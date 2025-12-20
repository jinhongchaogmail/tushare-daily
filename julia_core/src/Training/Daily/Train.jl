"""
    Train.jl

日线训练模块 - 完整版本

设计理念:
1. 使用 Types 中定义的抽象类型 (DailyTimeframe, ExcessReturn 等)
2. 使用 Targets 模块生成目标
3. 支持 TimeSeriesSplit 交叉验证 + Future Gap (防止数据泄露)
4. 支持 RobustScaler 特征标准化
5. 支持 Focal Loss 风格权重 + 买入类增强
6. 支持模型保存、后验评估、分类报告
7. 支持 Pruner 剪枝和 Study 持久化

与 Python 版本对齐的功能:
- TimeSeriesSplit (3-5 折交叉验证)
- Future Gap (训练集和验证集之间的间隔)
- RobustScaler (CV 内独立标准化)
- Focal Loss 权重增强
- 买入类 (Class 1) 权重 ×2
- MedianPruner 早停无效试验
- SQLite 持久化 Study
- 模型保存 (.cbm)
- 后验评估 + 分类报告
- 特征重要性导出
"""
module Train

using DataFrames
using PythonCall
using Statistics
using Random
using Dates
using JSON
using Printf
using CSV

# 使用新的模块化架构
using ....Shared.Types: DailyTimeframe, Daily20dTimeframe, ClassificationTargetConfig, timeframe_name, future_period
using ....Shared.Types: TargetType, AbsoluteReturn, ExcessReturn, target_type_name
using ....Shared.Features.Daily: apply_technical_indicators!
using ....Shared.Targets: add_future_returns!, create_targets!, load_index_data, get_index_returns
using ....Shared.DataFetcher: load_data_files

export run_daily_optimization

# ============================================================================
# 辅助函数
# ============================================================================

"""
    materialize_columns!(df::DataFrame)

将 DataFrame 中的惰性列（如 FillArrays）转换为普通 Vector。
"""
function materialize_columns!(df::DataFrame)
    for col in names(df)
        col_data = df[!, col]
        if !(col_data isa Vector)
            df[!, col] = collect(col_data)
        end
    end
end

"""
    time_series_split(n_samples::Int, n_splits::Int)

生成时间序列交叉验证的索引。
返回 Vector{Tuple{UnitRange, UnitRange}}，每个元素是 (train_idx, val_idx)。
"""
function time_series_split(n_samples::Int, n_splits::Int)
    splits = Tuple{UnitRange{Int}, UnitRange{Int}}[]
    fold_size = n_samples ÷ (n_splits + 1)
    
    for i in 1:n_splits
        train_end = fold_size * (i + 1)
        val_start = train_end + 1
        val_end = min(train_end + fold_size, n_samples)
        
        if val_start <= n_samples && val_end > val_start
            push!(splits, (1:train_end, val_start:val_end))
        end
    end
    
    return splits
end

"""
    apply_future_gap(train_idx::UnitRange, val_idx::UnitRange, gap)

应用 Future Gap，从训练集末尾移除 gap 个样本以防止标签泄露。
返回调整后的训练索引。如果 gap 为 nothing 或 <= 0，则返回原始索引。
"""
function apply_future_gap(train_idx::UnitRange, val_idx::UnitRange, gap::Union{Int, Nothing})
    # 如果 gap 为 nothing 或 <= 0，不做调整
    if gap === nothing || gap <= 0
        return train_idx
    end
    
    if last(train_idx) + gap >= first(val_idx)
        safe_train_end = first(val_idx) - gap - 1
        if safe_train_end >= first(train_idx)
            return first(train_idx):safe_train_end
        else
            return 1:0  # 空范围
        end
    end
    return train_idx
end

"""
    compute_focal_weights(y::Vector, gamma::Float64=2.0; boost_buy::Bool=true)

计算 Focal Loss 风格的类别权重。
- gamma: Focal Loss 的 gamma 参数
- boost_buy: 是否额外增强买入类 (Class 1) 的权重
"""
function compute_focal_weights(np, sklearn_class_weight, y::Vector; gamma::Float64=2.0, boost_buy::Bool=true, boost_factor::Float64=2.0)
    classes = sort(unique(y))
    
    # 基础平衡权重
    base_weights = sklearn_class_weight.compute_class_weight(
        "balanced", classes=np.array(classes), y=np.array(y)
    )
    
    weight_map = Dict{Int, Float64}()
    for (i, cls) in enumerate(classes)
        weight_map[cls] = pyconvert(Float64, base_weights[i-1])
    end
    
    # Focal Loss 风格增强
    total = length(y)
    class_counts = Dict{Int, Int}()
    for cls in classes
        class_counts[cls] = count(==(cls), y)
    end
    
    for cls in classes
        cls_ratio = class_counts[cls] / total
        focal_factor = (1 - cls_ratio) ^ gamma
        weight_map[cls] = weight_map[cls] * (1 + focal_factor * 0.5)
    end
    
    # 买入类增强 (Class 1 = 上涨)
    if boost_buy && haskey(weight_map, 1)
        weight_map[1] = weight_map[1] * boost_factor
    end
    
    # 归一化
    max_weight = maximum(values(weight_map))
    for cls in keys(weight_map)
        weight_map[cls] = weight_map[cls] / max_weight * 3.0
    end
    
    return weight_map
end

"""
    print_classification_report(sklearn_metrics, y_true, y_pred; labels=[-1, 0, 1])

打印分类报告（精确率、召回率、F1）。接受 Julia Vector 或 numpy 数组。
"""
function print_classification_report(sklearn_metrics, y_true, y_pred; labels=[-1, 0, 1])
    np = pyimport("numpy")
    # 将 numpy 数组转换为 Julia Vector
    y_t = y_true isa Vector ? y_true : pyconvert(Vector{Int}, y_true)
    y_p = y_pred isa Vector ? y_pred : pyconvert(Vector{Int}, y_pred)
    
    target_names = ["下跌 (-1)", "持平 (0)", "上涨 (1)"]
    
    println("\n" * "=" ^ 60)
    println("                    分类报告")
    println("=" ^ 60)
    
    for (i, (label, name)) in enumerate(zip(labels, target_names))
        mask_true = y_t .== label
        mask_pred = y_p .== label
        
        tp = sum(mask_true .& mask_pred)
        fp = sum(.!mask_true .& mask_pred)
        fn = sum(mask_true .& .!mask_pred)
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        support = sum(mask_true)
        
        println(@sprintf("  %s: Precision=%.3f, Recall=%.3f, F1=%.3f (支持=%d)", 
                        name, precision, recall, f1, support))
    end
    
    # Macro F1 - 使用 numpy 数组进行 sklearn 调用
    y_true_np = y_true isa Vector ? np.array(y_true) : y_true
    y_pred_np = y_pred isa Vector ? np.array(y_pred) : y_pred
    macro_f1 = pyconvert(Float64, sklearn_metrics.f1_score(y_true_np, y_pred_np, average="macro", labels=np.array(labels), zero_division=0))
    println("-" ^ 60)
    println(@sprintf("  Macro F1: %.4f", macro_f1))
    println("=" ^ 60)
    
    return macro_f1
end

"""
    print_confusion_matrix(y_true, y_pred; labels=[-1, 0, 1])

打印混淆矩阵。接受 Julia Vector 或 numpy 数组。
"""
function print_confusion_matrix(y_true, y_pred; labels=[-1, 0, 1])
    # 将 numpy 数组转换为 Julia Vector
    y_t = y_true isa Vector ? y_true : pyconvert(Vector{Int}, y_true)
    y_p = y_pred isa Vector ? y_pred : pyconvert(Vector{Int}, y_pred)
    
    println("\n混淆矩阵:")
    println("              预测下跌  预测持平  预测上涨")
    
    for (i, true_label) in enumerate(labels)
        row_name = ["实际下跌", "实际持平", "实际上涨"][i]
        counts = Int[]
        for pred_label in labels
            push!(counts, sum((y_t .== true_label) .& (y_p .== pred_label)))
        end
        println(@sprintf("  %s:    %6d    %6d    %6d", row_name, counts...))
    end
end

# ============================================================================
# 主训练函数
# ============================================================================

"""
    run_daily_optimization(; kwargs...)

运行日线 Optuna 超参数优化。

# Arguments
- `n_trials`: 试验次数
- `data_dir`: 数据目录
- `limit_files`: 限制文件数量 (用于快速测试)
- `timeframe`: 时间框架 (DailyTimeframe() 或 Daily20dTimeframe())
- `target_type`: 目标类型 (AbsoluteReturn() 或 ExcessReturn(index_code="000001.SH"))
- `config`: 完整配置字典 (包含搜索空间等参数)
- `output_dir`: 模型和结果输出目录

# Returns
- Optuna Study 对象
"""
function run_daily_optimization(; 
    n_trials::Int=20, 
    data_dir::String="data/daily", 
    limit_files::Union{Nothing, Int}=nothing, 
    timeframe=DailyTimeframe(), 
    target_type::TargetType=AbsoluteReturn(),
    config::Union{Nothing, Dict}=nothing,
    output_dir::String="optuna_results/daily_julia"
)
    # 导入 Python 库
    optuna = pyimport("optuna")
    catboost = pyimport("catboost")
    np = pyimport("numpy")
    sklearn_metrics = pyimport("sklearn.metrics")
    sklearn_preprocessing = pyimport("sklearn.preprocessing")
    sklearn_utils_class_weight = pyimport("sklearn.utils.class_weight")
    
    # 确保输出目录存在
    mkpath(output_dir)
    
    # ========== 打印训练配置 ==========
    println()
    println("=" ^ 70)
    println("                    日线模型训练配置")
    println("=" ^ 70)
    println("  预测周期:     $(future_period(timeframe)) 天 ($(timeframe_name(timeframe)))")
    println("  目标类型:     $(target_type_name(target_type))")
    if target_type isa ExcessReturn
        println("  基准指数:     $(target_type.index_code)")
    end
    println("  数据目录:     $data_dir")
    println("  Optuna 试验:  $n_trials")
    println("  限制文件数:   $(something(limit_files, "全部"))")
    println("  输出目录:     $output_dir")
    println("=" ^ 70)
    println()
    
    # 加载数据 (自动检测格式)
    println("从 $data_dir 加载原始数据...")
    raw_dfs = load_data_files(data_dir; limit=limit_files)
    
    if isempty(raw_dfs)
        println("❌ 未找到数据。")
        return nothing
    end
    
    println("✅ 已加载 $(length(raw_dfs)) 个文件。")
    
    # 如果是超额收益模式，加载指数数据
    index_returns = nothing
    if target_type isa ExcessReturn
        println("\n加载基准指数数据: $(target_type.index_code)...")
        index_df = load_index_data(target_type.index_code, data_dir)
        if index_df !== nothing
            index_returns = get_index_returns(index_df, future_period(timeframe))
            println("✅ 指数收益率计算完成 ($(length(index_returns)) 个交易日)")
        else
            println("⚠️  未找到指数数据，将使用绝对收益率模式")
        end
    end
    
    # 从配置读取各项参数
    feat_cfg = config !== nothing ? get(config, "features", Dict()) : Dict()
    model_cfg = config !== nothing ? get(config, "model", Dict()) : Dict()
    class_cfg = config !== nothing ? get(config, "classification", Dict()) : Dict()
    train_cfg = config !== nothing ? get(config, "training", Dict()) : Dict()
    gpu_cfg = config !== nothing ? get(config, "gpu", Dict()) : Dict()
    
    # 搜索空间辅助函数
    get_range(d, key, default) = get(d, key, default)
    
    # 训练参数
    n_cv_splits = get(train_cfg, "cv_splits", 3)
    # future_gap: 如果配置为 null/nothing，则使用预测周期作为默认值
    future_gap_raw = get(train_cfg, "future_gap", nothing)
    future_gap = something(future_gap_raw, future_period(timeframe))
    use_scaler = get(train_cfg, "use_scaler", true)
    split_ratio = get(train_cfg, "split_ratio", 0.8)
    early_stopping = get(train_cfg, "early_stopping", 50)
    min_samples = get(get(config !== nothing ? config : Dict(), "data", Dict()), "min_samples", 100)
    
    # 权重参数
    focal_gamma = get(class_cfg, "focal_gamma", 2.0)
    boost_buy = get(class_cfg, "boost_buy", true)
    boost_factor = get(class_cfg, "boost_factor", 2.0)
    
    # GPU 参数
    gpu_enabled = get(gpu_cfg, "enabled", true)
    gpu_device = get(gpu_cfg, "device", "0")
    
    # 模型参数
    loss_func = get(class_cfg, "loss_function", "MultiClass")
    eval_metric = get(class_cfg, "eval_metric", "MultiClass")
    classes_count = get(class_cfg, "classes_count", 3)
    
    # Study 持久化
    study_name = "daily_$(timeframe_name(timeframe))_$(target_type_name(target_type))_$(Dates.format(now(), "yyyymmdd"))"
    db_path = joinpath(output_dir, "optuna_study.db")
    
    println("\n训练参数:")
    println("  CV 折数:      $n_cv_splits")
    println("  Future Gap:   $future_gap")
    println("  使用 Scaler:  $use_scaler")
    println("  Focal Gamma:  $focal_gamma")
    println("  买入类增强:   $boost_buy (×$boost_factor)")
    println("  Study 名称:   $study_name")
    println()

    # 用于存储最佳模型信息
    best_model_info = Ref{Dict{String, Any}}(Dict())
    best_feature_cols = Ref{Vector{String}}(String[])
    
    # 定义目标函数
    function objective(trial)
        trial_start = time()
        trial_number = pyconvert(Int, trial.number) + 1
        
        # 1. 建议特征参数
        rsi_range = get_range(feat_cfg, "rsi_window", [6, 30])
        macd_fast_range = get_range(feat_cfg, "macd_fast", [8, 20])
        macd_slow_range = get_range(feat_cfg, "macd_slow", [20, 60])
        macd_signal_range = get_range(feat_cfg, "macd_signal", [5, 20])
        bb_window_range = get_range(feat_cfg, "bb_window", [10, 30])
        bb_std_range = get_range(feat_cfg, "bb_std", [1.5, 2.5])
        atr_length_range = get_range(feat_cfg, "atr_length", [10, 30])
        
        feat_params = Dict{String, Any}(
            "rsi_window" => pyconvert(Int, trial.suggest_int("rsi_window", rsi_range[1], rsi_range[2])),
            "macd_fast" => pyconvert(Int, trial.suggest_int("macd_fast", macd_fast_range[1], macd_fast_range[2])),
            "macd_slow" => pyconvert(Int, trial.suggest_int("macd_slow", macd_slow_range[1], macd_slow_range[2])),
            "macd_signal" => pyconvert(Int, trial.suggest_int("macd_signal", macd_signal_range[1], macd_signal_range[2])),
            "bb_window" => pyconvert(Int, trial.suggest_int("bb_window", bb_window_range[1], bb_window_range[2])),
            "bb_std" => pyconvert(Float64, trial.suggest_float("bb_std", bb_std_range[1], bb_std_range[2])),
            "atr_length" => pyconvert(Int, trial.suggest_int("atr_length", atr_length_range[1], atr_length_range[2])),
        )
        
        # 2. 建议模型参数
        iter_range = get_range(model_cfg, "iterations", [1000, 3000])
        depth_range = get_range(model_cfg, "depth", [4, 10])
        lr_range = get_range(model_cfg, "learning_rate", [0.005, 0.10])
        l2_range = get_range(model_cfg, "l2_leaf_reg", [1e-8, 20.0])
        rs_range = get_range(model_cfg, "random_strength", [1e-8, 10.0])
        bt_range = get_range(model_cfg, "bagging_temperature", [0.0, 1.0])
        bc_range = get_range(model_cfg, "border_count", [32, 255])
        
        iterations = pyconvert(Int, trial.suggest_int("iterations", iter_range[1], iter_range[2]))
        depth = pyconvert(Int, trial.suggest_int("depth", depth_range[1], depth_range[2]))
        learning_rate = pyconvert(Float64, trial.suggest_float("learning_rate", lr_range[1], lr_range[2]))
        l2_leaf_reg = pyconvert(Float64, trial.suggest_float("l2_leaf_reg", l2_range[1], l2_range[2], log=true))
        random_strength = pyconvert(Float64, trial.suggest_float("random_strength", rs_range[1], rs_range[2], log=true))
        bagging_temperature = pyconvert(Float64, trial.suggest_float("bagging_temperature", bt_range[1], bt_range[2]))
        border_count = pyconvert(Int, trial.suggest_int("border_count", bc_range[1], bc_range[2]))
        
        # 3. 建议目标参数 (波动率乘数)
        vol_range = get_range(class_cfg, "vol_multiplier", [0.5, 1.5])
        vol_multiplier = pyconvert(Float64, trial.suggest_float("vol_multiplier", vol_range[1], vol_range[2]))
        target_config = ClassificationTargetConfig(vol_multiplier=vol_multiplier)
        
        # 4. 处理数据
        processed_dfs = DataFrame[]
        
        for df in raw_dfs
            df_copy = copy(df)
            materialize_columns!(df_copy)
            
            # 应用特征
            apply_technical_indicators!(df_copy; params=feat_params)
            
            # 生成目标
            add_future_returns!(df_copy, timeframe; target_type=target_type, index_returns=index_returns)
            
            if "future_return" in names(df_copy) && "volatility_factor" in names(df_copy)
                create_targets!(df_copy, target_config)
                dropmissing!(df_copy, "future_return")
                
                if nrow(df_copy) > 0
                    push!(processed_dfs, df_copy)
                end
            end
        end
        
        if isempty(processed_dfs)
            return 0.0
        end
        
        full_df = reduce((a, b) -> vcat(a, b; cols=:intersect), processed_dfs)
        
        # 5. 准备特征和标签
        exclude_cols = [
            "ts_code", "trade_time", "target", 
            "open", "high", "low", "close", "volume", "amount", "pre_close", "date",
            "future_close", "future_return", "volatility_factor",
            "trade_date", "ann_date", "end_date", "f_ann_date"
        ]
        feature_cols = setdiff(names(full_df), exclude_cols)
        
        # 只保留数值列
        numeric_cols = String[]
        for col in feature_cols
            if eltype(full_df[!, col]) <: Union{Missing, Number}
                push!(numeric_cols, col)
            end
        end
        feature_cols = numeric_cols
        
        select!(full_df, [feature_cols; "target"])
        dropmissing!(full_df)
        
        if nrow(full_df) < min_samples
            return 0.0
        end
        
        X = Matrix(full_df[:, feature_cols])
        y = Vector(full_df.target)
        n_samples = length(y)
        
        # 检查买入信号数量 (防止过于保守的策略)
        count_up = count(==(1), y)
        if count_up < 10
            return 0.0  # 买入信号太少，无效试验
        end
        
        # 6. TimeSeriesSplit 交叉验证
        splits = time_series_split(n_samples, n_cv_splits)
        f1_scores = Float64[]
        
        for (fold, (train_idx_raw, val_idx)) in enumerate(splits)
            # 应用 Future Gap
            train_idx = apply_future_gap(train_idx_raw, val_idx, future_gap)
            
            if length(train_idx) < 50
                continue  # 训练集太小
            end
            
            X_tr = X[train_idx, :]
            y_tr = y[train_idx]
            X_val = X[val_idx, :]
            y_val = y[val_idx]
            
            # RobustScaler (CV 内独立标准化)
            if use_scaler
                scaler = sklearn_preprocessing.RobustScaler()
                X_tr_scaled = pyconvert(Matrix, scaler.fit_transform(np.array(X_tr)))
                X_val_scaled = pyconvert(Matrix, scaler.transform(np.array(X_val)))
            else
                X_tr_scaled = X_tr
                X_val_scaled = X_val
            end
            
            # Focal Loss 风格权重
            weight_map = compute_focal_weights(np, sklearn_utils_class_weight, y_tr; 
                                               gamma=focal_gamma, boost_buy=boost_buy, boost_factor=boost_factor)
            weights_dict = pydict(weight_map)
            
            # 转换为 NumPy
            X_tr_np = np.array(X_tr_scaled)
            y_tr_np = np.array(y_tr)
            X_val_np = np.array(X_val_scaled)
            y_val_np = np.array(y_val)
            
            # 训练 CatBoost
            model = catboost.CatBoostClassifier(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                random_strength=random_strength,
                bagging_temperature=bagging_temperature,
                border_count=border_count,
                loss_function=loss_func,
                eval_metric=eval_metric,
                classes_count=classes_count,
                class_weights=weights_dict,
                verbose=0,
                task_type=gpu_enabled ? "GPU" : "CPU",
                devices=gpu_device,
                allow_writing_files=false
            )
            
            try
                model.fit(X_tr_np, y_tr_np, eval_set=(X_val_np, y_val_np), early_stopping_rounds=early_stopping)
            catch e
                @warn "训练失败: $e"
                continue
            end
            
            # 评估 - 注意: CatBoost predict 返回 2D array，需要 flatten
            preds_raw = model.predict(X_val_np)
            preds_np = np.array(preds_raw).flatten()
            f1 = pyconvert(Float64, sklearn_metrics.f1_score(y_val_np, preds_np, average="macro", labels=np.array([-1, 0, 1]), zero_division=0))
            push!(f1_scores, f1)
            
            # Pruning (报告中间值)
            if !isempty(f1_scores)
                intermediate_value = mean(f1_scores)
                trial.report(intermediate_value, fold - 1)
                if pyconvert(Bool, trial.should_prune())
                    throw(pyimport("optuna").TrialPruned())
                end
            end
        end
        
        if isempty(f1_scores)
            return 0.0
        end
        
        mean_f1 = mean(f1_scores)
        
        # 记录试验结果
        elapsed = time() - trial_start
        println(@sprintf("[✓ 试验 %d] F1=%.4f | 耗时 %.1fs", trial_number, mean_f1, elapsed))
        
        # 保存最佳模型信息
        if mean_f1 > get(best_model_info[], "f1", 0.0)
            best_model_info[] = Dict(
                "f1" => mean_f1,
                "trial_number" => trial_number,
                "feat_params" => feat_params,
                "model_params" => Dict(
                    "iterations" => iterations,
                    "depth" => depth,
                    "learning_rate" => learning_rate,
                    "l2_leaf_reg" => l2_leaf_reg,
                    "random_strength" => random_strength,
                    "bagging_temperature" => bagging_temperature,
                    "border_count" => border_count
                ),
                "vol_multiplier" => vol_multiplier
            )
            best_feature_cols[] = feature_cols
        end
        
        return mean_f1
    end

    # 创建 Optuna Study (带 Pruner 和持久化)
    println("创建 Optuna Study...")
    println("  数据库: $db_path")
    
    pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1)
    storage_url = "sqlite:///$db_path"
    
    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        study_name=study_name,
        storage=storage_url,
        load_if_exists=true
    )
    
    # 运行优化
    println("\n开始 Optuna 优化 (目标试验数: $n_trials)...")
    println("-" ^ 70)
    
    try
        study.optimize(objective, n_trials=n_trials, show_progress_bar=false)
    catch e
        if occursin("TrialPruned", string(e))
            @info "部分试验被 Pruner 剪枝"
        else
            rethrow(e)
        end
    end
    
    println("-" ^ 70)
    println("\n优化完成!")
    
    # 输出结果
    best_value = pyconvert(Float64, study.best_value)
    best_params = pyconvert(Dict, study.best_params)
    
    println("\n最佳参数:")
    println(JSON.json(best_params, 2))
    println(@sprintf("\n最佳 Macro F1: %.4f", best_value))
    
    # ========== 后验评估 ==========
    println("\n" * "=" ^ 70)
    println("                    后验评估")
    println("=" ^ 70)
    
    if best_value > 0.0 && !isempty(best_model_info[])
        println("\n使用最佳参数重新训练模型...")
        
        info = best_model_info[]
        feat_params = info["feat_params"]
        model_params = info["model_params"]
        vol_multiplier = info["vol_multiplier"]
        target_config = ClassificationTargetConfig(vol_multiplier=vol_multiplier)
        
        # 重新处理数据
        processed_dfs = DataFrame[]
        for df in raw_dfs
            df_copy = copy(df)
            materialize_columns!(df_copy)
            apply_technical_indicators!(df_copy; params=feat_params)
            add_future_returns!(df_copy, timeframe; target_type=target_type, index_returns=index_returns)
            
            if "future_return" in names(df_copy) && "volatility_factor" in names(df_copy)
                create_targets!(df_copy, target_config)
                dropmissing!(df_copy, "future_return")
                if nrow(df_copy) > 0
                    push!(processed_dfs, df_copy)
                end
            end
        end
        
        full_df = reduce((a, b) -> vcat(a, b; cols=:intersect), processed_dfs)
        feature_cols = best_feature_cols[]
        
        select!(full_df, [feature_cols; "target"])
        dropmissing!(full_df)
        
        X = Matrix(full_df[:, feature_cols])
        y = Vector(full_df.target)
        n_samples = length(y)
        
        # 固定训练/测试划分
        split_idx = floor(Int, n_samples * split_ratio)
        X_train = X[1:split_idx, :]
        y_train = y[1:split_idx]
        X_test = X[split_idx+1:end, :]
        y_test = y[split_idx+1:end]
        
        # RobustScaler
        if use_scaler
            scaler = sklearn_preprocessing.RobustScaler()
            X_train_scaled = pyconvert(Matrix, scaler.fit_transform(np.array(X_train)))
            X_test_scaled = pyconvert(Matrix, scaler.transform(np.array(X_test)))
        else
            X_train_scaled = X_train
            X_test_scaled = X_test
        end
        
        # Focal 权重
        weight_map = compute_focal_weights(np, sklearn_utils_class_weight, y_train; 
                                           gamma=focal_gamma, boost_buy=boost_buy, boost_factor=boost_factor)
        weights_dict = pydict(weight_map)
        
        # 训练最终模型
        final_model = catboost.CatBoostClassifier(
            iterations=model_params["iterations"],
            depth=model_params["depth"],
            learning_rate=model_params["learning_rate"],
            l2_leaf_reg=model_params["l2_leaf_reg"],
            random_strength=model_params["random_strength"],
            bagging_temperature=model_params["bagging_temperature"],
            border_count=model_params["border_count"],
            loss_function=loss_func,
            eval_metric=eval_metric,
            classes_count=classes_count,
            class_weights=weights_dict,
            verbose=100,
            task_type=gpu_enabled ? "GPU" : "CPU",
            devices=gpu_device,
            allow_writing_files=false
        )
        
        X_train_np = np.array(X_train_scaled)
        y_train_np = np.array(y_train)
        X_test_np = np.array(X_test_scaled)
        y_test_np = np.array(y_test)
        
        final_model.fit(X_train_np, y_train_np, eval_set=(X_test_np, y_test_np), early_stopping_rounds=early_stopping)
        
        # 在测试集上评估 - 使用 numpy 数组
        y_pred_raw = final_model.predict(X_test_np)
        y_pred_np = np.array(y_pred_raw).flatten()
        
        println("\n测试集结果:")
        test_f1 = print_classification_report(sklearn_metrics, y_test_np, y_pred_np)
        print_confusion_matrix(y_test_np, y_pred_np)
        
        # ========== 保存模型 ==========
        model_filename = "catboost_$(timeframe_name(timeframe))_$(target_type_name(target_type)).cbm"
        model_path = joinpath(output_dir, model_filename)
        final_model.save_model(model_path)
        println("\n✅ 模型已保存: $model_path")
        
        # 保存模型参数
        params_filename = "model_params_$(timeframe_name(timeframe))_$(target_type_name(target_type)).json"
        params_path = joinpath(output_dir, params_filename)
        
        params_to_save = Dict(
            "timeframe" => timeframe_name(timeframe),
            "target_type" => target_type_name(target_type),
            "future_period" => future_period(timeframe),
            "vol_multiplier" => vol_multiplier,
            "feature_params" => feat_params,
            "model_params" => model_params,
            "feature_columns" => feature_cols,
            "test_f1" => test_f1,
            "cv_f1" => best_value,
            "training_date" => Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
            "focal_gamma" => focal_gamma,
            "boost_buy" => boost_buy,
            "boost_factor" => boost_factor
        )
        
        open(params_path, "w") do f
            JSON.print(f, params_to_save, 2)
        end
        println("✅ 参数已保存: $params_path")
        
        # ========== 特征重要性 ==========
        println("\n特征重要性 (前 20):")
        println("-" ^ 50)
        
        importance = pyconvert(Vector{Float64}, final_model.get_feature_importance())
        fi_df = DataFrame(feature=feature_cols, importance=importance)
        sort!(fi_df, :importance, rev=true)
        
        for row in eachrow(first(fi_df, 20))
            println(@sprintf("  %-30s  %.4f", row.feature, row.importance))
        end
        
        # 保存特征重要性
        fi_path = joinpath(output_dir, "feature_importance_$(timeframe_name(timeframe))_$(target_type_name(target_type)).csv")
        CSV.write(fi_path, fi_df)
        println("\n✅ 特征重要性已保存: $fi_path")
        
    else
        println("⚠️  没有成功的试验，跳过后验评估")
    end
    
    # 保存试验结果
    println("\n保存试验结果...")
    pd = pyimport("pandas")
    trials_pdf = study.trials_dataframe()
    trials_path = joinpath(output_dir, "optuna_trials_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv")
    # 直接用 Pandas 保存 CSV，避免转换问题
    trials_pdf.to_csv(trials_path, index=false)
    println("✅ 试验结果已保存: $trials_path")
    
    println("\n" * "=" ^ 70)
    println("                    训练完成")
    println("=" ^ 70)
    
    return study
end

end # module
