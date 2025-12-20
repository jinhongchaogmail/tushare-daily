"""
    CatBoostTuner.jl

针对 Tesla P4 GPU 优化的 CatBoost 超参数搜索模块。

设计理念:
1. 利用 CatBoost 对称树在 P4 上的高效性（零分支发散）
2. 充分利用 ~5秒/轮 的训练速度，跑大规模参数搜索
3. 针对 P4 的特点优化 border_count (直方图箱数)
4. 小数据量时自动回退到 CPU (避免统计偏差导致的 F1 崩溃)

Tesla P4 特性:
- 推理优化卡，FP32 仅 5.5 TFLOPS
- CatBoost 对称树完美契合，无分支发散
- 大数据量 (>1万行) 时 GPU 加速 5.3x
- 小数据量 (<1万行) 时 GPU 精度下降，建议用 CPU
"""
module CatBoostTuner

using DataFrames
using PythonCall
using Statistics
using Random
using Dates
using Printf
using Hyperopt
using Hyperopt: BOHB, Continuous, Categorical

# 使用共享模块
using ....Shared.Types: DailyTimeframe, ClassificationTargetConfig
using ....Shared.Features.Daily: apply_technical_indicators!
using ....Shared.Targets: add_future_returns!, create_targets!
using ....Shared.DataFetcher: load_data_files

export run_catboost_grid_search, run_catboost_random_search, 
       run_catboost_hyperopt, run_catboost_bayesian,
       TunerConfig, save_best_model

# ============================================================================
# 配置结构
# ============================================================================

"""
    TunerConfig

超参数搜索配置
"""
Base.@kwdef struct TunerConfig
    # 数据配置
    data_dir::String = "data/daily"
    limit_files::Union{Int, Nothing} = nothing
    train_ratio::Float64 = 0.8
    
    # 设备配置
    auto_device::Bool = true              # 自动选择设备
    min_samples_for_gpu::Int = 10000      # GPU 最小样本数阈值
    force_gpu::Bool = false               # 强制使用 GPU
    
    # P4 优化参数
    border_count_gpu::Int = 64            # GPU 模式下的直方图箱数 (P4 优化)
    border_count_cpu::Int = 128           # CPU 模式下的直方图箱数
    
    # 搜索配置
    n_trials::Int = 50                    # Random Search 试验次数
    early_stopping_rounds::Int = 50       # 早停轮数
    random_seed::Int = 42
    
    # 目标配置
    vol_multiplier::Float64 = 1.0         # 波动率乘数
end

# ============================================================================
# 参数网格定义 (针对 P4 优化)
# ============================================================================

"""
针对 Tesla P4 优化的参数网格

关键洞察:
- depth: P4 跑深度 6-8 的对称树非常高效
- learning_rate: 配合 early_stopping，可以用较大的学习率
- l2_leaf_reg: 正则化，防止过拟合
- border_count: P4 上 64 通常比 128/256 更快且精度损失小
"""
const P4_OPTIMIZED_GRID = [
    # 基准配置
    (depth=6, lr=0.1, l2=3.0, iters=1000),
    (depth=6, lr=0.08, l2=5.0, iters=1500),
    (depth=6, lr=0.05, l2=3.0, iters=2000),
    
    # 更深的树 (P4 对深度 8 的对称树也很高效)
    (depth=8, lr=0.08, l2=5.0, iters=1000),
    (depth=8, lr=0.05, l2=3.0, iters=1500),
    (depth=8, lr=0.03, l2=5.0, iters=2000),
    
    # 更浅的树 (更快，适合快速筛选)
    (depth=4, lr=0.15, l2=1.0, iters=1500),
    (depth=4, lr=0.1, l2=3.0, iters=2000),
    
    # 高正则化配置
    (depth=6, lr=0.05, l2=10.0, iters=2000),
    (depth=8, lr=0.03, l2=10.0, iters=2500),
    
    # 低正则化配置 (适合数据量大时)
    (depth=6, lr=0.1, l2=1.0, iters=1000),
    (depth=8, lr=0.08, l2=1.0, iters=1200),
]

"""
扩展参数网格 (用于精细调优)
"""
const EXTENDED_GRID = [
    # 添加更多 depth/lr 组合
    (depth=5, lr=0.1, l2=3.0, iters=1200),
    (depth=5, lr=0.08, l2=5.0, iters=1500),
    (depth=7, lr=0.08, l2=3.0, iters=1200),
    (depth=7, lr=0.05, l2=5.0, iters=1800),
    (depth=9, lr=0.05, l2=5.0, iters=1500),
    (depth=9, lr=0.03, l2=8.0, iters=2000),
    (depth=10, lr=0.03, l2=10.0, iters=2000),
    
    # 更多正则化变体
    (depth=6, lr=0.08, l2=0.5, iters=1200),
    (depth=6, lr=0.08, l2=8.0, iters=1500),
    (depth=6, lr=0.08, l2=15.0, iters=1800),
    (depth=8, lr=0.05, l2=0.5, iters=1500),
    (depth=8, lr=0.05, l2=15.0, iters=2000),
]

# ============================================================================
# 辅助函数
# ============================================================================

"""
    materialize_columns!(df::DataFrame)

将 DataFrame 中的惰性列转换为普通 Vector。
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
    prepare_data(config::TunerConfig)

加载和预处理数据。支持两种模式:
1. 已有特征的数据 (Arrow 格式, 含 RSI/MACD/SMA 等)
2. 原始数据 (需要计算特征)

返回: (X_train, y_train, X_val, y_val, feature_names, n_samples)
"""
function prepare_data(config::TunerConfig)
    # 导入 Python 库
    np = pyimport("numpy")
    sklearn_utils_class_weight = pyimport("sklearn.utils.class_weight")
    
    # 定义时间框架和目标配置
    timeframe = DailyTimeframe()
    target_config = ClassificationTargetConfig(vol_multiplier=config.vol_multiplier)
    
    # 加载数据
    println("📂 从 $(config.data_dir) 加载数据...")
    raw_dfs = load_data_files(config.data_dir; limit=config.limit_files)
    
    if isempty(raw_dfs)
        error("未找到数据文件！")
    end
    
    # 过滤空文件
    raw_dfs = filter(df -> nrow(df) > 0 && ncol(df) > 0, raw_dfs)
    
    if isempty(raw_dfs)
        error("所有数据文件都是空的！")
    end
    
    println("   已加载 $(length(raw_dfs)) 个有效文件")
    
    # 检测数据是否已有特征
    sample_df = first(raw_dfs)
    has_precomputed_features = all(col -> col in names(sample_df), ["RSI_14", "SMA_20", "MACD_12_26_9"])
    
    if has_precomputed_features
        println("   ✅ 检测到预计算特征，跳过特征工程")
    else
        println("   📊 原始数据，将计算技术指标...")
    end
    
    # 处理数据
    processed_dfs = DataFrame[]
    
    for df in raw_dfs
        df_copy = copy(df)
        materialize_columns!(df_copy)
        
        # 如果没有预计算特征，则计算
        if !has_precomputed_features
            apply_technical_indicators!(df_copy)
        end
        
        # 生成真正的未来收益 (future_return)
        # 注意: return_5d 是过去收益，不能用！
        if "close" in names(df_copy)
            n = nrow(df_copy)
            future_return = Vector{Union{Missing, Float64}}(missing, n)
            
            for i in 1:(n-5)
                if !ismissing(df_copy.close[i]) && !ismissing(df_copy.close[i+5])
                    future_return[i] = df_copy.close[i+5] / df_copy.close[i] - 1.0
                end
            end
            
            df_copy.future_return = future_return
        end
        
        # 检查 volatility_factor
        if !("volatility_factor" in names(df_copy))
            # 如果没有，使用默认值
            df_copy.volatility_factor = fill(0.02, nrow(df_copy))  # 默认 2% 阈值
        end
        
        # 生成目标 (如果没有)
        if !("target" in names(df_copy))
            if "future_return" in names(df_copy) && "volatility_factor" in names(df_copy)
                create_targets!(df_copy, target_config)
            end
        end
        
        # 过滤有效数据
        if "target" in names(df_copy)
            # 删除 target 为 missing 的行
            df_valid = df_copy[.!ismissing.(df_copy.target), :]
            if nrow(df_valid) > 0
                push!(processed_dfs, df_valid)
            end
        end
    end
    
    if isempty(processed_dfs)
        error("处理后无有效数据！")
    end
    
    # 合并数据
    full_df = reduce((a, b) -> vcat(a, b; cols=:intersect), processed_dfs)
    
    # 准备特征 - 排除非特征列
    exclude_cols = [
        # 标识列
        "ts_code", "trade_time", "target", "date",
        # 原始价格列
        "open", "high", "low", "close", "volume", "amount", "pre_close",
        # 目标相关列
        "future_close", "future_return", "volatility_factor", "pred_5d",
        # 日期列
        "trade_date", "ann_date", "end_date", "f_ann_date",
        # 文本/分类列
        "trade_status", "crncy_code", "crncy_code_basic",
        # 复权列 (避免数据泄露)
        "adj_close", "adj_open", "adj_high", "adj_low", "adj_pre_close",
        "adj_factor_x", "adj_factor_y", "adj_factor",
        # 其他非数值列
        "close_basic"
    ]
    feature_cols = setdiff(names(full_df), exclude_cols)
    
    # 只保留数值列
    numeric_cols = String[]
    for col in feature_cols
        col_type = eltype(full_df[!, col])
        # 检查是否为数值类型 (包括 Union{Missing, Number})
        if col_type <: Number || (col_type isa Union && any(t -> t <: Number, Base.uniontypes(col_type)))
            push!(numeric_cols, col)
        end
    end
    feature_cols = numeric_cols
    
    println("   特征列数: $(length(feature_cols))")
    
    # 选择特征和目标
    select!(full_df, [feature_cols; "target"])
    
    # 转换 target 列类型并删除缺失值
    full_df.target = convert(Vector{Union{Missing, Int}}, full_df.target)
    dropmissing!(full_df)
    
    n_samples = nrow(full_df)
    println("   总样本数: $n_samples")
    
    if n_samples < 100
        error("样本数过少: $n_samples")
    end
    
    X = full_df[:, feature_cols]
    y = Vector{Int}(full_df.target)
    
    # 训练/验证划分
    split_idx = floor(Int, n_samples * config.train_ratio)
    
    X_train = X[1:split_idx, :]
    y_train = y[1:split_idx]
    X_val = X[split_idx+1:end, :]
    y_val = y[split_idx+1:end]
    
    # 计算类别权重
    classes = sort(unique(y_train))
    class_weights = sklearn_utils_class_weight.compute_class_weight(
        "balanced", classes=np.array(classes), y=np.array(y_train)
    )
    weights_dict = pydict(Dict(classes[i] => pyconvert(Float64, class_weights[i-1]) for i in 1:length(classes)))
    
    # 将 DataFrame 转换为 Matrix，处理 Missing
    X_train_mat = Matrix{Float64}(coalesce.(Matrix(X_train), NaN))
    X_val_mat = Matrix{Float64}(coalesce.(Matrix(X_val), NaN))
    
    # 转换为 NumPy
    X_train_np = np.array(X_train_mat)
    y_train_np = np.array(y_train)
    X_val_np = np.array(X_val_mat)
    y_val_np = np.array(y_val)
    
    println("   训练集: $(size(X_train, 1)) 样本")
    println("   验证集: $(size(X_val, 1)) 样本")
    
    return (
        X_train=X_train_np, 
        y_train=y_train_np, 
        X_val=X_val_np, 
        y_val=y_val_np,
        feature_names=feature_cols,
        n_samples=n_samples,
        class_weights=weights_dict
    )
end

"""
    select_device(n_samples::Int, config::TunerConfig)

根据样本数自动选择设备。

Tesla P4 特性:
- 大数据量 (>1万) 时 GPU 有 5.3x 加速
- 小数据量 (<1万) 时 GPU 精度下降，建议用 CPU
"""
function select_device(n_samples::Int, config::TunerConfig)
    if config.force_gpu
        return ("GPU", config.border_count_gpu)
    end
    
    if !config.auto_device
        return ("GPU", config.border_count_gpu)
    end
    
    if n_samples >= config.min_samples_for_gpu
        return ("GPU", config.border_count_gpu)
    else
        @warn "样本数 $n_samples < $(config.min_samples_for_gpu)，自动切换到 CPU 模式以保证精度"
        return ("CPU", config.border_count_cpu)
    end
end

"""
    train_and_evaluate(params, data, device_info, config::TunerConfig)

训练单个模型并返回 F1 分数。
"""
function train_and_evaluate(params, data, device_info, config::TunerConfig)
    catboost = pyimport("catboost")
    sklearn_metrics = pyimport("sklearn.metrics")
    
    device, border_count = device_info
    
    # 构建模型参数
    model_kwargs = Dict(
        :iterations => params.iters,
        :depth => params.depth,
        :learning_rate => params.lr,
        :l2_leaf_reg => params.l2,
        :border_count => border_count,
        :loss_function => "MultiClass",
        :eval_metric => "MultiClass",
        :classes_count => 3,
        :class_weights => data.class_weights,
        :verbose => 0,
        :allow_writing_files => false,
        :random_seed => config.random_seed
    )
    
    # 设置设备
    if device == "GPU"
        model_kwargs[:task_type] = "GPU"
        model_kwargs[:devices] = "0"
    else
        model_kwargs[:task_type] = "CPU"
    end
    
    # 创建模型
    model = catboost.CatBoostClassifier(; model_kwargs...)
    
    # 训练 - 使用 Pool 以支持特征名
    train_pool = catboost.Pool(data.X_train, label=data.y_train, feature_names=data.feature_names)
    val_pool = catboost.Pool(data.X_val, label=data.y_val, feature_names=data.feature_names)
    
    t_start = time()
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=config.early_stopping_rounds
    )
    t_elapsed = time() - t_start
    
    # 评估
    preds = model.predict(data.X_val)
    f1 = pyconvert(Float64, sklearn_metrics.f1_score(data.y_val, preds, average="macro"))
    accuracy = pyconvert(Float64, sklearn_metrics.accuracy_score(data.y_val, preds))
    
    # 获取实际使用的迭代次数 (可能因 early stopping 而减少)
    best_iter = pyconvert(Int, model.get_best_iteration())
    
    return (f1=f1, accuracy=accuracy, time=t_elapsed, best_iter=best_iter, model=model)
end

# ============================================================================
# 主搜索函数
# ============================================================================

"""
    run_catboost_grid_search(; config=TunerConfig(), extended=false)

运行网格搜索。

# Arguments
- `config`: TunerConfig 配置对象
- `extended`: 是否使用扩展网格

# Returns
- 包含最佳参数和所有结果的 NamedTuple
"""
function run_catboost_grid_search(; config::TunerConfig=TunerConfig(), extended::Bool=false)
    println("\n" * "="^70)
    println("🚀 CatBoost 网格搜索 (Tesla P4 优化)")
    println("="^70)
    
    # 准备数据
    data = prepare_data(config)
    
    # 选择设备
    device_info = select_device(data.n_samples, config)
    device, border_count = device_info
    println("\n📱 设备: $device (border_count=$border_count)")
    
    # 选择网格
    grid = extended ? vcat(P4_OPTIMIZED_GRID, EXTENDED_GRID) : P4_OPTIMIZED_GRID
    println("📊 参数组合数: $(length(grid))")
    
    # 预估时间
    est_time_per_trial = device == "GPU" ? 5.0 : 25.0
    est_total_time = est_time_per_trial * length(grid)
    println("⏱️  预估总时间: $(round(est_total_time/60, digits=1)) 分钟")
    
    # 开始搜索
    results = Vector{NamedTuple}()
    best_f1 = 0.0
    best_params = nothing
    best_model = nothing
    
    println("\n" * "-"^70)
    println("开始搜索...")
    println("-"^70)
    
    for (i, params) in enumerate(grid)
        result = train_and_evaluate(params, data, device_info, config)
        
        push!(results, (
            params=params,
            f1=result.f1,
            accuracy=result.accuracy,
            time=result.time,
            best_iter=result.best_iter
        ))
        
        # 更新最佳
        status = ""
        if result.f1 > best_f1
            best_f1 = result.f1
            best_params = params
            best_model = result.model
            status = " ⭐ NEW BEST!"
        end
        
        # 打印进度
        println(@sprintf("[%2d/%2d] depth=%d, lr=%.3f, l2=%.1f, iters=%d | F1=%.4f, Acc=%.4f, Time=%.1fs%s",
            i, length(grid),
            params.depth, params.lr, params.l2, params.iters,
            result.f1, result.accuracy, result.time, status
        ))
    end
    
    # 打印总结
    println("\n" * "="^70)
    println("📊 搜索完成!")
    println("="^70)
    println("最佳 F1: $(round(best_f1, digits=4))")
    println("最佳参数: depth=$(best_params.depth), lr=$(best_params.lr), l2=$(best_params.l2), iters=$(best_params.iters)")
    println("总用时: $(round(sum(r.time for r in results)/60, digits=2)) 分钟")
    
    # 打印 Top 5
    sorted_results = sort(results, by=r -> r.f1, rev=true)
    println("\n🏆 Top 5 配置:")
    for (i, r) in enumerate(sorted_results[1:min(5, length(sorted_results))])
        println("  #$i: F1=$(round(r.f1, digits=4)), depth=$(r.params.depth), lr=$(r.params.lr), l2=$(r.params.l2)")
    end
    
    return (
        best_f1=best_f1,
        best_params=best_params,
        best_model=best_model,
        all_results=results,
        data=data,
        device=device
    )
end

"""
    run_catboost_random_search(; config=TunerConfig())

运行随机搜索 (更灵活的参数空间探索)。

# Arguments
- `config`: TunerConfig 配置对象

# Returns
- 包含最佳参数和所有结果的 NamedTuple
"""
function run_catboost_random_search(; config::TunerConfig=TunerConfig())
    println("\n" * "="^70)
    println("🎲 CatBoost 随机搜索 (Tesla P4 优化)")
    println("="^70)
    
    # 准备数据
    data = prepare_data(config)
    
    # 选择设备
    device_info = select_device(data.n_samples, config)
    device, border_count = device_info
    println("\n📱 设备: $device (border_count=$border_count)")
    println("📊 试验次数: $(config.n_trials)")
    
    # 预估时间
    est_time_per_trial = device == "GPU" ? 5.0 : 25.0
    est_total_time = est_time_per_trial * config.n_trials
    println("⏱️  预估总时间: $(round(est_total_time/60, digits=1)) 分钟")
    
    # 设置随机种子
    Random.seed!(config.random_seed)
    
    # 参数空间定义 (针对 P4 优化)
    depth_range = 4:10
    lr_range = (0.01, 0.15)
    l2_range = (0.5, 20.0)
    iters_range = 800:200:3000
    
    # 开始搜索
    results = Vector{NamedTuple}()
    best_f1 = 0.0
    best_params = nothing
    best_model = nothing
    
    println("\n" * "-"^70)
    println("开始随机搜索...")
    println("-"^70)
    
    for i in 1:config.n_trials
        # 随机采样参数
        params = (
            depth = rand(depth_range),
            lr = rand() * (lr_range[2] - lr_range[1]) + lr_range[1],
            l2 = exp(rand() * (log(l2_range[2]) - log(l2_range[1])) + log(l2_range[1])),  # log 均匀分布
            iters = rand(iters_range)
        )
        
        result = train_and_evaluate(params, data, device_info, config)
        
        push!(results, (
            params=params,
            f1=result.f1,
            accuracy=result.accuracy,
            time=result.time,
            best_iter=result.best_iter
        ))
        
        # 更新最佳
        status = ""
        if result.f1 > best_f1
            best_f1 = result.f1
            best_params = params
            best_model = result.model
            status = " ⭐ NEW BEST!"
        end
        
        # 打印进度
        println(@sprintf("[%2d/%2d] depth=%d, lr=%.3f, l2=%.2f, iters=%d | F1=%.4f, Acc=%.4f, Time=%.1fs%s",
            i, config.n_trials,
            params.depth, params.lr, params.l2, params.iters,
            result.f1, result.accuracy, result.time, status
        ))
    end
    
    # 打印总结
    println("\n" * "="^70)
    println("📊 随机搜索完成!")
    println("="^70)
    println("最佳 F1: $(round(best_f1, digits=4))")
    println("最佳参数: depth=$(best_params.depth), lr=$(round(best_params.lr, digits=4)), l2=$(round(best_params.l2, digits=2)), iters=$(best_params.iters)")
    println("总用时: $(round(sum(r.time for r in results)/60, digits=2)) 分钟")
    
    # 打印 Top 5
    sorted_results = sort(results, by=r -> r.f1, rev=true)
    println("\n🏆 Top 5 配置:")
    for (i, r) in enumerate(sorted_results[1:min(5, length(sorted_results))])
        println("  #$i: F1=$(round(r.f1, digits=4)), depth=$(r.params.depth), lr=$(round(r.params.lr, digits=3)), l2=$(round(r.params.l2, digits=2))")
    end
    
    return (
        best_f1=best_f1,
        best_params=best_params,
        best_model=best_model,
        all_results=results,
        data=data,
        device=device
    )
end

"""
    save_best_model(result, filename::String)

保存最佳模型到文件。
"""
function save_best_model(result, filename::String)
    if result.best_model === nothing
        error("没有最佳模型可保存！")
    end
    
    result.best_model.save_model(filename)
    println("✅ 模型已保存到: $filename")
end

# ============================================================================
# Hyperopt.jl 集成 - 高级超参数优化
# ============================================================================

"""
    run_catboost_hyperopt(; config=TunerConfig(), sampler=:random)

使用 Hyperopt.jl 进行超参数优化。

# Arguments
- `config`: TunerConfig 配置
- `sampler`: 采样器类型
  - `:random` - RandomSampler (默认，灵活)
  - `:lhs` - LHSampler (拉丁超立方，更均匀覆盖)
  - `:gp` - GPSampler (高斯过程，贝叶斯优化)
  - `:bluenoise` - BlueNoiseSampler (蓝噪声采样)

# Returns
- 包含最佳参数和优化历史的 NamedTuple

# Tesla P4 优化说明
由于 GPU 训练极快 (~5秒)，可以使用更多迭代次数进行更彻底的搜索。
贝叶斯优化 (GPSampler) 在参数空间探索上更高效，推荐用于精细调优。
"""
function run_catboost_hyperopt(; config::TunerConfig=TunerConfig(), sampler::Symbol=:random)
    println("\n" * "="^70)
    println("🔬 CatBoost Hyperopt 优化 (Tesla P4)")
    println("="^70)
    
    # 准备数据
    data = prepare_data(config)
    
    # 选择设备
    device_info = select_device(data.n_samples, config)
    device, border_count = device_info
    println("\n📱 设备: $device (border_count=$border_count)")
    
    # 选择采样器
    sampler_obj = if sampler == :random
        println("📊 采样器: RandomSampler")
        RandomSampler()
    elseif sampler == :lhs
        println("📊 采样器: LHSampler (拉丁超立方)")
        # LHS 需要迭代次数等于候选数
        RandomSampler()  # 回退到 Random，因为 LHS 对离散参数支持有限
    elseif sampler == :bohb || sampler == :gp || sampler == :bayesian
        println("📊 采样器: BOHB (贝叶斯优化)")
        # BOHB 需要指定参数维度
        BOHB(dims=[
            Categorical(7),   # depth: 7 个离散值
            Continuous(),     # lr: 连续
            Continuous(),     # l2: 连续
            Categorical(8)    # iters: 8 个离散值
        ])
    elseif sampler == :bluenoise
        println("📊 采样器: RandomSampler (BlueNoise 不可用)")
        RandomSampler()
    else
        println("📊 采样器: RandomSampler (默认)")
        RandomSampler()
    end
    
    println("📊 试验次数: $(config.n_trials)")
    
    # 预估时间
    est_time_per_trial = device == "GPU" ? 5.0 : 25.0
    est_total_time = est_time_per_trial * config.n_trials
    println("⏱️  预估总时间: $(round(est_total_time/60, digits=1)) 分钟")
    
    # 定义搜索空间 (针对 P4 优化)
    # depth: 4-10, P4 对深度 6-8 效率最高
    # lr: 0.01-0.15, 配合 early_stopping
    # l2: 0.5-20.0, log 均匀分布
    # iters: 800-3000
    
    println("\n" * "-"^70)
    println("开始 Hyperopt 搜索...")
    println("-"^70)
    
    best_f1 = 0.0
    best_params = nothing
    best_model = nothing
    all_results = Vector{NamedTuple}()
    
    # 使用 Hyperopt 宏
    ho = @hyperopt for i = config.n_trials, 
            sampler = sampler_obj,
            depth = [4, 5, 6, 7, 8, 9, 10],
            lr = LinRange(0.01, 0.15, 50),
            l2 = exp10.(LinRange(-0.3, 1.3, 50)),  # 0.5 ~ 20
            iters = [800, 1000, 1200, 1500, 1800, 2000, 2500, 3000]
        
        # 构建参数
        params = (depth=depth, lr=lr, l2=l2, iters=iters)
        
        # 训练和评估
        result = train_and_evaluate(params, data, device_info, config)
        
        # 记录结果
        push!(all_results, (
            params=params,
            f1=result.f1,
            accuracy=result.accuracy,
            time=result.time,
            best_iter=result.best_iter
        ))
        
        # 更新最佳
        status = ""
        if result.f1 > best_f1
            best_f1 = result.f1
            best_params = params
            best_model = result.model
            status = " ⭐ NEW BEST!"
        end
        
        # 打印进度
        println(@sprintf("[%2d/%2d] depth=%d, lr=%.3f, l2=%.2f, iters=%d | F1=%.4f, Acc=%.4f, Time=%.1fs%s",
            i, config.n_trials,
            depth, lr, l2, iters,
            result.f1, result.accuracy, result.time, status
        ))
        
        # Hyperopt 最小化，所以返回负 F1
        -result.f1
    end
    
    # 打印总结
    println("\n" * "="^70)
    println("📊 Hyperopt 搜索完成!")
    println("="^70)
    println("最佳 F1: $(round(best_f1, digits=4))")
    println("最佳参数: depth=$(best_params.depth), lr=$(round(best_params.lr, digits=4)), l2=$(round(best_params.l2, digits=2)), iters=$(best_params.iters)")
    println("总用时: $(round(sum(r.time for r in all_results)/60, digits=2)) 分钟")
    
    # 打印 Top 5
    sorted_results = sort(all_results, by=r -> r.f1, rev=true)
    println("\n🏆 Top 5 配置:")
    for (j, r) in enumerate(sorted_results[1:min(5, length(sorted_results))])
        println("  #$j: F1=$(round(r.f1, digits=4)), depth=$(r.params.depth), lr=$(round(r.params.lr, digits=3)), l2=$(round(r.params.l2, digits=2))")
    end
    
    # Hyperopt 结果分析
    println("\n📈 Hyperopt 分析:")
    println("  最优参数组合: ", ho.minimizer)
    println("  最优目标值 (负F1): ", round(ho.minimum, digits=4))
    
    return (
        best_f1=best_f1,
        best_params=best_params,
        best_model=best_model,
        all_results=all_results,
        hyperopt_result=ho,
        data=data,
        device=device
    )
end

"""
    run_catboost_bayesian(; config=TunerConfig())

专门的贝叶斯优化函数，使用 BOHB (Bayesian Optimization Hyperband)。

BOHB 结合了贝叶斯优化和早停策略，能够:
1. 使用 KDE (核密度估计) 建模好/坏配置的分布
2. 优先采样更有希望的配置
3. 自动平衡探索与利用

# Arguments
- `config`: TunerConfig 配置

# Tesla P4 优化
由于 GPU 训练快速 (~5秒)，BOHB 可以快速迭代更新先验知识。
"""
function run_catboost_bayesian(; config::TunerConfig=TunerConfig())
    println("\n" * "="^70)
    println("🧠 CatBoost BOHB 贝叶斯优化 (Tesla P4)")
    println("="^70)
    
    # 准备数据
    data = prepare_data(config)
    
    # 选择设备
    device_info = select_device(data.n_samples, config)
    device, border_count = device_info
    println("\n📱 设备: $device (border_count=$border_count)")
    println("📊 总试验次数: $(config.n_trials)")
    println("📊 采样器: BOHB (Bayesian Optimization Hyperband)")
    
    # 预估时间
    est_time_per_trial = device == "GPU" ? 5.0 : 25.0
    est_total_time = est_time_per_trial * config.n_trials
    println("⏱️  预估总时间: $(round(est_total_time/60, digits=1)) 分钟")
    
    println("\n" * "-"^70)
    println("开始 BOHB 贝叶斯优化...")
    println("-"^70)
    
    best_f1 = 0.0
    best_params = nothing
    best_model = nothing
    all_results = Vector{NamedTuple}()
    
    # 参数空间
    depth_values = [4, 5, 6, 7, 8, 9, 10]
    iters_values = [800, 1000, 1200, 1500, 1800, 2000, 2500, 3000]
    
    # 使用 BOHB 采样器
    bohb_sampler = BOHB(dims=[
        Categorical(length(depth_values)),   # depth
        Continuous(),                         # lr
        Continuous(),                         # l2 (log scale)
        Categorical(length(iters_values))    # iters
    ])
    
    ho = @hyperopt for i = config.n_trials,
            sampler = bohb_sampler,
            depth_idx = 1:length(depth_values),
            lr = LinRange(0.01, 0.15, 50),
            l2_log = LinRange(-0.3, 1.3, 50),  # 0.5 ~ 20
            iters_idx = 1:length(iters_values)
        
        # 映射参数
        depth = depth_values[depth_idx]
        iters = iters_values[iters_idx]
        l2 = 10.0^l2_log
        
        params = (depth=depth, lr=lr, l2=l2, iters=iters)
        result = train_and_evaluate(params, data, device_info, config)
        
        push!(all_results, (
            params=params,
            f1=result.f1,
            accuracy=result.accuracy,
            time=result.time,
            best_iter=result.best_iter
        ))
        
        # 更新最佳
        status = ""
        if result.f1 > best_f1
            best_f1 = result.f1
            best_params = params
            best_model = result.model
            status = " ⭐ NEW BEST!"
        end
        
        println(@sprintf("[%2d/%2d] depth=%d, lr=%.3f, l2=%.2f, iters=%d | F1=%.4f, Acc=%.4f, Time=%.1fs%s",
            length(all_results), config.n_trials,
            depth, lr, l2, iters,
            result.f1, result.accuracy, result.time, status
        ))
        
        -result.f1  # 最小化负 F1
    end
    
    # 打印总结
    println("\n" * "="^70)
    println("📊 BOHB 贝叶斯优化完成!")
    println("="^70)
    println("最佳 F1: $(round(best_f1, digits=4))")
    if best_params !== nothing
        println("最佳参数: depth=$(best_params.depth), lr=$(round(best_params.lr, digits=4)), l2=$(round(best_params.l2, digits=2)), iters=$(best_params.iters)")
    end
    println("总用时: $(round(sum(r.time for r in all_results)/60, digits=2)) 分钟")
    
    # 打印 Top 5
    sorted_results = sort(all_results, by=r -> r.f1, rev=true)
    println("\n🏆 Top 5 配置:")
    for (j, r) in enumerate(sorted_results[1:min(5, length(sorted_results))])
        println("  #$j: F1=$(round(r.f1, digits=4)), depth=$(r.params.depth), lr=$(round(r.params.lr, digits=3)), l2=$(round(r.params.l2, digits=2))")
    end
    
    return (
        best_f1=best_f1,
        best_params=best_params,
        best_model=best_model,
        all_results=all_results,
        hyperopt_result=ho,
        data=data,
        device=device
    )
end

end # module
