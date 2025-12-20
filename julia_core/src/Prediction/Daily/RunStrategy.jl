module RunStrategy

using Dates
using DataFrames
using PythonCall
using Printf
using Statistics
using ....Shared
using ....Shared.DataFetcher
using ....Shared.Features.Daily: apply_technical_indicators!
using ....Shared.Model: load_model, predict_catboost

export run_strategy, init_model, predict_stock

# 常量定义
const TS_ENV = "prd"
const TS_SERVER = "http://116.128.206.39:7172"
const OUT_DIR = "data/daily"
const MODEL_PATH_5D = "../models/daily/catboost_daily_julia.cbm"  # 使用 Julia 兼容模型
const MODEL_PATH_20D = "../models/daily/catboost_daily_20d.cbm"
const MODEL_PATH_MIN5 = "../models/min5/catboost_min5.cbm"

# 全局变量
const models = Dict{String, Any}()

"""
    init_model()

初始化模型，加载 5日、20日 和 5分钟模型。
"""
function init_model()
    cb = pyimport("catboost")
    
    # 加载 5日模型
    if isfile(MODEL_PATH_5D)
        try
            model_5d = cb.CatBoostClassifier()
            model_5d.load_model(MODEL_PATH_5D)
            models["5d"] = model_5d
            println("✅ 成功加载 5日模型: $MODEL_PATH_5D")
        catch e
            println("⚠️ 加载 5日模型失败: $e")
        end
    else
        println("⚠️ 5日模型不存在: $MODEL_PATH_5D")
    end

    # 加载 20日模型
    if isfile(MODEL_PATH_20D)
        try
            model_20d = cb.CatBoostClassifier()
            model_20d.load_model(MODEL_PATH_20D)
            models["20d"] = model_20d
            println("✅ 成功加载 20日模型: $MODEL_PATH_20D")
        catch e
            println("⚠️ 加载 20日模型失败: $e")
        end
    else
        println("⚠️ 20日模型不存在: $MODEL_PATH_20D")
    end
    
    # 加载 5分钟模型
    if isfile(MODEL_PATH_MIN5)
        try
            model_min5 = cb.CatBoostClassifier()
            model_min5.load_model(MODEL_PATH_MIN5)
            models["min5"] = model_min5
            println("✅ 成功加载 5分钟模型: $MODEL_PATH_MIN5")
        catch e
            println("⚠️ 加载 5分钟模型失败: $e")
        end
    else
        println("⚠️ 5分钟模型不存在: $MODEL_PATH_MIN5")
    end

    return !isempty(models)
end

"""
    predict_stock(ts_code, df)

对单只股票进行预测。
"""
function predict_stock(ts_code::String, df::DataFrame)
    if isempty(models) || nrow(df) < 60
        if nrow(df) < 60
            println("  [$ts_code] 数据不足 ($(nrow(df))行)，跳过预测")
        end
        return nothing
    end

    try
        # 检查是否已有预计算的特征（Arrow 数据）
        has_precomputed_features = "RSI_14" in names(df) || "SMA_20" in names(df)
        
        # 特征工程 - 只在需要时才计算
        df_features = copy(df)
        if !has_precomputed_features
            apply_technical_indicators!(df_features)
        end
        
        # 取最后一行
        latest_row = df_features[end, :]
        current_date = get(latest_row, :trade_date, Date(2020, 1, 1))
        
        result = Dict{String, Any}(
            "ts_code" => ts_code,
            "date" => current_date,
            "prob_up_5d" => 0.0, "prob_down_5d" => 0.0,
            "prob_up_20d" => 0.0, "prob_down_20d" => 0.0,
            "signal_5d" => "观望", "signal_20d" => "观望",
            "final_signal" => "", "reason" => "", "score" => 0.0
        )

        pd = pyimport("pandas")
        np = pyimport("numpy")

        # --- 5日模型预测 ---
        if haskey(models, "5d")
            model = models["5d"]
            # 正确获取特征名 - 使用 pyconvert 转换 Python list
            feats = pyconvert(Vector{String}, model.feature_names_)
            
            # 构建输入数据 - 使用 NumPy 数组
            data = Float64[]
            for f in feats
                if f in names(latest_row)
                    val = latest_row[f]
                    push!(data, ismissing(val) || isnan(val) ? 0.0 : Float64(val))
                else
                    push!(data, 0.0)
                end
            end
            X_np = np.array(pylist([pylist(data)]))
            
            probs_py = model.predict_proba(X_np)
            probs = pyconvert(Vector{Float64}, probs_py[0])
            
            # 三分类: 0=下跌, 1=中性, 2=上涨
            result["prob_down_5d"] = probs[1]  # 类别0
            result["prob_up_5d"] = probs[3]    # 类别2
            
            if result["prob_up_5d"] > 0.42
                result["signal_5d"] = "看多"
            elseif result["prob_down_5d"] > 0.42
                result["signal_5d"] = "看空"
            end
        end

        # --- 20日模型预测 ---
        # 注意: 20日模型是 Python 训练的，特征名不兼容 Julia Arrow 数据
        # 暂时跳过 20日模型预测
        # TODO: 训练一个 Julia 兼容的 20日模型
        # if haskey(models, "20d")
        #     ...
        # end

        # --- 立体化信号融合 ---
        final_signal = ""
        reason = ""
        score = 0.0
        
        p5_up = result["prob_up_5d"]
        p20_up = result["prob_up_20d"]
        p5_down = result["prob_down_5d"]
        p20_down = result["prob_down_20d"]
        
        # 1. 共振做多 (战略+战术)
        if p5_up > 0.4 && p20_up > 0.4
            final_signal = "🔴 强力做多 (共振)"
            reason = @sprintf("短线(%.2f)与中线(%.2f)共振", p5_up, p20_up)
            score = p5_up + p20_up
        # 2. 战术做多 (短线强，中线不差)
        elseif p5_up > 0.45 && p20_down < 0.35
            final_signal = "🟠 战术做多"
            reason = @sprintf("短线爆发(%.2f)", p5_up)
            score = p5_up
        # 3. 战略布局 (中线强，短线回调或震荡)
        elseif p20_up > 0.45 && p5_down < 0.4
            final_signal = "🟡 战略布局"
            reason = @sprintf("中线看好(%.2f)", p20_up)
            score = p20_up
        # 空头逻辑
        elseif p5_down > 0.4 && p20_down > 0.4
            final_signal = "🟢 强力做空 (共振)"
            reason = "双周期看空"
            score = p5_down + p20_down
        elseif p5_down > 0.45
            final_signal = "🔵 战术做空"
            reason = @sprintf("短线风险(%.2f)", p5_down)
            score = p5_down
        end

        result["final_signal"] = final_signal
        result["reason"] = reason
        result["score"] = score
        
        if !isempty(final_signal)
            println("  !!! [$ts_code] $final_signal | $reason")
        end

        return result
    catch e
        println("❌ [$ts_code] 预测出错: ", e)
        @error "预测错误详情" exception=(e, catch_backtrace())
        return nothing
    end
end

"""
    run_strategy(; stocks=nothing, batch_size=10, parallel_workers=2)

运行策略的主函数。
"""
function run_strategy(; stocks::Union{String, Nothing}=nothing, batch_size::Int=10, parallel_workers::Int=2)
    println("🚀 启动 Julia 版策略预测系统...")
    
    # 初始化 Tushare
    ts_token = get(ENV, "TUSHARE_TOKEN", "")
    if isempty(ts_token)
        error("缺少 TUSHARE_TOKEN 环境变量")
    end
    
    ts = pyimport("xcsc_tushare")
    ts.set_token(ts_token)
    pro = ts.pro_api(env=TS_ENV, server=TS_SERVER)
    
    # 初始化模型
    if !init_model()
        println("⚠️ 未加载任何模型，仅进行数据获取测试")
    end
    
    # 获取股票列表
    println("📋 正在获取股票列表...")
    df_list = DataFrame()
    try
        # 调用 Python Tushare 接口
        # 参照 run_strategy.py 的参数
        py_df = pro.stock_basic(market="CS", fields="ts_code,name,list_date,delist_date,list_board_name")
        
        # 转换为 Julia DataFrame
        pd = pyimport("pandas")
        if pyisinstance(py_df, pd.DataFrame)
             cols = ["ts_code", "name"]
             df_list = DataFrame()
             for col in cols
                 if col in py_df.columns
                     # 使用 pyconvert 安全转换
                     df_list[!, col] = [pyconvert(String, x) for x in PyArray(py_df[col].values)]
                 end
             end
        end
        
        println("✅ 获取到 $(nrow(df_list)) 只股票")
    catch e
        println("⚠️ 获取股票列表失败: $e")
        return
    end
    
    # 过滤股票
    if !isnothing(stocks)
        target_stocks = split(stocks, ",")
        filter!(row -> row.ts_code in target_stocks, df_list)
        println("🔍 筛选后剩余 $(nrow(df_list)) 只股票")
    end
    
    # 批处理
    tickers = df_list.ts_code
    total = length(tickers)
    
    println("🚀 开始处理 $total 只股票...")
    
    report = []
    
    # 简单的串行处理 (为了演示)
    for (i, ts_code) in enumerate(tickers)
        println("[$i/$total] 处理 $ts_code ...")
        
        # 获取数据
        df = get_hist(pro, ts_code)
        if isnothing(df)
            continue
        end
        
        # 预测
        result = predict_stock(ts_code, df)
        if !isnothing(result) && !isempty(result["final_signal"])
            push!(report, result)
        end
    end
    
    # 生成报告
    if !isempty(report)
        println("\n=== 每日策略报告 (总计: $(length(report))) ===")
        # 简单的文本报告
        for r in report
            println("$(r["ts_code"]): $(r["final_signal"]) [$(r["reason"])]")
        end
        # TODO: 保存为 CSV/HTML
    else
        println("ℹ️ 今日无符合条件的交易机会")
    end
    
    println("🎉 所有任务已完成")
end

"""
    get_hist(pro, ts_code)

获取单只股票的历史数据。
"""
function get_hist(pro, ts_code::String)
    try
        # 获取日线行情
        start_date = "20220101" # 默认
        py_daily = pro.daily(ts_code=ts_code, start_date=start_date, fields="ts_code,trade_date,open,high,low,close,volume,amount")
        
        if Bool(py_daily.empty)
            return nothing
        end
        
        # 转换为 Julia DataFrame
        df = DataFrame()
        cols = ["ts_code", "trade_date", "open", "high", "low", "close", "volume", "amount"]
        
        # 简单的列转换
        df.ts_code = [pyconvert(String, x) for x in PyArray(py_daily["ts_code"].values)]
        df.trade_date = [pyconvert(String, x) for x in PyArray(py_daily["trade_date"].values)]
        df.open = [pyconvert(Float64, x) for x in PyArray(py_daily["open"].values)]
        df.high = [pyconvert(Float64, x) for x in PyArray(py_daily["high"].values)]
        df.low = [pyconvert(Float64, x) for x in PyArray(py_daily["low"].values)]
        df.close = [pyconvert(Float64, x) for x in PyArray(py_daily["close"].values)]
        df.volume = [pyconvert(Float64, x) for x in PyArray(py_daily["volume"].values)]
        df.amount = [pyconvert(Float64, x) for x in PyArray(py_daily["amount"].values)]
        
        # 排序
        sort!(df, :trade_date)
        
        # 日期转换
        df.trade_date = Date.(df.trade_date, "yyyymmdd")
        
        return df
    catch e
        println("⚠️ 获取数据失败 [$ts_code]: $e")
        return nothing
    end
end

end # module
