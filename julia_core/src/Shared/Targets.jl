"""
    Targets.jl

目标生成模块。使用多重派发根据时间框架自动选择正确的目标生成逻辑。

设计理念:
- 目标生成逻辑独立于特征计算
- 通过类型派发支持不同时间框架
- 可配置的动态阈值
- 支持绝对收益和超额收益两种模式
"""
module Targets

using DataFrames
using Statistics
using RollingFunctions
using Arrow
using Parquet2
using ..Types
using ..Features.Utils: shift_series

export add_future_returns!, create_targets!, load_index_data, get_index_returns

# ============================================================================
# 指数数据加载 (用于超额收益计算)
# ============================================================================

# 全局缓存指数数据
const INDEX_CACHE = Dict{String, DataFrame}()

"""
    load_index_data(index_code::String, data_dir::String) -> DataFrame

加载指数日线数据。会缓存以避免重复加载。
"""
function load_index_data(index_code::String, data_dir::String)
    if haskey(INDEX_CACHE, index_code)
        return INDEX_CACHE[index_code]
    end
    
    # 尝试多种可能的文件路径
    possible_paths = [
        joinpath(data_dir, "$(index_code).arrow"),
        joinpath(data_dir, "$(index_code).parquet"),
        joinpath(dirname(data_dir), "index", "$(index_code).arrow"),
        joinpath(dirname(data_dir), "index", "$(index_code).parquet"),
    ]
    
    for path in possible_paths
        if isfile(path)
            df = if endswith(path, ".arrow")
                DataFrame(Arrow.Table(path))
            else
                DataFrame(Parquet2.Dataset(path))
            end
            INDEX_CACHE[index_code] = df
            @info "已加载指数数据: $(index_code) ($(nrow(df)) 行)"
            return df
        end
    end
    
    @warn "未找到指数数据: $(index_code)，将使用绝对收益率模式"
    return nothing
end

"""
    get_index_returns(index_df::DataFrame, period::Int) -> Dict{Date, Float64}

计算指数的 N 日收益率，返回日期到收益率的映射。
"""
function get_index_returns(index_df::DataFrame, period::Int)
    if index_df === nothing
        return nothing
    end
    
    # 确定日期列名
    date_col = "trade_date" in names(index_df) ? "trade_date" : "date"
    if !(date_col in names(index_df))
        @warn "指数数据缺少日期列"
        return nothing
    end
    
    close = Float64.(coalesce.(index_df.close, NaN))
    future_close = shift_series(close, -period)
    returns = (future_close .- close) ./ (close .+ 1e-8)
    
    # 构建日期到收益率的映射
    result = Dict{Any, Float64}()
    for i in 1:nrow(index_df)
        date = index_df[i, date_col]
        ret = returns[i]
        if !isnan(ret) && isfinite(ret)
            result[date] = ret
        end
    end
    
    return result
end

# ============================================================================
# Future Returns (基础目标数据)
# ============================================================================

"""
    add_future_returns!(df::DataFrame, tf::AbstractTimeframe; target_type=AbsoluteReturn(), index_returns=nothing)

添加未来收益列。根据时间框架自动确定预测周期。

# Arguments
- `df`: 输入 DataFrame，必须包含 `close` 列
- `tf`: 时间框架 (DailyTimeframe 或 Min5Timeframe)
- `target_type`: 目标类型 (AbsoluteReturn() 或 ExcessReturn())
- `index_returns`: 预计算的指数收益率 Dict (仅 excess 模式需要)

# 添加的列
- `future_close`: 未来收盘价
- `future_return`: 未来收益率 (绝对或超额)
- `volatility_factor`: 波动率因子
"""
function add_future_returns!(df::DataFrame, tf::AbstractTimeframe; 
                             target_type::TargetType=AbsoluteReturn(), 
                             index_returns::Union{Nothing, Dict}=nothing)
    if !("close" in names(df))
        @warn "DataFrame 缺少 close 列，跳过目标生成"
        return df
    end
    
    period = future_period(tf)
    n = nrow(df)
    
    # 转换 close 为 Float64 向量
    close = Float64.(coalesce.(df.close, NaN))
    
    # 计算未来收盘价 (负向偏移 = 向前看)
    future_close_vec = shift_series(close, -period)
    
    # 计算未来收益率 (绝对)
    stock_return_vec = (future_close_vec .- close) ./ (close .+ 1e-8)
    
    # 如果是超额收益模式，减去指数收益
    if target_type isa ExcessReturn && index_returns !== nothing
        # 确定日期列
        date_col = "trade_date" in names(df) ? "trade_date" : "date"
        if date_col in names(df)
            for i in 1:n
                date = df[i, date_col]
                if haskey(index_returns, date)
                    idx_ret = index_returns[date]
                    if !isnan(stock_return_vec[i])
                        stock_return_vec[i] -= idx_ret  # 超额收益 = 股票收益 - 指数收益
                    end
                end
            end
        end
    end
    
    # 转换 NaN 为 missing (便于后续 dropmissing)
    df[!, "future_close"] = [isnan(v) ? missing : v for v in future_close_vec]
    df[!, "future_return"] = [isnan(v) || !isfinite(v) ? missing : v for v in stock_return_vec]
    
    # 仅在 volatility_factor 不存在时计算
    # (通常已在 apply_technical_indicators! 中计算)
    if !("volatility_factor" in names(df))
        pct_change = zeros(Float64, n)
        @inbounds for i in 2:n
            if !isnan(close[i]) && !isnan(close[i-1]) && close[i-1] > 0
                pct_change[i] = (close[i] - close[i-1]) / close[i-1]
            end
        end
        volatility = runstd(pct_change, 20)
        df[!, "volatility_factor"] = [isnan(v) ? missing : v for v in volatility]
    end
    
    return df
end

# 便捷方法: 日线默认 (绝对收益)
add_future_returns!(df::DataFrame) = add_future_returns!(df, DailyTimeframe())

# ============================================================================
# Target Labels (分类标签)
# ============================================================================

"""
    create_targets!(df::DataFrame, config::ClassificationTargetConfig)

根据动态阈值生成三分类目标。

# 分类规则
- 2 (Buy):  future_return > vol_multiplier * volatility_factor
- 0 (Sell): future_return < -vol_multiplier * volatility_factor  
- 1 (Hold): 其他

# Arguments
- `df`: 输入 DataFrame，必须包含 `future_return` 和 `volatility_factor`
- `config`: 目标配置
"""
function create_targets!(df::DataFrame, config::ClassificationTargetConfig)
    required_cols = ["future_return", "volatility_factor"]
    for col in required_cols
        if !(col in names(df))
            @warn "DataFrame 缺少 $col 列，跳过目标生成"
            return df
        end
    end
    
    n = nrow(df)
    target = zeros(Int, n)
    
    future_ret = df.future_return
    vol_factor = df.volatility_factor
    
    @inbounds for i in 1:n
        ret = future_ret[i]
        vol = vol_factor[i]
        
        if ismissing(ret) || ismissing(vol)
            target[i] = 1  # 默认 Hold
            continue
        end
        
        # 动态阈值
        dyn_threshold = clamp(config.vol_multiplier * vol, config.min_threshold, config.max_threshold)
        
        if ret > dyn_threshold
            target[i] = 2  # Buy
        elseif ret < -dyn_threshold
            target[i] = 0  # Sell
        else
            target[i] = 1  # Hold
        end
    end
    
    df[!, "target"] = target
    return df
end

# 便捷方法: 使用默认配置
create_targets!(df::DataFrame; vol_multiplier::Float64=1.0) = 
    create_targets!(df, ClassificationTargetConfig(vol_multiplier=vol_multiplier))

end # module
