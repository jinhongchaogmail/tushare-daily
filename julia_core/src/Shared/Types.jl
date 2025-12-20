"""
    Types.jl

核心类型定义模块。利用 Julia 的类型系统和多重派发实现可扩展架构。

设计理念:
1. 使用抽象类型定义接口
2. 具体类型继承抽象类型
3. 多重派发根据类型自动选择正确的实现
"""
module Types

using DataFrames

export AbstractTimeframe, DailyTimeframe, Daily20dTimeframe, Min5Timeframe
export AbstractFeatureParams, DailyFeatureParams, Min5FeatureParams
export AbstractTargetConfig, ClassificationTargetConfig
export TargetType, AbsoluteReturn, ExcessReturn, target_type_name
export timeframe_name, future_period

# ============================================================================
# 时间框架抽象 (Timeframe Abstraction)
# ============================================================================

"""
    AbstractTimeframe

所有时间框架的抽象基类型。
通过多重派发，不同时间框架可以有不同的特征计算、目标生成逻辑。
"""
abstract type AbstractTimeframe end

"""
    DailyTimeframe <: AbstractTimeframe

日线时间框架 (5日预测)。
"""
struct DailyTimeframe <: AbstractTimeframe end

"""
    Daily20dTimeframe <: AbstractTimeframe

日线时间框架 (20日战略预测)。
"""
struct Daily20dTimeframe <: AbstractTimeframe end

"""
    Min5Timeframe <: AbstractTimeframe

5分钟时间框架。
"""
struct Min5Timeframe <: AbstractTimeframe end

# 时间框架属性 (通过多重派发)
timeframe_name(::DailyTimeframe) = "daily"
timeframe_name(::Daily20dTimeframe) = "daily_20d"
timeframe_name(::Min5Timeframe) = "min5"

future_period(::DailyTimeframe) = 5    # 日线预测5天后
future_period(::Daily20dTimeframe) = 20 # 日线预测20天后 (战略)
future_period(::Min5Timeframe) = 12    # 5分钟预测12根K线后 (1小时)

# ============================================================================
# 特征参数抽象 (Feature Parameters)
# ============================================================================

"""
    AbstractFeatureParams

特征参数的抽象基类型。
"""
abstract type AbstractFeatureParams end

"""
    DailyFeatureParams <: AbstractFeatureParams

日线特征参数。
"""
Base.@kwdef struct DailyFeatureParams <: AbstractFeatureParams
    # SMA 周期
    sma_5::Int = 5
    sma_10::Int = 10
    sma_20::Int = 20
    
    # RSI
    rsi_window::Int = 14
    
    # MACD
    macd_fast::Int = 12
    macd_slow::Int = 26
    macd_signal::Int = 9
    
    # Bollinger Bands
    bb_window::Int = 20
    bb_std::Float64 = 2.0
    
    # ADX / ATR
    adx_length::Int = 14
    atr_length::Int = 14
    
    # Stochastic
    stoch_k::Int = 14
    stoch_d::Int = 3
    stoch_smooth::Int = 3
end

"""
    Min5FeatureParams <: AbstractFeatureParams

5分钟特征参数 (周期更短)。
"""
Base.@kwdef struct Min5FeatureParams <: AbstractFeatureParams
    sma_5::Int = 5
    sma_10::Int = 10
    sma_20::Int = 20
    
    rsi_window::Int = 10
    
    macd_fast::Int = 8
    macd_slow::Int = 17
    macd_signal::Int = 9
    
    bb_window::Int = 15
    bb_std::Float64 = 2.0
    
    atr_length::Int = 10
end

# 从 Dict 构造参数 (用于 Optuna 调参)
function DailyFeatureParams(d::Dict)
    DailyFeatureParams(
        sma_5 = get(d, "sma_5", 5),
        sma_10 = get(d, "sma_10", 10),
        sma_20 = get(d, "sma_20", 20),
        rsi_window = get(d, "rsi_window", 14),
        macd_fast = get(d, "macd_fast", 12),
        macd_slow = get(d, "macd_slow", 26),
        macd_signal = get(d, "macd_signal", 9),
        bb_window = get(d, "bb_window", 20),
        bb_std = Float64(get(d, "bb_std", 2.0)),
        adx_length = get(d, "adx_length", 14),
        atr_length = get(d, "atr_length", 14),
        stoch_k = get(d, "stoch_k", 14),
        stoch_d = get(d, "stoch_d", 3),
        stoch_smooth = get(d, "stoch_smooth", 3)
    )
end

# ============================================================================
# 目标配置抽象 (Target Configuration)
# ============================================================================

"""
    AbstractTargetConfig

目标生成配置的抽象基类型。
"""
abstract type AbstractTargetConfig end

"""
    ClassificationTargetConfig <: AbstractTargetConfig

三分类目标配置 (买入/持有/卖出)。
"""
Base.@kwdef struct ClassificationTargetConfig <: AbstractTargetConfig
    vol_multiplier::Float64 = 1.0      # 波动率乘数
    min_threshold::Float64 = 0.005     # 最小阈值
    max_threshold::Float64 = 0.05      # 最大阈值
    vol_window::Int = 20               # 波动率计算窗口
end

# ============================================================================
# 目标类型 (Target Type: absolute vs excess)
# ============================================================================

"""
    TargetType

目标收益类型的抽象基类型。
"""
abstract type TargetType end

"""
    AbsoluteReturn <: TargetType

绝对收益率模式：预测股票自身涨跌。
"""
struct AbsoluteReturn <: TargetType end

"""
    ExcessReturn <: TargetType

超额收益率模式：预测股票相对指数的超额收益。
"""
Base.@kwdef struct ExcessReturn <: TargetType
    index_code::String = "000001.SH"   # 基准指数代码
end

# 描述函数
target_type_name(::AbsoluteReturn) = "absolute"
target_type_name(t::ExcessReturn) = "excess (vs $(t.index_code))"

end # module
