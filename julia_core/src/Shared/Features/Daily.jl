　module Daily

using DataFrames
using Statistics
using RollingFunctions
using ...ChanLun

export apply_technical_indicators!

safe_log(x) = log(x + 1e-8)

# --- Parameters Struct ---

"""
    FeatureParams

特征计算参数结构体。
"""
struct FeatureParams
    sma_5::Int
    sma_10::Int
    sma_20::Int
    rsi_window::Int
    macd_fast::Int
    macd_slow::Int
    macd_signal::Int
    bb_window::Int
    bb_std::Float64
    adx_length::Int
    atr_length::Int
    stoch_k::Int
    stoch_d::Int
    stoch_smooth::Int
end

# 默认参数构造函数
function FeatureParams(;
    sma_5=5, sma_10=10, sma_20=20,
    rsi_window=14,
    macd_fast=12, macd_slow=26, macd_signal=9,
    bb_window=5, bb_std=2.0,
    adx_length=14,
    atr_length=14,
    stoch_k=14, stoch_d=3, stoch_smooth=3
)
    return FeatureParams(
        sma_5, sma_10, sma_20,
        rsi_window,
        macd_fast, macd_slow, macd_signal,
        bb_window, bb_std,
        adx_length,
        atr_length,
        stoch_k, stoch_d, stoch_smooth
    )
end

"""
    apply_technical_indicators!(df::DataFrame; params::Dict=Dict())

计算技术指标 (Julia 实现)。
包含所有 Python 版本 (daily.py) 中的特征逻辑。
"""
function apply_technical_indicators!(df::DataFrame; params::Dict=Dict())
    # 解析参数
    default_p = FeatureParams()
    
    p = FeatureParams(
        sma_5 = get(params, "sma_5", default_p.sma_5),
        sma_10 = get(params, "sma_10", default_p.sma_10),
        sma_20 = get(params, "sma_20", default_p.sma_20),
        rsi_window = get(params, "rsi_window", default_p.rsi_window),
        macd_fast = get(params, "macd_fast", default_p.macd_fast),
        macd_slow = get(params, "macd_slow", default_p.macd_slow),
        macd_signal = get(params, "macd_signal", default_p.macd_signal),
        bb_window = get(params, "bb_window", default_p.bb_window),
        bb_std = Float64(get(params, "bb_std", default_p.bb_std)),
        adx_length = get(params, "adx_length", default_p.adx_length),
        atr_length = get(params, "atr_length", default_p.atr_length),
        stoch_k = get(params, "stoch_k", default_p.stoch_k),
        stoch_d = get(params, "stoch_d", default_p.stoch_d),
        stoch_smooth = get(params, "stoch_smooth", default_p.stoch_smooth)
    )

    # 确保按日期排序
    sort!(df, :trade_date)

    # --- 清洗数据 ---
    # 移除关键列为空的行
    cols_to_check = intersect(names(df), ["open", "high", "low", "close", "volume"])
    if nrow(df) > 0
        # Materialize immutable column types (e.g., FillArrays) before dropmissing! mutates them
        for col in names(df)
            col_data = df[!, col]
            if !(col_data isa Vector)
                df[!, col] = [col_data[i] for i in eachindex(col_data)]
            end
        end
        # println("Rows before dropmissing: ", nrow(df))
        # println("Type of close: ", eltype(df.close))
        # println("Any missing in close? ", any(ismissing, df.close))
        
        dropmissing!(df, cols_to_check)
        disallowmissing!(df, cols_to_check)
        
        println("Type of close after disallowmissing: ", eltype(df.close))
        println("Type of volume after disallowmissing: ", eltype(df.volume))
        
        # Double check
        for col in cols_to_check
            if any(ismissing, df[!, col])
                println("⚠️ 列 $col 在 dropmissing 后仍包含缺失值！")
            end
        end
        
        # println("Rows after dropmissing: ", nrow(df))
    end

    # --- 复权处理 ---
    if "adj_factor" in names(df)
        adj = df.adj_factor
        # 向量化操作，原地修改
        if "open" in names(df) df.open .*= adj end
        if "high" in names(df) df.high .*= adj end
        if "low" in names(df) df.low .*= adj end
        if "close" in names(df) df.close .*= adj end
        if "pre_close" in names(df) df.pre_close .*= adj end
        if "volume" in names(df) df.volume ./= adj end
    end

    # 提取向量 (Type Stable)
    if nrow(df) == 0
        return df
    end
    
    # println("Converting close...")
    # df[!, :close] = safe_log.(df[!, :close])
    # println("Close converted.")

    # println("Converting high...")
    # df[!, :high] = safe_log.(df[!, :high])
    # println("High converted.")

    # println("Converting low...")
    # df[!, :low] = safe_log.(df[!, :low])
    # println("Low converted.")

    # println("Converting volume...")
    # df[!, :volume] = safe_log.(df[!, :volume])
    # println("Volume converted.")
    
    # 将数据转换为 Float64 以支持所有技术指标计算函数
    close = Vector{Float64}(df.close)
    high = Vector{Float64}(df.high)
    low = Vector{Float64}(df.low)
    volume = Vector{Float64}(df.volume)
    
    # 预计算常用变量
    norm_denom = close .+ 1e-8
    # println("Norm denom calculated.")
    n_rows = length(close)

    # --- 基础指标 ---
    
    # SMA
    # println("Calculating SMA...")
    sma_5_col = "SMA_$(p.sma_5)"
    sma_10_col = "SMA_$(p.sma_10)"
    sma_20_col = "SMA_$(p.sma_20)"
    
    sma_5_val = runmean(close, p.sma_5)
    # println("SMA 5 calculated. Type: ", eltype(sma_5_val))
    sma_10_val = runmean(close, p.sma_10)
    sma_20_val = runmean(close, p.sma_20)
    
    println("正在计算 SMA...")
    df[!, sma_5_col] = sma_5_val
    df[!, sma_10_col] = sma_10_val
    df[!, sma_20_col] = sma_20_val
    println("SMA 计算完成。")
    
    # RSI
    println("正在计算 RSI...")
    rsi_col = "RSI_$(p.rsi_window)"
    rsi_val = calculate_rsi(close, p.rsi_window)
    df[!, rsi_col] = rsi_val
    println("RSI 计算完成。")
    
    # MACD
    println("正在计算 MACD...")
    macd_line, signal_line, hist_line = calculate_macd(close, p.macd_fast, p.macd_slow, p.macd_signal)
    macd_col = "MACD_$(p.macd_fast)_$(p.macd_slow)_$(p.macd_signal)"
    macd_s_col = "MACDs_$(p.macd_fast)_$(p.macd_slow)_$(p.macd_signal)"
    macd_h_col = "MACDh_$(p.macd_fast)_$(p.macd_slow)_$(p.macd_signal)"
    
    df[!, macd_col] = macd_line
    df[!, macd_s_col] = signal_line
    df[!, macd_h_col] = hist_line
    println("MACD 计算完成。")
    
    # BBANDS
    println("正在计算 BBANDS...")
    bb_upper, bb_middle, bb_lower = calculate_bbands(close, p.bb_window, p.bb_std)
    bb_col = "BB_$(p.bb_window)_$(p.bb_std)"
    bb_s_col = "BBs_$(p.bb_window)_$(p.bb_std)"
    bb_w_col = "BBw_$(p.bb_window)_$(p.bb_std)"
    df[!, bb_col] = bb_upper
    df[!, bb_s_col] = bb_middle
    df[!, bb_w_col] = (bb_upper .- bb_lower) ./ bb_middle
    println("BBANDS 计算完成。")
    
    # ATR
    println("正在计算 ATR...")
    atr_col = "ATRr_$(p.atr_length)"
    atr_val = calculate_atr(high, low, close, p.atr_length)
    df[!, atr_col] = atr_val
    println("ATR 计算完成。")
    
    # ADX
    println("正在计算 ADX...")
    adx_col = "ADX_$(p.adx_length)"
    adx_val = calculate_adx(high, low, close, p.adx_length)
    df[!, adx_col] = adx_val
    println("ADX 计算完成。")
    
    # Stochastic
    println("正在计算 Stochastic...")
    k, d = calculate_stoch(high, low, close, p.stoch_k, p.stoch_d, p.stoch_smooth)
    stoch_k_col = "STOCHk_$(p.stoch_k)_$(p.stoch_d)_$(p.stoch_smooth)"
    df[!, stoch_k_col] = k
    df[!, "STOCHd_$(p.stoch_k)_$(p.stoch_d)_$(p.stoch_smooth)"] = d
    println("Stochastic 计算完成。")
    
    # MACD norm
    println("正在计算 MACD norm...")
    macd_norm_col = "MACDnorm_$(p.macd_fast)_$(p.macd_slow)_$(p.macd_signal)"
    macd_norm_val = hist_line ./ (macd_line .+ 1e-8)
    df[!, macd_norm_col] = macd_norm_val
    println("MACD norm 计算完成。")
    
    # Converting open
    println("正在转换 open 数据...")
    if "open" in names(df)
        println("正在转换 open...")
        open_price = Float64.(df.open)
        df.open_close_gap = (close .- open_price) ./ (open_price .+ 1e-8)
        println("Open 转换完成并计算了 gap。")
    end

    # Bias SMA 20
    if sma_20_col in names(df)
        df.bias_sma20 = (close .- df[!, sma_20_col]) ./ (df[!, sma_20_col] .+ 1e-8)
    end

    # 量能特征
    vol_ma_20 = runmean(volume, 20) 
    df.volume_ma_ratio = volume ./ (vol_ma_20 .+ 1e-8)
    
    # 价格动量 * 量
    close_shift_1 = shift_series(close, 1)
    pct_change = (close .- close_shift_1) ./ close_shift_1
    df.price_volume_momentum = pct_change .* volume
    df.close_pct_change = pct_change # 保存供后续使用

    # 多周期收益率
    for lag in [2, 3, 5, 10]
        shifted = shift_series(close, lag)
        df[!, "return_$(lag)d"] = (close .- shifted) ./ shifted
    end

    # --- 量能指标 (OBV, AD) ---
    # 简化实现，不依赖 pandas_ta
    # OBV
    obv = zeros(Float64, n_rows)
    obv[1] = volume[1]
    for i in 2:n_rows
        if close[i] > close[i-1]
            obv[i] = obv[i-1] + volume[i]
        elseif close[i] < close[i-1]
            obv[i] = obv[i-1] - volume[i]
        else
            obv[i] = obv[i-1]
        end
    end
    df.OBV = obv
    
    # OBV Norm
    obv_mean_20 = runmean(obv, 20)
    # runsum 替代方案: mean * window
    vol_sum_20 = runmean(volume, 20) .* 20
    df.OBV_norm = (obv .- obv_mean_20) ./ (vol_sum_20 .+ 1e-8)

    # AD (Accumulation/Distribution)
    clv = ((close .- low) .- (high .- close)) ./ (high .- low .+ 1e-8)
    ad = cumsum(clv .* volume)
    df.AD = ad
    
    # AD Norm
    ad_mean_20 = runmean(ad, 20)
    df.AD_norm = (ad .- ad_mean_20) ./ (vol_sum_20 .+ 1e-8)

    # --- 分位数特征 ---
    # RSI Percentile 60
    rsi_min_60 = runmin(rsi_val, 60)
    rsi_max_60 = runmax(rsi_val, 60)
    df.rsi_percentile_60 = (rsi_val .- rsi_min_60) ./ (rsi_max_60 .- rsi_min_60 .+ 1e-8)

    # Price Percentile 60
    low_min_60 = runmin(low, 60)
    high_max_60 = runmax(high, 60)
    df.price_percentile_60 = (close .- low_min_60) ./ (high_max_60 .- low_min_60 .+ 1e-8)

    # Volume Percentile 60
    vol_min_60 = runmin(volume, 60)
    vol_max_60 = runmax(volume, 60)
    df.volume_percentile_60 = (volume .- vol_min_60) ./ (vol_max_60 .- vol_min_60 .+ 1e-8)

    # MACD Histogram Change
    hist_shift_1 = shift_series(hist_line, 1)
    df.macd_histogram_change = hist_line .- hist_shift_1

    # Trend Strength
    df.trend_strength = adx_val ./ 100.0

    # MA Alignment
    ma_align = ((sma_5_val .> sma_10_val) .+ (sma_10_val .> sma_20_val)) ./ 2.0
    df.ma_alignment = ma_align

    # BB Position
    bb_width_val = bb_upper .- bb_lower
    # bb_middle already calculated
    bb_pos = (close .- bb_middle) ./ (bb_width_val ./ 2.0 .+ 1e-8)
    df.bb_position = clamp.(bb_pos, -2.0, 2.0)

    # --- 交叉特征 ---
    df.rsi_volume_cross = (rsi_val ./ 50.0 .- 1.0) .* df.volume_ma_ratio
    df.macd_trend_cross = hist_line .* df.trend_strength
    df.bb_volume_cross = df.bb_position .* df.volume_percentile_60
    df.atr_volume_cross = atr_val .* df.volume_ma_ratio
    df.rsi_stoch_cross = (rsi_val ./ 100.0) .* (k ./ 100.0)

    # --- 波动率因子 ---
    # Top 2 特征，用于动态阈值
    # 20日滚动标准差
    df.volatility_factor = runstd(pct_change, 20)

    # 多周期波动率
    df.volatility_5d = runstd(pct_change, 5)
    df.volatility_10d = runstd(pct_change, 10)
    df.volatility_60d = runstd(pct_change, 60)

    # 指标波动率
    df.rsi_vol_20 = runstd(rsi_val, 20)
    df.bias_vol_20 = runstd(df.bias_sma20, 20)
    df.macd_h_vol_20 = runstd(hist_line, 20)

    # --- 趋势特征 ---
    # 1. 距离近期最低点的天数 (20日)
    # Julia RollingFunctions doesn't have argmin directly on window
    # We implement a simple loop for this specific feature
    days_since_low = fill(NaN, n_rows)
    for i in 20:n_rows
        window = view(low, i-19:i)
        min_val, min_idx = findmin(window)
        # min_idx is relative to window start (1-based)
        # if min_idx is last element (20), distance is 0
        days_since_low[i] = 20 - min_idx
    end
    df.days_since_low_20 = days_since_low

    # 2. 连涨天数
    consecutive_up = zeros(Int, n_rows)
    curr_run = 0
    for i in 2:n_rows
        if close[i] > close[i-1]
            curr_run += 1
        else
            curr_run = 0
        end
        consecutive_up[i] = curr_run
    end
    df.consecutive_up_days = consecutive_up

    # --- 基本面与资金流特征 (如果存在) ---
    
    # PE/PB Rank
    if "pe_ttm" in names(df)
        pe = Float64.(coalesce.(df.pe_ttm, NaN))
        pe_min = runmin(pe, 60)
        pe_max = runmax(pe, 60)
        df.pe_rank = (pe .- pe_min) ./ (pe_max .- pe_min .+ 1e-8)
    end

    if "pb_new" in names(df)
        pb = Float64.(coalesce.(df.pb_new, NaN))
        pb_min = runmin(pb, 60)
        pb_max = runmax(pb, 60)
        df.pb_rank = (pb .- pb_min) ./ (pb_max .- pb_min .+ 1e-8)
    end

    if "tot_mv" in names(df)
        df.log_mv = log.(df.tot_mv .+ 1e-8)
    end

    if "turn" in names(df)
        turn = Float64.(coalesce.(df.turn, NaN))
        df.turn_ma5 = runmean(turn, 5)
        df.turn_volatility = runstd(turn, 20)
    end

    if "free_turnover" in names(df)
        df.free_turn_ma5 = runmean(Float64.(coalesce.(df.free_turnover, NaN)), 5)
    end

    # --- 财务基本面特征 (Financials) ---
    if "n_income_attr_p" in names(df) && "total_revenue" in names(df)
        net_income = Float64.(coalesce.(df.n_income_attr_p, NaN))
        revenue = Float64.(coalesce.(df.total_revenue, NaN))
        
        # 净利率 (Net Profit Margin)
        df.net_profit_margin = net_income ./ (revenue .+ 1e-8)
        
        # 资产负债率 (Debt to Asset Ratio)
        if "total_assets" in names(df) && "total_liab" in names(df)
             assets = Float64.(coalesce.(df.total_assets, NaN))
             liabs = Float64.(coalesce.(df.total_liab, NaN))
             df.debt_to_asset_ratio = liabs ./ (assets .+ 1e-8)
        end
        
        # ROE (Return on Equity)
        if "total_hldr_eqy_exc_min_int" in names(df)
            equity = Float64.(coalesce.(df.total_hldr_eqy_exc_min_int, NaN))
            df.roe = net_income ./ (equity .+ 1e-8)
        end
        
        # 盈利质量 (Earnings Quality)
        if "n_cashflow_act" in names(df)
            cashflow = Float64.(coalesce.(df.n_cashflow_act, NaN))
            df.earnings_quality = cashflow ./ (net_income .+ 1e-8)
        end
    end

    # 资金流向
    if "buy_elg_vol" in names(df) && "sell_elg_vol" in names(df)
        buy_lg = Float64.(coalesce.(df.buy_lg_vol, 0.0))
        buy_elg = Float64.(coalesce.(df.buy_elg_vol, 0.0))
        sell_lg = Float64.(coalesce.(df.sell_lg_vol, 0.0))
        sell_elg = Float64.(coalesce.(df.sell_elg_vol, 0.0))
        
        main_in = buy_lg .+ buy_elg
        main_out = sell_lg .+ sell_elg
        
        df.main_force_net_inflow = main_in .- main_out
        df.main_force_ratio = (main_in .+ main_out) ./ (volume .+ 1e-8)
        
        buy_sm = Float64.(coalesce.(df.buy_sm_vol, 0.0))
        sell_sm = Float64.(coalesce.(df.sell_sm_vol, 0.0))
        retail_net = buy_sm .- sell_sm
        df.retail_sentiment = retail_net ./ (volume .+ 1e-8)
        
        # 中单
        if "buy_md_vol" in names(df) && "sell_md_vol" in names(df)
            mid_in = Float64.(coalesce.(df.buy_md_vol, 0.0))
            mid_out = Float64.(coalesce.(df.sell_md_vol, 0.0))
            
            df.mid_force_net_inflow = mid_in .- mid_out
            df.mid_force_ratio = (mid_in .+ mid_out) ./ (volume .+ 1e-8)
            
            mid_net_ratio = df.mid_force_net_inflow ./ (volume .+ 1e-8)
            df.mid_force_net_inflow_ratio = mid_net_ratio
            
            main_net_ratio = df.main_force_net_inflow ./ (volume .+ 1e-8)
            df.mid_force_divergence = main_net_ratio .- mid_net_ratio
        end
    end

    if "net_mf_amount" in names(df) && "amount" in names(df)
        df.net_mf_amt_ratio = df.net_mf_amount ./ (df.amount .+ 1e-8)
    end

    if "high_52w" in names(df) && "low_52w" in names(df)
        df.price_position_52w = (close .- df.low_52w) ./ (df.high_52w .- df.low_52w .+ 1e-8)
    end

    # 补全缺失特征
    if !("close_lag1" in names(df))
        df.close_lag1 = shift_series(close, 1)
    end
    
    if !("volume_change" in names(df))
        vol_shift = shift_series(volume, 1)
        df.volume_change = (volume .- vol_shift) ./ (vol_shift .+ 1e-8)
    end
    
    if !("price_vol_corr" in names(df))
        df.price_vol_corr = running(cor, close, volume, 20)
    end
    
    if !("excess_return" in names(df))
        df.excess_return = pct_change
    end

    # --- ChanLun Integration ---
    # 1. 包含处理
    indices = collect(1:nrow(df))
    merged_h, merged_l, merged_idx = process_kline_inclusion(high, low, indices)
    
    # 2. 识别分型
    f_types = identify_fractals(merged_h, merged_l)
    
    # 3. 映射回 DataFrame
    is_fractal = zeros(Int8, n_rows)
    for i in 1:length(f_types)
        if f_types[i] != 0
            orig_idx = merged_idx[i]
            is_fractal[orig_idx] = f_types[i]
        end
    end
    df[!, "is_fractal"] = is_fractal
    
    # 4. 识别笔 (Strokes)
    strokes = identify_strokes(f_types, merged_h, merged_l, merged_idx)
    
    stroke_dir = zeros(Int8, n_rows)
    for s in strokes
        stroke_dir[s.start_idx] = s.type
    end
    df[!, "stroke_dir"] = stroke_dir
    
    # 5. 计算中枢 (Pivots)
    pivots = calculate_pivots(strokes)
    
    prm_values = zeros(Float64, n_rows)
    if !isempty(pivots)
        current_pivot_idx = 1
        for i in 1:n_rows
            if current_pivot_idx < length(pivots)
                next_p = pivots[current_pivot_idx+1]
                if i >= next_p.start_idx
                    current_pivot_idx += 1
                end
            end
            p = pivots[current_pivot_idx]
            if i >= p.start_idx
                 prm_values[i] = calculate_prm(close[i], p)
            end
        end
    end
    df[!, "prm"] = prm_values

    # 注意: 目标生成 (future_return, target) 已移至 Shared/Targets.jl
    # 请使用 add_future_returns!(df, DailyTimeframe()) 和 create_targets!(df, config)

    return df
end

# --- 辅助计算函数 (Type Stable) ---

function shift_series(x::Vector{Float64}, lag::Int)
    n = length(x)
    res = fill(NaN, n)
    if lag >= 0
        if n > lag
            @inbounds res[lag+1:end] = view(x, 1:n-lag)
        end
    else
        lag = -lag
        if n > lag
            @inbounds res[1:end-lag] = view(x, lag+1:n)
        end
    end
    return res
end

function calculate_rsi(prices::Vector{Float64}, window::Int)
    n = length(prices)
    rsi = fill(NaN, n)
    if n < 2 return rsi end

    deltas = diff(prices)
    alpha = 1.0 / window
    
    d = deltas[1]
    avg_gain = d > 0 ? d : 0.0
    avg_loss = d < 0 ? -d : 0.0
    
    rs = avg_loss == 0 ? Inf : avg_gain / avg_loss
    rsi[2] = 100.0 - (100.0 / (1.0 + rs))

    @inbounds for i in 2:length(deltas)
        d = deltas[i]
        gain = d > 0 ? d : 0.0
        loss = d < 0 ? -d : 0.0
        
        avg_gain = alpha * gain + (1 - alpha) * avg_gain
        avg_loss = alpha * loss + (1 - alpha) * avg_loss
        
        rs = avg_loss == 0 ? Inf : avg_gain / avg_loss
        rsi[i+1] = 100.0 - (100.0 / (1.0 + rs))
    end
    return rsi
end

function calculate_ema(prices::Vector{Float64}, window::Int)
    n = length(prices)
    ema = fill(NaN, n)
    if n < window return ema end
    
    k = 2.0 / (window + 1)
    sum_val = 0.0
    @inbounds for i in 1:window
        sum_val += prices[i]
    end
    ema[window] = sum_val / window
    
    @inbounds for i in (window+1):n
        ema[i] = (prices[i] - ema[i-1]) * k + ema[i-1]
    end
    return ema
end

function calculate_macd(prices::Vector{Float64}, fast::Int, slow::Int, signal::Int)
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd_line = ema_fast .- ema_slow
    
    first_valid = slow
    signal_line = fill(NaN, length(prices))
    
    if length(prices) > first_valid + signal
        valid_macd = view(macd_line, first_valid:length(macd_line))
        valid_signal = calculate_ema(Vector(valid_macd), signal)
        signal_line[first_valid:end] = valid_signal
    end
    
    hist_line = macd_line .- signal_line
    return macd_line, signal_line, hist_line
end

function calculate_bbands(prices::Vector{Float64}, length::Int, std_dev::Float64)
    m = runmean(prices, length)
    s = runstd(prices, length)
    u = m .+ (std_dev .* s)
    l = m .- (std_dev .* s)
    return u, m, l
end

function calculate_atr(high::Vector{Float64}, low::Vector{Float64}, close::Vector{Float64}, window::Int)
    n = length(close)
    tr = zeros(Float64, n)
    tr[1] = high[1] - low[1]
    @inbounds for i in 2:n
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    end
    return calculate_rma(tr, window)
end

function calculate_adx(high::Vector{Float64}, low::Vector{Float64}, close::Vector{Float64}, window::Int)
    n = length(close)
    adx = fill(NaN, n)
    if n < window * 2 return adx end
    
    tr = zeros(Float64, n)
    dm_plus = zeros(Float64, n)
    dm_minus = zeros(Float64, n)
    
    tr[1] = high[1] - low[1]
    
    @inbounds for i in 2:n
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
        
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        
        if up > down && up > 0
            dm_plus[i] = up
        else
            dm_plus[i] = 0.0
        end
        
        if down > up && down > 0
            dm_minus[i] = down
        else
            dm_minus[i] = 0.0
        end
    end
    
    atr = calculate_rma(tr, window)
    plus_di = 100.0 .* calculate_rma(dm_plus, window) ./ (atr .+ 1e-8)
    minus_di = 100.0 .* calculate_rma(dm_minus, window) ./ (atr .+ 1e-8)
    
    dx = 100.0 .* abs.(plus_di .- minus_di) ./ (plus_di .+ minus_di .+ 1e-8)
    adx = calculate_rma(dx, window)
    
    return adx
end

function calculate_rma(data::Vector{Float64}, window::Int)
    n = length(data)
    rma = fill(NaN, n)
    if n < window return rma end
    
    sum_val = 0.0
    @inbounds for i in 1:window
        sum_val += data[i]
    end
    rma[window] = sum_val / window
    
    alpha = 1.0 / window
    @inbounds for i in (window+1):n
        rma[i] = alpha * data[i] + (1 - alpha) * rma[i-1]
    end
    return rma
end

function calculate_stoch(high::Vector{Float64}, low::Vector{Float64}, close::Vector{Float64}, k_win::Int, d_win::Int, smooth::Int)
    n = length(close)
    k_line = fill(NaN, n)
    d_line = fill(NaN, n)
    if n < k_win return k_line, d_line end
    
    lowest_low = runmin(low, k_win)
    highest_high = runmax(high, k_win)
    
    raw_k = 100.0 .* (close .- lowest_low) ./ (highest_high .- lowest_low .+ 1e-8)
    
    if smooth > 1
        k_line = runmean(raw_k, smooth)
    else
        k_line = raw_k
    end
    
    d_line = runmean(k_line, d_win)
    return k_line, d_line
end

end # module
