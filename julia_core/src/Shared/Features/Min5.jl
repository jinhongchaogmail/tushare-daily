module Min5

using DataFrames
using Statistics
using RollingFunctions
using Dates
using ...ChanLun
using ..Utils

export apply_min5_features!

"""
    apply_min5_features!(df::DataFrame; params::Dict=Dict())

Generate features for 5-minute frequency data.
"""
function apply_min5_features!(df::DataFrame; params::Dict=Dict())
    # Extract parameters with defaults
    rsi_window = get(params, "rsi_window", 14)
    macd_fast = get(params, "macd_fast", 12)
    macd_slow = get(params, "macd_slow", 26)
    macd_signal = get(params, "macd_signal", 9)
    bb_length = get(params, "bb_length", 20)
    bb_std = Float64(get(params, "bb_std", 2.0))
    atr_length = get(params, "atr_length", 14)

    # Ensure sorted by time
    if "trade_time" in names(df)
        sort!(df, :trade_time)
    elseif "trade_date" in names(df)
        sort!(df, :trade_date)
    end

    # Extract vectors for calculation
    close = Float64.(df.close)
    high = Float64.(df.high)
    low = Float64.(df.low)
    volume = Float64.(df.volume)
    n_rows = length(close)

    # --- Moving Averages ---
    # 1 hour = 12 periods, 1 day = 48 periods, 1 week = 240 periods
    df[!, "SMA_12"] = runmean(close, 12)
    df[!, "SMA_48"] = runmean(close, 48)
    df[!, "SMA_240"] = runmean(close, 240)
    df[!, "SMA_960"] = runmean(close, 960) # 1 month (20 days) - NEW for T+1 Trend

    # --- Momentum ---
    df[!, "RSI_$(rsi_window)"] = calculate_rsi(close, rsi_window)
    
    macd, signal, hist = calculate_macd(close, macd_fast, macd_slow, macd_signal)
    df[!, "MACD_$(macd_fast)_$(macd_slow)_$(macd_signal)"] = macd
    df[!, "MACDs_$(macd_fast)_$(macd_slow)_$(macd_signal)"] = signal
    df[!, "MACDh_$(macd_fast)_$(macd_slow)_$(macd_signal)"] = hist
    
    df[!, "ADX_14"] = calculate_adx(high, low, close, 14)

    # --- Volatility ---
    bb_u, bb_m, bb_l = calculate_bbands(close, bb_length, bb_std)
    df[!, "BBU_$(bb_length)_$(bb_std)"] = bb_u
    df[!, "BBM_$(bb_length)_$(bb_std)"] = bb_m
    df[!, "BBL_$(bb_length)_$(bb_std)"] = bb_l
    
    df[!, "ATRr_$(atr_length)"] = calculate_atr(high, low, close, atr_length)

    # --- New T+0 Features ---
    
    # 1. Bollinger Bandwidth
    df[!, "bb_bandwidth"] = (bb_u .- bb_l) ./ (bb_m .+ 1e-8)

    # 2. Light Speed MACD (5, 13, 1)
    macd_ls, signal_ls, hist_ls = calculate_macd(close, 5, 13, 1)
    df[!, "MACD_5_13_1"] = macd_ls
    df[!, "MACDs_5_13_1"] = signal_ls
    df[!, "MACDh_5_13_1"] = hist_ls

    # 3. Intraday VWAP
    if "trade_time" in names(df)
        try
            tp = (high .+ low .+ close) ./ 3.0
            vp = tp .* volume
            
            # Group by date to reset VWAP daily
            dates = Date.(df.trade_time)
            unique_dates = unique(dates)
            
            vwap = zeros(Float64, n_rows)
            
            for d in unique_dates
                mask = dates .== d
                daily_vp = vp[mask]
                daily_vol = volume[mask]
                
                daily_cum_vp = cumsum(daily_vp)
                daily_cum_vol = cumsum(daily_vol)
                
                vwap[mask] = daily_cum_vp ./ (daily_cum_vol .+ 1e-8)
            end
            
            df[!, "vwap"] = vwap
            df[!, "dist_vwap"] = (close .- vwap) ./ (vwap .+ 1e-8)
        catch e
            # println("VWAP calculation failed: $e")
        end
    end

    # Volume
    # OBV (Simplified)
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
    df[!, "OBV"] = obv
    
    df[!, "CMF_20"] = calculate_cmf(high, low, close, volume, 20)

    # --- Advanced Features ---
    
    # 1. Normalized MACD
    df[!, "MACD_$(macd_fast)_$(macd_slow)_$(macd_signal)_norm"] = hist ./ (close .+ 1e-8)

    # 2. Price Position
    for window in [48, 240]
        roll_low = runmin(low, window)
        roll_high = runmax(high, window)
        df[!, "price_pos_$(window)"] = (close .- roll_low) ./ (roll_high .- roll_low .+ 1e-8)
    end

    # 3. Volatility (Rolling Std Dev of Returns)
    returns = [0.0; diff(close) ./ close[1:end-1]]
    df[!, "returns"] = returns
    for window in [12, 48]
        df[!, "volatility_$(window)"] = runstd(returns, window)
    end

    # 4. Volume Ratio
    for window in [12, 48]
        vol_sma = runmean(volume, window)
        df[!, "vol_ratio_$(window)"] = volume ./ (vol_sma .+ 1e-8)
    end

    # 5. RSI of Volume
    df[!, "rsi_vol"] = calculate_rsi(volume, rsi_window)

    # --- Chan Lun Features ---
    try
        # 1. 包含处理
        # Julia ChanLun expects vectors
        indices = collect(1:n_rows)
        merged_h, merged_l, merged_idx = process_kline_inclusion(high, low, indices)
        
        # 2. 分型
        f_types = identify_fractals(merged_h, merged_l)
        
        # Map back to DataFrame
        is_fractal = zeros(Int, n_rows)
        # fractal_vol = zeros(Float64, n_rows) # Not implemented in ChanLun.jl yet?
        
        for i in 1:length(f_types)
            if f_types[i] != 0
                orig_idx = merged_idx[i]
                is_fractal[orig_idx] = f_types[i]
            end
        end
        
        # Shift features by 1 to avoid look-ahead bias
        df[!, "is_fractal"] = shift_series(Float64.(is_fractal), 1)
        df[!, "is_fractal"] = replace(df[!, "is_fractal"], NaN => 0.0)
        
        # 3. 笔
        strokes = identify_strokes(f_types, merged_h, merged_l, merged_idx)
        
        # 4. 中枢
        pivots = calculate_pivots(strokes)
        
        # PRM
        prm_values = zeros(Float64, n_rows)
        if !isempty(pivots)
            # Simple mapping: find the latest pivot for each bar
            # This is O(N*M) if naive, but pivots are sorted.
            # Optimized approach:
            pivot_idx = 1
            current_pivot = pivots[1]
            
            for i in 1:n_rows
                # Check if we moved past the current pivot's start
                while pivot_idx < length(pivots) && i >= pivots[pivot_idx+1].start_idx
                    pivot_idx += 1
                    current_pivot = pivots[pivot_idx]
                end
                
                if i >= current_pivot.start_idx
                    prm_values[i] = calculate_prm(close[i], current_pivot)
                end
            end
        end
        df[!, "prm"] = prm_values
        
    catch e
        # println("Chan Lun feature generation failed: $e")
    end

    # --- MACD Divergence ---
    df[!, "macd_divergence"] = zeros(Int, n_rows)
    macd_hist_col = "MACDh_$(macd_fast)_$(macd_slow)_$(macd_signal)"
    if macd_hist_col in names(df)
        window = 20
        roll_high = runmax(high, window)
        roll_low = runmin(low, window)
        roll_macd_max = runmax(df[!, macd_hist_col], window)
        roll_macd_min = runmin(df[!, macd_hist_col], window)
        
        # Vectorized check
        is_new_high = high .>= roll_high
        is_macd_weak = (df[!, macd_hist_col] .< roll_macd_max) .& (df[!, macd_hist_col] .> 0)
        
        df[is_new_high .& is_macd_weak, "macd_divergence"] .= -1
        
        is_new_low = low .<= roll_low
        is_macd_strong = (df[!, macd_hist_col] .> roll_macd_min) .& (df[!, macd_hist_col] .< 0)
        
        df[is_new_low .& is_macd_strong, "macd_divergence"] .= 1
    end

    # --- Custom Intraday Features ---
    if "trade_time" in names(df)
        # Julia Dates handling
        # Assuming trade_time is DateTime
        hours = Hour.(df.trade_time)
        minutes = Minute.(df.trade_time)
        
        df[!, "hour"] = Dates.value.(hours)
        df[!, "minute"] = Dates.value.(minutes)
        
        df[!, "is_market_open"] = ((df.hour .== 9) .& (df.minute .>= 30)) .|> Int
        df[!, "is_market_close"] = ((df.hour .== 14) .& (df.minute .>= 30)) .|> Int
    end

    if "SMA_48" in names(df)
        df[!, "dist_ma_1d"] = (close .- df.SMA_48) ./ (df.SMA_48 .+ 1e-8)
    end
    
    df[!, "return_1period"] = returns
    
    # return_12periods
    close_shift_12 = shift_series(close, 12)
    df[!, "return_12periods"] = (close .- close_shift_12) ./ (close_shift_12 .+ 1e-8)
    
    # volatility_12p
    df[!, "volatility_12p"] = runstd(returns, 12)
    
    # vol_surge
    vol_ma_12 = runmean(volume, 12) .+ 1e-8
    df[!, "vol_surge"] = volume ./ vol_ma_12
    
    # --- Bars Since Last Fractal ---
    # df['is_fractal'] is already shifted by 1
    # We need to count bars since is_fractal == 1 (Top)
    # Note: Python code only checked Top fractal (is_fractal == 1)
    
    bars_since = fill(100.0, n_rows)
    last_idx = -100
    
    for i in 1:n_rows
        if df[i, "is_fractal"] == 1.0
            last_idx = i
        end
        bars_since[i] = Float64(i - last_idx)
    end
    df[!, "bars_since_top"] = bars_since

    return df
end

end # module
