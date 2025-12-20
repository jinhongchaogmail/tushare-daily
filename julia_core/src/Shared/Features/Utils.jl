module Utils

using Statistics
using RollingFunctions

export shift_series, calculate_rsi, calculate_ema, calculate_macd, calculate_bbands, calculate_atr, calculate_adx, calculate_rma, calculate_stoch, calculate_cmf

function calculate_cmf(high::Vector{Float64}, low::Vector{Float64}, close::Vector{Float64}, volume::Vector{Float64}, window::Int)
    n = length(close)
    cmf = fill(NaN, n)
    if n < window return cmf end
    
    mfm = ((close .- low) .- (high .- close)) ./ (high .- low .+ 1e-8)
    mfv = mfm .* volume
    
    sum_mfv = runmean(mfv, window) .* window
    sum_vol = runmean(volume, window) .* window
    
    cmf = sum_mfv ./ (sum_vol .+ 1e-8)
    return cmf
end

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
