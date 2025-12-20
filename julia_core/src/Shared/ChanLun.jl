module ChanLun

using Statistics

# 定义轻量级 Struct，内存布局紧凑，速度极快
"""
Stroke (笔) 结构体：用于表示缠论中的“笔”，即由分型连接而成的价格走势段。
- start_idx: 笔的起始K线索引（在原始数据中的位置）
- end_idx: 笔的结束K线索引
- start_price: 笔的起始价格
- end_price: 笔的结束价格
- type: 笔的类型，1表示向上的笔，-1表示向下的笔。使用Int8节省内存。
"""
struct Stroke
    start_idx::Int
    end_idx::Int
    start_price::Float64
    end_price::Float64
    type::Int8 # 1: Up, -1: Down, 使用 Int8 节省内存
end

"""
Pivot (中枢) 结构体：用于表示缠论中的“中枢”，即由连续三笔重叠部分形成的价格区间。
- start_idx: 中枢的起始索引（第一笔开始的位置）
- end_idx: 中枢的结束索引（第三笔结束的位置）
- zg: 中枢高点 (ZG)，即中枢区间的上边界
- zd: 中枢低点 (ZD)，即中枢区间的下边界
- mean: 中枢均价，计算为 (zg + zd) / 2
- width: 中枢宽度，计算为 zg - zd
"""
struct Pivot
    start_idx::Int
    end_idx::Int
    zg::Float64   # 中枢高点 (High of Pivot)
    zd::Float64   # 中枢低点 (Low of Pivot)
    mean::Float64 # 中枢均价
    width::Float64 # 中枢宽度
end

export process_kline_inclusion, identify_fractals, identify_strokes, calculate_pivots, calculate_prm, Stroke, Pivot

"""
    process_kline_inclusion(highs, lows, indices)

处理K线包含关系 (标准缠论定义)
性能优化：预分配内存，避免 push!

参数:
- highs: 各K线的最高价组成的数组
- lows: 各K线的最低价组成的数组
- indices: 各K线在原始数据中的索引数组

返回值:
- 处理完包含关系后的最高价、最低价和索引数组
"""
function process_kline_inclusion(highs::AbstractVector{<:AbstractFloat}, lows::AbstractVector{<:AbstractFloat}, indices::AbstractVector{<:Integer})
    n = length(highs)
    if n == 0
        return Float64[], Float64[], Int[]
    end

    # 预分配最大可能空间
    out_high = Vector{eltype(highs)}(undef, n)
    out_low = Vector{eltype(lows)}(undef, n)
    out_idx = Vector{eltype(indices)}(undef, n)

    # 初始化第一根
    out_high[1] = highs[1]
    out_low[1] = lows[1]
    out_idx[1] = indices[1]
    
    count = 1
    
    @inbounds for i in 2:n
        curr_h = highs[i]
        curr_l = lows[i]
        curr_idx = indices[i]
        
        prev_h = out_high[count]
        prev_l = out_low[count]
        
        # 判断包含关系
        # 1. 前包含后 (Previous covers Current)
        prev_includes_curr = (prev_h >= curr_h) && (prev_l <= curr_l)
        # 2. 后包含前 (Current covers Previous)
        curr_includes_prev = (curr_h >= prev_h) && (curr_l <= prev_l)
        
        if prev_includes_curr || curr_includes_prev
            # 确定方向：根据合并后的 bar 与再前一根 bar 的关系
            direction = 1 # 默认向上
            if count >= 2
                prev_prev_h = out_high[count-1]
                # 严格定义：如果 prev_h > prev_prev_h，由于包含是延迟确认的，趋势由前两根非包含 K 线决定
                # 简化逻辑：比较 high 即可
                if prev_h < prev_prev_h
                    direction = -1
                end
            else
                # 第一根 K 线的特殊情况，使用当前与前一根的关系
                if curr_h < prev_h
                    direction = -1
                end
            end
            
            # 合并逻辑
            if direction == 1 # 向上趋势：高高低高 (取高点中的高点，低点中的高点)
                new_h = max(prev_h, curr_h)
                new_l = max(prev_l, curr_l)
            else # 向下趋势：低低低低 (取高点中的低点，低点中的低点)
                new_h = min(prev_h, curr_h)
                new_l = min(prev_l, curr_l)
            end
            
            # 原地更新 (Update the tail)
            out_high[count] = new_h
            out_low[count] = new_l
            # 时间通常取靠后的那根 (Current)
            out_idx[count] = curr_idx
        else
            # 无包含，推入新 K 线
            count += 1
            out_high[count] = curr_h
            out_low[count] = curr_l
            out_idx[count] = curr_idx
        end
    end
    
    # 调整数组大小到实际长度 (这一步开销很小)
    resize!(out_high, count)
    resize!(out_low, count)
    resize!(out_idx, count)
    
    return out_high, out_low, out_idx
end

"""
    identify_fractals(highs, lows)
识别分型，返回 Int8 数组

参数:
- highs: 经过包含处理后的K线最高价数组
- lows: 经过包含处理后的K线最低价数组

返回值:
- 一个与输入等长的Int8数组，其中：
  - 0: 不是分型
  - 1: 顶分型 (局部最高点)
  - -1: 底分型 (局部最低点)

说明:
- 顶分型：当前K线的高点同时高于前后两根K线的高点，且低点也高于前后两根K线的低点。
- 底分型：当前K线的低点同时低于前后两根K线的低点，且高点也低于前后两根K线的高点。
"""
function identify_fractals(highs::AbstractVector{<:AbstractFloat}, lows::AbstractVector{<:AbstractFloat})
    n = length(highs)
    fractal_type = zeros(Int8, n) # 使用 Int8
    
    if n < 3
        return fractal_type
    end
    
    @inbounds for i in 2:n-1
        h = highs[i]
        l = lows[i]
        
        # 顶分型：中间最高
        if h > highs[i-1] && h > highs[i+1] && l > lows[i-1] && l > lows[i+1]
            fractal_type[i] = 1
        # 底分型：中间最低
        elseif l < lows[i-1] && l < lows[i+1] && h < highs[i-1] && h < highs[i+1]
            fractal_type[i] = -1
        end
    end
    
    return fractal_type
end

"""
    identify_strokes(fractal_types, highs, lows, original_indices)
识别笔 (Strokes) - 返回 Vector{Stroke}

参数:
- fractal_types: 由identify_fractals函数生成的分型类型数组
- highs: K线最高价数组
- lows: K线最低价数组
- original_indices: K线在原始数据中的索引数组

返回值:
- 一个Stroke结构体的数组，表示识别出的所有“笔”

算法流程:
1. 首先找到所有分型的索引。
2. 遍历这些分型，将相邻的顶分型和底分型连接起来，形成“笔”。
3. 如果出现连续同向的分型（如两个顶分型），则保留更极端的那个（更高的顶或更低的底）。
4. 只有当两个分型之间的距离（K线数量）大于等于4时，才认为成功成笔。
"""
function identify_strokes(fractal_types::AbstractVector{Int8}, highs::AbstractVector{<:AbstractFloat}, lows::AbstractVector{<:AbstractFloat}, original_indices::AbstractVector{<:Integer})
    # 找到所有分型的索引 (在合并 K 线数组中的索引)
    fractal_indices = findall(!iszero, fractal_types)
    
    if length(fractal_indices) < 2
        return Stroke[]
    end
    
    strokes = Vector{Stroke}()
    sizehint!(strokes, length(fractal_indices) ÷ 2) # 预估笔的数量
    
    # 初始化
    curr_idx_in_arr = fractal_indices[1]
    curr_ft = fractal_types[curr_idx_in_arr]
    
    # 记录当前待定的分型极值 (用于处理连续同向分型)
    curr_extreme_h = highs[curr_idx_in_arr]
    curr_extreme_l = lows[curr_idx_in_arr]
    curr_orig_idx = original_indices[curr_idx_in_arr]
    
    for i in 2:length(fractal_indices)
        next_idx_in_arr = fractal_indices[i]
        next_ft = fractal_types[next_idx_in_arr]
        next_h = highs[next_idx_in_arr]
        next_l = lows[next_idx_in_arr]
        next_orig_idx = original_indices[next_idx_in_arr]
        
        # 1. 如果分型方向相同 (顶-顶 或 底-底)
        if curr_ft == next_ft
            # 延续逻辑：保留更极端的那个
            if curr_ft == 1 # 顶
                if next_h > curr_extreme_h
                    curr_extreme_h = next_h
                    curr_extreme_l = next_l # 伴随更新
                    curr_orig_idx = next_orig_idx
                    curr_idx_in_arr = next_idx_in_arr # 更新位置索引
                end
            else # 底
                if next_l < curr_extreme_l
                    curr_extreme_h = next_h
                    curr_extreme_l = next_l
                    curr_orig_idx = next_orig_idx
                    curr_idx_in_arr = next_idx_in_arr
                end
            end
            continue
        end
        
        # 2. 分型方向不同，检查成笔条件 (距离 > 3, 即中间至少有 3 根 K 线，或者 indices 差 >= 4)
        # 注意：为了与 Python 版本保持一致，这里使用原始索引差 (original_indices)
        # 标准缠论通常使用合并后的 K 线索引差，但 Python 版使用的是原始索引
        dist = next_orig_idx - curr_orig_idx
        
        if dist < 4
            # 距离不够，不成笔。
            # 这里有一个简化逻辑：如果不成笔，我们跳过 `next` 这个分型，继续找更远的反向分型
            # 这样 `curr` 保持不变
            continue
        end
        
        # 3. 有效成笔
        stroke_type = Int8(0)
        start_p = 0.0
        end_p = 0.0
        
        if curr_ft == 1 # 顶 -> 底 (向下笔)
            stroke_type = -1
            start_p = curr_extreme_h
            end_p = next_l # 下一笔的底
        else # 底 -> 顶 (向上笔)
            stroke_type = 1
            start_p = curr_extreme_l
            end_p = next_h # 下一笔的顶
        end
        
        push!(strokes, Stroke(curr_orig_idx, next_orig_idx, start_p, end_p, stroke_type))
        
        # 更新 Current 为 Next (这笔结束，作为下一笔的开始)
        curr_ft = next_ft
        curr_idx_in_arr = next_idx_in_arr
        curr_extreme_h = next_h
        curr_extreme_l = next_l
        curr_orig_idx = next_orig_idx
    end
    
    return strokes
end

"""
    calculate_pivots(strokes)
计算中枢 (Pivots) - 返回 Vector{Pivot}

参数:
- strokes: 由identify_strokes函数生成的笔的数组

返回值:
- 一个Pivot结构体的数组，表示识别出的所有“中枢”

算法流程:
1. 遍历笔的数组，每次取连续的三笔（i, i+1, i+2）。
2. 计算这三笔价格区间的最大重叠部分：
   - ZG (中枢高点) = min(三笔的最高点)
   - ZD (中枢低点) = max(三笔的最低点)
3. 如果 ZG > ZD，则形成有效中枢，并将其加入结果列表。
"""
function calculate_pivots(strokes::Vector{Stroke})
    n = length(strokes)
    if n < 3
        return Pivot[]
    end
    
    pivots = Vector{Pivot}()
    sizehint!(pivots, n - 2)
    
    @inbounds for i in 1:(n-2)
        s1 = strokes[i]
        s2 = strokes[i+1]
        s3 = strokes[i+2]
        
        # 确定每一笔的价格区间
        high1, low1 = max(s1.start_price, s1.end_price), min(s1.start_price, s1.end_price)
        high2, low2 = max(s2.start_price, s2.end_price), min(s2.start_price, s2.end_price)
        high3, low3 = max(s3.start_price, s3.end_price), min(s3.start_price, s3.end_price)
        
        # 中枢区间：取前三笔重叠部分
        # ZG = min(High1, High2, High3)
        # ZD = max(Low1, Low2, Low3)
        zg = min(high1, high2, high3)
        zd = max(low1, low2, low3)
        
        # 有效中枢条件：ZG > ZD
        if zg > zd
            width = zg - zd
            push!(pivots, Pivot(
                s1.start_idx, # 中枢起始索引 (第一笔开始)
                s3.end_idx,   # 中枢结束索引 (第三笔结束)
                zg,
                zd,
                (zg + zd) / 2,
                width
            ))
        end
    end
    
    return pivots
end

"""
    calculate_prm(price, pivot)
计算 PRM (Pivot Relative Momentum)，即价格相对于中枢的动量。

参数:
- price: 当前价格
- pivot: 一个Pivot结构体，代表当前有效的中枢

返回值:
- PRM 值，计算公式为 (price - pivot.mean) / pivot.width
  - 若PRM > 0，表示价格在中枢上方，且值越大表示偏离越多。
  - 若PRM < 0，表示价格在中枢下方，且绝对值越大表示偏离越多。
  - 若PRM ≈ 0，表示价格在中枢内部。
"""
function calculate_prm(price::Real, pivot::Pivot)
    if pivot.width == 0
        return 0.0
    end
    return (price - pivot.mean) / pivot.width
end

end # module
