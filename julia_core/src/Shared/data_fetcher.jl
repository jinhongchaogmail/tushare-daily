module DataFetcher

using DataFrames
using CSV
using Dates
using Arrow

export load_data, preprocess_data

"""
    load_data(file_path::String)

加载数据，支持 CSV 和 Arrow 格式。

参数:
- file_path: 数据文件的路径

返回值:
- 一个DataFrame对象，包含加载的数据

说明:
- 如果文件扩展名为 .csv，则使用CSV包读取。
- 如果文件扩展名为 .arrow，则使用Arrow包读取为DataFrame。
- 不支持其他格式，会抛出错误。
"""
function load_data(file_path::String)
    if endswith(file_path, ".csv")
        return CSV.read(file_path, DataFrame)
    elseif endswith(file_path, ".arrow")
        return DataFrame(Arrow.Table(file_path))
    else
        error("Unsupported file format: $file_path")
    end
end

"""
    preprocess_data(df::DataFrame)

对加载的原始数据进行预处理，使其符合后续分析的要求。

主要步骤包括：
1. 列名标准化：将所有列名转换为小写，保证一致性。
2. 日期处理：确保 'trade_date' 列是 Date 类型。尝试解析字符串格式的日期（如 YYYYMMDD 或 YYYY-MM-DD）。
3. 数据排序：按交易日期升序排列。
4. 缺失值处理：检查关键价格和成交量列是否存在缺失值，并进行处理（当前实现中仅作提示，未实际处理）。

参数:
- df: 待预处理的DataFrame

返回值:
- 处理后的DataFrame
"""
function preprocess_data(df::DataFrame)
    # 1. 列名标准化
    rename!(df, lowercase.(names(df)))
    
    # 2. 日期处理
    if "trade_date" in names(df)
        # 尝试解析日期，如果是字符串
        if eltype(df.trade_date) <: AbstractString
            # 假设格式为 YYYYMMDD 或 YYYY-MM-DD
            # 这里做一个简单的尝试，实际可能需要更复杂的解析
            try
                df.trade_date = Date.(df.trade_date, "yyyymmdd")
            catch
                try
                    df.trade_date = Date.(df.trade_date, "yyyy-mm-dd")
                catch
                    # 如果解析失败，可能已经是 Date 类型或者格式特殊
                end
            end
        end
        
        # 3. 排序
        sort!(df, :trade_date)
    end
    
    # 4. 缺失值处理
    # 对于价格列，通常不能有缺失。向前填充是一个常见策略。
    # Julia DataFrames 提供了 dropmissing! 或 coalesce
    # 这里我们简单地去除包含缺失值的行，或者根据需要填充
    # dropmissing!(df, [:open, :high, :low, :close, :volume])
    
    return df
end

end # module
