module JuliaCore

using Reexport
using PythonCall
using DataFrames
using Parquet2
using Dates
using Printf

# 包含各个子模块
include("Shared/Shared.jl")
@reexport using .Shared

include("Training/Training.jl")
@reexport using .Training

include("Prediction/Prediction.jl")
@reexport using .Prediction

# 声明将要导入的Python模块对象
const cb = PythonCall.pynew()
const optuna = PythonCall.pynew()
const sklearn_metrics = PythonCall.pynew()

"""
    __init__()

Julia模块的初始化函数。当此模块被加载时，会自动调用此函数。
它负责初始化与Python的连接，并导入所需的Python库（如catboost, optuna）。
如果导入失败，会发出警告，但不会中断程序。
"""
function __init__()
    # 初始化 Python 模块
    try
        PythonCall.pycopy!(cb, pyimport("catboost"))
        PythonCall.pycopy!(optuna, pyimport("optuna"))
        PythonCall.pycopy!(sklearn_metrics, pyimport("sklearn.metrics"))
    catch e
        @warn "导入 Python 模块失败: $e"
    end
end

export run_pipeline, load_and_process_data

"""
    load_and_process_data(data_dir::String; limit::Int=0)

从指定目录加载所有Parquet格式的数据文件，并对每个文件应用特征工程。

这是一个多线程函数，可以并行处理多个股票的数据，极大地提高了效率。

参数:
- data_dir: 存放Parquet数据文件的目录路径
- limit: 可选参数，用于限制处理的文件数量（常用于调试）

返回值:
- 一个合并了所有股票特征的大型DataFrame

流程:
1. 遍历目录，筛选出.parquet文件。
2. 使用多线程（Threads.@threads）并行处理每个文件：
   - 读取Parquet文件到DataFrame
   - 确保列类型正确
   - 调用apply_technical_indicators!计算特征
3. 将所有处理好的DataFrame合并成一个大表。
"""
function load_and_process_data(data_dir::String; limit::Int=0)
    if !isdir(data_dir)
        println("❌ 数据目录不存在: $data_dir")
        return DataFrame()
    end

    files = readdir(data_dir, join=true)
    # 过滤 parquet
    files = filter(f -> endswith(f, ".parquet"), files)
    
    if limit > 0
        println("⚠️ 限制处理前 $limit 个文件")
        files = files[1:min(limit, length(files))]
    end
    
    println("🚀 发现 $(length(files)) 个文件，开始并行处理...")

    # 使用线程安全的集合来收集 DataFrame
    # 初始化为 nothing
    processed_dfs = Vector{Union{DataFrame, Nothing}}(nothing, length(files))

    Threads.@threads for i in 1:length(files)
        try
            # 读取
            ds = Parquet2.readfile(files[i])
            df = DataFrame(ds; copycols=true)
            
            # 确保所有列都是标准 Vector (处理 FillArrays 等特殊类型)
            for col in names(df)
                df[!, col] = collect(df[!, col])
            end
            
            if i == 1
                println("列名: ", names(df))
            end
            
            # 特征工程 (Julia 原生速度)
            # 注意：apply_technical_indicators! 是原地修改
            apply_technical_indicators!(df)
            
            # 简单的过滤逻辑
            if nrow(df) > 20
                 processed_dfs[i] = df
            end
        catch e
            println("处理文件 $(files[i]) 时出错: $e")
            # 忽略错误，保持静默或记录日志
        end
    end

    # 收集有效结果
    valid_dfs = DataFrame[]
    for res in processed_dfs
        if !isnothing(res)
            push!(valid_dfs, res)
        end
    end
    
    # 合并
    println("📚 正在合并数据...")
    if isempty(valid_dfs)
        println("⚠️ 没有有效数据被处理！")
        return DataFrame()
    end
    
    # 使用 cols=:union 允许列不一致 (填充 missing)
    full_df = vcat(valid_dfs..., cols=:union)
    return full_df
end

"""
    run_pipeline(data_dir::String)

运行完整的量化分析流程：数据加载 -> 特征工程 -> 模型训练。

这是项目的一个高层接口，将复杂的流程封装起来。

参数:
- data_dir: 数据目录路径

说明:
- 此函数首先调用load_and_process_data获取带特征的数据。
- 然后准备训练集X和标签y。
- 最后使用CatBoost进行模型训练，并保存模型文件。
- 注意：此函数中的模型训练是简化的演示，实际项目中应使用train_model函数进行超参优化。
"""
function run_pipeline(data_dir::String)
    # 加载数据
    df = load_and_process_data(data_dir)
    
    if nrow(df) == 0
        return
    end
    
    println("🔧 准备训练数据...")
    # prepare_training_data 现在主要负责清洗和类型转换
    df_train = prepare_training_data(df)
    
    println("🤖 开始 Optuna 寻优...")
    # 使用 Training.Daily.Train 中的 train_model
    best_params = train_model(df_train, n_trials=20) # 增加 trial 次数
    
    println("✅ 优化完成。最佳参数: $best_params")
    
    # TODO: 使用最佳参数训练最终模型并保存
    # 目前 train_model 已经包含了完整的 Optuna 流程
end

end # module
