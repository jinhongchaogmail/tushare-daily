using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using JuliaCore
using JuliaCore.Shared.Types: DailyTimeframe, Daily20dTimeframe, AbsoluteReturn, ExcessReturn, timeframe_name, target_type_name
using JuliaCore.Training.Daily
using ArgParse
using YAML
using JSON
using Dates

# 模型变体映射
const MODEL_VARIANTS = Dict(
    "5d_absolute"  => "config/config_5d_absolute.yaml",
    "5d_excess"    => "config/config_5d_excess.yaml",
    "20d_absolute" => "config/config_20d_absolute.yaml",
    "20d_excess"   => "config/config_20d_excess.yaml",
)

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--model", "-m"
            help = "Model variant: 5d_absolute, 5d_excess, 20d_absolute, 20d_excess"
            arg_type = String
            default = "5d_absolute"
        "--config"
            help = "Override config file path"
            arg_type = String
            default = nothing
        "--trials"
            help = "Override number of Optuna trials"
            arg_type = Int
            default = nothing
        "--data-dir"
            help = "Override data directory"
            arg_type = String
            default = nothing
        "--limit"
            help = "Limit number of files to process"
            arg_type = Int
            default = nothing
        "--output-dir"
            help = "Override output directory for models and results"
            arg_type = String
            default = nothing
        "-v", "--verbose"
            help = "Verbose output"
            action = :store_true
    end

    return parse_args(s)
end

"""
    load_config_with_inherit(config_path::String) -> Dict

加载配置文件，支持 inherit 继承机制。
"""
function load_config_with_inherit(config_path::String)
    base_dir = dirname(config_path)
    
    if !isfile(config_path)
        # 尝试相对于脚本目录
        config_path = joinpath(@__DIR__, "..", config_path)
        base_dir = dirname(config_path)
    end
    
    if !isfile(config_path)
        error("Config file not found: $config_path")
    end
    
    config = YAML.load_file(config_path)
    
    # 处理继承
    if haskey(config, "inherit")
        parent_path = joinpath(base_dir, config["inherit"])
        parent_config = load_config_with_inherit(parent_path)
        
        # 深度合并：子配置覆盖父配置
        config = deep_merge(parent_config, config)
        delete!(config, "inherit")
    end
    
    return config
end

"""
    deep_merge(base::Dict, override::Dict) -> Dict

深度合并两个字典，override 中的值覆盖 base。
"""
function deep_merge(base::Dict, override::Dict)
    result = copy(base)
    for (k, v) in override
        if haskey(result, k) && isa(result[k], Dict) && isa(v, Dict)
            result[k] = deep_merge(result[k], v)
        else
            result[k] = v
        end
    end
    return result
end

"""
    save_training_snapshot(config::Dict, output_dir::String)

保存训练配置快照。
"""
function save_training_snapshot(config::Dict, output_dir::String)
    mkpath(output_dir)
    
    # 保存完整配置快照
    config_snapshot_path = joinpath(output_dir, "config_snapshot.yaml")
    open(config_snapshot_path, "w") do f
        YAML.write(f, config)
    end
    println("✅ 配置快照已保存: $config_snapshot_path")
    
    # 保存训练时间戳
    timestamp_path = joinpath(output_dir, "last_trained.txt")
    open(timestamp_path, "w") do f
        write(f, Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    end
end

function main()
    args = parse_commandline()
    
    # 确定配置文件路径
    config_path = if args["config"] !== nothing
        args["config"]
    elseif haskey(MODEL_VARIANTS, args["model"])
        MODEL_VARIANTS[args["model"]]
    else
        error("Unknown model variant: $(args["model"]). Available: $(keys(MODEL_VARIANTS))")
    end
    
    # 加载配置（支持继承）
    config = load_config_with_inherit(config_path)
    
    # 命令行参数覆盖配置
    period = get(config["target"], "period", 5)
    n_trials = something(args["trials"], get(config["optuna"], "n_trials", 20))
    data_dir = something(args["data-dir"], get(config["data"], "dir", "data/daily"))
    limit_files = something(args["limit"], get(config["data"], "limit_files", nothing))
    target_type_str = get(config["target"], "type", "absolute")
    index_code = get(config["target"], "index_code", "000001.SH")
    
    # 选择时间框架
    timeframe = period == 20 ? Daily20dTimeframe() : DailyTimeframe()
    
    # 选择目标类型
    target_type = if target_type_str == "excess"
        ExcessReturn(index_code=index_code)
    else
        AbsoluteReturn()
    end
    
    # 确定输出目录 (命令行 > 配置 > 默认)
    output_dir = if args["output-dir"] !== nothing
        args["output-dir"]
    else
        # 使用模型变体名称构建输出目录
        model_name = args["model"]
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        joinpath("optuna_results", "daily_$(model_name)_$timestamp")
    end
    
    # 打印模型变体信息
    println()
    println("🚀 训练模型变体: $(args["model"])")
    println("   预测周期: $(period) 天")
    println("   目标类型: $(target_type_str)")
    println("   输出目录: $output_dir")
    println()
    
    # 运行训练 (传递 output_dir)
    study = run_daily_optimization(
        n_trials=n_trials,
        data_dir=data_dir,
        limit_files=limit_files,
        timeframe=timeframe,
        target_type=target_type,
        config=config,
        output_dir=output_dir
    )
    
    # 保存配置快照
    if study !== nothing
        save_training_snapshot(config, output_dir)
    end
end

main()
