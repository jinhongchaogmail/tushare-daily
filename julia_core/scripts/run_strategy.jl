using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using JuliaCore
using JuliaCore.Prediction.Daily.RunStrategy
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--stocks"
            help = "指定要处理的股票代码，用逗号分隔"
            arg_type = String
            default = nothing
        "--batch-size"
            help = "每批次处理股票数量"
            arg_type = Int
            default = 10
        "--parallel-workers"
            help = "并行处理线程数"
            arg_type = Int
            default = 2
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()
    
    # 检查环境变量
    if !haskey(ENV, "TUSHARE_TOKEN")
        println("⚠️ 警告: 未设置 TUSHARE_TOKEN 环境变量")
    end
    
    run_strategy(
        stocks=args["stocks"],
        batch_size=args["batch-size"],
        parallel_workers=args["parallel-workers"]
    )
end

main()
