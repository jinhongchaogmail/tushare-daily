using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using JuliaCore
using JuliaCore.Prediction.Min5.RunStrategy
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--stocks"
            help = "指定要处理的股票代码，用逗号分隔"
            arg_type = String
            default = nothing
        "--top-k"
            help = "显示前K个结果"
            arg_type = Int
            default = 5
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()
    
    # 检查环境变量
    if !haskey(ENV, "TUSHARE_TOKEN")
        println("⚠️ 警告: 未设置 TUSHARE_TOKEN 环境变量")
    end
    
    run_min5_strategy(
        stocks=args["stocks"],
        top_k=args["top-k"]
    )
end

main()
