using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using JuliaCore
using JuliaCore.Training.Min5
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--trials"
            help = "Number of Optuna trials"
            arg_type = Int
            default = 20
        "--data-dir"
            help = "Path to data directory"
            arg_type = String
            default = "data/min5/raw"
        "--limit"
            help = "Limit number of files to process"
            arg_type = Int
            default = nothing
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()
    println("Starting Min5 Strategy Optimization...")
    run_min5_optimization(
        n_trials=args["trials"],
        data_dir=args["data-dir"],
        limit_files=args["limit"]
    )
end

main()
