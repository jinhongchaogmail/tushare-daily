
using PythonCall
using DataFrames
using Dates
using CSV
using Arrow

# Add project root to LOAD_PATH
push!(LOAD_PATH, joinpath(@__DIR__, "../src"))
using JuliaCore
using JuliaCore.Features

# Setup Python Environment
const sys = pyimport("sys")
const os = pyimport("os")
sys.path.append(joinpath(@__DIR__, "../..")) 
sys.path.append(joinpath(@__DIR__, "../../.venv/lib/python3.12/site-packages"))

const pd = pyimport("pandas")
const cb = pyimport("catboost")
const tushare = pyimport("xcsc_tushare")
const data_fetcher = pyimport("shared.data_fetcher")

# Configuration
const TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN")
const TS_SERVER = "http://116.128.206.39:7172"
const MODEL_PATH = joinpath(@__DIR__, "../../models/daily/catboost_daily.cbm")
const OUT_DIR = joinpath(@__DIR__, "../../prediction/daily/output")

# Initialize Tushare
if !pyis(TUSHARE_TOKEN, pybuiltins.None)
    tushare.set_token(TUSHARE_TOKEN)
    const pro = tushare.pro_api(server=TS_SERVER)
else
    println("⚠️ Warning: TUSHARE_TOKEN not found.")
    const pro = nothing
end

const CONFIG_PATH = joinpath(@__DIR__, "../../training/config.yaml")
const yaml = pyimport("yaml")

function load_config(path)
    if isfile(path)
        return pyconvert(Dict, yaml.safe_load(open(path)))
    else
        return Dict()
    end
end

const CONFIG = load_config(CONFIG_PATH)

"""
    get_stock_list(max_tickers::Int)

Fetch stock list from Tushare.
"""
function get_stock_list(max_tickers::Int)
    println("Fetching stock list...")
    if isnothing(pro)
        return ["000001.SZ", "600000.SH", "000002.SZ", "601398.SH", "601318.SH"]
    end
    
    try
        df = pro.stock_basic(exchange="", list_status="L", fields="ts_code")
        codes = pyconvert(Vector{String}, df["ts_code"].values)
        filter!(c -> endswith(c, ".SZ") || endswith(c, ".SH"), codes)
        
        if max_tickers > 0 && length(codes) > max_tickers
            return codes[1:max_tickers]
        end
        return codes
    catch e
        println("Error fetching stock list: $e")
        return ["000001.SZ"]
    end
end

function fetch_and_process_data(stock_list, start_date, end_date)
    all_dfs = DataFrame[]
    
    for (i, stock) in enumerate(stock_list)
        if i % 10 == 0
            println("Processing $i/$(length(stock_list)): $stock")
        end
        
        try
            # 1. Fetch Data (Python)
            py_df = data_fetcher.fetch_daily_data(pro, stock, start_date, end_date)
            if pyconvert(Bool, py_df.empty)
                continue
            end
            
            # 2. Convert to Julia DataFrame
            df = pyconvert(DataFrame, py_df)
            
            # 3. Feature Engineering (Julia)
            try
                apply_technical_indicators!(df)
            catch e
                println("Error calculating features for $stock: $e")
                continue
            end
            
            push!(all_dfs, df)
            
        catch e
            println("Error processing $stock: $e")
        end
    end
    
    if isempty(all_dfs)
        return nothing
    end
    
    return vcat(all_dfs...)
end

function predict_and_save(df)
    if !isfile(MODEL_PATH)
        println("Model not found at $MODEL_PATH")
        return
    end
    
    println("Loading model...")
    model = cb.CatBoostClassifier()
    model.load_model(MODEL_PATH)
    
    println("Preparing data...")
    # Select features
    exclude_cols = ["ts_code", "trade_date", "future_return_5d", "target_class"]
    feature_cols = [n for n in names(df) if eltype(df[!, n]) <: Number && !(n in exclude_cols)]
    
    # Fill missing
    for col in feature_cols
        df[!, col] = coalesce.(df[!, col], 0.0)
    end
    
    # Convert to Python
    data_dict = Dict{String, Any}()
    for col in feature_cols
        data_dict[col] = df[!, col]
    end
    py_X = pd.DataFrame(data_dict)
    
    # Reorder columns to match model
    # CatBoost is sensitive to column order if not using Pool with feature names
    # But we trained with DataFrame, so it should be fine if names match?
    # Ideally we should check model.feature_names_
    
    try
        model_features = model.feature_names_
        # Ensure all features exist
        for f in model_features
            f_str = pyconvert(String, f)
            if !haskey(data_dict, f_str)
                py_X[f] = 0.0
            end
        end
        py_X = py_X[model_features]
    catch e
        println("Warning: Could not align features with model: $e")
    end
    
    println("Predicting...")
    preds = model.predict(py_X)
    probs = model.predict_proba(py_X)
    
    df.pred_class = pyconvert(Vector{Int}, preds.flatten())
    # probs is (N, 3)
    probs_jl = pyconvert(Matrix{Float64}, probs)
    df.prob_down = probs_jl[:, 1]
    df.prob_flat = probs_jl[:, 2]
    df.prob_up = probs_jl[:, 3]
    
    # Save results
    mkpath(OUT_DIR)
    today_str = Dates.format(Dates.today(), "yyyymmdd")
    out_file = joinpath(OUT_DIR, "pred_$(today_str).csv")
    
    # Select output columns
    out_cols = ["ts_code", "trade_date", "close", "pred_class", "prob_up", "prob_down"]
    CSV.write(out_file, df[:, out_cols])
    println("Results saved to $out_file")
end

function main()
    println("Starting Julia Prediction Pipeline...")
    
    # 1. Get Stock List
    # Use a reasonable limit for prediction if not specified, or all
    max_files = get(get(CONFIG, "data", Dict()), "max_files_to_process", 500)
    stocks = get_stock_list(max_files)
    
    # Predict for last 60 days to see trends, or just today?
    # Usually prediction is for the latest available data.
    start_date = Dates.format(Dates.today() - Day(60), "yyyymmdd")
    end_date = Dates.format(Dates.today(), "yyyymmdd")
    
    # 2. Fetch
    df = fetch_and_process_data(stocks, start_date, end_date)
    
    if isnothing(df)
        println("No data.")
        return
    end
    
    # 3. Predict
    # We only care about the latest date for each stock usually, 
    # but for strategy validation we might want full history.
    # Let's predict all.
    predict_and_save(df)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
