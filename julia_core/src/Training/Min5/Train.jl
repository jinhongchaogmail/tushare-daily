module Train

using DataFrames
using PythonCall
using Statistics
using Random
using ....Shared.Features.Min5: apply_min5_features!
using ....Shared.DataFetcher: load_data_files

export run_min5_optimization

function run_min5_optimization(; n_trials=20, data_dir="data/min5/raw", limit_files=nothing)
    # Import Python libraries
    optuna = pyimport("optuna")
    catboost = pyimport("catboost")
    np = pyimport("numpy")
    sklearn_metrics = pyimport("sklearn.metrics")
    
    # Load Data (auto-detect format)
    println("Loading raw data from $data_dir...")
    raw_dfs = load_data_files(data_dir; limit=limit_files)
    
    if isempty(raw_dfs)
        println("No data found.")
        return
    end
    
    println("Loaded $(length(raw_dfs)) files.")

    # Define Objective Function
    function objective(trial)
        # Suggest Params and convert to Julia types
        params = Dict{String, Any}(
            "rsi_window" => pyconvert(Int, trial.suggest_int("rsi_window", 6, 30)),
            "macd_fast" => pyconvert(Int, trial.suggest_int("macd_fast", 8, 20)),
            "macd_slow" => pyconvert(Int, trial.suggest_int("macd_slow", 20, 40)),
            "macd_signal" => pyconvert(Int, trial.suggest_int("macd_signal", 5, 15)),
            "bb_length" => pyconvert(Int, trial.suggest_int("bb_length", 10, 30)),
            "bb_std" => pyconvert(Float64, trial.suggest_float("bb_std", 1.5, 2.5)),
            "atr_length" => pyconvert(Int, trial.suggest_int("atr_length", 10, 30)),
            
            "iterations" => pyconvert(Int, trial.suggest_int("iterations", 500, 1500)),
            "depth" => pyconvert(Int, trial.suggest_int("depth", 6, 10)),
            "learning_rate" => pyconvert(Float64, trial.suggest_float("learning_rate", 0.01, 0.1)),
            "l2_leaf_reg" => pyconvert(Float64, trial.suggest_float("l2_leaf_reg", 1e-8, 15.0, log=true)),
            "random_strength" => pyconvert(Float64, trial.suggest_float("random_strength", 1e-8, 10.0, log=true)),
            "bagging_temperature" => pyconvert(Float64, trial.suggest_float("bagging_temperature", 0.0, 1.0)),
            "border_count" => pyconvert(Int, trial.suggest_int("border_count", 32, 255)),
            "scale_pos_weight" => pyconvert(Float64, trial.suggest_float("scale_pos_weight", 0.1, 20.0))
        )
        
        processed_dfs = DataFrame[]
        
        for df in raw_dfs
            df_copy = copy(df)
            apply_min5_features!(df_copy; params=params)
            
            # Target Engineering (T+1 > Current)
            if "close" in names(df_copy)
                close = df_copy.close
                n = length(close)
                target = zeros(Int, n)
                shift_val = 48
                
                @inbounds for i in 1:(n - shift_val)
                    if close[i + shift_val] > close[i]
                        target[i] = 1
                    else
                        target[i] = 0
                    end
                end
                
                df_copy[!, "target"] = target
                
                # Remove last 48 rows (invalid targets) and initial rows (NaN features)
                # Simple approach: dropmissing
                dropmissing!(df_copy)
                
                push!(processed_dfs, df_copy)
            end
        end
        
        if isempty(processed_dfs)
            return 0.0
        end
        
        full_df = vcat(processed_dfs...)
        
        # Prepare X and y
        exclude_cols = ["ts_code", "trade_time", "target", "open", "high", "low", "close", "volume", "amount", "pre_close", "date", "future_close", "future_return_5d", "volatility_factor"]
        feature_cols = setdiff(names(full_df), exclude_cols)
        
        select!(full_df, [feature_cols; "target"])
        dropmissing!(full_df)
        
        if nrow(full_df) < 100
            return 0.0
        end
        
        X = full_df[:, feature_cols]
        y = full_df.target
        
        # Train/Test Split
        n_samples = nrow(full_df)
        split_idx = floor(Int, n_samples * 0.8)
        
        X_train = X[1:split_idx, :]
        y_train = y[1:split_idx]
        X_val = X[split_idx+1:end, :]
        y_val = y[split_idx+1:end]
        
        # Convert to Numpy
        X_train_np = np.array(Matrix(X_train))
        y_train_np = np.array(Vector(y_train))
        X_val_np = np.array(Matrix(X_val))
        y_val_np = np.array(Vector(y_val))
        
        # Train CatBoost
        model = catboost.CatBoostClassifier(
            iterations=params["iterations"],
            depth=params["depth"],
            learning_rate=params["learning_rate"],
            l2_leaf_reg=params["l2_leaf_reg"],
            random_strength=params["random_strength"],
            bagging_temperature=params["bagging_temperature"],
            border_count=params["border_count"],
            scale_pos_weight=params["scale_pos_weight"],
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=0,
            task_type="GPU",
            devices="0",
            allow_writing_files=false
        )
        
        model.fit(X_train_np, y_train_np, eval_set=(X_val_np, y_val_np), early_stopping_rounds=50)
        
        # Evaluate
        preds = model.predict_proba(X_val_np)
        preds_prob = pyconvert(Array, preds)[:, 2]
        
        auc = sklearn_metrics.roc_auc_score(y_val_np, preds_prob)
        
        return pyconvert(Float64, auc)
    end

    # Run Study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    println("Best params: ", study.best_params)
    return study
end

end
