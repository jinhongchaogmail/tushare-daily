import catboost as cb
import os

models = [
    "models/catboost_final_model.cbm",
    "optuna_results/run_20251202_174653/catboost_final_model.cbm"
]

for m_path in models:
    if os.path.exists(m_path):
        try:
            model = cb.CatBoostClassifier()
            model.load_model(m_path)
            print(f"=== Model: {m_path} ===")
            print(f"Feature count: {len(model.feature_names_)}")
            print(f"First 10 features: {model.feature_names_[:10]}")
            
            # Check for specific controversial features
            check_list = ['ma5', 'ma10', 'excess_return', 'volatility_factor']
            found = [f for f in check_list if f in model.feature_names_]
            print(f"Contains controversial features: {found}")
            print("-" * 30)
        except Exception as e:
            print(f"Error loading {m_path}: {e}")
    else:
        print(f"Model not found: {m_path}")
