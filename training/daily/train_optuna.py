
import os
import sys
import glob
import pandas as pd
import numpy as np
import optuna
import json
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import argparse

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.features.daily import apply_technical_indicators

# Global Lock for GPU (Threading lock is sufficient for Optuna n_jobs with threads)
gpu_lock = threading.Lock()

# --- Global Search Space Definition (Daily) ---
DEFAULT_BOUNDS = {
    # Feature Params (Dynamic)
    'rsi_window': {'type': 'int', 'min': 6, 'max': 30},
    'macd_fast': {'type': 'int', 'min': 8, 'max': 20},
    'macd_slow': {'type': 'int', 'min': 20, 'max': 60},
    'macd_signal': {'type': 'int', 'min': 5, 'max': 20},
    'bb_length': {'type': 'int', 'min': 10, 'max': 30},
    'bb_std': {'type': 'float', 'min': 1.5, 'max': 2.5},
    'atr_length': {'type': 'int', 'min': 10, 'max': 30},
    
    # Model Params
    'iterations': {'type': 'int', 'min': 1000, 'max': 3000},
    'depth': {'type': 'int', 'min': 4, 'max': 10},
    'learning_rate': {'type': 'float', 'min': 0.005, 'max': 0.10},
    'l2_leaf_reg': {'type': 'float', 'min': 1e-8, 'max': 20.0, 'log': True},
    'random_strength': {'type': 'float', 'min': 1e-8, 'max': 10.0, 'log': True},
    'bagging_temperature': {'type': 'float', 'min': 0.0, 'max': 1.0},
    'border_count': {'type': 'int', 'min': 32, 'max': 255},
    
    # Target Params
    'vol_multiplier': {'type': 'float', 'min': 0.5, 'max': 1.5}
}

def print_logic_summary():
    print("\n" + "="*60)
    print("   日线策略优化流程 (OPTUNA + CATBOOST)")
    print("="*60)
    print("1. [初始化] 从磁盘加载日线 Parquet 数据。")
    print("2. [寻优] 运行 Optuna 研究以寻找最佳超参数：")
    print("   - 特征参数: RSI/MACD/BB/ATR 窗口。")
    print("   - 模型参数: CatBoost 深度、学习率、正则化。")
    print("   - 目标参数: 用于动态标签的波动率乘数。")
    print("3. [逻辑] 对于每次试验 (Trial)：")
    print("   - 生成特征 (CPU 并行)。")
    print("   - 生成目标: 如果收益 > vol_multiplier * 20日波动率 则买入/卖出。")
    print("   - 切分: TimeSeriesSplit (3 折) 带间隔 (Gap)。")
    print("   - 缩放: RobustScaler (在训练集拟合，应用到验证集)。")
    print("   - 权重: 平衡权重 + Focal Loss (关注难分类样本)。")
    print("   - 训练: CatBoostClassifier (GPU/CPU)。")
    print("   - 评估: Macro F1 Score。")
    print("4. [报告] 使用最佳模型重训并输出详细指标。")
    print("="*60 + "\n")

def load_raw_data(data_dir='data/daily', limit_files=None):
    """
    Load raw parquet files without applying features.
    Returns a list of DataFrames.
    """
    files = glob.glob(os.path.join(data_dir, '*.parquet'))
    if not files:
        files = glob.glob(os.path.join(data_dir, '**', '*.parquet'), recursive=True)
        
    if limit_files:
        files = files[:limit_files]
        
    print(f"在 {data_dir} 中找到 {len(files)} 个文件")
    
    raw_dfs = []
    max_workers = min(os.cpu_count(), 8)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(pd.read_parquet, f) for f in files]
        
        iterator = as_completed(futures)
        if tqdm:
            iterator = tqdm(iterator, total=len(files), desc="加载原始数据")
            
        for future in iterator:
            try:
                df = future.result()
                if not df.empty:
                    if 'trade_date' in df.columns:
                        df['trade_date'] = pd.to_datetime(df['trade_date'])
                        df = df.set_index('trade_date').sort_index()
                    raw_dfs.append(df)
            except Exception:
                pass
                
    return raw_dfs

def process_single_df_wrapper(args):
    df, params = args
    try:
        df_copy = df.copy()
        
        # Apply features with dynamic params
        df_copy = apply_technical_indicators(df_copy, params=params)
        
        if df_copy is None or df_copy.empty:
            return None
            
        # Calculate Volatility Factor (if not present)
        if 'volatility_factor' not in df_copy.columns:
            df_copy['volatility_factor'] = df_copy['close'].pct_change().rolling(window=20).std()
            
        # Target Generation Base (5d return)
        future_period = 5
        df_copy['future_close'] = df_copy['close'].shift(-future_period)
        df_copy['future_return_5d'] = (df_copy['future_close'] - df_copy['close']) / df_copy['close']
        
        # Replace inf with nan and drop
        df_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_copy.dropna(inplace=True)
        
        return df_copy
    except Exception:
        return None

def process_features(raw_dfs, feat_params):
    """
    Apply features with specific parameters to raw DataFrames.
    """
    processed_dfs = []
    tasks = [(df, feat_params) for df in raw_dfs]
    
    # Use fewer workers for feature gen to avoid OOM if running parallel trials
    max_workers = min(os.cpu_count(), 8)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_df_wrapper, task) for task in tasks]
        
        iterator = as_completed(futures)
        if tqdm and len(tasks) > 100:
             iterator = tqdm(iterator, total=len(tasks), desc="计算特征中", leave=False)
             
        for future in iterator:
            try:
                res = future.result()
                if res is not None and not res.empty:
                    processed_dfs.append(res)
            except Exception:
                pass
            
    if not processed_dfs:
        return pd.DataFrame()
        
    return pd.concat(processed_dfs, ignore_index=True)

def create_targets(df, vol_multiplier):
    """
    Create 3-class targets based on dynamic volatility threshold.
    Logic ported from train.py
    """
    dfc = df.copy()
    target_col = 'future_return_5d'
    
    # Dynamic Threshold
    vol_raw = dfc['volatility_factor'].fillna(0.02)
    dynamic_threshold = vol_multiplier * vol_raw
    dynamic_threshold = dynamic_threshold.clip(0.005, 0.05) # Clip to reasonable range
    
    offset = 0.0
    lower_bound = offset - dynamic_threshold
    upper_bound = offset + dynamic_threshold
    
    dfc['target'] = 0 # Hold (Class 0 -> Mapped to 1 later)
    
    # Class 1: Buy (Mapped to 2)
    dfc.loc[dfc[target_col] > upper_bound, 'target'] = 1
    
    # Class -1: Sell (Mapped to 0)
    dfc.loc[dfc[target_col] < lower_bound, 'target'] = -1
    
    return dfc

def get_suggested_params(trial, bounds):
    params = {}
    for key, config in bounds.items():
        if config['type'] == 'int':
            params[key] = trial.suggest_int(key, config['min'], config['max'])
        elif config['type'] == 'float':
            log = config.get('log', False)
            params[key] = trial.suggest_float(key, config['min'], config['max'], log=log)
    return params

def train_gpu_worker(model_params, X_train, y_train, w_train, X_val, y_val):
    """
    Worker function to train model on GPU.
    """
    try:
        train_pool = Pool(X_train, y_train, weight=w_train)
        val_pool = Pool(X_val, y_val)
        
        model = CatBoostClassifier(**model_params)
        model.fit(train_pool, eval_set=val_pool, verbose=0, early_stopping_rounds=50)
        
        return model
    except Exception as e:
        print(f"GPU 工作线程失败: {e}")
        return None

def objective(trial, raw_dfs, bounds, use_gpu=False):
    # 1. Suggest Params
    all_params = get_suggested_params(trial, bounds)
    
    feat_keys = ['rsi_window', 'macd_fast', 'macd_slow', 'macd_signal', 'bb_length', 'bb_std', 'atr_length']
    feat_params = {k: v for k, v in all_params.items() if k in feat_keys}
    model_params = {k: v for k, v in all_params.items() if k not in feat_keys and k != 'vol_multiplier'}
    
    vol_multiplier = all_params.get('vol_multiplier', 1.0)
    
    model_params.update({
        'loss_function': 'MultiClass',
        'eval_metric': 'MultiClass',
        'classes_count': 3,
        'verbose': 0,
        'task_type': 'GPU' if use_gpu else 'CPU',
        'thread_count': 4
    })
    
    if use_gpu:
        model_params['devices'] = '0'

    # 2. Generate Features
    try:
        full_df = process_features(raw_dfs, feat_params)
    except Exception:
        return 0.0
        
    if full_df.empty:
        return 0.0
        
    # 3. Create Targets
    full_df = create_targets(full_df, vol_multiplier)
    
    # Map targets: -1 -> 0, 0 -> 1, 1 -> 2
    label_mapping = {-1: 0, 0: 1, 1: 2}
    full_df['target_mapped'] = full_df['target'].map(label_mapping)
    
    # 4. Prepare Data
    exclude_cols = ['ts_code', 'trade_time', 'target', 'target_mapped', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pre_close', 'date', 'future_close', 'future_return_5d', 'volatility_factor']
    feature_cols = [c for c in full_df.columns if c not in exclude_cols]
    
    X = full_df[feature_cols].select_dtypes(include=[np.number])
    y = full_df['target_mapped'].values
    
    # 5. Class Weights (Focal Loss Style from train.py)
    classes = np.unique(y)
    if len(classes) < 2:
        return 0.0
        
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    weight_map = dict(zip(classes, class_weights))
    
    # Focal Loss enhancement
    class_counts = pd.Series(y).value_counts()
    total = len(y)
    gamma = 2.0
    for cls in classes:
        cls_ratio = class_counts.get(cls, 0) / total
        focal_factor = (1 - cls_ratio) ** gamma
        weight_map[cls] = weight_map[cls] * (1 + focal_factor * 0.5)
        
    # Boost Buy Class (Class 2)
    if 2 in weight_map:
        weight_map[2] = weight_map[2] * 2.0
        
    # Normalize
    max_weight = max(weight_map.values())
    weight_map = {k: v / max_weight * 3 for k, v in weight_map.items()}
    
    weights = np.array([weight_map[val] for val in y])
    
    # 6. TimeSeriesSplit CV
    tscv = TimeSeriesSplit(n_splits=3) # Reduced to 3 for speed
    f1_scores = []
    
    for train_idx, val_idx in tscv.split(X):
        # Gap logic (simplified)
        gap = 5
        if train_idx[-1] + gap >= val_idx[0]:
             safe_train_end = val_idx[0] - gap
             train_idx = train_idx[train_idx < safe_train_end]
             
        if len(train_idx) < 50:
            continue
            
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w_tr = weights[train_idx]
        
        # RobustScaler
        scaler = RobustScaler()
        X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_val.columns)
        
        # Train
        model = None
        if use_gpu:
            with gpu_lock:
                model = train_gpu_worker(model_params, X_tr_scaled, y_tr, w_tr, X_val_scaled, y_val)
            if model is None: # Fallback
                 model_params['task_type'] = 'CPU'
                 if 'devices' in model_params: del model_params['devices']
                 model = train_gpu_worker(model_params, X_tr_scaled, y_tr, w_tr, X_val_scaled, y_val)
        else:
            model = train_gpu_worker(model_params, X_tr_scaled, y_tr, w_tr, X_val_scaled, y_val)
            
        if model is None:
            continue
            
        # Evaluate
        try:
            y_pred = model.predict(X_val_scaled).flatten()
            f1 = f1_score(y_val, y_pred, average='macro', labels=[0, 1, 2], zero_division=0)
            f1_scores.append(f1)
        except Exception:
            pass
            
    if not f1_scores:
        return 0.0
        
    return np.mean(f1_scores)

def evaluate_best_model(best_params, raw_dfs, use_gpu=False):
    print("\n" + "="*60)
    print("   最佳模型最终评估")
    print("="*60)
    
    # Re-construct params
    feat_keys = ['rsi_window', 'macd_fast', 'macd_slow', 'macd_signal', 'bb_length', 'bb_std', 'atr_length']
    feat_params = {k: v for k, v in best_params.items() if k in feat_keys}
    model_params = {k: v for k, v in best_params.items() if k not in feat_keys and k != 'vol_multiplier'}
    vol_multiplier = best_params.get('vol_multiplier', 1.0)
    
    model_params.update({
        'loss_function': 'MultiClass',
        'eval_metric': 'MultiClass',
        'classes_count': 3,
        'verbose': 0,
        'task_type': 'GPU' if use_gpu else 'CPU',
        'thread_count': 4
    })
    if use_gpu:
        model_params['devices'] = '0'

    print(f"1. 使用最佳参数生成特征: {feat_params}")
    full_df = process_features(raw_dfs, feat_params)
    if full_df.empty:
        print("错误: 未生成数据。")
        return

    print(f"2. 使用波动率乘数创建目标: {vol_multiplier:.4f}")
    full_df = create_targets(full_df, vol_multiplier)
    
    label_mapping = {-1: 0, 0: 1, 1: 2}
    full_df['target_mapped'] = full_df['target'].map(label_mapping)
    
    exclude_cols = ['ts_code', 'trade_time', 'target', 'target_mapped', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pre_close', 'date', 'future_close', 'future_return_5d', 'volatility_factor']
    feature_cols = [c for c in full_df.columns if c not in exclude_cols]
    
    X = full_df[feature_cols].select_dtypes(include=[np.number])
    y = full_df['target_mapped'].values
    
    # Recalculate weights
    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    weight_map = dict(zip(classes, class_weights))
    class_counts = pd.Series(y).value_counts()
    total = len(y)
    gamma = 2.0
    for cls in classes:
        cls_ratio = class_counts.get(cls, 0) / total
        focal_factor = (1 - cls_ratio) ** gamma
        weight_map[cls] = weight_map[cls] * (1 + focal_factor * 0.5)
    if 2 in weight_map:
        weight_map[2] = weight_map[2] * 2.0
    max_weight = max(weight_map.values())
    weight_map = {k: v / max_weight * 3 for k, v in weight_map.items()}
    weights = np.array([weight_map[val] for val in y])

    print("3. 运行交叉验证以生成混淆矩阵...")
    tscv = TimeSeriesSplit(n_splits=5) # Use 5 splits for final eval
    y_true_all = []
    y_pred_all = []
    
    iterator = tscv.split(X)
    if tqdm:
        iterator = tqdm(iterator, total=5, desc="最终交叉验证")

    for train_idx, val_idx in iterator:
        gap = 5
        if train_idx[-1] + gap >= val_idx[0]:
             safe_train_end = val_idx[0] - gap
             train_idx = train_idx[train_idx < safe_train_end]
        
        if len(train_idx) < 50: continue

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w_tr = weights[train_idx]
        
        scaler = RobustScaler()
        X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_val.columns)
        
        model = CatBoostClassifier(**model_params)
        model.fit(Pool(X_tr_scaled, y_tr, weight=w_tr), verbose=0)
        
        y_pred = model.predict(X_val_scaled).flatten()
        y_true_all.extend(y_val)
        y_pred_all.extend(y_pred)

    print("\n" + "="*30)
    print("最终结果报告")
    print("="*30)
    
    target_names = ['卖出', '持有', '买入']
    print("\n分类报告 (Classification Report):")
    print(classification_report(y_true_all, y_pred_all, target_names=target_names, zero_division=0))
    
    print("\n混淆矩阵 (Confusion Matrix):")
    cm = confusion_matrix(y_true_all, y_pred_all)
    print(cm)
    
    # Feature Importance
    print("\n4. 在所有数据上训练最终模型以获取特征重要性...")
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    final_model = CatBoostClassifier(**model_params)
    final_model.fit(Pool(X_scaled, y, weight=weights), verbose=0)
    
    feature_importances = final_model.get_feature_importance()
    fi_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importances})
    fi_df = fi_df.sort_values('importance', ascending=False).head(20)
    
    print("\n前 20 个重要特征:")
    print(fi_df.to_string(index=False))
    print("="*60)

def check_gpu_working():
    try:
        print("正在检查 GPU 可用性...")
        model = CatBoostClassifier(iterations=10, task_type="GPU", devices="0", verbose=0)
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)
        print("GPU 工作正常。")
        return True
    except Exception as e:
        print(f"GPU 检查失败: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--data_dir', type=str, default='data/daily')
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    args = parser.parse_args()
    
    print_logic_summary()
    
    if args.gpu:
        if not check_gpu_working():
            print("警告: GPU 不工作。切换到 CPU。")
            args.gpu = False
    
    print("正在加载数据...")
    raw_dfs = load_raw_data(args.data_dir, limit_files=args.limit)
    print(f"已加载 {len(raw_dfs)} 只股票。")
    
    if not raw_dfs:
        sys.exit(1)
        
    study = optuna.create_study(direction='maximize')
    
    # Use n_jobs=2 to allow CPU (Feature Gen) and GPU (Training) to overlap
    n_jobs = 2 if args.gpu else 1
    print(f"正在运行优化，n_jobs={n_jobs}...")
    
    study.optimize(lambda trial: objective(trial, raw_dfs, DEFAULT_BOUNDS, use_gpu=args.gpu), n_trials=args.trials, n_jobs=n_jobs, show_progress_bar=True)
    
    print("最佳参数:", study.best_params)
    print("最佳分数:", study.best_value)
    
    # Save results
    os.makedirs('optuna_results/daily_dynamic', exist_ok=True)
    df_results = study.trials_dataframe()
    df_results.to_csv(f'optuna_results/daily_dynamic/results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    
    # Final Evaluation
    evaluate_best_model(study.best_params, raw_dfs, use_gpu=args.gpu)
