import sys
import os
import json
import pandas as pd
import numpy as np
import catboost as cb
from sklearn.utils.class_weight import compute_class_weight
import shutil
import glob

# 添加 training 目录到 path
sys.path.append('training')
import train

# 配置
RUN_DIR = 'models'
GDRIVE_SAVE_PATH = RUN_DIR

# 用户提供的最佳参数 (Trial 13)
best_params = {
    'vol_multiplier': 0.9209494353703175,
    'iterations': 1811,
    'learning_rate': 0.07293939064484704,
    'depth': 10,
    'subsample': 0.6335506363776592,
    'l2_leaf_reg': 6.374309685723997,
    'min_data_in_leaf': 64,
    # 固定参数
    'loss_function': 'MultiClass',
    'eval_metric': 'MultiClass',
    'classes_count': 3,
    'random_seed': 42,
    'logging_level': 'Verbose', # 改为 Verbose 以便看到进度
    'bootstrap_type': 'Bernoulli'
}

# 强制使用 CPU
print("Forcing CPU training due to low GPU memory...")
best_params['task_type'] = 'CPU'
# best_params['rsm'] = 0.8 # CPU only parameter

# 1. 准备数据
print("Loading data...")
# 尝试加载缓存
paths = train.setup_paths(train.Environment.LOCAL)
# 查找缓存文件
cache_files = glob.glob(os.path.join(paths['GDRIVE_BASE_PATH'], 'feature_cache_*.parquet'))
combined_df = None
if cache_files:
    # 按时间排序取最新的
    cache_files.sort(key=os.path.getmtime, reverse=True)
    print(f"Loading cache: {cache_files[0]}")
    combined_df = pd.read_parquet(cache_files[0])
else:
    print("No cache found. Generating data from scratch...")
    # 使用 train.py 中的函数生成数据
    MAX_FILES = train.CONFIG.get('data.max_files_to_process', 10000)
    data_available = train.prepare_data_files(train.Environment.LOCAL, paths)
    if data_available:
        list_of_dfs = train.run_feature_engineering(paths['DATA_DIR'], MAX_FILES)
        if list_of_dfs:
            combined_df = train.post_process_data(list_of_dfs)
            
    if combined_df is None or combined_df.empty:
        print("Failed to generate data.")
        sys.exit(1)

# 2. 划分数据
print("Splitting data...")
TRAIN_TEST_SPLIT_RATIO = 0.8
train_size = int(len(combined_df) * TRAIN_TEST_SPLIT_RATIO)
train_data_raw = combined_df.iloc[:train_size].copy()
test_data_raw = combined_df.iloc[train_size:].copy()

# 3. 特征列
# 排除目标、未来收益，以及所有原始 O/H/L/C/V 数据
exclude = {
    'target_class_3', 'future_return_5d',
    'open', 'high', 'low', 'close', 'volume', 'amount',
    'stock_code_encoded'
}
# 排除所有以 'future_' 或 'shift' 开头的
exclude |= {c for c in train_data_raw.columns if c.lower().startswith('future_') or 'shift' in c.lower() or c in exclude}

num_cols = train_data_raw.select_dtypes(include=[np.number]).columns.to_list()
feature_columns = [c for c in num_cols if c not in exclude]
print(f"Features ({len(feature_columns)}): {feature_columns[:5]}...")

# 4. 创建目标 (使用最佳参数)
threshold_best = best_params.pop('vol_multiplier') # 从 params 中移除，因为它不是 catboost 参数
offset_best = 0.0 

def create_targets(df, base_threshold, offset):
    dfc = df.copy()
    vol_raw = dfc['volatility_factor'].fillna(0.02)
    dynamic_threshold = base_threshold * vol_raw
    dynamic_threshold = dynamic_threshold.clip(0.005, 0.05)
    lower_bound = offset - dynamic_threshold
    upper_bound = offset + dynamic_threshold
    
    dfc['target_class_3'] = np.nan
    valid = dfc['future_return_5d'].notna()
    dfc.loc[valid & (dfc['future_return_5d'] > upper_bound), 'target_class_3'] = 1
    dfc.loc[valid & (dfc['future_return_5d'] < lower_bound), 'target_class_3'] = -1
    dfc.loc[valid & dfc['target_class_3'].isna(), 'target_class_3'] = 0
    
    dfc = dfc.dropna(subset=['target_class_3']).copy()
    dfc['target_class_3'] = dfc['target_class_3'].astype('int16')
    return dfc

print(f"Creating targets with vol_multiplier={threshold_best}...")
train_final = create_targets(train_data_raw, threshold_best, offset_best)
test_final = create_targets(test_data_raw, threshold_best, offset_best)

X_train = train_final[feature_columns].fillna(0.0)
y_train = train_final['target_class_3'].values
X_test = test_final[feature_columns].fillna(0.0)
y_test = test_final['target_class_3'].values

# 映射标签
label_mapping = {-1: 0, 0: 1, 1: 2}
y_train_mapped = np.array([label_mapping[val] for val in y_train])
y_test_mapped = np.array([label_mapping[val] for val in y_test])

# 5. 计算权重
print("Computing weights...")
classes = np.unique(y_train_mapped)
cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_mapped)
wm = dict(zip(classes, cw))
# Focal loss style adjustment
class_counts = pd.Series(y_train_mapped).value_counts()
total = len(y_train_mapped)
for cls in classes:
    cls_ratio = class_counts.get(cls, 0) / total
    focal_factor = (1 - cls_ratio) ** 2.0
    wm[cls] = wm[cls] * (1 + focal_factor * 0.5)
if 1 in wm: wm[1] = wm[1] * 1.2 # Boost Up class
max_w = max(wm.values())
wm = {k: v / max_w * 3 for k, v in wm.items()}
sw_train = np.array([wm[val] for val in y_train_mapped])

# 6. 训练模型
print("Training model...")
model = cb.CatBoostClassifier(**best_params)
model.fit(X_train, y_train_mapped, sample_weight=sw_train, eval_set=(X_test, y_test_mapped), verbose=100)

# 7. 保存
print("Saving artifacts...")
os.makedirs(GDRIVE_SAVE_PATH, exist_ok=True)

# 保存模型
model_path = os.path.join(GDRIVE_SAVE_PATH, 'catboost_final_model.cbm')
model.save_model(model_path)
print(f"Model saved to {model_path}")

# 保存参数
final_params = {
    'vol_multiplier_best': threshold_best,
    'offset_best': offset_best,
    'mode': 'dynamic_volatility',
    'note': 'Recovered from Trial 13'
}
with open(os.path.join(GDRIVE_SAVE_PATH, 'final_model_params.json'), 'w') as f:
    json.dump(final_params, f, indent=2)

# 保存特征快照
features_src = 'shared/features.py'
if os.path.exists(features_src):
    shutil.copy2(features_src, os.path.join(GDRIVE_SAVE_PATH, 'frozen_features.py'))
    print("Features snapshot saved.")

print("Done.")
