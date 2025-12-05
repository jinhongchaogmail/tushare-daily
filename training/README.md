# 模型训练 (Training)

本目录包含模型训练的核心代码和配置。

## 核心文件

- **`train.py`**: 主训练脚本。集成 Optuna 超参数寻优和 CatBoost 模型训练。
- **`config.yaml`**: 训练配置文件。管理数据路径、Optuna 参数、日志设置等。
- **`v16_OPTIMIZATION_ARCHIVE.md`**: v16 版本的优化历史文档归档。

## 运行训练

```bash
# 确保在项目根目录下
python training/train.py
```

## 配置说明

通过修改 `config.yaml` 可以调整：
- `optuna.n_trials`: 寻优次数
- `data.max_files`: 训练使用的数据量
- `balance`: 类别平衡参数

## 历史文档

关于 v16 版本的详细优化记录（包括配置系统设计、缓存机制等），请参阅 [v16_OPTIMIZATION_ARCHIVE.md](v16_OPTIMIZATION_ARCHIVE.md)。

## 特征工程与模型一致性 (Feature Consistency)

为了解决训练与推断时的特征不一致问题 (Training-Inference Skew)，本系统采用了以下机制：

1.  **特征快照 (Feature Snapshot)**:
    - 每次训练成功后，`train.py` 会自动将当前的 `shared/features.py` 复制到 `models/frozen_features.py`。
    - 推断脚本 `daily_run.py` 优先加载 `models/frozen_features.py`，确保推断时使用的特征计算逻辑与训练时完全一致。

2.  **缓存机制 (Caching)**:
    - 特征工程结果会缓存为 `.parquet` 文件。
    - 缓存文件名包含配置哈希 (`config_hash`)。
    - **注意**: `config_hash` 计算包含了 `config.yaml` 中的 `features` 配置以及 `shared/features.py` 的代码哈希。修改特征代码会自动失效旧缓存。

3.  **兼容性说明**:
    - 当前模型 (v37) 依赖旧的特征命名 (`vol_ma5`, `volatility_10`)。
    - `shared/features.py` 中已添加兼容性代码，确保新版代码能生成旧版模型所需的特征。
