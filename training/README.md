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
