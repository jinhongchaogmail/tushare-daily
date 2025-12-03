# Tushare Daily Stock Prediction

这是一个基于 GitHub Actions 的每日 A 股量化预测系统。

## 项目结构

- **.github/workflows/**: GitHub Actions 工作流配置
  - `run.yml`: 每日定时运行的任务 (下载数据 -> 预测 -> 发送邮件)
- **training/**: 模型训练与优化
  - `train.py`: 核心训练脚本 (Optuna + CatBoost)
  - `config.yaml`: 训练配置文件
  - `v16_OPTIMIZATION_ARCHIVE.md`: 历史优化文档归档
- **shared/**: 公共模块
  - `features.py`: 统一的特征计算逻辑 (SSOT)
- **daily_run.py**: 生产环境主程序 (下载 -> 预测 -> 报告)
- **data/**: 存放下载的股票数据 (.parquet 格式)
- **optuna_results/**: 训练结果与日志

## 运行流程

1. **每日预测**:
   - GitHub Actions 触发 `daily_run.py`。
   - 脚本下载最新数据，调用 `shared/features.py` 计算特征。
   - 加载 `catboost_final_model.cbm` 进行预测 (支持做多与做空)。
   - 生成 HTML/Markdown 报告并通过邮件发送。

2. **模型训练**:
   - 运行 `training/train.py`。
   - 使用 Optuna 进行超参数寻优。
   - 生成新的模型文件 `catboost_final_model.cbm`。

## 本地运行

1. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```
2. 设置 Tushare Token:
   ```bash
   export TUSHARE_TOKEN="your_token_here"
   ```
3. 运行预测:
   ```bash
   python daily_run.py
   ```

