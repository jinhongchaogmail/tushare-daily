# Tushare Daily Stock Prediction

这是一个基于 GitHub Actions 的每日 A 股量化预测系统。

## 项目结构

- **.github/workflows/**: GitHub Actions 工作流配置
  - `daily_prediction.yml`: 每日定时运行的任务 (下载数据 -> 预测 -> 生成报告)
- **optuna/**: 核心策略代码
  - `quant/predict_strategy.py`: 预测脚本，加载模型并生成交易信号
  - `quant/feature_engineering.py`: 特征工程逻辑 (与训练时一致)
  - `model/catboost_final_model.cbm`: 训练好的 CatBoost 模型
  - `model/final_model_params.json`: 模型参数
- **main.py**: 数据下载脚本 (调用 Tushare API)
- **data/**: 存放下载的股票数据 (.parquet 格式)

## 运行流程

1. **下载数据**: `main.py` 获取最新的日线数据。
2. **特征计算**: `predict_strategy.py` 调用 `feature_engineering.py` 计算技术指标。
3. **模型推理**: 加载 `catboost_final_model.cbm` 对每只股票进行预测。
4. **生成报告**: 筛选出高胜率机会，生成 Markdown 报告并上传为 Artifact。

## 本地运行

1. 安装依赖:
   ```bash
   pip install -r optuna/quant/requirements.txt
   ```
2. 设置 Tushare Token:
   ```bash
   export TUSHARE_TOKEN="your_token_here"
   ```
3. 下载数据:
   ```bash
   python main.py
   ```
4. 运行策略:
   ```bash
   cd optuna/quant
   python predict_strategy.py
   ```
