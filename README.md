# Tushare Daily Stock Prediction

这是一个基于 GitHub Actions 的每日 A 股量化预测系统。

## 项目结构

- **.github/workflows/**: GitHub Actions 工作流配置
  - `daily_prediction.yml`: 每日定时运行的任务 (下载数据 -> 预测 -> 生成报告)
- **optuna/**: 核心策略代码
  - `quant/predict_strategy.py`: 预测脚本 (被 run_pipeline.py 调用)
  - `quant/feature_engineering.py`: 特征工程逻辑
  - `model/`: 存放模型文件 (.cbm) 和参数 (.json)
- **run_pipeline.py**: 主程序 (流式下载数据 + 实时预测)
- **data/**: 存放下载的股票数据 (.parquet 格式)
- **notebooks/**: 研究用的 Jupyter Notebooks
- **legacy/**: 旧版本的脚本和工具
- **artifacts/**: 历史运行生成的图片和报告

## 运行流程

1. **启动流水线**: `run_pipeline.py` 开始运行。
2. **流式处理**: 
   - 下载一只股票数据 -> 立即进行特征计算 -> 模型推理。
   - 这种方式比先下载后计算更高效。
3. **生成报告**: 筛选出高胜率机会，生成 Markdown 报告并上传为 Artifact。

## 本地运行

1. 安装依赖:
   ```bash
   pip install -r optuna/quant/requirements.txt
   ```
2. 设置 Tushare Token:
   ```bash
   export TUSHARE_TOKEN="your_token_here"
   ```
3. 运行流水线:
   ```bash
   python run_pipeline.py
   ```

