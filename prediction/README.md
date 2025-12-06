# Prediction 目录

这个目录包含了股票预测系统的运行文件，用于执行每日股票预测任务。

## 目录结构

- `daily_run.py`: 主要的预测脚本
- `features.py`: 特征工程模块（从模型目录复制）
- `data_fetcher.py`: 数据获取模块（符号链接到shared目录）
- `downloader.py`: 数据下载模块（符号链接到shared目录）
- `financials.py`: 财务数据处理模块（符号链接到shared目录）
- `run_prediction.sh`: 运行脚本

## 使用方法

```bash
cd prediction
./run_prediction.sh
```

或者直接运行：

```bash
cd prediction
python daily_run.py
```

## 参数说明

- `--batch-size`: 每批次处理股票数量 (默认: 10)
- `--stocks`: 指定要处理的股票代码，用逗号分隔 (例如: 000001.SZ,000002.SZ)
- `--skip-financials`: 跳过财务数据获取
- `--parallel-workers`: 并行处理线程数 (默认: 2)
- `--output-dir`: 数据文件输出目录 (默认: ../data)
- `--start-date`: 数据获取起始日期 (格式: YYYYMMDD)
- `--skip-predictions`: 跳过模型预测，仅下载数据

## 输出文件

预测结果将保存在 `../daily_prediction` 目录中：

- `daily_report_YYYYMMDD_HHMMSS.csv`: 日常报告
- `all_predictions_YYYYMMDD_HHMMSS.csv`: 所有预测结果