#!/bin/bash
# 本地运行预测
# 确保已配置 TUSHARE_TOKEN 环境变量

if [ -z "$TUSHARE_TOKEN" ]; then
    echo "Error: TUSHARE_TOKEN is not set."
    echo "Usage: export TUSHARE_TOKEN=your_token && ./predict_local.sh"
    exit 1
fi

echo "Running prediction pipeline locally..."
python3 run_pipeline.py
