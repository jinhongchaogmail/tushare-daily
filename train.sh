#!/bin/bash
# 运行本地训练流程
# 确保在项目根目录下运行

echo "Starting training pipeline..."
# 设置 PYTHONPATH 以便找到 shared 模块 (虽然脚本内部也处理了，但这样更稳健)
export PYTHONPATH=$PYTHONPATH:$(pwd)/shared

python3 training/optuna_catboost_pipeline.py
