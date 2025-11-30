#!/bin/bash
# 本地运行预测
# 确保已配置 TUSHARE_TOKEN 环境变量

# 加载本地密钥 (如果存在)
if [ -f "secrets.sh" ]; then
    source secrets.sh
fi

# 尝试激活虚拟环境
if [ -f "$HOME/.venv/bin/activate" ]; then
    source "$HOME/.venv/bin/activate"
fi

if [ -z "$TUSHARE_TOKEN" ]; then
    echo "Error: TUSHARE_TOKEN is not set."
    echo "Usage: export TUSHARE_TOKEN=your_token && ./predict_local.sh"
    exit 1
fi

echo "Running prediction pipeline locally..."
# 使用虚拟环境的 Python 绝对路径，确保依赖正确加载
"$HOME/.venv/bin/python" run_pipeline.py
