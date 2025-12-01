#!/bin/bash
# 运行本地训练流程
# 确保在项目根目录下运行

# 加载本地密钥 (如果存在)
if [ -f "secrets.sh" ]; then
    source secrets.sh
fi

# 尝试激活虚拟环境
if [ -f "$HOME/.venv/bin/activate" ]; then
    source "$HOME/.venv/bin/activate"
fi

echo "正在从 Google Drive 下载最新数据..."
rclone copyto "drive:tushare-data/parquet-data.tar.gz" ./parquet-data.tar.gz
rm data/* 
echo "正在启动训练流程..."
# 设置 PYTHONPATH 以便找到 shared 模块 (虽然脚本内部也处理了，但这样更稳健)
export PYTHONPATH=$PYTHONPATH:$(pwd)/shared

python3 training/模型训练.py
