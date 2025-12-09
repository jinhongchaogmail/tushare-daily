#!/bin/bash
# 自动激活项目的虚拟环境

PROJECT_VENV="/home/jin/work/.venv"

if [ -d "$PROJECT_VENV" ]; then
    echo "激活虚拟环境: $PROJECT_VENV"
    source "$PROJECT_VENV/bin/activate"
    echo "虚拟环境已激活，可以开始工作了！"
else
    echo "错误：找不到虚拟环境目录 $PROJECT_VENV"
fi