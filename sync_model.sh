#!/bin/bash
# 同步模型到 Git
# 用法: ./sync_model.sh [run_directory] [commit_message]
# 示例: ./sync_model.sh optuna_results/run_2023... "Update model"

echo "正在同步模型到 Git..."

# 检查参数
if [ -z "$1" ]; then
    echo "用法: ./sync_model.sh [run_directory] [commit_message]"
    echo "请提供训练结果目录 (例如 optuna_results/run_...)"
    exit 1
fi

RUN_DIR="$1"
COMMIT_MSG="${2:-"Update model artifacts $(date +%Y-%m-%d)"}"

# 检查源目录
if [ ! -d "$RUN_DIR" ]; then
    echo "错误：目录 $RUN_DIR 不存在！"
    exit 1
fi

# 检查模型文件
if [ ! -f "$RUN_DIR/catboost_final_model.cbm" ]; then
    echo "错误：在 $RUN_DIR 中未找到 catboost_final_model.cbm"
    exit 1
fi

# 确保 models 目录存在
if [ ! -d "models" ]; then
    mkdir -p models
fi

echo "=== 1. 同步模型文件 ==="
cp "$RUN_DIR/catboost_final_model.cbm" models/catboost_final_model.cbm
cp "$RUN_DIR/final_model_params.json" models/final_model_params.json
echo "✅ 模型文件已复制到 models/"

echo "=== 2. 同步特征快照 (frozen_features.py) ==="
if [ -f "$RUN_DIR/frozen_features.py" ]; then
    echo "✅ 发现特征快照，正在复制到 models/frozen_features.py..."
    cp "$RUN_DIR/frozen_features.py" models/frozen_features.py
else
    echo "⚠️ 警告：源目录中未找到 frozen_features.py！"
    echo "这可能导致预测时特征不一致。建议重新训练模型以生成快照。"
fi

# 添加文件到 Git (包含 frozen_features.py)
git add models/catboost_final_model.cbm models/final_model_params.json models/frozen_features.py

# 提交
git commit -m "$COMMIT_MSG"

# 推送
echo "正在推送到远程仓库..."
# 检查当前分支是否有上游分支
current_branch=$(git symbolic-ref --short HEAD)
upstream=$(git config --get branch.$current_branch.merge)
if [ -z "$upstream" ]; then
    git push --set-upstream origin "$current_branch"
else
    git push
fi

echo "模型同步成功。"
