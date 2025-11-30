#!/bin/bash
# 同步模型到 Git
# 用法: ./sync_model.sh [commit_message]

echo "Syncing models to Git..."

# 确保 models 目录存在
if [ ! -d "models" ]; then
    echo "Error: models directory not found!"
    exit 1
fi

# 添加模型文件
git add models/catboost_final_model.cbm models/final_model_params.json

# 提交
MSG=${1:-"Update model artifacts $(date +%Y-%m-%d)"}
git commit -m "$MSG"

# 推送
echo "Pushing to remote..."
git push

echo "Model synced successfully."
