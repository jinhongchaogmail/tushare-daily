#!/bin/bash
# 同步模型到 Git
# 用法: ./sync_model.sh [commit_message]

echo "正在同步模型到 Git..."

# 确保 models 目录存在
if [ ! -d "models" ]; then
    echo "错误：未找到 models 目录！"
    exit 1
fi

# 添加模型文件
git add models/catboost_final_model.cbm models/final_model_params.json

# 提交
MSG=${1:-"Update model artifacts $(date +%Y-%m-%d)"}
git commit -m "$MSG"

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
