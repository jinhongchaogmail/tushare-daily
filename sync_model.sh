#!/bin/bash
# 同步模型到 Git
# 用法: ./sync_model.sh [run_directory] [commit_message]
# 示例: ./sync_model.sh optuna_results/run_2023... "Update model"

# 定义颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== 同步模型到 Git 仓库 ===${NC}"

# 1. 确定源目录
RUN_DIR="$1"

# 如果未提供参数，尝试自动查找最新的运行结果
if [ -z "$RUN_DIR" ]; then
    echo -e "${YELLOW}[INFO] 未指定运行目录，正在查找最新的 optuna_results...${NC}"
    if [ -d "optuna_results" ]; then
        # 查找最新的 run_ 开头的目录
        LATEST_RUN=$(ls -td optuna_results/run_* 2>/dev/null | head -n 1)
        if [ -n "$LATEST_RUN" ]; then
            RUN_DIR="$LATEST_RUN"
            echo -e "${GREEN}✅ 自动定位到最新运行: $RUN_DIR${NC}"
        else
            echo -e "${RED}❌ 未找到任何训练结果目录。${NC}"
            exit 1
        fi
    else
        echo -e "${RED}❌ optuna_results 目录不存在。${NC}"
        exit 1
    fi
fi

COMMIT_MSG="${2:-"Update model artifacts from $(basename "$RUN_DIR")"}"

# 检查源目录是否存在
if [ ! -d "$RUN_DIR" ]; then
    echo -e "${RED}❌ 错误：目录 $RUN_DIR 不存在！${NC}"
    exit 1
fi

# 2. 检查关键文件
MODEL_FILE="$RUN_DIR/catboost_final_model.cbm"
PARAMS_FILE="$RUN_DIR/final_model_params.json"
FEATURES_FILE="$RUN_DIR/frozen_features.py"

MISSING_FILES=0

if [ ! -f "$MODEL_FILE" ]; then
    echo -e "${RED}❌ 缺失: catboost_final_model.cbm${NC}"
    MISSING_FILES=1
fi

if [ ! -f "$PARAMS_FILE" ]; then
    echo -e "${RED}❌ 缺失: final_model_params.json${NC}"
    MISSING_FILES=1
fi

if [ ! -f "$FEATURES_FILE" ]; then
    echo -e "${YELLOW}⚠️  缺失: frozen_features.py (特征快照)${NC}"
    echo -e "${YELLOW}   注意: 缺少此文件可能导致推断时特征不一致。${NC}"
    # 不强制退出，但给予强烈警告
else
    echo -e "${GREEN}✅ 发现特征快照: frozen_features.py${NC}"
fi

if [ $MISSING_FILES -eq 1 ]; then
    echo -e "${RED}❌ 关键文件缺失，无法同步。${NC}"
    exit 1
fi

# 3. 执行同步
echo -e "\n${CYAN}=== 开始复制文件 ===${NC}"
mkdir -p models

# 复制模型
cp "$MODEL_FILE" models/
MODEL_SIZE=$(du -h "models/catboost_final_model.cbm" | cut -f1)
echo -e "📦 模型文件已更新 (大小: $MODEL_SIZE)"

# 复制参数
cp "$PARAMS_FILE" models/
echo -e "⚙️  参数文件已更新"

# 复制特征快照
if [ -f "$FEATURES_FILE" ]; then
    cp "$FEATURES_FILE" models/
    echo -e "🧊 特征快照已更新 (frozen_features.py)"
fi

# 4. Git 操作
echo -e "\n${CYAN}=== Git 提交与推送 ===${NC}"

# 添加文件
git add models/catboost_final_model.cbm models/final_model_params.json models/frozen_features.py

# 检查状态
STATUS=$(git status --porcelain models/)
if [ -z "$STATUS" ]; then
    echo -e "${YELLOW}没有检测到模型文件变更，无需提交。${NC}"
    exit 0
fi

echo -e "检测到变更:\n$STATUS"

# 提交
echo -e "正在提交: ${GREEN}$COMMIT_MSG${NC}"
git commit -m "$COMMIT_MSG"

# 推送
echo -e "正在推送到远程仓库..."
CURRENT_BRANCH=$(git symbolic-ref --short HEAD)
UPSTREAM=$(git config --get branch.$CURRENT_BRANCH.merge)

if [ -z "$UPSTREAM" ]; then
    echo -e "${YELLOW}当前分支没有上游分支，正在设置 upstream...${NC}"
    git push --set-upstream origin "$CURRENT_BRANCH"
else
    git push
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}🎉 模型同步成功！${NC}"
else
    echo -e "${RED}❌ 推送失败，请检查网络或 Git 配置。${NC}"
    exit 1
fi
