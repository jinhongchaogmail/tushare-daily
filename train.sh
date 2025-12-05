#!/bin/bash

# 定义颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

LOG_FILE="training.log"
SCRIPT_PATH="training/train.py"

echo -e "${GREEN}=== 启动本地训练流程 ===${NC}"

# 0. 预检查
if [ ! -f "$SCRIPT_PATH" ]; then
    echo -e "${RED}❌ 错误: 找不到训练脚本 $SCRIPT_PATH${NC}"
    exit 1
fi

# 1. 检查运行状态
echo -e "${YELLOW}[1/6] 检查运行状态...${NC}"
if pgrep -f "training/train.py" > /dev/null; then
    PID=$(pgrep -f "training/train.py" | head -n 1)
    echo -e "${GREEN}✅ 检测到训练脚本正在后台运行 (PID: $PID)。${NC}"
    echo -e "${CYAN}⚠️  由于 GPU 资源限制，不启动新进程。${NC}"
    
    # 获取运行时间
    if ps -p $PID -o etime= >/dev/null 2>&1; then
        RUNTIME=$(ps -p $PID -o etime= | xargs)
    else
        RUNTIME="未知"
    fi
    
    echo -e "----------------------------------------"
    echo -e "⏱️  运行时间 : ${GREEN}$RUNTIME${NC}"
    echo -e "📄 日志文件 : ${GREEN}$LOG_FILE${NC}"
    echo -e "----------------------------------------"
    echo -e "📋 最新日志 (最后 10 行):"
    echo -e "${CYAN}"
    [ -f "$LOG_FILE" ] && tail -n 10 "$LOG_FILE" || echo "暂无日志"
    echo -e "${NC}"
    echo -e "----------------------------------------"
    echo -e "💡 常用命令:"
    echo -e "   👀 监控日志: ${YELLOW}tail -f $LOG_FILE${NC}"
    echo -e "   🛑 停止进程: ${YELLOW}kill $PID${NC}"
    exit 0
fi

# 2. 加载环境
echo -e "${YELLOW}[2/6] 加载运行环境...${NC}"
if [ -f "secrets.sh" ]; then
    source secrets.sh
    echo -e "✅ 已加载 secrets.sh"
fi

# 智能激活虚拟环境
VENV_ACTIVATED=0
if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "✅ 已在虚拟环境中: $VIRTUAL_ENV"
    VENV_ACTIVATED=1
elif [ -f "$HOME/.venv/bin/activate" ]; then
    source "$HOME/.venv/bin/activate"
    echo -e "✅ 已激活虚拟环境 ($HOME/.venv)"
    VENV_ACTIVATED=1
elif [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
    echo -e "✅ 已激活虚拟环境 (.venv)"
    VENV_ACTIVATED=1
fi

if [ $VENV_ACTIVATED -eq 0 ]; then
    echo -e "${CYAN}⚠️  未检测到虚拟环境，将使用系统 Python。${NC}"
fi

export PYTHONPATH=$PYTHONPATH:$(pwd)/shared

# 3. 下载数据
echo -e "${YELLOW}[3/6] 下载最新数据...${NC}"
DOWNLOAD_URL="https://github.com/jinhongchaogmail/tushare-daily/releases/download/latest/parquet-data.tar.gz"

# 检查 wget
if ! command -v wget &> /dev/null; then
    echo -e "${RED}❌ 错误: 未安装 wget。${NC}"
    exit 1
fi

echo "正在从 GitHub Releases 下载..."
if wget -q --show-progress -O parquet-data.tar.gz "$DOWNLOAD_URL"; then
    echo -e "${GREEN}✅ 数据下载成功。${NC}"
else
    echo -e "${RED}❌ 数据下载失败，请检查网络或 URL。${NC}"
    exit 1
fi

# 4. 清理旧数据
echo -e "${YELLOW}[4/6] 清理旧数据与缓存...${NC}"
mkdir -p data
# 清空 data 目录
rm -rf data/*
# 强制清理特征缓存 (因为数据源已更新)
COUNT=$(ls feature_cache_*.parquet 2>/dev/null | wc -l)
if [ "$COUNT" != "0" ]; then
    rm -f feature_cache_*.parquet
    echo -e "✅ 已删除 $COUNT 个旧特征缓存文件 (确保使用新数据)"
else
    echo -e "✅ 无旧缓存文件"
fi

# 5. 日志轮转
echo -e "${YELLOW}[5/6] 准备日志...${NC}"
if [ -f "$LOG_FILE" ]; then
    mv "$LOG_FILE" "${LOG_FILE}.bak"
    echo -e "✅ 旧日志已备份为 ${LOG_FILE}.bak"
fi

# 6. 后台启动
echo -e "${YELLOW}[6/6] 启动后台训练进程...${NC}"
nohup python3 "$SCRIPT_PATH" > "$LOG_FILE" 2>&1 &
NEW_PID=$!

sleep 2
if ps -p $NEW_PID > /dev/null; then
    echo -e "${GREEN}🎉 训练已成功在后台启动！${NC}"
    echo -e "----------------------------------------"
    echo -e "🆔 进程 PID : ${GREEN}$NEW_PID${NC}"
    echo -e "📄 日志文件 : ${GREEN}$LOG_FILE${NC}"
    echo -e "----------------------------------------"
    echo -e "💡 常用命令:"
    echo -e "   👀 监控日志: ${YELLOW}tail -f $LOG_FILE${NC}"
    echo -e "   🛑 停止进程: ${YELLOW}kill $NEW_PID${NC}"
    echo -e "----------------------------------------"
else
    echo -e "${RED}❌ 训练启动失败，请检查日志文件: $LOG_FILE${NC}"
    echo -e "${CYAN}--- 日志末尾 ---${NC}"
    tail -n 20 "$LOG_FILE"
    exit 1
fi
