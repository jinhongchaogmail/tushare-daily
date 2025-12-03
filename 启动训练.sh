#!/bin/bash

# 定义颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

LOG_FILE="training.log"
SCRIPT_PATH="training/模型训练.py"

echo -e "${GREEN}=== 启动本地训练流程 ===${NC}"

# 1. 检查是否已有训练在运行
echo -e "${YELLOW}[1/5] 检查运行状态...${NC}"
# 使用 pgrep 查找包含脚本文件名的进程 (放宽匹配条件，不强制 python3)
if pgrep -f "模型训练.py" > /dev/null; then
    PID=$(pgrep -f "模型训练.py" | head -n 1)
    echo -e "${GREEN}✅ 检测到训练脚本正在后台运行 (PID: $PID)。${NC}"
    echo -e "${CYAN}⚠️  由于 GPU 资源限制，不启动新进程，转为显示当前运行状态。${NC}"
    echo -e "----------------------------------------"
    echo -e "📄 日志文件 : ${GREEN}$LOG_FILE${NC}"
    # 获取运行时间
    RUNTIME=$(ps -p $PID -o etime= | xargs)
    echo -e "⏱️  运行时间 : ${GREEN}$RUNTIME${NC}"
    echo -e "----------------------------------------"
    echo -e "📋 最新日志 (最后 10 行):"
    echo -e "${CYAN}"
    if [ -f "$LOG_FILE" ]; then
        tail -n 10 "$LOG_FILE"
    else
        echo "暂无日志内容"
    fi
    echo -e "${NC}"
    echo -e "----------------------------------------"
    echo -e "💡 常用命令:"
    echo -e "   👀 继续监控日志: ${YELLOW}tail -f $LOG_FILE${NC}"
    echo -e "   🛑 停止训练进程: ${YELLOW}kill $PID${NC}"
    exit 0
else
    echo -e "${GREEN}✅ 无正在运行的训练进程，准备启动新任务。${NC}"
fi

# 2. 加载环境
echo -e "${YELLOW}[2/5] 加载运行环境...${NC}"
if [ -f "secrets.sh" ]; then
    source secrets.sh
    echo -e "✅ 已加载 secrets.sh"
fi

# 尝试激活虚拟环境
if [ -f "$HOME/.venv/bin/activate" ]; then
    source "$HOME/.venv/bin/activate"
    echo -e "✅ 已激活虚拟环境 ($HOME/.venv)"
elif [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
    echo -e "✅ 已激活虚拟环境 (.venv)"
fi

# 设置 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/shared

# 3. 下载数据
echo -e "${YELLOW}[3/5] 下载最新数据...${NC}"
DOWNLOAD_URL="https://github.com/jinhongchaogmail/tushare-daily/releases/download/latest/parquet-data.tar.gz"
echo "正在从 GitHub Releases 下载..."
if wget -q --show-progress -O parquet-data.tar.gz "$DOWNLOAD_URL"; then
    echo -e "${GREEN}✅ 数据下载成功。${NC}"
else
    echo -e "${RED}❌ 数据下载失败，请检查网络或 URL。${NC}"
    exit 1
fi

# 4. 清理旧数据
echo -e "${YELLOW}[4/5] 清理旧数据与缓存...${NC}"
# 确保 data 目录存在
mkdir -p data
# 清空 data 目录下的内容，强制触发 Python 脚本的解压逻辑
rm -rf data/*
# [新增] 删除旧的特征缓存文件，强制 Python 脚本使用新下载的数据重新计算特征
rm -f feature_cache_*.parquet
echo -e "${GREEN}✅ 已清理 data/ 目录及旧缓存，准备重新解压与计算。${NC}"

# 5. 后台启动训练
echo -e "${YELLOW}[5/5] 启动后台训练进程...${NC}"

# 使用 nohup 在后台运行，并将标准输出和标准错误重定向到日志文件
nohup python3 "$SCRIPT_PATH" > "$LOG_FILE" 2>&1 &
NEW_PID=$!

# 检查进程是否成功启动
sleep 1
if ps -p $NEW_PID > /dev/null; then
    echo -e "${GREEN}🎉 训练已成功在后台启动！${NC}"
    echo -e "----------------------------------------"
    echo -e "🆔 进程 PID : ${GREEN}$NEW_PID${NC}"
    echo -e "📄 日志文件 : ${GREEN}$LOG_FILE${NC}"
    echo -e "----------------------------------------"
    echo -e "💡 常用命令:"
    echo -e "   👀 查看实时日志: ${YELLOW}tail -f $LOG_FILE${NC}"
    echo -e "   🛑 停止训练进程: ${YELLOW}kill $NEW_PID${NC}"
    echo -e "----------------------------------------"
else
    echo -e "${RED}❌ 训练启动失败，请检查日志文件: $LOG_FILE${NC}"
    cat "$LOG_FILE"
    exit 1
fi
