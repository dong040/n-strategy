#!/bin/bash
# N字战法 每日扫描+推送脚本
# 建议配置 cron: 30 15 * * 1-5 /bin/bash ~/n-strategy/scripts/run_daily.sh > /tmp/nstrategy_daily.log 2>&1

set -e

PROJECT_DIR="$HOME/n-strategy"
cd "$PROJECT_DIR"

# 加载环境变量
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# 激活虚拟环境（如果存在）
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

echo "=== $(date '+%Y-%m-%d %H:%M:%S') 开始每日扫描 ==="

# 执行扫描+推送
python -m src.cli daily

echo "=== $(date '+%Y-%m-%d %H:%M:%S') 完成 ==="
