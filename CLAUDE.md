# N-strategy 项目配置

## 权限
- 默认允许所有 Bash 命令和文件读写，不需要手动确认

## Python 环境
- **核心策略系统**: 系统 Python 3.9 (`/usr/bin/python3`) — numpy, pandas, mootdx, akshare
- **TradingAgents (Batch 2)**: Python 3.12 venv (`/Users/moneisur/n-strategy/.venv/bin/python`) — 需要 3.10+ 支持 `X | None` 类型语法
- TradingAgents-CN 路径: `~/TradingAgents-CN`
- 回测用系统 Python 3.9 (mootdx 兼容 numpy<2.0)
- 扫描含 TradingAgents 用 venv Python 3.12

## 文件结构
- `src/strategy/n_pattern.py` — N字战法核心：形态识别 + 17因子打分
- `src/strategy/backtest.py` — Walk-forward 回测引擎
- `src/screener/daily_scan.py` — 每日全市场扫描 (含 TradingAgents 二次打分)
- `src/screener/data_fetcher.py` — 实时因子数据 (热点/行业/北向)
- `src/screener/tradingagents_scorer.py` — TradingAgents 多智能体二次打分
- `src/cli.py` — CLI 入口 (scan / backtest / daily / push)
- `scripts/run_scan.py` — 独立扫描脚本
- `scripts/bt_full.py` — 全市场回测脚本

## 当前状态 (2026-06-01)
- **22因子系统**: intraday_reversal, volume_climax, sector_relative + ML置信度因子
- **9个反效因子**已通过回测数据驱动反转
- **硬过滤v3**: 涨停基因60日(含7%+阳线), MA60斜率-1%, MA10容忍2%+分级罚分, 近2日涨停
- **退出优化v1**: ATR自适应止损 + 移动止损(3%保本/8%追踪5%) → 单此改进胜率30%→46% (最大单项改进)
- **ML模型**: GradientBoosting walk-forward训练, AUC~0.57, 保守阈值0.66
- **进场确认(放松版)**: 收盘>40%区间 + 成交量<2x20日均量 (严格版55%+阳线会杀太多信号)
- **分级移动止损**: 已测试但回退到原版(2%保本过早, 5%追踪过紧)

## 回测结果汇总 (3017只主板, 2年)

| 配置 | 交易数 | 胜率 | 盈亏比 | 总利润 |
|------|--------|------|--------|--------|
| 全信号+退出优化 (最佳基线) | 237 | 46.0% | 3.36 | 167% |
| +市场环境+进场确认(严格) | 71 | 36.6% | 2.23 | 33% |
| +市场环境+进场确认(放松) | 85 | 41.2% | 2.72 | 49% |
| +市场环境+进场确认+分级止损 | 158 | 38.6% | 1.96 | 51% |
| +市场环境+进场确认+MA100 | 143 | 32.2% | 1.80 | 43% |
| Top-3/day (进场放松,无市场过滤) | 141 | 40.4% | 2.65 | 76% |

**关键结论**:
- 退出优化是唯一有效改进(30%→46%), 其他过滤均降低胜率
- 强度≥90子集: ~62%胜率但仅~48笔/2年(无法满足日推3只)
- ML置信度≥0.7: 51.9%胜率但仅27笔/2年
- **N字战法现实天花板: ~46%全信号, ~52%精选信号**
- 70%胜率日推3只需要根本不同的策略组合或多策略体系

## 改进路线图
1. ✅ 放宽硬过滤 → 交易量~270笔
2. ✅ 22因子+ML置信度 → 胜率30%→35%
3. ✅ ATR止损+移动止损 → 胜率35%→46% (最大单项改进)
4. ✅ Walk-forward ML → 无look-ahead偏差
5. ✅ 强度阈值≥90 → 胜率~62% (但交易量降至~48笔/2年)
6. ✅ 市场环境过滤 — 测试完成, 降低胜率(太多好信号被过滤)
7. ✅ 日内确认入场 — 测试完成, 降低胜率(确认条件滞后)
8. ✅ Top-N每日排名 — 40.4%胜率, 日推1.8只, 未达目标
9. ⏳ 多策略体系 / 深度学习原始K线 → 需要超越N字战法的框架
