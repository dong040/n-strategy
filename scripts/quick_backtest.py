"""N字战法快速回测脚本

取一批股票的历史日线，用 n-strategy 回测引擎跑 walk-forward 回测。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
import pandas as pd

logging.basicConfig(level=logging.WARNING)

from strategy.n_pattern import NPatternParams
from strategy.backtest import BacktestConfig, backtest_single_stock
from mootdx.quotes import Quotes

# ── 回测标的 ──
UNIVERSE = [
    # 今天扫描Top10
    ("603283", "赛腾股份"),
    ("603078", "江化微"),
    ("603373", "安邦护卫"),
    ("603344", "星德胜"),
    ("600353", "旭光电子"),
    ("600903", "贵州燃气"),
    ("001269", "欧晶科技"),
    ("600962", "国投中鲁"),
    ("603726", "朗迪集团"),
    ("000695", "滨海能源"),
    # 大市值+各行业代表
    ("601012", "隆基绿能"),
    ("002594", "比亚迪"),
    ("600519", "贵州茅台"),
    ("000858", "五粮液"),
    ("300750", "宁德时代"),
    ("600036", "招商银行"),
    ("601318", "中国平安"),
    ("000651", "格力电器"),
    ("002415", "海康威视"),
    ("600276", "恒瑞医药"),
    ("000333", "美的集团"),
    ("002230", "科大讯飞"),
    ("601888", "中国中免"),
    ("600809", "山西汾酒"),
    ("300059", "东方财富"),
    # 中小盘活跃股
    ("600313", "农发种业"),
    ("601020", "华源控股"),
    ("002357", "富临运业"),
    ("300827", "上能电气"),
    ("688981", "中芯国际"),
]

# ── 参数 ──
params = NPatternParams(stop_loss_pct=0.02)  # 原始2%止损
config = BacktestConfig(
    commission_pct=0.00025,
    stamp_tax_pct=0.001,
    slippage_pct=0.001,
    init_cash=1_000_000,
    max_position_pct=0.2,
    min_strength=90,      # 只做中高信号
    max_wait_days=5,
    close_stop=True,       # 收盘价跌破才止损
)

client = Quotes.factory(market='std', timeout=10)

all_trades = []
all_results = []

print(f"{'代码':<10} {'名称':<10} {'交易':>5} {'胜率':>8} {'总收益':>10} {'年化':>8} {'夏普':>6} {'最大回撤':>8} {'盈亏比':>8} {'均盈%':>8} {'均损%':>8} {'均持天':>6}")
print("-" * 120)

for code, name in UNIVERSE:
    try:
        df = client.bars(symbol=code, frequency=9, start=0, offset=800)
        if df is None or len(df) < 150:
            print(f"{code:<10} {name:<10} {'数据不足':>5}")
            continue

        df = df.rename(columns={
            'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
            'volume': 'volume',
        })
        df['date'] = df.index.astype(str)

        result = backtest_single_stock(code, name, df, params, config)

        all_results.append(result)
        if result.total_trades > 0:
            for t in result.trades:
                all_trades.append(t)

        print(f"{code:<10} {name:<10} {result.total_trades:>5} {result.win_rate:>7.1f}% {result.total_return:>9.2f}% {result.annual_return:>7.2f}% {result.sharpe_ratio:>6.2f} {result.max_drawdown:>7.2f}% {result.profit_factor:>8.2f} {result.avg_profit:>7.1f}% {result.avg_loss:>7.1f}% {result.avg_hold_days:>5.1f}d")

    except Exception as e:
        print(f"{code:<10} {name:<10} {'错误':>5}: {e}")

# ── 汇总 ──
print()
print("=" * 120)
print("汇总统计")
print("=" * 120)

total_trades = sum(r.total_trades for r in all_results)
if all_trades:
    wins = [t for t in all_trades if t.profit > 0]
    losses = [t for t in all_trades if t.profit <= 0]
    win_rate = len(wins) / len(all_trades) * 100

    total_profit = sum(t.profit for t in wins)
    total_loss = abs(sum(t.profit for t in losses))
    profit_factor = total_profit / total_loss if total_loss > 0 else 999

    avg_profit = sum(t.profit_pct for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.profit_pct for t in losses) / len(losses) if losses else 0
    avg_hold = sum(
        (pd.to_datetime(t.exit_date) - pd.to_datetime(t.entry_date)).days
        for t in all_trades
    ) / len(all_trades)

    total_r = sum(t.profit for t in all_trades)

    print(f"总交易: {total_trades} 笔")
    print(f"盈利: {len(wins)} 笔 | 亏损: {len(losses)} 笔")
    print(f"胜率: {win_rate:.1f}%")
    print(f"总利润: {total_r:,.0f} 元")
    print(f"均盈: {avg_profit:.1f}% | 均损: {avg_loss:.1f}%")
    print(f"盈亏比: {profit_factor:.2f}")
    print(f"均持天数: {avg_hold:.1f} 天")

    exit_reasons = {}
    for t in all_trades:
        r = t.exit_reason
        exit_reasons[r] = exit_reasons.get(r, 0) + 1
    print(f"出场分布: {exit_reasons}")

    # 强度分层
    for label, lo, hi in [("强(≥110)", 110, 999), ("中(90-109)", 90, 109), ("弱(<90)", 0, 89)]:
        tier = [t for t in all_trades if lo <= t.strength < hi]
        if tier:
            t_win = len([t for t in tier if t.profit > 0]) / len(tier) * 100
            t_avg = sum(t.profit_pct for t in tier) / len(tier)
            print(f"  强度{label}: {len(tier)}笔 胜率{t_win:.1f}% 均收益{t_avg:.1f}%")
else:
    print("无交易记录")
