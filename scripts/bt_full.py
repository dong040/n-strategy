"""N字战法全市场回测 — 全部主板股票，近两年数据"""

import sys, os, logging, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)

from strategy.n_pattern import NPatternParams
from strategy.backtest import BacktestConfig, backtest_single_stock
from mootdx.quotes import Quotes

# ── 参数（最优配置） ──
params = NPatternParams(stop_loss_pct=0.02)
config = BacktestConfig(
    commission_pct=0.00025,
    stamp_tax_pct=0.001,
    slippage_pct=0.001,
    init_cash=1_000_000,
    max_position_pct=0.2,
    min_strength=65,
    max_wait_days=5,
    close_stop=True,
)

client = Quotes.factory(market='std', timeout=10)

# ── 获取 上证指数 日线，计算市场环境（MA60上方=可交易）──
print("获取上证指数...")
try:
    import akshare as ak
    idx = ak.stock_zh_index_daily(symbol='sh000001')
    idx_closes = idx['close'].values.astype(float)
    idx_dates = idx['date'].values.astype(str)
    market_regime = set()
    # Only use last 600 days to cover the backtest period
    recent_closes = idx_closes[-600:]
    recent_dates = idx_dates[-600:]
    for i in range(60, len(recent_closes)):
        ma60 = float(np.mean(recent_closes[i-60:i]))
        if recent_closes[i] > ma60:
            market_regime.add(str(recent_dates[i])[:10])
    print(f"  上证数据: {len(recent_closes)} 天, 可交易日: {len(market_regime)} 天 ({len(market_regime)/max(1,len(recent_closes))*100:.0f}%)")
except Exception as e:
    print(f"  上证数据获取失败: {e}, 跳过市场过滤")
    market_regime = None

# ── 获取全部主板标的 ──
print("获取主板股票列表...")
import akshare as ak
stock_info = ak.stock_info_a_code_name()
df = stock_info[['code', 'name']].copy()
main = df[df['code'].str.match(r'^(60\d{4}|00[0-4]\d{3})$')].copy()
main = main[~main['name'].str.contains('ST', na=False)]
universe = list(zip(main['code'], main['name']))
print(f"主板共 {len(universe)} 只，开始回测（每只 ~2年/500根日线）...")

all_trades = []
all_results = []
errors = 0
t0 = time.time()

print(f"{'代码':<10} {'名称':<10} {'交易':>5} {'胜率':>8} {'总收益':>10} {'年化':>8} {'夏普':>6} {'最大回撤':>8} {'盈亏比':>8} {'均盈%':>8} {'均损%':>8} {'均持天':>6}")
print("-" * 120)

for idx, (code, name) in enumerate(universe):
    try:
        df = client.bars(symbol=code, frequency=9, start=0, offset=500)
        if df is None or len(df) < 150:
            continue

        df['date'] = df.index.astype(str)
        result = backtest_single_stock(code, name, df, params, config, market_regime=market_regime)
        all_results.append(result)
        for t in result.trades:
            all_trades.append(t)

        if result.total_trades > 0:
            print(f"{code:<10} {name:<10} {result.total_trades:>5} {result.win_rate:>7.1f}% {result.total_return:>9.2f}% {result.annual_return:>7.2f}% {result.sharpe_ratio:>6.2f} {result.max_drawdown:>7.2f}% {result.profit_factor:>8.2f} {result.avg_profit:>7.1f}% {result.avg_loss:>7.1f}% {result.avg_hold_days:>5.1f}d")

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"... {idx+1}/{len(universe)} ({elapsed:.0f}s elapsed)")

    except Exception:
        errors += 1
        continue

elapsed = time.time() - t0

# ── 汇总 ──
print()
print("=" * 120)
print(f"全市场回测汇总 (耗时 {elapsed:.0f}s)")
print("=" * 120)

total_trades = len(all_trades)
if all_trades:
    wins = [t for t in all_trades if t.profit > 0]
    losses = [t for t in all_trades if t.profit <= 0]
    win_rate = len(wins) / total_trades * 100

    total_profit = sum(t.profit for t in wins)
    total_loss = abs(sum(t.profit for t in losses))
    profit_factor = total_profit / total_loss if total_loss > 0 else 999

    avg_profit = sum(t.profit_pct for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.profit_pct for t in losses) / len(losses) if losses else 0

    total_r = sum(t.profit for t in all_trades)

    print(f"回测股票: {len(all_results)} 只 (错误: {errors})")
    print(f"总交易: {total_trades} 笔")
    print(f"盈利: {len(wins)} 笔 | 亏损: {len(losses)} 笔")
    print(f"胜率: {win_rate:.1f}%")
    print(f"总利润: {total_r:,.0f} 元")
    print(f"均盈: {avg_profit:.1f}% | 均损: {avg_loss:.1f}%")
    print(f"盈亏比: {profit_factor:.2f}")

    exit_reasons = {}
    for t in all_trades:
        r = t.exit_reason
        exit_reasons[r] = exit_reasons.get(r, 0) + 1
    print(f"出场分布: {exit_reasons}")

    print()
    print("强度分层:")
    for label, lo, hi in [("强(≥110)", 110, 999), ("中(90-109)", 90, 109), ("弱(<90)", 0, 89)]:
        tier = [t for t in all_trades if lo <= t.strength < hi]
        if tier:
            t_win = len([t for t in tier if t.profit > 0]) / len(tier) * 100
            t_avg = sum(t.profit_pct for t in tier) / len(tier)
            print(f"  {label}: {len(tier)}笔 胜率{t_win:.1f}% 均收益{t_avg:.1f}%")

    print()
    print("出场原因分层:")
    for reason in ["take_profit", "stop_loss", "force_exit"]:
        tier = [t for t in all_trades if t.exit_reason == reason]
        if tier:
            t_win = len([t for t in tier if t.profit > 0]) / len(tier) * 100
            t_avg = sum(t.profit_pct for t in tier) / len(tier)
            print(f"  {reason}: {len(tier)}笔 胜率{t_win:.1f}% 均收益{t_avg:.1f}%")
else:
    print("无交易记录")
