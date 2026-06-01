"""Portfolio-level backtest with daily top-N ranking.

Collects all signals across all stocks, then for each entry date,
ranks by composite score and keeps only the top N trades.

This simulates the "recommend N stocks per day" requirement.
"""

import sys, os, logging, time, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)

from strategy.n_pattern import NPatternParams
from strategy.backtest import BacktestConfig, backtest_single_stock
from mootdx.quotes import Quotes

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

N_TOP = int(sys.argv[1]) if len(sys.argv) > 1 else 3

print(f"获取主板股票列表...")
import akshare as ak
stock_info = ak.stock_info_a_code_name()
df = stock_info[['code', 'name']].copy()
main = df[df['code'].str.match(r'^(60\d{4}|00[0-4]\d{3})$')].copy()
main = main[~main['name'].str.contains('ST', na=False)]
universe = list(zip(main['code'], main['name']))
print(f"主板共 {len(universe)} 只，收集信号中...")

all_trades = []
errors = 0
t0 = time.time()

for idx, (code, name) in enumerate(universe):
    try:
        df_ohlcv = client.bars(symbol=code, frequency=9, start=0, offset=500)
        if df_ohlcv is None or len(df_ohlcv) < 150:
            continue
        df_ohlcv['date'] = df_ohlcv.index.astype(str)
        result = backtest_single_stock(code, name, df_ohlcv, params, config)
        for t in result.trades:
            all_trades.append({
                'code': t.code,
                'name': t.name,
                'entry_date': t.entry_date,
                'exit_date': t.exit_date,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'profit': t.profit,
                'profit_pct': t.profit_pct,
                'strength': t.strength,
                'exit_reason': t.exit_reason,
                'factor_score': t.factor_score,
                'ml_confidence': t.ml_confidence,
                'ml_confidence_score': t.ml_confidence_score,
                'hold_days': (pd.to_datetime(t.exit_date) - pd.to_datetime(t.entry_date)).days,
            })
    except Exception:
        errors += 1
        continue

    if (idx + 1) % 500 == 0:
        elapsed = time.time() - t0
        print(f"... {idx+1}/{len(universe)} ({elapsed:.0f}s, {len(all_trades)} signals)")

elapsed = time.time() - t0
print(f"\n收集完成: {len(all_trades)} 笔交易 ({elapsed:.0f}s, errors={errors})")

if not all_trades:
    print("无交易记录")
    sys.exit(0)

# ── Post-process: Top-N per day ──
trades_df = pd.DataFrame(all_trades)
trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])

# Composite score: strength weighted by ML confidence
trades_df['composite'] = trades_df['strength'] * (0.5 + trades_df['ml_confidence'])

# For each entry date, keep top N by composite score
ranked = trades_df.copy()
ranked['day_rank'] = ranked.groupby('entry_date')['composite'].rank(method='first', ascending=False)
topn = ranked[ranked['day_rank'] <= N_TOP].copy()

print(f"\n{'='*100}")
print(f"Top-{N_TOP} per day 回测结果（全市场 {len(all_trades)} 笔 → Top-N {len(topn)} 笔）")
print(f"{'='*100}")

total_trades = len(topn)
wins = topn[topn['profit'] > 0]
losses = topn[topn['profit'] <= 0]
win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
total_profit = wins['profit'].sum() if len(wins) > 0 else 0
total_loss = abs(losses['profit'].sum()) if len(losses) > 0 else 0
profit_factor = total_profit / total_loss if total_loss > 0 else 999
avg_profit = wins['profit_pct'].mean() if len(wins) > 0 else 0
avg_loss = losses['profit_pct'].mean() if len(losses) > 0 else 0

print(f"总交易: {total_trades} 笔")
print(f"盈利: {len(wins)} 笔 | 亏损: {len(losses)} 笔")
print(f"胜率: {win_rate:.1f}%")
print(f"总利润: {topn['profit'].sum():,.0f} 元")
print(f"均盈: {avg_profit:.1f}% | 均损: {avg_loss:.1f}%")
print(f"盈亏比: {profit_factor:.2f}")
print(f"日均交易: {total_trades / max(1, trades_df['entry_date'].nunique()):.1f}")

# Compare to baseline
print(f"\n基线（全信号）: {len(all_trades)}笔 "
      f"胜率{len(trades_df[trades_df['profit']>0])/len(trades_df)*100:.1f}% "
      f"总利润{trades_df['profit'].sum():,.0f}元")

print(f"\n强度分层 (Top-{N_TOP}):")
for label, lo, hi in [("强(>=110)", 110, 999), ("中(90-109)", 90, 109), ("弱(<90)", 0, 89)]:
    tier = topn[(topn['strength'] >= lo) & (topn['strength'] < hi)]
    if len(tier) > 0:
        t_win = len(tier[tier['profit'] > 0]) / len(tier) * 100
        t_avg = tier['profit_pct'].mean()
        print(f"  {label}: {len(tier)}笔 胜率{t_win:.1f}% 均收益{t_avg:.1f}%")

print(f"\nML置信度分层 (Top-{N_TOP}):")
for label, lo, hi in [("高(>=0.7)", 0.7, 1.0), ("中(0.5-0.7)", 0.5, 0.7), ("低(<0.5)", 0.0, 0.5)]:
    tier = topn[(topn['ml_confidence'] >= lo) & (topn['ml_confidence'] < hi)]
    if len(tier) > 0:
        t_win = len(tier[tier['profit'] > 0]) / len(tier) * 100
        print(f"  {label}: {len(tier)}笔 胜率{t_win:.1f}%")

# Composite score threshold analysis
print(f"\nComposite score 阈值分析 (Top-{N_TOP}):")
for thresh in [80, 100, 120, 150]:
    tier = topn[topn['composite'] >= thresh]
    if len(tier) > 0:
        t_win = len(tier[tier['profit'] > 0]) / len(tier) * 100
        print(f"  composite>={thresh}: {len(tier)}笔 胜率{t_win:.1f}%")
