"""止损参数对比回测 — 同一数据集，不同止损档位"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd

from strategy.n_pattern import NPatternParams
from strategy.backtest import BacktestConfig, backtest_single_stock
from mootdx.quotes import Quotes

# ── 止损配置 ──
STOP_CONFIGS = [
    ("固定2%",  2.0, None),
    ("固定3%",  3.0, None),
    ("固定4%",  4.0, None),
    ("固定5%",  5.0, None),
    ("ATR 0.5x", None, 0.5),
    ("ATR 1.0x", None, 1.0),
    ("ATR 1.5x", None, 1.5),
    ("ATR 2.0x", None, 2.0),
]

client = Quotes.factory(market='std', timeout=10)

# ── 获取标的 ──
print("获取主板股票列表...")
import akshare as ak
stock_info = ak.stock_info_a_code_name()
df = stock_info[['code', 'name']].copy()
main = df[df['code'].str.match(r'^(60\d{4}|00[0-4]\d{3})$')].copy()
main = main[~main['name'].str.contains('ST', na=False)]
universe = list(zip(main['code'], main['name']))
print(f"主板 {len(universe)} 只，每只 500 根日线\n")

all_results = {}

for label, stop_pct, atr_mult in STOP_CONFIGS:
    if atr_mult is not None:
        params = NPatternParams(stop_loss_pct=0.02, stop_atr_mult=atr_mult)
    else:
        params = NPatternParams(stop_loss_pct=stop_pct / 100, stop_atr_mult=0.0)

    config = BacktestConfig(
        commission_pct=0.00025, stamp_tax_pct=0.001, slippage_pct=0.001,
        init_cash=1_000_000, max_position_pct=0.2,
        min_strength=65, max_wait_days=5, close_stop=True,
    )

    trades_list = []
    errors = 0
    t0 = time.time()

    for code, name in universe:
        try:
            stock_df = client.bars(symbol=code, frequency=9, start=0, offset=500)
            if stock_df is None or len(stock_df) < 150:
                continue
            stock_df['date'] = stock_df.index.astype(str)
            result = backtest_single_stock(code, name, stock_df, params, config)
            for t in result.trades:
                trades_list.append(t)
        except Exception:
            errors += 1

    elapsed = time.time() - t0

    if trades_list:
        wins = [t for t in trades_list if t.profit > 0]
        losses = [t for t in trades_list if t.profit <= 0]
        wr = len(wins) / len(trades_list) * 100
        tp = sum(t.profit for t in wins)
        tl = abs(sum(t.profit for t in losses))
        pf = tp / tl if tl > 0 else 999
        ap = np.mean([t.profit_pct for t in wins]) if wins else 0
        al = np.mean([t.profit_pct for t in losses]) if losses else 0
        total_r = sum(t.profit for t in trades_list)

        exits = {}
        for t in trades_list:
            exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1

        all_results[label] = {
            'trades': len(trades_list), 'wins': len(wins), 'losses': len(losses),
            'win_rate': wr, 'profit_factor': pf, 'total_profit': total_r,
            'avg_win': ap, 'avg_loss': al, 'exits': exits, 'elapsed': elapsed,
        }
    else:
        all_results[label] = {'trades': 0}

    print(f"[{label}] {len(trades_list)}笔 胜率{wr:.1f}% 盈亏比{pf:.2f} 总利润{total_r:,.0f} 均盈{ap:.1f}% 均损{al:.1f}% ({elapsed:.0f}s)")

# ── 汇总对比 ──
print()
print("=" * 130)
print(f"{'止损配置':<14} {'交易':>6} {'胜率':>8} {'盈亏比':>8} {'总利润':>14} {'均盈%':>8} {'均损%':>8} {'止盈':>6} {'止损':>6} {'强平':>6}")
print("-" * 130)
for label, r in all_results.items():
    if r['trades'] > 0:
        exits = r['exits']
        tp_n = exits.get('take_profit', 0)
        sl_n = exits.get('stop_loss', 0)
        fe_n = exits.get('force_exit', 0)
        print(f"{label:<14} {r['trades']:>6} {r['win_rate']:>7.1f}% {r['profit_factor']:>8.2f} {r['total_profit']:>14,.0f} {r['avg_win']:>7.1f}% {r['avg_loss']:>7.1f}% {tp_n:>6} {sl_n:>6} {fe_n:>6}")
print()
print("注: 每档配置都跑了同一批 ~2985 只股票的全量回测")
