"""提取回测所有交易的22因子数据，导出CSV供ML分析"""
import sys, os, logging, time, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
logging.basicConfig(level=logging.WARNING)

from strategy.n_pattern import NPatternParams
from strategy.backtest import BacktestConfig, backtest_single_stock, _factor_kwargs
from mootdx.quotes import Quotes

params = NPatternParams(stop_loss_pct=0.02)
config = BacktestConfig(
    commission_pct=0.00025, stamp_tax_pct=0.001, slippage_pct=0.001,
    init_cash=1_000_000, max_position_pct=0.2, min_strength=65,
    max_wait_days=10, close_stop=True,
)

client = Quotes.factory(market='std', timeout=15)

import akshare as ak
stock_info = ak.stock_info_a_code_name()
df = stock_info[['code', 'name']].copy()
main = df[df['code'].str.match(r'^(60\d{4}|00[0-4]\d{3})$')].copy()
main = main[~main['name'].str.contains('ST', na=False)]
universe = list(zip(main['code'], main['name']))
print(f"主板共 {len(universe)} 只")

all_trades = []
errors = 0
t0 = time.time()

for idx, (code, name) in enumerate(universe):
    try:
        df = client.bars(symbol=code, frequency=9, start=0, offset=500)
        if df is None or len(df) < 150:
            continue
        df['date'] = df.index.astype(str)
        result = backtest_single_stock(code, name, df, params, config)
        all_trades.extend(result.trades)

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"... {idx+1}/{len(universe)} ({elapsed:.0f}s, {len(all_trades)} trades)")

    except Exception:
        errors += 1
        continue

elapsed = time.time() - t0

# Export
factor_keys = list(_factor_kwargs({}).keys())
out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'trade_factors.csv')

with open(out_path, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['code', 'name', 'entry_date', 'exit_date', 'entry_price', 'exit_price',
              'profit_pct', 'strength', 'exit_reason', 'is_win'] + factor_keys
    writer.writerow(header)
    for t in all_trades:
        factors = {k: getattr(t, k, 0) for k in factor_keys}
        row = [t.code, t.name, t.entry_date, t.exit_date, t.entry_price, t.exit_price,
               round(t.profit_pct, 4), t.strength, t.exit_reason, 1 if t.profit > 0 else 0]
        row += [factors[k] for k in factor_keys]
        writer.writerow(row)

wins = [t for t in all_trades if t.profit > 0]
losses = [t for t in all_trades if t.profit <= 0]
total_r = sum(t.profit for t in all_trades)

print(f"\n=== 导出完成 (耗时 {elapsed:.0f}s) ===")
print(f"回测股票: {len(universe)} 只 (错误: {errors})")
print(f"总交易: {len(all_trades)} 笔")
print(f"盈利: {len(wins)} ({len(wins)/max(1,len(all_trades))*100:.1f}%) | 亏损: {len(losses)}")
print(f"总利润: {total_r:,.0f} 元")
print(f"均盈: {np.mean([t.profit_pct for t in wins]):.2f}%" if wins else "N/A")
print(f"均损: {np.mean([t.profit_pct for t in losses]):.2f}%" if losses else "N/A")
print(f"导出 → {out_path}")
