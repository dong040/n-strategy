"""止损后二次入场回测 — 对比原始策略 vs 跌破支撑90%再买入"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from strategy.n_pattern import NPatternParams, find_n_signals
from strategy.backtest import BacktestConfig, Trade, get_limit_pct

from mootdx.quotes import Quotes


def backtest_with_reentry(code, name, ohlcv, params, config):
    """带回补的回测：首次止损后，等跌到原入场价90%再买一次，跌破二次买入价即止损"""

    if len(ohlcv) < 120:
        return [], []

    ohlcv = ohlcv.sort_values('date').reset_index(drop=True)
    opens = ohlcv['open'].values
    highs = ohlcv['high'].values
    lows = ohlcv['low'].values
    closes = ohlcv['close'].values
    vols = ohlcv['volume'].values
    dates = ohlcv['date'].values

    limit_pct = get_limit_pct(code)
    cash = config.init_cash
    position = 0
    entry_price = 0.0
    entry_date = ""
    active_stop = 0.0
    active_target = 0.0
    active_strength = 0
    in_trade = False
    hold_days = 0
    pending_entry = None
    max_wait = config.max_wait_days
    min_window = 120
    is_reentry = False  # 标记是否为二次入场

    # 二次入场：记录首次止损后等待的机会
    reentry_pending = False
    reentry_price = 0.0     # 二次入场价 = 首次入场价 * 0.9
    reentry_target = 0.0    # 沿用首次目标
    reentry_strength = 0
    trades = []
    equity = []

    for i in range(min_window, len(closes)):
        # === 持仓管理 ===
        if in_trade:
            hold_days += 1

            # 止损
            stop_triggered = closes[i] <= active_stop if config.close_stop else lows[i] <= active_stop
            if active_stop > 0 and stop_triggered:
                exit_price = active_stop * (1 - config.slippage_pct)
                sell_value = position * exit_price * (1 - config.commission_pct - config.stamp_tax_pct)
                profit = sell_value - position * entry_price * (1 + config.commission_pct)
                cash += sell_value
                exit_reason = "reentry_stop" if is_reentry else "stop_loss"
                trades.append(Trade(
                    code=code, name=name,
                    entry_date=entry_date, exit_date=str(dates[i])[:10],
                    entry_price=entry_price, exit_price=exit_price,
                    shares=position, profit=profit,
                    profit_pct=(exit_price / entry_price - 1) * 100,
                    strength=active_strength, exit_reason=exit_reason,
                ))
                position = 0
                in_trade = False
                hold_days = 0
                active_stop = 0

                # —— 首次止损后，设置二次入场机会 ——
                if not is_reentry and not reentry_pending:
                    reentry_pending = True
                    reentry_price = round(entry_price * 0.9, 2)
                    reentry_target = active_target
                    reentry_strength = active_strength
                is_reentry = False

                equity.append(cash)
                continue

            # 止盈
            if active_target > 0 and highs[i] >= active_target:
                exit_price = active_target * (1 - config.slippage_pct)
                sell_value = position * exit_price * (1 - config.commission_pct - config.stamp_tax_pct)
                profit = sell_value - position * entry_price * (1 + config.commission_pct)
                cash += sell_value
                exit_reason = "reentry_tp" if is_reentry else "take_profit"
                trades.append(Trade(
                    code=code, name=name,
                    entry_date=entry_date, exit_date=str(dates[i])[:10],
                    entry_price=entry_price, exit_price=exit_price,
                    shares=position, profit=profit,
                    profit_pct=(exit_price / entry_price - 1) * 100,
                    strength=active_strength, exit_reason=exit_reason,
                ))
                position = 0
                in_trade = False
                hold_days = 0
                active_target = 0
                is_reentry = False
                reentry_pending = False
                equity.append(cash)
                continue

            # 强平 30 天
            if hold_days >= 30:
                exit_price = closes[i] * (1 - config.slippage_pct)
                sell_value = position * exit_price * (1 - config.commission_pct - config.stamp_tax_pct)
                profit = sell_value - position * entry_price * (1 + config.commission_pct)
                cash += sell_value
                exit_reason = "reentry_fe" if is_reentry else "force_exit"
                trades.append(Trade(
                    code=code, name=name,
                    entry_date=entry_date, exit_date=str(dates[i])[:10],
                    entry_price=entry_price, exit_price=exit_price,
                    shares=position, profit=profit,
                    profit_pct=(exit_price / entry_price - 1) * 100,
                    strength=active_strength, exit_reason=exit_reason,
                ))
                position = 0
                in_trade = False
                hold_days = 0
                is_reentry = False
                reentry_pending = False
                equity.append(cash)
                continue

            equity.append(cash + position * closes[i])
            continue

        # === 二次入场：等价格跌到首次入场价的90% ===
        if reentry_pending:
            # 涨跌停检查
            prev_close = closes[i - 1]
            if reentry_price > prev_close * (1 + limit_pct) * 1.001:
                reentry_pending = False
                equity.append(cash)
                continue
            if reentry_price < prev_close * (1 - limit_pct) * 0.999:
                reentry_pending = False
                equity.append(cash)
                continue

            # 当日最低价触及二次入场价
            if lows[i] <= reentry_price:
                # 确认：收盘高于二次入场价（有支撑）
                if closes[i] < reentry_price:
                    equity.append(cash)
                    continue

                # 放量砸盘不接
                if i > 0:
                    body_bottom = min(opens[i], closes[i])
                    shadow_ratio = (body_bottom - lows[i]) / closes[i] if closes[i] > 0 else 0
                    if vols[i] > vols[i - 1] * 1.5 and shadow_ratio < 0.005:
                        equity.append(cash)
                        continue

                buy_price = reentry_price
                max_shares = int(cash * config.max_position_pct / buy_price)
                shares = max(100, max_shares // 100 * 100)
                cost = shares * buy_price * (1 + config.commission_pct)
                if cost > cash:
                    shares = int(cash * 0.99 / buy_price) // 100 * 100
                    cost = shares * buy_price * (1 + config.commission_pct)
                if shares < 100:
                    equity.append(cash)
                    continue

                cash -= cost
                position = shares
                entry_price = buy_price
                entry_date = str(dates[i])[:10]
                # 二次入场止损：跌破买入价即止损（2% 止损）
                active_stop = round(buy_price * (1 - config.stop_loss_pct_secondary), 2)
                active_target = reentry_target
                active_strength = reentry_strength
                in_trade = True
                hold_days = 0
                is_reentry = True
                reentry_pending = False

                equity.append(cash + position * closes[i])
                continue

            equity.append(cash)
            continue

        # === 限价单等待成交（原始入场） ===
        if pending_entry is not None:
            pe = pending_entry

            if pe.get('confirmed'):
                buy_price = pe['confirm_close']
                max_shares = int(cash * config.max_position_pct / buy_price)
                shares = max(100, max_shares // 100 * 100)
                cost = shares * buy_price * (1 + config.commission_pct)
                if cost > cash:
                    shares = int(cash * 0.99 / buy_price) // 100 * 100
                    cost = shares * buy_price * (1 + config.commission_pct)
                if shares >= 100:
                    cash -= cost
                    position = shares
                    entry_price = buy_price
                    entry_date = str(dates[i - 1])[:10]
                    stop_pct = (pe['price'] - pe['stop']) / pe['price']
                    active_stop = round(buy_price * (1 - stop_pct), 2)
                    active_target = pe['target']
                    active_strength = pe.get('strength', 0)
                    in_trade = True
                    hold_days = 1
                pending_entry = None
                equity.append(cash + position * closes[i])
                continue

            limit_price = pe['price']

            prev_close = closes[i - 1]
            if limit_price > prev_close * (1 + limit_pct) * 1.001:
                pending_entry = None; equity.append(cash); continue
            if limit_price < prev_close * (1 - limit_pct) * 0.999:
                pending_entry = None; equity.append(cash); continue

            if lows[i] <= limit_price:
                if closes[i] < limit_price:
                    pending_entry = None; equity.append(cash); continue

                if i > 0:
                    body_bottom = min(opens[i], closes[i])
                    shadow_ratio = (body_bottom - lows[i]) / closes[i] if closes[i] > 0 else 0
                    vol_expanding = vols[i] > vols[i - 1] * 1.5
                    no_lower_shadow = shadow_ratio < 0.005
                    if vol_expanding and no_lower_shadow:
                        pending_entry = None; equity.append(cash); continue

                if i >= 20:
                    avg_vol_20 = float(np.mean(vols[i - 20:i]))
                    if vols[i] > avg_vol_20 * 1.2:
                        pending_entry = None; equity.append(cash); continue

                pe['confirmed'] = True
                pe['confirm_close'] = closes[i]
                equity.append(cash)
                continue

            pe['waited'] += 1
            if pe['waited'] > max_wait:
                pending_entry = None
            equity.append(cash)
            continue

        # === 信号检测 ===
        if i % 5 != 0:
            equity.append(cash)
            continue

        # 二次入场等待中不产生新信号
        if reentry_pending:
            equity.append(cash)
            continue

        window_slice = slice(max(0, i - 500), i)
        try:
            sigs = find_n_signals(
                opens[window_slice], highs[window_slice], lows[window_slice],
                closes[window_slice], vols[window_slice], params,
            )
        except Exception:
            equity.append(cash)
            continue

        if not sigs:
            equity.append(cash)
            continue

        best = max(sigs, key=lambda s: s['strength'])
        if best['strength'] < config.min_strength:
            equity.append(cash)
            continue

        pending_entry = {
            'price': best['entry_price'],
            'stop': best['stop_loss'],
            'target': best['target_price'],
            'strength': best['strength'],
            'waited': 0,
        }
        equity.append(cash)

    # 收盘强平
    if in_trade and position > 0:
        last_close = closes[-1]
        exit_price = last_close * (1 - config.slippage_pct)
        sell_value = position * exit_price * (1 - config.commission_pct - config.stamp_tax_pct)
        profit = sell_value - position * entry_price * (1 + config.commission_pct)
        cash += sell_value
        exit_reason = "reentry_fe" if is_reentry else "force_exit"
        trades.append(Trade(
            code=code, name=name,
            entry_date=entry_date, exit_date=str(dates[-1])[:10],
            entry_price=entry_price, exit_price=exit_price,
            shares=position, profit=profit,
            profit_pct=(exit_price / entry_price - 1) * 100,
            strength=active_strength, exit_reason=exit_reason,
        ))

    return trades, equity


# ── 运行对比 ──
print("获取主板股票列表...")
import akshare as ak
stock_info = ak.stock_info_a_code_name()
df = stock_info[['code', 'name']].copy()
main = df[df['code'].str.match(r'^(60\d{4}|00[0-4]\d{3})$')].copy()
main = main[~main['name'].str.contains('ST', na=False)]
universe = list(zip(main['code'], main['name']))
print(f"主板 {len(universe)} 只，每只 500 根日线\n")

client = Quotes.factory(market='std', timeout=10)

# ── 配置A：原始策略 5% 止损 ──
params_a = NPatternParams(stop_loss_pct=0.05)
config_a = BacktestConfig(
    commission_pct=0.00025, stamp_tax_pct=0.001, slippage_pct=0.001,
    init_cash=1_000_000, max_position_pct=0.2,
    min_strength=65, max_wait_days=5, close_stop=True,
)

# ── 配置B：二次入场策略，首次也用 5% 止损 ──
params_b = NPatternParams(stop_loss_pct=0.05)
config_b = BacktestConfig(
    commission_pct=0.00025, stamp_tax_pct=0.001, slippage_pct=0.001,
    init_cash=1_000_000, max_position_pct=0.2,
    min_strength=65, max_wait_days=5, close_stop=True,
)
config_b.stop_loss_pct_secondary = 0.02  # 二次入场止损 2%：跌破买入价即出

# ── 对比不同二次止损 ──
configs = [
    ("原始 (5%止损)", config_a, params_a, "original"),
    ("二次入场 (再止损2%)", config_b, params_b, "reentry"),
]

all_trades_data = {}
for label, cfg, prm, mode in configs:
    t0 = time.time()
    trades_list = []

    for code, name in universe:
        try:
            stock_df = client.bars(symbol=code, frequency=9, start=0, offset=500)
            if stock_df is None or len(stock_df) < 150:
                continue
            stock_df['date'] = stock_df.index.astype(str)
            if mode == "original":
                from strategy.backtest import backtest_single_stock
                result = backtest_single_stock(code, name, stock_df, prm, cfg)
                trades_list.extend(result.trades)
            else:
                raw_trades, _ = backtest_with_reentry(code, name, stock_df, prm, cfg)
                trades_list.extend(raw_trades)
        except Exception:
            pass

    elapsed = time.time() - t0
    all_trades_data[label] = (trades_list, elapsed)

print()
print("=" * 130)
print(f"{'策略':<30} {'交易':>6} {'胜率':>8} {'盈亏比':>8} {'总利润':>14} {'均盈%':>8} {'均损%':>8}", end="")
print(f" {'首次':>6} {'二次':>6} {'止盈':>6} {'止损':>6} {'强平':>6} {'二次止盈':>8} {'二次止损':>8} {'二次强平':>8}")
print("-" * 130)

for label, (trades, elapsed) in all_trades_data.items():
    if not trades:
        print(f"{label:<30} {'无交易':>6}")
        continue
    wins = [t for t in trades if t.profit > 0]
    losses = [t for t in trades if t.profit <= 0]
    wr = len(wins) / len(trades) * 100
    tp = sum(t.profit for t in wins)
    tl = abs(sum(t.profit for t in losses))
    pf = tp / tl if tl > 0 else 999
    ap = np.mean([t.profit_pct for t in wins]) if wins else 0
    al = np.mean([t.profit_pct for t in losses]) if losses else 0
    total_r = sum(t.profit for t in trades)

    exits = {}
    for t in trades:
        exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1

    tp_n = exits.get('take_profit', 0)
    sl_n = exits.get('stop_loss', 0)
    fe_n = exits.get('force_exit', 0)
    re_tp = exits.get('reentry_tp', 0)
    re_sl = exits.get('reentry_stop', 0)
    re_fe = exits.get('reentry_fe', 0)
    first = tp_n + sl_n + fe_n
    second = re_tp + re_sl + re_fe

    print(f"{label:<30} {len(trades):>6} {wr:>7.1f}% {pf:>8.2f} {total_r:>14,.0f} {ap:>7.1f}% {al:>7.1f}%", end="")
    print(f" {first:>6} {second:>6} {tp_n:>6} {sl_n:>6} {fe_n:>6} {re_tp:>8} {re_sl:>8} {re_fe:>8}")

    # 分层：首次 vs 二次
    print(f"  └ 首次入场: {first}笔 ", end="")
    f_wins = len([t for t in trades if t.profit > 0 and not t.exit_reason.startswith('reentry')])
    # 简单显示
    if sl_n > 0:
        avg_sl = np.mean([t.profit_pct for t in trades if t.exit_reason == 'stop_loss'])
        print(f"| 止损{sl_n}笔 均损{avg_sl:.1f}%", end="")
    if tp_n > 0:
        avg_tp = np.mean([t.profit_pct for t in trades if t.exit_reason == 'take_profit'])
        print(f" | 止盈{tp_n}笔 均盈{avg_tp:.1f}%", end="")
    if fe_n > 0:
        avg_fe = np.mean([t.profit_pct for t in trades if t.exit_reason == 'force_exit'])
        print(f" | 强平{fe_n}笔 均盈{avg_fe:.1f}%", end="")
    print()

    if second > 0:
        print(f"  └ 二次入场: {second}笔 ", end="")
        if re_sl > 0:
            avg_re_sl = np.mean([t.profit_pct for t in trades if t.exit_reason == 'reentry_stop'])
            print(f"| 止损{re_sl}笔 均损{avg_re_sl:.1f}%", end="")
        if re_tp > 0:
            avg_re_tp = np.mean([t.profit_pct for t in trades if t.exit_reason == 'reentry_tp'])
            print(f" | 止盈{re_tp}笔 均盈{avg_re_tp:.1f}%", end="")
        if re_fe > 0:
            avg_re_fe = np.mean([t.profit_pct for t in trades if t.exit_reason == 'reentry_fe'])
            print(f" | 强平{re_fe}笔 均盈{avg_re_fe:.1f}%", end="")
        print()

print()
print(f"注: 原始策略用固定5%止损 | 二次入场：首次止损后等跌到原入场价90%再买，跌破买入价{(config_b.stop_loss_pct_secondary*100):.0f}%止损")
