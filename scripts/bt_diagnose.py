"""N字战法回测诊断 — 找出赢家和输家的差异

不只是统计胜率，而是分析每笔交易的特征，找出哪些因子区分了赢家和输家。
"""

import sys, os, logging, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

logging.basicConfig(level=logging.WARNING)

from strategy.n_pattern import NPatternParams, find_n_signals
from strategy.backtest import BacktestConfig, backtest_single_stock
from mootdx.quotes import Quotes

params = NPatternParams(stop_loss_pct=0.02)
config = BacktestConfig(
    commission_pct=0.00025, stamp_tax_pct=0.001, slippage_pct=0.001,
    init_cash=1_000_000, max_position_pct=0.2,
    min_strength=50,  # 放宽，收集更多样本
    max_wait_days=5, close_stop=True,
)

client = Quotes.factory(market='std', timeout=10)

# ── 多维度诊断回测 ──
# 修改 backtest，在每笔交易入场的时刻记录信号特征
# 最快的办法：重写一个带诊断的轻量回测循环

import pandas as pd
from dataclasses import dataclass, field
from strategy.n_pattern import find_n_signals
from strategy.backtest import BacktestConfig, get_limit_pct

@dataclass
class DiagTrade:
    code: str
    name: str
    profit_pct: float
    profit: float
    exit_reason: str
    hold_days: int
    # 信号特征
    strength: int
    fib_level: float
    fib_dist: float
    first_rise_pct: float
    retrace_pct: float
    retrace_days: int
    entry_source: str
    stab_ok: bool
    has_vol_shrink: bool
    has_shadow: bool
    ma_bullish: bool
    bullish_days: int
    ma_consistent: bool
    ma_fib_ok: bool
    has_limit_up: bool
    limit_up_count: int
    ma10_broken_close: bool
    ma10_broken_intraday: bool
    rr_ratio: float
    # 新增
    consecutive_yin: int
    ma9: float
    ma10_val: float
    entry_price: float
    stop_loss: float
    target_price: float
    # 大盘相关
    market_pct: float


def diag_backtest(code, name, ohlcv, params, config):
    """带诊断的回测 — 记录每笔交易的信号特征"""
    if len(ohlcv) < 120:
        return []

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
    active_stop = 0.0
    active_target = 0.0
    active_sig_data = {}
    in_trade = False
    hold_days = 0
    pending_entry = None
    pending_sig_data = {}
    max_wait = config.max_wait_days
    min_window = 120

    trades = []

    for i in range(min_window, len(closes)):
        # 持仓管理
        if in_trade:
            hold_days += 1
            exit_now = False
            exit_price_val = 0.0
            exit_reason = ""

            stop_triggered = closes[i] <= active_stop if config.close_stop else lows[i] <= active_stop
            if active_stop > 0 and stop_triggered:
                exit_price_val = active_stop * (1 - config.slippage_pct)
                exit_reason = "stop_loss"
                exit_now = True
            elif active_target > 0 and highs[i] >= active_target:
                exit_price_val = active_target * (1 - config.slippage_pct)
                exit_reason = "take_profit"
                exit_now = True
            elif hold_days >= 30:
                exit_price_val = closes[i] * (1 - config.slippage_pct)
                exit_reason = "force_exit"
                exit_now = True

            if exit_now:
                sell_value = position * exit_price_val * (1 - config.commission_pct - config.stamp_tax_pct)
                profit = sell_value - position * entry_price * (1 + config.commission_pct)
                profit_pct = (exit_price_val / entry_price - 1) * 100

                trades.append(DiagTrade(
                    code=code, name=name,
                    profit_pct=profit_pct, profit=profit,
                    exit_reason=exit_reason, hold_days=hold_days,
                    strength=active_sig_data.get('strength', 0),
                    fib_level=active_sig_data.get('fib_level', 0),
                    fib_dist=active_sig_data.get('fib_dist', 0),
                    first_rise_pct=active_sig_data.get('first_rise_pct', 0),
                    retrace_pct=active_sig_data.get('retrace_pct', 0),
                    retrace_days=active_sig_data.get('retrace_days', 0),
                    entry_source=active_sig_data.get('entry_source', ''),
                    stab_ok=active_sig_data.get('stab_ok', False),
                    has_vol_shrink=active_sig_data.get('has_vol_shrink', False),
                    has_shadow=active_sig_data.get('has_shadow', False),
                    ma_bullish=active_sig_data.get('ma_bullish', False),
                    bullish_days=active_sig_data.get('bullish_days', 0),
                    ma_consistent=active_sig_data.get('ma_consistent', False),
                    ma_fib_ok=active_sig_data.get('ma_fib_ok', False),
                    has_limit_up=active_sig_data.get('has_limit_up', False),
                    limit_up_count=active_sig_data.get('limit_up_count', 0),
                    ma10_broken_close=active_sig_data.get('ma10_broken_close', False),
                    ma10_broken_intraday=active_sig_data.get('ma10_broken_intraday', False),
                    rr_ratio=active_sig_data.get('rr_ratio', 0),
                    consecutive_yin=active_sig_data.get('consecutive_yin', 0),
                    ma9=active_sig_data.get('ma9', 0),
                    ma10_val=active_sig_data.get('ma10', 0),
                    entry_price=entry_price,
                    stop_loss=active_stop,
                    target_price=active_target,
                    market_pct=active_sig_data.get('market_pct', 0),
                ))

                position = 0
                in_trade = False
                hold_days = 0
                active_stop = 0
                active_target = 0
                active_sig_data = {}
            continue

        # 限价单等待
        if pending_entry is not None:
            pe = pending_entry
            limit_price = pe['price']

            prev_close = closes[i - 1]
            if limit_price > prev_close * (1 + limit_pct) * 1.001:
                pending_entry = None; continue
            if limit_price < prev_close * (1 - limit_pct) * 0.999:
                pending_entry = None; continue

            if lows[i] <= limit_price:
                # 确认1: 收盘 > 入场
                if closes[i] < limit_price:
                    pending_entry = None; continue
                # 确认2: 不放量砸盘
                if i > 0:
                    body_bottom = min(opens[i], closes[i])
                    shadow_ratio = (body_bottom - lows[i]) / closes[i] if closes[i] > 0 else 0
                    if vols[i] > vols[i - 1] * 1.5 and shadow_ratio < 0.01:
                        pending_entry = None; continue

                buy_price = limit_price
                max_shares = int(cash * config.max_position_pct / buy_price)
                shares = max(100, max_shares // 100 * 100)
                cost = shares * buy_price * (1 + config.commission_pct)
                if cost > cash:
                    shares = int(cash * 0.99 / buy_price) // 100 * 100
                    cost = shares * buy_price * (1 + config.commission_pct)
                if shares < 100:
                    pending_entry = None; continue

                cash -= cost
                position = shares
                entry_price = buy_price
                active_stop = pe['stop']
                active_target = pe['target']
                active_sig_data = pending_sig_data
                in_trade = True
                hold_days = 0
                pending_entry = None
                pending_sig_data = {}
                continue

            pe['waited'] += 1
            if pe['waited'] > max_wait:
                pending_entry = None
            continue

        # 信号检测
        if i % 5 != 0:
            continue

        window_slice = slice(max(0, i - 500), i)
        try:
            sigs = find_n_signals(
                opens[window_slice], highs[window_slice], lows[window_slice],
                closes[window_slice], vols[window_slice], params,
            )
        except Exception:
            continue

        if not sigs:
            continue

        best = max(sigs, key=lambda s: s['strength'])
        if best['strength'] < config.min_strength:
            continue

        pending_entry = {
            'price': best['entry_price'],
            'stop': best['stop_loss'],
            'target': best['target_price'],
            'strength': best['strength'],
            'waited': 0,
        }
        pending_sig_data = best

    # 收盘强平
    if in_trade and position > 0:
        last_close = closes[-1]
        exit_price_val = last_close * (1 - config.slippage_pct)
        sell_value = position * exit_price_val * (1 - config.commission_pct - config.stamp_tax_pct)
        profit = sell_value - position * entry_price * (1 + config.commission_pct)
        profit_pct = (exit_price_val / entry_price - 1) * 100
        trades.append(DiagTrade(
            code=code, name=name,
            profit_pct=profit_pct, profit=profit,
            exit_reason="force_exit", hold_days=hold_days,
            strength=active_sig_data.get('strength', 0),
            fib_level=active_sig_data.get('fib_level', 0),
            fib_dist=active_sig_data.get('fib_dist', 0),
            first_rise_pct=active_sig_data.get('first_rise_pct', 0),
            retrace_pct=active_sig_data.get('retrace_pct', 0),
            retrace_days=active_sig_data.get('retrace_days', 0),
            entry_source=active_sig_data.get('entry_source', ''),
            stab_ok=active_sig_data.get('stab_ok', False),
            has_vol_shrink=active_sig_data.get('has_vol_shrink', False),
            has_shadow=active_sig_data.get('has_shadow', False),
            ma_bullish=active_sig_data.get('ma_bullish', False),
            bullish_days=active_sig_data.get('bullish_days', 0),
            ma_consistent=active_sig_data.get('ma_consistent', False),
            ma_fib_ok=active_sig_data.get('ma_fib_ok', False),
            has_limit_up=active_sig_data.get('has_limit_up', False),
            limit_up_count=active_sig_data.get('limit_up_count', 0),
            ma10_broken_close=active_sig_data.get('ma10_broken_close', False),
            ma10_broken_intraday=active_sig_data.get('ma10_broken_intraday', False),
            rr_ratio=active_sig_data.get('rr_ratio', 0),
            consecutive_yin=active_sig_data.get('consecutive_yin', 0),
            ma9=active_sig_data.get('ma9', 0),
            ma10_val=active_sig_data.get('ma10', 0),
            entry_price=entry_price,
            stop_loss=active_stop,
            target_price=active_target,
            market_pct=active_sig_data.get('market_pct', 0),
        ))

    return trades


# ── 运行诊断回测 ──
import random
try:
    import akshare as ak
    stock_info = ak.stock_info_a_code_name()
    df = stock_info[['code', 'name']].copy()
    main = df[df['code'].str.match(r'^(60\d{4}|00[0-4]\d{3})$')].copy()
    main = main[~main['name'].str.contains('ST', na=False)]
    universe = list(zip(main['code'], main['name']))
    print(f"主板 {len(universe)} 只, 抽样 200 只")
    sample = random.sample(universe, min(200, len(universe)))
except:
    sample = [(f"{600000+i:06d}", f"TEST{i}") for i in range(50)]

all_trades = []
for code, name in sample:
    try:
        df = client.bars(symbol=code, frequency=9, start=0, offset=800)
        if df is None or len(df) < 150:
            continue
        df['date'] = df.index.astype(str)
        trades = diag_backtest(code, name, df, params, config)
        all_trades.extend(trades)
    except Exception:
        continue

print(f"\n收集到 {len(all_trades)} 笔交易\n")

if not all_trades:
    print("无交易数据")
    sys.exit(1)

wins = [t for t in all_trades if t.profit > 0]
losses = [t for t in all_trades if t.profit <= 0]

def avg(vals): return sum(vals)/len(vals) if vals else 0

# ── 逐维度对比 ──
dims = [
    ("信号强度", "strength"),
    ("费波距离%", "fib_dist", lambda t: t.fib_dist * 100),
    ("首段涨幅%", "first_rise_pct"),
    ("回撤%", "retrace_pct"),
    ("回撤天数", "retrace_days"),
    ("多头持续天数", "bullish_days"),
    ("涨停次数", "limit_up_count"),
    ("盈亏比", "rr_ratio"),
    ("持仓天数", "hold_days"),
]

bool_dims = [
    ("企稳确认(stab_ok)", "stab_ok"),
    ("缩量", "has_vol_shrink"),
    ("下影线", "has_shadow"),
    ("多头排列(MA)", "ma_bullish"),
    ("MA持续(≥3天)", "ma_consistent"),
    ("MA-费波共振", "ma_fib_ok"),
    ("有涨停基因", "has_limit_up"),
    ("收盘跌破MA10", "ma10_broken_close"),
    ("日内触及MA10", "ma10_broken_intraday"),
]

print("=" * 90)
print(f"{'维度':<25} {'赢家均值':>12} {'输家均值':>12} {'差异':>12} {'结论'}")
print("=" * 90)

for label, attr, *transform in dims:
    if transform:
        wv = avg([transform[0](t) for t in wins])
        lv = avg([transform[0](t) for t in losses])
    else:
        wv = avg([getattr(t, attr) for t in wins])
        lv = avg([getattr(t, attr) for t in losses])
    diff = wv - lv
    direction = "赢家更高↑" if diff > 0 else "输家更高↓" if diff < 0 else "无差异"
    print(f"{label:<25} {wv:>12.2f} {lv:>12.2f} {diff:>+12.2f} {direction}")

print()
print("─ 布尔维度 ─")
for label, attr in bool_dims:
    wr = avg([1 if getattr(t, attr) else 0 for t in wins]) * 100
    lr = avg([1 if getattr(t, attr) else 0 for t in losses]) * 100
    diff = wr - lr
    direction = "赢家更多↑" if diff > 0 else "输家更多↓" if diff < 0 else "无差异"
    print(f"{label:<25} {wr:>11.1f}% {lr:>11.1f}% {diff:>+11.1f}% {direction}")

print()
print("─ 入场来源分布 ─")
for src in ['MA9', 'MA10', '费波', '回调低']:
    wc = len([1 for t in wins if t.entry_source == src])
    lc = len([1 for t in losses if t.entry_source == src])
    wr = wc / len(wins) * 100 if wins else 0
    lr = lc / len(losses) * 100 if losses else 0
    print(f"  {src:<8}: 赢家{wc}笔({wr:.0f}%)  输家{lc}笔({lr:.0f}%)")

print()
print("─ 费波级别分布 ─")
for fl in [0.236, 0.382, 0.5, 0.618]:
    wc = len([1 for t in wins if abs(t.fib_level - fl) < 0.01])
    lc = len([1 for t in losses if abs(t.fib_level - fl) < 0.01])
    wr = wc / len(wins) * 100 if wins else 0
    lr = lc / len(losses) * 100 if losses else 0
    diff = wr - lr
    print(f"  费波{fl}: 赢家{wc}笔({wr:.0f}%)  输家{lc}笔({lr:.0f}%)  diff={diff:+.0f}%")

print()
print("─ 连跌天数分布 ─")
for cd in range(0, 4):
    wc = len([1 for t in wins if t.consecutive_yin == cd])
    lc = len([1 for t in losses if t.consecutive_yin == cd])
    wr = wc / len(wins) * 100 if wins else 0
    lr = lc / len(losses) * 100 if losses else 0
    diff = wr - lr
    print(f"  连跌{cd}天: 赢家{wc}笔({wr:.0f}%)  输家{lc}笔({lr:.0f}%)  diff={diff:+.0f}%")

print()
print(f"总赢家: {len(wins)}笔, 总输家: {len(losses)}笔")
print(f"胜率: {len(wins)/len(all_trades)*100:.1f}%")
