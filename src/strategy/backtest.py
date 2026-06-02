"""N字战法历史回测引擎

模拟 A 股真实交易约束：
- T+1 制度
- 涨跌停限制（主板 10%）
- 佣金 + 印花税 + 滑点
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .n_pattern import NPatternParams, find_n_signals, NSignal

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    commission_pct: float = 0.00025
    stamp_tax_pct: float = 0.001
    slippage_pct: float = 0.001
    init_cash: float = 1_000_000
    max_position_pct: float = 0.2
    max_positions: int = 3           # 最大同时持仓数
    min_strength: int = 75
    max_wait_days: int = 5
    lookback_years: int = 3
    close_stop: bool = True  # True=收盘价跌破止损才平, False=日内低点触及即平


@dataclass
class Trade:
    code: str
    name: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    profit: float
    profit_pct: float
    strength: int = 0
    exit_reason: str = ""              # stop_loss / take_profit / force_exit
    factor_score: int = 0
    pullback_volume_score: int = 0
    turnover_crowding_score: int = 0
    relative_strength_score: int = 0
    volatility_contraction_score: int = 0
    support_reclaim_score: int = 0
    close_position_score: int = 0
    limit_up_followthrough_score: int = 0
    theme_heat_score: int = 0
    amount_quality_score: int = 0
    market_regime_score: int = 0
    northbound_flow_score: int = 0
    rsi_divergence_score: int = 0
    macd_signal_score: int = 0
    ma_alignment_score: int = 0
    boll_squeeze_score: int = 0
    kdj_oversold_score: int = 0
    mfi_score: int = 0
    shadow_quality_score: int = 0
    pullback_speed_score: int = 0
    intraday_reversal_score: int = 0
    volume_climax_score: int = 0
    sector_relative_score: int = 0
    adx_trend_score: int = 0
    obv_accumulation_score: int = 0
    cmf_score: int = 0
    gap_support_score: int = 0
    rr_ratio: float = 0.0
    entry_to_resistance_pct: float = 0.0
    ml_confidence: float = 0.0
    ml_confidence_score: int = 0
    sequence_confidence: float = 0.0
    sequence_score: int = 0


@dataclass
class BacktestResult:
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    total_return: float = 0.0
    annual_return: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    total_trades: int = 0
    avg_hold_days: float = 0.0


def get_limit_pct(code: str) -> float:
    if code.startswith("8"):
        return 0.30
    if code.startswith("68"):
        return 0.20
    return 0.10


def _factor_kwargs(source: dict) -> dict:
    keys = [
        "factor_score",
        "pullback_volume_score",
        "turnover_crowding_score",
        "relative_strength_score",
        "volatility_contraction_score",
        "support_reclaim_score",
        "close_position_score",
        "limit_up_followthrough_score",
        "theme_heat_score",
        "amount_quality_score",
        "market_regime_score",
        "northbound_flow_score",
        "rsi_divergence_score",
        "macd_signal_score",
        "ma_alignment_score",
        "boll_squeeze_score",
        "kdj_oversold_score",
        "mfi_score",
        "shadow_quality_score",
        "pullback_speed_score",
        "intraday_reversal_score",
        "volume_climax_score",
        "sector_relative_score",
        "adx_trend_score",
        "obv_accumulation_score",
        "cmf_score",
        "gap_support_score",
        "rr_ratio",
        "entry_to_resistance_pct",
        "ml_confidence",
        "ml_confidence_score",
        "sequence_confidence",
        "sequence_score",
    ]
    return {k: source.get(k, 0) for k in keys}


def _update_trailing_stop(
    entry_price: float,
    current_close: float,
    highs: np.ndarray,
    idx: int,
    hold_days: int,
    active_stop: float,
) -> float:
    """Apply the strategy's trailing-stop rules."""
    profit_from_entry = (current_close - entry_price) / entry_price
    if profit_from_entry >= 0.08:
        lookback = min(hold_days, 20)
        recent_high = max(highs[idx - lookback:idx + 1])
        return max(active_stop, recent_high * 0.95)
    if profit_from_entry >= 0.03:
        return max(active_stop, entry_price * 1.001)
    return active_stop


def _entry_confirmation_ok(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    vols: np.ndarray,
    idx: int,
    limit_price: float,
) -> bool:
    """Shared support-confirmation check for single-stock and portfolio backtests."""
    if closes[idx] < limit_price:
        return False

    if idx > 0:
        body_bottom = min(opens[idx], closes[idx])
        shadow_ratio = (body_bottom - lows[idx]) / closes[idx] if closes[idx] > 0 else 0
        vol_expanding = vols[idx] > vols[idx - 1] * 1.5
        no_lower_shadow = shadow_ratio < 0.005
        if vol_expanding and no_lower_shadow:
            return False

    if idx >= 20:
        avg_vol_20 = float(np.mean(vols[idx - 20:idx]))
        if vols[idx] > avg_vol_20 * 2.0:
            return False

    day_range = highs[idx] - lows[idx]
    if day_range > 0:
        close_position = (closes[idx] - lows[idx]) / day_range
        if close_position < 0.40:
            return False

    return True


def _calc_active_stop(
    buy_price: float,
    signal_price: float,
    signal_stop: float,
    highs: np.ndarray,
    lows: np.ndarray,
    idx: int,
) -> float:
    """Use the wider of signal stop and ATR stop."""
    signal_stop_pct = (signal_price - signal_stop) / signal_price
    atr20 = float(np.mean(highs[idx - 20:idx] - lows[idx - 20:idx])) if idx >= 20 else 0
    atr_stop_pct = max(signal_stop_pct, atr20 / buy_price * 1.0) if atr20 > 0 else signal_stop_pct
    return round(buy_price * (1 - atr_stop_pct), 2)


def backtest_single_stock(
    code: str,
    name: str,
    ohlcv: pd.DataFrame,
    params: NPatternParams,
    config: BacktestConfig = None,
    market_regime: set = None,
) -> BacktestResult:
    """单只股票 Walk-forward 回测

    每天用当日之前的数据扫描信号。
    买入：限价单 — 等待价格回落触及 fib 支撑位才成交。
    卖出：止损(成本价)/止盈(等幅目标)/30日强平。

    market_regime: set of date strings where market is favorable (e.g. 上证 > MA60)
                   如果提供，仅在 favorable dates 允许新入场。
    """
    if config is None:
        config = BacktestConfig()

    if len(ohlcv) < 120:
        return BacktestResult(total_trades=0)

    ohlcv = ohlcv.sort_values('date').reset_index(drop=True)
    opens = ohlcv['open'].values
    highs = ohlcv['high'].values
    lows = ohlcv['low'].values
    closes = ohlcv['close'].values
    vols = ohlcv['volume'].values
    amounts_arr = ohlcv['amount'].values if 'amount' in ohlcv.columns else None
    dates = ohlcv['date'].values

    limit_pct = get_limit_pct(code)
    cash = config.init_cash
    position = 0
    entry_price = 0.0
    entry_date = ""
    active_stop = 0.0
    active_target = 0.0
    active_strength = 0
    active_factors = {}
    trades = []
    equity = []
    in_trade = False
    hold_days = 0

    pending_entry = None  # 限价单: {'price','stop','target','strength','waited'}
    max_wait = config.max_wait_days
    min_window = 120

    for i in range(min_window, len(closes)):
        # === 持仓管理 ===
        if in_trade:
            hold_days += 1

            # 移动止损: 盈利>3%保本, >8%追踪最高价-5%
            active_stop = _update_trailing_stop(
                entry_price, closes[i], highs, i, hold_days, active_stop,
            )

            # 止损: 收盘价跌破（close_stop=True）或当日最低价触及（close_stop=False）
            stop_triggered = closes[i] <= active_stop if config.close_stop else lows[i] <= active_stop
            if active_stop > 0 and stop_triggered:
                exit_price = active_stop * (1 - config.slippage_pct)
                sell_value = position * exit_price * (1 - config.commission_pct - config.stamp_tax_pct)
                profit = sell_value - position * entry_price * (1 + config.commission_pct)
                cash += sell_value
                trades.append(Trade(
                    code=code, name=name,
                    entry_date=entry_date, exit_date=str(dates[i])[:10],
                    entry_price=entry_price, exit_price=exit_price,
                    shares=position, profit=profit,
                    profit_pct=(exit_price / entry_price - 1) * 100,
                    strength=active_strength, exit_reason="stop_loss",
                    **_factor_kwargs(active_factors),
                ))
                position = 0
                in_trade = False
                hold_days = 0
                active_stop = 0

            # 止盈: 当日最高价触及目标
            elif active_target > 0 and highs[i] >= active_target:
                exit_price = active_target * (1 - config.slippage_pct)
                sell_value = position * exit_price * (1 - config.commission_pct - config.stamp_tax_pct)
                profit = sell_value - position * entry_price * (1 + config.commission_pct)
                cash += sell_value
                trades.append(Trade(
                    code=code, name=name,
                    entry_date=entry_date, exit_date=str(dates[i])[:10],
                    entry_price=entry_price, exit_price=exit_price,
                    shares=position, profit=profit,
                    profit_pct=(exit_price / entry_price - 1) * 100,
                    strength=active_strength, exit_reason="take_profit",
                    **_factor_kwargs(active_factors),
                ))
                position = 0
                in_trade = False
                hold_days = 0
                active_target = 0

            # 强制平仓: 30 天
            elif hold_days >= 30:
                exit_price = closes[i] * (1 - config.slippage_pct)
                sell_value = position * exit_price * (1 - config.commission_pct - config.stamp_tax_pct)
                profit = sell_value - position * entry_price * (1 + config.commission_pct)
                cash += sell_value
                trades.append(Trade(
                    code=code, name=name,
                    entry_date=entry_date, exit_date=str(dates[i])[:10],
                    entry_price=entry_price, exit_price=exit_price,
                    shares=position, profit=profit,
                    profit_pct=(exit_price / entry_price - 1) * 100,
                    strength=active_strength, exit_reason="force_exit",
                    **_factor_kwargs(active_factors),
                ))
                position = 0
                in_trade = False
                hold_days = 0

            equity.append(cash + position * closes[i])
            continue

        # === 限价单等待成交 ===
        if pending_entry is not None:
            pe = pending_entry

            # ── 昨日已确认支撑 → 今天以昨日收盘价买入（无 lookahead）──
            if pe.get('confirmed'):
                buy_price = pe['confirm_close']  # 确认日收盘价 = 你可执行的价格
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
                    entry_date = str(dates[i - 1])[:10]  # 确认日即为入场日
                    active_stop = _calc_active_stop(
                        buy_price, pe['price'], pe['stop'], highs, lows, i,
                    )
                    active_target = pe['target']
                    active_strength = pe.get('strength', 0)
                    active_factors = _factor_kwargs(pe)
                    in_trade = True
                    hold_days = 1  # 已经过了一天（确认日→买入日）
                pending_entry = None
                equity.append(cash + position * closes[i])
                continue

            limit_price = pe['price']

            # 涨跌停检查（用前一日的收盘价）
            prev_close = closes[i - 1]
            if limit_price > prev_close * (1 + limit_pct) * 1.001:
                pending_entry = None
                equity.append(cash)
                continue
            if limit_price < prev_close * (1 - limit_pct) * 0.999:
                pending_entry = None
                equity.append(cash)
                continue

            # 当日最低价触及限价 → 需要确认支撑有效
            if lows[i] <= limit_price:
                if not _entry_confirmation_ok(opens, highs, lows, closes, vols, i, limit_price):
                    pending_entry = None
                    equity.append(cash)
                    continue

                # 当天收盘确认支撑后，存入 pending，等下一个 bar 买入
                pe['confirmed'] = True
                pe['confirm_close'] = closes[i]
                equity.append(cash)
                continue

                # 仓位
                max_shares = int(cash * config.max_position_pct / buy_price)
                shares = max(100, max_shares // 100 * 100)
                cost = shares * buy_price * (1 + config.commission_pct)
                if cost > cash:
                    shares = int(cash * 0.99 / buy_price) // 100 * 100
                    cost = shares * buy_price * (1 + config.commission_pct)

                if shares < 100:
                    pending_entry = None
                    equity.append(cash)
                    continue

                cash -= cost
                position = shares
                entry_price = buy_price
                entry_date = str(dates[i])[:10]
                active_stop = pe['stop']
                active_target = pe['target']
                active_strength = pe['strength']
                active_factors = _factor_kwargs(pe)
                in_trade = True
                hold_days = 0
                pending_entry = None

                equity.append(cash + position * closes[i])
                continue

            # 未成交，累计等待天数
            pe['waited'] += 1
            if pe['waited'] > max_wait:
                pending_entry = None  # 超时取消
            equity.append(cash)
            continue

        # === 信号检测: 每 5 个 bar 扫描一次 ===
        if i % 5 != 0:
            equity.append(cash)
            continue

        window_slice = slice(max(0, i - 500), i)
        try:
            sigs = find_n_signals(
                opens[window_slice], highs[window_slice], lows[window_slice],
                closes[window_slice], vols[window_slice], params,
                amounts=(amounts_arr[window_slice] if amounts_arr is not None else None),
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

        # 市场环境过滤：大盘不在 MA60 上方 → 不建新仓
        if market_regime is not None and str(dates[i])[:10] not in market_regime:
            equity.append(cash)
            continue

        # 下限价单，等待价格回落触及支撑位
        pending_entry = {
            'price': best['entry_price'],
            'stop': best['stop_loss'],
            'target': best['target_price'],
            'strength': best['strength'],
            'waited': 0,
            **_factor_kwargs(best),
        }
        equity.append(cash)

    # === 收盘强制平仓 ===
    if in_trade and position > 0:
        last_close = closes[-1]
        exit_price = last_close * (1 - config.slippage_pct)
        sell_value = position * exit_price * (1 - config.commission_pct - config.stamp_tax_pct)
        profit = sell_value - position * entry_price * (1 + config.commission_pct)
        cash += sell_value
        trades.append(Trade(
            code=code, name=name,
            entry_date=entry_date, exit_date=str(dates[-1])[:10],
            entry_price=entry_price, exit_price=exit_price,
            shares=position, profit=profit,
            profit_pct=(exit_price / entry_price - 1) * 100,
            strength=active_strength, exit_reason="force_exit",
            **_factor_kwargs(active_factors),
        ))
        equity[-1] = cash

    return _compute_result(trades, equity, config)


def _compute_result(trades: list, equity: list, config: BacktestConfig) -> BacktestResult:
    total = len(trades)
    if total == 0:
        return BacktestResult(total_trades=0, equity_curve=equity)

    wins = [t for t in trades if t.profit > 0]
    losses = [t for t in trades if t.profit <= 0]
    win_rate = len(wins) / total * 100

    total_profit = sum(t.profit for t in wins)
    total_loss = abs(sum(t.profit for t in losses))
    profit_factor = total_profit / total_loss if total_loss > 0 else 999

    total_return = (equity[-1] / config.init_cash - 1) * 100
    trading_days = len(equity)
    annual_return = ((1 + total_return / 100) ** (250 / trading_days) - 1) * 100 if trading_days > 0 else 0

    peak = config.init_cash
    max_dd = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100
        if dd > max_dd:
            max_dd = dd

    returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([0])
    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(250)) if np.std(returns) > 0 else 0

    avg_profit = float(np.mean([t.profit_pct for t in wins])) if wins else 0
    avg_loss = float(np.mean([t.profit_pct for t in losses])) if losses else 0
    avg_hold = float(np.mean([
        (pd.to_datetime(t.exit_date) - pd.to_datetime(t.entry_date)).days
        for t in trades
    ]))

    return BacktestResult(
        trades=trades,
        equity_curve=equity,
        total_trades=total,
        total_return=round(total_return, 2),
        annual_return=round(annual_return, 2),
        win_rate=round(win_rate, 1),
        profit_factor=round(profit_factor, 2),
        max_drawdown=round(max_dd, 2),
        sharpe_ratio=round(sharpe, 2),
        avg_profit=round(avg_profit, 1),
        avg_loss=round(avg_loss, 1),
        avg_hold_days=round(avg_hold, 1),
    )


def backtest_portfolio(
    stocks: list[tuple],        # [(code, name, ohlcv), ...]
    params: NPatternParams,
    config: BacktestConfig = None,
) -> BacktestResult:
    """组合回测 — 最大同时持仓 config.max_positions 只。

    按日期推进，每天生成所有股票的信号，取强度最高的入场，
    同时管理所有持仓的止损/止盈/强平。
    """
    if config is None:
        config = BacktestConfig()

    if len(stocks) < 1:
        return BacktestResult(total_trades=0)

    import heapq

    # 对齐所有股票到最早/最晚日期
    all_dates = set()
    stock_data = {}
    for code, name, df in stocks:
        if len(df) < 120:
            continue
        df = df.sort_values('date').reset_index(drop=True)
        stock_data[code] = {
            'name': name,
            'opens': df['open'].values,
            'highs': df['high'].values,
            'lows': df['low'].values,
            'closes': df['close'].values,
            'vols': df['volume'].values,
            'amounts': df['amount'].values if 'amount' in df.columns else None,
            'dates': df['date'].values,
        }
        for d in df['date'].values:
            all_dates.add(str(d)[:10])

    sorted_dates = sorted(all_dates)
    if len(sorted_dates) < 120:
        return BacktestResult(total_trades=0)

    cash = config.init_cash
    positions = {}     # code -> {entry_price, shares, stop, target, entry_date, strength, factors, hold_days}
    trades = []
    equity = []
    pending_entry = None  # {code, price, stop, target, strength, factors, waited}
    max_wait = config.max_wait_days

    min_window = max(120, params.retrace_days_max + 60)

    for di in range(min_window, len(sorted_dates)):
        date_str = sorted_dates[di]

        # === 持仓管理 ===
        closed = []
        for code, pos in positions.items():
            sd = stock_data.get(code)
            if sd is None:
                closed.append(code)
                continue

            # 找到该日期在股票数组中的索引
            date_matches = [j for j, d in enumerate(sd['dates']) if str(d)[:10] == date_str]
            if not date_matches:
                closed.append(code)
                continue
            idx = date_matches[0]
            close_px = sd['closes'][idx]
            high_px = sd['highs'][idx]
            low_px = sd['lows'][idx]

            pos['hold_days'] += 1
            pos['stop'] = _update_trailing_stop(
                pos['entry_price'], close_px, sd['highs'], idx, pos['hold_days'], pos['stop'],
            )
            exit_reason = None
            exit_price = 0

            # 止损
            if config.close_stop:
                if close_px <= pos['stop']:
                    exit_reason = 'stop_loss'
                    exit_price = pos['stop'] * (1 - config.slippage_pct)
            else:
                if low_px <= pos['stop']:
                    exit_reason = 'stop_loss'
                    exit_price = pos['stop'] * (1 - config.slippage_pct)

            # 止盈
            if exit_reason is None and pos['target'] > 0 and high_px >= pos['target']:
                exit_reason = 'take_profit'
                exit_price = pos['target'] * (1 - config.slippage_pct)

            # 强平
            if exit_reason is None and pos['hold_days'] >= 30:
                exit_reason = 'force_exit'
                exit_price = close_px * (1 - config.slippage_pct)

            if exit_reason:
                sell_value = pos['shares'] * exit_price * (1 - config.commission_pct - config.stamp_tax_pct)
                profit = sell_value - pos['shares'] * pos['entry_price'] * (1 + config.commission_pct)
                cash += sell_value
                trades.append(Trade(
                    code=code, name=sd['name'],
                    entry_date=pos['entry_date'], exit_date=date_str,
                    entry_price=pos['entry_price'], exit_price=exit_price,
                    shares=pos['shares'], profit=profit,
                    profit_pct=(exit_price / pos['entry_price'] - 1) * 100,
                    strength=pos['strength'], exit_reason=exit_reason,
                    **_factor_kwargs(pos['factors']),
                ))
                closed.append(code)

        for code in closed:
            del positions[code]

        # === 限价单成交检查 ===
        if pending_entry:
            pe_code = pending_entry['code']
            sd = stock_data.get(pe_code)
            if sd:
                date_matches = [j for j, d in enumerate(sd['dates']) if str(d)[:10] == date_str]
                if date_matches:
                    idx = date_matches[0]
                    close_px = sd['closes'][idx]
                    low_px = sd['lows'][idx]
                    high_px = sd['highs'][idx]
                    opens_i = sd['opens'][idx]
                    limit_price = pending_entry['price']

                    # 涨跌停检查
                    prev_close = sd['closes'][idx - 1] if idx > 0 else close_px
                    limit_pct = get_limit_pct(pe_code)
                    if limit_price > prev_close * (1 + limit_pct) * 1.001:
                        pending_entry = None
                    elif limit_price < prev_close * (1 - limit_pct) * 0.999:
                        pending_entry = None
                    elif low_px <= limit_price:
                        if not _entry_confirmation_ok(
                            sd['opens'], sd['highs'], sd['lows'], sd['closes'], sd['vols'],
                            idx, limit_price,
                        ):
                            pending_entry = None
                        else:
                            buy_price = close_px
                            per_position = config.max_position_pct
                            max_shares = int(cash * per_position / buy_price)
                            shares = max(100, max_shares // 100 * 100)
                            cost = shares * buy_price * (1 + config.commission_pct)
                            if cost > cash:
                                shares = int(cash * 0.99 / buy_price) // 100 * 100
                                cost = shares * buy_price * (1 + config.commission_pct)
                            if shares >= 100:
                                cash -= cost
                                positions[pe_code] = {
                                    'entry_price': buy_price,
                                    'shares': shares,
                                    'stop': _calc_active_stop(
                                        buy_price, pending_entry['price'], pending_entry['stop'],
                                        sd['highs'], sd['lows'], idx,
                                    ),
                                    'target': pending_entry['target'],
                                    'entry_date': date_str,
                                    'strength': pending_entry['strength'],
                                    'factors': {k: pending_entry[k] for k in _factor_kwargs({})},
                                    'hold_days': 0,
                                }
                            pending_entry = None
                    else:
                        pending_entry['waited'] += 1
                        if pending_entry['waited'] > max_wait:
                            pending_entry = None

        # === 扫描新信号 ===
        if len(positions) < config.max_positions and pending_entry is None:
            candidates = []
            for code, sd in stock_data.items():
                if code in positions:
                    continue
                date_matches = [j for j, d in enumerate(sd['dates']) if str(d)[:10] == date_str]
                if not date_matches:
                    continue
                idx = date_matches[0]
                if idx < min_window:
                    continue

                # 只用到今日为止的数据
                seg_opens = sd['opens'][:idx + 1]
                seg_highs = sd['highs'][:idx + 1]
                seg_lows = sd['lows'][:idx + 1]
                seg_closes = sd['closes'][:idx + 1]
                seg_vols = sd['vols'][:idx + 1]
                seg_amounts = sd['amounts'][:idx + 1] if sd['amounts'] is not None else None

                try:
                    raw_signals = find_n_signals(
                        seg_opens, seg_highs, seg_lows, seg_closes, seg_vols,
                        params, market_pct=0, amounts=seg_amounts,
                    )
                except Exception:
                    continue

                for s in raw_signals:
                    if s.get('strength', 0) >= config.min_strength:
                        s['_code'] = code
                        candidates.append(s)

            if candidates:
                candidates.sort(key=lambda s: s['strength'], reverse=True)
                best = candidates[0]
                best_code = best.pop('_code')
                pending_entry = {
                    'code': best_code,
                    'price': best['entry_price'],
                    'stop': best['stop_loss'],
                    'target': best['target_price'],
                    'strength': best['strength'],
                    'waited': 0,
                    **_factor_kwargs(best),
                }

        # === 权益 ===
        total_value = cash
        for code, pos in positions.items():
            sd = stock_data.get(code)
            if sd:
                date_matches = [j for j, d in enumerate(sd['dates']) if str(d)[:10] == date_str]
                if date_matches:
                    total_value += pos['shares'] * sd['closes'][date_matches[0]]
        equity.append(total_value)

    # 收盘强平所有持仓
    last_date = sorted_dates[-1]
    for code, pos in list(positions.items()):
        sd = stock_data.get(code)
        if sd is None:
            continue
        date_matches = [j for j, d in enumerate(sd['dates']) if str(d)[:10] == last_date]
        if not date_matches:
            continue
        idx = date_matches[0]
        close_px = sd['closes'][idx]
        exit_price = close_px * (1 - config.slippage_pct)
        sell_value = pos['shares'] * exit_price * (1 - config.commission_pct - config.stamp_tax_pct)
        profit = sell_value - pos['shares'] * pos['entry_price'] * (1 + config.commission_pct)
        cash += sell_value
        trades.append(Trade(
            code=code, name=sd['name'],
            entry_date=pos['entry_date'], exit_date=last_date,
            entry_price=pos['entry_price'], exit_price=exit_price,
            shares=pos['shares'], profit=profit,
            profit_pct=(exit_price / pos['entry_price'] - 1) * 100,
            strength=pos['strength'], exit_reason='force_exit',
            **_factor_kwargs(pos['factors']),
        ))

    return _compute_result(trades, equity, config)
