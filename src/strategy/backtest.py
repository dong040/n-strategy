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
    min_strength: int = 75
    max_wait_days: int = 5
    lookback_years: int = 3  # 兼容旧配置


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


def backtest_single_stock(
    code: str,
    name: str,
    ohlcv: pd.DataFrame,
    params: NPatternParams,
    config: BacktestConfig = None,
) -> BacktestResult:
    """单只股票 Walk-forward 回测

    每天用当日之前的数据扫描信号。
    买入：限价单 — 等待价格回落触及 fib 支撑位才成交。
    卖出：止损(成本价)/止盈(等幅目标)/30日强平。
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
    dates = ohlcv['date'].values

    limit_pct = get_limit_pct(code)
    cash = config.init_cash
    position = 0
    entry_price = 0.0
    entry_date = ""
    active_stop = 0.0
    active_target = 0.0
    active_strength = 0
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

            # 止损: 当日最低价触及成本价
            if active_stop > 0 and lows[i] <= active_stop:
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
                ))
                position = 0
                in_trade = False
                hold_days = 0

            equity.append(cash + position * closes[i])
            continue

        # === 限价单等待成交 ===
        if pending_entry is not None:
            pe = pending_entry
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

            # 当日最低价触及限价 → 成交
            if lows[i] <= limit_price:
                buy_price = limit_price  # 限价单成交，无滑点

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

        # 下限价单，等待价格回落触及支撑位
        pending_entry = {
            'price': best['entry_price'],
            'stop': best['stop_loss'],
            'target': best['target_price'],
            'strength': best['strength'],
            'waited': 0,
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
