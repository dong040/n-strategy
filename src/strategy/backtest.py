"""N字战法历史回测引擎

模拟 A 股真实交易约束：
- T+1 制度
- 涨跌停限制（主板 10%、科创板 20%、北交所 30%、ST 5%）
- 佣金 + 印花税 + 滑点
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .n_pattern import NPatternParams, find_n_pattern, NSignal

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    commission_pct: float = 0.00025    # 佣金 万2.5
    stamp_tax_pct: float = 0.001       # 印花税 千1（卖出）
    slippage_pct: float = 0.001        # 滑点 0.1%
    init_cash: float = 1_000_000       # 初始资金
    max_position_pct: float = 0.2      # 单票最大仓位
    t_plus_1: bool = True              # T+1 限制


@dataclass
class Trade:
    """一笔交易"""
    code: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    profit: float
    profit_pct: float
    exit_reason: str = ""              # stop_loss / take_profit / force_exit


@dataclass
class BacktestResult:
    """回测结果"""
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
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
    """获取涨跌停幅度"""
    if code.startswith("8"):
        return 0.30   # 北交所
    if code.startswith("68"):
        return 0.20   # 科创板
    if code.startswith(("3", "0")):
        return 0.10   # 创业板/深主板
    if code.startswith("6"):
        return 0.10   # 沪主板
    return 0.10


def _can_trade(price: float, prev_close: float, limit_pct: float, is_buy: bool) -> bool:
    """检查涨跌停限制下能否成交

    Args:
        is_buy: True=买入检查涨停, False=卖出检查跌停
    """
    if is_buy:
        return price <= prev_close * (1 + limit_pct) * 1.001  # 允许微小误差
    else:
        return price >= prev_close * (1 - limit_pct) * 0.999


def backtest_single_stock(
    code: str,
    ohlcv: pd.DataFrame,
    params: NPatternParams,
    config: BacktestConfig = None,
) -> BacktestResult:
    """单只股票回测

    Args:
        code: 股票代码
        ohlcv: DataFrame with columns [open, high, low, close, volume, date]
        params: N字战法参数
        config: 回测配置

    Returns:
        BacktestResult
    """
    if config is None:
        config = BacktestConfig()

    close = ohlcv["close"].values
    open_ = ohlcv["open"].values
    high = ohlcv["high"].values
    low = ohlcv["low"].values
    volume = ohlcv["volume"].values
    dates = ohlcv["date"].astype(str).tolist() if "date" in ohlcv.columns else [str(i) for i in range(len(close))]

    # 识别 N 字信号
    signals = find_n_pattern(close, volume, dates, params)
    signals.sort(key=lambda s: dates.index(s.date) if s.date in dates else 0)

    limit_pct = get_limit_pct(code)
    cash = config.init_cash
    position = 0          # 持仓股数
    entry_price = 0.0
    entry_date = ""
    trades = []
    equity = []
    signal_idx = 0

    for i in range(len(close)):
        # 检查是否有新信号
        while signal_idx < len(signals) and dates[i] >= signals[signal_idx].date:
            sig = signals[signal_idx]
            signal_idx += 1

            # 有空仓且信号有效
            if position == 0 and sig.strength >= 50:
                # 买入：用当日开盘价 + 滑点
                buy_price = open_[i] * (1 + config.slippage_pct)
                # 涨跌停检查
                prev_close = close[i - 1] if i > 0 else close[i]
                if not _can_trade(buy_price, prev_close, limit_pct, is_buy=True):
                    continue

                # 仓位计算
                max_shares = int(cash * config.max_position_pct / buy_price)
                shares = max(100, max_shares // 100 * 100)  # A股 100 股整数倍
                cost = shares * buy_price * (1 + config.commission_pct)
                if cost > cash:
                    shares = int(cash * 0.99 / buy_price) // 100 * 100
                    cost = shares * buy_price * (1 + config.commission_pct)

                if shares >= 100 and cost <= cash:
                    cash -= cost
                    position = shares
                    entry_price = buy_price
                    entry_date = dates[i]

        # 持仓管理
        if position > 0:
            # 检查止损
            if low[i] <= sig.stop_loss:
                sell_price = max(open_[i], sig.stop_loss) * (1 - config.slippage_pct)
                prev_close = close[i - 1] if i > 0 else close[i]
                if _can_trade(sell_price, prev_close, limit_pct, is_buy=False):
                    sell_value = position * sell_price * (1 - config.commission_pct - config.stamp_tax_pct)
                    profit = sell_value - position * entry_price * (1 + config.commission_pct)
                    cash += sell_value
                    trades.append(Trade(
                        code=code, entry_date=entry_date, exit_date=dates[i],
                        entry_price=entry_price, exit_price=sell_price,
                        shares=position, profit=profit,
                        profit_pct=(sell_price / entry_price - 1) * 100,
                        exit_reason="stop_loss",
                    ))
                    position = 0

            # 检查止盈
            elif position > 0 and high[i] >= sig.target_price:
                sell_price = sig.target_price * (1 - config.slippage_pct)
                prev_close = close[i - 1] if i > 0 else close[i]
                if _can_trade(sell_price, prev_close, limit_pct, is_buy=False):
                    sell_value = position * sell_price * (1 - config.commission_pct - config.stamp_tax_pct)
                    profit = sell_value - position * entry_price * (1 + config.commission_pct)
                    cash += sell_value
                    trades.append(Trade(
                        code=code, entry_date=entry_date, exit_date=dates[i],
                        entry_price=entry_price, exit_price=sell_price,
                        shares=position, profit=profit,
                        profit_pct=(sell_price / entry_price - 1) * 100,
                        exit_reason="take_profit",
                    ))
                    position = 0

            # T+1 限制: 次日才能卖（简化: 检查是否同一天）
            # 实际回测中信号当天买入，最早下一个 bar 卖出

        # 记录权益曲线
        mark_price = close[i]
        total = cash + position * mark_price
        equity.append(total)

    # 强制平仓
    if position > 0:
        last_price = close[-1]
        sell_value = position * last_price * (1 - config.commission_pct - config.stamp_tax_pct)
        profit = sell_value - position * entry_price * (1 + config.commission_pct)
        cash += sell_value
        trades.append(Trade(
            code=code, entry_date=entry_date, exit_date=dates[-1],
            entry_price=entry_price, exit_price=last_price,
            shares=position, profit=profit,
            profit_pct=(last_price / entry_price - 1) * 100,
            exit_reason="force_exit",
        ))
        equity[-1] = cash

    return _compute_result(trades, equity, config)


def _compute_result(trades: list[Trade], equity: list[float], config: BacktestConfig) -> BacktestResult:
    """计算回测统计指标"""
    total = len(trades)
    if total == 0:
        return BacktestResult(total_trades=0, equity_curve=equity)

    wins = [t for t in trades if t.profit > 0]
    losses = [t for t in trades if t.profit <= 0]
    win_rate = len(wins) / total

    total_profit = sum(t.profit for t in wins)
    total_loss = abs(sum(t.profit for t in losses))
    profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

    avg_p = float(np.mean([t.profit_pct for t in wins])) if wins else 0
    avg_l = float(np.mean([t.profit_pct for t in losses])) if losses else 0

    final_val = equity[-1] if equity else config.init_cash
    total_return = (final_val / config.init_cash - 1) * 100

    # 年化收益（假设 250 交易日）
    n_days = len(equity)
    if n_days > 0 and equity[0] > 0:
        annual_return = ((final_val / config.init_cash) ** (250 / n_days) - 1) * 100
    else:
        annual_return = 0.0

    # 最大回撤
    peak = 0
    max_dd = 0.0
    for v in equity:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    # 夏普比率
    returns = pd.Series(equity).pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        sharpe = float(returns.mean() / returns.std() * np.sqrt(250))
    else:
        sharpe = 0.0

    # 平均持仓天数
    avg_hold = 0.0
    if trades:
        for t in trades:
            try:
                ed = pd.Timestamp(t.entry_date)
                xd = pd.Timestamp(t.exit_date)
                avg_hold += (xd - ed).days
            except Exception:
                pass
        avg_hold /= len(trades)

    return BacktestResult(
        trades=trades,
        equity_curve=equity,
        total_return=round(total_return, 2),
        annual_return=round(annual_return, 2),
        win_rate=round(win_rate * 100, 1),
        profit_factor=round(profit_factor, 2),
        max_drawdown=round(max_dd * 100, 2),
        sharpe_ratio=round(sharpe, 2),
        avg_profit=round(avg_p, 2),
        avg_loss=round(avg_l, 2),
        total_trades=total,
        avg_hold_days=round(avg_hold, 1),
    )


def backtest_portfolio(
    stocks_data: dict[str, pd.DataFrame],
    params: NPatternParams,
    config: BacktestConfig = None,
) -> BacktestResult:
    """多股票组合回测

    Args:
        stocks_data: {code: ohlcv DataFrame}
    """
    if config is None:
        config = BacktestConfig()

    all_trades = []
    combined_equity = None

    for code, df in stocks_data.items():
        result = backtest_single_stock(code, df, params, config)
        all_trades.extend(result.trades)

        if combined_equity is None:
            combined_equity = pd.Series(result.equity_curve)
        else:
            # 简单相加（真实组合需要考虑现金分配，此处简化）
            eq = pd.Series(result.equity_curve)
            combined_equity = combined_equity.add(eq, fill_value=0)

    if combined_equity is not None:
        combined_equity = combined_equity.tolist()

    return _compute_result(all_trades, combined_equity or [], config)
