"""出场优化：ATR自适应止损 + 移动止损 + 时间止盈

在 backtest_single_stock 中替换固定2%止损为：
1. ATR止损：max(2%, 1.0 * ATR20)
2. 移动止损：盈利>5%后，止损上移至成本价
3. 分批止盈：目标一半仓位止盈，剩下一半移动止损
"""

import numpy as np


def calc_atr_stop(entry_price: float, highs: np.ndarray, lows: np.ndarray,
                  closes: np.ndarray, atr_mult: float = 1.0,
                  min_stop_pct: float = 0.02) -> float:
    """计算ATR自适应止损价"""
    n = len(closes)
    if n < 20:
        return entry_price * (1 - min_stop_pct)

    tr = np.maximum(
        highs[-20:] - lows[-20:],
        np.maximum(
            np.abs(highs[-20:] - np.concatenate([[closes[-21]], closes[-20:-1]])),
            np.abs(lows[-20:] - np.concatenate([[closes[-21]], closes[-20:-1]]))
        )
    )
    atr20 = float(np.mean(tr))
    atr_pct = atr20 / entry_price
    stop_pct = max(min_stop_pct, atr_pct * atr_mult)
    return round(entry_price * (1 - stop_pct), 2)


def calc_trailing_stop(entry_price: float, current_price: float,
                       stop_loss: float, high_since_entry: float,
                       profit_pct: float) -> float:
    """移动止损逻辑：
    - 盈利<3%: 原始止损不变
    - 盈利3-8%: 止损上移至成本价
    - 盈利>8%: 止损上移至最高价回撤5%
    """
    if profit_pct >= 0.08:
        # 盈利>8%: 移动止损到最高价下方5%
        trail_stop = high_since_entry * 0.95
        return max(stop_loss, trail_stop)
    elif profit_pct >= 0.03:
        # 盈利3-8%: 止损上移至成本价(保本)
        return max(stop_loss, entry_price * 1.001)
    return stop_loss


def calc_time_stop(entry_date: str, current_date: str, max_hold_days: int = 10) -> bool:
    """时间止损：超过最大持仓天数强制退出"""
    from datetime import datetime, timedelta
    try:
        entry_dt = datetime.strptime(entry_date[:10], '%Y-%m-%d')
        current_dt = datetime.strptime(current_date[:10], '%Y-%m-%d')
        days = (current_dt - entry_dt).days
        return days >= max_hold_days
    except Exception:
        return False


def calc_ma_exit_signal(closes: np.ndarray, ma_period: int = 5,
                        consecutive_days: int = 2) -> bool:
    """MA出场信号：连续N日收盘在MA下方则出"""
    n = len(closes)
    if n < ma_period + consecutive_days:
        return False
    ma = np.mean(closes[-ma_period:])
    for i in range(consecutive_days):
        if closes[-1 - i] >= ma:
            return False
    return True


def optimal_exit_check(entry_price: float, current_price: float,
                       highs_since: np.ndarray, lows_since: np.ndarray,
                       closes_since: np.ndarray,
                       stop_loss: float, target_price: float,
                       high_since_entry: float,
                       days_held: int, max_hold_days: int = 15) -> dict:
    """综合出场检查，返回 {'action': 'hold'|'exit', 'reason': str, 'new_stop': float}"""
    profit_pct = (current_price - entry_price) / entry_price

    # 1. 止损检查（含移动止损）
    current_stop = calc_trailing_stop(entry_price, current_price, stop_loss,
                                      high_since_entry, profit_pct)
    if current_price <= current_stop:
        return {'action': 'exit', 'reason': 'stop_loss', 'new_stop': current_stop}

    # 2. 止盈检查
    if current_price >= target_price:
        return {'action': 'exit', 'reason': 'take_profit', 'new_stop': current_stop}

    # 3. 时间止损
    if days_held >= max_hold_days:
        return {'action': 'exit', 'reason': 'force_exit', 'new_stop': current_stop}

    # 4. MA出场（仅在盈利时启用）
    if profit_pct > 0.03 and calc_ma_exit_signal(closes_since, ma_period=5):
        return {'action': 'exit', 'reason': 'ma_exit', 'new_stop': current_stop}

    return {'action': 'hold', 'reason': '', 'new_stop': current_stop}
