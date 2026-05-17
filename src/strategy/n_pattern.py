"""N字战法形态识别引擎 v2

核心逻辑（大神策略）:
  1. 第一波放量拉升后，等回调到费波位附近
  2. 放量下杀 → 缩量企稳 → 长下影确认 → 买入
  3. 止损放在费波位下方

N字结构定义：
    高点1 (第一波顶点)
      ↗     ↘
    起点    (回调低点 = 买点区域)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class NPatternParams:
    """N字战法参数"""
    # 形态参数
    min_rise_1st: float = 0.05
    max_rise_1st: float = 0.60
    retrace_min: float = 0.25
    retrace_max: float = 0.65
    retrace_days_min: int = 1
    retrace_days_max: int = 20
    vol_ratio_breakout: float = 1.5

    # 企稳确认
    require_stabilization: bool = True
    stabilization_vol_shrink: float = 0.7
    require_lower_shadow: bool = True
    lower_shadow_ratio: float = 0.03

    # 均线确认
    require_ma_confluence: bool = False
    ma_periods: list = field(default_factory=lambda: [9, 10])
    ma_fib_confluence_pct: float = 0.01

    # 交易参数
    stop_loss_mode: str = "fib050"
    stop_loss_pct: float = 0.02
    take_profit_mode: str = "equal"
    take_profit_pct: float = 0.10

    # 辅助条件
    max_mcap_yi: float = 500
    min_mcap_yi: float = 20
    min_turnover_wan: float = 5000


@dataclass
class NSignal:
    """N字战法买入信号"""
    code: str = ""
    name: str = ""
    date: str = ""
    entry_price: float = 0
    stop_loss: float = 0
    target_price: float = 0
    strength: int = 0
    fib_level: float = 0
    fib_price: float = 0
    first_rise_pct: float = 0
    retrace_pct: float = 0
    retrace_days: int = 0
    first_peak: float = 0
    retrace_low: float = 0
    # v2 fields
    stab_ok: bool = False
    has_vol_shrink: bool = False
    has_shadow: bool = False
    ma_bullish: bool = False
    ma9_gt_ma10: bool = False
    ma_consistent: bool = False
    bullish_days: int = 0
    has_limit_up: bool = False
    ma_fib_ok: bool = False
    ma10_broken_intraday: bool = False
    nearest_ma: tuple = None
    ma9: float = 0
    ma10: float = 0
    # 基本面
    pe: float = 0
    pb: float = 0
    net_profit_yi: float = 0
    fundamental_score: int = 0
    details: dict = field(default_factory=dict)


def _fib_to_level(retrace: float) -> float:
    """Map retrace ratio to nearest Fibonacci level."""
    if retrace < 0.30:
        return 0.236
    elif retrace < 0.43:
        return 0.382
    elif retrace < 0.56:
        return 0.5
    else:
        return 0.618


def find_extrema(highs: np.ndarray, lows: np.ndarray, min_pct: float = 0.04):
    """找局部极值点，带显著性过滤

    使用 high/low 分别检测，然后按时序合并，过滤掉幅度 < min_pct 的噪声转折。
    """
    n = len(highs)
    raw_peaks, raw_troughs = [], []

    for i in range(1, n - 1):
        if highs[i] >= highs[i - 1] and highs[i] >= highs[i + 1]:
            raw_peaks.append(i)
        if lows[i] <= lows[i - 1] and lows[i] <= lows[i + 1]:
            raw_troughs.append(i)

    if n >= 2:
        if highs[0] > highs[1]:
            raw_peaks.insert(0, 0)
        if highs[-1] > highs[-2]:
            raw_peaks.append(n - 1)
        if lows[0] < lows[1]:
            raw_troughs.insert(0, 0)
        if lows[-1] < lows[-2]:
            raw_troughs.append(n - 1)

    # 显著性过滤
    peaks, troughs = [], []
    last_pv, last_tv = None, None
    all_ev = sorted(
        [(i, 'P', highs[i]) for i in raw_peaks] +
        [(i, 'T', lows[i]) for i in raw_troughs],
        key=lambda x: x[0],
    )

    for idx, et, val in all_ev:
        if et == 'P':
            if last_tv is not None and (val - last_tv) / last_tv >= min_pct:
                if last_pv is None or val > last_pv:
                    peaks.append(idx)
                    last_pv = val
        else:
            if last_pv is not None and (last_pv - val) / last_pv >= min_pct:
                if last_tv is None or val < last_tv:
                    troughs.append(idx)
                    last_tv = val

    if len(peaks) < 2 or len(troughs) < 2:
        return raw_peaks, raw_troughs

    return peaks, troughs


def _check_stabilization(opens, highs, lows, closes, vols, peak_idx, trough_idx, params):
    """检查企稳确认：缩量 + 长下影/日内反转

    两种企稳模式：
    1. 标准长下影：body_bottom - low >= close * 3%（传统锤子线）
    2. 日内反转：开盘高开后大幅杀跌再拉回（open→low 跌幅显著，close 从 low 拉回）
    """
    retrace_vols = vols[peak_idx:trough_idx + 1]
    has_vol_shrink = False
    if len(retrace_vols) >= 2 and vols[trough_idx] < np.max(retrace_vols) * params.stabilization_vol_shrink:
        has_vol_shrink = True

    o = opens[trough_idx]
    c = closes[trough_idx]
    l = lows[trough_idx]
    h = highs[trough_idx]

    # 标准长下影
    body_bottom = min(o, c)
    shadow = (body_bottom - l) / c if c > 0 else 0
    has_shadow = shadow >= params.lower_shadow_ratio

    # 日内反转：开盘→最低跌幅大 + 收盘从最低拉回
    intraday_drop = (o - l) / o if o > 0 else 0  # 开盘到最低的跌幅
    recovery_from_low = (c - l) / l if l > 0 else 0  # 从最低拉回的幅度
    has_intraday_reversal = (intraday_drop >= 0.03 and recovery_from_low >= 0.005)

    # 企稳确认：缩量 + (长下影 或 日内反转)
    stab_ok = has_vol_shrink and (has_shadow or has_intraday_reversal)
    return stab_ok, has_vol_shrink, has_shadow or has_intraday_reversal



def _check_ma_bullish(closes, min_days=3, lookback=5):
    """检查 MA9 > MA10 > MA20 是否在近N日中至少持续M日

    大神要求：多头排列不能只是当日，需要在最近3-5个交易日内持续。
    """
    n = len(closes)
    if n < 20 + lookback:
        return False, False, 0, False, None, None, None

    ma9 = float(np.mean(closes[-9:]))
    ma10 = float(np.mean(closes[-10:]))
    ma20 = float(np.mean(closes[-20:]))

    bullish_days = 0
    for offset in range(lookback):
        end = n - offset
        if end < 20:
            continue
        m9 = float(np.mean(closes[end - 9:end]))
        m10 = float(np.mean(closes[end - 10:end]))
        m20 = float(np.mean(closes[end - 20:end]))
        if m9 > m10 > m20:
            bullish_days += 1

    ma_bullish_now = ma9 > ma10 > ma20
    ma_consistent = bullish_days >= min_days
    return ma_bullish_now, ma9 > ma10, bullish_days, ma_consistent, ma9, ma10, ma20


def _check_limit_up(closes, lookback=30):
    """检查近N日是否有涨停（>=9.5%近似涨停）"""
    n = len(closes)
    start = max(0, n - lookback)
    for i in range(start, n):
        if i > 0 and closes[i - 1] > 0:
            chg = (closes[i] - closes[i - 1]) / closes[i - 1]
            if chg >= 0.095:
                return True
    return False


def _calc_strength(signal_data: dict) -> int:
    """计算信号强度 0-100+"""
    strength = 40
    if signal_data.get('stab_ok'):
        strength += 20
    elif signal_data.get('has_vol_shrink') or signal_data.get('has_shadow'):
        strength += 8
    # 多头排列：需要持续3日以上才给满分
    if signal_data.get('ma_consistent'):
        strength += 15  # 持续多头排列 = 强趋势
    elif signal_data.get('ma9_gt_ma10'):
        strength += 5   # 仅当日排列 = 弱趋势
    if signal_data.get('ma_bullish'):
        strength += 5
    if signal_data.get('has_vol_shrink'):
        strength += 8
    if signal_data.get('has_shadow'):
        strength += 8
    if signal_data.get('ma_fib_ok'):
        strength += 10
    if signal_data.get('fib_level', 1) <= 0.382 and signal_data.get('first_rise_pct', 0) > 0.30:
        strength += 5
    if signal_data.get('has_limit_up'):
        strength += 10

    # 罚分：跌破 MA10 再拉回 → 支撑已被测试
    if signal_data.get('ma10_broken_intraday'):
        strength -= 15

    return max(0, strength)


def find_n_signals(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    vols: np.ndarray,
    params: NPatternParams,
) -> list[dict]:
    """扫描K线中的 N 字买点

    核心原则（大神策略）：
    1. 上涨 → 回调找支撑 → 支撑住 → 再拉一波
    2. 支撑位必须"撑住"，跌破再拉回的不算
    3. 费波位 + MA9/MA10 共振才是有效支撑
    """
    n = len(closes)
    peaks, troughs = find_extrema(highs, lows)

    has_limit_up = _check_limit_up(closes, 30)
    ma_bullish, ma9_gt_ma10, bullish_days, ma_consistent, ma9, ma10, ma20 = _check_ma_bullish(closes)

    last_close = closes[-1]

    # === 硬过滤1：收盘价跌破 MA10 → 支撑已失效，整只跳过 ===
    if ma10 is not None and last_close < ma10:
        return []

    # === 硬过滤2：多头排列不持续 → 至少3/5日 MA9>MA10>MA20 ===
    if not ma_consistent:
        return []

    # === 检测近期是否跌破过 MA10（日内跌破也算） ===
    recent_lows = lows[-3:] if n >= 3 else lows
    ma10_broken_intraday = (ma10 is not None and any(low < ma10 for low in recent_lows))

    signals = []

    for ti, ta in enumerate(troughs):
        for tb in troughs[ti + 1:]:
            if tb < n - 3:
                continue  # 第二低谷必须在近3日

            peaks_between = [p for p in peaks if ta < p < tb]
            if not peaks_between:
                continue
            best_p = max(peaks_between, key=lambda p: highs[p])

            first_low = lows[ta]
            first_high = highs[best_p]
            retrace_low = lows[tb]

            # 第一波涨幅
            first_rise = (first_high - first_low) / first_low
            if first_rise < params.min_rise_1st or first_rise > params.max_rise_1st:
                continue

            # 回调深度
            retrace = (first_high - retrace_low) / (first_high - first_low)
            if retrace < params.retrace_min or retrace > params.retrace_max:
                continue

            # 回调天数
            retrace_days = tb - best_p
            if retrace_days < params.retrace_days_min or retrace_days > params.retrace_days_max:
                continue

            # 费波位
            fib_level = _fib_to_level(retrace)
            fib_price = first_high - (first_high - first_low) * fib_level

            # 当前价必须接近费波位（有涨停基因放宽到15%，普通5%）
            fib_dist = abs(last_close - fib_price) / fib_price
            fib_dist_max = 0.15 if has_limit_up else 0.05
            if fib_dist > fib_dist_max:
                continue

            # === 关键：费波位必须与MA9或MA10形成共振 ===
            # 支撑位需要均线确认，费波位和均线不能差太远
            ma_fib_ok = False
            nearest_ma = None
            if ma10 is not None:
                ma10_fib_dist = abs(fib_price - ma10) / fib_price
                if ma10_fib_dist < 0.03:  # 放宽到3%检查共振
                    ma_fib_ok = True
                    nearest_ma = ('MA10', ma10)
            if not ma_fib_ok and ma9 is not None:
                ma9_fib_dist = abs(fib_price - ma9) / fib_price
                if ma9_fib_dist < 0.03:
                    ma_fib_ok = True
                    nearest_ma = ('MA9', ma9)

            # 企稳确认
            stab_ok, has_vol_shrink, has_shadow = _check_stabilization(
                opens, highs, lows, closes, vols, best_p, tb, params,
            )

            entry_price = round(last_close, 2)
            stop_loss = round(fib_price * 0.98, 2)

            # 等幅测距目标
            target = round(first_high + (first_high - first_low), 2)

            sig_data = {
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target_price': target,
                'fib_level': fib_level,
                'fib_price': round(fib_price, 2),
                'first_rise_pct': round(first_rise * 100, 1),
                'retrace_pct': round(retrace * 100, 1),
                'retrace_days': retrace_days,
                'first_peak': round(first_high, 2),
                'retrace_low': round(retrace_low, 2),
                'stab_ok': stab_ok,
                'has_vol_shrink': has_vol_shrink,
                'has_shadow': has_shadow,
                'ma_bullish': ma_bullish,
                'ma9_gt_ma10': ma9_gt_ma10,
                'ma_consistent': ma_consistent,
                'bullish_days': bullish_days,
                'has_limit_up': has_limit_up,
                'ma_fib_ok': ma_fib_ok,
                'ma10_broken_intraday': ma10_broken_intraday,
                'nearest_ma': nearest_ma,
                'ma9': round(ma9, 2) if ma9 is not None else None,
                'ma10': round(ma10, 2) if ma10 is not None else None,
            }
            sig_data['strength'] = _calc_strength(sig_data)
            signals.append(sig_data)

    return signals


def score_fundamental(code: str, close: float, client) -> dict:
    """获取并评分单只股票的基本面

    Returns:
        dict with pe, pb, net_profit_yi, score, is_loss_making
        如果 is_loss_making=True，则基本面无价值，应排除。
    """
    result = {'pe': 0, 'pb': 0, 'net_profit_yi': 0, 'score': 0, 'is_loss_making': False}

    try:
        fin = client.finance(symbol=code)
        if fin is None or len(fin) == 0:
            return result

        jlr = float(fin['jinglirun'].values[0])  # 净利润
        zgb = float(fin['zongguben'].values[0])  # 总股本
        mgjzc = float(fin['meigujingzichan'].values[0])  # 每股净资产

        mcap = close * zgb  # 总市值

        # 亏损股 → 不排除，但给负分
        if jlr <= 0:
            result['pe'] = 0
            result['pb'] = round(close / mgjzc, 1) if mgjzc > 0 else 0
            result['net_profit_yi'] = round(jlr / 1e8, 1)
            result['is_loss_making'] = True
            result['score'] = -15  # 亏损扣分，技术面强仍可入选
            return result

        pe = mcap / jlr
        pb = close / mgjzc if mgjzc > 0 else 999

        result['pe'] = round(pe, 1)
        result['pb'] = round(pb, 1)
        result['net_profit_yi'] = round(jlr / 1e8, 1)

        # 基本面评分 (0-20)
        score = 0
        if pe <= 30:
            score += 12  # 低估值
        elif pe <= 60:
            score += 8   # 合理
        elif pe <= 100:
            score += 3   # 偏高
        else:
            score -= 5   # 高估

        if pb <= 3:
            score += 8   # 低PB
        elif pb <= 6:
            score += 4   # 合理
        elif pb > 15:
            score -= 5   # 高PB

        result['score'] = score
        return result
    except Exception:
        return result


def scan_stock(
    code: str,
    name: str,
    df: pd.DataFrame,
    params: NPatternParams,
) -> list[NSignal]:
    """扫描单只股票的 N 字信号"""
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    vols = df['volume'].values

    raw_signals = find_n_signals(opens, highs, lows, closes, vols, params)

    result = []
    for s in raw_signals:
        sig = NSignal(
            code=code,
            name=name,
            entry_price=s['entry_price'],
            stop_loss=s['stop_loss'],
            target_price=s['target_price'],
            strength=s['strength'],
            fib_level=s['fib_level'],
            fib_price=s['fib_price'],
            first_rise_pct=s['first_rise_pct'],
            retrace_pct=s['retrace_pct'],
            retrace_days=s['retrace_days'],
            first_peak=s['first_peak'],
            retrace_low=s['retrace_low'],
            stab_ok=s['stab_ok'],
            has_vol_shrink=s['has_vol_shrink'],
            has_shadow=s['has_shadow'],
            ma_bullish=s['ma_bullish'],
            ma9_gt_ma10=s['ma9_gt_ma10'],
            ma_consistent=s['ma_consistent'],
            bullish_days=s['bullish_days'],
            has_limit_up=s['has_limit_up'],
            ma_fib_ok=s['ma_fib_ok'],
            ma10_broken_intraday=s.get('ma10_broken_intraday', False),
            nearest_ma=s.get('nearest_ma'),
            ma9=s.get('ma9') or 0,
            ma10=s.get('ma10') or 0,
        )
        result.append(sig)

    # 去重：同一只股票只保留最强信号
    if result:
        result.sort(key=lambda x: x.strength, reverse=True)
        result = result[:1]

    return result
