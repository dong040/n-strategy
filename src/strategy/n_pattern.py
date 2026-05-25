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
    fib_dist: float = 0
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
    limit_up_count: int = 0
    ma_fib_ok: bool = False
    ma10_broken_intraday: bool = False
    ma10_broken_close: bool = False
    nearest_ma: tuple = None
    ma9: float = 0
    ma10: float = 0
    # 基本面
    pe: float = 0
    pb: float = 0
    net_profit_yi: float = 0
    fundamental_score: int = 0
    # 压力位/卖出
    resistance_levels: list = field(default_factory=list)  # [(label, price, distance_pct), ...] 上方压力
    broken_levels: list = field(default_factory=list)  # [(label, price, distance_pct), ...] 刚突破的
    nearest_resistance: tuple = None  # (label, price, distance_pct)
    entry_to_resistance_pct: float = 0
    rr_ratio: float = 0
    fib_extension_1272: float = 0
    fib_extension_1618: float = 0
    is_big_n: bool = False
    market_pct: float = 0  # 当日大盘涨跌幅
    entry_source: str = ""  # 入场价来源: MA9/MA10/回调低/费波
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


def _check_limit_up(closes, lookback=20):
    """检查近N日涨停次数（>=9.5%近似涨停），返回次数"""
    n = len(closes)
    start = max(0, n - lookback)
    count = 0
    for i in range(start, n):
        if i > 0 and closes[i - 1] > 0:
            chg = (closes[i] - closes[i - 1]) / closes[i - 1]
            if chg >= 0.095:
                count += 1
    return count


def _calc_strength(signal_data: dict) -> int:
    """计算信号强度 0-100+"""
    strength = 40
    if signal_data.get('stab_ok'):
        strength += 20
    elif signal_data.get('has_vol_shrink') or signal_data.get('has_shadow'):
        strength += 8
    # 多头排列：持续3日+满分，2日半奖，1日无加分，0日重罚
    ma_bullish = signal_data.get('ma_bullish', False)
    bullish_days = signal_data.get('bullish_days', 0)
    if ma_bullish:
        if bullish_days >= 3:
            strength += 15  # 持续多头排列 = 强趋势
        elif bullish_days >= 2:
            strength += 5   # 刚形成多头排列
        strength += 5  # 今日多头排列
    else:
        strength -= 10  # 今日不是多头排列（含刚失去的情况）
    if signal_data.get('has_vol_shrink'):
        strength += 8
    if signal_data.get('has_shadow'):
        strength += 8
    if signal_data.get('ma_fib_ok'):
        strength += 10

    # 费波偏离度：贴近(0.5-2%)最优 — 回调途中逼近支撑，动手窗口
    # 到位(<0.5%)次之 — 已踩在支撑上，可能直接弹走
    # 太远(>5%)扣分 — 还没跌到位
    fib_dist = signal_data.get('fib_dist', 0)
    if 0.005 <= fib_dist <= 0.02:
        strength += 10  # 贴近支撑，回调途中最佳买点
    elif fib_dist < 0.005:
        strength += 5   # 已到位，可能已错过最佳窗口
    elif fib_dist > 0.05:
        strength -= 10  # 偏离太远，还需等
    if signal_data.get('fib_level', 1) <= 0.382 and signal_data.get('first_rise_pct', 0) > 0.30:
        strength += 5
    if signal_data.get('has_limit_up'):
        cnt = signal_data.get('limit_up_count', 0)
        if cnt >= 3:
            strength += 15  # 多次涨停，股性极活
        else:
            strength += 10  # 2次涨停

    # 罚分：MA10 支撑质量
    if signal_data.get('ma10_broken_close'):
        strength -= 15  # 收盘跌破 = 支撑失败
    elif signal_data.get('ma10_broken_intraday'):
        strength -= 5   # 日内触及但收在上方 = 支撑确认

    # 压力位罚分：区分前高压力 vs 纯整数关口
    entry_to_res = signal_data.get('entry_to_resistance_pct', 100)
    nearest_res = signal_data.get('nearest_resistance')
    is_swing_high = nearest_res and '前高' in str(nearest_res[0]) if nearest_res else False

    if entry_to_res < 2:
        strength -= 25 if is_swing_high else 15  # 前高重压 → 重罚
    elif entry_to_res < 5:
        strength -= 10 if is_swing_high else 5   # 整数关口 → 轻罚
    elif entry_to_res < 8 and is_swing_high:
        strength -= 5  # 接近前高，轻扣

    # 盈亏比奖励/罚分
    rr = signal_data.get('rr_ratio', 0)
    if rr >= 3:
        strength += 10
    elif rr >= 2:
        strength += 5
    elif rr < 1:
        strength -= 10  # 盈亏比 < 1:1 不划算

    # 大盘环境调整：弱势守住支撑=强庄，强势守住=随大流
    market_pct = signal_data.get('market_pct', 0)
    market_bonus = -round(market_pct * 10)  # -2% → +20, +1% → -10
    market_bonus = max(-25, min(25, market_bonus))  # 封顶 ±25
    strength += market_bonus

    return max(0, strength)


def _get_round_numbers(price: float) -> list:
    """生成当前价上方的整数关口"""
    levels = []
    if price <= 0:
        return levels
    if price < 10:
        step = 1
    elif price < 50:
        step = 5
    elif price < 100:
        step = 10
    elif price < 500:
        step = 50
    else:
        step = 100

    base = int(price // step) * step
    for i in range(1, 6):
        level = base + step * i
        if level > price:
            levels.append(level)
    return levels


def _cluster_levels(raw_levels: list, current_price: float, merge_pct: float = 0.015):
    """合并相近压力位（间距 < 1.5%），合并后保留最高价并标注来源

    返回 (active_levels, broken_levels):
      - active: 价格 > current_price 的压力位（上方）
      - broken: 价格在 current_price ±5% 范围内，包括刚被突破的（下方）
    """
    if not raw_levels:
        return [], []
    raw_levels.sort(key=lambda x: x[1])
    clustered = []
    cur_label, cur_price = raw_levels[0]
    for label, price in raw_levels[1:]:
        if (price - cur_price) / cur_price < merge_pct:
            cur_label = cur_label + '+' + label if label not in cur_label else cur_label
            cur_price = max(cur_price, price)
        else:
            dist = round((cur_price - current_price) / current_price * 100, 1)
            clustered.append((cur_label, round(cur_price, 2), dist))
            cur_label, cur_price = label, price
    dist = round((cur_price - current_price) / current_price * 100, 1)
    clustered.append((cur_label, round(cur_price, 2), dist))

    # 分离：上方压力位 vs 已被突破但仍在10%内的关键位
    active = [(l, p, d) for l, p, d in clustered if d > 0]
    broken = [(l, p, d) for l, p, d in clustered if -10 <= d <= 0]
    active.sort(key=lambda x: x[2])
    broken.sort(key=lambda x: x[2], reverse=True)  # 最近的突破位在前
    return active[:5], broken[:3]


def _find_resistance_levels(highs, peaks, current_price, first_high, first_low, retrace_low=None):
    """找到当前价附近的压力位（含刚被突破的）

    Returns:
        (resistance_levels, broken_levels)
        - resistance_levels: 上方压力位, sorted by distance asc
        - broken_levels: 刚被突破的(-5%内), sorted by proximity desc
    """
    raw = []

    # 1. 历史前高（只要在 ±5% 范围内的都收集）
    for p_idx in peaks:
        h = highs[p_idx]
        if h > current_price * 0.90:  # 包括刚被突破的 (-5%内)
            raw.append(('前高', float(h)))

    # 2. 整数关口（价格上方 + 紧邻下方）
    # 也生成一个低于当前价的最近关口
    all_rounds = _get_round_numbers(current_price)
    # 添加紧邻下方的关口
    if current_price < 10:
        step = 1
    elif current_price < 50:
        step = 5
    elif current_price < 100:
        step = 10
    elif current_price < 500:
        step = 50
    else:
        step = 100
    base = int(current_price // step) * step
    if base < current_price and base > current_price * 0.90:
        raw.append(('整数关口', float(base)))
    for rl in all_rounds:
        raw.append(('整数关口', float(rl)))

    # 3. 前高突破延伸位
    ext_base_low = retrace_low if retrace_low is not None else first_low
    pullback_range = first_high - ext_base_low
    if pullback_range > 0:
        for ratio, label in [(0.236, '延伸23.6%'), (0.382, '延伸38.2%')]:
            level = first_high + pullback_range * ratio
            if level > current_price * 0.90:
                raw.append((label, round(level, 2)))

        # 3b. 前高供应区上限：前高 - 回调段 × 10%
        # 大胜达案例: 前高20.80 - (20.80-17.83)×0.10 = 20.50
        if retrace_low is not None:
            supply_ceiling = first_high - pullback_range * 0.10
            if supply_ceiling > current_price * 0.90:
                raw.append(('供应区上限', round(supply_ceiling, 2)))

    # 4. Fib 扩展位 (基于第一波幅度)
    move = first_high - first_low
    if move > 0:
        for ext_pct, label in [(1.272, 'Fib127.2%'), (1.618, 'Fib161.8%')]:
            level = first_low + move * ext_pct
            if level > current_price * 0.90:
                raw.append((label, round(level, 2)))

    # 5. 大N = 小N × 0.90（小N没撑住→跌一个板到下一个支撑）
    # 航天电器案例: 大N 70.4 = 小N(前高) 78.22 × 0.90
    # A股主板10%涨跌停逻辑：支撑破后下一个自然支撑位
    small_n_candidates = [first_high]  # 前高是小N的第一候选
    if retrace_low is not None:
        small_n_candidates.append(first_high - (first_high - retrace_low) * 0.10)  # 供应区上限
    # 也从相邻前高延伸（铭普光磁案例：前高+(前高-上一前高)×0.236）
    peaks_sorted = sorted([p for p in peaks if highs[p] < first_high], key=lambda p: highs[p], reverse=True)
    if peaks_sorted:
        prev_peak = highs[peaks_sorted[0]]
        if prev_peak < first_high:
            ext_from_prev = first_high + (first_high - prev_peak) * 0.236
            small_n_candidates.append(ext_from_prev)

    for sn in small_n_candidates:
        bn = sn * 0.90
        if bn > current_price * 0.85:  # 宽松阈值，覆盖下方支撑
            raw.append(('大N', round(bn, 2)))

    return _cluster_levels(raw, current_price)


def find_n_signals(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    vols: np.ndarray,
    params: NPatternParams,
    market_pct: float = 0,
) -> list[dict]:
    """扫描K线中的 N 字买点

    核心原则（大神策略）：
    1. 上涨 → 回调找支撑 → 支撑住 → 再拉一波
    2. 支撑位必须"撑住"，跌破再拉回的不算
    3. 费波位 + MA9/MA10 共振才是有效支撑
    """
    n = len(closes)
    peaks, troughs = find_extrema(highs, lows)

    limit_up_count = _check_limit_up(closes, lookback=30)
    # 硬过滤：近30日至少1次涨停
    if limit_up_count < 1:
        return []
    has_limit_up = True  # 能走到这里说明 >=2
    ma_bullish, ma9_gt_ma10, bullish_days, ma_consistent, ma9, ma10, ma20 = _check_ma_bullish(closes)

    last_close = closes[-1]

    # === 硬过滤：中期趋势向下 → 下跌趋势中的反弹不参与 ===
    if n >= 65:
        ma60_now = float(np.mean(closes[-60:]))
        ma60_5ago = float(np.mean(closes[-65:-5]))
        if ma60_5ago > 0:
            ma60_slope = (ma60_now - ma60_5ago) / ma60_5ago
            if ma60_slope < -0.002:  # MA60 下降 >0.2%
                return []

    # === 硬过滤：收盘价跌破 MA10 → 支撑已失效，整只跳过 ===
    if ma10 is not None and last_close < ma10:
        return []

    # === MA 多头排列不持续 → 不硬过滤，改在强度中罚分 ===

    # === 硬过滤：今天涨停 → 回调结束已启动，不是买点 ===
    if n >= 2 and closes[-2] > 0:
        today_chg = (closes[-1] - closes[-2]) / closes[-2]
        if today_chg >= 0.095:
            return []

    # === 硬过滤：今天收盘跌破MA9/MA10 → 支撑失败，排除 ===
    # 盘中触及但收盘站回 = 支撑确认为有效，不排除
    today_low = lows[-1]
    if ma10 is not None and closes[-1] < ma10:
        return []
    if ma9 is not None and closes[-1] < ma9 * 0.98:
        return []

    # === 检测近期是否跌破过 MA10 ===
    # 收盘跌破 = 支撑失败(严重)；仅日内触及但收在上方 = 支撑确认(轻微)
    recent_lows = lows[-3:] if n >= 3 else lows
    recent_closes = closes[-3:] if n >= 3 else closes
    ma10_broken_close = (ma10 is not None and any(c < ma10 for c in recent_closes))
    ma10_touched_intraday = (ma10 is not None and any(low < ma10 for low in recent_lows))

    signals = []

    for ti, ta in enumerate(troughs):
        for tb in troughs[ti + 1:]:
            if tb < n - 7:
                continue  # 第二低谷必须在近7日（配合反弹过滤，不会追高）

            # N结构总时长限制：避免匹配过于久远的历史形态
            total_n_days = tb - ta
            if total_n_days > 90:
                continue

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

            # 弱趋势 + 深回调 = 动能衰竭
            if first_rise < 0.15 and retrace > 0.40:
                continue

            # 回调天数
            retrace_days = tb - best_p
            if retrace_days < 1 or retrace_days > params.retrace_days_max:
                continue

            # V型反转排除 — 跌完立刻涨停拉回 = 没有横盘撑住的过程
            v_reversal = False
            for di in range(1, min(4, n - tb)):  # 检查低谷后3天
                if closes[tb + di - 1] > 0:
                    chg_di = (closes[tb + di] - closes[tb + di - 1]) / closes[tb + di - 1]
                    if chg_di >= 0.095:
                        v_reversal = True
                        break
            if v_reversal:
                continue

            # 费波位
            fib_level = _fib_to_level(retrace)
            fib_price = first_high - (first_high - first_low) * fib_level

            # 当前价必须接近费波位（涨停基因 12%，普通 5%）
            fib_dist = abs(last_close - fib_price) / fib_price
            fib_dist_max = 0.12 if has_limit_up else 0.05
            if fib_dist > fib_dist_max:
                continue

            # === 关键：费波位必须与MA9或MA10形成共振 ===
            # 支撑位需要均线确认，费波位和均线不能差太远
            ma_fib_ok = False
            nearest_ma = None
            if ma10 is not None:
                ma10_fib_dist = abs(fib_price - ma10) / fib_price
                if ma10_fib_dist < 0.04:
                    ma_fib_ok = True
                    nearest_ma = ('MA10', ma10)
            if not ma_fib_ok and ma9 is not None:
                ma9_fib_dist = abs(fib_price - ma9) / fib_price
                if ma9_fib_dist < 0.04:
                    ma_fib_ok = True
                    nearest_ma = ('MA9', ma9)

            # 企稳确认 — 至少要有一种企稳信号
            stab_ok, has_vol_shrink, has_shadow = _check_stabilization(
                opens, highs, lows, closes, vols, best_p, tb, params,
            )
            if not stab_ok and not has_vol_shrink and not has_shadow:
                continue  # 无任何企稳信号，支撑位没人接盘

            # === 入场价：四候选 + 大盘环境 ===
            # 候选: MA9, MA10, 回调低点, 费波位 (必须 < 当前价)
            candidates = []
            if ma9 is not None and ma9 < last_close:
                candidates.append(('MA9', round(ma9, 2)))
            if ma10 is not None and ma10 < last_close:
                candidates.append(('MA10', round(ma10, 2)))
            if retrace_low < last_close:
                candidates.append(('回调低', round(retrace_low, 2)))
            fib_price_rounded = round(fib_price, 2)
            if fib_price_rounded < last_close:
                candidates.append(('费波', fib_price_rounded))

            if not candidates:
                continue

            candidates.sort(key=lambda x: x[1])  # 价格升序

            # 大盘环境选择入场价
            if market_pct > 0.5:
                # 强势: 取最高候选 (资金积极，浅回调即承接)
                entry_source, entry_price = candidates[-1]
            elif market_pct < -0.5:
                # 弱势: 取最低候选 (深回调才接，支撑容易破)
                entry_source, entry_price = candidates[0]
            else:
                # 平盘: 取中间候选
                mid = len(candidates) // 2
                entry_source, entry_price = candidates[mid]

            # 现价距入场支撑必须 ≤6.5%
            dist_to_entry = abs(last_close - entry_price) / entry_price
            if dist_to_entry > 0.065:
                continue

            fib_dist = dist_to_entry

            # 收盘跌破入场价 → 支撑失败，排除
            if closes[-1] < entry_price:
                continue

            # 今日盘中贴近/触及入场价 → 需要判断是否已有效反弹
            # 反弹>3% = 支撑已测+强力拉起，买点已过（雅克科技+5.6%）
            # 反弹<3% = 只是蹭了一下支撑，明天仍可能是买点（万润科技+1.5%）
            if lows[-1] <= entry_price * 1.005:
                bounce_from_low = (closes[-1] - lows[-1]) / lows[-1]
                if bounce_from_low > 0.03:
                    continue

            # 止损 = 入场价（跌破支撑位即离场，逻辑失效）
            stop_loss = round(entry_price * 0.995, 2)

            # 等幅测距目标
            target = round(first_high + (first_high - first_low), 2)

            # === 压力位/卖出位检测 (以入场价为基准) ===
            resistance_levels, broken_levels = _find_resistance_levels(
                highs, peaks, entry_price, first_high, first_low, retrace_low,
            )
            nearest_resistance = resistance_levels[0] if resistance_levels else None
            entry_to_resistance_pct = nearest_resistance[2] if nearest_resistance else 0

            # Fib 扩展位 (卖出参考)
            move = first_high - first_low
            fib_ext_1272 = round(first_low + move * 1.272, 2) if move > 0 else 0
            fib_ext_1618 = round(first_low + move * 1.618, 2) if move > 0 else 0

            # 盈亏比: (目标 - 入场) / (入场 - 止损)
            risk = entry_price - stop_loss
            reward = target - entry_price
            rr_ratio = round(reward / risk, 1) if risk > 0 else 0

            sig_data = {
                'entry_price': entry_price,
                'entry_source': entry_source,
                'market_pct': market_pct,
                'stop_loss': stop_loss,
                'target_price': target,
                'fib_level': fib_level,
                'fib_price': round(fib_price, 2),
                'fib_dist': round(fib_dist, 3),
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
                'limit_up_count': limit_up_count,
                'ma_fib_ok': ma_fib_ok,
                'ma10_broken_intraday': ma10_touched_intraday,
                'ma10_broken_close': ma10_broken_close,
                'nearest_ma': nearest_ma,
                'ma9': round(ma9, 2) if ma9 is not None else None,
                'ma10': round(ma10, 2) if ma10 is not None else None,
                # 压力位/卖出
                'resistance_levels': resistance_levels,
                'broken_levels': broken_levels,
                'nearest_resistance': nearest_resistance,
                'entry_to_resistance_pct': entry_to_resistance_pct,
                'rr_ratio': rr_ratio,
                'fib_extension_1272': fib_ext_1272,
                'fib_extension_1618': fib_ext_1618,
            }
            sig_data['strength'] = _calc_strength(sig_data)
            sig_data['_tb'] = tb  # 用于去重
            signals.append(sig_data)

    # === 大N扫描：涨停基因 + 跌破小N后在MA10×0.9深度支撑入场 ===
    # 大N: 股价跌破MA9(小N已破)，在MA10×0.9处寻求深度支撑。
    # 不要求MA/Fib共振，不要求多头排列。
    if ma10 is not None and ma10 < last_close and ma9 is not None:
        big_n_entry = round(ma10 * 0.9, 2)
        near_or_below_ma9 = last_close <= ma9 * 1.02  # 小N已破
        above_big_n = last_close > big_n_entry
        if near_or_below_ma9 and above_big_n:
            dist_to_big_n = abs(last_close - big_n_entry) / big_n_entry
            if dist_to_big_n <= 0.10:  # 深度入场，容忍更宽距离
                for ti, ta in enumerate(troughs):
                    for tb in troughs[ti + 1:]:
                        if tb < n - 7:
                            continue
                        total_n_days = tb - ta
                        if total_n_days > 90:
                            continue

                        peaks_between = [p for p in peaks if ta < p < tb]
                        if not peaks_between:
                            continue
                        best_p = max(peaks_between, key=lambda p: highs[p])

                        first_low = lows[ta]
                        first_high = highs[best_p]
                        retrace_low = lows[tb]

                        first_rise = (first_high - first_low) / first_low
                        if first_rise < 0.05 or first_rise > 0.70:
                            continue

                        retrace = (first_high - retrace_low) / (first_high - first_low)
                        if retrace < 0.25 or retrace > 0.65:
                            continue

                        if first_rise < 0.15 and retrace > 0.40:
                            continue

                        retrace_days = tb - best_p
                        if retrace_days < 1 or retrace_days > 15:
                            continue

                        # V型反转排除
                        v_reversal = False
                        for di in range(1, min(4, n - tb)):
                            if closes[tb + di - 1] > 0:
                                chg_di = (closes[tb + di] - closes[tb + di - 1]) / closes[tb + di - 1]
                                if chg_di >= 0.095:
                                    v_reversal = True
                                    break
                        if v_reversal:
                            continue

                        # 企稳确认
                        stab_ok, has_vol_shrink, has_shadow = _check_stabilization(
                            opens, highs, lows, closes, vols, best_p, tb, params,
                        )
                        if not stab_ok and not has_vol_shrink and not has_shadow:
                            continue

                        # 大N入场已测（今日最低已触及入场价）
                        if lows[-1] <= big_n_entry:
                            continue

                        # === 大N 信号构建 ===
                        entry_price = big_n_entry
                        stop_loss = round(entry_price * 0.995, 2)
                        target = round(first_high + (first_high - first_low), 2)

                        fib_level = _fib_to_level(retrace)
                        fib_price = first_high - (first_high - first_low) * fib_level

                        dist_to_entry = abs(last_close - entry_price) / entry_price

                        # 盈亏比
                        risk = entry_price - stop_loss
                        reward = target - entry_price
                        rr_ratio = round(reward / risk, 1) if risk > 0 else 0

                        # 压力位
                        resistance_levels, broken_levels = _find_resistance_levels(
                            highs, peaks, entry_price, first_high, first_low, retrace_low,
                        )
                        nearest_resistance = resistance_levels[0] if resistance_levels else None
                        entry_to_resistance_pct = nearest_resistance[2] if nearest_resistance else 0

                        move = first_high - first_low
                        fib_ext_1272 = round(first_low + move * 1.272, 2) if move > 0 else 0
                        fib_ext_1618 = round(first_low + move * 1.618, 2) if move > 0 else 0

                        big_sig = {
                            'entry_price': entry_price,
                            'entry_source': '大N',
                            'market_pct': market_pct,
                            'stop_loss': stop_loss,
                            'target_price': target,
                            'fib_level': fib_level,
                            'fib_price': round(fib_price, 2),
                            'fib_dist': round(dist_to_entry, 3),
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
                            'limit_up_count': limit_up_count,
                            'ma_fib_ok': False,  # 大N不要求共振
                            'ma10_broken_intraday': ma10_touched_intraday,
                            'ma10_broken_close': ma10_broken_close,
                            'nearest_ma': ('MA10', ma10),
                            'ma9': round(ma9, 2),
                            'ma10': round(ma10, 2),
                            'resistance_levels': resistance_levels,
                            'broken_levels': broken_levels,
                            'nearest_resistance': nearest_resistance,
                            'entry_to_resistance_pct': entry_to_resistance_pct,
                            'rr_ratio': rr_ratio,
                            'fib_extension_1272': fib_ext_1272,
                            'fib_extension_1618': fib_ext_1618,
                            'is_big_n': True,
                        }
                        big_sig['strength'] = _calc_strength(big_sig)
                        big_sig['_tb'] = f"big_{tb}"
                        signals.append(big_sig)

    # 去重：每个第二低谷保留最强信号
    if signals:
        best_per_tb = {}
        for s in signals:
            tb = s.pop('_tb')
            if tb not in best_per_tb or s['strength'] > best_per_tb[tb]['strength']:
                best_per_tb[tb] = s
        signals = list(best_per_tb.values())

    return signals


def score_fundamental(code: str, close: float, client) -> dict:
    """获取并评分单只股票的基本面

    Returns:
        dict with pe, pb, net_profit_yi, mcap_yi, score, is_loss_making, is_garbage_profitable
        is_loss_making: 亏损股，技术面强仍可入选
        is_garbage_profitable: 盈利但微利+高PE/小市值，明显垃圾，应排除
    """
    result = {
        'pe': 0, 'pb': 0, 'net_profit_yi': 0, 'mcap_yi': 0,
        'score': 0, 'is_loss_making': False, 'is_garbage_profitable': False,
    }

    try:
        fin = client.finance(symbol=code)
        if fin is None or len(fin) == 0:
            return result

        jlr = float(fin['jinglirun'].values[0])
        zgb = float(fin['zongguben'].values[0])
        mgjzc = float(fin['meigujingzichan'].values[0])

        mcap = close * zgb
        mcap_yi = round(mcap / 1e8, 1)
        net_profit_yi = round(jlr / 1e8, 1)
        result['net_profit_yi'] = net_profit_yi
        result['mcap_yi'] = mcap_yi

        # 亏损股 — 纯技术面策略不排除，不给负分
        if jlr <= 0:
            result['pb'] = round(close / mgjzc, 1) if mgjzc > 0 else 0
            result['is_loss_making'] = True
            result['score'] = 0
            return result

        pe = mcap / jlr
        pb = close / mgjzc if mgjzc > 0 else 999

        result['pe'] = round(pe, 1)
        result['pb'] = round(pb, 1)

        # === 盈利垃圾股检测 ===
        # 微利（<500万）且高PE（>150）且小市值（<30亿）→ 明显垃圾
        if net_profit_yi < 0.05 and pe > 150 and mcap_yi < 30:
            result['is_garbage_profitable'] = True
            result['score'] = -30
            return result
        # 微利+极高PE → 垃圾
        if net_profit_yi < 0.1 and pe > 200:
            result['is_garbage_profitable'] = True
            result['score'] = -30
            return result
        # 微利+极小市值（<15亿）→ 壳资源/垃圾
        if net_profit_yi < 0.05 and mcap_yi < 15:
            result['is_garbage_profitable'] = True
            result['score'] = -30
            return result

        # 基本面评分 (0-20)
        score = 0
        if pe <= 30:
            score += 12
        elif pe <= 60:
            score += 8
        elif pe <= 100:
            score += 3
        else:
            score -= 5

        if pb <= 3:
            score += 8
        elif pb <= 6:
            score += 4
        elif pb > 15:
            score -= 5

        result['score'] = score
        return result
    except Exception:
        return result


def scan_stock(
    code: str,
    name: str,
    df: pd.DataFrame,
    params: NPatternParams,
    market_pct: float = 0,
) -> list[NSignal]:
    """扫描单只股票的 N 字信号"""
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    vols = df['volume'].values

    raw_signals = find_n_signals(opens, highs, lows, closes, vols, params, market_pct)

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
            fib_dist=s.get('fib_dist', 0),
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
            limit_up_count=s.get('limit_up_count', 0),
            ma_fib_ok=s['ma_fib_ok'],
            ma10_broken_intraday=s.get('ma10_broken_intraday', False),
            ma10_broken_close=s.get('ma10_broken_close', False),
            nearest_ma=s.get('nearest_ma'),
            ma9=s.get('ma9') or 0,
            ma10=s.get('ma10') or 0,
            resistance_levels=s.get('resistance_levels', []),
            broken_levels=s.get('broken_levels', []),
            nearest_resistance=s.get('nearest_resistance'),
            entry_to_resistance_pct=s.get('entry_to_resistance_pct', 0),
            rr_ratio=s.get('rr_ratio', 0),
            fib_extension_1272=s.get('fib_extension_1272', 0),
            fib_extension_1618=s.get('fib_extension_1618', 0),
            is_big_n=s.get('is_big_n', False),
            market_pct=s.get('market_pct', 0),
            entry_source=s.get('entry_source', ''),
        )
        result.append(sig)

    # 去重：同类型每只股票只保留最强信号，普通N和大N各保留一个
    if result:
        best_regular = None
        best_big_n = None
        for s in result:
            if s.is_big_n:
                if best_big_n is None or s.strength > best_big_n.strength:
                    best_big_n = s
            else:
                if best_regular is None or s.strength > best_regular.strength:
                    best_regular = s
        result = [s for s in (best_regular, best_big_n) if s is not None]

    return result
