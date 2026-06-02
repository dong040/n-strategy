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
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .kline_sequence import (
    build_kline_tensor,
    load_sequence_model as _load_sequence_model,
    predict_sequence_prob as _predict_sequence_prob,
)
from .ml_filter import load_model as _load_ml_model, predict_signal as _predict_ml_signal

logger = logging.getLogger(__name__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEQUENCE_MODEL_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "data", "kline_sequence_model_tradefactors.pt"),
    os.path.join(PROJECT_ROOT, "data", "kline_sequence_model.pt"),
]
_SEQUENCE_MODEL_CACHE = None


# ====== 技术指标辅助函数（与 TradingAgents 同口径） ======

def _calc_rsi(closes: np.ndarray, n: int = 14) -> np.ndarray:
    """RSI (Wilder's EMA 方法)，返回完整序列，末值即 RSI(n)"""
    closes = np.asarray(closes, dtype=float)
    if len(closes) < n + 1:
        return np.full(len(closes), np.nan)
    delta = np.diff(closes)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.full(len(closes), np.nan)
    avg_loss = np.full(len(closes), np.nan)
    avg_gain[n] = np.mean(gain[:n])
    avg_loss[n] = np.mean(loss[:n])
    alpha = 1.0 / n
    for i in range(n + 1, len(closes)):
        avg_gain[i] = alpha * gain[i - 1] + (1 - alpha) * avg_gain[i - 1]
        avg_loss[i] = alpha * loss[i - 1] + (1 - alpha) * avg_loss[i - 1]
    rsi = np.full(len(closes), np.nan)
    for i in range(n, len(closes)):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _calc_macd(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """返回 (dif, dea, hist)，均为与 closes 同长的数组"""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    dif = np.full(n, np.nan)
    dea = np.full(n, np.nan)
    hist = np.full(n, np.nan)
    if n < slow:
        return dif, dea, hist
    ema_fast = closes.copy()
    ema_slow = closes.copy()
    alpha_f = 2.0 / (fast + 1)
    alpha_s = 2.0 / (slow + 1)
    alpha_sig = 2.0 / (signal + 1)
    for i in range(1, n):
        ema_fast[i] = alpha_f * closes[i] + (1 - alpha_f) * ema_fast[i - 1]
        ema_slow[i] = alpha_s * closes[i] + (1 - alpha_s) * ema_slow[i - 1]
    for i in range(n):
        dif[i] = ema_fast[i] - ema_slow[i]
    first_dea = slow + signal - 1
    if first_dea < n:
        dea[first_dea] = np.mean(dif[slow:first_dea + 1])
        for i in range(first_dea + 1, n):
            dea[i] = alpha_sig * dif[i] + (1 - alpha_sig) * dea[i - 1]
    for i in range(n):
        if not np.isnan(dif[i]) and not np.isnan(dea[i]):
            hist[i] = (dif[i] - dea[i]) * 2  # 同花顺口径
    return dif, dea, hist


def _calc_kdj(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
              n: int = 9, m1: int = 3, m2: int = 3):
    """返回 (k, d, j)，均为与 closes 同长的数组"""
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    length = len(closes)
    k = np.full(length, np.nan)
    d = np.full(length, np.nan)
    j = np.full(length, np.nan)
    if length < n:
        return k, d, j
    alpha_k = 1.0 / m1
    alpha_d = 1.0 / m2
    last_k, last_d = 50.0, 50.0
    for i in range(n - 1, length):
        hh = np.max(highs[i - n + 1:i + 1])
        ll = np.min(lows[i - n + 1:i + 1])
        rsv = (closes[i] - ll) / (hh - ll) * 100 if hh > ll else 50.0
        last_k = (1 - alpha_k) * last_k + alpha_k * rsv
        last_d = (1 - alpha_d) * last_d + alpha_d * last_k
        k[i] = last_k
        d[i] = last_d
        j[i] = 3 * last_k - 2 * last_d
    return k, d, j


def _calc_boll(closes: np.ndarray, n: int = 20, k: float = 2.0):
    """返回 (mid, upper, lower, bandwidth) 序列，bandwidth = (upper-lower)/mid"""
    closes = np.asarray(closes, dtype=float)
    length = len(closes)
    mid = np.full(length, np.nan)
    upper = np.full(length, np.nan)
    lower = np.full(length, np.nan)
    bw = np.full(length, np.nan)
    if length < n:
        return mid, upper, lower, bw
    for i in range(n - 1, length):
        win = closes[i - n + 1:i + 1]
        mid[i] = np.mean(win)
        std = np.std(win)
        upper[i] = mid[i] + k * std
        lower[i] = mid[i] - k * std
        bw[i] = (upper[i] - lower[i]) / mid[i] if mid[i] > 0 else np.nan
    return mid, upper, lower, bw


def _calc_mfi(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
              vols: np.ndarray, n: int = 14) -> np.ndarray:
    """Money Flow Index — 量价结合的 RSI"""
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    vols = np.asarray(vols, dtype=float)
    length = len(closes)
    mfi = np.full(length, np.nan)
    if length < n + 1:
        return mfi
    tp = (highs + lows + closes) / 3.0
    mf = tp * vols
    pos_flow = np.zeros(length)
    neg_flow = np.zeros(length)
    for i in range(1, length):
        if tp[i] > tp[i - 1]:
            pos_flow[i] = mf[i]
        elif tp[i] < tp[i - 1]:
            neg_flow[i] = mf[i]
    for i in range(n, length):
        pos_sum = np.sum(pos_flow[i - n + 1:i + 1])
        neg_sum = np.sum(neg_flow[i - n + 1:i + 1])
        if neg_sum == 0:
            mfi[i] = 100.0
        else:
            mr = pos_sum / neg_sum
            mfi[i] = 100.0 - 100.0 / (1.0 + mr)
    return mfi


def _calc_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, n: int = 14) -> np.ndarray:
    """Average Directional Index."""
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    length = len(closes)
    adx = np.full(length, np.nan)
    if length < n + 2:
        return adx

    tr = np.zeros(length)
    plus_dm = np.zeros(length)
    minus_dm = np.zeros(length)
    for i in range(1, length):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0.0

    atr = np.full(length, np.nan)
    plus_di = np.full(length, np.nan)
    minus_di = np.full(length, np.nan)
    dx = np.full(length, np.nan)
    atr[n] = np.sum(tr[1:n + 1])
    plus_sum = np.sum(plus_dm[1:n + 1])
    minus_sum = np.sum(minus_dm[1:n + 1])

    for i in range(n, length):
        if i > n:
            atr[i] = atr[i - 1] - atr[i - 1] / n + tr[i]
            plus_sum = plus_sum - plus_sum / n + plus_dm[i]
            minus_sum = minus_sum - minus_sum / n + minus_dm[i]
        if atr[i] > 0:
            plus_di[i] = 100 * plus_sum / atr[i]
            minus_di[i] = 100 * minus_sum / atr[i]
            denom = plus_di[i] + minus_di[i]
            if denom > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / denom

    valid_dx = dx[n:n * 2]
    valid_dx = valid_dx[np.isfinite(valid_dx)]
    if len(valid_dx) == 0:
        return adx
    first_idx = n * 2 - 1
    if first_idx < length:
        adx[first_idx] = np.mean(valid_dx)
        for i in range(first_idx + 1, length):
            if np.isfinite(dx[i]) and np.isfinite(adx[i - 1]):
                adx[i] = ((adx[i - 1] * (n - 1)) + dx[i]) / n
    return adx


def _calc_obv(closes: np.ndarray, vols: np.ndarray) -> np.ndarray:
    closes = np.asarray(closes, dtype=float)
    vols = np.asarray(vols, dtype=float)
    obv = np.zeros(len(closes), dtype=float)
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv[i] = obv[i - 1] + vols[i]
        elif closes[i] < closes[i - 1]:
            obv[i] = obv[i - 1] - vols[i]
        else:
            obv[i] = obv[i - 1]
    return obv


def _calc_cmf(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, vols: np.ndarray, n: int = 20) -> np.ndarray:
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    vols = np.asarray(vols, dtype=float)
    length = len(closes)
    out = np.full(length, np.nan)
    mfv = np.zeros(length, dtype=float)
    for i in range(length):
        hl = highs[i] - lows[i]
        mult = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / hl if hl > 0 else 0.0
        mfv[i] = mult * vols[i]
    for i in range(n - 1, length):
        vol_sum = np.sum(vols[i - n + 1:i + 1])
        if vol_sum > 0:
            out[i] = np.sum(mfv[i - n + 1:i + 1]) / vol_sum
    return out


# ====== 参数与信号 ======


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
    stop_loss_mode: str = "fixed"
    stop_loss_pct: float = 0.02
    stop_atr_mult: float = 0.0  # >0 时用 ATR 止损, e.g. 1.0 = 1倍ATR
    take_profit_mode: str = "equal"
    take_profit_pct: float = 0.10

    # 辅助条件
    max_mcap_yi: float = 500
    min_mcap_yi: float = 20
    min_turnover_wan: float = 5000
    enable_trading_factors: bool = True
    high_win_mode: bool = True
    high_win_min_strength: int = 85
    high_win_min_close_position_score: int = 5
    high_win_min_volatility_contraction_score: int = 5
    high_win_min_sequence_confidence: float = 0.47
    high_win_min_factor_score: int = 0
    high_win_min_resistance_pct: float = 3.0
    high_win_min_rr: float = 1.5
    high_win_min_ml_confidence: float = 0.50
    high_win_require_ml: bool = True


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
    ml_confidence: float = 0.0       # XGBoost 模型预测胜率
    sequence_confidence: float = 0.0
    sequence_score: int = 0
    # Batch 2: TradingAgents LLM 打分
    tradingagents_confidence: float = 0.0
    tradingagents_action: str = ""
    tradingagents_divergence: float = 0.0
    factor_score: int = 0
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


def _check_limit_up(closes, lookback=20, threshold=0.095):
    """检查近N日涨幅>=threshold的次数，返回次数"""
    n = len(closes)
    start = max(0, n - lookback)
    count = 0
    for i in range(start, n):
        if i > 0 and closes[i - 1] > 0:
            chg = (closes[i] - closes[i - 1]) / closes[i - 1]
            if chg >= threshold:
                count += 1
    return count


def _safe_mean(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    return float(np.mean(arr))


def _clip_score(value: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(value))))


def _calc_trading_factors(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    vols: np.ndarray,
    start_idx: int,
    peak_idx: int,
    trough_idx: int,
    amounts: np.ndarray = None,
    live_data: dict = None,
) -> dict:
    """计算适合 N 字回调的交易行为因子。

    这些因子只依赖历史 OHLCV，方便在每日扫描和历史回测里保持同一口径。
    amounts 可选，传入后可计算成交额质量因子（均价趋势、异常放量检测）。
    live_data 可选，传入后启用实时数据增强（同花顺热点/行业排名/北向资金）。
    """
    if live_data is None:
        live_data = {}
    n = len(closes)
    factors = {
        "pullback_volume_score": 0,
        "turnover_crowding_score": 0,
        "relative_strength_score": 0,
        "volatility_contraction_score": 0,
        "support_reclaim_score": 0,
        "close_position_score": 0,
        "limit_up_followthrough_score": 0,
        "theme_heat_score": 0,
        "amount_quality_score": 0,
        "market_regime_score": 0,
        "northbound_flow_score": 0,
        "shadow_quality_score": 0,
        "pullback_speed_score": 0,
        "intraday_reversal_score": 0,
        "volume_climax_score": 0,
        "sector_relative_score": 0,
        "adx_trend_score": 0,
        "obv_accumulation_score": 0,
        "cmf_score": 0,
        "gap_support_score": 0,
        "ml_confidence": 0.0,
        "ml_confidence_score": 0,
        "sequence_confidence": 0.0,
        "sequence_score": 0,
        "factor_score": 0,
        "factor_details": {},
    }
    if n < 30:
        return factors

    eps = 1e-9
    rise_vol = vols[start_idx:peak_idx + 1]
    pullback_vol = vols[peak_idx + 1:trough_idx + 1]
    if len(pullback_vol) == 0:
        pullback_vol = vols[peak_idx:trough_idx + 1]

    rise_avg_vol = _safe_mean(rise_vol)
    pullback_avg_vol = _safe_mean(pullback_vol)
    support_vol = float(vols[trough_idx]) if trough_idx < len(vols) else float(vols[-1])
    pullback_max_vol = float(np.max(pullback_vol)) if len(pullback_vol) else support_vol

    support_shrink = 1 - support_vol / max(pullback_max_vol, eps)
    pullback_vs_rise = pullback_avg_vol / max(rise_avg_vol, eps)
    pull_score = 0
    if support_shrink >= 0.55:
        pull_score += 8
    elif support_shrink >= 0.35:
        pull_score += 4
    elif support_shrink < 0.10:
        pull_score -= 6
    if pullback_vs_rise <= 0.65:
        pull_score += 6
    elif pullback_vs_rise <= 0.85:
        pull_score += 3
    elif pullback_vs_rise > 1.20:
        pull_score -= 8
    elif pullback_vs_rise > 1.00:
        pull_score -= 4
    factors["pullback_volume_score"] = _clip_score(-pull_score, -14, 10)

    vol60 = _safe_mean(vols[-60:]) if n >= 60 else _safe_mean(vols)
    vol20 = _safe_mean(vols[-20:])
    recent3_vol = _safe_mean(vols[-3:])
    recent5_vol = _safe_mean(vols[-5:])
    recent3_to_60 = recent3_vol / max(vol60, eps)
    recent5_to_20 = recent5_vol / max(vol20, eps)
    ret5 = closes[-1] / closes[-6] - 1 if n >= 6 and closes[-6] > 0 else 0
    crowd_score = 0
    if recent3_to_60 > 2.5:
        crowd_score -= 10
    elif recent3_to_60 > 1.8 and ret5 < 0.03:
        crowd_score -= 8
    elif recent3_to_60 > 1.8:
        crowd_score -= 2
    elif 0.80 <= recent3_to_60 <= 1.50:
        crowd_score += 4
    elif recent3_to_60 < 0.50:
        crowd_score += 2  # 极度缩量 = 卖压枯竭，利好
    if recent5_to_20 > 1.6 and ret5 < 0:
        crowd_score -= 4
    factors["turnover_crowding_score"] = _clip_score(-crowd_score, -8, 10)

    ret20 = closes[-1] / closes[-21] - 1 if n >= 21 and closes[-21] > 0 else 0
    ret60 = closes[-1] / closes[-61] - 1 if n >= 61 and closes[-61] > 0 else ret20
    high60 = float(np.max(highs[-60:])) if n >= 60 else float(np.max(highs))
    low20 = float(np.min(lows[-20:])) if n >= 20 else float(np.min(lows))
    dist_from_high60 = closes[-1] / max(high60, eps) - 1
    recovery_from_low20 = closes[-1] / max(low20, eps) - 1
    rs_score = 0
    if 0.12 <= ret60 <= 0.45:
        rs_score += 5
    elif 0.04 <= ret60 < 0.12:
        rs_score += 3
    elif ret60 > 0.70:
        rs_score -= 6
    elif ret60 > 0.45:
        rs_score -= 2
    elif ret60 < -0.10:
        rs_score -= 6
    if 0.02 <= ret20 <= 0.18:
        rs_score += 4
    elif ret20 > 0.28:
        rs_score -= 3
    elif ret20 < -0.06:
        rs_score -= 4
    if dist_from_high60 > -0.18:
        rs_score += 3
    elif dist_from_high60 < -0.35:
        rs_score -= 4
    if recovery_from_low20 > 0.04:
        rs_score += 2
    factors["relative_strength_score"] = _clip_score(rs_score, -10, 12)

    ranges = highs - lows
    atr5 = _safe_mean(ranges[-5:])
    atr20 = _safe_mean(ranges[-20:])
    range3 = _safe_mean(ranges[-3:])
    atr5_to_20 = atr5 / max(atr20, eps)
    range3_to_20 = range3 / max(atr20, eps)
    vol_score = 0
    if atr5_to_20 <= 0.75:
        vol_score += 7
    elif atr5_to_20 <= 0.90:
        vol_score += 3
    elif atr5_to_20 > 1.35:
        vol_score -= 7
    if range3_to_20 <= 0.75:
        vol_score += 4
    elif range3_to_20 > 1.30:
        vol_score -= 4
    factors["volatility_contraction_score"] = _clip_score(vol_score, -10, 11)

    support_low = float(lows[trough_idx]) if trough_idx < n else float(lows[-1])
    support_close = float(closes[trough_idx]) if trough_idx < n else float(closes[-1])
    recent_low = float(np.min(lows[-3:]))
    reclaim_from_support = support_close / max(support_low, eps) - 1
    recent_reclaim = closes[-1] / max(recent_low, eps) - 1
    support_pierce = min(0.0, support_low / max(closes[trough_idx - 1], eps) - 1) if trough_idx > 0 else 0.0
    reclaim_score = 0
    if 0.012 <= reclaim_from_support <= 0.05:
        reclaim_score += 7
    elif 0.006 <= reclaim_from_support < 0.012:
        reclaim_score += 3
    elif reclaim_from_support > 0.08:
        reclaim_score -= 5
    if 0.008 <= recent_reclaim <= 0.04:
        reclaim_score += 4
    elif recent_reclaim > 0.07:
        reclaim_score -= 4
    if support_pierce < -0.04:
        reclaim_score -= 6
    elif -0.02 <= support_pierce < 0:
        reclaim_score += 2
    factors["support_reclaim_score"] = _clip_score(-reclaim_score, -12, 12)

    close_pos = (closes[-1] - lows[-1]) / max(highs[-1] - lows[-1], eps)
    close_pos_3 = _safe_mean([
        (closes[i] - lows[i]) / max(highs[i] - lows[i], eps)
        for i in range(max(0, n - 3), n)
    ])
    upper_shadow = (highs[-1] - max(opens[-1], closes[-1])) / max(closes[-1], eps)
    close_score = 0
    if close_pos >= 0.75:
        close_score += 7
    elif close_pos >= 0.60:
        close_score += 4
    elif close_pos < 0.35:
        close_score -= 7
    if close_pos_3 >= 0.62:
        close_score += 4
    elif close_pos_3 < 0.42:
        close_score -= 4
    if upper_shadow > 0.045:
        close_score -= 6
    elif upper_shadow > 0.025:
        close_score -= 3
    factors["close_position_score"] = _clip_score(close_score, -12, 12)

    limit_indices = []
    for i in range(max(1, n - 30), n):
        if closes[i - 1] > 0 and (closes[i] - closes[i - 1]) / closes[i - 1] >= 0.095:
            limit_indices.append(i)
    follow_score = 0
    limit_days_ago = None
    limit_next_ret = 0.0
    limit_post_drawdown = 0.0
    if limit_indices:
        li = limit_indices[-1]
        limit_days_ago = n - 1 - li
        if li + 1 < n and closes[li] > 0:
            limit_next_ret = closes[li + 1] / closes[li] - 1
            if limit_next_ret >= 0.02:
                follow_score += 5
            elif limit_next_ret >= -0.02:
                follow_score += 2
            elif limit_next_ret < -0.06:
                follow_score -= 8
            else:
                follow_score -= 4
        end = min(n, li + 6)
        if li + 1 < end:
            limit_post_drawdown = float(np.min(lows[li + 1:end]) / max(closes[li], eps) - 1)
            if limit_post_drawdown > -0.06:
                follow_score += 2
            elif limit_post_drawdown < -0.14:
                follow_score -= 8
            elif limit_post_drawdown < -0.10:
                follow_score -= 4
        if limit_days_ago <= 3:
            follow_score -= 3
        elif 5 <= limit_days_ago <= 20:
            follow_score += 2
    factors["limit_up_followthrough_score"] = _clip_score(-follow_score, -9, 12)

    # ── 因子8: 题材热度 (theme_heat_score) ──
    # 用 OHLCV 代理题材活跃度：振幅扩张 + 超额收益 + 涨停密度
    ranges_arr = highs - lows
    amp5 = _safe_mean(ranges_arr[-5:]) / max(_safe_mean(closes[-5:]), eps)
    amp20 = _safe_mean(ranges_arr[-20:]) / max(_safe_mean(closes[-20:]), eps)
    amp_expansion = amp5 / max(amp20, eps)
    ret5_vs_self = closes[-1] / max(closes[-6], eps) - 1 if n >= 6 and closes[-6] > 0 else 0
    theme_score = 0
    if amp_expansion > 1.5:
        theme_score += 4
    elif amp_expansion > 1.2:
        theme_score += 2
    elif amp_expansion < 0.6:
        theme_score -= 3
    if ret5_vs_self > 0.05:
        theme_score += 3
    elif ret5_vs_self > 0.02:
        theme_score += 1
    elif ret5_vs_self < -0.03:
        theme_score -= 4
    limit_cnt = sum(
        1 for i in range(max(1, n - 30), n)
        if closes[i - 1] > 0 and (closes[i] - closes[i - 1]) / closes[i - 1] >= 0.095
    )
    if limit_cnt >= 3:
        theme_score += 5
    elif limit_cnt >= 1:
        theme_score += 2
    # 成交额扩张 (amounts available 时)
    if amounts is not None and len(amounts) >= 20:
        amt5 = _safe_mean(amounts[-5:])
        amt20_val = _safe_mean(amounts[-20:])
        amt_exp = amt5 / max(amt20_val, eps)
        if amt_exp > 2.0:
            theme_score += 3
        elif amt_exp > 1.3:
            theme_score += 1
        elif amt_exp < 0.4:
            theme_score -= 3
    # 实时增强：同花顺热点 + 行业排名
    hot_set = live_data.get("hot_code_set", set())
    sector_rank_map = live_data.get("sector_rank", {})
    if hot_set:
        # 该股在今日同花顺强势股列表中 → 有题材催化
        # 注意: _calc_trading_factors 没有 code 参数，需要外部传入
        # 这里先用调用方传入的 live_data["is_hot"] 来判断
        if live_data.get("is_hot"):
            theme_score += 4
        if live_data.get("sector_rank_pct", 1.0) < 0.2:  # 行业排名前20%
            theme_score += 3
        elif live_data.get("sector_rank_pct", 1.0) < 0.4:
            theme_score += 1
    # 反向：热门题材 = 拥挤交易，N字机会在冷门/被忽视的标的中更可靠
    factors["theme_heat_score"] = _clip_score(theme_score, -8, 12)

    # ── 因子9: 成交额质量 (amount_quality_score) ──
    amount_score = 0
    if amounts is not None and len(amounts) >= 20:
        # 回调期成交额衰减
        pullback_amt = amounts[peak_idx + 1:trough_idx + 1]
        if len(pullback_amt) == 0:
            pullback_amt = amounts[peak_idx:trough_idx + 1]
        pb_amt_max = float(np.max(pullback_amt)) if len(pullback_amt) else float(amounts[-1])
        support_amt = float(amounts[trough_idx]) if trough_idx < len(amounts) else float(amounts[-1])
        amt_shrink = 1 - support_amt / max(pb_amt_max, eps)
        if amt_shrink >= 0.55:
            amount_score += 5
        elif amt_shrink >= 0.35:
            amount_score += 3
        elif amt_shrink < 0.10:
            amount_score -= 4
        # 均价趋势: amount/vol = 近似均价, 均价上行 = 量价齐升
        avg_price3 = _safe_mean(amounts[-3:]) / max(_safe_mean(vols[-3:]), eps)
        avg_price20_val = _safe_mean(amounts[-20:]) / max(_safe_mean(vols[-20:]), eps)
        price_trend = avg_price3 / max(avg_price20_val, eps)
        if price_trend > 1.12:
            amount_score += 3
        elif price_trend < 0.85:
            amount_score -= 3
        # 异常放量检测
        amt5_vs_20 = _safe_mean(amounts[-5:]) / max(_safe_mean(amounts[-20:]), eps)
        if amt5_vs_20 > 3.0:
            amount_score -= 6
        elif amt5_vs_20 > 2.0:
            amount_score -= 3
        # 成交额稳定性 (变异系数)
        amt5_std = float(np.std(amounts[-5:]))
        amt5_mean = _safe_mean(amounts[-5:])
        amt_cv = amt5_std / max(amt5_mean, eps)
        if amt_cv < 0.3:
            amount_score += 4
        elif amt_cv > 0.7:
            amount_score -= 4
    factors["amount_quality_score"] = _clip_score(-amount_score, -12, 10)

    # ── 因子10: 大盘环境 (market_regime_score) ──
    # 用个股自身 MA 关系代理市场牛熊（回测兼容），扫描时可叠加真实指数数据
    regime_score = 0
    if n >= 60:
        ma20 = _safe_mean(closes[-20:])
        ma60_val = _safe_mean(closes[-60:])
        # MA20 vs MA60: 短期在中期上方 = 多头环境
        if ma20 > ma60_val * 1.03:
            regime_score += 3
        elif ma20 > ma60_val:
            regime_score += 1
        elif ma20 < ma60_val * 0.97:
            regime_score -= 3
        # MA60 方向
        ma60_10ago = _safe_mean(closes[-70:-10]) if n >= 70 else ma60_val
        ma60_slope_10d = (ma60_val - ma60_10ago) / max(ma60_10ago, eps)
        if ma60_slope_10d > 0.02:
            regime_score += 2
        elif ma60_slope_10d > 0.005:
            regime_score += 1
        elif ma60_slope_10d < -0.01:
            regime_score -= 4
        # 波动率环境: ATR20 / close, 低波 = 适合均值回归
        atr20_val = _safe_mean(highs[-20:] - lows[-20:])
        atr20_pct = atr20_val / max(closes[-1], eps)
        if atr20_pct < 0.02:
            regime_score += 3
        elif atr20_pct < 0.04:
            regime_score += 1
        elif atr20_pct > 0.07:
            regime_score -= 3
    # 反向：N字回调适合在中性/震荡市入场，强牛市中回调多为陷阱
    factors["market_regime_score"] = _clip_score(-regime_score, -8, 8)

    # ── 因子11: 北向资金/机构痕迹 (northbound_flow_score) ──
    # 回测用 OHLCV 代理：上涨段连续大阳线 + 回调缩量 = 机构吸筹痕迹
    nf_score = 0
    if start_idx < peak_idx:
        rise_slice_close = closes[start_idx:peak_idx + 1]
        rise_slice_vol = vols[start_idx:peak_idx + 1]
        if len(rise_slice_close) >= 3:
            # 上涨段中大阳线(>3%)天数
            big_green = sum(
                1 for i in range(1, len(rise_slice_close))
                if rise_slice_close[i - 1] > 0 and
                (rise_slice_close[i] - rise_slice_close[i - 1]) / rise_slice_close[i - 1] > 0.03
            )
            # 上涨段中放量天数(vol > 1.3x avg)
            rise_avg_v = _safe_mean(rise_slice_vol)
            big_vol_days = sum(1 for v in rise_slice_vol if v > rise_avg_v * 1.3)
            rise_len = len(rise_slice_close)
            if big_green >= 2 and big_vol_days >= 2:
                nf_score += 4
            elif big_green >= 1:
                nf_score += 2
            # 上涨斜率: 越陡 = 越像主力拉升
            rise_slope = (rise_slice_close[-1] - rise_slice_close[0]) / max(rise_slice_close[0], eps) / rise_len
            if rise_slope > 0.02:
                nf_score += 3
            elif rise_slope > 0.01:
                nf_score += 1
    # 回调有序 = 主力没走，不是散户踩踏
    if trough_idx > peak_idx:
        pullback_len = trough_idx - peak_idx
        pullback_closes_arr = closes[peak_idx:trough_idx + 1]
        if pullback_len >= 2:
            orderly = sum(
                1 for i in range(1, len(pullback_closes_arr))
                if abs(pullback_closes_arr[i] - pullback_closes_arr[i - 1]) / max(pullback_closes_arr[i - 1], eps) < 0.04
            )
            if orderly >= pullback_len - 1:
                nf_score += 3
            elif orderly >= pullback_len // 2:
                nf_score += 1
            else:
                nf_score -= 3
    # 实时增强：北向资金流向
    nb_real = live_data.get("northbound_score", 0)
    if nb_real != 0:
        nf_score = nf_score + nb_real  # 在 OHLCV 代理分基础上叠加实时数据
    factors["northbound_flow_score"] = _clip_score(-nf_score, -8, 5)

    # ── 因子12: RSI 背离 (rsi_divergence_score) ──
    rsi_arr = _calc_rsi(closes, 14)
    rsi_score = 0
    if not np.isnan(rsi_arr[-1]):
        # RSI 底背离: 回调低点是新低，但 RSI 高于上一个低点
        rsi_now = float(rsi_arr[-1])
        rsi_at_trough = float(rsi_arr[trough_idx]) if trough_idx < len(rsi_arr) else rsi_now
        # 在回踩区域 search 低点
        search_start = max(peak_idx, 20)
        if search_start < trough_idx:
            seg_lows = lows[search_start:trough_idx + 1]
            seg_rsi = rsi_arr[search_start:trough_idx + 1]
            valid = ~np.isnan(seg_rsi)
            if valid.any() and len(seg_lows) >= 3:
                price_min_idx = int(np.argmin(seg_lows[valid]))
                rsi_at_price_min = seg_rsi[valid][price_min_idx] if price_min_idx < len(seg_rsi[valid]) else rsi_now
                rsi_earlier = float(np.nanmin(seg_rsi[valid][:max(1, len(seg_rsi[valid]) // 2)]))
                # 底背离: 价格创新低但RSI走高
                if rsi_at_price_min > rsi_earlier + 5:
                    rsi_score += 8
                elif rsi_at_price_min > rsi_earlier + 2:
                    rsi_score += 4
        # RSI 绝对值
        if rsi_at_trough < 30:
            rsi_score += 4
        elif rsi_at_trough < 40:
            rsi_score += 2
        elif rsi_at_trough > 65:
            rsi_score -= 4
        # RSI 近期拐头向上
        if n >= 5:
            rsi_recent = rsi_arr[-5:]
            valid_recent = rsi_recent[~np.isnan(rsi_recent)]
            if len(valid_recent) >= 3:
                rsi_slope = float(valid_recent[-1] - valid_recent[0])
                if rsi_slope > 8:
                    rsi_score += 3
                elif rsi_slope < -8:
                    rsi_score -= 4
    factors["rsi_divergence_score"] = _clip_score(rsi_score, -10, 10)

    # ── 因子13: MACD 信号 (macd_signal_score) ──
    dif, dea, hist = _calc_macd(closes)
    macd_score = 0
    if n >= 30 and not np.isnan(dif[-1]) and not np.isnan(dea[-1]):
        # 金叉检测（最近 3 根 bar）
        cross_bars = []
        for i in range(max(1, n - 5), n):
            if not np.isnan(dif[i]) and not np.isnan(dea[i]) and not np.isnan(dif[i - 1]) and not np.isnan(dea[i - 1]):
                if dif[i - 1] <= dea[i - 1] and dif[i] > dea[i]:
                    cross_bars.append(i)
        if cross_bars:
            last_cross = cross_bars[-1]
            days_since = n - 1 - last_cross
            if days_since <= 3:
                macd_score += 6
                # 零轴上方金叉更强
                if float(dif[last_cross]) > 0:
                    macd_score += 4
                elif float(dif[last_cross]) > -0.5:
                    macd_score += 2
            elif days_since <= 5:
                macd_score += 3
        # 死叉惩罚
        for i in range(max(1, n - 5), n):
            if not np.isnan(dif[i]) and not np.isnan(dea[i]) and not np.isnan(dif[i - 1]) and not np.isnan(dea[i - 1]):
                if dif[i - 1] >= dea[i - 1] and dif[i] < dea[i]:
                    if n - 1 - i <= 3:
                        macd_score -= 6
                    break
        # 柱状图扩张（动量确认）
        if n >= 5:
            recent_hist = hist[-5:]
            valid_h = recent_hist[~np.isnan(recent_hist)]
            if len(valid_h) >= 3:
                if valid_h[-1] > valid_h[0] * 1.5 and valid_h[-1] > 0:
                    macd_score += 3
    factors["macd_signal_score"] = _clip_score(macd_score, -8, 10)

    # ── 因子14: MA 排列完整度 (ma_alignment_score) ──
    ma_score = 0
    if n >= 60:
        ma5 = _safe_mean(closes[-5:])
        ma10 = _safe_mean(closes[-10:])
        ma20 = _safe_mean(closes[-20:])
        ma60 = _safe_mean(closes[-60:])
        if ma5 > ma10 > ma20 > ma60:
            ma_score += 8
        elif ma5 > ma10 > ma20:
            ma_score += 3
        elif ma5 < ma10 and ma20 < ma60:
            ma_score -= 6
        # 均线粘合（收敛到 3% 以内 = 变盘前兆）
        mas = np.array([ma5, ma10, ma20, ma60])
        mas_range = (np.max(mas) - np.min(mas)) / np.mean(mas)
        if mas_range < 0.03:
            ma_score += 4
        # 价格站上 MA20
        if closes[-1] > ma20 and closes[-2] <= ma20:
            ma_score += 3
    # 反向：完美多头排列 = 趋势成熟/尾声，N字回调需要均线收敛而非发散
    factors["ma_alignment_score"] = _clip_score(-ma_score, -12, 10)

    # ── 因子15: BOLL 挤压 (boll_squeeze_score) ──
    boll_mid, boll_upper, boll_lower, boll_bw = _calc_boll(closes)
    boll_score = 0
    if n >= 20 and not np.isnan(boll_bw[-1]):
        bw_now = float(boll_bw[-1])
        # 带宽低点 = 波动挤压
        bw_20 = boll_bw[-20:]
        bw_valid = bw_20[~np.isnan(bw_20)]
        if len(bw_valid) >= 10:
            bw_pct_rank = np.sum(bw_valid < bw_now) / len(bw_valid)
            if bw_pct_rank < 0.2:
                boll_score += 5
            elif bw_pct_rank < 0.4:
                boll_score += 2
        # %B = (close - lower) / (upper - lower)
        if not np.isnan(boll_upper[-1]) and not np.isnan(boll_lower[-1]):
            bp = boll_upper[-1] - boll_lower[-1]
            if bp > 0:
                pct_b = (closes[-1] - boll_lower[-1]) / bp
                if 0.1 <= pct_b <= 0.3:
                    boll_score += 3
                elif pct_b > 0.8:
                    boll_score -= 3
    # 反向：N字已形成=挤压已结束，此时低带宽=动能衰竭，需要带宽扩张确认趋势
    factors["boll_squeeze_score"] = _clip_score(-boll_score, -8, 8)

    # ── 因子16: KDJ 超卖反弹 (kdj_oversold_score) ──
    kdj_k, kdj_d, kdj_j = _calc_kdj(highs, lows, closes)
    kdj_score = 0
    if n >= 9 and not np.isnan(kdj_j[-1]):
        j_now = float(kdj_j[-1])
        j_prev = float(kdj_j[-2]) if not np.isnan(kdj_j[-2]) else j_now
        if j_now < 0:
            kdj_score += 6
        elif j_now < 20:
            kdj_score += 3
        if j_now > 80:
            kdj_score -= 5
        # K/D 金叉
        if not np.isnan(kdj_k[-1]) and not np.isnan(kdj_d[-1]):
            if float(kdj_k[-2]) <= float(kdj_d[-2]) and float(kdj_k[-1]) > float(kdj_d[-1]):
                kdj_score += 5
            elif float(kdj_k[-3]) <= float(kdj_d[-3]) and float(kdj_k[-1]) > float(kdj_d[-1]):
                kdj_score += 3
        # J 线拐头
        if j_now > j_prev and j_prev < 20:
            kdj_score += 4
    factors["kdj_oversold_score"] = _clip_score(kdj_score, -8, 10)

    # ── 因子17: MFI 资金流量 (mfi_score) ──
    mfi_arr = _calc_mfi(highs, lows, closes, vols)
    mfi_score = 0
    if n >= 15 and not np.isnan(mfi_arr[-1]):
        mfi_now = float(mfi_arr[-1])
        mfi_prev = float(mfi_arr[-2]) if not np.isnan(mfi_arr[-2]) else mfi_now
        if mfi_now < 20:
            mfi_score += 6
        elif mfi_now < 30:
            mfi_score += 3
        elif mfi_now > 80:
            mfi_score -= 5
        # MFI 金叉 50
        if mfi_prev < 50 <= mfi_now:
            mfi_score += 3
        # MFI 底背离
        if trough_idx > 20:
            price_low_seg = lows[trough_idx - 5:trough_idx + 1]
            mfi_seg = mfi_arr[trough_idx - 5:trough_idx + 1]
            valid_seg = ~np.isnan(mfi_seg)
            if valid_seg.sum() >= 3:
                price_min_i = int(np.argmin(price_low_seg[valid_seg]))
                mfi_at_min = mfi_seg[valid_seg][price_min_i] if price_min_i < len(mfi_seg[valid_seg]) else mfi_now
                mfi_early = float(np.nanmin(mfi_seg[valid_seg][:max(1, len(mfi_seg[valid_seg]) // 2)]))
                if mfi_at_min > mfi_early + 5:
                    mfi_score += 5
    factors["mfi_score"] = _clip_score(-mfi_score, -10, 8)

    # ── 因子18: 影线质量 (shadow_quality_score) ──
    # 回调/确认日的长下影线 = 买方在支撑位主动接盘，机构托盘痕迹
    shadow_score = 0
    if peak_idx < trough_idx:
        pb_start = peak_idx + 1
        pb_end = min(trough_idx + 1, n)
        shadow_ratios = []
        for i in range(pb_start, pb_end):
            body_bottom = min(opens[i], closes[i])
            bar_range = highs[i] - lows[i]
            if bar_range > 0 and closes[i] > 0:
                lower_shadow = (body_bottom - lows[i]) / closes[i]
                shadow_ratios.append(lower_shadow)
        if shadow_ratios:
            strong_shadows = sum(1 for r in shadow_ratios if r > 0.008)  # >0.8% of price
            total_bars = len(shadow_ratios)
            strong_pct = strong_shadows / total_bars
            avg_shadow = float(np.mean(shadow_ratios))
            if strong_pct >= 0.6:
                shadow_score += 6  # 多数回调日都有托盘
            elif strong_pct >= 0.3:
                shadow_score += 3
            elif strong_pct < 0.15:
                shadow_score -= 4  # 几乎没有下影线 = 无支撑
            if avg_shadow > 0.012:
                shadow_score += 3  # 平均下影线超过 1.2% → 强力托盘
            # 确认日（最后一根）下影线检查
            last_shadow = shadow_ratios[-1]
            if last_shadow > 0.015:
                shadow_score += 4  # 确认日长下影 → 强支撑确认
            elif last_shadow > 0.008:
                shadow_score += 2
            elif last_shadow < 0.003 and closes[-1] < opens[-1]:
                shadow_score -= 5  # 确认日无下影+收阴 = 无支撑
    factors["shadow_quality_score"] = _clip_score(-shadow_score, -10, 8)

    # ── 因子19: 回调速度 (pullback_speed_score) ──
    # 快速浅回调 = 强趋势中的短暂休整，慢速深回调 = 趋势可能反转
    speed_score = 0
    if start_idx < peak_idx < trough_idx and n > trough_idx:
        start_price = closes[start_idx]
        peak_price = closes[peak_idx]
        trough_price = closes[trough_idx]
        if peak_price > start_price and trough_idx > peak_idx:
            retrace_pct = (peak_price - trough_price) / (peak_price - start_price)
            retrace_days = trough_idx - peak_idx
            if retrace_days >= 1:
                speed = retrace_pct / retrace_days  # 每天回调幅度
                if speed > 0.04:
                    speed_score += 6  # 快速回调（每天>4%回调比例）→ 强趋势
                elif speed > 0.02:
                    speed_score += 3
                elif speed < 0.01:
                    speed_score -= 4  # 慢速回调 → 动能不足
            # 浅回调加分
            if retrace_pct < 0.382:
                speed_score += 3
            elif retrace_pct > 0.618:
                speed_score -= 4  # 深度回调 → 趋势可能反转
            if retrace_days <= 3:
                speed_score += 3  # 3天内结束回调 → 高效
            elif retrace_days > 8:
                speed_score -= 3  # 超过8天 → 时间太长
    factors["pullback_speed_score"] = _clip_score(speed_score, -8, 10)

    # ── 因子20: 日内反转强度 (intraday_reversal_score) ──
    # 今天开盘低走、探底、收高 = 支撑位买方反攻
    intraday_score = 0
    today_open = opens[-1]
    today_low = lows[-1]
    today_close = closes[-1]
    today_high = highs[-1]
    today_range = today_high - today_low
    if today_range > 0 and today_open > 0:
        # 收盘位置 (0=最低, 1=最高)
        close_pos_today = (today_close - today_low) / today_range
        if close_pos_today >= 0.80:
            intraday_score += 5  # 收在最高20%区间 → 强反转
        elif close_pos_today >= 0.65:
            intraday_score += 2
        elif close_pos_today < 0.35:
            intraday_score -= 5  # 收在低位 → 支撑被砸穿
        # 下影线: 开盘后先跌再涨
        lower_shadow = (min(today_open, today_close) - today_low) / today_close
        if lower_shadow > 0.02:
            intraday_score += 4  # 长下影 → 支撑位接盘
        elif lower_shadow > 0.01:
            intraday_score += 2
        # 对比昨日收盘: 低开高走 = 更强
        if n >= 2 and closes[-2] > 0:
            gap = today_open / closes[-2] - 1
            if gap < -0.01 and today_close > today_open:
                intraday_score += 3  # 低开>1%但收阳 → 支撑确认
            elif gap > 0.02:
                intraday_score -= 2  # 高开 → 支撑没测到
    factors["intraday_reversal_score"] = _clip_score(intraday_score, -8, 12)

    # ── 因子21: 量能衰竭 (volume_climax_score) ──
    # 回调过程中成交量先放大后急剧萎缩 = 卖压释放完毕
    climax_score = 0
    if start_idx < peak_idx < trough_idx and n > trough_idx:
        pb_vols = vols[peak_idx + 1:trough_idx + 1]
        if len(pb_vols) >= 3:
            pb_vol_max = float(np.max(pb_vols))
            pb_vol_max_i = int(np.argmax(pb_vols))
            pb_vol_now = vols[-1]
            # 量能峰值已过（峰值不在最后2天）
            if pb_vol_max_i < len(pb_vols) - 2:
                decay = pb_vol_now / max(pb_vol_max, eps)
                if decay < 0.4:
                    climax_score += 6  # 量缩到峰值的40%以下 → 卖压枯竭
                elif decay < 0.55:
                    climax_score += 3
                elif decay > 0.85:
                    climax_score -= 3  # 量没缩 → 卖压还在
            # 峰值当天如果是天量（>60日均量2.5倍）→ 经典放量见底
            avg_vol_60 = _safe_mean(vols[-60:]) if n >= 60 else _safe_mean(vols)
            if pb_vol_max > avg_vol_60 * 2.5 and pb_vol_max_i < len(pb_vols) - 1:
                climax_score += 3
    factors["volume_climax_score"] = _clip_score(climax_score, -6, 10)

    # ── 因子22: 行业相对强度 (sector_relative_score) ──
    # 同行业内，个股是否强于板块均值 → N字龙头 vs 跟风
    sector_score = 0
    sector_map = live_data.get("_sector_map", {}) if live_data else {}
    # Note: _calc_trading_factors has no 'code' param, so we check via live_data injection
    # The caller should set live_data['_stock_ret20'] and live_data['_sector_avg_ret20']
    stock_ret20 = live_data.get("_stock_ret20") if live_data else None
    sector_avg_ret20 = live_data.get("_sector_avg_ret20") if live_data else None
    if stock_ret20 is not None and sector_avg_ret20 is not None and sector_avg_ret20 != 0:
        relative = stock_ret20 - sector_avg_ret20
        if relative > 0.10:
            sector_score += 6  # 大幅跑赢行业 → 龙头
        elif relative > 0.03:
            sector_score += 3
        elif relative > 0:
            sector_score += 1
        elif relative < -0.08:
            sector_score -= 5  # 跑输行业 → 弱势股
        elif relative < -0.03:
            sector_score -= 2
    else:
        # Fallback: use the stock's own ret20 vs a neutral baseline
        ret20_val = closes[-1] / closes[-21] - 1 if n >= 21 and closes[-21] > 0 else 0
        if ret20_val > 0.15:
            sector_score += 2
        elif ret20_val < -0.08:
            sector_score -= 2
    factors["sector_relative_score"] = _clip_score(sector_score, -8, 10)

    # ── 因子23: ADX 趋势强度 (adx_trend_score) ──
    adx_arr = _calc_adx(highs, lows, closes)
    adx_score = 0
    if len(adx_arr) and np.isfinite(adx_arr[-1]):
        adx_now = float(adx_arr[-1])
        if 18 <= adx_now <= 38:
            adx_score += 6
        elif 38 < adx_now <= 55:
            adx_score += 2
        elif adx_now < 12:
            adx_score -= 6
        elif adx_now > 60:
            adx_score -= 4
    factors["adx_trend_score"] = _clip_score(adx_score, -8, 8)

    # ── 因子24: OBV 吸筹 (obv_accumulation_score) ──
    obv_arr = _calc_obv(closes, vols)
    obv_score = 0
    if len(obv_arr) >= 20:
        obv5 = obv_arr[-1] - obv_arr[-6]
        obv20 = obv_arr[-1] - obv_arr[-21]
        if obv20 > 0:
            obv_score += 4
        elif obv20 < 0:
            obv_score -= 3
        if obv5 > 0:
            obv_score += 4
        elif obv5 < 0:
            obv_score -= 4
    factors["obv_accumulation_score"] = _clip_score(obv_score, -8, 8)

    # ── 因子25: CMF 资金流 (cmf_score) ──
    cmf_arr = _calc_cmf(highs, lows, closes, vols)
    cmf_score = 0
    if len(cmf_arr) and np.isfinite(cmf_arr[-1]):
        cmf_now = float(cmf_arr[-1])
        if cmf_now > 0.10:
            cmf_score += 6
        elif cmf_now > 0:
            cmf_score += 3
        elif cmf_now < -0.12:
            cmf_score -= 6
        elif cmf_now < -0.03:
            cmf_score -= 3
    factors["cmf_score"] = _clip_score(cmf_score, -8, 8)

    # ── 因子26: 缺口支撑 (gap_support_score) ──
    gap_score = 0
    for i in range(max(1, n - 15), n):
        prev_high = highs[i - 1]
        prev_low = lows[i - 1]
        if lows[i] > prev_high * 1.01:
            gap_floor = prev_high
            if closes[-1] >= gap_floor * 0.98:
                gap_score += 4
        elif highs[i] < prev_low * 0.99:
            gap_score -= 3
    factors["gap_support_score"] = _clip_score(gap_score, -6, 8)

    # ── 因子27: XGBoost ML 预测胜率 (ml_confidence) ──
    ml_prob = _predict_ml_win_prob(factors)
    factors["ml_confidence"] = round(ml_prob, 3)
    if ml_prob >= 0.65:
        ml_score = 15
    elif ml_prob >= 0.55:
        ml_score = 10
    elif ml_prob >= 0.45:
        ml_score = 5
    elif ml_prob >= 0.35:
        ml_score = 0
    elif ml_prob >= 0.25:
        ml_score = -8
    else:
        ml_score = -15
    factors["ml_confidence_score"] = ml_score

    # ── 因子28: K线序列模型胜率 (sequence_confidence) ──
    seq_prob = _predict_sequence_win_prob(opens, highs, lows, closes, vols)
    factors["sequence_confidence"] = round(seq_prob, 3)
    if seq_prob >= 0.65:
        seq_score = 8
    elif seq_prob >= 0.58:
        seq_score = 5
    elif seq_prob >= 0.52:
        seq_score = 2
    elif seq_prob <= 0.35:
        seq_score = -6
    elif seq_prob <= 0.42:
        seq_score = -3
    else:
        seq_score = 0
    factors["sequence_score"] = seq_score

    factor_score = (
        factors["pullback_volume_score"]
        + factors["turnover_crowding_score"]
        + factors["relative_strength_score"]
        + factors["volatility_contraction_score"]
        + factors["support_reclaim_score"]
        + factors["close_position_score"]
        + round(factors["limit_up_followthrough_score"] * 0.5)
        + factors["theme_heat_score"]
        + factors["amount_quality_score"]
        + factors["market_regime_score"]
        + round(factors["northbound_flow_score"] * 0.6)
        + round(factors["rsi_divergence_score"] * 0.4)
        + round(factors["macd_signal_score"] * 0.3)
        + round(factors["ma_alignment_score"] * 0.3)
        + round(factors["boll_squeeze_score"] * 0.3)
        + round(factors["kdj_oversold_score"] * 0.4)
        + round(factors["mfi_score"] * 0.3)
        + round(factors["shadow_quality_score"] * 0.8)
        + round(factors["pullback_speed_score"] * 1.0)
        + round(factors["intraday_reversal_score"] * 0.6)
        + round(factors["volume_climax_score"] * 0.5)
        + round(factors["sector_relative_score"] * 0.4)
        + round(factors["adx_trend_score"] * 0.5)
        + round(factors["obv_accumulation_score"] * 0.5)
        + round(factors["cmf_score"] * 0.5)
        + round(factors["gap_support_score"] * 0.4)
        + round(factors.get("ml_confidence_score", 0) * 0.8)
        + round(factors.get("sequence_score", 0) * 0.5)
    )
    factors["factor_score"] = _clip_score(factor_score, -95, 118)
    factors["factor_details"] = {
        "support_shrink": round(support_shrink, 3),
        "pullback_vs_rise_vol": round(pullback_vs_rise, 3),
        "recent3_to_60_vol": round(recent3_to_60, 3),
        "recent5_to_20_vol": round(recent5_to_20, 3),
        "ret20": round(ret20, 3),
        "ret60": round(ret60, 3),
        "dist_from_high60": round(dist_from_high60, 3),
        "atr5_to_20": round(atr5_to_20, 3),
        "range3_to_20": round(range3_to_20, 3),
        "reclaim_from_support": round(reclaim_from_support, 3),
        "recent_reclaim": round(recent_reclaim, 3),
        "support_pierce": round(support_pierce, 3),
        "close_pos": round(close_pos, 3),
        "close_pos_3": round(close_pos_3, 3),
        "upper_shadow": round(upper_shadow, 3),
        "limit_days_ago": limit_days_ago,
        "limit_next_ret": round(limit_next_ret, 3),
        "limit_post_drawdown": round(limit_post_drawdown, 3),
        "amp_expansion": round(amp_expansion, 3),
        "limit_cnt_30d": limit_cnt,
        "theme_heat_score": factors["theme_heat_score"],
        "amount_quality_score": factors["amount_quality_score"],
        "ma60_slope_10d_pct": round(ma60_slope_10d * 100, 2) if n >= 70 else 0,
        "atr20_pct": round(atr20_pct * 100, 2) if n >= 20 else 0,
        "market_regime_score": factors["market_regime_score"],
        "northbound_flow_score": factors["northbound_flow_score"],
        "rsi_divergence_score": factors["rsi_divergence_score"],
        "macd_signal_score": factors["macd_signal_score"],
        "ma_alignment_score": factors["ma_alignment_score"],
        "boll_squeeze_score": factors["boll_squeeze_score"],
        "kdj_oversold_score": factors["kdj_oversold_score"],
        "mfi_score": factors["mfi_score"],
        "shadow_quality_score": factors["shadow_quality_score"],
        "pullback_speed_score": factors["pullback_speed_score"],
        "intraday_reversal_score": factors["intraday_reversal_score"],
        "volume_climax_score": factors["volume_climax_score"],
        "sector_relative_score": factors["sector_relative_score"],
        "adx_trend_score": factors["adx_trend_score"],
        "obv_accumulation_score": factors["obv_accumulation_score"],
        "cmf_score": factors["cmf_score"],
        "gap_support_score": factors["gap_support_score"],
        "sequence_confidence": factors["sequence_confidence"],
        "sequence_score": factors["sequence_score"],
        "rsi_now": round(float(_calc_rsi(closes, 14)[-1]), 1) if n >= 15 else 0,
        "macd_dif": round(float(dif[-1]), 4) if n >= 30 and not np.isnan(dif[-1]) else 0,
        "kdj_j": round(float(kdj_j[-1]), 1) if n >= 9 and not np.isnan(kdj_j[-1]) else 0,
        "mfi_now": round(float(mfi_arr[-1]), 1) if n >= 15 and not np.isnan(mfi_arr[-1]) else 0,
    }
    return factors


def _factor_strength_bonus(factor_score: int) -> int:
    """把原始因子分映射为强度加减分。

    22因子总分范围约 -110~140。
    旧版本对超高分再次扣分，容易把本来已经通过多因子共振的强信号打回去。
    这里改成更平滑、单调的奖励曲线。
    """
    if factor_score < 0:
        return -15
    if factor_score < 10:
        return -6
    if factor_score < 25:
        return 4
    if factor_score <= 50:
        return 9
    if factor_score <= 80:
        return 12
    return 10


# ── XGBoost ML 模型预测 ──

ML_FEATURE_COLS = [
    "pullback_volume_score", "turnover_crowding_score", "relative_strength_score",
    "volatility_contraction_score", "support_reclaim_score", "close_position_score",
    "limit_up_followthrough_score", "theme_heat_score", "amount_quality_score",
    "market_regime_score", "northbound_flow_score", "rsi_divergence_score",
    "macd_signal_score", "ma_alignment_score", "boll_squeeze_score",
    "kdj_oversold_score", "mfi_score", "shadow_quality_score", "pullback_speed_score",
    "intraday_reversal_score", "volume_climax_score", "sector_relative_score",
    "adx_trend_score", "obv_accumulation_score", "cmf_score", "gap_support_score",
]


def _predict_ml_win_prob(factors: dict) -> float:
    """使用 ML/Ensemble 模型预测信号胜率。模型不存在时返回 0.5。"""
    try:
        model_data = _load_ml_model()
        if model_data is None:
            return 0.5
        prob, _ = _predict_ml_signal(model_data, factors, strength=0)
        return prob
    except Exception:
        return 0.5


def _get_sequence_model():
    global _SEQUENCE_MODEL_CACHE
    if _SEQUENCE_MODEL_CACHE is not None:
        return _SEQUENCE_MODEL_CACHE
    for path in SEQUENCE_MODEL_CANDIDATES:
        if os.path.exists(path):
            try:
                _SEQUENCE_MODEL_CACHE = _load_sequence_model(path)
                return _SEQUENCE_MODEL_CACHE
            except Exception:
                logger.debug("load sequence model failed: %s", path, exc_info=True)
    _SEQUENCE_MODEL_CACHE = False
    return None


def _predict_sequence_win_prob(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    vols: np.ndarray,
) -> float:
    model = _get_sequence_model()
    if not model:
        return 0.5
    try:
        window = int(model.get("config", {}).get("window", 30))
        seq = build_kline_tensor(opens, highs, lows, closes, vols, end_idx=len(closes) - 1, window=window)
        if seq is None:
            return 0.5
        return float(_predict_sequence_prob(model, seq))
    except Exception:
        logger.debug("sequence inference failed", exc_info=True)
        return 0.5


def _passes_high_win_filter(signal_data: dict, params: NPatternParams) -> bool:
    """高胜率模式：只保留收盘强、波动缩、空间足的回踩信号。"""
    if not params.high_win_mode:
        return True
    if signal_data.get('strength', 0) < params.high_win_min_strength:
        return False
    if signal_data.get('close_position_score', 0) < params.high_win_min_close_position_score:
        return False
    if signal_data.get('volatility_contraction_score', 0) < params.high_win_min_volatility_contraction_score:
        return False
    if signal_data.get('sequence_confidence', 0.5) < params.high_win_min_sequence_confidence:
        return False
    if signal_data.get('factor_score', 0) < params.high_win_min_factor_score:
        return False
    if signal_data.get('rr_ratio', 0) < params.high_win_min_rr:
        return False

    resistance_pct = signal_data.get('entry_to_resistance_pct', 0)
    nearest_resistance = signal_data.get('nearest_resistance')
    if nearest_resistance and 0 < resistance_pct < params.high_win_min_resistance_pct:
        return False

    model_data = _load_ml_model()
    if params.high_win_require_ml and model_data is not None:
        model_threshold = float(model_data.get('threshold', 0.5))
        min_ml_conf = max(params.high_win_min_ml_confidence, model_threshold)
        if signal_data.get('ml_confidence', 0.5) < min_ml_conf:
            return False

    # 输家清单：因子区分度显示4个反效特征，≥3个命中则排除
    loser_flags = 0
    if signal_data.get('sector_relative_score', 0) > 5:
        loser_flags += 1  # 行业相对过强 → 均值回归风险
    if signal_data.get('relative_strength_score', 0) > 15:
        loser_flags += 1  # 相对强度过高 → 追高
    if signal_data.get('kdj_oversold_score', 0) > 8:
        loser_flags += 1  # KDJ不够超卖 → 入场成本偏高
    if signal_data.get('intraday_reversal_score', 0) < -3:
        loser_flags += 1  # 日内反转弱 → 买方力度不足
    if loser_flags >= 3:
        return False

    return True


def _calc_strength(signal_data: dict) -> int:
    """计算信号强度 0-100+"""
    strength = 40
    if signal_data.get('stab_ok'):
        strength += 12  # 缩量+下影确认，但赢输差异小，降权
    elif signal_data.get('has_vol_shrink') or signal_data.get('has_shadow'):
        strength += 5
    # 多头排列：新鲜多头(1-2天)=加分，持续多头(≥5天)=高位追入扣分
    ma_bullish = signal_data.get('ma_bullish', False)
    bullish_days = signal_data.get('bullish_days', 0)
    if ma_bullish:
        if bullish_days >= 5:
            strength -= 10  # 持续多头 → 高位追入
        elif bullish_days >= 3:
            strength += 5   # 趋势确立
        elif bullish_days >= 1:
            strength += 12  # 刚形成多头 → 最佳时机
    else:
        strength -= 3   # 非多头也可以做(回调入场)，轻罚
    if signal_data.get('has_vol_shrink'):
        strength += 5
    if signal_data.get('has_shadow'):
        strength += 5
    if signal_data.get('ma_fib_ok'):
        strength += 8

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
        if cnt >= 5:
            strength += 22  # 极活股性
        elif cnt >= 3:
            strength += 18  # 多次涨停，股性很活
        elif cnt >= 2:
            strength += 14  # 2次涨停
        else:
            strength += 8   # 1次涨停

    # 罚分：MA10 支撑质量（分级罚分）
    close_vs_ma10 = signal_data.get('close_vs_ma10_pct', 0)
    if close_vs_ma10 < -1.5:
        strength -= 18  # 跌破MA10>1.5% = 支撑失败(重罚)
    elif close_vs_ma10 < -0.5:
        strength -= 10  # 跌破MA10 0.5-1.5% = 中度罚
    elif close_vs_ma10 < 0:
        strength -= 5   # 跌破MA10<0.5% = 测试支撑，轻罚
    elif signal_data.get('ma10_broken_intraday'):
        strength -= 4   # 日内触及但收在上方 = 支撑确认，轻罚

    # 罚分：深回调 (费波0.618) → 支撑不可靠
    if signal_data.get('fib_level', 0) >= 0.60:
        strength -= 12

    # 罚分：暴涨后回调 → 第一波>40%支撑弱
    if signal_data.get('first_rise_pct', 0) > 40:
        strength -= 8

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

    strength += _factor_strength_bonus(signal_data.get('factor_score', 0))

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

    # 4. Fib 扩展位 (基于第一波幅度，从起点推算)
    move = first_high - first_low
    if move > 0:
        for ext_pct, label in [(1.272, 'Fib127.2%'), (1.618, 'Fib161.8%')]:
            level = first_low + move * ext_pct
            if level > current_price * 0.90:
                raw.append((label, round(level, 2)))

        # 4b. 二波目标位 (从回调低点/小N支撑起算)
        # 第一波走完后回调到小N，第二波从 retrace_low 出发
        # 目标位 = retrace_low + move × ratio
        # 北投科技案例: 3.91→5.69 move=1.78, retrace_low=5.35
        #   二波38.2%=6.03, 二波50%=6.24, 二波61.8%=6.45, 二波等幅=7.13
        if retrace_low is not None:
            for ratio, label in [(0.382, '二波38%'), (0.50, '二波50%'), (0.618, '二波62%'), (1.0, '二波等幅')]:
                level = retrace_low + move * ratio
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
    amounts: np.ndarray = None,
    live_data: dict = None,
) -> list[dict]:
    """扫描K线中的 N 字买点

    核心原则（大神策略）：
    1. 上涨 → 回调找支撑 → 支撑住 → 再拉一波
    2. 支撑位必须"撑住"，跌破再拉回的不算
    3. 费波位 + MA9/MA10 共振才是有效支撑
    """
    n = len(closes)
    peaks, troughs = find_extrema(highs, lows)

    limit_up_count = _check_limit_up(closes, lookback=60)
    big_yang_count = _check_limit_up(closes, lookback=60, threshold=0.07)
    # 硬过滤：近60日至少1次涨停(9.5%+) 或 2次大阳线(7%+)
    if limit_up_count < 1 and big_yang_count < 2:
        return []
    has_limit_up = limit_up_count >= 1  # 有涨停 = 涨停基因更纯
    ma_bullish, ma9_gt_ma10, bullish_days, ma_consistent, ma9, ma10, ma20 = _check_ma_bullish(closes)

    last_close = closes[-1]

    # === 硬过滤：MA60 趋势方向 ===
    if n >= 65:
        ma60_now = float(np.mean(closes[-60:]))
        ma60_5ago = float(np.mean(closes[-65:-5]))
        if ma60_5ago > 0:
            ma60_slope = (ma60_now - ma60_5ago) / ma60_5ago
            if ma60_slope < -0.01:   # MA60 下降 >1% → 下跌趋势不参与
                return []
            if ma60_slope > 0.06:    # MA60 10日升幅 >6% → 抛物线，均值回归风险极大
                return []             # 三孚股份案例: 6天4涨停后 MA60 斜率+12.5%，必然暴力回调

    # === 硬过滤：MA60 < MA120 → 中期空头排列 ===
    if n >= 125:
        ma120_now = float(np.mean(closes[-120:]))
        if ma60_now < ma120_now:     # 昭衍新药案例: MA60(35.66) < MA120(35.75)，中期趋势已转空
            return []

    # === 硬过滤：收盘价跌破 MA10 >2% → 支撑已失效，整只跳过 ===
    # 轻微跌破(<2%)不硬排除，改在强度中罚分
    if ma10 is not None and last_close < ma10 * 0.98:
        return []

    # === MA 多头排列不持续 → 不硬过滤，改在强度中罚分 ===

    # === 硬过滤：近2日有涨停 → 涨停后主力出货窗口，不接飞刀 ===
    for i in range(1, min(3, n)):
        if closes[-i - 1] > 0:
            if (closes[-i] - closes[-i - 1]) / closes[-i - 1] >= 0.095:
                return []  # 昭衍新药(5/21涨停)、三孚股份(5/20涨停)、滨化股份(5/18涨停)

    # === 硬过滤：入场前已连跌 ≥2 天 → 砸盘进行中，不是企稳 ===
    consecutive_down = 0
    for i in range(1, min(6, n)):
        if closes[-i] < closes[-i - 1]:
            consecutive_down += 1
        else:
            break
    if consecutive_down >= 3:
        return []  # 连跌3天以上才排除，2天回调属正常

    # === 硬过滤：今天涨停 → 回调结束已启动，不是买点 ===
    if n >= 2 and closes[-2] > 0:
        today_chg = (closes[-1] - closes[-2]) / closes[-2]
        if today_chg >= 0.095:
            return []

    # === 硬过滤：今天收盘跌破MA10 >2% 或 跌破MA9 >5% → 支撑失败，排除 ===
    if ma10 is not None and closes[-1] < ma10 * 0.98:
        return []
    if ma9 is not None and closes[-1] < ma9 * 0.93:
        return []

    # === 检测近期是否跌破过 MA10 ===
    # 收盘跌破 = 支撑失败(严重)；仅日内触及但收在上方 = 支撑确认(轻微)
    recent_lows = lows[-3:] if n >= 3 else lows
    recent_closes = closes[-3:] if n >= 3 else closes
    ma10_broken_close = (ma10 is not None and any(c < ma10 for c in recent_closes))
    ma10_touched_intraday = (ma10 is not None and any(low < ma10 for low in recent_lows))

    # === 软罚：近3日收盘跌破过 MA10 → 在强度中重罚，但不硬排除 ===

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
            fib_dist_max = 0.15 if has_limit_up else 0.08
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
            # 反弹>5% = 支撑已测+强力拉起，买点已过
            # 反弹<5% = 只是蹭了一下支撑，明天仍可能是买点
            if lows[-1] <= entry_price * 1.005:
                bounce_from_low = (closes[-1] - lows[-1]) / lows[-1]
                if bounce_from_low > 0.05:
                    continue

            # 止损 = 入场价下方
            if params.stop_atr_mult > 0:
                atr20 = float(np.mean(highs[-20:] - lows[-20:]))
                stop_loss = round(entry_price - atr20 * params.stop_atr_mult, 2)
            else:
                stop_loss = round(entry_price * (1 - params.stop_loss_pct), 2)

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
            factor_data = (
                _calc_trading_factors(opens, highs, lows, closes, vols, ta, best_p, tb, amounts, live_data)
                if params.enable_trading_factors else {}
            )

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
                'close_vs_ma10_pct': round((closes[-1] / ma10 - 1) * 100, 1) if ma10 is not None else 0,
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
                'pullback_volume_score': factor_data.get('pullback_volume_score', 0),
                'turnover_crowding_score': factor_data.get('turnover_crowding_score', 0),
                'relative_strength_score': factor_data.get('relative_strength_score', 0),
                'volatility_contraction_score': factor_data.get('volatility_contraction_score', 0),
                'support_reclaim_score': factor_data.get('support_reclaim_score', 0),
                'close_position_score': factor_data.get('close_position_score', 0),
                'limit_up_followthrough_score': factor_data.get('limit_up_followthrough_score', 0),
                'theme_heat_score': factor_data.get('theme_heat_score', 0),
                'amount_quality_score': factor_data.get('amount_quality_score', 0),
                'market_regime_score': factor_data.get('market_regime_score', 0),
                'northbound_flow_score': factor_data.get('northbound_flow_score', 0),
                'rsi_divergence_score': factor_data.get('rsi_divergence_score', 0),
                'macd_signal_score': factor_data.get('macd_signal_score', 0),
                'ma_alignment_score': factor_data.get('ma_alignment_score', 0),
                'boll_squeeze_score': factor_data.get('boll_squeeze_score', 0),
                'kdj_oversold_score': factor_data.get('kdj_oversold_score', 0),
                'mfi_score': factor_data.get('mfi_score', 0),
                'shadow_quality_score': factor_data.get('shadow_quality_score', 0),
                'pullback_speed_score': factor_data.get('pullback_speed_score', 0),
                'intraday_reversal_score': factor_data.get('intraday_reversal_score', 0),
                'volume_climax_score': factor_data.get('volume_climax_score', 0),
                'sector_relative_score': factor_data.get('sector_relative_score', 0),
                'adx_trend_score': factor_data.get('adx_trend_score', 0),
                'obv_accumulation_score': factor_data.get('obv_accumulation_score', 0),
                'cmf_score': factor_data.get('cmf_score', 0),
                'gap_support_score': factor_data.get('gap_support_score', 0),
                'ml_confidence': factor_data.get('ml_confidence', 0.0),
                'ml_confidence_score': factor_data.get('ml_confidence_score', 0),
                'sequence_confidence': factor_data.get('sequence_confidence', 0.0),
                'sequence_score': factor_data.get('sequence_score', 0),
                'factor_score': factor_data.get('factor_score', 0),
                'factor_details': factor_data.get('factor_details', {}),
            }
            if not _passes_high_win_filter(sig_data, params):
                continue
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
                        if params.stop_atr_mult > 0:
                            atr20 = float(np.mean(highs[-20:] - lows[-20:]))
                            stop_loss = round(entry_price - atr20 * params.stop_atr_mult, 2)
                        else:
                            stop_loss = round(entry_price * (1 - params.stop_loss_pct), 2)
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
                        factor_data = (
                            _calc_trading_factors(opens, highs, lows, closes, vols, ta, best_p, tb, amounts, live_data)
                            if params.enable_trading_factors else {}
                        )

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
                            'close_vs_ma10_pct': round((closes[-1] / ma10 - 1) * 100, 1) if ma10 is not None else 0,
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
                            'pullback_volume_score': factor_data.get('pullback_volume_score', 0),
                            'turnover_crowding_score': factor_data.get('turnover_crowding_score', 0),
                            'relative_strength_score': factor_data.get('relative_strength_score', 0),
                            'volatility_contraction_score': factor_data.get('volatility_contraction_score', 0),
                            'support_reclaim_score': factor_data.get('support_reclaim_score', 0),
                            'close_position_score': factor_data.get('close_position_score', 0),
                            'limit_up_followthrough_score': factor_data.get('limit_up_followthrough_score', 0),
                            'theme_heat_score': factor_data.get('theme_heat_score', 0),
                            'amount_quality_score': factor_data.get('amount_quality_score', 0),
                            'market_regime_score': factor_data.get('market_regime_score', 0),
                            'northbound_flow_score': factor_data.get('northbound_flow_score', 0),
                            'rsi_divergence_score': factor_data.get('rsi_divergence_score', 0),
                            'macd_signal_score': factor_data.get('macd_signal_score', 0),
                            'ma_alignment_score': factor_data.get('ma_alignment_score', 0),
                            'boll_squeeze_score': factor_data.get('boll_squeeze_score', 0),
                            'kdj_oversold_score': factor_data.get('kdj_oversold_score', 0),
                            'mfi_score': factor_data.get('mfi_score', 0),
                            'shadow_quality_score': factor_data.get('shadow_quality_score', 0),
                            'pullback_speed_score': factor_data.get('pullback_speed_score', 0),
                            'intraday_reversal_score': factor_data.get('intraday_reversal_score', 0),
                            'volume_climax_score': factor_data.get('volume_climax_score', 0),
                            'sector_relative_score': factor_data.get('sector_relative_score', 0),
                            'adx_trend_score': factor_data.get('adx_trend_score', 0),
                            'obv_accumulation_score': factor_data.get('obv_accumulation_score', 0),
                            'cmf_score': factor_data.get('cmf_score', 0),
                            'gap_support_score': factor_data.get('gap_support_score', 0),
                            'ml_confidence': factor_data.get('ml_confidence', 0.0),
                            'ml_confidence_score': factor_data.get('ml_confidence_score', 0),
                            'sequence_confidence': factor_data.get('sequence_confidence', 0.0),
                            'sequence_score': factor_data.get('sequence_score', 0),
                            'factor_score': factor_data.get('factor_score', 0),
                            'factor_details': factor_data.get('factor_details', {}),
                        }
                        if not _passes_high_win_filter(big_sig, params):
                            continue
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
    live_data: dict = None,
) -> list[NSignal]:
    """扫描单只股票的 N 字信号"""
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    vols = df['volume'].values
    amounts_arr = df['amount'].values if 'amount' in df.columns else None

    # 注入该股的实时因子数据
    stock_live = dict(live_data) if live_data else {}
    if live_data:
        hot_set = live_data.get("hot_code_set", set())
        stock_live["is_hot"] = code in hot_set
        # 查找行业排名
        sector_map = live_data.get("_sector_map", {})
        industry = sector_map.get(code, "")
        sector_rank_map = live_data.get("sector_rank", {})
        stock_live["sector_rank_pct"] = sector_rank_map.get(industry, 1.0) if industry else 1.0

    raw_signals = find_n_signals(opens, highs, lows, closes, vols, params, market_pct, amounts=amounts_arr, live_data=stock_live)

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
            pullback_volume_score=s.get('pullback_volume_score', 0),
            turnover_crowding_score=s.get('turnover_crowding_score', 0),
            relative_strength_score=s.get('relative_strength_score', 0),
            volatility_contraction_score=s.get('volatility_contraction_score', 0),
            support_reclaim_score=s.get('support_reclaim_score', 0),
            close_position_score=s.get('close_position_score', 0),
            limit_up_followthrough_score=s.get('limit_up_followthrough_score', 0),
            theme_heat_score=s.get('theme_heat_score', 0),
            amount_quality_score=s.get('amount_quality_score', 0),
            market_regime_score=s.get('market_regime_score', 0),
            northbound_flow_score=s.get('northbound_flow_score', 0),
            rsi_divergence_score=s.get('rsi_divergence_score', 0),
            macd_signal_score=s.get('macd_signal_score', 0),
            ma_alignment_score=s.get('ma_alignment_score', 0),
            boll_squeeze_score=s.get('boll_squeeze_score', 0),
            kdj_oversold_score=s.get('kdj_oversold_score', 0),
            mfi_score=s.get('mfi_score', 0),
            shadow_quality_score=s.get('shadow_quality_score', 0),
            pullback_speed_score=s.get('pullback_speed_score', 0),
            intraday_reversal_score=s.get('intraday_reversal_score', 0),
            volume_climax_score=s.get('volume_climax_score', 0),
            sector_relative_score=s.get('sector_relative_score', 0),
            adx_trend_score=s.get('adx_trend_score', 0),
            obv_accumulation_score=s.get('obv_accumulation_score', 0),
            cmf_score=s.get('cmf_score', 0),
            gap_support_score=s.get('gap_support_score', 0),
            ml_confidence=s.get('ml_confidence', 0.0),
            sequence_confidence=s.get('sequence_confidence', 0.0),
            sequence_score=s.get('sequence_score', 0),
            factor_score=s.get('factor_score', 0),
            details=s.get('factor_details', {}),
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
