"""诊断脚本：检查两只大神标的为何未被 N-pattern scanner 捕获

Usage: python diagnose_expert_stocks.py
"""

import logging
import sys
import os
import numpy as np

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from screener.data_fetcher import get_daily_klines
from strategy.n_pattern import (
    NPatternParams,
    find_n_signals,
    find_extrema,
    _check_limit_up,
    _check_ma_bullish,
    _check_stabilization,
    _fib_to_level,
    _find_resistance_levels,
    scan_stock,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("diagnose")

TARGETS = [
    ("600172", "黄河旋风"),
    ("603052", "可川科技"),
]

params = NPatternParams()

# ──────────────────────────────────────────────────────────
# 1. Fetch data
# ──────────────────────────────────────────────────────────

def load_data(code, name):
    """Fetch daily K-lines for a single stock."""
    logger.info(f"Fetching K-lines for {code} {name} ...")
    df = get_daily_klines(code, days=250)
    if df.empty:
        logger.error(f"  EMPTY DataFrame for {code} {name}")
        return None
    # Sort by date ascending for the scanner (it assumes oldest first)
    df = df.sort_values("date", ascending=True).reset_index(drop=True)
    logger.info(f"  Got {len(df)} rows, date range {df['date'].iloc[0]} → {df['date'].iloc[-1]}")
    return df


# ──────────────────────────────────────────────────────────
# 2. Verbose diagnostic
# ──────────────────────────────────────────────────────────

def diagnose(code, name, df):
    """Step-by-step diagnostic to find which filter rejects the stock."""
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    vols = df["volume"].values
    n = len(closes)

    print(f"\n{'=' * 70}")
    print(f"DIAGNOSTIC: {code} {name}")
    print(f"{'=' * 70}")
    print(f"K-lines: {n} rows, latest date: {df['date'].iloc[-1]}")
    print(f"Latest OHLC: O={opens[-1]:.2f} H={highs[-1]:.2f} L={lows[-1]:.2f} C={closes[-1]:.2f}")

    # ── 1. 涨停基因硬过滤 ──
    limit_up_count = _check_limit_up(closes, lookback=30)
    has_limit_up = limit_up_count >= 1
    print(f"\n[1] 涨停基因 (近30日): count={limit_up_count}  {'PASS' if has_limit_up else '>>> FAIL <<<'}")
    if not has_limit_up:
        print("    >>> REJECTED: 近30日无涨停 <<<")
        return

    # ── 2. MA values ──
    ma_bullish, ma9_gt_ma10, bullish_days, ma_consistent, ma9, ma10, ma20 = _check_ma_bullish(closes)
    print(f"\n[2] MA 均线:")
    print(f"    MA9={ma9:.2f}  MA10={ma10:.2f}  MA20={ma20:.2f}")
    print(f"    MA9>MA10>MA20 (today): {ma_bullish}")
    print(f"    MA9>MA10 (today):       {ma9_gt_ma10}")
    print(f"    bull days (last 5):      {bullish_days} (need >=3 for consistent)")
    print(f"    ma_consistent:           {ma_consistent}")

    # ── 3. 收盘价 vs MA10 ──
    last_close = closes[-1]
    close_above_ma10 = ma10 is None or last_close >= ma10
    ma10_str = f"{ma10:.2f}" if ma10 else "None"
    print(f"\n[3] 收盘价 vs MA10: close={last_close:.2f}  ma10={ma10_str}")
    if not close_above_ma10:
        print(f"    >>> REJECTED: 收盘价跌破 MA10 <<<")
        return
    print(f"    PASS (close >= MA10)")

    # ── 4. 今天涨停 ──
    if n >= 2 and closes[-2] > 0:
        today_chg = (closes[-1] - closes[-2]) / closes[-2]
        print(f"\n[4] 今天涨停检查: chg={today_chg*100:.1f}%")
        if today_chg >= 0.095:
            print(f"    >>> REJECTED: 今天涨停，不是买点 <<<")
            return
        print(f"    PASS (not limit-up today)")
    else:
        print(f"\n[4] 今天涨停检查: insufficient data, PASS")

    # ── 5. 今天摸 MA9/MA10 ──
    today_low = lows[-1]
    print(f"\n[5] 今日最低 vs MA 支撑:")
    print(f"    today_low={today_low:.2f}")
    ma10_touch_rejected = False
    ma9_touch_rejected = False
    if ma10 is not None:
        ma10_threshold = ma10 * 1.002
        print(f"    MA10*1.002={ma10_threshold:.2f}  low <= MA10*1.002? {today_low <= ma10_threshold}")
        if today_low <= ma10_threshold:
            print(f"    >>> REJECTED: 今日摸过 MA10 <<<")
            ma10_touch_rejected = True
    if ma9 is not None:
        ma9_threshold = ma9 * 1.02
        print(f"    MA9*1.02={ma9_threshold:.2f}  low <= MA9*1.02? {today_low <= ma9_threshold}")
        if today_low <= ma9_threshold:
            print(f"    >>> REJECTED: 今日摸过 MA9(2%) <<<")
            ma9_touch_rejected = True
    if not ma10_touch_rejected and not ma9_touch_rejected:
        print(f"    PASS (no MA touch today)")
    else:
        print(f"\n  === CONTINUING PAST MA-TOUCH FILTER TO SHOW WHAT SIGNALS WOULD EXIST ===")

    # ── 6. MA10 broken intraday (recent 3 days) ──
    recent_lows = lows[-3:] if n >= 3 else lows
    ma10_broken_intraday = ma10 is not None and any(low < ma10 for low in recent_lows)
    print(f"\n[6] 近日跌破MA10 (intraday): {ma10_broken_intraday}")
    if ma10 is not None:
        for i, lo in enumerate(recent_lows):
            print(f"    recent lows[-{len(recent_lows)-i} or less]: {lo:.2f} vs MA10={ma10:.2f}")

    # ── 7. Find extrema ──
    peaks, troughs = find_extrema(highs, lows)
    print(f"\n[7] 极值点: peaks={peaks} ({len(peaks)}), troughs={troughs} ({len(troughs)})")
    if len(troughs) < 2:
        print(f"    >>> 低谷少于2个，无法构成N字 <<<")
        return

    # Show last few extrema for context
    print(f"    Recent extrema (last 30 bars):")
    last_idx = n - 1
    for p in peaks:
        if p >= last_idx - 30:
            print(f"      Peak  idx={p} date={df['date'].iloc[p]} high={highs[p]:.2f}")
    for t in troughs:
        if t >= last_idx - 30:
            print(f"      Trough idx={t} date={df['date'].iloc[t]} low={lows[t]:.2f}")

    # ── 8. Check each trough pair ──
    print(f"\n[8] 遍历 trough pairs 检查N形态:")

    signal_count = 0
    rejection_reasons = {}
    checked_any_pair = False

    for ti_idx, ti in enumerate(troughs):
        for tj_idx, tb in enumerate(troughs[ti_idx + 1:], start=ti_idx + 1):
            # Filter: tb must be within last 7 bars
            if tb < n - 7:
                continue

            total_n_days = tb - ti
            if total_n_days > 90:
                continue

            peaks_between = [p for p in peaks if ti < p < tb]
            if not peaks_between:
                continue
            best_p = max(peaks_between, key=lambda p: highs[p])

            first_low = lows[ti]
            first_high = highs[best_p]
            retrace_low = lows[tb]

            first_rise = (first_high - first_low) / first_low
            retrace = (first_high - retrace_low) / (first_high - first_low)
            retrace_days = tb - best_p

            checked_any_pair = True
            print(f"\n  --- Checking pair: ti={ti} ({df['date'].iloc[ti]}) tb={tb} ({df['date'].iloc[tb]}) best_p={best_p} ({df['date'].iloc[best_p]}) ---")
            print(f"    first_low={first_low:.2f}  first_high={first_high:.2f}  retrace_low={retrace_low:.2f}")
            print(f"    total_n_days={total_n_days}  retrace_days={retrace_days}")
            print(f"    first_rise={first_rise*100:.1f}%  retrace={retrace*100:.1f}%")

            # Check 1: first_rise range
            if first_rise < params.min_rise_1st or first_rise > params.max_rise_1st:
                print(f"    >>> REJECT: first_rise out of range [{params.min_rise_1st*100:.0f}%, {params.max_rise_1st*100:.0f}%]")
                rejection_reasons.setdefault("first_rise_range", 0)
                rejection_reasons["first_rise_range"] += 1
                continue

            # Check 2: retrace range
            if retrace < params.retrace_min or retrace > params.retrace_max:
                print(f"    >>> REJECT: retrace out of range [{params.retrace_min*100:.0f}%, {params.retrace_max*100:.0f}%]")
                rejection_reasons.setdefault("retrace_range", 0)
                rejection_reasons["retrace_range"] += 1
                continue

            # Check 3: weak trend + deep retrace
            if first_rise < 0.15 and retrace > 0.40:
                print(f"    >>> REJECT: 弱趋势深回调 (first_rise={first_rise*100:.1f}%<15% AND retrace={retrace*100:.1f}%>40%)")
                rejection_reasons.setdefault("weak_trend_deep_retrace", 0)
                rejection_reasons["weak_trend_deep_retrace"] += 1
                continue

            # Check 4: retrace_days range
            if retrace_days < 1 or retrace_days > params.retrace_days_max:
                print(f"    >>> REJECT: retrace_days out of range [1, {params.retrace_days_max}]")
                rejection_reasons.setdefault("retrace_days_range", 0)
                rejection_reasons["retrace_days_range"] += 1
                continue

            # Check 5: V型反转
            v_reversal = False
            for di in range(1, min(4, n - tb)):
                if closes[tb + di - 1] > 0:
                    chg_di = (closes[tb + di] - closes[tb + di - 1]) / closes[tb + di - 1]
                    if chg_di >= 0.095:
                        v_reversal = True
                        break
            if v_reversal:
                print(f"    >>> REJECT: V型反转排除")
                rejection_reasons.setdefault("v_reversal", 0)
                rejection_reasons["v_reversal"] += 1
                continue

            # Check 6: fib distance
            fib_level = _fib_to_level(retrace)
            fib_price = first_high - (first_high - first_low) * fib_level
            fib_dist = abs(last_close - fib_price) / fib_price
            fib_dist_max = 0.12 if has_limit_up else 0.05
            print(f"    fib_level={fib_level:.3f}  fib_price={fib_price:.2f}  fib_dist={fib_dist*100:.1f}%  max={fib_dist_max*100:.0f}%")
            if fib_dist > fib_dist_max:
                print(f"    >>> REJECT: fib_dist too large")
                rejection_reasons.setdefault("fib_dist", 0)
                rejection_reasons["fib_dist"] += 1
                continue

            # Check 7: MA/Fib 共振 (NOT required since require_ma_confluence=false)
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
            print(f"    MA/Fib共振: {ma_fib_ok}  nearest_ma={nearest_ma}")
            if ma10 is not None:
                print(f"      MA10-fib dist: {abs(fib_price-ma10)/fib_price*100:.1f}%")
            if ma9 is not None:
                print(f"      MA9-fib dist: {abs(fib_price-ma9)/fib_price*100:.1f}%")
            # NOTE: require_ma_confluence is False, so this is NOT a hard filter

            # Check 8: 企稳确认
            stab_ok, has_vol_shrink, has_shadow = _check_stabilization(
                opens, highs, lows, closes, vols, best_p, tb, params,
            )
            o_tb = opens[tb]
            c_tb = closes[tb]
            l_tb = lows[tb]
            h_tb = highs[tb]
            body_bottom = min(o_tb, c_tb)
            shadow_ratio = (body_bottom - l_tb) / c_tb if c_tb > 0 else 0
            intraday_drop = (o_tb - l_tb) / o_tb if o_tb > 0 else 0
            recovery = (c_tb - l_tb) / l_tb if l_tb > 0 else 0

            retrace_vols = vols[best_p:tb + 1]
            max_vol = np.max(retrace_vols)
            tb_vol = vols[tb]

            print(f"    企稳确认 day (tb={tb}, date={df['date'].iloc[tb]}):")
            print(f"      O={o_tb:.2f} H={h_tb:.2f} L={l_tb:.2f} C={c_tb:.2f}")
            print(f"      shadow_ratio={shadow_ratio*100:.1f}%  (need >= {params.lower_shadow_ratio*100:.0f}%)")
            print(f"      intraday_drop={intraday_drop*100:.1f}%  (need >= 3%)")
            print(f"      recovery_from_low={recovery*100:.1f}%  (need >= 0.5%)")
            print(f"      has_intraday_reversal: {intraday_drop >= 0.03 and recovery >= 0.005}")
            print(f"      retrace_max_vol={max_vol:.0f}  tb_vol={tb_vol:.0f}  shrink_ratio={tb_vol/max_vol:.2f}  (need < {params.stabilization_vol_shrink})")
            print(f"      has_vol_shrink={has_vol_shrink}  has_shadow={has_shadow}  stab_ok={stab_ok}")

            if not stab_ok and not has_vol_shrink and not has_shadow:
                print(f"    >>> REJECT: 无企稳信号")
                rejection_reasons.setdefault("no_stabilization", 0)
                rejection_reasons["no_stabilization"] += 1
                continue

            # Check 9: entry price (MA support)
            entry_price = None
            ma_candidates = []
            if ma10 is not None and ma10 < last_close:
                ma_candidates.append(round(ma10, 2))
            if ma9 is not None and ma9 < last_close:
                ma_candidates.append(round(ma9, 2))

            print(f"    Entry price calculation:")
            print(f"      last_close={last_close:.2f}")
            print(f"      ma9={ma9:.2f}  ma9 < last_close? {ma9 < last_close}")
            print(f"      ma10={ma10:.2f}  ma10 < last_close? {ma10 < last_close}")
            print(f"      ma_candidates={ma_candidates}")

            if ma_candidates:
                entry_price = max(ma_candidates)
            if entry_price is None:
                print(f"    >>> REJECT: 无 MA 支撑 (both MA >= close)")
                rejection_reasons.setdefault("no_ma_support", 0)
                rejection_reasons["no_ma_support"] += 1
                continue

            # Check 10: distance to entry <= 6.5%
            dist_to_entry = abs(last_close - entry_price) / entry_price
            print(f"    entry_price={entry_price:.2f}  dist_to_entry={dist_to_entry*100:.1f}%  (max 6.5%)")
            if dist_to_entry > 0.065:
                print(f"    >>> REJECT: 现价距入场价太远")
                rejection_reasons.setdefault("dist_to_entry", 0)
                rejection_reasons["dist_to_entry"] += 1
                continue

            # Check 11: today's low already at entry
            print(f"    today_low={lows[-1]:.2f}  entry_price={entry_price:.2f}  low <= entry? {lows[-1] <= entry_price}")
            if lows[-1] <= entry_price:
                print(f"    >>> REJECT: 今日已测支撑")
                rejection_reasons.setdefault("already_tested", 0)
                rejection_reasons["already_tested"] += 1
                continue

            # === PASSED ALL FILTERS ===
            signal_count += 1
            print(f"    >>> SIGNAL FOUND! entry={entry_price:.2f} stop={entry_price*0.995:.2f} fib_level={fib_level:.3f}")

    if not checked_any_pair:
        # Check if any trough pair even reached the inner loop
        print(f"    没有 trough pair 进入内层检查。")
        print(f"    可能原因: 所有 trough 都在 n-7={n-7} 之前，或 total_n_days > 90，或 peaks_between 为空")
        # Let's print more detail about which pairs were filtered
        for ti_idx, ti in enumerate(troughs):
            for tj_idx, tb in enumerate(troughs[ti_idx + 1:], start=ti_idx + 1):
                reasons = []
                if tb < n - 7:
                    reasons.append(f"tb={tb} not in last 7 (n-7={n-7})")
                if tb - ti > 90:
                    reasons.append(f"total_n_days={tb-ti} > 90")
                peaks_between = [p for p in peaks if ti < p < tb]
                if not peaks_between:
                    reasons.append("no peaks between")
                if reasons:
                    print(f"    ti={ti} tb={tb}: skipped because {', '.join(reasons)}")

    if signal_count == 0:
        print(f"\n  >>> NO SIGNALS FOUND <<<")
        print(f"  Rejection summary: {rejection_reasons}")

    # ── 9. Big N check ──
    print(f"\n[9] 大N扫描检查:")
    if ma10 is not None and ma10 < last_close and ma9 is not None:
        big_n_entry = round(ma10 * 0.9, 2)
        near_or_below_ma9 = last_close <= ma9 * 1.02
        above_big_n = last_close > big_n_entry
        print(f"    ma10={ma10:.2f}  ma9={ma9:.2f}  close={last_close:.2f}")
        print(f"    big_n_entry (ma10*0.9)={big_n_entry:.2f}")
        print(f"    near/below ma9(2%): {near_or_below_ma9}  (close <= ma9*1.02={ma9*1.02:.2f})")
        print(f"    above_big_n: {above_big_n}")
        if near_or_below_ma9 and above_big_n:
            dist_to_big_n = abs(last_close - big_n_entry) / big_n_entry
            print(f"    dist_to_big_n={dist_to_big_n*100:.1f}% (max 10%) -> {'PASS' if dist_to_big_n <= 0.10 else 'FAIL'}")
            if dist_to_big_n <= 0.10:
                print(f"    >>> Big N entry zone! Would check trough pairs now.")
            else:
                print(f"    >>> Big N: dist too far")
        else:
            print(f"    >>> Big N pre-conditions not met")
    else:
        print(f"    ma10 < close? {ma10 is not None and ma10 < last_close}, ma9 exists? {ma9 is not None}")
        print(f"    >>> Big N pre-conditions not met")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    for code, name in TARGETS:
        df = load_data(code, name)
        if df is None:
            continue
        diagnose(code, name, df)

        # Also run the actual scanner to confirm
        print(f"\n  --- Actual scan_stock result ---")
        sigs = scan_stock(code, name, df, params)
        print(f"  Signals found: {len(sigs)}")
        for s in sigs:
            print(f"    entry={s.entry_price} stop={s.stop_loss} target={s.target_price} strength={s.strength}")
            print(f"    fib={s.fib_level:.3f}@{s.fib_price} first_rise={s.first_rise_pct}% retrace={s.retrace_pct}%")
            print(f"    stab_ok={s.stab_ok} vol_shrink={s.has_vol_shrink} shadow={s.has_shadow}")
            print(f"    MA bullish={s.ma_bullish} consistent={s.ma_consistent} bull_days={s.bullish_days}")
            print(f"    is_big_n={s.is_big_n}")
        if not sigs:
            print(f"    (none)")

if __name__ == "__main__":
    main()
