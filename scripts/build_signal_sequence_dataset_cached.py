"""Build a broader cached signal dataset for sequence / meta-model training.

This script scans cached OHLCV files with relaxed N-pattern rules and records
the strongest signal every few bars. Labels are assigned from a short-term
forward path:

- positive: after the next bar touches the entry zone, price reaches +6%
  before hitting stop-loss within the next 15 trading days
- negative: otherwise

Usage:
    python3 scripts/build_signal_sequence_dataset_cached.py [cache_dir] [limit]
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.strategy.kline_sequence import build_kline_tensor, load_sequence_model, predict_sequence_prob
from src.strategy.n_pattern import NPatternParams, find_n_signals


def _load_sequence_artifact():
    for name in ["kline_sequence_model_tradefactors.pt", "kline_sequence_model_cachedsignals.pt", "kline_sequence_model.pt"]:
        path = PROJECT_ROOT / "data" / name
        if path.exists():
            try:
                return load_sequence_model(path)
            except Exception:
                continue
    return None


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns:
        if "datetime" in out.columns:
            out["date"] = pd.to_datetime(out["datetime"]).dt.strftime("%Y-%m-%d")
        else:
            out["date"] = pd.to_datetime(out.index).strftime("%Y-%m-%d")
    if "volume" not in out.columns and "vol" in out.columns:
        out["volume"] = out["vol"]
    return out.sort_values("date").reset_index(drop=True)


def _label_signal(df: pd.DataFrame, idx: int, signal: dict, horizon: int = 15) -> tuple[int, float]:
    """Label based on whether a touched entry reaches +6% before stop-loss."""
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    entry = float(signal["entry_price"])
    stop = float(signal["stop_loss"])
    target = entry * 1.06

    start = idx + 1
    end = min(len(df), idx + 1 + horizon)
    if start >= end:
        return 0, 0.0

    touched = False
    entry_fill = entry
    for j in range(start, end):
        if lows[j] <= entry:
            touched = True
            entry_fill = max(entry, min(closes[j], highs[j]))
            break
    if not touched:
        return 0, 0.0

    realized = 0.0
    for j in range(j, end):
        if lows[j] <= stop:
            realized = (stop / entry_fill - 1) * 100
            return 0, realized
        if highs[j] >= target:
            realized = (target / entry_fill - 1) * 100
            return 1, realized

    realized = (closes[end - 1] / entry_fill - 1) * 100
    return (1 if realized > 0 else 0), realized


def main():
    cache_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "data" / "bt_cache"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    params = NPatternParams(**cfg["n_pattern"])
    params.high_win_mode = False

    files = sorted(cache_dir.glob("*.pkl"))
    if limit > 0:
        files = files[:limit]

    rows = []
    X_seq = []
    y = []
    seq_artifact = _load_sequence_artifact()
    scanned = 0
    for p in files:
        try:
            df = _prepare_df(pickle.load(open(p, "rb")))
        except Exception:
            continue
        if len(df) < 180:
            continue

        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        vols = df["volume"].values
        amounts = df["amount"].values if "amount" in df.columns else None

        for idx in range(120, len(df) - 20, 5):
            try:
                sigs = find_n_signals(
                    opens[:idx],
                    highs[:idx],
                    lows[:idx],
                    closes[:idx],
                    vols[:idx],
                    params,
                    amounts=amounts[:idx] if amounts is not None else None,
                )
            except Exception:
                continue
            if not sigs:
                continue

            best = max(sigs, key=lambda s: (s.get("strength", 0), s.get("factor_score", 0)))
            seq = build_kline_tensor(opens, highs, lows, closes, vols, end_idx=idx - 1, window=30)
            if seq is None:
                continue
            seq_prob = float(predict_sequence_prob(seq_artifact, seq)) if seq_artifact is not None else 0.5
            label, realized = _label_signal(df, idx, best)
            rows.append(
                {
                    "code": p.stem,
                    "signal_date": str(df.loc[idx - 1, "date"])[:10],
                    "label": int(label),
                    "realized_pct": round(float(realized), 2),
                    "strength": int(best.get("strength", 0)),
                    "factor_score": int(best.get("factor_score", 0)),
                    "ml_confidence": float(best.get("ml_confidence", 0.5)),
                    "sequence_confidence": round(seq_prob, 4),
                    "rr_ratio": float(best.get("rr_ratio", 0.0)),
                    "pullback_volume_score": int(best.get("pullback_volume_score", 0)),
                    "turnover_crowding_score": int(best.get("turnover_crowding_score", 0)),
                    "relative_strength_score": int(best.get("relative_strength_score", 0)),
                    "volatility_contraction_score": int(best.get("volatility_contraction_score", 0)),
                    "support_reclaim_score": int(best.get("support_reclaim_score", 0)),
                    "close_position_score": int(best.get("close_position_score", 0)),
                    "limit_up_followthrough_score": int(best.get("limit_up_followthrough_score", 0)),
                    "market_regime_score": int(best.get("market_regime_score", 0)),
                    "shadow_quality_score": int(best.get("shadow_quality_score", 0)),
                    "pullback_speed_score": int(best.get("pullback_speed_score", 0)),
                    "intraday_reversal_score": int(best.get("intraday_reversal_score", 0)),
                    "sector_relative_score": int(best.get("sector_relative_score", 0)),
                    "adx_trend_score": int(best.get("adx_trend_score", 0)),
                    "obv_accumulation_score": int(best.get("obv_accumulation_score", 0)),
                    "cmf_score": int(best.get("cmf_score", 0)),
                    "gap_support_score": int(best.get("gap_support_score", 0)),
                }
            )
            X_seq.append(seq)
            y.append(float(label))

        scanned += 1
        if scanned % 50 == 0:
            print(f"scanned {scanned}/{len(files)} files -> samples {len(rows)}")

    out_rows = pd.DataFrame(rows)
    out_rows.to_csv(PROJECT_ROOT / "data" / "cached_signal_dataset.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        PROJECT_ROOT / "data" / "cached_signal_sequences.npz",
        X_seq=np.asarray(X_seq, dtype=np.float32),
        y=np.asarray(y, dtype=np.float32),
        meta=out_rows[["code", "signal_date", "realized_pct"]].astype(object).values,
    )
    print(f"saved rows -> {PROJECT_ROOT / 'data' / 'cached_signal_dataset.csv'}")
    print(f"saved seq  -> {PROJECT_ROOT / 'data' / 'cached_signal_sequences.npz'}")
    print(f"samples={len(rows)} positives={int(sum(y))} negatives={len(y)-int(sum(y))}")


if __name__ == "__main__":
    main()
