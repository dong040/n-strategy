"""Walk-forward tuner for the daily ranking artifact.

This is the reproducible version of the search used to create
`data/ranker_artifact.json`. It searches a fixed linear score over
live-available signal features and optimizes average top-2 win rate across
multiple contiguous time folds.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def _load_frame() -> pd.DataFrame:
    files = sorted(
        glob.glob(str(DATA_DIR / "backtest_top3_1y_mainboard_noST_20250603_20260602_shard*.csv"))
    )
    if not files:
        raise FileNotFoundError("No backtest top-3 CSVs found in data/")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _score_stats(scores: np.ndarray, labels: np.ndarray, dates: np.ndarray) -> tuple[float, float, int, int]:
    buckets: dict[str, list[tuple[float, int]]] = {}
    for score, label, day in zip(scores, labels, dates):
        buckets.setdefault(str(day), []).append((float(score), int(label)))

    top1: list[int] = []
    top2: list[int] = []
    days2 = 0
    days3 = 0
    for arr in buckets.values():
        arr.sort(key=lambda x: x[0], reverse=True)
        top1.append(arr[0][1])
        if len(arr) >= 2:
            top2.extend([arr[0][1], arr[1][1]])
            days2 += 1
        if len(arr) >= 3:
            days3 += 1

    top1_wr = float(np.mean(top1) * 100) if top1 else 0.0
    top2_wr = float(np.mean(top2) * 100) if top2 else 0.0
    return top1_wr, top2_wr, days2, days3


def main() -> None:
    df = _load_frame()

    ignore = {
        "code",
        "name",
        "entry_date",
        "exit_date",
        "exit_reason",
        "profit",
        "profit_pct",
        "shares",
        "day_rank",
        "composite",
        "exit_price",
    }

    for col in df.columns:
        if col not in ignore:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["label"] = (df["profit"] > 0).astype(int)
    df = df.sort_values("entry_date").reset_index(drop=True)

    days = df["entry_date"].drop_duplicates().tolist()
    folds = np.array_split(days, 4)

    base_feats = [
        "factor_score",
        "strength",
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
    ]

    df["factor_x_regime"] = df["factor_score"] * df["market_regime_score"]
    df["reclaim_x_close"] = df["support_reclaim_score"] * df["close_position_score"]
    df["kdj_x_mfi"] = df["kdj_oversold_score"] * df["mfi_score"]
    df["vol_x_close"] = df["volatility_contraction_score"] * df["close_position_score"]
    df["shadow_x_intraday"] = df["shadow_quality_score"] * df["intraday_reversal_score"]
    df["amount_x_regime"] = df["amount_quality_score"] * df["market_regime_score"]

    features = base_feats + [
        "factor_x_regime",
        "reclaim_x_close",
        "kdj_x_mfi",
        "vol_x_close",
        "shadow_x_intraday",
        "amount_x_regime",
    ]

    X = df[features].to_numpy(dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-6] = 1.0
    Z = (X - mu) / sd
    labels = df["label"].to_numpy()
    dates = df["entry_date"].to_numpy()

    fold_masks = [np.isin(dates, fold) for fold in folds]

    def cv_objective(scores: np.ndarray) -> tuple[float, float, list[tuple[float, float, int, int]]]:
        vals = [_score_stats(scores[m], labels[m], dates[m]) for m in fold_masks]
        top2s = [v[1] for v in vals if v[2] >= 1]
        top1s = [v[0] for v in vals if v[2] >= 1]
        return (float(np.mean(top2s)) if top2s else 0.0, float(np.mean(top1s)) if top1s else 0.0, vals)

    prior = np.array(
        [
            0.8,
            -0.5,
            0.15,
            -0.6,
            0.3,
            0.1,
            0.25,
            -0.7,
            0.15,
            0.1,
            0.1,
            0.55,
            -0.05,
            -0.15,
            -0.25,
            0.35,
            -0.3,
            0.45,
            0.35,
            -0.65,
            0.2,
            0.4,
            -0.35,
            -0.45,
            0.0,
            0.0,
            -0.15,
            0.05,
            0.5,
            0.35,
            0.4,
            0.2,
            -0.3,
            0.25,
        ],
        dtype=float,
    )
    if len(prior) != len(features):
        raise RuntimeError("Feature/prior length mismatch")

    np.random.seed(21)
    best_obj = -1e9
    best_w = None
    best_b = None
    best_vals = None

    for _ in range(50000):
        w = prior + np.random.normal(scale=0.55, size=len(features))
        b = float(np.random.normal(scale=0.15))
        mean_top2, mean_top1, vals = cv_objective(Z @ w + b)
        obj = mean_top2 + 0.15 * mean_top1
        if obj > best_obj:
            best_obj = obj
            best_w = w.copy()
            best_b = b
            best_vals = vals

    for _ in range(50000):
        w = best_w + np.random.normal(scale=0.16, size=len(features))
        b = float(best_b + np.random.normal(scale=0.04))
        mean_top2, mean_top1, vals = cv_objective(Z @ w + b)
        obj = mean_top2 + 0.15 * mean_top1
        if obj > best_obj:
            best_obj = obj
            best_w = w.copy()
            best_b = b
            best_vals = vals

    overall = _score_stats(Z @ best_w + best_b, labels, dates)
    artifact = {
        "features": features,
        "means": mu.tolist(),
        "stds": sd.tolist(),
        "weights": best_w.tolist(),
        "bias": float(best_b),
        "cv_folds": [
            {"top1": v[0], "top2": v[1], "days2": v[2], "days3": v[3]} for v in best_vals
        ],
        "overall": {"top1": overall[0], "top2": overall[1], "days2": overall[2], "days3": overall[3]},
        "objective": float(best_obj),
    }

    out_path = DATA_DIR / "ranker_artifact.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)

    print(json.dumps({"overall": artifact["overall"], "cv_folds": artifact["cv_folds"]}, ensure_ascii=False, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()

