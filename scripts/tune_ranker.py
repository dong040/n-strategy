"""Fit a daily ranking artifact from historical top-3 candidate ledgers.

The script searches for a linear score over live-available signal features
using a chronological train/validation split and writes
`data/ranker_artifact.json`.

Example:
    python scripts/tune_ranker.py
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


def _load_top3_frame() -> pd.DataFrame:
    files = sorted(
        glob.glob(str(DATA_DIR / "backtest_top3_1y_mainboard_noST_20250603_20260602_shard*.csv"))
    )
    if not files:
        raise FileNotFoundError("No backtest top-3 CSVs found in data/")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _eval_scores(scores: np.ndarray, labels: np.ndarray, dates: np.ndarray) -> tuple[float, float, int, int]:
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
    df = _load_top3_frame()

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
    split = int(len(days) * 0.68)
    train_days = set(days[:split])
    val_days = set(days[split:])
    train = df[df["entry_date"].isin(train_days)].copy()
    val = df[df["entry_date"].isin(val_days)].copy()

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

    for d in (train, val):
        d["factor_x_regime"] = d["factor_score"] * d["market_regime_score"]
        d["reclaim_x_close"] = d["support_reclaim_score"] * d["close_position_score"]
        d["kdj_x_mfi"] = d["kdj_oversold_score"] * d["mfi_score"]
        d["vol_x_close"] = d["volatility_contraction_score"] * d["close_position_score"]
        d["shadow_x_intraday"] = d["shadow_quality_score"] * d["intraday_reversal_score"]
        d["amount_x_regime"] = d["amount_quality_score"] * d["market_regime_score"]

    features = base_feats + [
        "factor_x_regime",
        "reclaim_x_close",
        "kdj_x_mfi",
        "vol_x_close",
        "shadow_x_intraday",
        "amount_x_regime",
    ]

    x_train = train[features].to_numpy(dtype=float)
    x_val = val[features].to_numpy(dtype=float)
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd[sd < 1e-6] = 1.0
    z_train = (x_train - mu) / sd
    z_val = (x_val - mu) / sd
    y_train = train["label"].to_numpy()
    y_val = val["label"].to_numpy()
    d_train = train["entry_date"].to_numpy()
    d_val = val["entry_date"].to_numpy()

    # Starting point from a coarse manual search.
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

    np.random.seed(11)
    best_obj = -1e9
    best_w = None
    best_b = None
    best_train = None
    best_val = None

    def objective(w: np.ndarray, b: float) -> tuple[float, tuple[float, float, int, int], tuple[float, float, int, int]]:
        tr = _eval_scores(z_train @ w + b, y_train, d_train)
        va = _eval_scores(z_val @ w + b, y_val, d_val)
        # We care most about validation top-2, with mild guardrails for top-1 and train stability.
        obj = va[1] + 0.15 * va[0] + 0.05 * tr[1]
        return obj, tr, va

    # Broad search
    for _ in range(30000):
        w = prior + np.random.normal(scale=0.55, size=len(features))
        b = float(np.random.normal(scale=0.15))
        obj, tr, va = objective(w, b)
        if va[2] < 20:
            continue
        if obj > best_obj:
            best_obj = obj
            best_w = w.copy()
            best_b = b
            best_train = tr
            best_val = va

    # Local refinement
    assert best_w is not None and best_b is not None
    for _ in range(40000):
        w = best_w + np.random.normal(scale=0.14, size=len(features))
        b = float(best_b + np.random.normal(scale=0.04))
        obj, tr, va = objective(w, b)
        if va[2] < 20:
            continue
        if obj > best_obj:
            best_obj = obj
            best_w = w.copy()
            best_b = b
            best_train = tr
            best_val = va

    artifact = {
        "features": features,
        "means": mu.tolist(),
        "stds": sd.tolist(),
        "weights": best_w.tolist(),
        "bias": float(best_b),
        "train": {"top1": best_train[0], "top2": best_train[1], "days2": best_train[2], "days3": best_train[3]},
        "validation": {"top1": best_val[0], "top2": best_val[1], "days2": best_val[2], "days3": best_val[3]},
        "objective": float(best_obj),
    }

    out_path = DATA_DIR / "ranker_artifact.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)

    print(json.dumps(artifact["validation"], ensure_ascii=False, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()

