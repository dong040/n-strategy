"""Search scan-time signal rules on cached signal datasets.

This works directly on `cached_signal_dataset.csv`, which represents the signal
surface at scan time rather than only realized trades.

Usage:
    python3 scripts/optimize_scan_signal_rules.py data/cached_signal_dataset.csv
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"samples": 0, "win_rate": 0.0, "avg_realized_pct": 0.0, "days": 0, "avg_per_day": 0.0}
    wins = df[df["label"] == 1]
    days = pd.to_datetime(df["signal_date"]).nunique()
    return {
        "samples": int(len(df)),
        "win_rate": round(len(wins) / len(df) * 100, 4),
        "avg_realized_pct": round(float(df["realized_pct"].mean()), 4),
        "days": int(days),
        "avg_per_day": round(len(df) / max(days, 1), 4),
    }


def _rank_topn(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    ranked = df.copy()
    ranked["signal_date"] = pd.to_datetime(ranked["signal_date"])
    ranked["composite"] = (
        ranked["strength"] * (0.35 + ranked["ml_confidence"] + ranked["sequence_confidence"] * 0.45)
        + ranked["factor_score"] * 0.35
    )
    ranked["day_rank"] = ranked.groupby("signal_date")["composite"].rank(method="first", ascending=False)
    return ranked[ranked["day_rank"] <= top_n].copy()


def _search_watchlist(df: pd.DataFrame) -> pd.DataFrame:
    grids = {
        "ml_confidence": [0.25, 0.30, 0.35, 0.40, 0.45],
        "sequence_confidence": [0.47, 0.48, 0.49, 0.50],
        "strength": [70, 75, 80, 85, 90],
        "factor_score": [-10, 0, 10, 20],
        "close_position_score": [-10, -5, 0, 5],
        "volatility_contraction_score": [-5, 0, 3, 5],
    }
    rows = []
    combos = itertools.product(*grids.values())
    for ml_thr, seq_thr, strength_thr, factor_thr, close_thr, vol_thr in combos:
        sub = df[
            (df["ml_confidence"] >= ml_thr)
            & (df["sequence_confidence"] >= seq_thr)
            & (df["strength"] >= strength_thr)
            & (df["factor_score"] >= factor_thr)
            & (df["close_position_score"] >= close_thr)
            & (df["volatility_contraction_score"] >= vol_thr)
        ].copy()
        top = _rank_topn(sub, top_n=3)
        stats = _summary(top)
        if stats["samples"] < 8:
            continue
        score = stats["win_rate"] + stats["avg_realized_pct"] * 2.0 - abs(stats["avg_per_day"] - 3.0) * 10.0
        rows.append(
            {
                "layer": "watchlist",
                "score": round(score, 4),
                "ml_confidence_min": ml_thr,
                "sequence_confidence_min": seq_thr,
                "strength_min": strength_thr,
                "factor_score_min": factor_thr,
                "close_position_score_min": close_thr,
                "volatility_contraction_score_min": vol_thr,
                **stats,
            }
        )
    return pd.DataFrame(rows)


def _search_execution(df: pd.DataFrame) -> pd.DataFrame:
    grids = {
        "ml_confidence": [0.45, 0.50, 0.55, 0.60],
        "sequence_confidence": [0.48, 0.49, 0.50],
        "strength": [75, 80, 85, 90],
        "factor_score": [0, 10, 15, 20],
        "close_position_score": [0, 3, 5, 7],
        "volatility_contraction_score": [0, 3, 5, 7],
    }
    rows = []
    combos = itertools.product(*grids.values())
    for ml_thr, seq_thr, strength_thr, factor_thr, close_thr, vol_thr in combos:
        sub = df[
            (df["ml_confidence"] >= ml_thr)
            & (df["sequence_confidence"] >= seq_thr)
            & (df["strength"] >= strength_thr)
            & (df["factor_score"] >= factor_thr)
            & (df["close_position_score"] >= close_thr)
            & (df["volatility_contraction_score"] >= vol_thr)
        ].copy()
        stats = _summary(sub)
        if stats["samples"] < 3:
            continue
        score = stats["win_rate"] * 2.0 + min(stats["samples"], 10) * 2.0 + stats["avg_realized_pct"]
        rows.append(
            {
                "layer": "execution",
                "score": round(score, 4),
                "ml_confidence_min": ml_thr,
                "sequence_confidence_min": seq_thr,
                "strength_min": strength_thr,
                "factor_score_min": factor_thr,
                "close_position_score_min": close_thr,
                "volatility_contraction_score_min": vol_thr,
                **stats,
            }
        )
    return pd.DataFrame(rows)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python3 scripts/optimize_scan_signal_rules.py data/cached_signal_dataset.csv")
    path = Path(sys.argv[1])
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit("empty dataset")

    watch = _search_watchlist(df)
    exec_df = _search_execution(df)
    out = pd.concat([watch, exec_df], ignore_index=True)
    if out.empty:
        print("no rules found")
        return
    out = out.sort_values(["layer", "score", "win_rate", "samples"], ascending=[True, False, False, False])
    out_path = PROJECT_ROOT / "data" / f"scan_signal_rule_search_{path.stem}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(out.groupby("layer").head(20).to_string(index=False))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
