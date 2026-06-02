"""Search execution-layer rules for high precision on benchmark trade ledgers.

Goal: widen strict mode while preserving very high win rate on the filtered set.

Usage:
    python3 scripts/optimize_execution_rules.py data/backtest_trades_xxx.csv
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"trades": 0, "win_rate": 0.0, "avg_profit_pct": 0.0, "signal_days": 0}
    wins = df[df["profit_pct"] > 0]
    return {
        "trades": int(len(df)),
        "win_rate": round(len(wins) / len(df) * 100, 4),
        "avg_profit_pct": round(float(df["profit_pct"].mean()), 4),
        "signal_days": int(pd.to_datetime(df["entry_date"]).nunique()),
    }


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python3 scripts/optimize_execution_rules.py data/backtest_trades_xxx.csv")

    path = Path(sys.argv[1])
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit("empty trades csv")

    if "sequence_confidence" not in df.columns:
        df["sequence_confidence"] = 0.0

    rules = []
    grids = {
        "ml_confidence": [0.45, 0.50, 0.55, 0.60, 0.65],
        "sequence_confidence": [0.47, 0.48, 0.49, 0.50],
        "strength": [80, 85, 90, 95, 100],
        "factor_score": [0, 10, 15, 20, 25],
        "close_position_score": [0, 3, 5, 7],
        "volatility_contraction_score": [0, 3, 5, 7],
    }

    combos = itertools.product(
        grids["ml_confidence"],
        grids["sequence_confidence"],
        grids["strength"],
        grids["factor_score"],
        grids["close_position_score"],
        grids["volatility_contraction_score"],
    )

    for ml_thr, seq_thr, strength_thr, factor_thr, close_thr, vol_thr in combos:
        sub = df[
            (df["ml_confidence"] >= ml_thr)
            & (df["sequence_confidence"] >= seq_thr)
            & (df["strength"] >= strength_thr)
            & (df["factor_score"] >= factor_thr)
            & (df["close_position_score"] >= close_thr)
            & (df["volatility_contraction_score"] >= vol_thr)
        ].copy()
        summary = _summarize(sub)
        if summary["trades"] < 1:
            continue
        # Strongly prefer precision, then sample size, then average return.
        score = summary["win_rate"] * 2.0 + min(summary["trades"], 10) * 3.0 + summary["avg_profit_pct"]
        rules.append(
            {
                "score": round(score, 4),
                "ml_confidence_min": ml_thr,
                "sequence_confidence_min": seq_thr,
                "strength_min": strength_thr,
                "factor_score_min": factor_thr,
                "close_position_score_min": close_thr,
                "volatility_contraction_score_min": vol_thr,
                **summary,
            }
        )

    if not rules:
        print("no rules found")
        return

    out = pd.DataFrame(rules).sort_values(
        ["win_rate", "trades", "avg_profit_pct", "score"],
        ascending=[False, False, False, False],
    ).head(40)
    out_path = PROJECT_ROOT / "data" / f"execution_rule_search_{path.stem}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
