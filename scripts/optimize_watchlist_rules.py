"""Search watchlist filtering rules toward ~3 trades/day with highest win rate.

Input should be a benchmark trade ledger produced by
`scripts/backtest_online_mainboard_2y.py` in relaxed/dual mode.

Usage:
    python3 scripts/optimize_watchlist_rules.py data/backtest_trades_xxx.csv
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _summarize(df: pd.DataFrame, top_n: int = 3) -> dict:
    if df.empty:
        return {"trades": 0, "win_rate": 0.0, "avg_profit_pct": 0.0, "signal_days": 0, "avg_per_day": 0.0}
    ranked = df.copy()
    ranked["entry_date"] = pd.to_datetime(ranked["entry_date"])
    ranked["composite"] = (
        ranked["strength"] * (0.35 + ranked["ml_confidence"] + ranked.get("sequence_confidence", 0.0) * 0.45)
        + ranked["factor_score"] * 0.35
    )
    ranked["day_rank"] = ranked.groupby("entry_date")["composite"].rank(method="first", ascending=False)
    top = ranked[ranked["day_rank"] <= top_n].copy()
    if top.empty:
        return {"trades": 0, "win_rate": 0.0, "avg_profit_pct": 0.0, "signal_days": 0, "avg_per_day": 0.0}
    wins = top[top["profit_pct"] > 0]
    signal_days = top["entry_date"].nunique()
    return {
        "trades": int(len(top)),
        "win_rate": round(len(wins) / len(top) * 100, 4),
        "avg_profit_pct": round(float(top["profit_pct"].mean()), 4),
        "signal_days": int(signal_days),
        "avg_per_day": round(len(top) / max(signal_days, 1), 4),
    }


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python3 scripts/optimize_watchlist_rules.py data/backtest_trades_xxx.csv")

    path = Path(sys.argv[1])
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit("empty trades csv")

    if "sequence_confidence" not in df.columns:
        df["sequence_confidence"] = 0.0

    rules = []
    grids = {
        "ml_confidence": [0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        "sequence_confidence": [0.40, 0.45, 0.50],
        "strength": [80, 90, 100, 110],
        "factor_score": [0, 10, 20, 30, 40],
        "close_position_score": [-20, -10, 0, 5],
        "volatility_contraction_score": [-10, -5, 0, 5],
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
        summary = _summarize(sub, top_n=3)
        if summary["trades"] < 5:
            continue
        # Reward win rate and avg return; penalize deviating too far from ~3/day.
        target_gap = abs(summary["avg_per_day"] - 3.0)
        score = summary["win_rate"] + summary["avg_profit_pct"] * 2.0 - target_gap * 10.0
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

    out = pd.DataFrame(rules).sort_values(["score", "win_rate", "trades"], ascending=[False, False, False]).head(30)
    out_path = PROJECT_ROOT / "data" / f"watchlist_rule_search_{path.stem}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
