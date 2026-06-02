"""Auto-iterate the high-win filter from historical trade factors.

Usage:
    python3 scripts/auto_iterate_strategy.py [csv_path] [target_precision] [min_selected]

What it does:
1. Trains a walk-forward precision ensemble.
2. Saves the model artifact used by the scanner.
3. Mines a few interpretable single-factor rules for human review.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from strategy.ml_filter import (
    compute_factor_discrimination,
    load_trade_data,
    save_model,
    train_ensemble_model,
)


RULE_COLS = [
    "factor_score",
    "relative_strength_score",
    "volatility_contraction_score",
    "limit_up_followthrough_score",
    "rsi_divergence_score",
    "macd_signal_score",
    "close_position_score",
]


def mine_single_rules(df: pd.DataFrame, min_selected: int = 8) -> pd.DataFrame:
    rows = []
    for col in RULE_COLS:
        if col not in df.columns:
            continue
        cuts = sorted(
            {
                round(float(x), 2)
                for x in np.quantile(df[col].dropna(), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            }
        )
        for thr in cuts:
            for op in (">=", "<="):
                mask = df[col] >= thr if op == ">=" else df[col] <= thr
                selected = int(mask.sum())
                if selected < min_selected:
                    continue
                wr = float(df.loc[mask, "is_win"].mean())
                rows.append(
                    {
                        "rule": f"{col}{op}{thr:.2f}",
                        "selected": selected,
                        "coverage": round(selected / len(df), 4),
                        "win_rate": round(wr, 4),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["win_rate", "selected"], ascending=[False, False]).head(20)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_ROOT, "data", "trade_factors.csv")
    target_precision = float(sys.argv[2]) if len(sys.argv) > 2 else 0.90
    min_selected = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    df = load_trade_data(csv_path)
    print(f"样本数: {len(df)}")
    print(f"基线胜率: {df['is_win'].mean() * 100:.1f}%")

    artifact, feature_cols, report = train_ensemble_model(
        df,
        cv_folds=4,
        target_precision=target_precision,
        min_selected=min_selected,
        include_strength=False,
    )
    save_model(artifact, feature_cols, threshold=artifact["threshold"])

    print("\n=== Precision Ensemble ===")
    print(f"阈值: {report['threshold']:.2f}")
    print(f"目标胜率: {target_precision * 100:.1f}%")
    print(f"选中样本: {report['selected']} ({report['coverage'] * 100:.1f}%)")
    print(f"OOF胜率: {report['selected_precision'] * 100:.1f}%")
    print(f"OOF AUC: {report['ensemble_auc']:.4f}")

    print("\n=== Top Feature Importance ===")
    for name, score in report["importance"][:12]:
        print(f"{name:>28}: {score:.4f}")

    print("\n=== Factor Discrimination ===")
    disc = compute_factor_discrimination(df)
    for _, row in disc.head(10).iterrows():
        print(
            f"{row['factor']:>28}: disc={row['discrimination']:+.3f} "
            f"win={row['win_mean']:.2f} loss={row['loss_mean']:.2f}"
        )

    print("\n=== Interpretable Rules ===")
    rules = mine_single_rules(df, min_selected=max(8, min_selected))
    if rules.empty:
        print("无可用规则")
    else:
        for _, row in rules.iterrows():
            print(
                f"{row['rule']:>36} | n={int(row['selected']):>3} "
                f"| cov={row['coverage'] * 100:>5.1f}% | wr={row['win_rate'] * 100:>5.1f}%"
            )


if __name__ == "__main__":
    main()
