"""Walk-forward ensemble training for N-pattern signals.

This replaces the old single-GBM training flow with a precision-oriented ensemble:
- GradientBoosting
- RandomForest
- ExtraTrees
- optional XGBoost

The output artifact is still written to data/xgboost_n_pattern.pkl for compatibility.
"""

import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from strategy.ml_filter import load_trade_data, save_model, train_ensemble_model


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_ROOT, "data", "trade_factors.csv")
    target_precision = float(sys.argv[2]) if len(sys.argv) > 2 else 0.90
    min_selected = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    df = load_trade_data(csv_path)
    if "entry_date" in df.columns:
        df["entry_date"] = pd.to_datetime(df["entry_date"].astype(str).str[:10], errors="coerce")
        df = df.sort_values("entry_date")

    print(f"总交易: {len(df)}")
    print(f"整体胜率: {df['is_win'].mean() * 100:.1f}%")
    if "entry_date" in df.columns and df["entry_date"].notna().any():
        print(f"日期范围: {df['entry_date'].min().date()} ~ {df['entry_date'].max().date()}")

    artifact, feature_cols, report = train_ensemble_model(
        df=df,
        cv_folds=4,
        target_precision=target_precision,
        min_selected=min_selected,
        include_strength=False,
    )

    print("\n=== Ensemble Summary ===")
    print(f"base_wr={report['base_win_rate'] * 100:.1f}%")
    print(f"ensemble_auc={report['ensemble_auc']:.4f}")
    print(f"threshold={report['threshold']:.2f}")
    print(f"selected={report['selected']} ({report['coverage'] * 100:.1f}%)")
    print(f"selected_wr={report['selected_precision'] * 100:.1f}%")

    print("\n=== Fold AUC ===")
    fold_df = pd.DataFrame(report["fold_results"])
    if not fold_df.empty:
        for _, row in fold_df.iterrows():
            print(
                f"{row['model']:>18} fold={int(row['fold'])} "
                f"auc={row['auc']:.4f} train={int(row['train_n'])} val={int(row['val_n'])}"
            )

    print("\n=== Top Feature Importance ===")
    for name, score in report["importance"][:12]:
        print(f"{name:>28}: {score:.4f}")

    save_model(artifact, feature_cols, threshold=artifact["threshold"])


if __name__ == "__main__":
    main()
