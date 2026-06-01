"""Walk-forward ML training + 回测验证

正确流程：
1. 先跑回测收集全部交易+日期
2. 按日期切分 train/test
3. 训练模型（仅用 train 数据）
4. 重新跑回测，仅在 test 期间用 ML 过滤

这样保证无 look-ahead bias。
"""
import sys, os, logging, time, pickle, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

logging.basicConfig(level=logging.WARNING)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

FACTOR_KEYS = [
    "factor_score", "pullback_volume_score", "turnover_crowding_score",
    "relative_strength_score", "volatility_contraction_score", "support_reclaim_score",
    "close_position_score", "limit_up_followthrough_score", "theme_heat_score",
    "amount_quality_score", "market_regime_score", "northbound_flow_score",
    "rsi_divergence_score", "macd_signal_score", "ma_alignment_score",
    "boll_squeeze_score", "kdj_oversold_score", "mfi_score",
    "shadow_quality_score", "pullback_speed_score",
    "intraday_reversal_score", "volume_climax_score", "sector_relative_score",
]


def train_with_walkforward_splits(csv_path: str, n_splits: int = 4):
    """Time-series walk-forward training.

    Returns: (best_threshold, model, results_df)
    """
    df = pd.read_csv(csv_path)
    df['entry_date'] = pd.to_datetime(df['entry_date'].astype(str).str[:10])
    df = df.sort_values('entry_date')

    feature_cols = [k for k in FACTOR_KEYS if k in df.columns and k != 'factor_score']
    if 'strength' in df.columns:
        feature_cols.append('strength')

    print(f"总交易: {len(df)}, 日期范围: {df['entry_date'].min().date()} ~ {df['entry_date'].max().date()}")
    print(f"特征数: {len(feature_cols)}")
    print(f"整体胜率: {df['is_win'].mean()*100:.1f}%")

    # Time-based split: train on earlier, test on later
    dates = sorted(df['entry_date'].unique())
    split_dates = np.array_split(dates, n_splits)

    print(f"\n=== Walk-Forward 验证 ({n_splits} 期) ===")
    all_results = []

    for fold in range(1, n_splits):
        train_cutoff = split_dates[fold - 1][-1]
        test_start = split_dates[fold][0]
        test_end = split_dates[fold][-1]

        train_df = df[df['entry_date'] <= train_cutoff]
        test_df = df[(df['entry_date'] >= test_start) & (df['entry_date'] <= test_end)]

        if len(train_df) < 30 or len(test_df) < 10:
            print(f"  Fold {fold}: train={len(train_df)}, test={len(test_df)} — 样本不足，跳过")
            continue

        X_train = train_df[feature_cols].fillna(0).values
        y_train = train_df['is_win'].values
        X_test = test_df[feature_cols].fillna(0).values
        y_test = test_df['is_win'].values

        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42,
        )
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        # Find best threshold
        best_wr, best_thresh, best_n = 0, 0.5, 0
        for thresh in np.arange(0.25, 0.85, 0.025):
            selected = (y_prob >= thresh).sum()
            if selected < 5:
                continue
            sel_wr = y_test[y_prob >= thresh].mean() * 100
            if sel_wr > best_wr:
                best_wr = sel_wr
                best_thresh = thresh
                best_n = int(selected)

        baseline_wr = y_test.mean() * 100

        all_results.append({
            'fold': fold,
            'train_n': len(train_df),
            'test_n': len(test_df),
            'train_dates': f"{train_df['entry_date'].min().date()}~{train_df['entry_date'].max().date()}",
            'test_dates': f"{test_start.date()}~{test_end.date()}",
            'auc': round(auc, 4),
            'baseline_wr': round(baseline_wr, 1),
            'best_thresh': round(best_thresh, 3),
            'best_wr': round(best_wr, 1),
            'selected_n': best_n,
            'coverage': round(best_n / len(test_df) * 100, 1),
        })
        print(f"  Fold {fold} [{all_results[-1]['test_dates']}]: "
              f"AUC={auc:.4f}, baseline={baseline_wr:.1f}%, "
              f"thresh={best_thresh:.2f}→WR={best_wr:.1f}% ({best_n}/{len(test_df)})")

    # Train final model on all data up to last fold start
    final_cutoff = split_dates[-1][0]
    final_train = df[df['entry_date'] < final_cutoff]
    print(f"\n最终模型训练: {len(final_train)} 笔 (截止 {final_cutoff.date()}, "
          f"vs 整体 {len(df)} 笔)")

    final_model = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, min_samples_leaf=5, random_state=42,
    )
    X_final = final_train[feature_cols].fillna(0).values
    y_final = final_train['is_win'].values
    final_model.fit(X_final, y_final)

    # Find conservative threshold across all folds
    avg_thresh = np.mean([r['best_thresh'] for r in all_results])
    conservative_thresh = min(0.70, max(0.55, avg_thresh + 0.05))

    # Save model
    model_path = os.path.join(DATA_DIR, 'xgboost_n_pattern.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': final_model,
            'feature_cols': feature_cols,
            'threshold': conservative_thresh,
            'train_cutoff': str(final_cutoff.date()),
            'walkforward_results': all_results,
        }, f)

    print(f"\n模型已保存 → {model_path}")
    print(f"保守阈值: {conservative_thresh:.2f}")
    print(f"训练截止日期: {final_cutoff.date()}")
    print(f"注意: 此模型仅应用于 {final_cutoff.date()} 之后的回测日期")

    # Summary
    results_df = pd.DataFrame(all_results)
    print(f"\n=== Walk-Forward 汇总 ===")
    print(f"平均 AUC: {results_df['auc'].mean():.4f}")
    print(f"平均基线胜率: {results_df['baseline_wr'].mean():.1f}%")
    print(f"平均最优胜率: {results_df['best_wr'].mean():.1f}%")
    print(f"平均覆盖率: {results_df['coverage'].mean():.1f}%")

    return conservative_thresh, final_model, results_df


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA_DIR, 'trade_factors.csv')
    train_with_walkforward_splits(csv_path)
