"""训练XGBoost模型预测N字信号胜率，并集成到策略中"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'data', 'xgboost_n_pattern.pkl')

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


def load_trade_data(csv_path: str = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = os.path.join(PROJECT_ROOT, 'data', 'trade_factors.csv')
    df = pd.read_csv(csv_path)
    # Ensure date columns are strings
    for col in ['entry_date', 'exit_date']:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def compute_factor_discrimination(df: pd.DataFrame) -> pd.DataFrame:
    """计算每个因子对胜负的区分度: (win_mean - loss_mean) / pooled_std"""
    results = []
    wins = df[df['is_win'] == 1]
    losses = df[df['is_win'] == 0]

    for key in FACTOR_KEYS:
        if key not in df.columns:
            continue
        w_mean = wins[key].mean()
        l_mean = losses[key].mean()
        w_std = wins[key].std()
        l_std = losses[key].std()
        pooled_std = np.sqrt((w_std**2 + l_std**2) / 2)
        disc = (w_mean - l_mean) / max(pooled_std, 0.001)

        # Also compute win rate by quantile
        q_bins = pd.qcut(df[key], q=5, duplicates='drop', labels=False)
        q_wr = df.groupby(q_bins)['is_win'].mean()

        results.append({
            'factor': key,
            'win_mean': round(w_mean, 2),
            'loss_mean': round(l_mean, 2),
            'discrimination': round(disc, 3),
            'abs_disc': abs(disc),
            'q1_wr': q_wr.get(0, 0) if 0 in q_wr.index else 0,
            'q5_wr': q_wr.get(q_wr.index.max(), 0) if len(q_wr) > 0 else 0,
        })

    return pd.DataFrame(results).sort_values('abs_disc', ascending=False)


def train_model(df: pd.DataFrame, cv_folds: int = 5) -> tuple:
    """Train XGBoost classifier with time-series CV"""
    feature_cols = [k for k in FACTOR_KEYS if k in df.columns and k != 'factor_score']
    # Add strength as feature
    if 'strength' in df.columns:
        feature_cols.append('strength')

    X = df[feature_cols].fillna(0).values
    y = df['is_win'].values

    print(f"训练数据: {len(X)} 笔, {len(feature_cols)} 特征, 胜率={y.mean()*100:.1f}%")

    # TimeSeriesSplit for realistic validation
    tscv = TimeSeriesSplit(n_splits=cv_folds)

    model = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, min_samples_leaf=5,
        random_state=42,
    )

    # Cross-validation
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)

        # Find best threshold
        best_f1 = 0
        best_thresh = 0.5
        best_wr = 0
        for thresh in np.arange(0.2, 0.85, 0.025):
            y_pred = (y_prob >= thresh).astype(int)
            if y_pred.sum() < 3:  # need min samples
                continue
            tp = ((y_pred == 1) & (y_val == 1)).sum()
            fp = ((y_pred == 1) & (y_val == 0)).sum()
            prec = tp / max(tp + fp, 1)
            wr = y_val[y_pred == 1].mean()
            f1 = 2 * prec * wr / max(prec + wr, 0.001)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                best_wr = wr

        cv_scores.append({
            'fold': fold,
            'auc': round(auc, 4),
            'best_thresh': round(best_thresh, 3),
            'best_winrate': round(best_wr * 100, 1),
            'n_selected': int((y_prob >= best_thresh).sum()),
            'n_total': len(y_val),
        })

    # Train final model on all data
    model.fit(X, y)

    # Global threshold sweep
    y_prob_all = model.predict_proba(X)[:, 1]
    print(f"\n=== CV Results ===")
    for s in cv_scores:
        print(f"  Fold {s['fold']}: AUC={s['auc']}, best_thresh={s['best_thresh']}, "
              f"sel_winrate={s['best_winrate']}%, selected={s['n_selected']}/{s['n_total']}")

    print(f"\n=== Threshold Sweep (full data) ===")
    print(f"{'Thresh':>8} {'Selected':>10} {'WinRate':>10} {'Coverage':>10}")
    for thresh in np.arange(0.2, 0.9, 0.05):
        selected = (y_prob_all >= thresh).sum()
        if selected < 10:
            continue
        sel_wr = y[y_prob_all >= thresh].mean() * 100
        coverage = selected / len(y) * 100
        marker = ' ←' if sel_wr >= 70 else ''
        print(f"{thresh:>8.2f} {selected:>10} {sel_wr:>9.1f}% {coverage:>9.1f}%{marker}")

    # Feature importance
    importance = list(zip(feature_cols, model.feature_importances_))
    importance.sort(key=lambda x: x[1], reverse=True)
    print(f"\n=== Top 10 Feature Importance ===")
    for name, imp in importance[:10]:
        print(f"  {name}: {imp:.4f}")

    return model, feature_cols, cv_scores


def save_model(model, feature_cols, threshold: float = 0.5):
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_cols': feature_cols,
            'threshold': threshold,
        }, f)
    print(f"\n模型已保存 → {MODEL_PATH}")


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def predict_signal(model_data: dict, factor_scores: dict, strength: int) -> tuple:
    """Predict win probability for a signal. Returns (prob, pass_filter)."""
    if model_data is None:
        return 0.5, True  # no model = pass through

    model = model_data['model']
    feature_cols = model_data['feature_cols']
    threshold = model_data.get('threshold', 0.5)

    features = []
    for col in feature_cols:
        if col == 'strength':
            features.append(strength)
        else:
            features.append(factor_scores.get(col, 0))

    X = np.array([features])
    prob = float(model.predict_proba(X)[0, 1])
    return prob, prob >= threshold


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    df = load_trade_data(csv_path)
    print(f"加载 {len(df)} 笔交易")
    print(f"胜率: {df['is_win'].mean()*100:.1f}%")

    # Factor discrimination
    disc = compute_factor_discrimination(df)
    print(f"\n=== 因子区分度 Top 10 ===")
    for _, row in disc.head(10).iterrows():
        direction = '→' if row['discrimination'] > 0 else '← 反效'
        print(f"  {row['factor']}: disc={row['discrimination']:.3f} {direction} "
              f"(win={row['win_mean']:.1f}, loss={row['loss_mean']:.1f})")

    # Train model
    model, feature_cols, cv_scores = train_model(df)
    save_model(model, feature_cols, threshold=0.5)
