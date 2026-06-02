"""ML meta-filter for N-pattern signals.

Goals:
1. Keep backward compatibility with the existing pickle artifact.
2. Support a higher-precision ensemble model trained with time-series splits.
3. Make it easy to auto-iterate on factor data without changing the scanner.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "xgboost_n_pattern.pkl")

FACTOR_KEYS = [
    "factor_score",
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

_MODEL_CACHE: dict[str, Any] | None = None


@dataclass
class ThresholdResult:
    threshold: float
    precision: float
    selected: int
    coverage: float


def load_trade_data(csv_path: str = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = os.path.join(PROJECT_ROOT, "data", "trade_factors.csv")
    df = pd.read_csv(csv_path)
    for col in ["entry_date", "exit_date"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def compute_factor_discrimination(df: pd.DataFrame) -> pd.DataFrame:
    """(win_mean - loss_mean) / pooled_std."""
    results = []
    wins = df[df["is_win"] == 1]
    losses = df[df["is_win"] == 0]

    for key in FACTOR_KEYS:
        if key not in df.columns:
            continue
        w_mean = wins[key].mean()
        l_mean = losses[key].mean()
        w_std = wins[key].std()
        l_std = losses[key].std()
        pooled_std = np.sqrt((w_std**2 + l_std**2) / 2)
        disc = (w_mean - l_mean) / max(pooled_std, 0.001)

        q_bins = pd.qcut(df[key], q=5, duplicates="drop", labels=False)
        q_wr = df.groupby(q_bins)["is_win"].mean()

        results.append(
            {
                "factor": key,
                "win_mean": round(w_mean, 2),
                "loss_mean": round(l_mean, 2),
                "discrimination": round(disc, 3),
                "abs_disc": abs(disc),
                "q1_wr": q_wr.get(0, 0) if 0 in q_wr.index else 0,
                "q5_wr": q_wr.get(q_wr.index.max(), 0) if len(q_wr) > 0 else 0,
            }
        )

    return pd.DataFrame(results).sort_values("abs_disc", ascending=False)


def _feature_columns(df: pd.DataFrame, include_strength: bool = False) -> list[str]:
    feature_cols = [k for k in FACTOR_KEYS if k in df.columns]
    if include_strength and "strength" in df.columns:
        feature_cols.append("strength")
    return feature_cols


def _clean_feature_frame(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    feat = df[feature_cols].copy()
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feat.values.astype(float)


def _candidate_models() -> list[tuple[str, Any]]:
    models = [
        (
            "gradient_boosting",
            GradientBoostingClassifier(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                min_samples_leaf=5,
                random_state=42,
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=5,
                min_samples_leaf=5,
                random_state=42,
                class_weight="balanced_subsample",
            ),
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=400,
                max_depth=5,
                min_samples_leaf=5,
                random_state=42,
                class_weight="balanced",
            ),
        ),
    ]
    try:
        from xgboost import XGBClassifier

        models.append(
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=350,
                    max_depth=4,
                    learning_rate=0.03,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=3,
                    reg_lambda=2.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=1,
                ),
            )
        )
    except Exception:
        pass
    return models


def _search_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    target_precision: float = 0.90,
    min_selected: int = 10,
) -> ThresholdResult:
    """Prefer thresholds that hit target precision, otherwise maximize precision."""
    candidates: list[ThresholdResult] = []
    for threshold in np.arange(0.35, 0.96, 0.01):
        mask = probs >= threshold
        selected = int(mask.sum())
        if selected < min_selected:
            continue
        precision = float(y_true[mask].mean())
        coverage = selected / max(len(y_true), 1)
        candidates.append(
            ThresholdResult(
                threshold=round(float(threshold), 2),
                precision=precision,
                selected=selected,
                coverage=coverage,
            )
        )

    if not candidates:
        return ThresholdResult(threshold=0.80, precision=0.0, selected=0, coverage=0.0)

    good = [c for c in candidates if c.precision >= target_precision]
    if good:
        return max(good, key=lambda c: (c.coverage, c.precision, -c.threshold))
    return max(candidates, key=lambda c: (c.precision, c.coverage, -c.threshold))


def _collect_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 4,
) -> tuple[list[dict], np.ndarray]:
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    fold_rows: list[dict] = []
    model_rows: list[dict] = []

    for model_name, model in _candidate_models():
        oof_prob = np.full(len(y), np.nan)
        fold_aucs: list[float] = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                continue

            model.fit(X_train, y_train)
            val_prob = model.predict_proba(X_val)[:, 1]
            oof_prob[val_idx] = val_prob
            auc = roc_auc_score(y_val, val_prob)
            fold_aucs.append(float(auc))
            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "auc": round(float(auc), 4),
                    "train_n": len(train_idx),
                    "val_n": len(val_idx),
                }
            )

        valid_mask = ~np.isnan(oof_prob)
        if valid_mask.sum() == 0:
            continue

        valid_auc = roc_auc_score(y[valid_mask], oof_prob[valid_mask])
        model_rows.append(
            {
                "name": model_name,
                "prototype": model,
                "oof_prob": oof_prob,
                "oof_auc": float(valid_auc),
                "weight": max(float(valid_auc) - 0.5, 0.01),
                "fold_auc_mean": float(np.mean(fold_aucs)) if fold_aucs else 0.5,
            }
        )

    if not model_rows:
        raise RuntimeError("No ML model could be trained from trade_factors.csv")

    ensemble_num = np.zeros(len(y), dtype=float)
    ensemble_den = np.zeros(len(y), dtype=float)
    for row in model_rows:
        mask = ~np.isnan(row["oof_prob"])
        ensemble_num[mask] += row["oof_prob"][mask] * row["weight"]
        ensemble_den[mask] += row["weight"]
    ensemble_prob = np.divide(
        ensemble_num,
        np.maximum(ensemble_den, 1e-9),
        out=np.full(len(y), 0.5, dtype=float),
    )
    return model_rows, ensemble_prob, fold_rows


def _aggregate_feature_importance(
    members: list[dict],
    feature_cols: list[str],
) -> list[tuple[str, float]]:
    scores = {col: 0.0 for col in feature_cols}
    total_weight = 0.0
    for member in members:
        model = member["prototype"]
        weight = member["weight"]
        importance = getattr(model, "feature_importances_", None)
        if importance is None:
            continue
        for col, imp in zip(feature_cols, importance):
            scores[col] += float(imp) * weight
        total_weight += weight
    if total_weight <= 0:
        return []
    rows = [(col, val / total_weight) for col, val in scores.items()]
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def train_ensemble_model(
    df: pd.DataFrame,
    cv_folds: int = 4,
    target_precision: float = 0.90,
    min_selected: int = 10,
    include_strength: bool = False,
) -> tuple[dict, list[str], dict]:
    """Train a walk-forward ensemble optimized for precision."""
    feature_cols = _feature_columns(df, include_strength=include_strength)
    X = _clean_feature_frame(df, feature_cols)
    y = df["is_win"].values.astype(int)

    members, ensemble_oof, fold_rows = _collect_oof_predictions(X, y, cv_folds=cv_folds)
    threshold_row = _search_threshold(
        y_true=y,
        probs=ensemble_oof,
        target_precision=target_precision,
        min_selected=min_selected,
    )

    trained_members = []
    for member in members:
        model = member["prototype"]
        model.fit(X, y)
        trained_members.append(
            {
                "name": member["name"],
                "model": model,
                "oof_auc": round(member["oof_auc"], 4),
                "weight": round(member["weight"], 4),
            }
        )

    importance_rows = _aggregate_feature_importance(members, feature_cols)
    report = {
        "rows": len(df),
        "base_win_rate": float(y.mean()),
        "ensemble_auc": round(float(roc_auc_score(y, ensemble_oof)), 4),
        "threshold": threshold_row.threshold,
        "selected": threshold_row.selected,
        "selected_precision": round(threshold_row.precision, 4),
        "coverage": round(threshold_row.coverage, 4),
        "fold_results": fold_rows,
        "importance": [(name, round(score, 4)) for name, score in importance_rows[:15]],
    }
    artifact = {
        "version": 2,
        "model_kind": "precision_ensemble",
        "feature_cols": feature_cols,
        "threshold": threshold_row.threshold,
        "target_precision": target_precision,
        "min_selected": min_selected,
        "report": report,
        "members": trained_members,
    }
    return artifact, feature_cols, report


def train_model(df: pd.DataFrame, cv_folds: int = 5) -> tuple:
    """Backward-compatible entrypoint.

    Historically returned a single model; now returns the ensemble artifact.
    """
    artifact, feature_cols, report = train_ensemble_model(
        df,
        cv_folds=cv_folds,
        target_precision=0.90,
        min_selected=max(8, len(df) // 30),
        include_strength=False,
    )
    cv_scores = report["fold_results"]
    return artifact, feature_cols, cv_scores


def save_model(model, feature_cols, threshold: float = 0.5):
    global _MODEL_CACHE
    if isinstance(model, dict) and model.get("model_kind") == "precision_ensemble":
        payload = model
    else:
        payload = {
            "version": 1,
            "model": model,
            "feature_cols": feature_cols,
            "threshold": threshold,
        }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    _MODEL_CACHE = payload
    print(f"\n模型已保存 -> {MODEL_PATH}")


def load_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        _MODEL_CACHE = pickle.load(f)
    return _MODEL_CACHE


def predict_signal(model_data: dict, factor_scores: dict, strength: int = 0) -> tuple[float, bool]:
    """Predict win probability for a signal. Returns (prob, pass_filter)."""
    if model_data is None:
        return 0.5, True

    threshold = model_data.get("threshold", 0.5)
    feature_cols = model_data.get("feature_cols", FACTOR_KEYS)
    features = []
    for col in feature_cols:
        if col == "strength":
            features.append(strength)
        else:
            features.append(factor_scores.get(col, 0))
    X = np.array([features], dtype=float)

    if model_data.get("model_kind") == "precision_ensemble":
        probs = []
        weights = []
        for member in model_data.get("members", []):
            model = member["model"]
            weight = float(member.get("weight", 1.0))
            try:
                prob = float(model.predict_proba(X)[0, 1])
            except Exception:
                continue
            probs.append(prob)
            weights.append(weight)
        if not probs:
            return 0.5, True
        prob = float(np.average(probs, weights=weights))
        return prob, prob >= threshold

    model = model_data["model"]
    prob = float(model.predict_proba(X)[0, 1])
    return prob, prob >= threshold


if __name__ == "__main__":
    df = load_trade_data()
    print(f"加载 {len(df)} 笔交易")
    print(f"胜率: {df['is_win'].mean() * 100:.1f}%")

    disc = compute_factor_discrimination(df)
    print("\n=== 因子区分度 Top 10 ===")
    for _, row in disc.head(10).iterrows():
        direction = "->" if row["discrimination"] > 0 else "<- 反效"
        print(
            f"  {row['factor']}: disc={row['discrimination']:.3f} {direction} "
            f"(win={row['win_mean']:.1f}, loss={row['loss_mean']:.1f})"
        )

    artifact, feature_cols, report = train_ensemble_model(df)
    print("\n=== Ensemble Report ===")
    print(
        f"  base_wr={report['base_win_rate'] * 100:.1f}% "
        f"auc={report['ensemble_auc']:.4f} "
        f"thresh={report['threshold']:.2f} "
        f"selected={report['selected']} "
        f"sel_wr={report['selected_precision'] * 100:.1f}%"
    )
    print("\n=== Top Feature Importance ===")
    for name, score in report["importance"][:10]:
        print(f"  {name}: {score:.4f}")
    save_model(artifact, feature_cols, threshold=artifact["threshold"])
