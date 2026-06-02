"""Shared ranking helpers for daily scan and backtests.

This module loads a learned ranker artifact when available and falls back to
the older heuristic score when the artifact is missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = PROJECT_ROOT / "data" / "ranker_artifact.json"

_ARTIFACT_CACHE: dict[str, Any] | None = None


def _get_value(obj: Any, key: str, default: float = 0.0) -> float:
    if isinstance(obj, Mapping):
        value = obj.get(key, default)
    else:
        value = getattr(obj, key, default)
    try:
        return float(value)
    except Exception:
        return float(default)


def load_ranker_artifact() -> dict[str, Any] | None:
    global _ARTIFACT_CACHE
    if _ARTIFACT_CACHE is not None:
        return _ARTIFACT_CACHE
    if not ARTIFACT_PATH.exists():
        return None
    try:
        with open(ARTIFACT_PATH, "r", encoding="utf-8") as f:
            _ARTIFACT_CACHE = json.load(f)
            return _ARTIFACT_CACHE
    except Exception:
        return None


def _heuristic_score(obj: Any) -> float:
    seq_prob = _get_value(obj, "sequence_confidence", 0.5)
    ml_prob = _get_value(obj, "ml_confidence", 0.5)
    rr_ratio = max(_get_value(obj, "rr_ratio", 0.0), 0.0)
    strength = _get_value(obj, "strength", 0.0)
    factor_score = _get_value(obj, "factor_score", 0.0)
    return (
        strength * (0.35 + ml_prob + 0.45 * seq_prob)
        + factor_score * 0.35
        + min(rr_ratio, 5.0) * 8.0
    )


def rank_score(obj: Any) -> float:
    """Compute a ranking score for NSignal or row-like objects."""
    artifact = load_ranker_artifact()
    if not artifact:
        return _heuristic_score(obj)

    features = artifact.get("features", [])
    means = artifact.get("means", [])
    stds = artifact.get("stds", [])
    weights = artifact.get("weights", [])
    bias = float(artifact.get("bias", 0.0))

    if not (len(features) == len(means) == len(stds) == len(weights)):
        return _heuristic_score(obj)

    score = bias
    for feature, mean, std, weight in zip(features, means, stds, weights):
        denom = float(std) if abs(float(std)) > 1e-9 else 1.0
        value = _get_value(obj, feature, float(mean))
        score += float(weight) * ((value - float(mean)) / denom)

    # Mild live confidence bonus. These are zero when the optional ML/sequence
    # models are unavailable, so they do not change the historical fit.
    score += 0.20 * (_get_value(obj, "ml_confidence", 0.5) - 0.5)
    score += 0.15 * (_get_value(obj, "sequence_confidence", 0.5) - 0.5)
    rr_ratio = max(_get_value(obj, "rr_ratio", 0.0), 0.0)
    score += 0.08 * max(rr_ratio - 1.0, 0.0)
    return float(score)

