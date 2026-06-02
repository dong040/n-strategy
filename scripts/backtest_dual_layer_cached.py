"""Dual-layer backtest on locally cached OHLCV data.

This evaluates:
1. Execution layer: strict high-win subset
2. Watchlist layer: relaxed candidates ranked to Top-N per day

It is intentionally cache-first so we can iterate fast without waiting for
full-market online data fetches every time.
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.strategy.backtest import BacktestConfig, backtest_single_stock
from src.strategy.n_pattern import NPatternParams


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns:
        if "datetime" in out.columns:
            out["date"] = pd.to_datetime(out["datetime"]).dt.strftime("%Y-%m-%d")
        else:
            out["date"] = pd.to_datetime(out.index).strftime("%Y-%m-%d")
    if "volume" not in out.columns and "vol" in out.columns:
        out["volume"] = out["vol"]
    return out


def _summarize(label: str, df: pd.DataFrame) -> dict:
    if df.empty:
        return {"label": label, "trades": 0, "win_rate": 0, "avg_profit_pct": 0, "days": 0, "avg_per_day": 0}
    wins = df[df["profit_pct"] > 0]
    days = df["entry_date"].nunique()
    return {
        "label": label,
        "trades": int(len(df)),
        "win_rate": round(len(wins) / len(df) * 100, 2),
        "avg_profit_pct": round(float(df["profit_pct"].mean()), 2),
        "median_profit_pct": round(float(df["profit_pct"].median()), 2),
        "days": int(days),
        "avg_per_day": round(len(df) / max(days, 1), 2),
    }


def main():
    cache_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "data" / "bt_cache"
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    scan_cfg = cfg.get("screener", {})
    strict_params = NPatternParams(**cfg["n_pattern"])
    relaxed_params = replace(strict_params, high_win_mode=False)
    bt_cfg = BacktestConfig(**cfg["backtest"])

    files = sorted(cache_dir.glob("*.pkl"))
    if limit > 0:
        files = files[:limit]

    rows = []
    for p in files:
        try:
            df = _prepare_df(pickle.load(open(p, "rb")))
        except Exception:
            continue

        result = backtest_single_stock(p.stem, p.stem, df, relaxed_params, bt_cfg)
        for t in result.trades:
            rows.append(
                {
                    "code": t.code,
                    "name": t.name,
                    "entry_date": str(t.entry_date)[:10],
                    "exit_date": str(t.exit_date)[:10],
                    "profit_pct": float(t.profit_pct),
                    "strength": int(t.strength),
                    "factor_score": int(t.factor_score),
                    "ml_confidence": float(t.ml_confidence),
                    "sequence_confidence": float(getattr(t, "sequence_confidence", 0.5)),
                    "close_position_score": int(t.close_position_score),
                    "volatility_contraction_score": int(t.volatility_contraction_score),
                    "rr_ratio_proxy": round(abs(float(t.profit_pct)) / 2.0, 2),  # loose proxy for quick slicing
                    "layer_execution": (
                        t.ml_confidence >= strict_params.high_win_min_ml_confidence
                        and t.close_position_score >= strict_params.high_win_min_close_position_score
                        and t.volatility_contraction_score >= strict_params.high_win_min_volatility_contraction_score
                    ),
                    "layer_watchlist": (
                        t.ml_confidence >= scan_cfg.get("watchlist_min_ml_confidence", 0.30)
                        and t.strength >= scan_cfg.get("watchlist_min_strength", 90)
                    ),
                }
            )

    trades = pd.DataFrame(rows)
    if trades.empty:
        print("无交易")
        return

    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["composite"] = (
        trades["strength"] * (0.35 + trades["ml_confidence"] + trades["sequence_confidence"] * 0.45)
        + trades["factor_score"] * 0.35
    )

    execution = trades[trades["layer_execution"]].copy()
    watch_pool = trades[trades["layer_watchlist"]].copy()
    ranked = watch_pool.copy()
    ranked["day_rank"] = ranked.groupby("entry_date")["composite"].rank(method="first", ascending=False)
    watch_topn = ranked[ranked["day_rank"] <= top_n].copy()

    overlap = watch_topn[watch_topn["layer_execution"]].copy()

    summary_rows = [
        _summarize("all_relaxed", trades),
        _summarize("execution_only", execution),
        _summarize(f"watchlist_top{top_n}", watch_topn),
        _summarize(f"watchlist_top{top_n}_strict_overlap", overlap),
    ]
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))

    out_dir = PROJECT_ROOT / "data"
    summary_path = out_dir / f"dual_layer_cached_summary_top{top_n}.csv"
    trades_path = out_dir / f"dual_layer_cached_trades_top{top_n}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    watch_topn.to_csv(trades_path, index=False, encoding="utf-8-sig")
    print(f"\nsummary -> {summary_path}")
    print(f"trades -> {trades_path}")


if __name__ == "__main__":
    main()
