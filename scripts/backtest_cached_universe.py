"""Backtest the current strategy on locally cached OHLCV files.

Cache directory format: data/bt_cache/<code>.pkl
Each pickle should contain a pandas DataFrame with OHLCV columns.
"""

from __future__ import annotations

import os
import pickle
import sys
from collections import Counter
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
            raise ValueError("cache dataframe missing date/datetime column")
    if "volume" not in out.columns and "vol" in out.columns:
        out["volume"] = out["vol"]
    return out


def main():
    cache_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "data" / "bt_cache"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    params = NPatternParams(**cfg["n_pattern"])
    config = BacktestConfig(**cfg["backtest"])

    files = sorted(cache_dir.glob("*.pkl"))
    if limit > 0:
        files = files[:limit]

    trades = []
    for p in files:
        try:
            df = _prepare_df(pickle.load(open(p, "rb")))
            result = backtest_single_stock(p.stem, p.stem, df, params, config)
            trades.extend(result.trades)
        except Exception as exc:
            print(f"skip {p.name}: {exc}")

    if not trades:
        print("无交易")
        return

    wins = [t for t in trades if t.profit > 0]
    losses = [t for t in trades if t.profit <= 0]
    profits = [t.profit_pct for t in trades]
    profit_factor = (
        sum(t.profit for t in wins) / abs(sum(t.profit for t in losses))
        if losses else 999
    )
    print(f"样本股票: {len(files)}")
    print(f"交易数: {len(trades)}")
    print(f"胜率: {len(wins) / len(trades) * 100:.2f}%")
    print(f"平均收益: {sum(profits) / len(profits):.2f}%")
    print(f"中位数收益: {pd.Series(profits).median():.2f}%")
    print(f"均盈: {sum(t.profit_pct for t in wins) / len(wins):.2f}%" if wins else "均盈: 0.00%")
    print(f"均损: {sum(t.profit_pct for t in losses) / len(losses):.2f}%" if losses else "均损: 0.00%")
    print(f"盈利因子: {profit_factor:.2f}")
    print(f"出场分布: {dict(Counter(t.exit_reason for t in trades))}")


if __name__ == "__main__":
    main()
