"""Build a K-line sequence dataset from backtest trades.

Usage:
    python3 scripts/build_kline_sequence_dataset.py [trades_csv] [output_npz]

The script tries local cache first (`data/bt_cache/<code>.pkl`), then falls back to
online mootdx fetch when needed.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from mootdx.quotes import Quotes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.strategy.kline_sequence import build_kline_tensor


def _load_cached_df(code: str):
    p = PROJECT_ROOT / "data" / "bt_cache" / f"{code}.pkl"
    if not p.exists():
        return None
    df = pickle.load(open(p, "rb"))
    df = df.copy()
    if "date" not in df.columns:
        if "datetime" in df.columns:
            df["date"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d")
        else:
            df["date"] = pd.to_datetime(df.index).strftime("%Y-%m-%d")
    if "volume" not in df.columns and "vol" in df.columns:
        df["volume"] = df["vol"]
    return df


def _fetch_online_df(code: str, client):
    df = client.bars(symbol=code, frequency=9, start=0, offset=700)
    if df is None or len(df) < 100:
        return None
    df = df.copy()
    if "date" not in df.columns:
        if "datetime" in df.columns:
            df["date"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d")
        else:
            df["date"] = pd.to_datetime(df.index).strftime("%Y-%m-%d")
    if "volume" not in df.columns and "vol" in df.columns:
        df["volume"] = df["vol"]
    return df


def main():
    trades_path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "data" / "backtest_trades_2y_mainboard_all.csv"
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else PROJECT_ROOT / "data" / "kline_sequences.npz"

    trades = pd.read_csv(trades_path)
    if trades.empty:
        raise SystemExit("交易文件为空")

    client = Quotes.factory(market="std", timeout=10)
    cache: dict[str, pd.DataFrame] = {}
    X_seq = []
    y = []
    meta = []

    for _, row in trades.iterrows():
        code = str(row["code"]).zfill(6)
        if code not in cache:
            df = _load_cached_df(code)
            if df is None:
                df = _fetch_online_df(code, client)
            cache[code] = df
        df = cache.get(code)
        if df is None or df.empty:
            continue

        date_series = pd.to_datetime(df["date"].astype(str).str[:10], errors="coerce")
        entry_date = pd.to_datetime(str(row["entry_date"])[:10], errors="coerce")
        matches = np.where(date_series == entry_date)[0]
        if len(matches) == 0:
            continue
        idx = int(matches[0])
        seq = build_kline_tensor(
            df["open"].values,
            df["high"].values,
            df["low"].values,
            df["close"].values,
            df["volume"].values,
            end_idx=idx,
            window=30,
        )
        if seq is None:
            continue
        X_seq.append(seq)
        y.append(1.0 if float(row["profit_pct"]) > 0 else 0.0)
        meta.append((code, str(row["entry_date"]), float(row["profit_pct"])))

    if not X_seq:
        raise SystemExit("没有成功构建任何序列样本")

    np.savez_compressed(
        output_path,
        X_seq=np.asarray(X_seq, dtype=np.float32),
        y=np.asarray(y, dtype=np.float32),
        meta=np.asarray(meta, dtype=object),
    )
    print(f"saved -> {output_path}")
    print(f"samples={len(X_seq)} positives={int(sum(y))} negatives={len(y) - int(sum(y))}")


if __name__ == "__main__":
    main()
