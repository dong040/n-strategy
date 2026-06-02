"""Online benchmark backtest for A-shares.

Despite the historical filename, this script is now the canonical benchmark
entrypoint and defaults to:

- Universe: all A-shares
- Window: trailing 1 year
- Data source: akshare universe + mootdx daily bars
- Outputs: trade ledger, summary JSON, Top-3/day ledger, per-run universe/errors

Examples:
    python3 scripts/backtest_online_mainboard_2y.py
    python3 scripts/backtest_online_mainboard_2y.py --universe mainboard --exclude-st
    python3 scripts/backtest_online_mainboard_2y.py --start 0 --end 1000 --tag shard1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml
from mootdx.quotes import Quotes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.screener.data_fetcher import get_all_stocks
from src.strategy.backtest import BacktestConfig, backtest_single_stock
from src.strategy.n_pattern import NPatternParams

OUT_DIR = PROJECT_ROOT / "data"
OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = OUT_DIR / "bt_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _load_config() -> tuple[dict, NPatternParams, BacktestConfig]:
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg, NPatternParams(**cfg["n_pattern"]), BacktestConfig(**cfg["backtest"])


def _build_universe(universe_name: str, exclude_st: bool) -> list[tuple[str, str]]:
    stock_info = get_all_stocks()[["code", "name"]].copy()
    stock_info["code"] = stock_info["code"].astype(str).str.zfill(6)
    stock_info["name"] = stock_info["name"].astype(str)

    if universe_name == "mainboard":
        stock_info = stock_info[stock_info["code"].str.match(r"^(60\d{4}|00[0-4]\d{3})$")].copy()
    else:
        stock_info = stock_info[stock_info["code"].str.match(r"^\d{6}$")].copy()

    if exclude_st:
        stock_info = stock_info[~stock_info["name"].str.contains("ST", na=False)].copy()

    stock_info = stock_info.drop_duplicates(subset=["code"]).reset_index(drop=True)
    return list(stock_info.itertuples(index=False, name=None))


def _prepare_df(df: pd.DataFrame, warmup_from: pd.Timestamp, date_to: pd.Timestamp) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns:
        if "datetime" in out.columns:
            out["date"] = pd.to_datetime(out["datetime"])
        else:
            out["date"] = pd.to_datetime(out.index)
    else:
        out["date"] = pd.to_datetime(out["date"])
    if "volume" not in out.columns and "vol" in out.columns:
        out["volume"] = out["vol"]
    out = out[(out["date"] >= warmup_from) & (out["date"] <= date_to)].copy()
    if len(out) < 140:
        return out.iloc[0:0].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)


def _load_or_fetch_bars(code: str, client, refresh: bool = False, offset: int = 700) -> pd.DataFrame | None:
    cache_path = CACHE_DIR / f"{code}.pkl"
    if cache_path.exists() and not refresh:
        try:
            return pd.read_pickle(cache_path)
        except Exception:
            pass

    df = client.bars(symbol=code, frequency=9, start=0, offset=offset)
    if df is None or len(df) < 150:
        return None
    try:
        df.to_pickle(cache_path)
    except Exception:
        pass
    return df


def _trade_rows(result, date_from: pd.Timestamp, date_to: pd.Timestamp) -> list[dict]:
    rows = []
    for t in result.trades:
        entry_ts = pd.to_datetime(t.entry_date)
        if entry_ts < date_from or entry_ts > date_to:
            continue
        rows.append(
            {
                "code": t.code,
                "name": t.name,
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "shares": t.shares,
                "profit": t.profit,
                "profit_pct": t.profit_pct,
                "strength": t.strength,
                "exit_reason": t.exit_reason,
                "factor_score": t.factor_score,
                "pullback_volume_score": t.pullback_volume_score,
                "turnover_crowding_score": t.turnover_crowding_score,
                "relative_strength_score": t.relative_strength_score,
                "volatility_contraction_score": t.volatility_contraction_score,
                "support_reclaim_score": t.support_reclaim_score,
                "close_position_score": t.close_position_score,
                "limit_up_followthrough_score": t.limit_up_followthrough_score,
                "theme_heat_score": t.theme_heat_score,
                "amount_quality_score": t.amount_quality_score,
                "market_regime_score": t.market_regime_score,
                "northbound_flow_score": t.northbound_flow_score,
                "rsi_divergence_score": t.rsi_divergence_score,
                "macd_signal_score": t.macd_signal_score,
                "ma_alignment_score": t.ma_alignment_score,
                "boll_squeeze_score": t.boll_squeeze_score,
                "kdj_oversold_score": t.kdj_oversold_score,
                "mfi_score": t.mfi_score,
                "shadow_quality_score": t.shadow_quality_score,
                "pullback_speed_score": t.pullback_speed_score,
                "intraday_reversal_score": t.intraday_reversal_score,
                "volume_climax_score": t.volume_climax_score,
                "sector_relative_score": t.sector_relative_score,
                "adx_trend_score": t.adx_trend_score,
                "obv_accumulation_score": t.obv_accumulation_score,
                "cmf_score": t.cmf_score,
                "gap_support_score": t.gap_support_score,
                "ml_confidence": t.ml_confidence,
                "ml_confidence_score": t.ml_confidence_score,
                "sequence_confidence": getattr(t, "sequence_confidence", 0.0),
                "sequence_score": getattr(t, "sequence_score", 0),
            }
        )
    return rows


def _summarize_trades(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "avg_profit_pct": 0,
            "median_profit_pct": 0,
            "avg_win_pct": 0,
            "avg_loss_pct": 0,
            "profit_factor": 0,
            "avg_hold_days": 0,
            "exit_reasons": {},
        }

    wins = trades_df[trades_df["profit"] > 0].copy()
    losses = trades_df[trades_df["profit"] <= 0].copy()
    hold_days = (pd.to_datetime(trades_df["exit_date"]) - pd.to_datetime(trades_df["entry_date"])).dt.days
    total_loss = abs(losses["profit"].sum()) if not losses.empty else 0
    profit_factor = wins["profit"].sum() / total_loss if total_loss > 0 else 999
    return {
        "trades": int(len(trades_df)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": round(len(wins) / len(trades_df) * 100, 4),
        "avg_profit_pct": round(float(trades_df["profit_pct"].mean()), 4),
        "median_profit_pct": round(float(trades_df["profit_pct"].median()), 4),
        "avg_win_pct": round(float(wins["profit_pct"].mean()), 4) if not wins.empty else 0,
        "avg_loss_pct": round(float(losses["profit_pct"].mean()), 4) if not losses.empty else 0,
        "profit_factor": round(float(profit_factor), 4),
        "avg_hold_days": round(float(hold_days.mean()), 4),
        "exit_reasons": dict(Counter(trades_df["exit_reason"])),
    }


def _topn_by_day(trades_df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df.copy()
    ranked = trades_df.copy()
    ranked["entry_date"] = pd.to_datetime(ranked["entry_date"])
    ranked["composite"] = (
        ranked["strength"] * (0.35 + ranked["ml_confidence"] + ranked["sequence_confidence"] * 0.45)
        + ranked["factor_score"] * 0.35
    )
    ranked["day_rank"] = ranked.groupby("entry_date")["composite"].rank(method="first", ascending=False)
    return ranked[ranked["day_rank"] <= top_n].copy()


def _strict_subset(trades_df: pd.DataFrame, params: NPatternParams) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df.copy()
    mask = (
        (trades_df["ml_confidence"] >= params.high_win_min_ml_confidence)
        & (trades_df["close_position_score"] >= params.high_win_min_close_position_score)
        & (trades_df["volatility_contraction_score"] >= params.high_win_min_volatility_contraction_score)
    )
    return trades_df[mask].copy()


def _watchlist_pool(trades_df: pd.DataFrame, scan_cfg: dict) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df.copy()
    mask = (
        (trades_df["ml_confidence"] >= scan_cfg.get("watchlist_min_ml_confidence", 0.30))
        & (trades_df["sequence_confidence"] >= scan_cfg.get("watchlist_min_sequence_confidence", 0.40))
        & (trades_df["strength"] >= scan_cfg.get("watchlist_min_strength", 90))
        & (trades_df["factor_score"] >= scan_cfg.get("watchlist_min_factor_score", 0))
        & (trades_df["close_position_score"] >= scan_cfg.get("watchlist_min_close_position_score", -10))
        & (
            trades_df["volatility_contraction_score"]
            >= scan_cfg.get("watchlist_min_volatility_contraction_score", 0)
        )
    )
    return trades_df[mask].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="0-based inclusive start index")
    parser.add_argument("--end", type=int, default=0, help="0-based exclusive end index, 0 means all")
    parser.add_argument("--tag", type=str, default="", help="suffix tag for output files")
    parser.add_argument("--universe", type=str, choices=["all", "mainboard"], default="all")
    parser.add_argument("--exclude-st", action="store_true", help="exclude ST stocks")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--date-to", type=str, default="")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--high-win", type=str, choices=["on", "off"], default="on")
    parser.add_argument("--eval-mode", type=str, choices=["strict", "relaxed", "dual"], default="dual")
    args = parser.parse_args()

    date_to = pd.Timestamp(args.date_to) if args.date_to else pd.Timestamp.today().normalize()
    date_from = date_to - pd.Timedelta(days=max(args.lookback_days - 1, 1))
    warmup_from = date_from - pd.Timedelta(days=260)

    suffix_bits = [
        f"1y_{args.universe}",
        "noST" if args.exclude_st else "withST",
        date_from.strftime("%Y%m%d"),
        date_to.strftime("%Y%m%d"),
    ]
    if args.tag:
        suffix_bits.append(args.tag)
    suffix = "_".join(suffix_bits)

    summary_path = OUT_DIR / f"backtest_results_{suffix}.json"
    trades_path = OUT_DIR / f"backtest_trades_{suffix}.csv"
    top3_path = OUT_DIR / f"backtest_top3_{suffix}.csv"
    strict_path = OUT_DIR / f"backtest_execution_{suffix}.csv"
    watch_top3_path = OUT_DIR / f"backtest_watchtop3_{suffix}.csv"
    universe_path = OUT_DIR / f"backtest_universe_{suffix}.csv"
    errors_path = OUT_DIR / f"backtest_errors_{suffix}.csv"

    cfg, params, config = _load_config()
    scan_cfg = cfg.get("screener", {})
    if args.eval_mode in {"relaxed", "dual"} or args.high_win == "off":
        params = replace(params, high_win_mode=False)
    universe = _build_universe(args.universe, args.exclude_st)
    if args.end and args.end > args.start:
        universe = universe[args.start:args.end]
    elif args.start > 0:
        universe = universe[args.start:]
    pd.DataFrame(universe, columns=["code", "name"]).to_csv(universe_path, index=False, encoding="utf-8-sig")
    print(f"universe_size={len(universe)}", flush=True)

    client = Quotes.factory(market="std", timeout=10)
    all_trades = []
    errors = []
    no_data = 0
    success = 0
    t0 = time.time()

    for idx, (code, name) in enumerate(universe, start=1):
        try:
            raw_df = _load_or_fetch_bars(code, client, refresh=args.refresh_cache)
            if raw_df is None or len(raw_df) < 150:
                no_data += 1
                continue
            df = _prepare_df(raw_df, warmup_from, date_to)
            if df.empty:
                no_data += 1
                continue

            result = backtest_single_stock(code, name, df, params, config)
            success += 1
            all_trades.extend(_trade_rows(result, date_from, date_to))
        except Exception as exc:
            errors.append({"code": code, "name": name, "error": str(exc)[:300]})

        if idx % 20 == 0:
            elapsed = time.time() - t0
            speed = idx / elapsed if elapsed > 0 else 0
            remain = (len(universe) - idx) / speed if speed > 0 else 0
            print(
                json.dumps(
                    {
                        "progress": f"{idx}/{len(universe)}",
                        "success": success,
                        "no_data": no_data,
                        "errors": len(errors),
                        "trades": len(all_trades),
                        "elapsed_min": round(elapsed / 60, 1),
                        "eta_min": round(remain / 60, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    elapsed = time.time() - t0
    trades_df = pd.DataFrame(all_trades)
    errors_df = pd.DataFrame(errors)
    top3_df = _topn_by_day(trades_df, top_n=3)
    strict_df = _strict_subset(trades_df, NPatternParams(**cfg["n_pattern"]))
    watch_pool_df = _watchlist_pool(trades_df, scan_cfg)
    watch_top3_df = _topn_by_day(watch_pool_df, top_n=int(scan_cfg.get("watchlist_top_n", 3)))

    if not trades_df.empty:
        trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
    if not top3_df.empty:
        top3_df.to_csv(top3_path, index=False, encoding="utf-8-sig")
    if not strict_df.empty:
        strict_df.to_csv(strict_path, index=False, encoding="utf-8-sig")
    if not watch_top3_df.empty:
        watch_top3_df.to_csv(watch_top3_path, index=False, encoding="utf-8-sig")
    if not errors_df.empty:
        errors_df.to_csv(errors_path, index=False, encoding="utf-8-sig")

    summary = {
        "date_from": str(date_from.date()),
        "date_to": str(date_to.date()),
        "warmup_from": str(warmup_from.date()),
        "lookback_days": args.lookback_days,
        "universe_name": args.universe,
        "exclude_st": args.exclude_st,
        "universe_size": int(len(universe)),
        "success": int(success),
        "no_data": int(no_data),
        "errors": int(len(errors)),
        "elapsed_sec": round(elapsed, 1),
        "config_min_strength": config.min_strength,
        "high_win_mode": params.high_win_mode,
        "eval_mode": args.eval_mode,
        "ml_threshold": getattr(params, "high_win_min_ml_confidence", None),
        "cache_dir": str(CACHE_DIR),
    }
    summary["all_trades"] = _summarize_trades(trades_df)
    summary["top3_per_day"] = _summarize_trades(top3_df)
    summary["top3_per_day"]["signal_days"] = int(top3_df["entry_date"].nunique()) if not top3_df.empty else 0
    summary["top3_per_day"]["avg_trades_per_signal_day"] = (
        round(len(top3_df) / max(top3_df["entry_date"].nunique(), 1), 4) if not top3_df.empty else 0
    )
    summary["execution_layer"] = _summarize_trades(strict_df)
    summary["watchlist_pool"] = _summarize_trades(watch_pool_df)
    summary["watchlist_top3"] = _summarize_trades(watch_top3_df)
    summary["watchlist_top3"]["signal_days"] = int(watch_top3_df["entry_date"].nunique()) if not watch_top3_df.empty else 0
    summary["watchlist_top3"]["avg_trades_per_signal_day"] = (
        round(len(watch_top3_df) / max(watch_top3_df["entry_date"].nunique(), 1), 4) if not watch_top3_df.empty else 0
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("FINAL_SUMMARY", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
