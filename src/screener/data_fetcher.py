"""A-share data access layer with local cache support.

This module keeps the existing public API but adds durable local caching for:
- daily K-lines, with incremental refresh on first run per day
- A-share universe list
- stock -> industry mapping

The cache lives under ``data/cache`` in the repository root. Daily K-lines are
stored per stock code so repeated backtests do not need to re-download the same
history every time.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from datetime import date as date_cls
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = PROJECT_ROOT / "data" / "cache"
DAILY_KLINE_CACHE_DIR = CACHE_ROOT / "daily_klines"
STATIC_CACHE_DIR = CACHE_ROOT / "static"
ALL_STOCKS_CACHE_PATH = STATIC_CACHE_DIR / "all_stocks.pkl"
INDUSTRY_MAP_CACHE_PATH = STATIC_CACHE_DIR / "industry_map.pkl"

DEFAULT_DAILY_FETCH_DAYS = 250
DEFAULT_DAILY_REFRESH_BUFFER = 10
STATIC_CACHE_TTL_DAYS = 7

_industry_cache: dict[str, str] = {}
_all_stocks_cache: pd.DataFrame | None = None


def _ensure_cache_dirs() -> None:
    DAILY_KLINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_stock_code(code: str) -> str:
    return str(code).strip().zfill(6)


def _normalize_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw K-line frame into the schema expected by the strategy."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount"])

    frame = df.copy()

    if "date" not in frame.columns:
        for candidate in ("date", "datetime"):
            if candidate in frame.columns:
                frame = frame.rename(columns={candidate: "date"})
                break
        if "date" not in frame.columns:
            try:
                frame = frame.reset_index()
            except ValueError:
                frame = frame.reset_index(drop=True)
            for candidate in ("date", "datetime", "index"):
                if candidate in frame.columns:
                    frame = frame.rename(columns={candidate: "date"})
                    break
    if "date" in frame.columns:
        dt = pd.to_datetime(frame["date"], errors="coerce")
        if dt.isna().all():
            dt = pd.to_datetime(frame["date"].astype(str).str.slice(0, 8), format="%Y%m%d", errors="coerce")
        frame["date"] = dt

    rename_map = {
        "vol": "volume",
        "amount_wan": "amount",
        "amountWan": "amount",
    }
    frame = frame.rename(columns={k: v for k, v in rename_map.items() if k in frame.columns})

    # Ensure required columns exist.
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in frame.columns:
            frame[col] = np.nan
    if "amount" not in frame.columns:
        frame["amount"] = np.nan

    keep_cols = ["date", "open", "high", "low", "close", "volume", "amount"]
    frame = frame[keep_cols]
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return frame


def _load_pickle_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_pickle(path)
    except Exception as exc:
        logger.warning("Failed to load cache %s: %s", path, exc)
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    return _normalize_ohlcv_frame(df) if "date" in df.columns or not isinstance(df.index, pd.RangeIndex) else df


def _save_pickle_frame(path: Path, df: pd.DataFrame) -> None:
    _ensure_cache_dirs()
    df.to_pickle(path)


def _cache_age_days(path: Path) -> int | None:
    if not path.exists():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime).date()
    return (date_cls.today() - mtime).days


def _merge_ohlcv_frames(base: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    if base is None or base.empty:
        return _normalize_ohlcv_frame(fresh)
    if fresh is None or fresh.empty:
        return _normalize_ohlcv_frame(base)

    merged = pd.concat([base, fresh], ignore_index=True)
    merged = _normalize_ohlcv_frame(merged)
    return merged


# ====== A-share universe ======


def get_all_stocks() -> pd.DataFrame:
    """Fetch the A-share universe and cache it locally."""
    global _all_stocks_cache
    _ensure_cache_dirs()

    if _all_stocks_cache is not None:
        return _all_stocks_cache.copy()

    if ALL_STOCKS_CACHE_PATH.exists() and _cache_age_days(ALL_STOCKS_CACHE_PATH) is not None:
        age = _cache_age_days(ALL_STOCKS_CACHE_PATH)
        if age is not None and age <= STATIC_CACHE_TTL_DAYS:
            try:
                cached = pd.read_pickle(ALL_STOCKS_CACHE_PATH)
                if isinstance(cached, pd.DataFrame) and not cached.empty:
                    _all_stocks_cache = cached
                    return cached.copy()
            except Exception:
                pass

    import akshare as ak

    df = ak.stock_info_a_code_name()
    if df.empty:
        raise RuntimeError("Unable to fetch A-share stock list (akshare returned empty).")

    mapping = {}
    for col in df.columns:
        lower = str(col).lower()
        if "代码" in str(col) or lower == "code":
            mapping[col] = "code"
        elif "名称" in str(col) or lower == "name":
            mapping[col] = "name"
    df = df.rename(columns=mapping)
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.zfill(6)

    _save_pickle_frame(ALL_STOCKS_CACHE_PATH, df)
    _all_stocks_cache = df
    return df.copy()


# ====== K-line data (mootdx) ======


def get_klines_mootdx(code: str, period: int = 4, count: int = 250) -> pd.DataFrame:
    """Fetch raw K-line data from mootdx."""
    from mootdx.quotes import Quotes

    code = _normalize_stock_code(code)
    client = Quotes.factory(market="std")
    df = client.bars(symbol=code, category=period, offset=count)

    if df is None or df.empty:
        logger.warning("mootdx returned empty K-lines for %s", code)
        return pd.DataFrame()

    frame = df.copy()
    if "date" not in frame.columns:
        for candidate in ("date", "datetime"):
            if candidate in frame.columns:
                frame = frame.rename(columns={candidate: "date"})
                break
        if "date" not in frame.columns:
            try:
                frame = frame.reset_index()
            except ValueError:
                frame = frame.reset_index(drop=True)
            for candidate in ("date", "datetime", "index"):
                if candidate in frame.columns:
                    frame = frame.rename(columns={candidate: "date"})
                    break
    return _normalize_ohlcv_frame(frame)


def _daily_kline_cache_path(code: str) -> Path:
    return DAILY_KLINE_CACHE_DIR / f"{_normalize_stock_code(code)}.pkl"


def _refresh_window_days(requested_days: int, cached_rows: int, cache_age_days: int | None) -> int:
    """Choose a recent refresh window large enough to cover stale local caches."""
    if cache_age_days is None:
        return requested_days

    if cached_rows < requested_days:
        return max(requested_days, cache_age_days + DEFAULT_DAILY_REFRESH_BUFFER, 30)

    # For an already sufficient cache, only fetch the recent overlap window.
    return min(max(cache_age_days + DEFAULT_DAILY_REFRESH_BUFFER, 30), max(cached_rows, requested_days))


def get_daily_klines(
    code: str,
    days: int = DEFAULT_DAILY_FETCH_DAYS,
    *,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch daily K-lines with a local incremental cache.

    Behavior:
    - First call: fetch ``days`` bars and save them locally.
    - Later calls on the same day: return the cache immediately.
    - New day: fetch a recent overlap window, merge, de-duplicate by date, and
      keep the merged full history on disk.
    """
    code = _normalize_stock_code(code)
    cache_path = _daily_kline_cache_path(code)
    _ensure_cache_dirs()

    cached = _normalize_ohlcv_frame(_load_pickle_frame(cache_path)) if cache_path.exists() else pd.DataFrame()
    age_days = _cache_age_days(cache_path)

    if use_cache and not force_refresh and not cached.empty and age_days == 0 and len(cached) >= days:
        return cached.tail(days).reset_index(drop=True)

    # No usable cache or a new day has started: refresh from the network.
    if cached.empty or force_refresh:
        fetch_days = days
    else:
        fetch_days = _refresh_window_days(days, len(cached), age_days)

    fresh = get_klines_mootdx(code, period=4, count=fetch_days)
    if fresh.empty:
        return cached.tail(days).reset_index(drop=True) if not cached.empty else pd.DataFrame()

    merged = _merge_ohlcv_frames(cached, fresh)
    if not merged.empty:
        _save_pickle_frame(cache_path, merged)

    if days > 0:
        return merged.tail(days).reset_index(drop=True)
    return merged.reset_index(drop=True)


def sync_daily_klines(code: str, days: int = DEFAULT_DAILY_FETCH_DAYS) -> pd.DataFrame:
    """Force a refresh of a stock's daily K-lines and return the merged frame."""
    return get_daily_klines(code, days=days, use_cache=True, force_refresh=True)


# ====== Tencent quotes ======


def _get_prefix(code: str) -> str:
    if code.startswith(("6", "9")):
        return "sh"
    if code.startswith("8"):
        return "bj"
    return "sz"


def get_tencent_quotes(codes: list[str]) -> dict[str, dict]:
    """Batch fetch real-time quotes from Tencent."""
    prefixed = [_get_prefix(c) + _normalize_stock_code(c) for c in codes]
    url = "https://qt.gtimg.cn/q=" + ",".join(prefixed)
    req = urllib.request.Request(url)
    req.add_header("User-Agent", UA)
    resp = urllib.request.urlopen(req, timeout=10)
    data = resp.read().decode("gbk", errors="ignore")

    result: dict[str, dict] = {}
    for line in data.strip().split(";"):
        if not line.strip() or "=" not in line or '"' not in line:
            continue
        key = line.split("=")[0].split("_")[-1]
        vals = line.split('"')[1].split("~")
        if len(vals) < 53:
            continue
        code = key[2:]
        result[code] = {
            "name": vals[1],
            "price": float(vals[3]) if vals[3] else 0,
            "last_close": float(vals[4]) if vals[4] else 0,
            "open": float(vals[5]) if vals[5] else 0,
            "change_pct": float(vals[32]) if vals[32] else 0,
            "high": float(vals[33]) if vals[33] else 0,
            "low": float(vals[34]) if vals[34] else 0,
            "amount_wan": float(vals[37]) if vals[37] else 0,
            "turnover_pct": float(vals[38]) if vals[38] else 0,
            "pe_ttm": float(vals[39]) if vals[39] else 0,
            "amplitude_pct": float(vals[43]) if vals[43] else 0,
            "mcap_yi": float(vals[44]) if vals[44] else 0,
            "float_mcap_yi": float(vals[45]) if vals[45] else 0,
            "pb": float(vals[46]) if vals[46] else 0,
            "limit_up": float(vals[47]) if vals[47] else 0,
            "limit_down": float(vals[48]) if vals[48] else 0,
            "vol_ratio": float(vals[49]) if vals[49] else 0,
        }
    return result


# ====== Hot stocks and industry data ======


def get_hot_stocks(date: str | None = None) -> pd.DataFrame:
    """Fetch the current hot-stock list from 10jqka."""
    from datetime import date as _date

    if date is None:
        date = _date.today().strftime("%Y-%m-%d")

    url = (
        "http://zx.10jqka.com.cn/event/api/getharden/"
        f"date/{date}/orderby/date/orderway/desc/charset/GBK/"
    )
    headers = {"User-Agent": UA}
    r = requests.get(url, headers=headers, timeout=10)
    data = r.json()
    if data.get("errocode", 0) != 0:
        raise RuntimeError(f"Hot-stock fetch failed: {data.get('errormsg', '')}")

    rows = data.get("data") or []
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    rename_map = {
        "code": "代码",
        "name": "名称",
        "reason": "题材归因",
        "close": "收盘价",
        "zhangfu": "涨幅%",
        "huanshou": "换手率",
        "chengjiaoe": "成交额",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
    return df


def get_industry_comparison() -> pd.DataFrame:
    """Fetch industry board performance from Tonghuashun."""
    import akshare as ak

    df = ak.stock_board_industry_summary_ths()
    return df


def get_stock_industry_map() -> dict[str, str]:
    """Build and cache a stock-code -> industry-name map."""
    global _industry_cache
    _ensure_cache_dirs()

    if _industry_cache:
        return _industry_cache

    if INDUSTRY_MAP_CACHE_PATH.exists():
        age = _cache_age_days(INDUSTRY_MAP_CACHE_PATH)
        if age is not None and age <= STATIC_CACHE_TTL_DAYS:
            try:
                cached = pd.read_pickle(INDUSTRY_MAP_CACHE_PATH)
                if isinstance(cached, dict) and cached:
                    _industry_cache = cached
                    return _industry_cache
            except Exception:
                pass

    import akshare as ak

    try:
        industry_df = ak.stock_board_industry_summary_ths()
        if industry_df.empty:
            return {}

        name_col = next((c for c in ["板块名称", "name", "行业"] if c in industry_df.columns), industry_df.columns[0])
        industry_names = industry_df[name_col].tolist()
        logger.info("Building industry map from %s industries", len(industry_names))

        code_to_industry: dict[str, str] = {}
        for idx, name in enumerate(industry_names):
            try:
                cons_df = ak.stock_board_industry_cons_ths(symbol=name)
                if cons_df is not None and not cons_df.empty:
                    code_col = "代码" if "代码" in cons_df.columns else "code"
                    for code in cons_df[code_col]:
                        code_to_industry[_normalize_stock_code(code)] = name
            except Exception:
                continue
            if (idx + 1) % 20 == 0:
                logger.debug("Industry map progress: %s/%s", idx + 1, len(industry_names))

        _industry_cache = code_to_industry
        pd.to_pickle(code_to_industry, INDUSTRY_MAP_CACHE_PATH)
        logger.info("Industry map ready: %s stocks", len(code_to_industry))
        return code_to_industry

    except Exception as e:
        logger.warning("Failed to build industry map: %s", e)
        return {}


def get_live_factor_data() -> dict:
    """Fetch current market factors once per scan."""
    result = {
        "hot_code_set": set(),
        "sector_rank": {},
        "northbound_score": 0,
    }

    # 1. Hot stocks
    try:
        hot_df = get_hot_stocks()
        if not hot_df.empty:
            code_col = next((c for c in ["代码", "code"] if c in hot_df.columns), None)
            if code_col:
                result["hot_code_set"] = set(_normalize_stock_code(c) for c in hot_df[code_col])
                logger.info("Hot-stock set loaded: %s", len(result["hot_code_set"]))
    except Exception as e:
        logger.debug("Hot-stock fetch failed: %s", e)

    # 2. Sector ranking
    try:
        ind_df = get_industry_comparison()
        if not ind_df.empty and "涨跌幅" in ind_df.columns:
            sorted_df = ind_df.sort_values("涨跌幅", ascending=False)
            total = len(sorted_df)
            for rank, (_, row) in enumerate(sorted_df.iterrows()):
                name = row.get("板块名称", "")
                result["sector_rank"][name] = rank / max(total - 1, 1)
    except Exception as e:
        logger.debug("Industry comparison failed: %s", e)

    # 3. Northbound flow
    try:
        import akshare as ak

        try:
            north = ak.stock_hsgt_fund_flow_em().head(1)
            if not north.empty:
                val = float(north.iloc[0].get("北向资金净流入-沪深股通", 0) or 0)
                result["northbound_score"] = 5 if val > 0 else (-5 if val < 0 else 0)
        except Exception:
            pass
    except Exception:
        pass

    return result
