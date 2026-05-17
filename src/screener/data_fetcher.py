"""A 股数据获取层 — 封装 mootdx/akshare/腾讯财经等接口

复用 a-stock-data skill 的接口：
- mootdx: K线 + 盘口
- 腾讯财经: PE/PB/市值/涨跌停
- akshare: 股票列表/一致预期/龙虎榜
- 同花顺热点: 题材归因
- 百度股市通: 概念板块 + 资金流向
"""

import logging
import time
import urllib.request
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


# ====== 全 A 股列表 ======


def get_all_stocks() -> pd.DataFrame:
    """获取全 A 股列表（含 ST/停牌标记）"""
    import akshare as ak

    df = ak.stock_info_a_code_name()
    if df.empty:
        raise RuntimeError("无法获取股票列表（akshare 返回空）")
    # 列名可能是 代码/名称 或 code/name
    mapping = {}
    for col in df.columns:
        if "代码" in col or col.lower() == "code":
            mapping[col] = "code"
        elif "名称" in col or col.lower() == "name":
            mapping[col] = "name"
    df = df.rename(columns=mapping)
    return df


# ====== K 线数据 (mootdx) ======


def get_klines_mootdx(code: str, period: int = 4, count: int = 250) -> pd.DataFrame:
    """mootdx K线数据

    Args:
        code: 6 位股票代码
        period: 4=日线, 5=周线, 6=月线
        count: 取多少根 K 线

    Returns:
        DataFrame with columns: open, high, low, close, volume, amount, date
    """
    from mootdx.quotes import Quotes

    client = Quotes.factory(market="std")
    market = 1 if code.startswith(("6", "9")) else 0
    df = client.bars(symbol=code, category=period, offset=count)

    if df.empty:
        logger.warning(f"mootdx {code} K线返回空")
        return df

    # 日期处理：通达信格式 YYYYMMDD
    if "date" not in df.columns:
        # 从 index 提取
        pass
    df["date"] = pd.to_datetime(df["date"] if "date" in df.columns else df.index, format="%Y%m%d")
    return df


def get_daily_klines(code: str, days: int = 250) -> pd.DataFrame:
    """获取日线 K 线（带重试）"""
    for attempt in range(3):
        try:
            df = get_klines_mootdx(code, period=4, count=days)
            if not df.empty:
                return df
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"mootdx {code} 第{attempt + 1}次失败: {e}")
            time.sleep(1)
    return pd.DataFrame()


# ====== 腾讯财经行情 ======


def _get_prefix(code: str) -> str:
    """6位代码 → 市场前缀"""
    if code.startswith(("6", "9")):
        return "sh"
    elif code.startswith("8"):
        return "bj"
    return "sz"


def get_tencent_quotes(codes: list[str]) -> dict[str, dict]:
    """批量获取腾讯财经实时行情（PE/PB/市值/换手率/涨跌停）"""
    prefixed = [_get_prefix(c) + c for c in codes]
    url = "https://qt.gtimg.cn/q=" + ",".join(prefixed)
    req = urllib.request.Request(url)
    req.add_header("User-Agent", UA)
    resp = urllib.request.urlopen(req, timeout=10)
    data = resp.read().decode("gbk")

    result = {}
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


# ====== 同花顺热点（题材归因） ======


def get_hot_stocks(date: str = None) -> pd.DataFrame:
    """同花顺当日强势股 + 题材归因"""
    from datetime import date as _date

    if date is None:
        date = _date.today().strftime("%Y-%m-%d")

    url = (
        f"http://zx.10jqka.com.cn/event/api/getharden/"
        f"date/{date}/orderby/date/orderway/desc/charset/GBK/"
    )
    headers = {"User-Agent": UA}
    r = requests.get(url, headers=headers, timeout=10)
    data = r.json()
    if data.get("errocode", 0) != 0:
        raise RuntimeError(f"同花顺热点错误: {data.get('errormsg', '')}")

    rows = data.get("data") or []
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    rename_map = {
        "code": "代码", "name": "名称", "reason": "题材归因",
        "close": "收盘价", "zhangfu": "涨幅%",
        "huanshou": "换手率%", "chengjiaoe": "成交额",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
    return df


# ====== 行业对比 ======


def get_industry_comparison() -> pd.DataFrame:
    """同花顺行业板块涨跌排名"""
    import akshare as ak

    df = ak.stock_board_industry_summary_ths()
    return df


# ====== 北向资金 ======


def get_northbound_realtime() -> pd.DataFrame:
    """北向资金实时分钟流向"""
    headers = {
        "User-Agent": UA,
        "Host": "data.hexin.cn",
        "Referer": "https://data.hexin.cn/",
    }
    url = "https://data.hexin.cn/market/hsgtApi/method/dayChart/"
    r = requests.get(url, headers=headers, timeout=10)
    d = r.json()
    return pd.DataFrame({
        "time": d.get("time", []),
        "hgt_yi": d.get("hgt", []),
        "sgt_yi": d.get("sgt", []),
    })


# ====== 批量数据获取（扫描用） ======


def fetch_batch_daily_klines(
    codes: list[str],
    days: int = 250,
    delay: float = 0.3,
) -> dict[str, pd.DataFrame]:
    """批量获取日线 K 线

    Args:
        codes: 股票代码列表
        days: 回溯天数
        delay: 请求间隔（防封IP）

    Returns:
        {code: DataFrame}
    """
    results = {}
    total = len(codes)

    for i, code in enumerate(codes):
        try:
            df = get_daily_klines(code, days=days)
            if not df.empty and len(df) >= 60:
                results[code] = df
        except Exception as e:
            logger.debug(f"获取 {code} 失败: {e}")

        if (i + 1) % 50 == 0:
            logger.info(f"  K线进度: {i + 1}/{total}")

        if delay > 0 and i < total - 1:
            time.sleep(delay)

    return results


def filter_candidates(
    stocks: pd.DataFrame,
    quotes: dict[str, dict],
    min_mcap: float = 20,
    max_mcap: float = 500,
    min_amount: float = 5000,
    exclude_st: bool = True,
) -> list[str]:
    """初步筛选候选股票

    Args:
        stocks: 全 A 股列表 (columns: code, name)
        quotes: 腾讯行情 dict
        min_mcap: 最小市值(亿)
        max_mcap: 最大市值(亿)
        min_amount: 最小日成交额(万)
        exclude_st: 是否排除 ST

    Returns:
        通过筛选的股票代码列表
    """
    candidates = []

    for _, row in stocks.iterrows():
        code = str(row.get("code", "")).zfill(6)
        name = str(row.get("name", ""))

        # 排除 ST
        if exclude_st and ("ST" in name or "*ST" in name):
            continue

        # 排除退市整理
        if "退" in name:
            continue

        # 市值和成交额过滤（从腾讯行情）
        q = quotes.get(code, {})
        mcap = q.get("mcap_yi", 0)
        amount = q.get("amount_wan", 0)

        if mcap < min_mcap or mcap > max_mcap:
            continue
        if amount < min_amount:
            continue

        candidates.append(code)

    return candidates
