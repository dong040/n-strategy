"""每日全市场 N 字战法扫描器

流程：
1. 获取主板 A 股列表 (akshare)
2. 逐只获取 K 线 (mootdx)
3. N 字形态匹配 + 涨停基因 + 多头排列过滤
4. 按信号强度排序输出
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

from ..strategy.n_pattern import NPatternParams, NSignal, scan_stock, score_fundamental

logger = logging.getLogger(__name__)


@dataclass
class ScanConfig:
    top_n: int = 10
    markets: list = field(default_factory=lambda: ["sh", "sz"])
    exclude_st: bool = True
    exclude_suspend: bool = True
    main_board_only: bool = True
    industry_filter: list = field(default_factory=list)


@dataclass
class ScanResult:
    date: str
    total_scanned: int
    signals: list
    elapsed_seconds: float = 0


def _get_main_board_stocks():
    """获取主板股票列表（60xxxx + 000xxx/001xxx，排除ST）"""
    import akshare as ak
    stock_info = ak.stock_info_a_code_name()
    df = stock_info[['code', 'name']].copy()
    main = df[df['code'].str.match(r'^(60\d{4}|00[0-4]\d{3})$')].copy()
    main = main[~main['name'].str.contains('ST', na=False)]
    return main


def _fetch_klines(code: str, client, offset: int = 80):
    """获取单只股票日K线"""
    try:
        df = client.bars(symbol=code, frequency=9, start=0, offset=offset)
        if df is None or len(df) < 40:
            return None
        return df.reset_index(drop=True)
    except Exception:
        return None


def _get_market_pct(client) -> float:
    """获取今日上证指数涨跌幅(%)"""
    try:
        df = client.quotes(symbol='999999')
        if df is not None and not df.empty:
            price = float(df.iloc[-1]['price'])
            last_close = float(df.iloc[-1]['last_close'])
            if last_close > 0:
                return round((price - last_close) / last_close * 100, 2)
    except Exception:
        pass
    return 0


def run_daily_scan(
    params: NPatternParams = None,
    scan_config: ScanConfig = None,
) -> ScanResult:
    """执行每日全市场扫描"""
    if params is None:
        params = NPatternParams()
    if scan_config is None:
        scan_config = ScanConfig()

    start_time = time.time()
    today = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"=== N字战法 每日扫描 {today} ===")

    # 1. 获取主板股票列表
    logger.info("1/4 获取主板股票列表...")
    try:
        stocks_df = _get_main_board_stocks()
        codes = stocks_df['code'].tolist()
        names = dict(zip(stocks_df['code'], stocks_df['name']))
        logger.info(f"  主板 {len(codes)} 只")
    except Exception as e:
        logger.error(f"获取股票列表失败: {e}")
        return ScanResult(date=today, total_scanned=0, signals=[])

    # 2. 初始化 mootdx
    from mootdx.quotes import Quotes
    client = Quotes.factory(market='std', timeout=10)

    # 2.5 获取今日大盘涨跌
    market_pct = _get_market_pct(client)
    mkt_label = "强势" if market_pct > 0.5 else ("弱势" if market_pct < -0.5 else "平盘")
    logger.info(f"2/3 大盘 {market_pct:+.2f}% [{mkt_label}]")

    # 3. 逐只扫描
    logger.info(f"3/4 扫描形态 ({len(codes)} 只)...")
    all_signals = []
    errors = 0
    report_interval = max(1, len(codes) // 5)

    for idx, code in enumerate(codes):
        if idx % report_interval == 0:
            logger.info(f"  {idx}/{len(codes)}... ({len(all_signals)} found)")

        try:
            df = _fetch_klines(code, client)
            if df is None:
                continue

            signals = scan_stock(code, names.get(code, ''), df, params, market_pct)

            # 基本面过滤 — 亏损可留，盈利垃圾必除
            if signals:
                last_close = df['close'].values[-1]
                fin = score_fundamental(code, last_close, client)
                sig = signals[0]
                sig.pe = fin['pe']
                sig.pb = fin['pb']
                sig.net_profit_yi = fin['net_profit_yi']
                sig.fundamental_score = fin['score']
                if fin['is_garbage_profitable']:
                    continue  # 盈利但微利+高PE/小市值 → 垃圾股排除
                sig.strength += fin['score']

            all_signals.extend(signals)
        except Exception:
            errors += 1
            continue

    # 4. 排序
    all_signals.sort(key=lambda s: s.strength, reverse=True)

    elapsed = time.time() - start_time
    logger.info(f"4/4 完成: {len(all_signals)} 信号, {errors} 错误, 耗时 {elapsed:.0f}s")

    return ScanResult(
        date=today,
        total_scanned=len(codes),
        signals=all_signals[:scan_config.top_n],
        elapsed_seconds=round(elapsed, 1),
    )
