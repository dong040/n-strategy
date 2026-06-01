"""每日全市场 N 字战法扫描器

流程：
1. 获取主板 A 股列表 (akshare)
2. 逐只获取 K 线 (mootdx)
3. N 字形态匹配 + 涨停基因 + 多头排列过滤
4. 按信号强度排序输出
"""

import logging
import time
from dataclasses import dataclass, field, replace
from datetime import datetime

from ..strategy.n_pattern import NPatternParams, NSignal, scan_stock, score_fundamental
from .data_fetcher import get_live_factor_data, get_stock_industry_map

logger = logging.getLogger(__name__)


@dataclass
class ScanConfig:
    top_n: int = 3               # 每日输出前 N 只 (受 max_positions 约束)
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

    # 2.6 获取实时因子数据（同花顺热点/行业排名/北向资金/行业映射）
    logger.info("2.5/4 获取实时因子数据...")
    live_data = get_live_factor_data()
    try:
        industry_map = get_stock_industry_map()
        live_data["_sector_map"] = industry_map
        logger.info(f"  行业映射: {len(industry_map)} 只")
    except Exception:
        live_data["_sector_map"] = {}

    # 3. 逐只扫描
    logger.info(f"3/4 扫描形态 ({len(codes)} 只)...")
    all_signals = []
    fallback_signals = []
    errors = 0
    report_interval = max(1, len(codes) // 5)
    relaxed_params = replace(params, high_win_mode=False) if params.high_win_mode else params

    def _enrich_and_filter(signals: list, df, code: str, tier: str) -> list:
        if not signals:
            return []
        last_close = df['close'].values[-1]
        fin = score_fundamental(code, last_close, client)
        out = []
        for sig in signals:
            sig.pe = fin['pe']
            sig.pb = fin['pb']
            sig.net_profit_yi = fin['net_profit_yi']
            sig.fundamental_score = fin['score']
            sig.details = dict(sig.details or {})
            sig.details["selection_tier"] = tier
            if fin['is_garbage_profitable']:
                continue
            sig.strength += fin['score']
            out.append(sig)
        return out

    for idx, code in enumerate(codes):
        if idx % report_interval == 0:
            logger.info(f"  {idx}/{len(codes)}... ({len(all_signals)} found)")

        try:
            df = _fetch_klines(code, client)
            if df is None:
                continue

            signals = scan_stock(code, names.get(code, ''), df, params, market_pct, live_data=live_data)
            all_signals.extend(_enrich_and_filter(signals, df, code, "high_win"))

            # 高胜率模式下保留一份宽松候选池，只有严格池不足时才兜底使用。
            if params.high_win_mode and not signals:
                relaxed = scan_stock(
                    code, names.get(code, ''), df, relaxed_params, market_pct, live_data=live_data,
                )
                fallback_signals.extend(_enrich_and_filter(relaxed, df, code, "fallback"))
        except Exception:
            errors += 1
            continue

    # 4. 排序
    all_signals.sort(key=lambda s: s.strength, reverse=True)
    fallback_signals.sort(key=lambda s: s.strength, reverse=True)

    # 4.5 高胜率兜底：严格信号不足 top_n 时，才用宽松候选补齐并标注 selection_tier=fallback。
    if params.high_win_mode and len(all_signals) < scan_config.top_n:
        need = scan_config.top_n - len(all_signals)
        selected_codes = {s.code for s in all_signals}
        fill = [s for s in fallback_signals if s.code not in selected_codes][:need]
        all_signals = all_signals + fill
        logger.info(f"高胜率信号不足 ({len(all_signals) - len(fill)}/{scan_config.top_n})，兜底补齐 {len(fill)} 只")

    # 5. TradingAgents 多智能体二次打分 (Top N)
    ta_count = 0
    try:
        from .tradingagents_scorer import score_top_candidates
        top_n_ta = min(scan_config.top_n, 5)
        score_top_candidates(all_signals[:top_n_ta], top_n=top_n_ta)
        ta_count = top_n_ta
        logger.info(f"TradingAgents 二次打分完成 ({ta_count} 只)")
    except ImportError:
        logger.debug("TradingAgents 未安装，跳过二次打分")
    except Exception as e:
        logger.warning(f"TradingAgents 二次打分失败: {e}")

    elapsed = time.time() - start_time
    logger.info(f"4/4 完成: {len(all_signals)} 信号, {errors} 错误, 耗时 {elapsed:.0f}s")

    return ScanResult(
        date=today,
        total_scanned=len(codes),
        signals=all_signals[:scan_config.top_n],
        elapsed_seconds=round(elapsed, 1),
    )
