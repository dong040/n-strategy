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


def _rank_score(sig: NSignal) -> float:
    seq_prob = getattr(sig, "sequence_confidence", 0.5)
    ml_prob = getattr(sig, "ml_confidence", 0.5)
    rr_ratio = max(getattr(sig, "rr_ratio", 0.0), 0.0)
    return (
        sig.strength * (0.35 + ml_prob + 0.45 * seq_prob)
        + getattr(sig, "factor_score", 0) * 0.35
        + min(rr_ratio, 5.0) * 8.0
    )


def _sort_signals(signals: list[NSignal]) -> list[NSignal]:
    return sorted(signals, key=_rank_score, reverse=True)


@dataclass
class ScanConfig:
    top_n: int = 3               # 兼容旧配置，等价于 watchlist_top_n
    markets: list = field(default_factory=lambda: ["sh", "sz"])
    exclude_st: bool = True
    exclude_suspend: bool = True
    main_board_only: bool = True
    industry_filter: list = field(default_factory=list)
    enable_tradingagents: bool = False
    execution_top_n: int = 3
    watchlist_top_n: int = 3
    watchlist_min_ml_confidence: float = 0.30
    watchlist_min_sequence_confidence: float = 0.40
    watchlist_min_rr: float = 1.2
    watchlist_min_strength: int = 80
    watchlist_min_factor_score: int = 0
    watchlist_min_close_position_score: int = -10
    watchlist_min_volatility_contraction_score: int = 0


@dataclass
class ScanResult:
    date: str
    total_scanned: int
    signals: list
    execution_signals: list = field(default_factory=list)
    watchlist_signals: list = field(default_factory=list)
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
    if not getattr(scan_config, "watchlist_top_n", 0):
        scan_config.watchlist_top_n = scan_config.top_n

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
    fallback_relaxed_raw = []
    errors = 0
    report_interval = max(1, len(codes) // 5)
    relaxed_params = replace(params, high_win_mode=False) if params.high_win_mode else params

    def _enrich_and_filter(signals: list, df, code: str, tier: str, layer: str) -> list:
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
            sig.details["strategy_layer"] = layer
            if fin['is_garbage_profitable']:
                continue
            sig.strength += fin['score']
            sig.details["rank_score"] = round(_rank_score(sig), 2)
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
            all_signals.extend(_enrich_and_filter(signals, df, code, "high_win", "execution"))

            # 高胜率模式下保留一份宽松候选池，只有严格池不足时才兜底使用。
            if params.high_win_mode and not signals:
                relaxed = scan_stock(code, names.get(code, ''), df, relaxed_params, market_pct, live_data=live_data)
                relaxed = _enrich_and_filter(relaxed, df, code, "fallback", "watchlist")
                fallback_relaxed_raw.extend(relaxed)
                for sig in relaxed:
                    if sig.ml_confidence < scan_config.watchlist_min_ml_confidence:
                        continue
                    if getattr(sig, "sequence_confidence", 0.5) < scan_config.watchlist_min_sequence_confidence:
                        continue
                    if sig.rr_ratio < scan_config.watchlist_min_rr:
                        continue
                    if sig.strength < scan_config.watchlist_min_strength:
                        continue
                    if getattr(sig, "factor_score", 0) < scan_config.watchlist_min_factor_score:
                        continue
                    if getattr(sig, "close_position_score", 0) < scan_config.watchlist_min_close_position_score:
                        continue
                    if getattr(sig, "volatility_contraction_score", 0) < scan_config.watchlist_min_volatility_contraction_score:
                        continue
                    fallback_signals.append(sig)
        except Exception:
            errors += 1
            continue

    # 4. 排序
    all_signals = _sort_signals(all_signals)
    fallback_signals = _sort_signals(fallback_signals)
    fallback_relaxed_raw = _sort_signals(fallback_relaxed_raw)

    execution_signals = all_signals[:scan_config.execution_top_n]

    watchlist_signals = execution_signals.copy()
    selected_codes = {s.code for s in watchlist_signals}
    if len(watchlist_signals) < scan_config.watchlist_top_n:
        need = scan_config.watchlist_top_n - len(watchlist_signals)
        fill = [s for s in fallback_signals if s.code not in selected_codes][:need]
        watchlist_signals.extend(fill)
        selected_codes.update(s.code for s in fill)
        logger.info(
            f"高胜率信号不足 ({len(execution_signals)}/{scan_config.watchlist_top_n})，兜底补齐 {len(fill)} 只"
        )

    if len(watchlist_signals) < scan_config.watchlist_top_n:
        need = scan_config.watchlist_top_n - len(watchlist_signals)
        loose_fill = []
        for sig in fallback_relaxed_raw:
            if sig.code in selected_codes:
                continue
            sig.details = dict(sig.details or {})
            sig.details["selection_tier"] = "fallback_loose"
            loose_fill.append(sig)
            if len(loose_fill) >= need:
                break
        watchlist_signals.extend(loose_fill)
        if loose_fill:
            logger.info(f"严格观察层仍不足，宽松候选再补齐 {len(loose_fill)} 只")

    # 5. TradingAgents 多智能体二次打分 (Top N)
    ta_count = 0
    if scan_config.enable_tradingagents:
        try:
            from .tradingagents_scorer import score_top_candidates
            top_n_ta = min(scan_config.watchlist_top_n, 5)
            score_top_candidates(watchlist_signals[:top_n_ta], top_n=top_n_ta)
            ta_count = top_n_ta
            logger.info(f"TradingAgents 二次打分完成 ({ta_count} 只)")
        except ImportError:
            logger.debug("TradingAgents 未安装，跳过二次打分")
        except Exception as e:
            logger.warning(f"TradingAgents 二次打分失败: {e}")
    else:
        logger.info("TradingAgents 二次打分已关闭")

    elapsed = time.time() - start_time
    logger.info(
        f"4/4 完成: 执行层{len(execution_signals)} 只 / 观察层{len(watchlist_signals)} 只, {errors} 错误, 耗时 {elapsed:.0f}s"
    )

    return ScanResult(
        date=today,
        total_scanned=len(codes),
        signals=watchlist_signals[:scan_config.watchlist_top_n],
        execution_signals=execution_signals[:scan_config.execution_top_n],
        watchlist_signals=watchlist_signals[:scan_config.watchlist_top_n],
        elapsed_seconds=round(elapsed, 1),
    )
