"""运行每日扫描并输出结果（含实时因子数据）"""
import sys, os, time, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
import yaml
from mootdx.quotes import Quotes
from strategy.n_pattern import NPatternParams, NSignal, scan_stock, score_fundamental
from screener.data_fetcher import get_live_factor_data, get_stock_industry_map

# ── 参数 ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
with open(os.path.join(PROJECT_ROOT, "config.yaml")) as f:
    cfg = yaml.safe_load(f)
params = NPatternParams(**cfg.get("n_pattern", {}))
top_n = int(cfg.get("screener", {}).get("top_n", 15))

# ── 股票列表 ──
import akshare as ak
stock_info = ak.stock_info_a_code_name()
main = stock_info[stock_info['code'].str.match(r'^(60\d{4}|00[0-4]\d{3})$')].copy()
main = main[~main['name'].str.contains('ST', na=False)]
codes = main['code'].tolist()
names = dict(zip(main['code'], main['name']))
logger.info(f"主板 {len(codes)} 只")

# ── 大盘 ──
client = Quotes.factory(market='std', timeout=10)
try:
    idx_df = client.quotes(symbol='999999')
    price = float(idx_df.iloc[-1]['price'])
    last_close = float(idx_df.iloc[-1]['last_close'])
    market_pct = round((price - last_close) / last_close * 100, 2) if last_close > 0 else 0
except Exception:
    market_pct = 0
logger.info(f"大盘 {market_pct:+.2f}%")

# ── 实时因子数据 ──
logger.info("获取实时因子数据...")
live_data = get_live_factor_data()
try:
    industry_map = get_stock_industry_map()
    live_data["_sector_map"] = industry_map
    logger.info(f"行业映射: {len(industry_map)} 只")
except Exception as e:
    logger.warning(f"行业映射失败: {e}")
    live_data["_sector_map"] = {}

# ── 扫描 ──
logger.info(f"扫描形态 ({len(codes)} 只)...")
all_signals = []
errors = 0
t0 = time.time()

for idx, code in enumerate(codes):
    if (idx + 1) % 500 == 0:
        logger.info(f"  {idx+1}/{len(codes)}... ({len(all_signals)} 信号)")

    try:
        df = client.bars(symbol=code, frequency=9, start=0, offset=80)
        if df is None or len(df) < 40:
            continue

        signals = scan_stock(code, names.get(code, ''), df, params, market_pct, live_data=live_data)

        if signals:
            last_close_val = df['close'].values[-1]
            fin = score_fundamental(code, last_close_val, client)
            sig = signals[0]
            sig.pe = fin['pe']
            sig.pb = fin['pb']
            sig.net_profit_yi = fin['net_profit_yi']
            sig.fundamental_score = fin['score']
            if fin.get('is_garbage_profitable'):
                continue
            sig.strength += fin['score']

        all_signals.extend(signals)
    except Exception:
        errors += 1
        continue

# ── 排序 ──
all_signals.sort(key=lambda s: s.strength, reverse=True)
top_signals = all_signals[:top_n]

# ── TradingAgents 多智能体二次打分 ──
try:
    from screener.tradingagents_scorer import score_top_candidates
    ta_n = min(top_n, 5)
    score_top_candidates(top_signals[:ta_n], top_n=ta_n)
    logger.info(f"TradingAgents 二次打分完成 ({ta_n} 只)")
except ImportError:
    logger.debug("TradingAgents 未安装，跳过二次打分")
except Exception as e:
    logger.warning(f"TradingAgents 二次打分失败: {e}")

elapsed = time.time() - t0

# ── 输出 ──
print()
print('=' * 80)
print(f'扫描结果: {len(top_signals)} 信号 / 共 {len(all_signals)} 个 (扫描 {len(codes)} 只, {errors} 错误, {elapsed:.0f}s)')
hot_cnt = len(live_data.get("hot_code_set", set()))
sec_cnt = len(live_data.get("sector_rank", {}))
nb_score = live_data.get("northbound_score", 0)
print(f'实时数据: 热点{hot_cnt}只 行业{sec_cnt}个 北向{nb_score:+d}')
print('=' * 80)

for i, s in enumerate(top_signals, 1):
    flag_str = ''
    if s.resistance_levels:
        r = s.resistance_levels[0]
        flag_str += f'压力 {r[0]} {r[1]}(+{r[2]}%) '
    if s.ma10_broken_close:
        flag_str += '⚠️MA10破位'
    elif s.ma10_broken_intraday:
        flag_str += 'MA10支撑确认'

    emoji = '🔴' if s.strength >= 90 else ('🟡' if s.strength >= 75 else '⚪')
    n_type = '大N' if s.is_big_n else '小N'
    ta_info = ''
    if s.tradingagents_action:
        ta_info = f' | TA:{s.tradingagents_action}(conf={s.tradingagents_confidence:.1f})'

    print(f"""\
{i}. {emoji} **{s.name}**({s.code}) | {n_type} | 强**{s.strength}**{ta_info}
   买入**{s.entry_price}** | 止损{s.stop_loss} | 目标{s.target_price} | 费波{s.fib_level}
   {flag_str}
   因子{s.factor_score:+d}: 缩量{s.pullback_volume_score:+d} 拥挤{s.turnover_crowding_score:+d} 强弱{s.relative_strength_score:+d} 波动{s.volatility_contraction_score:+d}
        收回{s.support_reclaim_score:+d} 收盘{s.close_position_score:+d} 涨停承接{s.limit_up_followthrough_score:+d}
        题材{s.theme_heat_score:+d} 额质{s.amount_quality_score:+d} 大盘{s.market_regime_score:+d} 北向{s.northbound_flow_score:+d}
        RSI背离{s.rsi_divergence_score:+d} MACD{s.macd_signal_score:+d} MA排列{s.ma_alignment_score:+d} BOLL{s.boll_squeeze_score:+d} KDJ{s.kdj_oversold_score:+d} MFI{s.mfi_score:+d} 影线{s.shadow_quality_score:+d} 回速{s.pullback_speed_score:+d} 日内反转{s.intraday_reversal_score:+d} 量衰{s.volume_climax_score:+d} 行业{s.sector_relative_score:+d}
""")

if not top_signals:
    print("今日无符合条件的 N 字信号")
