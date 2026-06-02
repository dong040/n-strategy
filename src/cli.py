"""CLI 入口 — N字战法系统命令行工具

Usage:
    python -m src.cli import <文件路径>   导入飞书导出的聊天记录
    python -m src.cli extract             提取战法规则
    python -m src.cli backtest            回测当前参数
    python -m src.cli optimize            优化参数
    python -m src.cli scan                扫描今日标的
    python -m src.cli push                推送到飞书（需先 scan）
    python -m src.cli daily               一键 scan + push
    python -m src.cli test-webhook        测试飞书 webhook 连接
    python -m src.cli notify <消息>       发送手动飞书提醒
    python -m src.cli notify-backtest     推送最近回测摘要到飞书
"""

import logging
import pickle
import sys
import re
import json
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ====== 消息导入 ======

def _parse_feishu_export(path: str) -> list[dict]:
    """解析飞书桌面端导出的 txt 聊天记录"""
    content = Path(path).read_text(encoding="utf-8")
    messages = []

    lines = content.split("\n")
    current_sender = ""
    current_time = ""
    current_msg = []

    for line in lines:
        time_match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
        if time_match:
            if current_msg and current_time:
                text = "\n".join(current_msg).strip()
                if text:
                    messages.append({
                        "sender": current_sender,
                        "content": text,
                        "msg_type": "text",
                        "created_at": current_time,
                    })
            current_time = time_match.group(1)
            prefix = line[:time_match.start()].strip()
            current_sender = prefix.lstrip("[").rstrip("]").strip()
            current_msg = [line[time_match.end():].strip()] if line[time_match.end():].strip() else []
        else:
            if current_sender:
                current_msg.append(line.strip())

    if current_msg and current_time:
        text = "\n".join(current_msg).strip()
        if text:
            messages.append({
                "sender": current_sender,
                "content": text,
                "msg_type": "text",
                "created_at": current_time,
            })

    return messages


def cmd_import():
    if len(sys.argv) < 3:
        print("用法: python -m src.cli import <文件路径>")
        sys.exit(1)

    filepath = sys.argv[2]
    if not Path(filepath).exists():
        print(f"文件不存在: {filepath}")
        sys.exit(1)

    print(f"解析文件: {filepath}")
    messages = _parse_feishu_export(filepath)

    if not messages:
        print("未识别到消息")
        sys.exit(1)

    from src.feishu.message_store import MessageStore

    store = MessageStore()
    inserted = 0
    skipped = 0
    for i, msg in enumerate(messages):
        fake_id = f"import_{Path(filepath).stem}_{i}"
        if store.insert(
            msg_id=fake_id,
            sender=msg["sender"],
            content=msg["content"],
            msg_type=msg["msg_type"],
            created_at=msg["created_at"],
        ):
            inserted += 1
        else:
            skipped += 1

    print(f"导入完成: {inserted} 条新增, {skipped} 条重复")


# ====== 规则提取 ======

def cmd_extract():
    from src.feishu.message_store import MessageStore
    from src.strategy.extractor import extract_rules, extract_stock_cases, summarize_rules

    store = MessageStore()
    messages = store.get_all(limit=500)
    if not messages:
        logger.warning("没有消息数据，请先运行 import")
        return

    senders = store.get_senders()
    print(f"\n消息来源统计（共 {len(messages)} 条）:")
    for s in senders[:10]:
        print(f"   {s['sender'][:20]:20s} | {s['cnt']} 条")

    print(f"\n开始提取战法规则...")
    rules = extract_rules(messages)
    summary = summarize_rules(rules)
    cases = extract_stock_cases(messages)

    print("\n" + "=" * 60)
    print("规则提取结果")
    print("=" * 60)
    for cat, info in summary.items():
        if info["count"] == 0:
            continue
        print(f"\n[{cat}] 共 {info['count']} 条规则")
        print(f"  热门关键词: {info['top_keywords'][:5]}")

    print(f"\n提取到 {len(cases)} 个实战案例")
    for c in cases[:10]:
        print(f"  {c.code} {c.date} {'✅' if c.direction == 'long' else '📉'} {c.raw_text[:80]}")


# ====== 回测 ======

def cmd_backtest():
    import yaml
    from src.strategy.n_pattern import NPatternParams
    from src.strategy.backtest import BacktestConfig, backtest_single_stock
    from src.screener.data_fetcher import get_daily_klines

    with open(_PROJECT_ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    params = NPatternParams(**cfg.get("n_pattern", {}))
    backtest_cfg = BacktestConfig(**cfg.get("backtest", {}))

    test_codes = ["600379", "000062", "002475", "000858"]
    for code in test_codes:
        logger.info(f"回测 {code}...")
        df = get_daily_klines(code, days=500)
        if df.empty:
            logger.warning(f"  {code} 无数据")
            continue
        result = backtest_single_stock(code, code, df, params, backtest_cfg)
        print(
            f"  {code}: {result.total_trades}笔 | "
            f"胜率{result.win_rate}% | 收益{result.total_return}% | "
            f"夏普{result.sharpe_ratio} | 回撤{result.max_drawdown}%"
        )


# ====== 参数优化 ======

def cmd_optimize():
    import yaml
    from src.strategy.n_pattern import NPatternParams
    from src.strategy.backtest import BacktestConfig
    from src.strategy.optimizer import optimize, OptimizerConfig
    from src.screener.data_fetcher import get_daily_klines

    with open(_PROJECT_ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    opt_cfg = OptimizerConfig(**cfg.get("optimizer", {}))
    backtest_cfg = BacktestConfig(**cfg.get("backtest", {}))

    train_codes = [
        "600379", "000062", "002475", "000858", "600519",
        "300308", "002463", "688111", "300476", "603259",
    ]
    stocks_data = {}
    logger.info("加载训练数据...")
    for code in train_codes:
        df = get_daily_klines(code, days=500)
        if not df.empty and len(df) >= 100:
            stocks_data[code] = df
            logger.info(f"  {code}: {len(df)} 根 K线")

    logger.info(f"开始{opt_cfg.method}优化 ({len(stocks_data)} 只股票)...")
    best_params, best_score, all_results = optimize(
        stocks_data, method=opt_cfg.method, config=backtest_cfg, opt_config=opt_cfg,
    )

    print("\n" + "=" * 60)
    print("优化完成")
    print("=" * 60)
    print(f"方法: {opt_cfg.method} | 目标: {opt_cfg.target_metric} | 最优: {best_score:.3f}")
    print(f"\n最优参数:")
    for k, v in vars(best_params).items():
        print(f"  {k}: {v}")


# ====== 每日扫描 ======

def cmd_scan():
    import yaml
    from src.strategy.n_pattern import NPatternParams
    from src.screener.daily_scan import run_daily_scan, ScanConfig

    with open(_PROJECT_ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    params = NPatternParams(**cfg.get("n_pattern", {}))
    scan_cfg = ScanConfig(**cfg.get("screener", {}))

    result = run_daily_scan(params, scan_cfg)

    # Market label
    mkt = result.signals[0].market_pct if result.signals else 0
    mkt_label = f"大盘{mkt:+.2f}% " + ("强势" if mkt > 0.5 else ("弱势" if mkt < -0.5 else "平盘"))
    execution = getattr(result, "execution_signals", []) or []
    watchlist = getattr(result, "watchlist_signals", []) or result.signals

    # Print report
    print(f"\n{'='*80}")
    print(f"N字战法 每日扫描 — {result.date} | {mkt_label}")
    print(f"{'='*80}")
    print(
        f"扫描: {result.total_scanned} 只主板 | 执行层: {len(execution)} 个 | "
        f"观察层: {len(watchlist)} 个 | 耗时: {result.elapsed_seconds}s"
    )
    print()

    print("[执行层]")
    if not execution:
        print("  今日无严格高胜率信号")
    for i, s in enumerate(execution, 1):
        flags = []
        if s.is_big_n: flags.append("大N")
        if s.stab_ok: flags.append("量价双确认")
        elif s.has_vol_shrink: flags.append("量缩")
        elif s.has_shadow: flags.append("下影")
        if s.ma_bullish: flags.append("多头排列")
        if s.has_limit_up: flags.append("涨停基因")
        if s.ma_fib_ok: flags.append("MA共振")
        if s.ma10_broken_close: flags.append("⚠️MA10破位")
        elif s.ma10_broken_intraday: flags.append("MA10确认")
        flag_str = " | ".join(flags)
        warn = " ⚠️MA10破位" if s.ma10_broken_close else (" MA10支撑确认" if s.ma10_broken_intraday else "")
        src = f" [{s.entry_source}]" if s.entry_source else ""
        tier = (s.details or {}).get("selection_tier", "high_win")
        if tier == "fallback":
            tier_label = "兜底"
        elif tier == "fallback_loose":
            tier_label = "宽松兜底"
        else:
            tier_label = "高胜率"

        print(f"{i:2d}. {s.name}({s.code}) 强{s.strength} [{tier_label}]{warn}")
        print(f"    买入{s.entry_price}{src} 止损{s.stop_loss} 目标{s.target_price} 盈亏比{s.rr_ratio}")
        print(f"    ML={s.ml_confidence:.2f} SEQ={getattr(s, 'sequence_confidence', 0.0):.2f} Rank={(s.details or {}).get('rank_score', 0)} 因子={s.factor_score:+d}")
        if flag_str:
            print(f"    {flag_str}")
    print()

    print("[观察层]")
    for i, s in enumerate(watchlist, 1):
        flags = []
        if s.is_big_n: flags.append("大N")
        if s.stab_ok: flags.append("量价双确认")
        elif s.has_vol_shrink: flags.append("量缩")
        elif s.has_shadow: flags.append("下影")
        if s.ma_bullish: flags.append("多头排列")
        if s.has_limit_up: flags.append("涨停基因")
        if s.ma_fib_ok: flags.append("MA共振")
        if s.ma10_broken_close: flags.append("⚠️MA10破位")
        elif s.ma10_broken_intraday: flags.append("MA10确认")
        flag_str = " | ".join(flags)
        warn = " ⚠️MA10破位" if s.ma10_broken_close else (" MA10支撑确认" if s.ma10_broken_intraday else "")

        src = f" [{s.entry_source}]" if s.entry_source else ""
        tier = (s.details or {}).get("selection_tier", "high_win")
        if tier == "fallback":
            tier_label = "兜底"
        elif tier == "fallback_loose":
            tier_label = "宽松兜底"
        else:
            tier_label = "高胜率"

        print(f"{i:2d}. {s.name}({s.code}) 强{s.strength} [{tier_label}]{warn}")
        if s.tradingagents_action:
            print(f"    🤖TA: {s.tradingagents_action}(conf={s.tradingagents_confidence:.1f} 分歧={s.tradingagents_divergence:.1f})")
        print(f"    买入{s.entry_price}{src} 止损{s.stop_loss} 目标{s.target_price} 盈亏比{s.rr_ratio}")
        print(f"    ML={s.ml_confidence:.2f} SEQ={getattr(s, 'sequence_confidence', 0.0):.2f} Rank={(s.details or {}).get('rank_score', 0)} 因子={s.factor_score:+d}")
        print(f"    费波{s.fib_level}({s.fib_price}) MA9={s.ma9} MA10={s.ma10} | 首波+{s.first_rise_pct}% 回调{s.retrace_pct}%({s.retrace_days}天)")
        if s.broken_levels:
            brk_strs = [f"{label}{price}(已突破)" for label, price, dist in s.broken_levels[:2]]
            print(f"    已突破: {' | '.join(brk_strs)}")
        if s.resistance_levels:
            res_strs = [f"{label}{price}(+{dist}%)" for label, price, dist in s.resistance_levels[:3]]
            print(f"    压力位: {' | '.join(res_strs)}")
        if s.fib_extension_1272 > s.entry_price:
            print(f"    Fib扩展 127.2%={s.fib_extension_1272} 161.8%={s.fib_extension_1618}")
        if flag_str:
            print(f"    {flag_str}")
        print(
            f"    因子{s.factor_score:+d}: "
            f"缩量{s.pullback_volume_score:+d} "
            f"拥挤{s.turnover_crowding_score:+d} "
            f"强弱{s.relative_strength_score:+d} "
            f"波动{s.volatility_contraction_score:+d} "
            f"收回{s.support_reclaim_score:+d} "
            f"收盘{s.close_position_score:+d} "
            f"涨停承接{s.limit_up_followthrough_score:+d} "
            f"题材{s.theme_heat_score:+d} "
            f"额质{s.amount_quality_score:+d} "
            f"大盘{s.market_regime_score:+d} "
            f"北向{s.northbound_flow_score:+d} "
            f"RSI{s.rsi_divergence_score:+d} "
            f"MACD{s.macd_signal_score:+d} "
            f"MA排{s.ma_alignment_score:+d} "
            f"BOLL{s.boll_squeeze_score:+d} "
            f"KDJ{s.kdj_oversold_score:+d} "
            f"MFI{s.mfi_score:+d} "
            f"影线{s.shadow_quality_score:+d} "
            f"回速{s.pullback_speed_score:+d} "
            f"日内反转{s.intraday_reversal_score:+d} "
            f"量衰{s.volume_climax_score:+d} "
            f"行业{s.sector_relative_score:+d}"
        )

    # Cache for push
    cache_path = _PROJECT_ROOT / "data" / "last_scan.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    print(f"\n结果已缓存 → {cache_path}")


# ====== 推送 ======

def cmd_push():
    from src.push.reporter import push_to_feishu

    cache_path = _PROJECT_ROOT / "data" / "last_scan.pkl"
    if not cache_path.exists():
        logger.error("没有缓存扫描结果，请先运行 scan")
        sys.exit(1)

    with open(cache_path, "rb") as f:
        result = pickle.load(f)

    push_to_feishu(result)
    print("已推送到飞书")


# ====== 一键 ======

def cmd_daily():
    logger.info("=== N字战法 每日任务 ===")
    cmd_scan()
    logger.info("推送结果到飞书...")
    cmd_push()
    logger.info("=== 完成 ===")


# ====== 测试 Webhook ======

def cmd_test_webhook():
    from src.feishu.bot import send_text_via_webhook

    msg = f"N字战法系统 Webhook 测试\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    if send_text_via_webhook(msg):
        print("Webhook 测试成功")
    else:
        print("Webhook 测试失败，请检查 .env 中的 FEISHU_WEBHOOK_URL")


def cmd_notify():
    from src.push.reporter import push_notification

    if len(sys.argv) < 3:
        print("用法: python -m src.cli notify <消息>")
        sys.exit(1)

    message = " ".join(sys.argv[2:]).strip()
    title = f"N字战法通知 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    if push_notification(title, message, use_card=False):
        print("通知已发送")
    else:
        print("通知发送失败")


def cmd_notify_backtest():
    from src.push.reporter import push_notification

    default_path = _PROJECT_ROOT / "data" / "backtest_results_2y_mainboard_all.json"
    summary_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else default_path
    if not summary_path.exists():
        print(f"回测摘要不存在: {summary_path}")
        sys.exit(1)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lines = [
        f"**区间**: {summary.get('date_from')} ~ {summary.get('date_to')}",
        f"**股票池**: {summary.get('universe_size', 0)} | 成功 {summary.get('success', 0)} | 无数据 {summary.get('no_data', 0)}",
        f"**交易**: {summary.get('trades', 0)} 笔",
        f"**胜率**: {summary.get('win_rate', 0)}%",
        f"**平均收益**: {summary.get('avg_profit_pct', 0)}%",
        f"**中位数收益**: {summary.get('median_profit_pct', 0)}%",
        f"**盈利因子**: {summary.get('profit_factor', 0)}",
        f"**平均持仓**: {summary.get('avg_hold_days', 0)} 天",
        f"**出场分布**: `{summary.get('exit_reasons', {})}`",
    ]
    title = f"N字战法回测完成 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    if push_notification(title, "\n".join(lines), use_card=True):
        print("回测摘要已发送到飞书")
    else:
        print("回测摘要发送失败")


# ====== 命令注册 ======

COMMANDS = {
    "import": cmd_import,
    "extract": cmd_extract,
    "backtest": cmd_backtest,
    "optimize": cmd_optimize,
    "scan": cmd_scan,
    "push": cmd_push,
    "daily": cmd_daily,
    "test-webhook": cmd_test_webhook,
    "notify": cmd_notify,
    "notify-backtest": cmd_notify_backtest,
}


def main():
    if len(sys.argv) < 2:
        print("用法: python -m src.cli <command>")
        print(f"可用命令: {', '.join(COMMANDS.keys())}")
        print()
        print("常用流程:")
        print("  python -m src.cli import 导出记录.txt    # 导入飞书聊天记录")
        print("  python -m src.cli extract               # 提取战法规则")
        print("  python -m src.cli test-webhook          # 测试飞书推送")
        print("  python -m src.cli daily                 # 扫描+推送")
        print("  python -m src.cli notify 任务完成       # 手动发送提醒")
        print("  python -m src.cli notify-backtest       # 发送回测摘要")
        sys.exit(1)

    cmd_name = sys.argv[1]
    if cmd_name not in COMMANDS:
        print(f"未知命令: {cmd_name}")
        print(f"可用命令: {', '.join(COMMANDS.keys())}")
        sys.exit(1)

    COMMANDS[cmd_name]()


if __name__ == "__main__":
    main()
