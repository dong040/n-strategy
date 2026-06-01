"""推送报告生成 + 飞书发送"""

import logging
from datetime import datetime

from ..screener.daily_scan import ScanResult
from ..strategy.n_pattern import NSignal
from ..feishu.bot import send_markdown_card, send_text_to_chat, send_interactive_via_webhook

logger = logging.getLogger(__name__)


def _strength_emoji(strength: int) -> str:
    if strength >= 90:
        return "🔴"
    elif strength >= 75:
        return "🟡"
    return "⚪"


def format_signal_line(s: NSignal, idx: int) -> str:
    """格式化单条信号为精简报告行"""
    emoji = _strength_emoji(s.strength)
    n_type = "大N" if s.is_big_n else "小N"
    support = f"MA9={s.ma9}" if not s.is_big_n else f"MA10={s.ma10}"
    resistance = ""
    if s.resistance_levels:
        r = s.resistance_levels[0]
        resistance = f" | {r[0]} {r[1]}(+{r[2]}%)"
    if s.ma10_broken_close:
        warn = " ⚠️MA10破位"
    elif s.ma10_broken_intraday:
        warn = " MA10支撑确认"
    else:
        warn = ""
    tier = (s.details or {}).get("selection_tier", "high_win")
    tier_label = "兜底" if tier == "fallback" else "高胜率"

    return (
        f"{idx}. {emoji} **{s.name}**({s.code}) "
        f"买入**{s.entry_price}** | {support} | {n_type} | {tier_label} | 强**{s.strength}**{resistance}{warn}"
        f" | 因子{s.factor_score:+d}"
    )


def build_report(result: ScanResult) -> str:
    """根据扫描结果构建精简 Markdown 报告"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"## N字战法 {result.date}",
        f"扫描 {result.total_scanned} 只 → {len(result.signals)} 信号 | {result.elapsed_seconds}s",
        "",
    ]

    for i, s in enumerate(result.signals, 1):
        lines.append(format_signal_line(s, i))

    if not result.signals:
        lines.append("> 今日无符合条件的 N 字信号")

    lines.append("")
    lines.append("---")
    lines.append(f"> {now} · 仅供参考")

    return "\n".join(lines)


def build_text_report(result: ScanResult) -> str:
    """纯文本版本"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"N字战法 每日扫描 {result.date}",
        f"扫描 {result.total_scanned} 只 → 信号 {len(result.signals)} 个 | 耗时 {result.elapsed_seconds}s",
        "",
    ]

    for i, s in enumerate(result.signals, 1):
        emoji = _strength_emoji(s.strength)
        tier = (s.details or {}).get("selection_tier", "high_win")
        tier_label = "兜底" if tier == "fallback" else "高胜率"
        lines.append(
            f"{i}. {emoji} {s.name}({s.code}) "
            f"买入{s.entry_price} 止损{s.stop_loss} 目标{s.target_price} "
            f"强{s.strength} {tier_label} 费波{s.fib_level}"
        )
    return "\n".join(lines)


def push_to_feishu(
    result: ScanResult,
    chat_id: str = None,
    use_card: bool = True,
) -> bool:
    """推送扫描结果到飞书"""
    try:
        ok = False
        if use_card:
            report = build_report(result)
            title = f"N字战法 每日扫描 {result.date}"
            ok = send_markdown_card(title, report, chat_id=chat_id)
        else:
            report = build_text_report(result)
            ok = send_text_to_chat(report, chat_id=chat_id)
        if ok:
            logger.info("推送成功")
        else:
            logger.error("推送失败: 飞书返回错误")
        return ok
    except Exception as e:
        logger.error(f"推送失败: {e}")
        return False


def push_via_webhook(results: list, top_n: int = 15) -> bool:
    """直接通过 webhook 推送信号列表（不依赖 ScanResult）"""
    now = datetime.now().strftime("%m/%d")
    lines = []

    for i, r in enumerate(results[:top_n]):
        emoji = _strength_emoji(r.strength)
        n_type = "大N" if r.is_big_n else "小N"
        support = f"MA9={r.ma9}" if not r.is_big_n else f"MA10={r.ma10}"
        resistance = ""
        if r.resistance_levels:
            res = r.resistance_levels[0]
            resistance = f" | {res[0]} {res[1]}(+{res[2]}%)"
        warn = " ⚠️MA10已测" if r.ma10_broken_intraday else ""

        lines.append(
            f"{i+1}. {emoji} **{r.name}**({r.code}) "
            f"买入**{r.entry_price}** | {support} | {n_type} | 强**{r.strength}**"
            f" | 因子{r.factor_score:+d}{resistance}{warn}"
        )

    lines.append("")
    lines.append(f"共 {len(results)} 只标的 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    return send_interactive_via_webhook(
        f"N字战法 — {now} TOP {top_n}",
        "\n".join(lines),
    )


def push_simple_message(content: str, chat_id: str = None) -> bool:
    """推送简单文本消息到飞书"""
    try:
        send_text_to_chat(content, chat_id=chat_id)
        return True
    except Exception as e:
        logger.error(f"推送失败: {e}")
        return False
