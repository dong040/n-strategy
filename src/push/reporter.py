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
    """格式化单条信号为报告行"""
    emoji = _strength_emoji(s.strength)

    flags = []
    if s.stab_ok:
        flags.append("量价双确认")
    elif s.has_vol_shrink:
        flags.append("量缩")
    elif s.has_shadow:
        flags.append("下影")
    if s.ma_bullish:
        flags.append("多头排列")
    if s.has_limit_up:
        flags.append("涨停基因")
    if s.ma_fib_ok:
        flags.append("MA共振")

    # 支撑质量标记
    support_note = ""
    if s.ma10_broken_intraday:
        support_note = " ⚠️MA10日内跌破"
        flags.append("MA10已测")

    flag_str = " ".join(flags)

    lines = [
        f"{idx}. {emoji} **{s.name}**({s.code}) 强**{s.strength}**{support_note}",
        f"   买入 **{s.entry_price}** | 止损 {s.stop_loss} | 目标 {s.target_price}",
        f"   费波**{s.fib_level}**({s.fib_price}) | MA9={s.ma9} MA10={s.ma10} | 首波+{s.first_rise_pct}% 回调{s.retrace_pct}%({s.retrace_days}天)",
    ]
    if flag_str:
        lines.append(f"   {flag_str}")

    return "\n".join(lines)


def build_report(result: ScanResult) -> str:
    """根据扫描结果构建 Markdown 报告"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"## N字战法 每日扫描",
        f"**{result.date}** | 扫描 {result.total_scanned} 只主板 | 信号 {len(result.signals)} 个",
        f"耗时 {result.elapsed_seconds}s",
        "",
    ]

    strong = [s for s in result.signals if s.strength >= 90]
    medium = [s for s in result.signals if 75 <= s.strength < 90]
    weak = [s for s in result.signals if s.strength < 75]

    if strong:
        lines.append(f"### 强烈信号 [{len(strong)}只]")
        lines.append("")
        for i, s in enumerate(strong, 1):
            lines.append(format_signal_line(s, i))
            lines.append("")

    if medium:
        lines.append(f"### 一般信号 [{len(medium)}只]")
        lines.append("")
        for i, s in enumerate(medium, 1):
            lines.append(format_signal_line(s, i))
            lines.append("")

    if weak and not strong and not medium:
        lines.append(f"### 弱信号 [{len(weak)}只]")
        lines.append("")
        for i, s in enumerate(weak, 1):
            lines.append(format_signal_line(s, i))
            lines.append("")

    if not result.signals:
        lines.append("> 今日无符合条件的 N 字信号")
        lines.append("")

    lines.append("---")
    lines.append(f"> N字战法自动扫描 · {now}")
    lines.append("> 本报告由AI生成，仅供参考，不构成投资建议")

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
        lines.append(
            f"{i}. {emoji} {s.name}({s.code}) "
            f"买入{s.entry_price} 止损{s.stop_loss} 目标{s.target_price} "
            f"强{s.strength} 费波{s.fib_level}"
        )
    return "\n".join(lines)


def push_to_feishu(
    result: ScanResult,
    chat_id: str = None,
    use_card: bool = True,
) -> bool:
    """推送扫描结果到飞书"""
    try:
        if use_card:
            report = build_report(result)
            title = f"N字战法 每日扫描 {result.date}"
            send_markdown_card(title, report, chat_id=chat_id)
        else:
            report = build_text_report(result)
            send_text_to_chat(report, chat_id=chat_id)
        logger.info("推送成功")
        return True
    except Exception as e:
        logger.error(f"推送失败: {e}")
        return False


def push_via_webhook(results: list, top_n: int = 15) -> bool:
    """直接通过 webhook 推送信号列表（不依赖 ScanResult）"""
    now = datetime.now().strftime("%m/%d")
    lines = []

    for i, r in enumerate(results[:top_n]):
        flags = []
        if r.stab_ok:
            flags.append("量价双确认")
        elif r.has_vol_shrink:
            flags.append("量缩")
        elif r.has_shadow:
            flags.append("下影")
        if r.ma_bullish:
            flags.append("多头排列")
        if r.has_limit_up:
            flags.append("涨停基因")
        if r.ma_fib_ok:
            flags.append("MA共振")

        support_note = ""
        if r.ma10_broken_intraday:
            support_note = " ⚠️MA10日内跌破"
            flags.append("MA10已测")

        flag_str = " ".join(flags)

        lines.append(f"{i+1}. **{r.name}**({r.code}) 强**{r.strength}**{support_note}")
        lines.append(f"   买入 **{r.entry_price}** | 止损 {r.stop_loss} | 目标 {r.target_price}")
        lines.append(f"   费波**{r.fib_level}**({r.fib_price}) | MA9={r.ma9} MA10={r.ma10} | 首波+{r.first_rise_pct}% 回调{r.retrace_pct}%({r.retrace_days}天)")
        if flag_str:
            lines.append(f"   {flag_str}")
        lines.append("")

    lines.append("---")
    lines.append(f"共 {len(results)} 只主板标的 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")

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
