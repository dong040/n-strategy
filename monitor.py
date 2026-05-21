"""N字战法 盘中监控 + 飞书推送

5 秒轮询，仅在竞价结束后推送（9:25-11:30, 13:00-15:00）。
只在状态切换时推送一次，不重复轰炸。

Usage:
    python3 monitor.py                           # 监控最近一次扫描的 Top 15
    python3 monitor.py <scan_file>               # 从指定 pickle 文件加载
    python3 monitor.py --all                     # 监控所有信号（而非仅 Top 15）

开盘前启动，Ctrl+C 退出。
"""

import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from mootdx.quotes import Quotes

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")
WEBHOOK_URL = os.getenv("FEISHU_WEBHOOK_URL", "")

POLL_SECONDS = 5

# 状态定义
IDLE = "idle"
APPROACHING = "approaching"   # 距入场 ≤2% 但未触及
TRIGGERED = "triggered"       # 入场 ≤ price ≤ 止损之上
STOPPED_OUT = "stopped_out"   # 跌破止损


def _in_trading_hours():
    now = datetime.now()
    h, m = now.hour, now.minute
    morning = (h == 9 and m >= 25) or h == 10 or (h == 11 and m <= 30)
    afternoon = h == 13 or h == 14 or (h == 15 and m == 0)
    return morning or afternoon


def _push_card(title: str, content: str, color: str = "orange"):
    if not WEBHOOK_URL:
        print("[WARN] FEISHU_WEBHOOK_URL 未设置，仅终端输出")
        return False
    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": color,
        },
        "elements": [{"tag": "markdown", "content": content}],
    }
    body = {"msg_type": "interactive", "card": card}
    try:
        r = requests.post(WEBHOOK_URL, json=body, timeout=10)
        return r.json().get("code") == 0
    except Exception:
        return False


def main():
    scan_path = _PROJECT_ROOT / "data" / "last_scan.pkl"
    top_n = 15

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--all":
            top_n = None
        else:
            scan_path = Path(arg)

    if not scan_path.exists():
        print(f"扫描结果不存在: {scan_path}")
        sys.exit(1)

    with open(scan_path, "rb") as f:
        result = pickle.load(f)

    signals = result.signals if top_n is None else result.signals[:top_n]
    if not signals:
        print("无信号可监控")
        sys.exit(0)

    print(f"加载 {len(signals)} 只标的, 扫描日 {result.date}")

    pool = {}
    for s in signals:
        pool[s.code] = {
            "name": s.name,
            "entry": s.entry_price,
            "stop": s.stop_loss,
            "target": s.target_price,
            "strength": s.strength,
            "state": IDLE,
        }

    client = Quotes.factory(market="std", timeout=5)

    print(f"飞书推送: {'✓' if WEBHOOK_URL else '✗ 未配置'}")
    print(f"轮询: {POLL_SECONDS}s | 仅在状态切换时推送")
    print(f"推送时段: 9:30-11:30, 13:00-15:00 | Ctrl+C 退出\n")

    while True:
        try:
            now = datetime.now()
            ts = now.strftime("%H:%M:%S")
            in_trading = _in_trading_hours()

            for code, p in pool.items():
                try:
                    quote = client.quotes(symbol=code)
                    if quote is None or quote.empty:
                        continue

                    row = quote.iloc[-1]
                    price = float(row["price"])
                    entry = p["entry"]
                    name = p["name"]
                    dist = (price - entry) / entry * 100

                    # 状态判定
                    if price <= p["stop"]:
                        new_state = STOPPED_OUT
                    elif price <= entry:
                        new_state = TRIGGERED
                    elif abs(dist) <= 2:
                        new_state = APPROACHING
                    else:
                        new_state = IDLE

                    # 无变化不推送
                    if new_state == p["state"]:
                        continue

                    old_state = p["state"]
                    p["state"] = new_state

                    # 非交易时段记录状态变化但不推送
                    if not in_trading:
                        print(f"[{ts}] 状态追踪: {name} {old_state} → {new_state} (非交易时段)")
                        continue

                    # === 状态切换 → 推送一次 ===

                    if new_state == APPROACHING:
                        title = f"贴近买入区 — {name}({code})"
                        body = (
                            f"**{name}**({code}) 强**{p['strength']}**\n"
                            f"现价 **{price:.2f}** → 入场价 **{entry:.2f}** (+{dist:.1f}%)\n"
                            f"止损 {p['stop']} | 目标 {p['target']}\n"
                            f"时间: {ts}"
                        )
                        print(f"[{ts}] 🔴 {name}({code}) 贴近买入区 +{dist:.1f}% 现价{price:.2f}")
                        _push_card(title, body, "orange")

                    elif new_state == TRIGGERED:
                        title = f"到达买入区 — {name}({code})"
                        body = (
                            f"**{name}**({code}) 强**{p['strength']}**\n"
                            f"现价 **{price:.2f}** 已进入买入区 (入场 **{entry:.2f}**, {dist:.1f}%)\n"
                            f"止损 {p['stop']} | 目标 {p['target']}\n"
                            f"时间: {ts}"
                        )
                        print(f"[{ts}] 🟢 {name}({code}) 到达买入区 {dist:.1f}% 现价{price:.2f}")
                        _push_card(title, body, "green")

                    elif new_state == STOPPED_OUT:
                        title = f"跌破止损 — {name}({code})"
                        body = (
                            f"**{name}**({code}) 强**{p['strength']}**\n"
                            f"现价 **{price:.2f}** 已跌破止损 **{p['stop']}**\n"
                            f"入场 {entry} | 时间: {ts}"
                        )
                        print(f"[{ts}] ⚠️  {name}({code}) 跌破止损 现价{price:.2f}")
                        _push_card(title, body, "red")

                    elif new_state == IDLE:
                        # 从关注区回到空闲 → 通知离开
                        title = f"离开买入区 — {name}({code})"
                        body = (
                            f"**{name}**({code}) 强**{p['strength']}**\n"
                            f"现价 **{price:.2f}** 已远离入场价 **{entry:.2f}** (+{dist:.1f}%)\n"
                            f"时间: {ts}"
                        )
                        print(f"[{ts}] 🔵 {name}({code}) 离开买入区 +{dist:.1f}% 现价{price:.2f}")
                        _push_card(title, body, "blue")

                except Exception:
                    continue

            time.sleep(POLL_SECONDS)

        except KeyboardInterrupt:
            print(f"\n监控已停止 — {datetime.now().strftime('%H:%M:%S')}")
            break


if __name__ == "__main__":
    main()
