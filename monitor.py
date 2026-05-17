"""N字战法 盘中监控 + 飞书推送

每分钟轮询标的实时价格，接近入场价时通过飞书 Webhook 推送提醒。

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


def _push_card(title: str, content: str):
    """发送飞书卡片消息"""
    if not WEBHOOK_URL:
        print("[WARN] FEISHU_WEBHOOK_URL 未设置，仅终端输出")
        return False
    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": "red" if "跌破" in title else "green" if "到位" in title else "orange",
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

    # 监控状态: idle → approaching → triggered → stopped_out
    # 冷却: 每只股票 30 分钟内最多推送一次
    COOLDOWN = 1800  # seconds
    pool = {}
    for s in signals:
        pool[s.code] = {
            "name": s.name,
            "entry": s.entry_price,
            "stop": s.stop_loss,
            "target": s.target_price,
            "strength": s.strength,
            "state": "idle",  # idle | approaching | triggered | stopped_out
            "last_notify": 0,  # unix timestamp of last push
        }

    client = Quotes.factory(market="std", timeout=5)
    interval = 60

    print(f"飞书推送: {'✓' if WEBHOOK_URL else '✗ 未配置'}")
    print(f"轮询间隔: {interval}s | Ctrl+C 退出\n")

    while True:
        try:
            ts = datetime.now().strftime("%H:%M:%S")

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

                    # === 状态机 ===
                    prev_state = p["state"]

                    if price <= p["stop"]:
                        new_state = "stopped_out"
                    elif price <= entry:
                        new_state = "triggered"
                    elif abs(dist) <= 2:
                        new_state = "approaching"
                    else:
                        new_state = "idle"

                    # 只在状态变化时推送
                    if new_state == prev_state:
                        continue

                    now_ts = time.time()
                    in_cooldown = (now_ts - p["last_notify"]) < COOLDOWN
                    p["state"] = new_state

                    if in_cooldown:
                        continue

                    p["last_notify"] = now_ts

                    # 构建消息
                    if new_state == "approaching":
                        title = f"贴近买入区 — {name}({code})"
                        body_lines = [
                            f"**{name}**({code}) 强**{p['strength']}**",
                            f"现价 **{price:.2f}** → 入场价 **{entry:.2f}** (+{dist:.1f}%)",
                            f"止损 {p['stop']} | 目标 {p['target']}",
                            f"时间: {ts}",
                        ]
                        print(f"[{ts}] 🔴 {name}({code}) 贴近 +{dist:.1f}% 现价{price:.2f}")
                        _push_card(title, "\n".join(body_lines))

                    elif new_state == "triggered":
                        title = f"到位买入区 — {name}({code})"
                        body_lines = [
                            f"**{name}**({code}) 强**{p['strength']}**",
                            f"现价 **{price:.2f}** 已触及入场价 **{entry:.2f}** ({dist:.1f}%)",
                            f"止损 {p['stop']} | 目标 {p['target']}",
                            f"时间: {ts}",
                        ]
                        print(f"[{ts}] 🟢 {name}({code}) 到位 {dist:.1f}% 现价{price:.2f}")
                        _push_card(title, "\n".join(body_lines))

                    elif new_state == "stopped_out":
                        title = f"已跌破止损 — {name}({code})"
                        body_lines = [
                            f"**{name}**({code}) 强**{p['strength']}**",
                            f"现价 **{price:.2f}** 已跌破止损 **{p['stop']}**",
                            f"入场 {entry} → 止损位 {p['stop']} 无效",
                            f"时间: {ts}",
                        ]
                        print(f"[{ts}] ⚠️  {name}({code}) 跌破止损 现价{price:.2f}")
                        _push_card(title, "\n".join(body_lines))

                except Exception:
                    continue

            time.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n监控已停止 — {datetime.now().strftime('%H:%M:%S')}")
            break


if __name__ == "__main__":
    main()
