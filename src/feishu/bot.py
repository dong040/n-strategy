"""飞书推送模块 — 通过自定义机器人 Webhook 发送消息

外部群无法添加企业自建应用，但可以添加自定义机器人（Webhook）。
Webhook 只能发消息，不能收消息。消息接收走本地文件导入。
"""
import json
import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

WEBHOOK_URL = os.getenv("FEISHU_WEBHOOK_URL", "")


def send_text_via_webhook(content: str) -> bool:
    """通过 webhook 发送纯文本消息"""
    if not WEBHOOK_URL:
        raise RuntimeError("FEISHU_WEBHOOK_URL 未设置，请检查 ~/n-strategy/.env")

    body = {"msg_type": "text", "content": {"text": content}}
    r = requests.post(WEBHOOK_URL, json=body, timeout=10)
    data = r.json()
    if data.get("code") != 0:
        logger.error(f"Webhook 发送失败: {data}")
        return False
    return True


def send_interactive_via_webhook(title: str, content: str) -> bool:
    """通过 webhook 发送 Markdown 卡片消息"""
    if not WEBHOOK_URL:
        raise RuntimeError("FEISHU_WEBHOOK_URL 未设置，请检查 ~/n-strategy/.env")

    card = {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": title}},
        "elements": [{"tag": "markdown", "content": content}],
    }
    body = {"msg_type": "interactive", "card": card}
    r = requests.post(WEBHOOK_URL, json=body, timeout=10)
    data = r.json()
    if data.get("code") != 0:
        logger.error(f"Webhook 卡片发送失败: {data}")
        return False
    return True


def send_text_to_chat(content: str, chat_id: str = None) -> bool:
    """兼容旧接口：发送文本消息"""
    return send_text_via_webhook(content)


def send_markdown_card(title: str, content: str, chat_id: str = None) -> bool:
    """兼容旧接口：发送卡片消息"""
    return send_interactive_via_webhook(title, content)
