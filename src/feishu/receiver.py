"""飞书事件接收器 — 接收群聊消息

支持两种模式：
1. WebSocket 长连接（本地开发，无需公网 URL）
2. Flask HTTP Webhook（生产部署）

使用方式：
    # WebSocket 模式（默认）
    python -m src.feishu.receiver

    # 指定大神 open_id
    python -m src.feishu.receiver --user-id ou_xxxxxxxx
"""

import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from lark_oapi import Client

from .message_store import MessageStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class FeishuReceiver:
    """飞书消息接收器（WebSocket 长连接模式）"""

    def __init__(self, target_user_open_id: str = None):
        app_id = os.getenv("FEISHU_APP_ID")
        app_secret = os.getenv("FEISHU_APP_SECRET")
        if not app_id or not app_secret:
            raise RuntimeError("FEISHU_APP_ID / FEISHU_APP_SECRET 未设置")

        self.client = Client.builder().app_id(app_id).app_secret(app_secret).build()
        self.target_chat_id = os.getenv("FEISHU_TARGET_CHAT_ID", "")
        self.target_user_id = target_user_open_id or os.getenv(
            "FEISHU_TARGET_USER_OPEN_ID", ""
        )
        self.store = MessageStore()
        self._running = False

    def _is_target_message(self, event: dict) -> bool:
        """判断是否为目标群聊 + 目标用户的消息"""
        msg_event = event.get("message", {})
        chat_id = msg_event.get("chat_id", "")
        sender_id = msg_event.get("sender", {}).get("id", "")

        # 检查群聊
        if self.target_chat_id and chat_id != self.target_chat_id:
            return False
        # 检查发送者（配置了才过滤）
        if self.target_user_id and sender_id != self.target_user_id:
            return False
        return True

    def _handle_message(self, event: dict):
        """处理单条消息事件"""
        msg = event.get("message", {})
        msg_id = msg.get("message_id", "")
        msg_type = msg.get("msg_type", "text")
        sender_id = msg.get("sender", {}).get("id", "")
        chat_id = msg.get("chat_id", "")
        create_time = msg.get("create_time", "")

        # 提取文本内容
        content = ""
        if msg_type == "text":
            body = msg.get("body", {})
            if isinstance(body, str):
                body = json.loads(body)
            content = body.get("text", "") or body.get("content", "")
        elif msg_type == "image":
            content = "[图片消息]"
        elif msg_type == "post":
            content = json.dumps(msg.get("body", {}), ensure_ascii=False)
        else:
            content = f"[{msg_type} 类型消息]"

        # 入库
        self.store.insert(
            msg_id=msg_id,
            sender=sender_id,
            content=content,
            msg_type=msg_type,
            chat_id=chat_id,
            created_at=create_time or datetime.now().isoformat(),
        )

        logger.info(
            f"📩 收到消息 | 发送者={sender_id} | "
            f"类型={msg_type} | "
            f"内容={content[:80]}{'...' if len(content) > 80 else ''}"
        )

    def start(self):
        """启动 WebSocket 长连接，持续接收消息"""
        self._running = True
        logger.info("🟢 飞书消息接收器启动中...")
        logger.info(f"   目标群聊: {self.target_chat_id or '(未指定，接收全部)'}")
        logger.info(f"   目标用户: {self.target_user_id or '(未指定，接收全部)'}")

        try:
            # 飞书 WebSocket 事件订阅
            ws_url = self._get_ws_connection_url()
            logger.info(f"   WebSocket URL 获取成功")

            import websocket

            ws = websocket.WebSocketApp(
                ws_url,
                on_message=lambda ws, raw: self._on_ws_message(raw),
                on_error=lambda ws, err: logger.error(f"WebSocket 错误: {err}"),
                on_close=lambda ws, code, msg: logger.info(f"WebSocket 关闭: {code} {msg}"),
            )

            # 启动心跳
            def send_ping(ws):
                import time
                while self._running:
                    time.sleep(30)
                    try:
                        ws.send(json.dumps({"type": "ping"}))
                    except Exception:
                        break

            import threading
            threading.Thread(target=send_ping, args=(ws,), daemon=True).start()

            ws.run_forever()
        except KeyboardInterrupt:
            self._running = False
            logger.info("🛑 接收器已停止")
        except Exception as e:
            logger.error(f"接收器异常: {e}")
            raise

    def _get_ws_connection_url(self) -> str:
        """获取 WebSocket 长连接 URL"""
        resp = self.client.ws.connect({})
        if not resp.success():
            raise RuntimeError(f"获取 WebSocket URL 失败: {resp.msg}")
        return resp.data.get("url", "")

    def _on_ws_message(self, raw: str):
        """处理 WebSocket 原始消息"""
        try:
            payload = json.loads(raw)
            event_type = payload.get("type", "")

            if event_type == "ping":
                return  # 心跳包，忽略

            if event_type == "event":
                event_data = payload.get("event", {})
                event_subtype = event_data.get("type", "")
                if event_subtype == "im.message.receive_v1":
                    if self._is_target_message(event_data):
                        self._handle_message(event_data)
        except json.JSONDecodeError:
            logger.warning(f"无法解析消息: {raw[:200]}")


def main():
    parser = argparse.ArgumentParser(description="飞书消息接收器")
    parser.add_argument("--user-id", help="目标用户 open_id（大神）")
    parser.add_argument("--chat-id", help="目标群聊 ID（覆盖 .env 配置）")
    args = parser.parse_args()

    if args.chat_id:
        os.environ["FEISHU_TARGET_CHAT_ID"] = args.chat_id

    receiver = FeishuReceiver(target_user_open_id=args.user_id)
    receiver.start()


if __name__ == "__main__":
    main()
