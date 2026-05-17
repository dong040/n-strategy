"""飞书消息 SQLite 持久化存储"""
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "messages.db"


class MessageStore:
    """群聊消息存储（SQLite，msg_id 去重）"""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or _DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    msg_id TEXT UNIQUE NOT NULL,
                    sender TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL DEFAULT '',
                    msg_type TEXT NOT NULL DEFAULT 'text',
                    chat_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT '',
                    ingested_at TEXT NOT NULL DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_sender
                ON messages(sender)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_created
                ON messages(created_at)
            """)
            conn.commit()

    def insert(
        self,
        msg_id: str,
        sender: str,
        content: str,
        msg_type: str = "text",
        chat_id: str = "",
        created_at: str = "",
    ) -> bool:
        """插入消息，msg_id 已存在返回 False"""
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            try:
                conn.execute(
                    """INSERT INTO messages (msg_id, sender, content, msg_type, chat_id, created_at, ingested_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (msg_id, sender, content, msg_type, chat_id, created_at or now, now),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False  # 已存在，跳过

    def get_all(self, limit: int = 500) -> list[dict]:
        """获取全部消息"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM messages ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_by_sender(self, sender: str, limit: int = 500) -> list[dict]:
        """按发送者过滤"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE sender = ? ORDER BY created_at DESC LIMIT ?",
                (sender, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_senders(self) -> list[dict]:
        """获取所有发送者及其消息数量"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT sender, COUNT(*) as cnt FROM messages GROUP BY sender ORDER BY cnt DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def count(self) -> int:
        with self._get_conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

    def search(self, keyword: str, limit: int = 200) -> list[dict]:
        """按关键词搜索消息内容"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE content LIKE ? ORDER BY created_at DESC LIMIT ?",
                (f"%{keyword}%", limit),
            ).fetchall()
            return [dict(r) for r in rows]
