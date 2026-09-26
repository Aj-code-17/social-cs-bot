"""Conversation memory stored in SQLite (stdlib) so chats survive restarts,
plus message-ID deduplication and human-handoff pause flags."""

from __future__ import annotations

import sqlite3
import threading
import time


class Memory:
    def __init__(self, db_path: str = "conversations.db"):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    sender_id  TEXT NOT NULL,
                    role       TEXT NOT NULL,      -- 'user' | 'assistant'
                    content    TEXT NOT NULL,
                    ts         REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_sender
                    ON messages(sender_id, id);
                CREATE TABLE IF NOT EXISTS seen_mids (
                    mid TEXT PRIMARY KEY
                );
                CREATE TABLE IF NOT EXISTS flags (
                    sender_id    TEXT PRIMARY KEY,
                    paused_until REAL NOT NULL
                );
                """
            )

    # ----------------------------------------------------------- messages
    def add_message(self, sender_id: str, role: str, content: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO messages(sender_id, role, content, ts) VALUES (?,?,?,?)",
                (sender_id, role, content, time.time()),
            )

    def last_messages(self, sender_id: str, limit: int = 12) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM messages WHERE sender_id=? "
                "ORDER BY id DESC LIMIT ?",
                (sender_id, limit),
            ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def reset(self, sender_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM messages WHERE sender_id=?", (sender_id,))
            self._conn.execute("DELETE FROM flags WHERE sender_id=?", (sender_id,))

    # ------------------------------------------------------------- dedupe
    def seen_mid(self, mid: str | None) -> bool:
        """True if this Meta message id was already processed (prevents double
        replies when Meta retries a webhook delivery)."""
        if not mid:
            return False
        with self._lock, self._conn:
            cur = self._conn.execute("INSERT OR IGNORE INTO seen_mids(mid) VALUES (?)", (mid,))
            return cur.rowcount == 0

    # ------------------------------------------------------- human handoff
    def set_handoff_pause(self, sender_id: str, minutes: float) -> None:
        until = time.time() + minutes * 60
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO flags(sender_id, paused_until) VALUES (?,?) "
                "ON CONFLICT(sender_id) DO UPDATE SET paused_until=excluded.paused_until",
                (sender_id, until),
            )

    def is_paused(self, sender_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT paused_until FROM flags WHERE sender_id=?", (sender_id,)
            ).fetchone()
        return bool(row and row[0] > time.time())

    def clear_pause(self, sender_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM flags WHERE sender_id=?", (sender_id,))
