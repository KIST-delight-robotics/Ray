"""Storage backend implementations for conversation history."""

from __future__ import annotations

import copy
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from voice_pipeline.core.config import ConversationHistoryConfig
from voice_pipeline.core.interfaces import IStorageBackend

logger = logging.getLogger("voice_pipeline.history")

# Unified timestamp format (UTC, no timezone offset)
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


class MemoryStorageBackend(IStorageBackend):
    """In-memory storage backend. Data is lost when the process exits.

    Used for unit tests.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    def create_session(self, session_id: str, started_at: str) -> None:
        """Create a new session record."""
        self._sessions[session_id] = {
            "started_at": started_at,
            "ended_at": None,
            "messages": [],
        }

    def end_session(self, session_id: str, ended_at: str) -> None:
        """Mark a session as ended."""
        if session_id in self._sessions:
            self._sessions[session_id]["ended_at"] = ended_at

    def load_session(self, session_id: str) -> list[tuple[int, int, dict[str, Any], int]]:
        """Load all messages for a session."""
        session = self._sessions.get(session_id)
        if session is None:
            return []
        return [
            (m["msg_id"], m["turn_id"], copy.deepcopy(m["item"]), m["token_count"])
            for m in session["messages"]
        ]

    def append_message(
        self,
        session_id: str,
        msg_id: int,
        turn_id: int,
        item: dict[str, Any],
        token_count: int,
        metrics_json: str | None = None,
    ) -> None:
        """Append a message to the session."""
        if session_id not in self._sessions:
            return
        self._sessions[session_id]["messages"].append(
            {
                "msg_id": msg_id,
                "turn_id": turn_id,
                "item": copy.deepcopy(item),
                "token_count": token_count,
                "metrics_json": metrics_json,
            }
        )

    def update_message(
        self,
        session_id: str,
        msg_id: int,
        item: dict[str, Any],
        token_count: int,
    ) -> None:
        """Update an existing message."""
        session = self._sessions.get(session_id)
        if session is None:
            return
        for msg in session["messages"]:
            if msg["msg_id"] == msg_id:
                msg["item"] = copy.deepcopy(item)
                msg["token_count"] = token_count
                return

    def delete_session(self, session_id: str) -> None:
        """Delete all data for a session."""
        self._sessions.pop(session_id, None)

    def get_recent_sessions(
        self,
        limit: int,
        exclude_session_id: str | None = None,
    ) -> list[tuple[str, str, str | None]]:
        """Return the most recent completed sessions."""
        completed = [
            (sid, s["started_at"], s["ended_at"])
            for sid, s in self._sessions.items()
            if s["ended_at"] is not None and sid != exclude_session_id
        ]
        completed.sort(key=lambda x: x[1], reverse=True)
        return completed[:limit]


class SQLiteStorageBackend(IStorageBackend):
    """SQLite write-through storage backend.

    Uses WAL mode for concurrent read/write safety.
    Graduated corruption recovery: normal open → WAL delete → new DB.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn = self._open_db(db_path)
        self._create_tables()

    def _open_db(self, db_path: str) -> sqlite3.Connection:
        """Open DB with graduated corruption recovery."""
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Step 1: Normal open
        try:
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.execute("PRAGMA integrity_check")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            return conn
        except sqlite3.DatabaseError:
            logger.warning("DB integrity check failed, attempting WAL recovery")

        # Step 2: Delete WAL and retry
        wal = Path(db_path + "-wal")
        shm = Path(db_path + "-shm")
        if wal.exists() or shm.exists():
            if wal.exists():
                wal.unlink()
            if shm.exists():
                shm.unlink()
            try:
                conn = sqlite3.connect(db_path, check_same_thread=False)
                conn.execute("PRAGMA integrity_check")
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                logger.warning("WAL recovery succeeded — recent writes may be lost")
                return conn
            except sqlite3.DatabaseError:
                logger.warning("WAL recovery failed, creating new DB")

        # Step 3: Backup corrupt file and create new
        if path.exists():
            backup = Path(db_path + ".corrupt")
            path.rename(backup)
            logger.error("DB corrupted — backed up to %s, creating new DB", backup)

        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                ended_at   TEXT,
                summary    TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                session_id   TEXT    NOT NULL REFERENCES sessions(session_id),
                msg_id       INTEGER NOT NULL,
                turn_id      INTEGER NOT NULL,
                item_json    TEXT    NOT NULL,
                token_count  INTEGER NOT NULL,
                metrics_json TEXT,
                created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (session_id, msg_id)
            );
        """)

    def create_session(self, session_id: str, started_at: str) -> None:
        """Create a new session record."""
        try:
            self._conn.execute(
                "INSERT INTO sessions (session_id, started_at) VALUES (?, ?)",
                (session_id, started_at),
            )
            self._conn.commit()
        except sqlite3.Error:
            logger.warning("Failed to create session %s", session_id, exc_info=True)

    def end_session(self, session_id: str, ended_at: str) -> None:
        """Mark session as ended and checkpoint WAL."""
        try:
            self._conn.execute(
                "UPDATE sessions SET ended_at = ? WHERE session_id = ?",
                (ended_at, session_id),
            )
            self._conn.commit()
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            logger.warning("Failed to end session %s", session_id, exc_info=True)

    def load_session(self, session_id: str) -> list[tuple[int, int, dict[str, Any], int]]:
        """Load all messages for a session."""
        try:
            cursor = self._conn.execute(
                "SELECT msg_id, turn_id, item_json, token_count "
                "FROM messages WHERE session_id = ? ORDER BY msg_id",
                (session_id,),
            )
            return [(row[0], row[1], json.loads(row[2]), row[3]) for row in cursor]
        except sqlite3.Error:
            logger.warning("Failed to load session %s", session_id, exc_info=True)
            return []

    def append_message(
        self,
        session_id: str,
        msg_id: int,
        turn_id: int,
        item: dict[str, Any],
        token_count: int,
        metrics_json: str | None = None,
    ) -> None:
        """Append a message (write-through). Graceful on failure."""
        try:
            self._conn.execute(
                "INSERT INTO messages "
                "(session_id, msg_id, turn_id, item_json, token_count, metrics_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    msg_id,
                    turn_id,
                    json.dumps(item, ensure_ascii=False),
                    token_count,
                    metrics_json,
                ),
            )
            self._conn.commit()
        except sqlite3.Error:
            logger.warning(
                "Failed to append message %d to session %s",
                msg_id,
                session_id,
                exc_info=True,
            )

    def update_message(
        self,
        session_id: str,
        msg_id: int,
        item: dict[str, Any],
        token_count: int,
    ) -> None:
        """Update an existing message (write-through)."""
        try:
            self._conn.execute(
                "UPDATE messages SET item_json = ?, token_count = ? "
                "WHERE session_id = ? AND msg_id = ?",
                (
                    json.dumps(item, ensure_ascii=False),
                    token_count,
                    session_id,
                    msg_id,
                ),
            )
            self._conn.commit()
        except sqlite3.Error:
            logger.warning(
                "Failed to update message %d in session %s",
                msg_id,
                session_id,
                exc_info=True,
            )

    def delete_session(self, session_id: str) -> None:
        """Delete all data for a session."""
        try:
            self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            self._conn.commit()
        except sqlite3.Error:
            logger.warning("Failed to delete session %s", session_id, exc_info=True)

    def get_recent_sessions(
        self,
        limit: int,
        exclude_session_id: str | None = None,
    ) -> list[tuple[str, str, str | None]]:
        """Return the most recent completed sessions."""
        try:
            if exclude_session_id:
                cursor = self._conn.execute(
                    "SELECT session_id, started_at, ended_at FROM sessions "
                    "WHERE ended_at IS NOT NULL AND session_id != ? "
                    "ORDER BY started_at DESC LIMIT ?",
                    (exclude_session_id, limit),
                )
            else:
                cursor = self._conn.execute(
                    "SELECT session_id, started_at, ended_at FROM sessions "
                    "WHERE ended_at IS NOT NULL "
                    "ORDER BY started_at DESC LIMIT ?",
                    (limit,),
                )
            return [(row[0], row[1], row[2]) for row in cursor]
        except sqlite3.Error:
            logger.warning("Failed to get recent sessions", exc_info=True)
            return []

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except sqlite3.Error:
            logger.debug("Error closing DB connection", exc_info=True)


def create_storage_backend(config: ConversationHistoryConfig) -> IStorageBackend:
    """Factory: create a storage backend from config."""
    if config.storage_backend == "memory":
        return MemoryStorageBackend()
    elif config.storage_backend == "sqlite":
        if not config.storage_path:
            raise ValueError("storage_path is required for sqlite backend")
        return SQLiteStorageBackend(config.storage_path)
    else:
        raise ValueError(f"Unknown storage backend: {config.storage_backend!r}")
