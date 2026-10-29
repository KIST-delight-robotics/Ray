"""세션 대화 히스토리.

- ``ConversationHistory``: 세션 중 읽기는 메모리에서, 모든 변경은 SQLite에 write-through.
  메시지 1건 = 1 row (Responses API 입력 형식), 메시지마다 ``turn_id`` 를 매긴다.
- ``SQLiteStorageBackend``: WAL 모드. 손상 시 단계적 복구(정상 open → WAL 삭제 → 새 DB).
  INSERT 실패는 경고만 남기고 메모리 전용으로 계속 진행한다. 테스트는 ``":memory:"`` 경로를 쓴다.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from voice_pipeline.types import LLMMetrics, TokenCounter

logger = logging.getLogger("voice_pipeline.history")


@dataclass(frozen=True)
class HistoryTurn:
    """Atomic history unit — one or more message items that belong to the same turn.

    Attributes:
        items: Message dicts in Responses API input format.
        token_count: Pre-computed total token count for all items.
        turn_id: Monotonically increasing turn identifier within the session.
    """

    items: tuple[dict[str, Any], ...]
    token_count: int
    turn_id: int


# Module-level constants
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"  # UTC 세션 timestamp 포맷 (no timezone offset)


class SQLiteStorageBackend:
    """SQLite write-through storage backend.

    Uses WAL mode for concurrent read/write safety.
    Graduated corruption recovery: normal open → WAL delete → new DB.

    Thread-safe via an internal lock.
    """

    def __init__(self, db_path: str) -> None:
        self._lock = threading.Lock()
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
            with self._lock:
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
            with self._lock:
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
            with self._lock:
                cursor = self._conn.execute(
                    "SELECT msg_id, turn_id, item_json, token_count FROM messages WHERE session_id = ? ORDER BY msg_id",
                    (session_id,),
                )
                rows = cursor.fetchall()
            return [(row[0], row[1], json.loads(row[2]), row[3]) for row in rows]
        except sqlite3.Error:
            logger.warning("Failed to load session %s", session_id, exc_info=True)
            return []

    def load_message(self, session_id: str, msg_id: int) -> tuple[int, int, dict[str, Any], int] | None:
        """Load a single message by ID."""
        try:
            with self._lock:
                cursor = self._conn.execute(
                    "SELECT msg_id, turn_id, item_json, token_count FROM messages WHERE session_id = ? AND msg_id = ?",
                    (session_id, msg_id),
                )
                row = cursor.fetchone()
            if row is None:
                return None
            return (row[0], row[1], json.loads(row[2]), row[3])
        except sqlite3.Error:
            logger.warning("Failed to load message %d from session %s", msg_id, session_id, exc_info=True)
            return None

    def append_message(
        self,
        session_id: str,
        msg_id: int,
        turn_id: int,
        item: dict[str, Any],
        token_count: int,
        metrics_json: str | None = None,
    ) -> None:
        """Append a message. Graceful on failure."""
        try:
            with self._lock:
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
            with self._lock:
                self._conn.execute(
                    "UPDATE messages SET item_json = ?, token_count = ? WHERE session_id = ? AND msg_id = ?",
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

    def close(self) -> None:
        """Close the database connection."""
        try:
            with self._lock:
                self._conn.close()
        except sqlite3.Error:
            logger.debug("Error closing DB connection", exc_info=True)


class ConversationHistory:
    """Entry-based conversation history backed by a storage backend.

    The storage backend is the single source of truth for all reads
    and writes. Every mutation is persisted immediately for crash safety.

    Thread-safe via threading.Lock.
    """

    def __init__(self, backend: SQLiteStorageBackend, token_counter: TokenCounter) -> None:
        self._backend = backend
        self._token_counter = token_counter
        self._lock = threading.Lock()
        self._session_id: str | None = None
        self._next_msg_id: int = 0
        self._next_turn_id: int = 0

    def _require_session(self) -> str:
        if self._session_id is None:
            raise RuntimeError("No active session. Call new_session() first.")
        return self._session_id

    def _allocate_msg_id(self) -> int:
        msg_id = self._next_msg_id
        self._next_msg_id += 1
        return msg_id

    def _allocate_turn_id(self) -> int:
        turn_id = self._next_turn_id
        self._next_turn_id += 1
        return turn_id

    def _compute_token_count(self, text: str, metrics: LLMMetrics | None) -> int:
        """Use LLM output_tokens if available, fallback to token_counter."""
        if metrics is not None:
            return metrics.usage.output_tokens
        return self._safe_count(text)

    def _safe_count(self, text: str) -> int:
        """Count tokens with graceful fallback on error."""
        try:
            return self._token_counter(text)
        except Exception:
            logger.warning("Token counter failed, using 0", exc_info=True)
            return 0

    @staticmethod
    def _serialize_metrics(metrics: LLMMetrics | None) -> str | None:
        """Serialize LLMMetrics to JSON string for DB storage."""
        if metrics is None:
            return None
        return json.dumps(
            {
                "usage": {
                    "input_tokens": metrics.usage.input_tokens,
                    "output_tokens": metrics.usage.output_tokens,
                    "cached_tokens": metrics.usage.cached_tokens,
                    "reasoning_tokens": metrics.usage.reasoning_tokens,
                },
                "model": metrics.model,
                "latency_ms": metrics.latency_ms,
                "ttft_ms": metrics.ttft_ms,
            }
        )

    def _write_message(
        self,
        session_id: str,
        msg_id: int,
        turn_id: int,
        item: dict[str, Any],
        token_count: int,
        metrics: LLMMetrics | None = None,
    ) -> None:
        """Persist message to backend. Logs warning on failure."""
        try:
            self._backend.append_message(
                session_id,
                msg_id,
                turn_id,
                item,
                token_count,
                self._serialize_metrics(metrics),
            )
        except Exception:
            logger.warning(
                "Failed to persist message %d — message will be missing from history",
                msg_id,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def new_session(self, session_id: str) -> None:
        """Start a new session, clearing any previous state."""
        with self._lock:
            self._session_id = session_id
            self._next_msg_id = 0
            self._next_turn_id = 0
            started_at = datetime.now(UTC).strftime(TIMESTAMP_FORMAT)
            try:
                self._backend.create_session(session_id, started_at)
            except Exception:
                logger.warning("Failed to create session in backend", exc_info=True)
            logger.debug("Started new session: %s", session_id)

    def add_user_message(self, text: str) -> int:
        """Append a user message. Auto-assigns turn_id."""
        with self._lock:
            session_id = self._require_session()
            msg_id = self._allocate_msg_id()
            turn_id = self._allocate_turn_id()
            token_count = self._safe_count(text)
            item: dict[str, Any] = {"role": "user", "content": text}
            self._write_message(session_id, msg_id, turn_id, item, token_count)
            return msg_id

    def add_assistant_message(self, text: str, metrics: LLMMetrics | None = None) -> int:
        """Append an assistant text message. Auto-assigns turn_id."""
        with self._lock:
            session_id = self._require_session()
            msg_id = self._allocate_msg_id()
            turn_id = self._allocate_turn_id()
            token_count = self._compute_token_count(text, metrics)
            item: dict[str, Any] = {"role": "assistant", "content": text}
            self._write_message(session_id, msg_id, turn_id, item, token_count, metrics)
            return msg_id

    def update_message(self, msg_id: int, text: str) -> None:
        """Update message text. Recomputes token_count internally."""
        with self._lock:
            session_id = self._require_session()
            result = self._backend.load_message(session_id, msg_id)
            if result is None:
                raise RuntimeError(f"No message with ID {msg_id}")
            _, _, item, _ = result
            item["content"] = text
            token_count = self._safe_count(text)
            try:
                self._backend.update_message(session_id, msg_id, item, token_count)
            except Exception:
                logger.warning(
                    "Failed to update message %d in backend",
                    msg_id,
                    exc_info=True,
                )

    def get_turns(self) -> list[HistoryTurn]:
        """Retrieve messages grouped by turn."""
        with self._lock:
            session_id = self._require_session()
            rows = self._backend.load_session(session_id)

            groups: dict[int, list[tuple[dict[str, Any], int]]] = {}
            first_msg_id: dict[int, int] = {}
            for msg_id, turn_id, item, token_count in rows:
                if turn_id not in groups:
                    groups[turn_id] = []
                    first_msg_id[turn_id] = msg_id
                groups[turn_id].append((item, token_count))

            sorted_turn_ids = sorted(groups.keys(), key=lambda tid: first_msg_id[tid])
            return [
                HistoryTurn(
                    items=tuple(item for item, _ in groups[tid]),
                    token_count=sum(tc for _, tc in groups[tid]),
                    turn_id=tid,
                )
                for tid in sorted_turn_ids
            ]

    def save(self) -> None:
        """Finalize the current session (sets ended_at)."""
        with self._lock:
            session_id = self._require_session()
            ended_at = datetime.now(UTC).strftime(TIMESTAMP_FORMAT)
            try:
                self._backend.end_session(session_id, ended_at)
            except Exception:
                logger.warning("Failed to end session in backend", exc_info=True)
