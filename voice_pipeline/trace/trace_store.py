"""Pipeline trace storage implementations."""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

from voice_pipeline.core.types import PipelineTrace

logger = logging.getLogger("voice_pipeline.trace")

_COLUMNS = (
    "session_id",
    "run_id",
    "pipeline_mode",
    "created_at",
    "outcome",
    "speculative_attempts",
    "user_msg_id",
    "memory_ms",
    "context_ms",
    "llm_ms",
    "llm_ttft_ms",
    "tts_ms",
    "tts_ttfc_ms",
    "prepare_to_streaming_ms",
    "turn_shift_to_playback_ms",
    "speculative_ms",
    "bridge_ms",
)


class SQLiteTraceStore:
    """Persists PipelineTrace records to a SQLite database.

    Opens its own connection to the shared DB file (WAL mode).
    Thread-safe: a lock serializes all connection access.
    """

    def __init__(self, db_path: str) -> None:
        self._lock = threading.Lock()
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_traces (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id              TEXT    NOT NULL,
                run_id                  INTEGER NOT NULL,
                pipeline_mode           TEXT    NOT NULL,
                created_at              TEXT    NOT NULL,
                outcome                 TEXT    NOT NULL,
                speculative_attempts    INTEGER NOT NULL DEFAULT 1,
                user_msg_id             INTEGER NOT NULL DEFAULT 0,
                memory_ms               REAL    NOT NULL DEFAULT 0,
                context_ms              REAL    NOT NULL DEFAULT 0,
                llm_ms                  REAL    NOT NULL DEFAULT 0,
                llm_ttft_ms             REAL    NOT NULL DEFAULT 0,
                tts_ms                  REAL    NOT NULL DEFAULT 0,
                tts_ttfc_ms             REAL    NOT NULL DEFAULT 0,
                prepare_to_streaming_ms REAL    NOT NULL DEFAULT 0,
                turn_shift_to_playback_ms REAL  NOT NULL DEFAULT 0,
                speculative_ms          REAL    NOT NULL DEFAULT 0,
                bridge_ms               REAL    NOT NULL DEFAULT 0
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_traces_session "
            "ON pipeline_traces(session_id)"
        )
        self._conn.commit()

    def save(self, trace: PipelineTrace) -> None:
        """Persist a trace record."""
        record = trace.to_record()
        values = tuple(record[col] for col in _COLUMNS)
        placeholders = ", ".join("?" for _ in _COLUMNS)
        col_names = ", ".join(_COLUMNS)
        with self._lock:
            self._conn.execute(
                f"INSERT INTO pipeline_traces ({col_names}) VALUES ({placeholders})",
                values,
            )
            self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()


class InMemoryTraceStore:
    """In-memory trace store for unit tests."""

    def __init__(self) -> None:
        self.traces: list[PipelineTrace] = []

    def save(self, trace: PipelineTrace) -> None:
        """Append trace to in-memory list."""
        self.traces.append(trace)

    def close(self) -> None:
        """No-op."""
