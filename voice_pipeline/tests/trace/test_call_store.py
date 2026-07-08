"""Tests for call store implementations."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from voice_pipeline.core.types import CallRecord
from voice_pipeline.trace.trace_store import InMemoryCallStore, SQLiteCallStore


def _make_record(**overrides: object) -> CallRecord:
    defaults: dict[str, object] = {
        "session_id": "test-session",
        "timestamp": "2026-06-09 12:00:00",
        "module": "embedder",
        "operation": "embed_batch",
        "model": "all-MiniLM-L6-v2",
        "elapsed_ms": 42.5,
        "status": "ok",
        "metadata": None,
    }
    defaults.update(overrides)
    return CallRecord(**defaults)  # type: ignore[arg-type]


class TestSQLiteCallStore:
    @pytest.fixture(autouse=True)
    def _setup_db(self, tmp_path: Path) -> None:
        self._db_path = str(tmp_path / "test.db")

    def _make_store(self) -> SQLiteCallStore:
        return SQLiteCallStore(self._db_path)

    def test_table_creation(self) -> None:
        store = self._make_store()
        conn = sqlite3.connect(self._db_path)
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "call_records" in tables
        conn.close()
        store.close()

    def test_record_and_query(self) -> None:
        store = self._make_store()
        store.record(_make_record())
        store.close()

        conn = sqlite3.connect(self._db_path)
        rows = conn.execute("SELECT * FROM call_records").fetchall()
        assert len(rows) == 1
        conn.close()

    def test_record_preserves_values(self) -> None:
        store = self._make_store()
        store.record(_make_record(elapsed_ms=99.9, status="error", metadata='{"error": "timeout"}'))
        store.close()

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM call_records").fetchone()
        assert row["session_id"] == "test-session"
        assert row["module"] == "embedder"
        assert row["operation"] == "embed_batch"
        assert row["model"] == "all-MiniLM-L6-v2"
        assert row["elapsed_ms"] == pytest.approx(99.9)
        assert row["status"] == "error"
        assert row["metadata"] == '{"error": "timeout"}'
        conn.close()

    def test_record_multiple(self) -> None:
        store = self._make_store()
        for i in range(5):
            store.record(_make_record(elapsed_ms=float(i)))
        store.close()

        conn = sqlite3.connect(self._db_path)
        count = conn.execute("SELECT COUNT(*) FROM call_records").fetchone()[0]
        assert count == 5
        conn.close()

    def test_null_metadata(self) -> None:
        store = self._make_store()
        store.record(_make_record(metadata=None))
        store.close()

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM call_records").fetchone()
        assert row["metadata"] is None
        conn.close()

    def test_default_status_ok(self) -> None:
        store = self._make_store()
        store.record(_make_record())
        store.close()

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM call_records").fetchone()
        assert row["status"] == "ok"
        conn.close()

    def test_turn_index_persisted(self) -> None:
        store = self._make_store()
        store.record(_make_record(turn_index=3))
        store.close()

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM call_records").fetchone()
        assert row["turn_index"] == 3
        conn.close()

    def test_turn_index_defaults_to_zero(self) -> None:
        store = self._make_store()
        store.record(_make_record())
        store.close()

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM call_records").fetchone()
        assert row["turn_index"] == 0
        conn.close()

    def test_legacy_db_migrated(self) -> None:
        # A pre-change DB without the turn_index column must gain it on open.
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "CREATE TABLE call_records ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, "
            "timestamp TEXT NOT NULL, module TEXT NOT NULL, operation TEXT NOT NULL, "
            "model TEXT NOT NULL, elapsed_ms REAL NOT NULL, "
            "status TEXT NOT NULL DEFAULT 'ok', metadata TEXT)"
        )
        conn.commit()
        conn.close()

        store = self._make_store()
        store.record(_make_record(turn_index=2))
        store.close()

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        cols = {r[1] for r in conn.execute("PRAGMA table_info(call_records)")}
        assert "turn_index" in cols
        row = conn.execute("SELECT * FROM call_records").fetchone()
        assert row["turn_index"] == 2
        conn.close()

    def test_turn_index_counter(self) -> None:
        store = self._make_store()
        assert store.current_turn_index == 0
        store.set_turn_index(5)
        assert store.current_turn_index == 5
        store.close()

    def test_thread_safety(self) -> None:
        store = self._make_store()
        errors: list[Exception] = []

        def writer(thread_id: int) -> None:
            try:
                for i in range(20):
                    store.record(_make_record(elapsed_ms=float(thread_id * 100 + i)))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        store.close()

        assert not errors
        conn = sqlite3.connect(self._db_path)
        count = conn.execute("SELECT COUNT(*) FROM call_records").fetchone()[0]
        assert count == 80
        conn.close()


class TestInMemoryCallStore:
    def test_record(self) -> None:
        store = InMemoryCallStore()
        rec = _make_record()
        store.record(rec)
        assert len(store.records) == 1
        assert store.records[0] is rec

    def test_record_multiple(self) -> None:
        store = InMemoryCallStore()
        for i in range(3):
            store.record(_make_record(elapsed_ms=float(i)))
        assert len(store.records) == 3

    def test_close_noop(self) -> None:
        store = InMemoryCallStore()
        store.close()
