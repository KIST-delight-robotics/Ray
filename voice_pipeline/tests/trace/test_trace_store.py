"""Tests for trace storage implementations."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from voice_pipeline.core.types import PipelineTrace
from voice_pipeline.trace.trace_store import InMemoryTraceStore, SQLiteTraceStore


def _make_trace(**overrides: object) -> PipelineTrace:
    defaults: dict[str, object] = {
        "session_id": "test-session",
        "run_id": 1,
        "pipeline_mode": "full",
        "created_at": "2026-04-07 12:00:00",
        "outcome": "completed",
        "prepare_ts": 100.0,
        "turn_shift_ts": 100.8,
        "begin_streaming_ts": 101.0,
        "playback_started_ts": 101.05,
        "pipeline_start_ts": 100.01,
        "memory_done_ts": 100.05,
        "context_done_ts": 100.06,
        "llm_start_ts": 100.06,
        "llm_done_ts": 100.7,
        "tts_start_ts": 100.7,
        "tts_first_chunk_ts": 100.85,
        "tts_done_ts": 101.0,
        "llm_ttft_ms": 180.0,
    }
    defaults.update(overrides)
    return PipelineTrace(**defaults)  # type: ignore[arg-type]


class TestSQLiteTraceStore:
    @pytest.fixture(autouse=True)
    def _setup_db(self, tmp_path: Path) -> None:
        self._db_path = str(tmp_path / "test.db")

    def _make_store(self) -> SQLiteTraceStore:
        return SQLiteTraceStore(self._db_path)

    def test_table_creation(self) -> None:
        store = self._make_store()
        conn = sqlite3.connect(self._db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {row[0] for row in tables}
        assert "pipeline_traces" in table_names
        conn.close()
        store.close()

    def test_save_and_query(self) -> None:
        store = self._make_store()
        trace = _make_trace()
        store.save(trace)
        store.close()

        conn = sqlite3.connect(self._db_path)
        rows = conn.execute("SELECT * FROM pipeline_traces").fetchall()
        assert len(rows) == 1
        conn.close()

    def test_save_preserves_values(self) -> None:
        store = self._make_store()
        trace = _make_trace(speculative_attempts=3)
        store.save(trace)
        store.close()

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM pipeline_traces").fetchone()
        assert row["session_id"] == "test-session"
        assert row["outcome"] == "completed"
        assert row["speculative_attempts"] == 3
        assert row["llm_ttft_ms"] == 180.0
        assert row["memory_ms"] == pytest.approx(40.0, abs=1)
        conn.close()

    def test_save_multiple(self) -> None:
        store = self._make_store()
        for i in range(5):
            store.save(_make_trace(run_id=i))
        store.close()

        conn = sqlite3.connect(self._db_path)
        count = conn.execute("SELECT COUNT(*) FROM pipeline_traces").fetchone()[0]
        assert count == 5
        conn.close()

    def test_interrupt_latency_stored(self) -> None:
        store = self._make_store()
        trace = _make_trace(
            outcome="truncated",
            interrupt_ts=100.8,
            interrupt_ack_ts=100.85,
        )
        store.save(trace)
        store.close()

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM pipeline_traces").fetchone()
        assert row["interrupt_latency_ms"] == pytest.approx(50.0, abs=1)
        conn.close()

    def test_cancelled_trace_zero_durations(self) -> None:
        store = self._make_store()
        trace = PipelineTrace(
            session_id="s1",
            run_id=1,
            pipeline_mode="full",
            created_at="2026-04-07 12:00:00",
            outcome="cancelled",
            prepare_ts=100.0,
            pipeline_start_ts=100.01,
            memory_done_ts=100.05,
        )
        store.save(trace)
        store.close()

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM pipeline_traces").fetchone()
        assert row["outcome"] == "cancelled"
        assert row["llm_ms"] == 0.0
        assert row["tts_ms"] == 0.0
        assert row["turn_shift_to_playback_ms"] == 0.0
        conn.close()


class TestInMemoryTraceStore:
    def test_save(self) -> None:
        store = InMemoryTraceStore()
        trace = _make_trace()
        store.save(trace)
        assert len(store.traces) == 1
        assert store.traces[0] is trace

    def test_save_multiple(self) -> None:
        store = InMemoryTraceStore()
        for i in range(3):
            store.save(_make_trace(run_id=i))
        assert len(store.traces) == 3

    def test_close_noop(self) -> None:
        store = InMemoryTraceStore()
        store.close()  # should not raise
