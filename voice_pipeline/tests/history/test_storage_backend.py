"""Tests for voice_pipeline.history.storage_backend."""

import json
from pathlib import Path

import pytest

from voice_pipeline.core.interfaces import IStorageBackend
from voice_pipeline.history.storage_backend import (
    MemoryStorageBackend,
    SQLiteStorageBackend,
)


def _user_item(text: str = "hello") -> dict:
    return {"role": "user", "content": text}


def _assistant_item(text: str = "hi") -> dict:
    return {"role": "assistant", "content": text}


def _metrics_json() -> str:
    return json.dumps(
        {
            "usage": {"input_tokens": 50, "output_tokens": 10},
            "model": "gpt-4o",
            "latency_ms": 300,
            "ttft_ms": 80,
        }
    )


class _BackendTests:
    """Shared test suite for all IStorageBackend implementations."""

    def make_backend(self) -> IStorageBackend:
        raise NotImplementedError

    def test_load_empty_session(self) -> None:
        b = self.make_backend()
        assert b.load_session("nonexistent") == []

    def test_create_and_load_session(self) -> None:
        b = self.make_backend()
        b.create_session("s1", "2026-03-29 10:00:00")
        assert b.load_session("s1") == []

    def test_append_and_load_message(self) -> None:
        b = self.make_backend()
        b.create_session("s1", "2026-03-29 10:00:00")
        item = _user_item()
        b.append_message("s1", msg_id=0, turn_id=0, item=item, token_count=3)
        loaded = b.load_session("s1")
        assert len(loaded) == 1
        msg_id, turn_id, loaded_item, tc = loaded[0]
        assert msg_id == 0
        assert turn_id == 0
        assert loaded_item == item
        assert tc == 3

    def test_append_multiple_messages(self) -> None:
        b = self.make_backend()
        b.create_session("s1", "2026-03-29 10:00:00")
        b.append_message("s1", 0, 0, _user_item("hello"), 3)
        b.append_message("s1", 1, 1, _assistant_item("hi"), 2)
        b.append_message("s1", 2, 2, _user_item("bye"), 3)
        loaded = b.load_session("s1")
        assert len(loaded) == 3
        assert loaded[0][0] == 0  # msg_id order
        assert loaded[2][0] == 2

    def test_append_with_metrics(self) -> None:
        b = self.make_backend()
        b.create_session("s1", "2026-03-29 10:00:00")
        b.append_message("s1", 0, 0, _assistant_item(), 2, metrics_json=_metrics_json())
        loaded = b.load_session("s1")
        assert len(loaded) == 1

    def test_update_message(self) -> None:
        b = self.make_backend()
        b.create_session("s1", "2026-03-29 10:00:00")
        b.append_message("s1", 0, 0, _assistant_item("full text"), 5)
        b.update_message("s1", 0, _assistant_item("truncated"), 3)
        loaded = b.load_session("s1")
        assert loaded[0][2] == _assistant_item("truncated")
        assert loaded[0][3] == 3

    def test_delete_session(self) -> None:
        b = self.make_backend()
        b.create_session("s1", "2026-03-29 10:00:00")
        b.append_message("s1", 0, 0, _user_item(), 3)
        b.delete_session("s1")
        assert b.load_session("s1") == []

    def test_delete_nonexistent_is_noop(self) -> None:
        b = self.make_backend()
        b.delete_session("nonexistent")  # should not raise

    def test_multiple_sessions(self) -> None:
        b = self.make_backend()
        b.create_session("s1", "2026-03-29 10:00:00")
        b.create_session("s2", "2026-03-29 11:00:00")
        b.append_message("s1", 0, 0, _user_item("a"), 3)
        b.append_message("s2", 0, 0, _user_item("b"), 3)
        assert b.load_session("s1")[0][2]["content"] == "a"
        assert b.load_session("s2")[0][2]["content"] == "b"

    def test_end_session(self) -> None:
        b = self.make_backend()
        b.create_session("s1", "2026-03-29 10:00:00")
        b.end_session("s1", "2026-03-29 10:05:00")
        # end_session should not affect messages
        assert b.load_session("s1") == []

    def test_load_returns_deep_copy(self) -> None:
        b = self.make_backend()
        b.create_session("s1", "2026-03-29 10:00:00")
        b.append_message("s1", 0, 0, _user_item("hello"), 3)
        loaded = b.load_session("s1")
        loaded[0][2]["content"] = "tampered"
        reloaded = b.load_session("s1")
        assert reloaded[0][2]["content"] == "hello"

    def test_load_message_existing(self) -> None:
        b = self.make_backend()
        b.create_session("s1", "2026-03-29 10:00:00")
        b.append_message("s1", 0, 0, _user_item("hello"), 3)
        result = b.load_message("s1", 0)
        assert result is not None
        assert result == (0, 0, _user_item("hello"), 3)

    def test_load_message_nonexistent(self) -> None:
        b = self.make_backend()
        b.create_session("s1", "2026-03-29 10:00:00")
        assert b.load_message("s1", 999) is None

    def test_load_message_nonexistent_session(self) -> None:
        b = self.make_backend()
        assert b.load_message("nonexistent", 0) is None

    def test_get_latest_session(self) -> None:
        b = self.make_backend()
        b.create_session("s1", "2026-03-29 10:00:00")
        b.create_session("s2", "2026-03-29 11:00:00")
        b.append_message("s1", 0, 0, _user_item("a"), 3)
        b.append_message("s2", 0, 0, _user_item("b"), 3)
        assert b.get_latest_session() == ("s2", "2026-03-29 11:00:00")

    def test_get_latest_session_excludes_given(self) -> None:
        b = self.make_backend()
        b.create_session("s1", "2026-03-29 10:00:00")
        b.create_session("s2", "2026-03-29 11:00:00")
        b.append_message("s1", 0, 0, _user_item("a"), 3)
        b.append_message("s2", 0, 0, _user_item("b"), 3)
        assert b.get_latest_session(exclude_session_id="s2") == ("s1", "2026-03-29 10:00:00")

    def test_get_latest_session_skips_empty_sessions(self) -> None:
        """Sessions without messages (e.g. silent wake) are not carryover candidates."""
        b = self.make_backend()
        b.create_session("s1", "2026-03-29 10:00:00")
        b.create_session("s2", "2026-03-29 11:00:00")  # newer but empty
        b.append_message("s1", 0, 0, _user_item("a"), 3)
        assert b.get_latest_session() == ("s1", "2026-03-29 10:00:00")

    def test_get_latest_session_none(self) -> None:
        b = self.make_backend()
        assert b.get_latest_session() is None
        b.create_session("s1", "2026-03-29 10:00:00")  # empty session only
        assert b.get_latest_session() is None

    def test_rolling_summary_roundtrip(self) -> None:
        b = self.make_backend()
        b.create_session("s1", "2026-03-29 10:00:00")
        b.save_rolling_summary("s1", "They discussed the weather.", 7)
        assert b.load_rolling_summary("s1") == ("They discussed the weather.", 7)

    def test_rolling_summary_missing(self) -> None:
        b = self.make_backend()
        assert b.load_rolling_summary("s1") is None

    def test_rolling_summary_keeps_latest_session_only(self) -> None:
        b = self.make_backend()
        b.save_rolling_summary("s1", "First summary.", 3)
        b.save_rolling_summary("s2", "Second summary.", 5)
        assert b.load_rolling_summary("s1") is None
        assert b.load_rolling_summary("s2") == ("Second summary.", 5)

    def test_rolling_summary_overwrite_same_session(self) -> None:
        b = self.make_backend()
        b.save_rolling_summary("s1", "Early summary.", 3)
        b.save_rolling_summary("s1", "Merged summary.", 9)
        assert b.load_rolling_summary("s1") == ("Merged summary.", 9)

    def test_tool_call_turn(self) -> None:
        """Store a multi-message tool call turn with shared turn_id."""
        b = self.make_backend()
        b.create_session("s1", "2026-03-29 10:00:00")
        fc = {
            "type": "function_call",
            "call_id": "fc1",
            "name": "get_weather",
            "arguments": '{"city":"Seoul"}',
        }
        fco = {"type": "function_call_output", "call_id": "fc1", "output": '{"temp":20}'}
        asst = {"role": "assistant", "content": "It's 20 degrees"}
        b.append_message("s1", 0, 0, _user_item("weather?"), 3)
        b.append_message("s1", 1, 1, fc, 10, metrics_json=_metrics_json())
        b.append_message("s1", 2, 1, fco, 8)
        b.append_message("s1", 3, 1, asst, 5, metrics_json=_metrics_json())
        loaded = b.load_session("s1")
        assert len(loaded) == 4
        # turn_id grouping: msg 1,2,3 share turn_id=1
        assert loaded[1][1] == loaded[2][1] == loaded[3][1] == 1


class TestMemoryStorageBackend(_BackendTests):
    def make_backend(self) -> IStorageBackend:
        return MemoryStorageBackend()


class TestSQLiteStorageBackend(_BackendTests):
    @pytest.fixture(autouse=True)
    def _setup_db(self, tmp_path: Path) -> None:
        self._db_path = str(tmp_path / "test.db")

    def make_backend(self) -> IStorageBackend:
        return SQLiteStorageBackend(self._db_path)

    def test_wal_mode_enabled(self) -> None:
        b = SQLiteStorageBackend(self._db_path)
        cursor = b._conn.execute("PRAGMA journal_mode")
        assert cursor.fetchone()[0] == "wal"

    def test_persistence_across_instances(self) -> None:
        b1 = SQLiteStorageBackend(self._db_path)
        b1.create_session("s1", "2026-03-29 10:00:00")
        b1.append_message("s1", 0, 0, _user_item("hello"), 3)
        b1.close()

        b2 = SQLiteStorageBackend(self._db_path)
        loaded = b2.load_session("s1")
        assert len(loaded) == 1
        assert loaded[0][2]["content"] == "hello"
        b2.close()

    def test_rolling_summary_persists_across_instances(self) -> None:
        b1 = SQLiteStorageBackend(self._db_path)
        b1.save_rolling_summary("s1", "Persisted summary.", 4)
        b1.close()

        b2 = SQLiteStorageBackend(self._db_path)
        assert b2.load_rolling_summary("s1") == ("Persisted summary.", 4)
        b2.close()

    def test_graceful_insert_failure(self) -> None:
        """append_message should not crash on duplicate msg_id."""
        b = SQLiteStorageBackend(self._db_path)
        b.create_session("s1", "2026-03-29 10:00:00")
        b.append_message("s1", 0, 0, _user_item(), 3)
        # Duplicate msg_id — should log warning, not raise
        b.append_message("s1", 0, 0, _user_item("dup"), 3)
        loaded = b.load_session("s1")
        assert len(loaded) == 1  # original preserved
        b.close()
