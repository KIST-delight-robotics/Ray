"""Tests for voice_pipeline.history."""

import pytest

from voice_pipeline.history import ConversationHistory, SQLiteStorageBackend
from voice_pipeline.types import LLMMetrics, Usage


def _token_counter(text: str) -> int:
    """Simple token counter: 1 token per word."""
    return len(text.split())


def _make_history() -> tuple[ConversationHistory, SQLiteStorageBackend]:
    backend = SQLiteStorageBackend(":memory:")
    h = ConversationHistory(backend=backend, token_counter=_token_counter)
    return h, backend


def _make_metrics(output_tokens: int = 10) -> LLMMetrics:
    return LLMMetrics(
        usage=Usage(input_tokens=50, output_tokens=output_tokens),
        model="gpt-4o",
        latency_ms=300,
        ttft_ms=80,
    )


class TestSessionLifecycle:
    def test_no_session_raises(self) -> None:
        h, _ = _make_history()
        with pytest.raises(RuntimeError):
            h.get_messages()

    def test_new_session_and_get_messages(self) -> None:
        h, _ = _make_history()
        h.new_session("s1")
        assert h.get_messages() == []

    def test_new_session_clears_previous(self) -> None:
        h, _ = _make_history()
        h.new_session("s1")
        h.add_user_message("hello")
        h.new_session("s2")
        assert h.get_messages() == []

    def test_new_session_resets_ids(self) -> None:
        h, _ = _make_history()
        h.new_session("s1")
        id0 = h.add_user_message("hello")
        h.new_session("s2")
        id1 = h.add_user_message("world")
        assert id0 == 0
        assert id1 == 0


class TestAddMessages:
    def test_add_user_message(self) -> None:
        h, _ = _make_history()
        h.new_session("s1")
        msg_id = h.add_user_message("hello world")
        assert msg_id == 0
        msgs = h.get_messages()
        assert msgs == [{"role": "user", "content": "hello world"}]

    def test_add_assistant_message(self) -> None:
        h, _ = _make_history()
        h.new_session("s1")
        msg_id = h.add_assistant_message("hi there")
        assert msg_id == 0
        msgs = h.get_messages()
        assert msgs == [{"role": "assistant", "content": "hi there"}]

    def test_add_assistant_with_metrics(self) -> None:
        h, _ = _make_history()
        h.new_session("s1")
        metrics = _make_metrics(output_tokens=5)
        h.add_assistant_message("hi there", metrics=metrics)
        # token_count should use output_tokens (5), not word count (2)
        turns = h.get_turns()
        assert turns[0].token_count == 5

    def test_add_assistant_without_metrics_uses_counter(self) -> None:
        h, _ = _make_history()
        h.new_session("s1")
        h.add_assistant_message("hi there")
        turns = h.get_turns()
        assert turns[0].token_count == 2  # word count

    def test_sequential_ids(self) -> None:
        h, _ = _make_history()
        h.new_session("s1")
        id0 = h.add_user_message("hello")
        id1 = h.add_assistant_message("hi")
        id2 = h.add_user_message("bye")
        assert (id0, id1, id2) == (0, 1, 2)

    def test_conversation_flow(self) -> None:
        h, _ = _make_history()
        h.new_session("s1")
        h.add_user_message("hello")
        h.add_assistant_message("hi")
        h.add_user_message("how are you")
        h.add_assistant_message("good")
        msgs = h.get_messages()
        assert len(msgs) == 4
        assert msgs[0] == {"role": "user", "content": "hello"}
        assert msgs[3] == {"role": "assistant", "content": "good"}


class TestAddMessage:
    def test_add_message_auto_turn(self) -> None:
        h, _ = _make_history()
        h.new_session("s1")
        msg_id, turn_id = h.add_message({"role": "user", "content": "hi"})
        assert msg_id == 0
        assert turn_id == 0

    def test_add_message_explicit_turn(self) -> None:
        h, _ = _make_history()
        h.new_session("s1")
        tid = h.begin_turn()
        fc = {
            "type": "function_call",
            "call_id": "fc1",
            "name": "get_weather",
            "arguments": '{"city":"Seoul"}',
        }
        fco = {"type": "function_call_output", "call_id": "fc1", "output": '{"temp":20}'}
        asst = {"role": "assistant", "content": "It's 20 degrees"}

        m1, t1 = h.add_message(fc, turn_id=tid, metrics=_make_metrics())
        m2, t2 = h.add_message(fco, turn_id=tid)
        m3, t3 = h.add_message(asst, turn_id=tid, metrics=_make_metrics())

        assert t1 == t2 == t3 == tid
        assert m1 == 0 and m2 == 1 and m3 == 2

    def test_begin_turn_without_session_raises(self) -> None:
        h, _ = _make_history()
        with pytest.raises(RuntimeError):
            h.begin_turn()


class TestUpdateMessage:
    def test_update_message(self) -> None:
        h, _ = _make_history()
        h.new_session("s1")
        msg_id = h.add_assistant_message("full text here")
        h.update_message(msg_id, "truncated")
        msgs = h.get_messages()
        assert msgs[0]["content"] == "truncated"

    def test_update_message_recomputes_token_count(self) -> None:
        h, _ = _make_history()
        h.new_session("s1")
        msg_id = h.add_assistant_message("one two three", metrics=_make_metrics(10))
        turns_before = h.get_turns()
        assert turns_before[0].token_count == 10  # from metrics

        h.update_message(msg_id, "one")
        turns_after = h.get_turns()
        assert turns_after[0].token_count == 1  # recomputed by token_counter

    def test_update_nonexistent_raises(self) -> None:
        h, _ = _make_history()
        h.new_session("s1")
        with pytest.raises(RuntimeError):
            h.update_message(999, "nope")


class TestGetTurns:
    def test_simple_turns(self) -> None:
        h, _ = _make_history()
        h.new_session("s1")
        h.add_user_message("hello")
        h.add_assistant_message("hi")
        turns = h.get_turns()
        assert len(turns) == 2
        assert turns[0].items == ({"role": "user", "content": "hello"},)
        assert turns[1].items == ({"role": "assistant", "content": "hi"},)

    def test_tool_call_turn_grouped(self) -> None:
        h, _ = _make_history()
        h.new_session("s1")
        h.add_user_message("weather?")
        tid = h.begin_turn()
        h.add_message(
            {"type": "function_call", "call_id": "fc1", "name": "w", "arguments": "{}"},
            turn_id=tid,
        )
        h.add_message(
            {"type": "function_call_output", "call_id": "fc1", "output": "sunny"},
            turn_id=tid,
        )
        h.add_message(
            {"role": "assistant", "content": "It's sunny"},
            turn_id=tid,
        )
        turns = h.get_turns()
        assert len(turns) == 2  # user turn + tool call turn
        assert len(turns[1].items) == 3  # 3 items in tool call turn

    def test_turn_token_count_is_sum(self) -> None:
        h, _ = _make_history()
        h.new_session("s1")
        tid = h.begin_turn()
        h.add_message({"role": "assistant", "content": "one two"}, turn_id=tid)
        h.add_message({"role": "assistant", "content": "three"}, turn_id=tid)
        turns = h.get_turns()
        assert turns[0].token_count == 3  # 2 + 1

    def test_returns_copies(self) -> None:
        h, _ = _make_history()
        h.new_session("s1")
        h.add_user_message("hello")
        turns = h.get_turns()
        # Mutating returned data should not affect internal state
        turns[0].items[0]["content"] = "tampered"
        assert h.get_messages()[0]["content"] == "hello"


class TestWriteThrough:
    def test_messages_persisted_immediately(self) -> None:
        h, backend = _make_history()
        h.new_session("s1")
        h.add_user_message("hello")
        h.add_assistant_message("hi")
        # Check backend directly without calling save()
        loaded = backend.load_session("s1")
        assert len(loaded) == 2
        assert loaded[0][2]["content"] == "hello"
        assert loaded[1][2]["content"] == "hi"

    def test_update_persisted_immediately(self) -> None:
        h, backend = _make_history()
        h.new_session("s1")
        msg_id = h.add_assistant_message("full text")
        h.update_message(msg_id, "truncated")
        loaded = backend.load_session("s1")
        assert loaded[0][2]["content"] == "truncated"

    def test_save_calls_end_session(self) -> None:
        h, backend = _make_history()
        h.new_session("s1")
        h.save()
        row = backend._conn.execute("SELECT ended_at FROM sessions WHERE session_id = ?", ("s1",)).fetchone()
        assert row is not None and row[0] is not None


class TestGetMessagesReturnsCleanCopies:
    def test_no_internal_ids(self) -> None:
        h, _ = _make_history()
        h.new_session("s1")
        h.add_user_message("hello")
        msgs = h.get_messages()
        assert "msg_id" not in msgs[0]
        assert "turn_id" not in msgs[0]

    def test_mutation_safety(self) -> None:
        h, _ = _make_history()
        h.new_session("s1")
        h.add_user_message("hello")
        msgs = h.get_messages()
        msgs[0]["content"] = "tampered"
        assert h.get_messages()[0]["content"] == "hello"

    def test_list_mutation_safety(self) -> None:
        h, _ = _make_history()
        h.new_session("s1")
        h.add_user_message("hello")
        msgs = h.get_messages()
        msgs.append({"role": "user", "content": "injected"})
        assert len(h.get_messages()) == 1
