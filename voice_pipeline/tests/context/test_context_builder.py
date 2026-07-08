"""Tests for voice_pipeline.context.context_builder."""

from __future__ import annotations

from typing import Any

import pytest

from voice_pipeline.context.context_builder import (
    _PER_MESSAGE_OVERHEAD_TOKENS,
    ContextBuilder,
)
from voice_pipeline.core.types import HistorySummarySnapshot, HistoryTurn, LLMMetrics
from voice_pipeline.memory.types import Episode, MemoryReadResult, Profile

# Per-message overhead shorthand
_MO = _PER_MESSAGE_OVERHEAD_TOKENS


class StubHistory:
    """Minimal history stub for testing ContextBuilder."""

    def __init__(self, messages: list[dict[str, Any]] | None = None) -> None:
        self._turns: list[HistoryTurn] = []
        if messages:
            for msg in messages:
                text = msg.get("content", "")
                tc = len(text.split()) if text and text.strip() else 0
                self._append_turn((msg,), tc)

    def _append_turn(self, items: tuple[dict[str, Any], ...], token_count: int) -> None:
        self._turns.append(HistoryTurn(items=items, token_count=token_count, turn_id=len(self._turns)))

    def new_session(self, session_id: str) -> None:
        pass

    def add_user_message(self, text: str) -> int:
        tc = len(text.split()) if text.strip() else 0
        self._append_turn(({"role": "user", "content": text},), tc)
        return len(self._turns) - 1

    def add_assistant_message(self, text: str, metrics: LLMMetrics | None = None) -> int:
        tc = len(text.split()) if text.strip() else 0
        self._append_turn(({"role": "assistant", "content": text},), tc)
        return len(self._turns) - 1

    def add_message(
        self, item: dict[str, Any], turn_id: int | None = None, metrics: LLMMetrics | None = None
    ) -> tuple[int, int]:
        return 0, 0

    def begin_turn(self) -> int:
        return 0

    def update_message(self, msg_id: int, text: str) -> None:
        pass

    def get_messages(self) -> list[dict[str, Any]]:
        return [item for turn in self._turns for item in turn.items]

    def get_turns(self) -> list[HistoryTurn]:
        return list(self._turns)

    def save(self) -> None:
        pass

    def inject_turn(self, items: list[dict[str, Any]], token_count: int) -> None:
        self._append_turn(tuple(items), token_count)


class StubSummarizer:
    """Records maybe_schedule calls and serves a fixed snapshot."""

    def __init__(self, snapshot: HistorySummarySnapshot | None = None) -> None:
        self._snapshot = snapshot
        self.scheduled: list[list[HistoryTurn]] = []

    def snapshot(self) -> HistorySummarySnapshot | None:
        return self._snapshot

    def maybe_schedule(self, turns: list[HistoryTurn]) -> None:
        self.scheduled.append(list(turns))

    def close(self) -> None:
        pass


def _word_counter(text: str) -> int:
    return len(text.split()) if text.strip() else 0


def _turn_cost(*msg_tokens: int) -> int:
    """History-budget cost of single-item turns with the given token counts."""
    return sum(t + _MO for t in msg_tokens)


def _set_budgets(
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_history: int = ContextBuilder._MAX_HISTORY_TOKENS,
    max_memory: int = ContextBuilder._MAX_MEMORY_TOKENS,
    max_profile: int = ContextBuilder._MAX_PROFILE_TOKENS,
    max_prev: int = ContextBuilder._MAX_PREV_SESSION_TOKENS,
) -> None:
    """Override ContextBuilder token budget class vars for a test.

    Uses monkeypatch so values auto-revert at test scope.
    """
    monkeypatch.setattr(ContextBuilder, "_MAX_HISTORY_TOKENS", max_history)
    monkeypatch.setattr(ContextBuilder, "_MAX_MEMORY_TOKENS", max_memory)
    monkeypatch.setattr(ContextBuilder, "_MAX_PROFILE_TOKENS", max_profile)
    monkeypatch.setattr(ContextBuilder, "_MAX_PREV_SESSION_TOKENS", max_prev)


class TestContextBuilder:
    def test_basic_with_system_prompt(self) -> None:
        history = StubHistory()
        cb = ContextBuilder(history, "You are helpful.", _word_counter)
        result = cb.build("hello")
        assert result == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
        ]

    def test_no_system_prompt(self) -> None:
        history = StubHistory()
        cb = ContextBuilder(history, "", _word_counter)
        result = cb.build("hello")
        assert result == [{"role": "user", "content": "hello"}]

    def test_includes_history(self) -> None:
        history = StubHistory(
            [
                {"role": "user", "content": "hi"},  # 1 token
                {"role": "assistant", "content": "hello there"},  # 2 tokens
            ]
        )
        cb = ContextBuilder(history, "", _word_counter)
        result = cb.build("how are you")
        assert len(result) == 3
        assert result[0] == {"role": "user", "content": "hi"}
        assert result[1] == {"role": "assistant", "content": "hello there"}
        assert result[2] == {"role": "user", "content": "how are you"}

    def test_history_budget_trims_oldest_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        history = StubHistory(
            [
                {"role": "user", "content": "old message here"},  # 3 tokens
                {"role": "assistant", "content": "old reply here"},  # 3 tokens
                {"role": "user", "content": "recent"},  # 1 token
                {"role": "assistant", "content": "recent reply"},  # 2 tokens
            ]
        )
        # History budget fits only the two recent messages
        _set_budgets(monkeypatch, max_history=_turn_cost(1, 2))
        cb = ContextBuilder(history, "", _word_counter)
        result = cb.build("now")
        assert len(result) == 3
        assert result[0] == {"role": "user", "content": "recent"}
        assert result[1] == {"role": "assistant", "content": "recent reply"}
        assert result[2] == {"role": "user", "content": "now"}

    def test_zero_history_budget_keeps_fixed_blocks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        history = StubHistory([{"role": "user", "content": "old"}])
        _set_budgets(monkeypatch, max_history=0)
        cb = ContextBuilder(history, "sys", _word_counter)
        result = cb.build("hi")
        assert result == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]

    def test_current_text_always_included(self, monkeypatch: pytest.MonkeyPatch) -> None:
        history = StubHistory([{"role": "user", "content": "old"}])
        _set_budgets(monkeypatch, max_history=0)
        cb = ContextBuilder(history, "", _word_counter)
        result = cb.build("a very long current user message that has no budget cap")
        assert result == [{"role": "user", "content": "a very long current user message that has no budget cap"}]

    def test_empty_history(self) -> None:
        history = StubHistory()
        cb = ContextBuilder(history, "sys", _word_counter)
        result = cb.build("hello")
        assert result == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]

    def test_empty_current_text(self) -> None:
        history = StubHistory()
        cb = ContextBuilder(history, "", _word_counter)
        result = cb.build("")
        assert result == [{"role": "user", "content": ""}]

    def test_single_turn_exactly_fills_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        history = StubHistory(
            [
                {"role": "user", "content": "two words"},  # 2 tokens
            ]
        )
        _set_budgets(monkeypatch, max_history=_turn_cost(2))
        cb = ContextBuilder(history, "", _word_counter)
        result = cb.build("hi")
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "two words"}
        assert result[1] == {"role": "user", "content": "hi"}

    def test_tool_call_turn_atomic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        history = StubHistory([{"role": "user", "content": "weather"}])  # 1 token
        history.inject_turn(
            [
                {"type": "function_call", "call_id": "fc1", "name": "w", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "fc1", "output": "sunny"},
                {"role": "assistant", "content": "It is sunny"},
            ],
            token_count=15,
        )
        # History budget covers the user turn + 3-item tool turn
        _set_budgets(monkeypatch, max_history=_turn_cost(1) + 15 + 3 * _MO)
        cb = ContextBuilder(history, "", _word_counter)
        result = cb.build("next")
        assert len(result) == 5  # user + 3 tool items + current

    def test_tool_call_turn_excluded_when_too_large(self, monkeypatch: pytest.MonkeyPatch) -> None:
        history = StubHistory([{"role": "user", "content": "q"}])
        history.inject_turn(
            [
                {"type": "function_call", "call_id": "fc1", "name": "w", "arguments": "{}"},
                {"role": "assistant", "content": "answer"},
            ],
            token_count=50,
        )
        # Tool turn (50 + 2*_MO) exceeds the budget; nothing older fits either
        _set_budgets(monkeypatch, max_history=10)
        cb = ContextBuilder(history, "", _word_counter)
        result = cb.build("hi")
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "hi"}


# ---------------------------------------------------------------------------
# Rolling in-session summary view
# ---------------------------------------------------------------------------


def _make_snapshot(
    text: str = "[Earlier in this conversation]\nThey talked.", through: int = 1
) -> HistorySummarySnapshot:
    return HistorySummarySnapshot(
        block_text=text,
        token_count=_word_counter(text),
        through_turn_id=through,
    )


class TestContextBuilderWithSummary:
    def test_summary_block_replaces_covered_turns(self) -> None:
        history = StubHistory(
            [
                {"role": "user", "content": "old question"},  # turn 0
                {"role": "assistant", "content": "old answer"},  # turn 1
                {"role": "user", "content": "new question"},  # turn 2
                {"role": "assistant", "content": "new answer"},  # turn 3
            ]
        )
        snapshot = _make_snapshot(through=1)
        cb = ContextBuilder(history, "", _word_counter, summarizer=StubSummarizer(snapshot))
        result = cb.build("now")

        contents = [m.get("content") for m in result]
        assert snapshot.block_text in contents
        assert "old question" not in contents
        assert "old answer" not in contents
        assert "new question" in contents
        assert "new answer" in contents
        # Summary block sits right before the live turns
        assert contents.index(snapshot.block_text) < contents.index("new question")

    def test_summary_cost_counts_against_history_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        history = StubHistory(
            [
                {"role": "user", "content": "old"},  # turn 0 (covered)
                {"role": "user", "content": "live one"},  # turn 1, 2 tokens
                {"role": "assistant", "content": "live two ok"},  # turn 2, 3 tokens
            ]
        )
        snapshot = _make_snapshot(text="five word summary block here", through=0)  # 5 tokens
        # Budget: summary (5+_MO) + most recent turn (3+_MO) — turn 1 must drop
        _set_budgets(monkeypatch, max_history=(5 + _MO) + _turn_cost(3))
        cb = ContextBuilder(history, "", _word_counter, summarizer=StubSummarizer(snapshot))
        result = cb.build("now")

        contents = [m.get("content") for m in result]
        assert "five word summary block here" in contents
        assert "live two ok" in contents
        assert "live one" not in contents

    def test_summary_placed_after_prev_sessions(self) -> None:
        history = StubHistory(
            [
                {"role": "user", "content": "old"},  # turn 0 (covered)
                {"role": "user", "content": "live"},  # turn 1
            ]
        )
        snapshot = _make_snapshot(text="summary text", through=0)
        cb = ContextBuilder(
            history,
            "sys",
            _word_counter,
            session_summaries=["[2026-03-28 14:00 session]\n- Prev ep."],
            summarizer=StubSummarizer(snapshot),
        )
        result = cb.build("now")

        contents = [m.get("content") for m in result]
        prev_idx = next(i for i, c in enumerate(contents) if "2026-03-28" in c)
        assert contents.index("summary text") == prev_idx + 1
        assert contents.index("summary text") < contents.index("live")

    def test_maybe_schedule_called_with_all_turns(self) -> None:
        history = StubHistory(
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
            ]
        )
        summarizer = StubSummarizer()
        cb = ContextBuilder(history, "", _word_counter, summarizer=summarizer)
        cb.build("now")

        assert len(summarizer.scheduled) == 1
        assert [t.turn_id for t in summarizer.scheduled[0]] == [0, 1]

    def test_no_snapshot_sends_all_turns_verbatim(self) -> None:
        history = StubHistory(
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
            ]
        )
        cb = ContextBuilder(history, "", _word_counter, summarizer=StubSummarizer(None))
        result = cb.build("now")
        contents = [m.get("content") for m in result]
        assert contents == ["one", "two", "now"]


# ---------------------------------------------------------------------------
# Memory-aware tests
# ---------------------------------------------------------------------------


def _make_profile(topic: str, sub: str, content: str) -> Profile:
    return Profile(id=1, topic=topic, sub_topic=sub, content=content, updated_at="2026-03-15")


def _make_episode(text: str, ts: str = "2026-03-15 14:00:00", eid: int = 1) -> Episode:
    return Episode(
        id=eid,
        text=text,
        timestamp=ts,
        session_id="s1",
        importance=1.0,
        last_cited_at=ts,
    )


class TestContextBuilderWithMemory:
    """Tests for profile, memory, and session summary injection."""

    def test_profile_injected_as_developer_msg(self) -> None:
        profiles = [_make_profile("basic_info", "name", "Alice")]
        history = StubHistory()
        cb = ContextBuilder(history, "sys", _word_counter, profiles=profiles)
        result = cb.build("hi")
        # system, profile(developer), user
        assert len(result) == 3
        assert result[1]["role"] == "developer"
        assert "basic_info::name: Alice" in result[1]["content"]

    def test_memory_injected_last(self) -> None:
        history = StubHistory()
        history.add_user_message("prev")
        history.add_assistant_message("reply")
        cb = ContextBuilder(history, "", _word_counter)

        ep = _make_episode("User likes SF.")
        mem = MemoryReadResult([ep], [0.9], {1: 1})
        result = cb.build("now", memory_result=mem)
        # history(2) + current user + memory(last)
        assert result[-1]["role"] == "developer"
        assert "[M1]" in result[-1]["content"]
        assert result[-2] == {"role": "user", "content": "now"}

    def test_session_summaries_injected(self) -> None:
        history = StubHistory()
        summary = "[2026-03-28 14:00 session]\n- User talked about Dune."
        cb = ContextBuilder(
            history,
            "sys",
            _word_counter,
            session_summaries=[summary],
        )
        result = cb.build("hi")
        # system, summary(developer), user
        assert any("2026-03-28 14:00 session" in m.get("content", "") for m in result)

    def test_block_ordering(self) -> None:
        """Verify: system → profile → summary → history → user → memory."""
        profiles = [_make_profile("basic_info", "name", "Alice")]
        summary = "[2026-03-28 14:00 session]\n- Summary ep."
        history = StubHistory()
        history.add_user_message("old")
        cb = ContextBuilder(
            history,
            "sys",
            _word_counter,
            profiles=profiles,
            session_summaries=[summary],
        )

        mem_ep = _make_episode("Memory ep.", eid=2)
        mem = MemoryReadResult([mem_ep], [0.8], {1: 2})
        result = cb.build("now", memory_result=mem)

        roles = [m.get("role") for m in result]
        # system, profile, summary, history(user), current(user), memory
        assert roles[0] == "system"
        assert roles[1] == "developer"  # profile
        assert roles[2] == "developer"  # summary
        assert roles[-1] == "developer"  # memory (last)
        assert roles[-2] == "user"  # current text

    def test_memory_trimmed_to_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Episodes over the memory cap are dropped from the tail (lowest salience)."""
        eps = [
            _make_episode("first episode with several words inside", eid=1),
            _make_episode("second episode with several words inside", eid=2),
            _make_episode("third episode with several words inside", eid=3),
        ]
        mem = MemoryReadResult(eps, [0.9, 0.8, 0.7], {1: 1, 2: 2, 3: 3})
        _set_budgets(monkeypatch, max_memory=20)
        cb = ContextBuilder(StubHistory(), "", _word_counter)
        result = cb.build("hi", memory_result=mem)

        memory_msg = result[-1]
        assert memory_msg["role"] == "developer"
        assert "[M1]" in memory_msg["content"]
        assert "[M3]" not in memory_msg["content"]

    def test_memory_included_even_with_large_history(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Memory has its own budget — a full history cannot crowd it out."""
        history = StubHistory()
        for i in range(20):
            history.add_user_message(f"turn {i} user message here")
            history.add_assistant_message(f"turn {i} assistant reply here")

        ep = _make_episode("Important memory episode text here.")
        mem = MemoryReadResult([ep], [0.9], {1: 1})
        _set_budgets(monkeypatch, max_history=30)
        cb = ContextBuilder(history, "", _word_counter)
        result = cb.build("hi", memory_result=mem)

        assert result[-1]["role"] == "developer"
        assert "[M1]" in result[-1]["content"]

    def test_backward_compatible_build_call(self) -> None:
        """build(text) without memory_result still works."""
        history = StubHistory()
        cb = ContextBuilder(history, "sys", _word_counter)
        result = cb.build("hello")
        assert result[0] == {"role": "system", "content": "sys"}
        assert result[-1] == {"role": "user", "content": "hello"}

    def test_profile_exceeding_cap_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Profile larger than max_profile_tokens is not injected."""
        # Create a profile with long content
        profiles = [_make_profile("basic_info", "bio", "a " * 200)]
        history = StubHistory()
        _set_budgets(monkeypatch, max_profile=10)  # very tight cap
        cb = ContextBuilder(history, "", _word_counter, profiles=profiles)
        result = cb.build("hi")
        # No developer message (profile too large)
        assert all(m.get("role") != "developer" for m in result)

    def test_summary_overflow_drops_later(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When summaries exceed budget, later ones are dropped."""
        summaries = [
            "[2026-03-26 10:00 session]\n- Old session ep.",
            "[2026-03-27 10:00 session]\n- Mid session ep.",
            "[2026-03-28 10:00 session]\n- Recent session ep.",
        ]
        # Each summary ≈ 7 words + 3 overhead = 10 tokens. Cap at 15 → only 1 fits.
        _set_budgets(monkeypatch, max_prev=15)
        history = StubHistory()
        cb = ContextBuilder(
            history,
            "",
            _word_counter,
            session_summaries=summaries,
        )
        result = cb.build("hi")
        dev_msgs = [m for m in result if m.get("role") == "developer"]
        assert len(dev_msgs) == 1
        # First summary (oldest) is included
        assert "2026-03-26" in dev_msgs[0]["content"]
