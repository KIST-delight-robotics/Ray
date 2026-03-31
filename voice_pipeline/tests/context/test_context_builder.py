"""Tests for voice_pipeline.context.context_builder."""

from __future__ import annotations

from typing import Any

from voice_pipeline.context.context_builder import (
    _BASE_OVERHEAD_TOKENS,
    _PER_MESSAGE_OVERHEAD_TOKENS,
    ContextBuilder,
)
from voice_pipeline.core.config import ConversationHistoryConfig
from voice_pipeline.core.types import HistoryTurn, LLMMetrics
from voice_pipeline.memory.types import Episode, MemoryReadResult, Profile

# Per-message overhead shorthand
_MO = _PER_MESSAGE_OVERHEAD_TOKENS
_BO = _BASE_OVERHEAD_TOKENS


class StubHistory:
    """Minimal history stub for testing ContextBuilder."""

    def __init__(self, messages: list[dict[str, Any]] | None = None) -> None:
        self._turns: list[HistoryTurn] = []
        if messages:
            for msg in messages:
                text = msg.get("content", "")
                tc = len(text.split()) if text and text.strip() else 0
                self._turns.append(HistoryTurn(items=(msg,), token_count=tc))

    def new_session(self, session_id: str) -> None:
        pass

    def add_user_message(self, text: str) -> int:
        tc = len(text.split()) if text.strip() else 0
        self._turns.append(HistoryTurn(items=({"role": "user", "content": text},), token_count=tc))
        return len(self._turns) - 1

    def add_assistant_message(self, text: str, metrics: LLMMetrics | None = None) -> int:
        tc = len(text.split()) if text.strip() else 0
        self._turns.append(
            HistoryTurn(items=({"role": "assistant", "content": text},), token_count=tc)
        )
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
        self._turns.append(HistoryTurn(items=tuple(items), token_count=token_count))


def _word_counter(text: str) -> int:
    return len(text.split()) if text.strip() else 0


def _budget(*msg_tokens: int, system: int = 0, tools: int = 0) -> int:
    """Calculate the minimum budget needed for given messages.

    Each value in msg_tokens is the content token count for one message.
    Automatically adds base overhead, per-message overhead, and tool cost.
    """
    n_msgs = len(msg_tokens) + (1 if system else 0)
    return sum(msg_tokens) + system + tools + _BO + n_msgs * _MO


class TestContextBuilder:
    def test_basic_with_system_prompt(self) -> None:
        history = StubHistory()
        config = ConversationHistoryConfig(max_context_tokens=200)
        cb = ContextBuilder(history, config, "You are helpful.", _word_counter)
        result = cb.build("hello")
        assert result == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
        ]

    def test_no_system_prompt(self) -> None:
        history = StubHistory()
        config = ConversationHistoryConfig(max_context_tokens=200)
        cb = ContextBuilder(history, config, "", _word_counter)
        result = cb.build("hello")
        assert result == [{"role": "user", "content": "hello"}]

    def test_includes_history(self) -> None:
        history = StubHistory(
            [
                {"role": "user", "content": "hi"},  # 1 token
                {"role": "assistant", "content": "hello there"},  # 2 tokens
            ]
        )
        config = ConversationHistoryConfig(max_context_tokens=200)
        cb = ContextBuilder(history, config, "", _word_counter)
        result = cb.build("how are you")
        assert len(result) == 3
        assert result[0] == {"role": "user", "content": "hi"}
        assert result[1] == {"role": "assistant", "content": "hello there"}
        assert result[2] == {"role": "user", "content": "how are you"}

    def test_token_budget_trims_oldest_first(self) -> None:
        history = StubHistory(
            [
                {"role": "user", "content": "old message here"},  # 3 tokens
                {"role": "assistant", "content": "old reply here"},  # 3 tokens
                {"role": "user", "content": "recent"},  # 1 token
                {"role": "assistant", "content": "recent reply"},  # 2 tokens
            ]
        )
        # Budget enough for current(1) + recent(1) + recent_reply(2) + overhead
        # but not enough for old messages
        budget = _budget(1) + (1 + _MO) + (2 + _MO)  # current + 2 history msgs
        config = ConversationHistoryConfig(max_context_tokens=budget)
        cb = ContextBuilder(history, config, "", _word_counter)
        result = cb.build("now")
        assert len(result) == 3
        assert result[0] == {"role": "user", "content": "recent"}
        assert result[1] == {"role": "assistant", "content": "recent reply"}
        assert result[2] == {"role": "user", "content": "now"}

    def test_system_prompt_budget(self) -> None:
        history = StubHistory(
            [
                {"role": "user", "content": "hello"},  # 1 token
            ]
        )
        # Budget for system(2) + current(1) + history(1) + overhead
        budget = _budget(1, system=2) + (1 + _MO)
        config = ConversationHistoryConfig(max_context_tokens=budget)
        cb = ContextBuilder(history, config, "be nice", _word_counter)
        result = cb.build("hi")
        assert len(result) == 3
        assert result[0] == {"role": "system", "content": "be nice"}
        assert result[1] == {"role": "user", "content": "hello"}
        assert result[2] == {"role": "user", "content": "hi"}

    def test_empty_history(self) -> None:
        history = StubHistory()
        config = ConversationHistoryConfig(max_context_tokens=200)
        cb = ContextBuilder(history, config, "sys", _word_counter)
        result = cb.build("hello")
        assert result == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]

    def test_current_text_always_included(self) -> None:
        history = StubHistory(
            [
                {"role": "user", "content": "very long old message"},
            ]
        )
        # Tight budget: only room for current message
        budget = _budget(1)
        config = ConversationHistoryConfig(max_context_tokens=budget)
        cb = ContextBuilder(history, config, "", _word_counter)
        result = cb.build("hi")
        assert result == [{"role": "user", "content": "hi"}]

    def test_system_prompt_exceeds_budget(self) -> None:
        history = StubHistory([{"role": "user", "content": "old"}])
        config = ConversationHistoryConfig(max_context_tokens=2)
        cb = ContextBuilder(history, config, "a very long system prompt", _word_counter)
        result = cb.build("hi")
        assert result[0] == {"role": "system", "content": "a very long system prompt"}
        assert result[-1] == {"role": "user", "content": "hi"}
        assert len(result) == 2

    def test_zero_budget(self) -> None:
        history = StubHistory([{"role": "user", "content": "old"}])
        config = ConversationHistoryConfig(max_context_tokens=0)
        cb = ContextBuilder(history, config, "sys", _word_counter)
        result = cb.build("hi")
        assert result[0] == {"role": "system", "content": "sys"}
        assert result[-1] == {"role": "user", "content": "hi"}
        assert len(result) == 2

    def test_empty_current_text(self) -> None:
        history = StubHistory()
        config = ConversationHistoryConfig(max_context_tokens=200)
        cb = ContextBuilder(history, config, "", _word_counter)
        result = cb.build("")
        assert result == [{"role": "user", "content": ""}]

    def test_single_message_exactly_fills_budget(self) -> None:
        history = StubHistory(
            [
                {"role": "user", "content": "two words"},  # 2 tokens
            ]
        )
        # Exact budget for current(1) + history(2) + overhead
        budget = _budget(1) + (2 + _MO)
        config = ConversationHistoryConfig(max_context_tokens=budget)
        cb = ContextBuilder(history, config, "", _word_counter)
        result = cb.build("hi")
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "two words"}
        assert result[1] == {"role": "user", "content": "hi"}

    def test_tool_call_turn_atomic(self) -> None:
        history = StubHistory([{"role": "user", "content": "weather"}])  # 1 token
        history.inject_turn(
            [
                {"type": "function_call", "call_id": "fc1", "name": "w", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "fc1", "output": "sunny"},
                {"role": "assistant", "content": "It is sunny"},
            ],
            token_count=15,
        )
        # Budget enough for all: current + user_turn + tool_turn(3 items)
        budget = _budget(1) + (1 + _MO) + (15 + 3 * _MO)
        config = ConversationHistoryConfig(max_context_tokens=budget)
        cb = ContextBuilder(history, config, "", _word_counter)
        result = cb.build("next")
        assert len(result) == 5  # user + 3 tool items + current

    def test_tool_call_turn_excluded_when_too_large(self) -> None:
        history = StubHistory([{"role": "user", "content": "q"}])
        history.inject_turn(
            [
                {"type": "function_call", "call_id": "fc1", "name": "w", "arguments": "{}"},
                {"role": "assistant", "content": "answer"},
            ],
            token_count=50,
        )
        # Budget only for current message
        budget = _budget(1)
        config = ConversationHistoryConfig(max_context_tokens=budget)
        cb = ContextBuilder(history, config, "", _word_counter)
        result = cb.build("hi")
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "hi"}

    def test_tools_token_cost_deducted(self) -> None:
        """Tool definitions reduce available budget."""
        history = StubHistory(
            [
                {"role": "user", "content": "hello"},  # 1 token
            ]
        )
        # Budget for current + 1 history message + overhead, NO tools
        budget = _budget(1) + (1 + _MO)
        # With tool cost, history message no longer fits
        tool_cost = 100
        config = ConversationHistoryConfig(max_context_tokens=budget)
        cb = ContextBuilder(history, config, "", _word_counter, tools_token_cost=tool_cost)
        result = cb.build("hi")
        # Only current message fits (tool cost ate the history budget)
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "hi"}


# ---------------------------------------------------------------------------
# Memory-aware tests (Phase 4)
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
        config = ConversationHistoryConfig(max_context_tokens=500)
        cb = ContextBuilder(history, config, "sys", _word_counter, profiles=profiles)
        result = cb.build("hi")
        # system, profile(developer), user
        assert len(result) == 3
        assert result[1]["role"] == "developer"
        assert "basic_info::name: Alice" in result[1]["content"]

    def test_memory_injected_last(self) -> None:
        history = StubHistory()
        history.add_user_message("prev")
        history.add_assistant_message("reply")
        config = ConversationHistoryConfig(max_context_tokens=500)
        cb = ContextBuilder(history, config, "", _word_counter)

        ep = _make_episode("User likes SF.")
        mem = MemoryReadResult([ep], [0.9], {1: 1})
        result = cb.build("now", memory_result=mem)
        # history(2) + current user + memory(last)
        assert result[-1]["role"] == "developer"
        assert "[M1]" in result[-1]["content"]
        assert result[-2] == {"role": "user", "content": "now"}

    def test_session_summaries_injected(self) -> None:
        history = StubHistory()
        episodes = [_make_episode("User talked about Dune.")]
        config = ConversationHistoryConfig(max_context_tokens=500)
        cb = ContextBuilder(
            history,
            config,
            "sys",
            _word_counter,
            session_summaries=[("2026-03-28 14:00:00", episodes)],
        )
        result = cb.build("hi")
        # system, summary(developer), user
        assert any("2026-03-28 14:00 session" in m.get("content", "") for m in result)

    def test_block_ordering(self) -> None:
        """Verify: system → profile → summary → history → user → memory."""
        profiles = [_make_profile("basic_info", "name", "Alice")]
        episodes = [_make_episode("Summary ep.")]
        history = StubHistory()
        history.add_user_message("old")
        config = ConversationHistoryConfig(max_context_tokens=1000)
        cb = ContextBuilder(
            history,
            config,
            "sys",
            _word_counter,
            profiles=profiles,
            session_summaries=[("2026-03-28 14:00:00", episodes)],
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

    def test_memory_budget_reserved_before_history(self) -> None:
        """Memory gets its dedicated budget even when history is large."""
        # Fill history with many turns
        history = StubHistory()
        for i in range(20):
            history.add_user_message(f"turn {i} user message here")
            history.add_assistant_message(f"turn {i} assistant reply here")

        ep = _make_episode("Important memory episode text here.")
        mem = MemoryReadResult([ep], [0.9], {1: 1})

        # Tight budget: memory reservation eats into history space
        config = ConversationHistoryConfig(
            max_context_tokens=80,
            max_memory_tokens=20,
        )
        cb = ContextBuilder(history, config, "", _word_counter)
        result = cb.build("hi", memory_result=mem)

        # Memory block must be present (last message)
        assert result[-1]["role"] == "developer"
        assert "[M1]" in result[-1]["content"]

    def test_no_memory_gives_budget_to_history(self) -> None:
        """Without memory, history gets the full remaining budget."""
        history = StubHistory()
        for i in range(5):
            history.add_user_message(f"msg {i}")

        config = ConversationHistoryConfig(max_context_tokens=200, max_memory_tokens=50)
        cb = ContextBuilder(history, config, "", _word_counter)

        result_no_mem = cb.build("now")
        result_with_mem = cb.build("now", memory_result=MemoryReadResult([], [], {}))

        # Both should have same result (empty memory = no reservation)
        assert len(result_no_mem) == len(result_with_mem)

    def test_backward_compatible_build_call(self) -> None:
        """build(text) without memory_result still works."""
        history = StubHistory()
        config = ConversationHistoryConfig(max_context_tokens=200)
        cb = ContextBuilder(history, config, "sys", _word_counter)
        result = cb.build("hello")
        assert result[0] == {"role": "system", "content": "sys"}
        assert result[-1] == {"role": "user", "content": "hello"}

    def test_profile_exceeding_cap_is_skipped(self) -> None:
        """Profile larger than max_profile_tokens is not injected."""
        # Create a profile with long content
        profiles = [_make_profile("basic_info", "bio", "a " * 200)]
        history = StubHistory()
        config = ConversationHistoryConfig(
            max_context_tokens=500,
            max_profile_tokens=10,  # very tight cap
        )
        cb = ContextBuilder(history, config, "", _word_counter, profiles=profiles)
        result = cb.build("hi")
        # No developer message (profile too large)
        assert all(m.get("role") != "developer" for m in result)

    def test_summary_overflow_drops_later(self) -> None:
        """When summaries exceed budget, later ones are dropped."""
        summaries = [
            ("2026-03-26 10:00:00", [_make_episode("Old session ep.")]),
            ("2026-03-27 10:00:00", [_make_episode("Mid session ep.")]),
            ("2026-03-28 10:00:00", [_make_episode("Recent session ep.")]),
        ]
        # Each summary ≈ 7 words + 3 overhead = 10 tokens. Cap at 15 → only 1 fits.
        config = ConversationHistoryConfig(
            max_context_tokens=500,
            max_prev_session_tokens=15,
        )
        history = StubHistory()
        cb = ContextBuilder(
            history,
            config,
            "",
            _word_counter,
            session_summaries=summaries,
        )
        result = cb.build("hi")
        dev_msgs = [m for m in result if m.get("role") == "developer"]
        assert len(dev_msgs) == 1
        # First summary (oldest) is included
        assert "2026-03-26" in dev_msgs[0]["content"]
