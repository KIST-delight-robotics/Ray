"""Tests for voice_pipeline.context.context_builder."""

from __future__ import annotations

from typing import Any

from voice_pipeline.context.context_builder import ContextBuilder
from voice_pipeline.core.config import ConversationHistoryConfig
from voice_pipeline.core.interfaces import IConversationHistory, IStorageBackend


class StubStorageBackend(IStorageBackend):
    def load(self, session_id: str) -> list[dict[str, Any]]:
        return []

    def save(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        pass

    def delete(self, session_id: str) -> None:
        pass


class StubHistory(IConversationHistory):
    """Minimal history stub for testing ContextBuilder."""

    def __init__(self, messages: list[dict[str, Any]] | None = None) -> None:
        self._messages = messages or []

    def new_session(self, session_id: str) -> None:
        pass

    def add_user_message(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str) -> None:
        self._messages.append({"role": "assistant", "content": text})

    def get_messages(self) -> list[dict[str, Any]]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def save(self) -> None:
        pass


def _word_counter(text: str) -> int:
    """Simple token counter: count words."""
    return len(text.split()) if text.strip() else 0


class TestContextBuilder:
    def test_basic_with_system_prompt(self) -> None:
        history = StubHistory()
        config = ConversationHistoryConfig(max_context_tokens=100)
        cb = ContextBuilder(history, config, "You are helpful.", _word_counter)
        result = cb.build("hello")
        assert result == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
        ]

    def test_no_system_prompt(self) -> None:
        history = StubHistory()
        config = ConversationHistoryConfig(max_context_tokens=100)
        cb = ContextBuilder(history, config, "", _word_counter)
        result = cb.build("hello")
        assert result == [{"role": "user", "content": "hello"}]

    def test_includes_history(self) -> None:
        history = StubHistory(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello there"},
            ]
        )
        config = ConversationHistoryConfig(max_context_tokens=100)
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
        # Budget: 4 tokens total
        # current_text "now" = 1 token → remaining = 3
        # reversed: "recent reply"(2) → 1, "recent"(1) → 0
        # "old reply here"(3) exceeds → stop
        config = ConversationHistoryConfig(max_context_tokens=4)
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
        # Budget: 5, system "be nice" = 2, current "hi" = 1 → remaining = 2
        # history "hello" = 1 → fits
        config = ConversationHistoryConfig(max_context_tokens=5)
        cb = ContextBuilder(history, config, "be nice", _word_counter)
        result = cb.build("hi")
        assert len(result) == 3
        assert result[0] == {"role": "system", "content": "be nice"}
        assert result[1] == {"role": "user", "content": "hello"}
        assert result[2] == {"role": "user", "content": "hi"}

    def test_empty_history(self) -> None:
        history = StubHistory()
        config = ConversationHistoryConfig(max_context_tokens=100)
        cb = ContextBuilder(history, config, "sys", _word_counter)
        result = cb.build("hello")
        assert result == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]

    def test_current_text_always_included(self) -> None:
        """Current user text must always be in the result, even with tight budget."""
        history = StubHistory(
            [
                {"role": "user", "content": "very long old message"},
            ]
        )
        # Budget: 2 tokens, current "hi" = 1 → remaining = 1
        # history message = 5 tokens → won't fit
        config = ConversationHistoryConfig(max_context_tokens=2)
        cb = ContextBuilder(history, config, "", _word_counter)
        result = cb.build("hi")
        assert result == [{"role": "user", "content": "hi"}]

    def test_system_prompt_exceeds_budget(self) -> None:
        """When system prompt alone exceeds budget, history is dropped but both
        system prompt and current text are still included."""
        history = StubHistory([{"role": "user", "content": "old"}])
        # Budget: 2, system "a very long system prompt" = 5 → budget goes negative
        config = ConversationHistoryConfig(max_context_tokens=2)
        cb = ContextBuilder(history, config, "a very long system prompt", _word_counter)
        result = cb.build("hi")
        assert result[0] == {"role": "system", "content": "a very long system prompt"}
        assert result[-1] == {"role": "user", "content": "hi"}
        # history should be excluded (budget negative)
        assert len(result) == 2

    def test_zero_budget(self) -> None:
        """Zero budget still includes system prompt and current text."""
        history = StubHistory([{"role": "user", "content": "old"}])
        config = ConversationHistoryConfig(max_context_tokens=0)
        cb = ContextBuilder(history, config, "sys", _word_counter)
        result = cb.build("hi")
        assert result[0] == {"role": "system", "content": "sys"}
        assert result[-1] == {"role": "user", "content": "hi"}
        assert len(result) == 2

    def test_empty_current_text(self) -> None:
        """Empty current text produces a user message with empty content."""
        history = StubHistory()
        config = ConversationHistoryConfig(max_context_tokens=100)
        cb = ContextBuilder(history, config, "", _word_counter)
        result = cb.build("")
        assert result == [{"role": "user", "content": ""}]

    def test_single_message_exactly_fills_budget(self) -> None:
        """A history message that exactly uses the remaining budget is included."""
        history = StubHistory(
            [{"role": "user", "content": "two words"}]  # 2 tokens
        )
        # Budget: 3, current "hi" = 1 → remaining = 2
        # history "two words" = 2 → exactly fits
        config = ConversationHistoryConfig(max_context_tokens=3)
        cb = ContextBuilder(history, config, "", _word_counter)
        result = cb.build("hi")
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "two words"}
        assert result[1] == {"role": "user", "content": "hi"}
