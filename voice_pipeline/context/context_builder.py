"""LLM context assembly from conversation history and current input."""

from __future__ import annotations

import logging
from typing import Any

from voice_pipeline.core.config import ConversationHistoryConfig
from voice_pipeline.core.interfaces import IContextBuilder, IConversationHistory
from voice_pipeline.core.types import TokenCounter

logger = logging.getLogger("voice_pipeline.context")


class ContextBuilder(IContextBuilder):
    """Assembles LLM context with token budget management.

    Fills context by reserving tokens for the system prompt and current
    user message first, then adding history messages in reverse
    chronological order until the budget is exhausted.

    The ``system_prompt`` parameter is a plain string for now.
    In Phase 3, it will be sourced from ``llm/prompts.py``.
    """

    def __init__(
        self,
        history: IConversationHistory,
        config: ConversationHistoryConfig,
        system_prompt: str,
        token_counter: TokenCounter,
    ) -> None:
        self._history = history
        self._config = config
        self._system_prompt = system_prompt
        self._token_counter = token_counter

    def build(self, current_text: str) -> list[dict[str, Any]]:
        """Build the message list for an LLM call.

        Token budget allocation:
        1. Reserve tokens for system prompt (if non-empty) + current user message.
        2. Fill remaining budget with history messages, most recent first.
        """
        budget = self._config.max_context_tokens
        result: list[dict[str, Any]] = []

        # Reserve system prompt
        if self._system_prompt:
            system_msg: dict[str, Any] = {"role": "system", "content": self._system_prompt}
            budget -= self._token_counter(self._system_prompt)
            result.append(system_msg)

        # Reserve current user message
        user_msg: dict[str, Any] = {"role": "user", "content": current_text}
        budget -= self._token_counter(current_text)

        # Fill history in reverse chronological order
        all_messages = self._history.get_messages()
        selected: list[dict[str, Any]] = []
        for msg in reversed(all_messages):
            cost = self._token_counter(msg["content"])
            if cost > budget:
                break
            selected.append(msg)
            budget -= cost

        selected.reverse()
        result.extend(selected)
        result.append(user_msg)

        return result
