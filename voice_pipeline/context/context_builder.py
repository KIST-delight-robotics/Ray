"""LLM context assembly from conversation history and current input."""

from __future__ import annotations

import logging
from typing import Any

from voice_pipeline.core.config import ConversationHistoryConfig
from voice_pipeline.core.interfaces import IContextBuilder, IConversationHistory
from voice_pipeline.core.types import TokenCounter

logger = logging.getLogger("voice_pipeline.context")

# API framing overhead (empirically measured for OpenAI Responses API):
# - Base: fixed overhead per request (internal structure)
# - Per-message: role markers, separators per message
_BASE_OVERHEAD_TOKENS = 5
_PER_MESSAGE_OVERHEAD_TOKENS = 3


class ContextBuilder(IContextBuilder):
    """Assembles LLM context with token budget management.

    Uses turn-level atomic budgeting: each HistoryTurn is included
    or excluded as a whole. Pre-computed token_count is used directly
    — no re-tokenization at build time.

    Budget accounts for:
    - Tool definition tokens (from tools_token_cost)
    - API framing overhead (base + per-message)
    - System prompt + current user message
    - History turns (reverse chronological, atomic)
    """

    def __init__(
        self,
        history: IConversationHistory,
        config: ConversationHistoryConfig,
        system_prompt: str,
        token_counter: TokenCounter,
        tools_token_cost: int = 0,
    ) -> None:
        self._history = history
        self._config = config
        self._system_prompt = system_prompt
        self._token_counter = token_counter
        self._tools_token_cost = tools_token_cost

    def build(self, current_text: str) -> list[dict[str, Any]]:
        """Build the message list for an LLM call.

        Token budget allocation:
        1. Reserve fixed overhead (base + tool definitions).
        2. Reserve system prompt + current user message (with per-message overhead).
        3. Fill remaining budget with history turns, most recent first.
           Each turn is atomic — never split. Per-message overhead
           is added for each message in the turn.
        """
        budget = self._config.max_context_tokens

        # Fixed overhead: API framing + tool definitions
        budget -= _BASE_OVERHEAD_TOKENS + self._tools_token_cost

        result: list[dict[str, Any]] = []

        # Reserve system prompt (with per-message overhead)
        if self._system_prompt:
            system_msg: dict[str, Any] = {"role": "system", "content": self._system_prompt}
            budget -= self._token_counter(self._system_prompt) + _PER_MESSAGE_OVERHEAD_TOKENS
            result.append(system_msg)

        # Reserve current user message (with per-message overhead)
        user_msg: dict[str, Any] = {"role": "user", "content": current_text}
        budget -= self._token_counter(current_text) + _PER_MESSAGE_OVERHEAD_TOKENS

        # Fill history turns in reverse chronological order (atomic)
        turns = self._history.get_turns()
        selected: list[list[dict[str, Any]]] = []
        for turn in reversed(turns):
            # Turn cost = stored token_count + per-message overhead for each item
            turn_cost = turn.token_count + len(turn.items) * _PER_MESSAGE_OVERHEAD_TOKENS
            if turn_cost > budget:
                break
            selected.append(list(turn.items))
            budget -= turn_cost

        # Flatten in chronological order
        selected.reverse()
        for turn_items in selected:
            result.extend(turn_items)

        result.append(user_msg)
        return result
