"""LLM context assembly with 4-block priority-based token budgeting.

Block layout (optimised for prefix caching — stable blocks first):

  1. System prompt  (instructions, static)
  2. Profile        (developer msg, session-level fixed)
  3. Prev sessions  (developer msgs, session-level fixed)
  4. History turns   (user/assistant, grows within session)
  5. Current user    (user msg, varies per call)
  6. Memory          (developer msg, varies per call — placed last)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from voice_pipeline.context.formatters import (
    format_memory_block,
    format_profile_block,
)
from voice_pipeline.core.interfaces import IContextBuilder, IConversationHistory
from voice_pipeline.core.types import TokenCounter

if TYPE_CHECKING:
    from voice_pipeline.memory.types import MemoryReadResult, Profile

logger = logging.getLogger("voice_pipeline.context")

# API framing overhead (empirically measured for OpenAI Responses API):
# - Base: fixed overhead per request (internal structure)
# - Per-message: role markers, separators per message
_BASE_OVERHEAD_TOKENS = 5
_PER_MESSAGE_OVERHEAD_TOKENS = 3


class ContextBuilder(IContextBuilder):
    """Assembles LLM context with 4-block priority-based token budgeting.

    Session-level data (profiles, previous session summaries) is injected
    at construction and pre-formatted.  Per-turn memory results are passed
    to ``build()``.

    Token budget allocation order:
      1. Base overhead + tool definitions
      2. System prompt  (Block 1)
      3. Profile block  (Block 2, capped at ``_MAX_PROFILE_TOKENS``)
      4. Prev sessions  (Block 3 front, capped at ``_MAX_PREV_SESSION_TOKENS``)
      5. Memory block   (Block 4, **dedicated** ``_MAX_MEMORY_TOKENS``)
      6. Current user message
      7. Current session history  (remainder, reverse-chronological atomic)
    """

    _MAX_CONTEXT_TOKENS = 4096  # LLM 입력 전체 토큰 예산 (모든 블록 합산 상한)
    _MAX_MEMORY_TOKENS = 512  # retrieved memory 블록 전용 예산 (초과 시 낮은 salience 순 drop)
    _MAX_PROFILE_TOKENS = 256  # profile 블록 전용 예산 (초과 시 블록 skip)
    _MAX_PREV_SESSION_TOKENS = 512  # previous session summary 블록 전용 예산 (초과 시 오래된 순 drop)

    def __init__(
        self,
        history: IConversationHistory,
        system_prompt: str,
        token_counter: TokenCounter,
        tools_token_cost: int = 0,
        profiles: list[Profile] | None = None,
        session_summaries: list[str] | None = None,
    ) -> None:
        self._history = history
        self._system_prompt = system_prompt
        self._token_counter = token_counter
        self._tools_token_cost = tools_token_cost

        # Pre-format and pre-count session-level blocks (immutable)
        self._profile_text = format_profile_block(profiles or [])
        self._profile_tokens = (
            self._token_counter(self._profile_text) + _PER_MESSAGE_OVERHEAD_TOKENS if self._profile_text else 0
        )

        self._summary_msgs: list[tuple[str, int]] = []  # (text, token_cost)
        if session_summaries:
            for text in session_summaries:
                cost = self._token_counter(text) + _PER_MESSAGE_OVERHEAD_TOKENS
                self._summary_msgs.append((text, cost))

    def build(
        self,
        current_text: str,
        memory_result: MemoryReadResult | None = None,
    ) -> list[dict[str, Any]]:
        """Build the message list for an LLM call.

        Token budget allocation:
          1. Reserve fixed overhead (base + tool definitions).
          2. Reserve system prompt.
          3. Reserve profile block (capped).
          4. Reserve previous session summaries (capped, oldest dropped first).
          5. Reserve memory block (dedicated budget, capped).
          6. Reserve current user message.
          7. Fill remaining budget with history turns (most recent first).
        """
        budget = self._MAX_CONTEXT_TOKENS

        # 1. Fixed overhead
        budget -= _BASE_OVERHEAD_TOKENS + self._tools_token_cost

        messages: list[dict[str, Any]] = []

        # 2. System prompt (Block 1)
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
            budget -= self._token_counter(self._system_prompt) + _PER_MESSAGE_OVERHEAD_TOKENS

        # 3. Profile (Block 2) — capped at _MAX_PROFILE_TOKENS
        if self._profile_text and self._profile_tokens <= self._MAX_PROFILE_TOKENS and self._profile_tokens <= budget:
            messages.append({"role": "developer", "content": self._profile_text})
            budget -= self._profile_tokens

        # 4. Previous session summaries (Block 3 front) — capped total
        summary_budget = min(self._MAX_PREV_SESSION_TOKENS, budget)
        summary_spent = 0
        summary_msgs_to_add: list[dict[str, Any]] = []
        for text, cost in self._summary_msgs:
            if summary_spent + cost > summary_budget:
                break
            summary_msgs_to_add.append({"role": "developer", "content": text})
            summary_spent += cost
        messages.extend(summary_msgs_to_add)
        budget -= summary_spent

        # 5. Memory (Block 4) — dedicated budget, truncate episodes to fit
        memory_text = ""
        memory_cost = 0
        if memory_result and memory_result.episodes:
            memory_text = format_memory_block(memory_result)
            memory_cost = self._token_counter(memory_text) + _PER_MESSAGE_OVERHEAD_TOKENS
            # If over cap, drop lowest-salience episodes (from tail) until it fits
            max_mem = self._MAX_MEMORY_TOKENS
            if memory_cost > max_mem:
                from voice_pipeline.memory.types import MemoryReadResult

                eps = list(memory_result.episodes)
                scores = list(memory_result.scores)
                idx_map = dict(memory_result.index_to_id)
                while eps and memory_cost > max_mem:
                    eps.pop()
                    scores.pop()
                    idx_map.pop(len(eps) + 1, None)
                    trimmed = MemoryReadResult(eps, scores, idx_map)
                    memory_text = format_memory_block(trimmed)
                    memory_cost = self._token_counter(memory_text) + _PER_MESSAGE_OVERHEAD_TOKENS if memory_text else 0
                if not memory_text:
                    memory_cost = 0
        budget -= memory_cost  # reserve even before history fill

        # 6. Current user message
        user_msg: dict[str, Any] = {"role": "user", "content": current_text}
        budget -= self._token_counter(current_text) + _PER_MESSAGE_OVERHEAD_TOKENS

        # 7. History turns (fill remainder, reverse chronological, atomic)
        turns = self._history.get_turns()
        selected: list[list[dict[str, Any]]] = []
        for turn in reversed(turns):
            turn_cost = turn.token_count + len(turn.items) * _PER_MESSAGE_OVERHEAD_TOKENS
            if turn_cost > budget:
                break
            selected.append(list(turn.items))
            budget -= turn_cost

        # Flatten history in chronological order
        selected.reverse()
        for turn_items in selected:
            messages.extend(turn_items)

        # Current user message
        messages.append(user_msg)

        # Memory block last (for prefix caching — varies per call)
        if memory_text:
            messages.append({"role": "developer", "content": memory_text})

        return messages
