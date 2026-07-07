"""LLM context assembly with fixed per-block token budgets.

Block layout (optimised for prefix caching — stable blocks first):

  1. System prompt   (instructions, static)
  2. Profile         (developer msg, session-level fixed)
  3. Prev sessions   (developer msgs, session-level fixed)
  4. History summary (developer msg, swaps only when the rolling summary updates)
  5. History turns   (user/assistant, grows within session)
  6. Current user    (user msg, varies per call)
  7. Memory          (developer msg, varies per call — placed last)

Each block has its own independent budget; there is no shared global budget.
History (summary block + live turns) gets a fixed ``_MAX_HISTORY_TOKENS``.
When usage approaches it, the injected IHistorySummarizer folds older turns
into a rolling summary in the background; until the swap lands, overflow is
handled by dropping the oldest live turns from the view (fallback).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from voice_pipeline.context.formatters import (
    format_memory_block,
    format_profile_block,
    format_raw_transcript_block,
    format_session_summary_block,
)
from voice_pipeline.core.interfaces import IConversationHistory, IHistorySummarizer, IMemoryStorage
from voice_pipeline.core.types import TokenCounter

if TYPE_CHECKING:
    from voice_pipeline.memory.types import MemoryReadResult, Profile

logger = logging.getLogger("voice_pipeline.context")

# Per-message API framing overhead (role markers, separators — empirically
# measured for the OpenAI Responses API).
_PER_MESSAGE_OVERHEAD_TOKENS = 3


class ContextBuilder:
    """Assembles LLM context with fixed per-block token budgets.

    Session-level data (profiles, previous session summaries) is injected
    at construction and pre-formatted.  Per-turn memory results are passed
    to ``build()``.

    Per-block budgets (independent — no shared global budget):
      - System prompt / current user message: always included, uncapped.
      - Profile: capped at ``_MAX_PROFILE_TOKENS`` (skip if over).
      - Prev sessions: capped at ``_MAX_PREV_SESSION_TOKENS`` (oldest kept first).
      - Memory: capped at ``_MAX_MEMORY_TOKENS`` (lowest-salience dropped).
      - History (summary block + live turns): fixed ``_MAX_HISTORY_TOKENS``,
        reverse-chronological atomic fill, oldest dropped on overflow.
    """

    _MAX_HISTORY_TOKENS = 8192  # 현재 세션 히스토리 전용 고정 예산 (요약 블록 포함)
    _MAX_MEMORY_TOKENS = 512  # retrieved memory 블록 전용 예산 (초과 시 낮은 salience 순 drop)
    _MAX_PROFILE_TOKENS = 256  # profile 블록 전용 예산 (초과 시 블록 skip)
    _MAX_PREV_SESSION_TOKENS = 512  # previous session summary 블록 전용 예산 (초과 시 오래된 순 drop)
    _PREVIOUS_SESSION_COUNT = 3  # 이전 세션 요약 최대 로딩 건수

    def __init__(
        self,
        history: IConversationHistory,
        system_prompt: str,
        token_counter: TokenCounter,
        profiles: list[Profile] | None = None,
        session_summaries: list[str] | None = None,
        *,
        memory_storage: IMemoryStorage | None = None,
        session_id: str | None = None,
        summarizer: IHistorySummarizer | None = None,
    ) -> None:
        self._history = history
        self._system_prompt = system_prompt
        self._token_counter = token_counter
        self._summarizer = summarizer

        # Load session context if storage is provided
        self.exclude_session_ids: set[str] = set()
        if memory_storage is not None and session_id is not None:
            profiles, session_summaries, self.exclude_session_ids = self._load_session_context(
                memory_storage, session_id
            )

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

        Assembly order (each block against its own budget):
          1. System prompt (always).
          2. Profile block (capped).
          3. Previous session summaries (capped, oldest kept first).
          4. History: rolling summary block + live turns within
             ``_MAX_HISTORY_TOKENS`` (most recent first, oldest dropped).
          5. Current user message (always).
          6. Memory block (capped, placed last for prefix caching).

        Also schedules a background history summarization when usage
        crosses the summarizer's trigger threshold.
        """
        messages: list[dict[str, Any]] = []

        # 1. System prompt (Block 1)
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        # 2. Profile (Block 2) — capped at _MAX_PROFILE_TOKENS
        if self._profile_text and self._profile_tokens <= self._MAX_PROFILE_TOKENS:
            messages.append({"role": "developer", "content": self._profile_text})

        # 3. Previous session summaries (Block 3) — capped total
        summary_spent = 0
        for text, cost in self._summary_msgs:
            if summary_spent + cost > self._MAX_PREV_SESSION_TOKENS:
                break
            messages.append({"role": "developer", "content": text})
            summary_spent += cost

        # 4. History — fixed budget shared by the summary block and live turns
        snapshot = self._summarizer.snapshot() if self._summarizer is not None else None
        turns = self._history.get_turns()
        history_budget = self._MAX_HISTORY_TOKENS
        watermark = -1
        if snapshot is not None:
            watermark = snapshot.through_turn_id
            history_budget -= snapshot.token_count + _PER_MESSAGE_OVERHEAD_TOKENS
            messages.append({"role": "developer", "content": snapshot.block_text})

        live_turns = [t for t in turns if t.turn_id > watermark]
        selected: list[list[dict[str, Any]]] = []
        for turn in reversed(live_turns):
            turn_cost = turn.token_count + len(turn.items) * _PER_MESSAGE_OVERHEAD_TOKENS
            if turn_cost > history_budget:
                break
            selected.append(list(turn.items))
            history_budget -= turn_cost

        # Flatten history in chronological order
        selected.reverse()
        for turn_items in selected:
            messages.extend(turn_items)

        # 5. Current user message (always included)
        messages.append({"role": "user", "content": current_text})

        # 6. Memory block last (for prefix caching — varies per call)
        memory_text = self._build_memory_text(memory_result)
        if memory_text:
            messages.append({"role": "developer", "content": memory_text})

        if self._summarizer is not None:
            self._summarizer.maybe_schedule(turns)

        return messages

    def _build_memory_text(self, memory_result: MemoryReadResult | None) -> str:
        """Format the memory block, trimming lowest-salience episodes to fit the cap."""
        if not memory_result or not memory_result.episodes:
            return ""
        memory_text = format_memory_block(memory_result)
        memory_cost = self._token_counter(memory_text) + _PER_MESSAGE_OVERHEAD_TOKENS
        if memory_cost <= self._MAX_MEMORY_TOKENS:
            return memory_text

        from voice_pipeline.memory.types import MemoryReadResult as _MemoryReadResult

        eps = list(memory_result.episodes)
        scores = list(memory_result.scores)
        idx_map = dict(memory_result.index_to_id)
        while eps and memory_cost > self._MAX_MEMORY_TOKENS:
            eps.pop()
            scores.pop()
            idx_map.pop(len(eps) + 1, None)
            memory_text = format_memory_block(_MemoryReadResult(eps, scores, idx_map))
            memory_cost = self._token_counter(memory_text) + _PER_MESSAGE_OVERHEAD_TOKENS if memory_text else 0
        return memory_text

    def _load_session_context(
        self,
        memory_storage: IMemoryStorage,
        session_id: str,
    ) -> tuple[list[Profile], list[str], set[str]]:
        profiles = memory_storage.get_all_profiles()
        recent = memory_storage.get_recent_sessions(self._PREVIOUS_SESSION_COUNT, exclude_session_id=session_id)
        recent_session_ids = [s[0] for s in recent]
        session_episodes = memory_storage.get_episodes_by_session_ids(recent_session_ids)
        processed_ids = memory_storage.get_processed_session_ids(recent_session_ids)

        session_summaries: list[str] = []
        for sid, started_at in recent:
            episodes = session_episodes.get(sid, [])
            if episodes:
                session_summaries.append(format_session_summary_block(started_at, episodes))
            elif sid in processed_ids:
                continue
            else:
                utterances = memory_storage.get_utterances(sid)
                if utterances:
                    session_summaries.append(format_raw_transcript_block(started_at, utterances))

        exclude_session_ids = {session_id} | set(recent_session_ids)
        return profiles, session_summaries, exclude_session_ids
