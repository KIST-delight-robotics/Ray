"""LLM context assembly with fixed per-block token budgets.

Block layout (optimised for prefix caching — stable blocks first):

  1. System prompt   (instructions, static)
  2. Profile         (developer msg, session-level fixed)
  3. Recent sessions (developer msgs, episodes per session; grows once when
                      the carryover is evicted)
  4. Carryover       (developer header + previous session's raw turns +
                      boundary marker; present until evicted)
  5. History summary (developer msg, swaps only when the rolling summary updates)
  6. History turns   (user/assistant, grows within session)
  7. Current user    (user msg, varies per call)
  8. Memory          (developer msg, varies per call — placed last)

Each block has its own independent budget; there is no shared global budget.
History (carryover + summary block + live turns) gets a fixed
``_MAX_HISTORY_TOKENS``.

The previous session is carried verbatim into block 4 at session start.
When history usage crosses ``_CARRYOVER_EVICT_RATIO`` of the budget, the
carryover is evicted: its episodes join the recent-sessions block (block 3)
and the raw turns disappear — no LLM call involved. Rolling summarization
(block 5) stays paused until then, so previous- and current-session content
never mix into one summary. Overflow beyond the budget is handled by
dropping the oldest turns from the view (carryover turns first).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from voice_pipeline.context.formatters import (
    format_carryover_block,
    format_memory_block,
    format_profile_block,
    format_session_boundary,
    format_session_summary_block,
)
from voice_pipeline.core.interfaces import (
    IConversationHistory,
    IHistorySummarizer,
    IMemoryStorage,
    IStorageBackend,
)
from voice_pipeline.core.types import HistoryTurn, TokenCounter, utc_now_str

if TYPE_CHECKING:
    from voice_pipeline.memory.types import MemoryReadResult, Profile

logger = logging.getLogger("voice_pipeline.context")

# Per-message API framing overhead (role markers, separators — empirically
# measured for the OpenAI Responses API).
_PER_MESSAGE_OVERHEAD_TOKENS = 3


@dataclass(frozen=True)
class _Carryover:
    """Previous session's raw view carried into the current session's context.

    Token counts include per-message framing overhead. ``turn_units`` holds
    (items, cost) per turn in chronological order — turns the persisted
    rolling summary already covers are excluded at load time.
    """

    session_id: str
    started_at: str
    header_text: str  # "[Previous session — …]" (+ persisted rolling summary)
    header_tokens: int
    marker_text: str  # "[New session — …]"
    marker_tokens: int
    turn_units: tuple[tuple[tuple[dict[str, Any], ...], int], ...]
    total_tokens: int


class ContextBuilder:
    """Assembles LLM context with fixed per-block token budgets.

    Session-level data (profiles, recent-session episodes, previous-session
    carryover) is loaded at construction.  Per-turn memory results are passed
    to ``build()``.

    Per-block budgets (independent — no shared global budget):
      - System prompt / current user message: always included, uncapped.
      - Profile: capped at ``_MAX_PROFILE_TOKENS`` (skip if over).
      - Recent sessions: soft-capped at ``_MAX_RECENT_SESSIONS_TOKENS`` —
        whole sessions newest-first, the newest always included.
      - Memory: capped at ``_MAX_MEMORY_TOKENS`` (lowest-salience dropped).
      - History (carryover + summary block + live turns): fixed
        ``_MAX_HISTORY_TOKENS``, reverse-chronological atomic fill,
        oldest dropped on overflow.

    ``exclude_session_ids`` holds the sessions already represented in the
    context view (current, carryover, recent block) — the retriever filters
    them out so their episodes are never injected twice. The set is fixed
    for the session lifetime: eviction only moves the carryover session
    from block 4 to block 3.
    """

    _MAX_HISTORY_TOKENS = 8192  # 히스토리 뷰 전용 고정 예산 (이월 + 요약 블록 + 라이브 턴)
    _MAX_MEMORY_TOKENS = 512  # retrieved memory 블록 전용 예산 (초과 시 낮은 salience 순 drop)
    _MAX_PROFILE_TOKENS = 256  # profile 블록 전용 예산 (초과 시 블록 skip)
    _MAX_RECENT_SESSIONS_TOKENS = 512  # 최근 세션 블록 soft cap — 최신 세션 1개는 캡 무관 보장
    _RECENT_SESSION_CANDIDATES = 10  # 최근 세션 후보 조회 수 (실제 포함 수는 캡이 결정)
    _CARRYOVER_EVICT_RATIO = 0.75  # 히스토리 수요가 예산 대비 이 비율을 넘으면 이월분 퇴거

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
        history_backend: IStorageBackend | None = None,
    ) -> None:
        """Initialize the builder and load session-level context.

        Args:
            history: 현재 세션 대화 이력.
            system_prompt: LLM 시스템 프롬프트.
            token_counter: 토큰 카운터 콜러블.
            profiles: 직접 주입할 프로필 (memory_storage 미사용 시).
            session_summaries: 직접 주입할 최근 세션 블록 텍스트, 시간순
                (오래된 것 먼저). soft cap이 동일하게 적용된다.
            memory_storage: 프로필·최근 세션 로딩 및 이월 퇴거용 스토리지.
            session_id: 현재 세션 ID.
            summarizer: 세션 내 히스토리 롤링 요약기.
            history_backend: 직전 세션 이월(carryover) 로딩용 히스토리 백엔드.
                ``None``이면 이월 없음.
        """
        self._history = history
        self._system_prompt = system_prompt
        self._token_counter = token_counter
        self._summarizer = summarizer
        self._memory_storage = memory_storage
        self._evict_trigger_tokens = int(self._MAX_HISTORY_TOKENS * self._CARRYOVER_EVICT_RATIO)
        # Guards carryover + recent-block state: eviction swaps both, and
        # build() may run from concurrent generation threads.
        self._state_lock = threading.Lock()

        # Block 4: previous-session carryover — loaded before the recent
        # sessions block so the carried session can be excluded from it.
        self._carryover: _Carryover | None = None
        if history_backend is not None and session_id is not None:
            self._carryover = self._load_carryover(history_backend, session_id)

        # Block 3: recent sessions (episode summaries, chronological)
        self._recent_block_texts: list[str] = []
        self.exclude_session_ids: set[str] = set()
        if memory_storage is not None and session_id is not None:
            profiles, self._recent_block_texts, included_ids = self._load_session_context(memory_storage, session_id)
            self.exclude_session_ids = {session_id} | included_ids
            if self._carryover is not None:
                self.exclude_session_ids.add(self._carryover.session_id)
        elif session_summaries:
            selected = self._select_recent_blocks([(None, text) for text in reversed(session_summaries)])
            self._recent_block_texts = [text for _, text in selected]

        # Pre-format and pre-count session-level blocks (immutable)
        self._profile_text = format_profile_block(profiles or [])
        self._profile_tokens = (
            self._token_counter(self._profile_text) + _PER_MESSAGE_OVERHEAD_TOKENS if self._profile_text else 0
        )

    def build(
        self,
        current_text: str,
        memory_result: MemoryReadResult | None = None,
    ) -> list[dict[str, Any]]:
        """Build the message list for an LLM call.

        Assembly order (each block against its own budget):
          1. System prompt (always).
          2. Profile block (capped).
          3. Recent sessions block (soft-capped at load time).
          4. Carryover: previous-session header + raw turns + boundary marker.
          5. Rolling summary block + live turns within ``_MAX_HISTORY_TOKENS``
             (most recent first, oldest dropped).
          6. Current user message (always).
          7. Memory block (capped, placed last for prefix caching).

        Past the eviction trigger the carryover is demoted to episodes in
        block 3 before assembly. Rolling summarization is scheduled only
        once no carryover remains.
        """
        messages: list[dict[str, Any]] = []

        # 1. System prompt (Block 1)
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        # 2. Profile (Block 2) — capped at _MAX_PROFILE_TOKENS
        if self._profile_text and self._profile_tokens <= self._MAX_PROFILE_TOKENS:
            messages.append({"role": "developer", "content": self._profile_text})

        turns = self._history.get_turns()

        # Eviction decision precedes assembly (no LLM call involved).
        with self._state_lock:
            if self._carryover is not None and self._history_demand(turns) >= self._evict_trigger_tokens:
                self._evict_carryover()
            carryover = self._carryover
            recent_texts = list(self._recent_block_texts)

        # 3. Recent sessions (Block 3) — chronological, soft-capped at load
        for text in recent_texts:
            messages.append({"role": "developer", "content": text})

        # 4-6. History — fixed budget shared by carryover, summary block,
        # and live turns. Live turns fill first (newest priority), the
        # carryover renders into whatever budget remains.
        snapshot = self._summarizer.snapshot() if self._summarizer is not None else None
        history_budget = self._MAX_HISTORY_TOKENS
        watermark = -1
        snapshot_msg: dict[str, Any] | None = None
        if snapshot is not None:
            watermark = snapshot.through_turn_id
            history_budget -= snapshot.token_count + _PER_MESSAGE_OVERHEAD_TOKENS
            snapshot_msg = {"role": "developer", "content": snapshot.block_text}

        live_turns = [t for t in turns if t.turn_id > watermark]
        selected: list[list[dict[str, Any]]] = []
        for turn in reversed(live_turns):
            turn_cost = turn.token_count + len(turn.items) * _PER_MESSAGE_OVERHEAD_TOKENS
            if turn_cost > history_budget:
                break
            selected.append(list(turn.items))
            history_budget -= turn_cost
        selected.reverse()

        if carryover is not None:
            messages.extend(self._render_carryover(carryover, history_budget))
        if snapshot_msg is not None:
            messages.append(snapshot_msg)
        for turn_items in selected:
            messages.extend(turn_items)

        # 7. Current user message (always included)
        messages.append({"role": "user", "content": current_text})

        # 8. Memory block last (for prefix caching — varies per call)
        memory_text = self._build_memory_text(memory_result)
        if memory_text:
            messages.append({"role": "developer", "content": memory_text})

        # Rolling summarization stays paused while the carryover holds the
        # view — previous and current session must never mix into one summary.
        if self._summarizer is not None and carryover is None:
            self._summarizer.maybe_schedule(turns)

        return messages

    # ------------------------------------------------------------------
    # Carryover
    # ------------------------------------------------------------------

    def _load_carryover(self, backend: IStorageBackend, session_id: str) -> _Carryover | None:
        """Assemble the previous session's raw view from storage.

        Turns already covered by the persisted rolling summary are skipped;
        the summary text rides along in the header instead. Returns None
        when there is no previous session or on any failure — carryover is
        a continuity enhancement, never a startup blocker.
        """
        try:
            latest = backend.get_latest_session(exclude_session_id=session_id)
            if latest is None:
                return None
            prev_sid, started_at = latest
            rows = backend.load_session(prev_sid)
            summary = backend.load_rolling_summary(prev_sid)
        except Exception:
            logger.warning("Failed to load previous-session carryover", exc_info=True)
            return None

        summary_text: str | None = None
        watermark = -1
        if summary is not None:
            summary_text, watermark = summary

        turn_units: list[tuple[tuple[dict[str, Any], ...], int]] = []
        cur_items: list[dict[str, Any]] = []
        cur_cost = 0
        cur_tid: int | None = None
        for _msg_id, turn_id, item, token_count in rows:
            if turn_id <= watermark:
                continue
            if turn_id != cur_tid and cur_items:
                turn_units.append((tuple(cur_items), cur_cost))
                cur_items, cur_cost = [], 0
            cur_tid = turn_id
            cur_items.append(item)
            cur_cost += token_count + _PER_MESSAGE_OVERHEAD_TOKENS
        if cur_items:
            turn_units.append((tuple(cur_items), cur_cost))
        if not turn_units and summary_text is None:
            return None

        header_text = format_carryover_block(started_at, summary_text)
        marker_text = format_session_boundary(utc_now_str())
        header_tokens = self._token_counter(header_text) + _PER_MESSAGE_OVERHEAD_TOKENS
        marker_tokens = self._token_counter(marker_text) + _PER_MESSAGE_OVERHEAD_TOKENS
        total = header_tokens + marker_tokens + sum(cost for _, cost in turn_units)
        logger.info("Carryover loaded: session %s, %d turn(s), %d tokens", prev_sid, len(turn_units), total)
        return _Carryover(
            session_id=prev_sid,
            started_at=started_at,
            header_text=header_text,
            header_tokens=header_tokens,
            marker_text=marker_text,
            marker_tokens=marker_tokens,
            turn_units=tuple(turn_units),
            total_tokens=total,
        )

    def _history_demand(self, turns: list[HistoryTurn]) -> int:
        """Would-be history cost if everything were shown: carryover + all turns."""
        demand = self._carryover.total_tokens if self._carryover is not None else 0
        for turn in turns:
            demand += turn.token_count + len(turn.items) * _PER_MESSAGE_OVERHEAD_TOKENS
        return demand

    def _evict_carryover(self) -> None:
        """Demote the carried previous session to episodes in the recent block.

        Must be called under ``_state_lock``. Deferred (no-op) while the
        previous session's episode extraction is still pending — the oldest-
        turn-drop overflow fallback covers the wait. Without memory storage
        there is nothing to demote into, so the carryover is simply dropped.
        """
        if self._carryover is None:
            return
        sid = self._carryover.session_id
        if self._memory_storage is None:
            self._carryover = None
            logger.info("Carryover dropped (no memory storage): session %s", sid)
            return
        try:
            if sid not in self._memory_storage.get_processed_session_ids([sid]):
                return
            episodes = self._memory_storage.get_episodes_by_session_ids([sid]).get(sid, [])
        except Exception:
            logger.warning("Carryover eviction failed — will retry on next build", exc_info=True)
            return
        if episodes:
            self._recent_block_texts.append(format_session_summary_block(self._carryover.started_at, episodes))
        self._carryover = None
        logger.info("Carryover evicted: session %s → %d episode(s) in recent block", sid, len(episodes))

    def _render_carryover(self, carryover: _Carryover, budget: int) -> list[dict[str, Any]]:
        """Render the carryover into messages within the remaining history budget.

        Header and boundary marker are fixed cost; turns fill the rest
        newest-first (oldest dropped), mirroring live-turn overflow handling.
        Returns [] when nothing meaningful fits.
        """
        budget -= carryover.header_tokens + carryover.marker_tokens
        if budget < 0:
            return []
        chosen: list[tuple[dict[str, Any], ...]] = []
        for items, cost in reversed(carryover.turn_units):
            if cost > budget:
                break
            chosen.append(items)
            budget -= cost
        if carryover.turn_units and not chosen:
            return []
        chosen.reverse()
        messages: list[dict[str, Any]] = [{"role": "developer", "content": carryover.header_text}]
        for items in chosen:
            messages.extend(items)
        messages.append({"role": "developer", "content": carryover.marker_text})
        return messages

    # ------------------------------------------------------------------
    # Recent sessions block
    # ------------------------------------------------------------------

    def _load_session_context(
        self,
        memory_storage: IMemoryStorage,
        session_id: str,
    ) -> tuple[list[Profile], list[str], set[str]]:
        """Load profiles and the recent-sessions block from memory storage.

        Sessions without episodes (extraction pending, failed, or judged
        meaningless) are skipped — the carryover covers the only session
        whose extraction can still be legitimately in flight.

        Returns:
            (profiles, block texts in chronological order,
            session IDs actually included in the block).
        """
        profiles = memory_storage.get_all_profiles()
        carryover_sid = self._carryover.session_id if self._carryover is not None else None
        recent = memory_storage.get_recent_sessions(self._RECENT_SESSION_CANDIDATES, exclude_session_id=session_id)
        sids = [sid for sid, _ in recent if sid != carryover_sid]
        episodes_by_sid = memory_storage.get_episodes_by_session_ids(sids)

        candidates: list[tuple[str | None, str]] = []  # newest first
        for sid, started_at in recent:
            if sid == carryover_sid:
                continue
            episodes = episodes_by_sid.get(sid, [])
            if not episodes:
                continue
            candidates.append((sid, format_session_summary_block(started_at, episodes)))

        selected = self._select_recent_blocks(candidates)
        block_texts = [text for _, text in selected]
        included_ids = {sid for sid, _ in selected if sid is not None}
        return profiles, block_texts, included_ids

    def _select_recent_blocks(self, candidates: list[tuple[str | None, str]]) -> list[tuple[str | None, str]]:
        """Fill whole sessions newest-first under the soft cap; return chronological.

        The newest candidate is always included regardless of size (soft
        cap); older ones are appended while the running total stays within
        ``_MAX_RECENT_SESSIONS_TOKENS``, stopping at the first that no
        longer fits (keeps the block temporally contiguous).
        """
        selected: list[tuple[str | None, str]] = []
        spent = 0
        for sid, text in candidates:
            cost = self._token_counter(text) + _PER_MESSAGE_OVERHEAD_TOKENS
            if selected and spent + cost > self._MAX_RECENT_SESSIONS_TOKENS:
                break
            selected.append((sid, text))
            spent += cost
        selected.reverse()
        return selected

    # ------------------------------------------------------------------
    # Memory block
    # ------------------------------------------------------------------

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
