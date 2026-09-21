"""LLM 입력 조립 (cascade) — 턴마다 블록별 고정 토큰 예산으로 messages 를 쌓는다.

- ``DEFAULT_SYSTEM_PROMPT``
- ``format_*``: 턴 단위 블록 포매터 (carryover / history summary / memory). 프로필·최근 세션 블록은
  두 엔진이 공유하므로 :mod:`voice_pipeline.session_context` 에 있다.
- ``ContextBuilder``: 블록별 고정 토큰 예산으로 messages 조립. 롤링 요약은 :mod:`.summarizer`.

블록 순서는 prefix cache를 살리기 위해 고정 블록 먼저, 변동 블록 나중:
  1 system  2 profile  3 recent sessions  4 carryover  5 history summary
  6 history turns  7 current user  8 memory (매 호출 바뀌므로 마지막)

예산은 블록별로 독립이며 전체 공용 예산은 없다. 히스토리(4+5+6) 예산은 ``settings.HISTORY_TOKEN_BUDGET``.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from voice_pipeline.engines.cascade.summarizer import HistorySummarizer
from voice_pipeline.history import ConversationHistory, HistoryTurn, SQLiteStorageBackend
from voice_pipeline.memory.storage import SQLiteMemoryStorage
from voice_pipeline.session_context import (
    PER_MESSAGE_OVERHEAD_TOKENS,
    RECENT_SESSIONS_MAX_TOKENS,
    SESSION_PAGE_SIZE,
    format_profile_block,
    format_session_summary_block,
    load_session_context,
    select_recent_blocks,
)
from voice_pipeline.settings import HISTORY_TOKEN_BUDGET
from voice_pipeline.types import TokenCounter, utc_now_str

if TYPE_CHECKING:
    from voice_pipeline.memory.types import MemoryReadResult, Profile

logger = logging.getLogger("voice_pipeline.prompt")


DEFAULT_SYSTEM_PROMPT = """\
You are Ray, a friendly conversational companion.
Your response is converted to speech via TTS — write only what sounds natural spoken aloud.
Keep responses to 1-3 sentences.

If you used any retrieved memories in your response, append [MEMORIES: M1, M2] (listing only the ones \
you used) at the very end. If you did not use any, do not append anything.\
"""


# Match "[MEMORIES: M1, M2, ...]" at the end of text (with optional trailing whitespace)
_CITATION_RE = re.compile(r"\[MEMORIES:\s*(M\d+(?:\s*,\s*M\d+)*)\s*\]\s*$")

# Markdown link [text](url) — including optional wrapping parentheses
_MD_LINK_WITH_PARENS_RE = re.compile(r"\(\[[^\]]*\]\([^)]*\)\)")
_MD_LINK_RE = re.compile(r"\[[^\]]*\]\([^)]*\)")


# ---------------------------------------------------------------------------
# Previous-session carryover
# ---------------------------------------------------------------------------


def format_carryover_block(started_at: str, summary_text: str | None = None) -> str:
    """Header opening the previous session's raw turns carried into context.

    Args:
        started_at: Previous session start timestamp (UTC, '%Y-%m-%d %H:%M:%S').
        summary_text: Persisted rolling summary covering the earlier part of
            that session, or None if it never summarized.

    Output example::

        [Previous session — 2026-03-28 21:30]
        Earlier in that session: User asked about weekend plans; Ray suggested
        a movie night. ...
    """
    display_time = started_at[:16] if len(started_at) >= 16 else started_at
    header = f"[Previous session — {display_time}]"
    if summary_text:
        return f"{header}\nEarlier in that session: {summary_text}"
    return header


def format_session_boundary(started_at: str) -> str:
    """Marker separating carried-over previous-session turns from the current session.

    Args:
        started_at: Current session start timestamp (UTC, '%Y-%m-%d %H:%M:%S').

    Output example::

        [New session — 2026-03-29 09:15]
    """
    display_time = started_at[:16] if len(started_at) >= 16 else started_at
    return f"[New session — {display_time}]"


# ---------------------------------------------------------------------------
# Block 4: Retrieved memories
# ---------------------------------------------------------------------------


def format_memory_block(memory_result: MemoryReadResult) -> str:
    """Format retrieved memories for LLM context injection (Block 4).

    Output example::

        [Retrieved Memories]
        [M1] User cried watching Interstellar on a rainy day. (2026-03-15)
        [M2] User said Dune 2 was better than the original. (2026-03-20)
    """
    if not memory_result.episodes:
        return ""
    lines = ["[Retrieved Memories]"]
    for i, ep in enumerate(memory_result.episodes, 1):
        date = ep.timestamp[:10] if len(ep.timestamp) >= 10 else ep.timestamp
        lines.append(f"[M{i}] {ep.text} ({date})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Citation parsing
# ---------------------------------------------------------------------------


def strip_urls(text: str) -> str:
    """Remove markdown links from text.

    Handles both ``([text](url))`` and ``[text](url)`` forms.
    """
    text = _MD_LINK_WITH_PARENS_RE.sub("", text)
    text = _MD_LINK_RE.sub("", text)
    return re.sub(r"  +", " ", text).strip()


def parse_citation_tag(text: str) -> tuple[str, list[int]]:
    """Parse ``[MEMORIES: M1, M2]`` from the end of LLM output.

    Args:
        text: Raw LLM response text.

    Returns:
        Tuple of (clean_text, cited_indices) where cited_indices are
        1-based integers (e.g. [1, 3] for M1, M3). If no tag is found,
        returns (text, []).
    """
    match = _CITATION_RE.search(text)
    if not match:
        return (text, [])

    clean = text[: match.start()].rstrip()
    raw_indices = match.group(1).split(",")
    cited: list[int] = []
    for token in raw_indices:
        token = token.strip().lstrip("Mm")
        try:
            cited.append(int(token))
        except ValueError:
            continue
    return (clean, cited)


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

    _MAX_HISTORY_TOKENS = HISTORY_TOKEN_BUDGET  # 히스토리 뷰 예산 (이월 + 요약 블록 + 라이브 턴)
    _MAX_MEMORY_TOKENS = 512  # retrieved memory 블록 전용 예산 (초과 시 낮은 salience 순 drop)
    _MAX_PROFILE_TOKENS = 256  # profile 블록 전용 예산 (초과 시 블록 skip)
    _MAX_RECENT_SESSIONS_TOKENS = RECENT_SESSIONS_MAX_TOKENS  # 최근 세션 블록 soft cap
    _SESSION_PAGE_SIZE = SESSION_PAGE_SIZE  # 세션 에피소드 lazy 로딩 배치 크기
    _CARRYOVER_EVICT_RATIO = 0.75  # 히스토리 수요가 예산 대비 이 비율을 넘으면 이월분 퇴거

    def __init__(
        self,
        history: ConversationHistory,
        system_prompt: str,
        token_counter: TokenCounter,
        profiles: list[Profile] | None = None,
        session_summaries: list[str] | None = None,
        *,
        memory_storage: SQLiteMemoryStorage | None = None,
        session_id: str | None = None,
        summarizer: HistorySummarizer | None = None,
        history_backend: SQLiteStorageBackend | None = None,
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
            carryover_sid = self._carryover.session_id if self._carryover is not None else None
            profiles, self._recent_block_texts, included_ids = load_session_context(
                memory_storage,
                session_id,
                token_counter,
                carryover_session_id=carryover_sid,
                max_tokens=self._MAX_RECENT_SESSIONS_TOKENS,
                page_size=self._SESSION_PAGE_SIZE,
            )
            self.exclude_session_ids = {session_id} | included_ids
            if self._carryover is not None:
                self.exclude_session_ids.add(self._carryover.session_id)
        elif session_summaries:
            selected = select_recent_blocks(
                [(None, text) for text in reversed(session_summaries)],
                token_counter,
                max_tokens=self._MAX_RECENT_SESSIONS_TOKENS,
            )
            self._recent_block_texts = [text for _, text in selected]

        # Pre-format and pre-count session-level blocks (immutable)
        self._profile_text = format_profile_block(profiles or [])
        self._profile_tokens = (
            self._token_counter(self._profile_text) + PER_MESSAGE_OVERHEAD_TOKENS if self._profile_text else 0
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
            history_budget -= snapshot.token_count + PER_MESSAGE_OVERHEAD_TOKENS
            snapshot_msg = {"role": "developer", "content": snapshot.block_text}

        live_turns = [t for t in turns if t.turn_id > watermark]
        selected: list[list[dict[str, Any]]] = []
        for turn in reversed(live_turns):
            turn_cost = turn.token_count + len(turn.items) * PER_MESSAGE_OVERHEAD_TOKENS
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

    def _load_carryover(self, backend: SQLiteStorageBackend, session_id: str) -> _Carryover | None:
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
            cur_cost += token_count + PER_MESSAGE_OVERHEAD_TOKENS
        if cur_items:
            turn_units.append((tuple(cur_items), cur_cost))
        if not turn_units and summary_text is None:
            return None

        header_text = format_carryover_block(started_at, summary_text)
        marker_text = format_session_boundary(utc_now_str())
        header_tokens = self._token_counter(header_text) + PER_MESSAGE_OVERHEAD_TOKENS
        marker_tokens = self._token_counter(marker_text) + PER_MESSAGE_OVERHEAD_TOKENS
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
            demand += turn.token_count + len(turn.items) * PER_MESSAGE_OVERHEAD_TOKENS
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
    # Memory block
    # ------------------------------------------------------------------

    def _build_memory_text(self, memory_result: MemoryReadResult | None) -> str:
        """Format the memory block, trimming lowest-salience episodes to fit the cap."""
        if not memory_result or not memory_result.episodes:
            return ""
        memory_text = format_memory_block(memory_result)
        memory_cost = self._token_counter(memory_text) + PER_MESSAGE_OVERHEAD_TOKENS
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
            memory_cost = self._token_counter(memory_text) + PER_MESSAGE_OVERHEAD_TOKENS if memory_text else 0
        return memory_text
