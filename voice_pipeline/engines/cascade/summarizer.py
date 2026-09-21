"""히스토리 롤링 요약 (cascade) — 히스토리 예산을 넘기 전에 백그라운드 LLM 으로 앞부분을 요약한다.

:class:`HistorySummarizer` 가 요약 잡을 돌리고, 결과 :class:`HistorySummarySnapshot` 을
:class:`~voice_pipeline.engines.cascade.context_builder.ContextBuilder` 가 요약된 턴 대신 넣는다.
저장소의 원문 히스토리는 바꾸지 않는다.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from voice_pipeline.history import HistoryTurn, SQLiteStorageBackend
from voice_pipeline.session_context import PER_MESSAGE_OVERHEAD_TOKENS
from voice_pipeline.settings import SUMMARY_MAX_TOKENS
from voice_pipeline.trace import record_call
from voice_pipeline.types import ILLM, TokenCounter

logger = logging.getLogger("voice_pipeline.prompt")


@dataclass(frozen=True)
class HistorySummarySnapshot:
    """Immutable view of the rolling in-session history summary.

    Produced by HistorySummarizer and consumed by ContextBuilder in place
    of the turns it covers. Raw history in storage is never modified.

    Attributes:
        block_text: Formatted developer-message content (header included).
        token_count: Pre-counted tokens of ``block_text``.
        through_turn_id: Last turn_id covered by this summary. Turns with
            a greater turn_id are still live and sent verbatim.
    """

    block_text: str
    token_count: int
    through_turn_id: int


def format_history_summary_block(summary_text: str) -> str:
    """Format the rolling summary of earlier turns in the current session.

    Shown in place of the turns it covers, right before the live history.

    Output example::

        [Earlier in this conversation]
        User asked about weekend plans; Ray suggested a movie night. ...
    """
    return f"[Earlier in this conversation]\n{summary_text}"


_SUMMARY_SYSTEM_PROMPT = """\
You maintain a running summary of an ongoing spoken conversation between User and Ray, \
a voice assistant robot. Merge the previous summary and the new transcript into one updated summary. \
Preserve concrete facts, names, dates, user preferences, decisions, commitments, and unresolved topics. \
Drop filler and pleasantries. Write plain English prose, under {max_words} words. \
Output only the summary text.\
"""


class HistorySummarizer:
    """Background rolling summarizer over ConversationHistory turns.

    Threading: ``maybe_schedule`` is called from SpeechGenerator's pipeline
    thread (via ContextBuilder.build). At most one summarization job runs at
    a time on a daemon worker thread; state swaps are atomic under a lock.
    Failed or truncated jobs leave the previous snapshot intact and are
    retried on the next trigger.
    """

    _TRIGGER_RATIO = 0.75  # 히스토리 예산 대비 사용률이 이 값을 넘으면 요약 발동
    _KEEP_RECENT_TURNS = 20  # 요약 대상에서 제외하고 원문 유지할 최근 턴 수 (user/assistant 각 1턴)
    _SUMMARY_MAX_WORDS = 250  # 프롬프트 soft limit (≈384 tokens)
    _HARD_CAP_TOKENS = SUMMARY_MAX_TOKENS  # 요약 LLM max_output_tokens (wiring의 summary LLM과 동기)

    def __init__(
        self,
        llm: ILLM,
        token_counter: TokenCounter,
        history_budget_tokens: int,
        *,
        session_id: str = "",
        summary_backend: SQLiteStorageBackend | None = None,
    ) -> None:
        """Initialize the summarizer.

        Args:
            llm: 요약 전용 LLM. tools 비활성, ``max_tokens=_HARD_CAP_TOKENS``로
                구성된 인스턴스를 기대한다.
            token_counter: 요약 블록 토큰 계산용 카운터.
            history_budget_tokens: ContextBuilder의 히스토리 예산
                (트리거 판정 기준).
            session_id: 요약 영속화(다음 세션 이월)에 쓸 세션 ID.
            summary_backend: 요약 영속화 대상 히스토리 백엔드. 스왑 성공 시
                다음 세션의 이월(carryover) 로딩을 위해 저장한다.
                ``None``이면 저장 안 함.
        """
        self._llm = llm
        self._token_counter = token_counter
        self._trigger_tokens = int(history_budget_tokens * self._TRIGGER_RATIO)
        self._session_id = session_id
        self._summary_backend = summary_backend

        self._lock = threading.Lock()
        self._snapshot: HistorySummarySnapshot | None = None
        self._raw_text = ""  # 헤더 없는 요약 원문 — 롤링 병합 입력용
        self._in_flight = False
        self._closed = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # HistorySummarizer
    # ------------------------------------------------------------------

    def snapshot(self) -> HistorySummarySnapshot | None:
        """Return the current summary snapshot, or None if none exists yet."""
        with self._lock:
            return self._snapshot

    def maybe_schedule(self, turns: list[HistoryTurn]) -> None:
        """Kick off a background summarization when usage crosses the trigger.

        Non-blocking. No-op below the threshold, while a job is in flight,
        or after close().
        """
        with self._lock:
            if self._closed or self._in_flight:
                return
            watermark = self._snapshot.through_turn_id if self._snapshot else -1
            live = [t for t in turns if t.turn_id > watermark]
            if self._usage_tokens(live) < self._trigger_tokens:
                return

            candidates = live[: -self._KEEP_RECENT_TURNS] if len(live) > self._KEEP_RECENT_TURNS else []
            # Align the cutoff to an exchange boundary: the first kept turn
            # must start with a user message, so a question is never split
            # from its answer.
            while candidates and not _starts_exchange(live[len(candidates)]):
                candidates.pop()
            if not candidates:
                return
            transcript = _render_transcript(candidates)
            if not transcript:
                return

            prev_summary = self._raw_text
            through_turn_id = candidates[-1].turn_id
            self._in_flight = True

        self._thread = threading.Thread(
            target=self._run_job,
            args=(prev_summary, transcript, through_turn_id),
            daemon=True,
            name="history-summarizer",
        )
        self._thread.start()

    def close(self) -> None:
        """Reject further scheduling and discard any in-flight result."""
        with self._lock:
            self._closed = True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _usage_tokens(self, live_turns: list[HistoryTurn]) -> int:
        """Current history-view cost: summary block + live turns (with framing)."""
        used = self._snapshot.token_count + PER_MESSAGE_OVERHEAD_TOKENS if self._snapshot else 0
        for turn in live_turns:
            used += turn.token_count + len(turn.items) * PER_MESSAGE_OVERHEAD_TOKENS
        return used

    def _run_job(self, prev_summary: str, transcript: str, through_turn_id: int) -> None:
        start = time.monotonic()
        status = "ok"
        try:
            summary, output_tokens = self._call_llm(prev_summary, transcript)
            if output_tokens is not None and output_tokens >= self._HARD_CAP_TOKENS:
                # Hard-stop hit → likely cut mid-sentence. A corrupted summary
                # is worse than none; keep the previous one and retry later.
                status = "truncated"
                logger.warning(
                    "History summary hit hard cap (%d tokens) — discarded, will retry",
                    output_tokens,
                )
                return
            if not summary:
                status = "empty"
                logger.warning("History summary came back empty — discarded, will retry")
                return

            block = format_history_summary_block(summary)
            block_tokens = self._token_counter(block)
            with self._lock:
                if self._closed:
                    return
                self._raw_text = summary
                self._snapshot = HistorySummarySnapshot(
                    block_text=block,
                    token_count=block_tokens,
                    through_turn_id=through_turn_id,
                )
            self._persist(summary, through_turn_id)
            logger.info(
                "History summary updated: through turn %d, block %d tokens",
                through_turn_id,
                block_tokens,
            )
        except Exception:
            status = "error"
            logger.warning("History summarization failed — will retry on next trigger", exc_info=True)
        finally:
            self._record_call(status, (time.monotonic() - start) * 1000)
            with self._lock:
                self._in_flight = False

    def _call_llm(self, prev_summary: str, transcript: str) -> tuple[str, int | None]:
        """Run the summary LLM call. Returns (summary text, output token count)."""
        system = _SUMMARY_SYSTEM_PROMPT.format(max_words=self._SUMMARY_MAX_WORDS)
        user_parts = []
        if prev_summary:
            user_parts.append(f"Previous summary:\n{prev_summary}")
        user_parts.append(f"New transcript:\n{transcript}")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]
        stream = self._llm.generate(messages, tools=[])
        text = "".join(stream)
        result = stream.result
        output_tokens = result.metrics.usage.output_tokens if result.metrics is not None else None
        return text.strip(), output_tokens

    def _persist(self, summary: str, through_turn_id: int) -> None:
        """Persist the summary for next-session carryover. Failure is non-fatal."""
        if self._summary_backend is None:
            return
        try:
            self._summary_backend.save_rolling_summary(self._session_id, summary, through_turn_id)
        except Exception:
            logger.warning("Failed to persist rolling summary", exc_info=True)

    def _record_call(self, status: str, elapsed_ms: float) -> None:
        record_call(
            "history_summarizer", "summarize", getattr(self._llm, "model", "unknown"), elapsed_ms, status=status
        )


def _starts_exchange(turn: HistoryTurn) -> bool:
    """True if the turn begins a new user exchange."""
    return bool(turn.items) and turn.items[0].get("role") == "user"


def _render_transcript(turns: list[HistoryTurn]) -> str:
    """Render user/assistant text items as a transcript; other items skipped."""
    lines = []
    for turn in turns:
        for item in turn.items:
            role = item.get("role")
            content = item.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                label = "User" if role == "user" else "Ray"
                lines.append(f"{label}: {content.strip()}")
    return "\n".join(lines)
