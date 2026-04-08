"""Shared data types for the voice pipeline.

Types defined here are passed across module boundaries. Module-internal
types belong in their own modules.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("voice_pipeline.core")

# ---------------------------------------------------------------------------
# Callable type aliases
# ---------------------------------------------------------------------------

TokenCounter = Callable[[str], int]
"""Counts tokens in a string. Vendor-specific implementations provided in Phase 3."""

# ---------------------------------------------------------------------------
# Primitive aliases
# ---------------------------------------------------------------------------

AudioFrame = bytes
"""Raw PCM audio bytes for one capture frame. Size determined by AudioConfig."""


# ---------------------------------------------------------------------------
# System-level enums
# ---------------------------------------------------------------------------


class SystemMode(enum.Enum):
    """Top-level state machine modes."""

    SLEEP = "sleep"
    GREETING = "greeting"
    ACTIVE = "active"
    FAREWELL = "farewell"


class GeneratorState(enum.Enum):
    """SpeechGenerator background preparation state.

    IDLE      — no preparation in progress, ready to accept prepare().
    PREPARING — background LLM+TTS generation is running.
    STREAMING — LLM text collected, TTS audio chunks available via poll_audio().
    FAILED    — generation failed, Orchestrator should skip this turn.
    """

    IDLE = "idle"
    PREPARING = "preparing"
    STREAMING = "streaming"
    FAILED = "failed"


class PlaybackState(enum.Enum):
    """Audio playback state tracked by Orchestrator.

    IDLE         — no audio being played or pending.
    PLAYING      — C++ is actively playing TTS audio.
    STOP_PENDING — interrupt sent to C++, awaiting stop confirmation.
    """

    IDLE = "idle"
    PLAYING = "playing"
    STOP_PENDING = "stop_pending"


class LEDState(enum.Enum):
    """LED display states triggered by the pipeline.

    Implementations map these states to specific colors/animations.
    """

    OFF = "off"
    SLEEPING = "sleeping"
    IDLE = "idle"


# ---------------------------------------------------------------------------
# Turn-taking types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnDecision:
    """Output of TurnDetector per audio frame.

    At most one signal is True per decision. All False means no action.
    """

    turn_shift: bool = False
    interrupt: bool = False
    prepare: bool = False

    def __post_init__(self) -> None:
        if sum([self.turn_shift, self.interrupt, self.prepare]) > 1:
            raise ValueError(
                "TurnDecision: at most one signal may be True. "
                f"Got turn_shift={self.turn_shift}, interrupt={self.interrupt}, "
                f"prepare={self.prepare}"
            )

    @classmethod
    def none(cls) -> TurnDecision:
        """Return a no-op decision (all signals False)."""
        return cls()


@dataclass(frozen=True)
class VAPResult:
    """Output of the VAP model per audio frame.

    Attributes:
        p_now: Probability of current user voice activity.
        p_fut: Probability of near-future user voice activity.
        user_is_speaking: Derived boolean from VAP thresholds.
    """

    p_now: float
    p_fut: float
    user_is_speaking: bool

    def __post_init__(self) -> None:
        if not (0.0 <= self.p_now <= 1.0):
            raise ValueError(f"p_now must be in [0, 1], got {self.p_now}")
        if not (0.0 <= self.p_fut <= 1.0):
            raise ValueError(f"p_fut must be in [0, 1], got {self.p_fut}")


# ---------------------------------------------------------------------------
# TTS / response types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WordTimestamp:
    """Word-level timestamp from TTS synthesis.

    Attributes:
        word: The spoken word.
        start_sec: Start time in seconds from audio start.
        end_sec: End time in seconds from audio start.
    """

    word: str
    start_sec: float
    end_sec: float

    def __post_init__(self) -> None:
        if self.start_sec < 0:
            raise ValueError(f"start_sec must be non-negative, got {self.start_sec}")
        if self.end_sec < 0:
            raise ValueError(f"end_sec must be non-negative, got {self.end_sec}")
        if self.start_sec > self.end_sec:
            raise ValueError(
                f"start_sec ({self.start_sec}) must not exceed end_sec ({self.end_sec})"
            )


@dataclass(frozen=True)
class TTSResult:
    """Result of a TTS synthesis call.

    Attributes:
        audio: Raw PCM audio bytes.
        timestamps: Word-level timestamps if supported, empty list otherwise.
    """

    audio: bytes
    timestamps: tuple[WordTimestamp, ...] = ()


class TTSStream(Iterator[bytes]):
    """Streaming TTS result. Yields PCM audio chunks.

    After full iteration, ``.audio`` / ``.timestamps`` / ``.result`` become
    available.  Must be closed (full iteration or ``.close()``) to release
    resources.  Supports ``with`` statement for automatic cleanup.

    Threading: consume from a single thread only. If cancellation from
    another thread is needed, call ``.close()`` which sets the closed flag.
    """

    __slots__ = ("_gen", "_close_fn", "_ts_fn", "_audio", "_done", "_closed", "_ts_cache")

    def __init__(
        self,
        gen: Generator[bytes, None, None],
        *,
        close_fn: Callable[[], None] | None = None,
        timestamps_fn: Callable[[], tuple[WordTimestamp, ...]] | None = None,
    ) -> None:
        self._gen = gen
        self._close_fn = close_fn
        self._ts_fn = timestamps_fn
        self._audio = bytearray()
        self._done = False
        self._closed = False
        self._ts_cache: tuple[WordTimestamp, ...] | None = None

    def __next__(self) -> bytes:
        if self._closed:
            raise StopIteration
        try:
            chunk = next(self._gen)
            self._audio.extend(chunk)
            return chunk
        except StopIteration:
            self._done = True
            raise

    def __iter__(self) -> TTSStream:
        return self

    def __enter__(self) -> TTSStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the generator and release resources."""
        if self._closed:
            return
        self._closed = True
        try:
            self._gen.close()
        finally:
            if self._close_fn is not None:
                try:
                    self._close_fn()
                except Exception:
                    logger.debug("Error in close_fn (suppressed)", exc_info=True)

    @property
    def audio(self) -> bytes:
        """Full audio data. Only available after complete iteration."""
        if not self._done:
            raise RuntimeError("Audio not available until stream is fully consumed")
        return bytes(self._audio)

    @property
    def timestamps(self) -> tuple[WordTimestamp, ...]:
        """Word-level timestamps. Only available after complete iteration."""
        if not self._done:
            raise RuntimeError("Timestamps not available until stream is fully consumed")
        if self._ts_cache is None:
            self._ts_cache = self._ts_fn() if self._ts_fn is not None else ()
        return self._ts_cache

    @property
    def result(self) -> TTSResult:
        """Convenience: return a TTSResult from the completed stream."""
        return TTSResult(audio=self.audio, timestamps=self.timestamps)


@dataclass
class ResponseData:
    """Complete robot response: text, audio, optional timestamps, and LLM metadata.

    Produced by SpeechGenerator after LLM + TTS pipeline completes.
    Consumed by Orchestrator, CppBridge, UtteranceTruncator, ConversationHistory.

    Attributes:
        text: The final assistant text (after tool loop if applicable).
        audio: Raw PCM audio bytes.
        timestamps: Word-level timestamps from TTS.
        turn_items: Ordered Responses API input items for the entire
            assistant turn. For simple responses: one assistant message.
            For tool calls: tool_call + tool_output + ... + final assistant.
        metrics_list: LLMMetrics from each LLM call in the generation
            (multiple when tool loop runs).
        cited_memory_ids: Database IDs of episodes cited by the LLM
            (resolved from ``[MEMORIES: M1, M2]`` tag). Empty when
            memory is not active or no citations were produced.
    """

    text: str
    audio: bytes
    timestamps: list[WordTimestamp] = field(default_factory=list)
    turn_items: list[dict[str, Any]] = field(default_factory=list)
    metrics_list: list[LLMMetrics] = field(default_factory=list)
    cited_memory_ids: list[int] = field(default_factory=list)

    @property
    def has_timestamps(self) -> bool:
        """True if word-level timestamp data is available."""
        return len(self.timestamps) > 0


# ---------------------------------------------------------------------------
# LLM types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Usage:
    """Token usage from a single LLM API call.

    Direct mapping from OpenAI Responses API usage fields.
    """

    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass(frozen=True)
class LLMMetrics:
    """Performance metadata from a single LLM generation call.

    Stored as metrics_json in the messages table.
    Extensible via JSON serialization — new fields can be added
    without schema changes.

    Attributes:
        usage: Token usage breakdown.
        model: Model identifier string from the API response.
        latency_ms: Total wall time from request start to final token.
        ttft_ms: Time to first token in milliseconds.
    """

    usage: Usage
    model: str
    latency_ms: int
    ttft_ms: int


@dataclass(frozen=True)
class ToolCall:
    """A function tool call requested by the LLM.

    Matches OpenAI Responses API function_call output item.

    Attributes:
        call_id: Unique identifier for this tool call (from API).
        name: Function name to invoke.
        arguments: JSON-encoded string of function arguments.
    """

    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class LLMResult:
    """Complete result from an LLM generation call.

    Available after fully iterating an LLMStream.

    Attributes:
        text: Full response text (concatenation of all yielded chunks).
        tool_calls: Tool calls from the response, empty tuple if none.
        metrics: LLM call metrics, None if unavailable.
    """

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    metrics: LLMMetrics | None = None


class LLMStream(Iterator[str]):
    """Streaming LLM result. Yields text chunks as they arrive.

    After full iteration, ``.result`` becomes available with the
    complete LLMResult (text, tool_calls, metrics).

    Follows the same consume-then-access pattern as TTSStream.
    Must be closed (full iteration or ``.close()``) to release
    the underlying HTTP connection.

    Threading: consume from a single thread only. Call ``.close()``
    from another thread for cancellation (sets closed flag).
    """

    __slots__ = (
        "_gen",
        "_close_fn",
        "_result_fn",
        "_text_chunks",
        "_done",
        "_closed",
        "_result_cache",
    )

    def __init__(
        self,
        gen: Generator[str, None, None],
        *,
        close_fn: Callable[[], None] | None = None,
        result_fn: Callable[[str], LLMResult] | None = None,
    ) -> None:
        self._gen = gen
        self._close_fn = close_fn
        self._result_fn = result_fn
        self._text_chunks: list[str] = []
        self._done = False
        self._closed = False
        self._result_cache: LLMResult | None = None

    def __next__(self) -> str:
        if self._closed:
            raise StopIteration
        try:
            chunk = next(self._gen)
            self._text_chunks.append(chunk)
            return chunk
        except StopIteration:
            self._done = True
            raise

    def __iter__(self) -> LLMStream:
        return self

    def __enter__(self) -> LLMStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the stream and release resources. Idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            self._gen.close()
        finally:
            if self._close_fn is not None:
                try:
                    self._close_fn()
                except Exception:
                    logger.debug("Error in close_fn (suppressed)", exc_info=True)

    @property
    def text(self) -> str:
        """Full text assembled from yielded chunks. Available after complete iteration."""
        if not self._done:
            raise RuntimeError("Text not available until stream is fully consumed")
        return "".join(self._text_chunks)

    @property
    def result(self) -> LLMResult:
        """Complete LLM result. Available after full iteration."""
        if not self._done:
            raise RuntimeError("Result not available until stream is fully consumed")
        if self._result_cache is None:
            full_text = "".join(self._text_chunks)
            if self._result_fn is not None:
                self._result_cache = self._result_fn(full_text)
            else:
                self._result_cache = LLMResult(text=full_text)
        return self._result_cache


# ---------------------------------------------------------------------------
# History types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistoryTurn:
    """Atomic history unit for ContextBuilder.

    Groups one or more message items that belong to the same turn.
    Included or excluded as a whole during token budget allocation.

    Attributes:
        items: Message dicts in Responses API input format.
        token_count: Pre-computed total token count for all items.
    """

    items: tuple[dict[str, Any], ...]
    token_count: int


# ---------------------------------------------------------------------------
# CppBridge event types
# ---------------------------------------------------------------------------


class CppEventType(enum.Enum):
    """Event types sent from C++ to Python via CppBridge."""

    PLAYBACK_STARTED = "playback_started"
    PLAYBACK_COMPLETE = "playback_complete"


@dataclass(frozen=True)
class CppEvent:
    """An event received from the C++ audio process.

    Attributes:
        event_type: Type of event.
    """

    event_type: CppEventType


# ---------------------------------------------------------------------------
# Pipeline latency tracing
# ---------------------------------------------------------------------------


@dataclass
class PipelineTrace:
    """Timing trace for one turn's response generation pipeline.

    Accumulates monotonic timestamps from SpeechGenerator (background
    thread) and Orchestrator (main thread).  Stored once per turn —
    speculative prepare() replacements within a turn are not stored
    individually; only the final pipeline run's timing is recorded.

    ``to_record()`` converts raw timestamps to millisecond durations
    for SQLite storage (monotonic values are meaningless across
    process restarts).

    Thread safety: the SpeechGenerator background thread writes
    pipeline-stage fields while the Orchestrator main thread writes
    orchestrator-level fields.  Individual float assignments are
    atomic under CPython's GIL, and the fields written by each thread
    do not overlap.
    """

    # -- Identity / metadata --
    session_id: str = ""
    run_id: int = 0
    pipeline_mode: str = "full"
    created_at: str = ""
    outcome: str = ""
    speculative_attempts: int = 1

    # -- Link to conversation history --
    user_msg_id: int = 0

    # -- Orchestrator-level monotonic timestamps --
    prepare_ts: float = 0.0
    turn_shift_ts: float = 0.0
    begin_streaming_ts: float = 0.0
    playback_started_ts: float = 0.0

    # -- SpeechGenerator pipeline-stage monotonic timestamps --
    pipeline_start_ts: float = 0.0
    memory_done_ts: float = 0.0
    context_done_ts: float = 0.0
    llm_start_ts: float = 0.0
    llm_first_token_ts: float = 0.0
    llm_done_ts: float = 0.0
    tts_start_ts: float = 0.0
    tts_first_chunk_ts: float = 0.0
    tts_done_ts: float = 0.0

    # -- From LLMMetrics (already in ms) --
    llm_ttft_ms: float = 0.0

    @staticmethod
    def _delta_ms(start: float, end: float) -> float:
        """Compute millisecond delta, returning 0.0 if either timestamp is missing."""
        if start <= 0 or end <= 0 or end < start:
            return 0.0
        return (end - start) * 1000

    def to_record(self) -> dict[str, object]:
        """Convert to a flat dict of computed durations for DB storage."""
        speculative_ms = (
            max(0.0, (self.turn_shift_ts - self.prepare_ts) * 1000)
            if self.turn_shift_ts > 0 and self.prepare_ts > 0
            else 0.0
        )
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "pipeline_mode": self.pipeline_mode,
            "created_at": self.created_at,
            "outcome": self.outcome,
            "speculative_attempts": self.speculative_attempts,
            "user_msg_id": self.user_msg_id,
            "memory_ms": self._delta_ms(self.pipeline_start_ts, self.memory_done_ts),
            "context_ms": self._delta_ms(self.memory_done_ts, self.context_done_ts),
            "llm_ms": self._delta_ms(self.llm_start_ts, self.llm_done_ts),
            "llm_ttft_ms": self.llm_ttft_ms,
            "tts_ms": self._delta_ms(self.tts_start_ts, self.tts_done_ts),
            "tts_ttfc_ms": self._delta_ms(self.tts_start_ts, self.tts_first_chunk_ts),
            "prepare_to_streaming_ms": self._delta_ms(self.prepare_ts, self.tts_first_chunk_ts),
            "turn_shift_to_playback_ms": self._delta_ms(
                self.turn_shift_ts, self.playback_started_ts
            ),
            "speculative_ms": speculative_ms,
            "bridge_ms": self._delta_ms(self.begin_streaming_ts, self.playback_started_ts),
        }

    def summary(self) -> str:
        """One-line latency summary for logging."""
        r = self.to_record()
        parts = [f"outcome={self.outcome}"]
        ts_to_pb = r["turn_shift_to_playback_ms"]
        if ts_to_pb:
            parts.append(f"ts→pb={ts_to_pb:.0f}ms")
        spec = r["speculative_ms"]
        if spec:
            parts.append(f"spec={spec:.0f}ms")
        for key in ("memory_ms", "context_ms", "llm_ms", "tts_ms", "bridge_ms"):
            v = r[key]
            if v:
                label = key.removesuffix("_ms")
                parts.append(f"{label}={v:.0f}ms")
        ttft = r["llm_ttft_ms"]
        if ttft:
            parts.append(f"llm_ttft={ttft:.0f}ms")
        ttfc = r["tts_ttfc_ms"]
        if ttfc:
            parts.append(f"tts_ttfc={ttfc:.0f}ms")
        if self.speculative_attempts > 1:
            parts.append(f"attempts={self.speculative_attempts}")
        return " | ".join(parts)
