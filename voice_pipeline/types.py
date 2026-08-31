"""벤더 교체 대상 인터페이스(IASR / ILLM / ITTS / IEmbedder)와 그 계약 타입.

인터페이스 시그니처가 주고받는 스트림·결과 타입(TTSStream, LLMStream, LLMMetrics, …)과
공통 별칭(AudioFrame, TokenCounter)만 여기 둔다. 그 외 타입은 만들어 내는 모듈이 소유한다
(TurnDecision → turn_detector, ResponseData → generator, CppEvent → adapters/cpp_bridge, …).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("voice_pipeline.types")


def utc_now_str() -> str:
    """Current UTC wall-clock as a sortable string with microsecond resolution.

    Microsecond precision lets call records produced by different module
    threads be ordered against each other; second resolution collapses
    cross-stage interleaving within the same second.
    """
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")


# ---------------------------------------------------------------------------
# Callable type aliases
# ---------------------------------------------------------------------------

TokenCounter = Callable[[str], int]
"""Counts tokens in a string. Vendor-specific implementations provided in Phase 3."""

# ---------------------------------------------------------------------------
# Primitive aliases
# ---------------------------------------------------------------------------

AudioFrame = bytes
"""Raw PCM audio bytes for one capture frame. Size determined by audio.constants."""


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
            raise ValueError(f"start_sec ({self.start_sec}) must not exceed end_sec ({self.end_sec})")


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
# 벤더 교체 대상 인터페이스 (ASR / LLM / TTS / Embedder)
# ---------------------------------------------------------------------------


class IASR(ABC):
    """Automatic speech recognition interface.

    Lifecycle: Orchestrator calls start() on ACTIVE entry, feed_audio()/get_text()
    per frame, reset() after turn confirmation, stop() on ACTIVE exit.

    Threading: all methods are called from the Orchestrator (main) thread only.
    Implementations do not need to be thread-safe.
    """

    @abstractmethod
    def start(self) -> None:
        """Start the recognition session."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the recognition session and release resources."""

    @abstractmethod
    def feed_audio(self, frame: AudioFrame) -> None:
        """Feed a single audio frame to the recognizer.

        Args:
            frame: Raw PCM audio bytes for one capture frame.
        """

    @abstractmethod
    def get_text(self) -> str:
        """Return the current (interim or final) transcription.

        Returns:
            Current transcription text, or empty string if none available.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset recognizer state for the next turn."""


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


class ILLM(ABC):
    """Large language model interface for response generation.

    Returns LLMStream (Iterator[str] compatible) with post-iteration
    access to LLMResult (text, tool_calls, metrics).
    Synchronous: fits threading+queue model.
    """

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMStream:
        """Generate a streaming response from the given message history.

        Args:
            messages: List of message dicts in vendor-specific format.
            tools: Tool definitions. None uses config defaults.
                Empty list explicitly disables tools for this call.
            response_format: Structured output format specification.
                None = free-form text (default). When provided, passed
                through to the API (e.g. ``{"type": "json_schema", ...}``).

        Returns:
            LLMStream yielding text chunks. After full iteration,
            .result provides LLMResult with text, tool_calls, metrics.
        """


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


class ITTS(ABC):
    """Text-to-speech synthesis interface.

    synthesize() returns a TTSStream that yields PCM audio chunks.
    After iteration, .timestamps and .audio are available on the stream.
    Thread-safe: concurrent synthesize() calls are independent.
    """

    @property
    @abstractmethod
    def output_sample_rate(self) -> int:
        """PCM 출력 샘플레이트 (Hz). vendor/모델별로 고정값."""

    @property
    @abstractmethod
    def voice_id(self) -> str:
        """동일 음성 설정을 식별하는 문자열 (vendor + 설정 조합).

        cache 무효화 등 "같은 음성인지" 비교에 사용.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier for logging and tracing."""

    @abstractmethod
    def synthesize(self, text: str) -> TTSStream:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize.

        Returns:
            TTSStream yielding PCM audio chunks. Iterate to receive audio.
            After iteration, access .audio, .timestamps, or .result.
        """


class IEmbedder(ABC):
    """Text embedding provider.

    Converts text to dense vectors for semantic search.
    Shared across modules (memory, similarity, etc.).

    Implementations may use local models or external APIs.
    """

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text.

        Args:
            text: Input text.

        Returns:
            1-D float32 array of shape (dimension,).
        """

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts in a single call.

        Args:
            texts: List of input texts.

        Returns:
            2-D float32 array of shape (len(texts), dimension).
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier for logging and tracing."""
