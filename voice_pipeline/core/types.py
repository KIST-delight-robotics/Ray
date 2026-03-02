"""Shared data types for the voice pipeline.

Types defined here are passed across module boundaries. Module-internal
types belong in their own modules.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field

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
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


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
    resources.
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
    """Complete robot response: text, audio, and optional word timestamps.

    Produced by SpeechGenerator after LLM + TTS pipeline completes.
    Consumed by Orchestrator, CppBridge, UtteranceTruncator, ConversationHistory.
    """

    text: str
    audio: bytes
    timestamps: list[WordTimestamp] = field(default_factory=list)

    @property
    def has_timestamps(self) -> bool:
        """True if word-level timestamp data is available."""
        return len(self.timestamps) > 0


# ---------------------------------------------------------------------------
# CppBridge event types
# ---------------------------------------------------------------------------


class CppEventType(enum.Enum):
    """Event types sent from C++ to Python via CppBridge."""

    PLAYBACK_STARTED = "playback_started"
    PLAYBACK_POSITION = "playback_position"
    PLAYBACK_COMPLETE = "playback_complete"
    PLAYBACK_STOPPED = "playback_stopped"


@dataclass(frozen=True)
class CppEvent:
    """An event received from the C++ audio process.

    Attributes:
        event_type: Type of event.
        position_sec: Playback position in seconds. Meaningful for
            PLAYBACK_POSITION and PLAYBACK_STOPPED. None otherwise.
    """

    event_type: CppEventType
    position_sec: float | None = None
