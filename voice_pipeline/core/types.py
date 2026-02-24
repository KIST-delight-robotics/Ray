"""Shared data types for the voice pipeline.

Types defined here are passed across module boundaries. Module-internal
types belong in their own modules.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field

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
