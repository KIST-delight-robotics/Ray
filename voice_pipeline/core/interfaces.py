"""Module interfaces for the voice pipeline.

Only interfaces needed by the next implementation phase are defined here.
New interfaces are added just before their consuming phase begins.

Current: Phase 2 + Phase 3 interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from voice_pipeline.core.types import AudioFrame, CppEvent, LEDState, TTSResult, WordTimestamp

# ---------------------------------------------------------------------------
# StorageBackend
# ---------------------------------------------------------------------------


class IStorageBackend(ABC):
    """Persistence backend for conversation history.

    Implementations: memory, file, database.
    """

    @abstractmethod
    def load(self, session_id: str) -> list[dict[str, Any]]:
        """Load messages for a session.

        Args:
            session_id: Unique session identifier.

        Returns:
            List of message dicts, or empty list if session not found.
        """

    @abstractmethod
    def save(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Persist messages for a session.

        Args:
            session_id: Unique session identifier.
            messages: List of message dicts to persist.
        """

    @abstractmethod
    def delete(self, session_id: str) -> None:
        """Delete stored messages for a session.

        Args:
            session_id: Unique session identifier.
        """


# ---------------------------------------------------------------------------
# ConversationHistory
# ---------------------------------------------------------------------------


class IConversationHistory(ABC):
    """Session-scoped conversation history store.

    Pure data repository. Message dict schema is vendor-specific
    and determined by LLM implementation.
    """

    @abstractmethod
    def new_session(self, session_id: str) -> None:
        """Start a new session, clearing any in-memory state.

        Args:
            session_id: Unique identifier for this conversation session.
        """

    @abstractmethod
    def add_user_message(self, text: str) -> None:
        """Append a user message to the current session.

        Args:
            text: The user's transcribed utterance (final ASR result).
        """

    @abstractmethod
    def add_assistant_message(self, text: str) -> None:
        """Append an assistant message to the current session.

        Called with full response text on normal playback completion,
        or with truncated text on barge-in interruption.

        Args:
            text: The robot's spoken text (full or truncated).
        """

    @abstractmethod
    def get_messages(self) -> list[dict[str, Any]]:
        """Retrieve all conversation messages.

        Returns:
            List of message dicts in vendor-specific format.
        """

    @abstractmethod
    def clear(self) -> None:
        """Remove all messages from the current session in memory."""

    @abstractmethod
    def save(self) -> None:
        """Persist the current session to the storage backend."""


# ---------------------------------------------------------------------------
# UtteranceTruncator
# ---------------------------------------------------------------------------


class IUtteranceTruncator(ABC):
    """Truncates spoken text to match a barge-in stop position.

    Strategy interface with two implementations:
    - TimestampTruncator: uses word-level timestamps for precision.
    - DurationRatioTruncator: estimates from total audio duration ratio.
    """

    @abstractmethod
    def truncate(
        self,
        text: str,
        stop_position_sec: float,
        timestamps: list[WordTimestamp],
    ) -> str:
        """Return the portion of text spoken before the stop point.

        Args:
            text: Full response text that was being played.
            stop_position_sec: Playback position in seconds when the
                robot was interrupted.
            timestamps: Word-level timestamps from TTS. May be empty
                if the TTS implementation does not support them.

        Returns:
            Truncated text representing what was actually spoken.
        """


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------


class IContextBuilder(ABC):
    """Assembles LLM context from conversation history and current input.

    IConversationHistory is injected via the constructor — it does not
    change per call. The build() method only takes the current turn's text.
    """

    @abstractmethod
    def build(self, current_text: str) -> list[dict[str, Any]]:
        """Build the message list for an LLM call.

        Args:
            current_text: Current ASR transcription (the user's in-progress
                turn, not yet committed to history).

        Returns:
            List of message dicts suitable for passing to ILLM.generate().
        """


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------


class IASR(ABC):
    """Automatic speech recognition interface.

    Lifecycle: Orchestrator calls start() on ACTIVE entry, feed_audio()/get_text()
    per frame, reset() after turn confirmation, stop() on ACTIVE exit.
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

    Returns Iterator[str] (not Generator) — consumer only iterates,
    never sends/throws. Synchronous: fits threading+queue model.
    """

    @abstractmethod
    def generate(self, messages: list[dict[str, Any]]) -> Iterator[str]:
        """Generate a streaming response from the given message history.

        Args:
            messages: List of message dicts in vendor-specific format.

        Returns:
            Iterator yielding text chunks as they become available.
        """


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


class ITTS(ABC):
    """Text-to-speech synthesis interface."""

    @abstractmethod
    def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize.

        Returns:
            TTSResult containing audio bytes and optional word timestamps.
        """


# ---------------------------------------------------------------------------
# CppBridge
# ---------------------------------------------------------------------------


class ICppBridge(ABC):
    """Interface to the C++ audio playback process via WebSocket.

    send_audio takes raw bytes (not ResponseData) — bridge doesn't need text.
    poll_event is non-blocking (returns None if empty) — fits frame-driven
    sync loop.
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the C++ process."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection to the C++ process."""

    @abstractmethod
    def send_audio(self, audio: bytes) -> None:
        """Send audio data for playback.

        Args:
            audio: Raw PCM audio bytes.
        """

    @abstractmethod
    def send_stop(self) -> None:
        """Send a stop/interrupt signal to halt playback."""

    @abstractmethod
    def send_greeting(self) -> None:
        """Send a greeting trigger to the C++ process."""

    @abstractmethod
    def send_farewell(self) -> None:
        """Send a farewell trigger to the C++ process."""

    @abstractmethod
    def poll_event(self) -> CppEvent | None:
        """Poll for the next event from the C++ process.

        Returns:
            CppEvent if available, None if the event queue is empty.
        """


# ---------------------------------------------------------------------------
# WakewordDetector
# ---------------------------------------------------------------------------


class IWakewordDetector(ABC):
    """Wakeword detection interface.

    Minimal — no start/stop/reset; SessionManager controls when to feed frames.
    """

    @abstractmethod
    def feed_audio(self, frame: AudioFrame) -> bool:
        """Feed an audio frame and check for wakeword detection.

        Args:
            frame: Raw PCM audio bytes for one capture frame.

        Returns:
            True if the wakeword was detected in this frame.
        """


# ---------------------------------------------------------------------------
# LEDController
# ---------------------------------------------------------------------------


class ILEDController(ABC):
    """LED display controller interface.

    Implementations map LEDState values to specific colors/animations.
    """

    @abstractmethod
    def set_state(self, state: LEDState) -> None:
        """Set the LED display to the given state.

        Args:
            state: The desired LED display state.
        """
