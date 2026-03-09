"""Module interfaces for the voice pipeline.

Only interfaces needed by the next implementation phase are defined here.
New interfaces are added just before their consuming phase begins.

Current: Phase 2 + Phase 3 + Phase 4 + Phase 5 + Phase 6 interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Literal

from voice_pipeline.core.types import (
    AudioFrame,
    CppEvent,
    GeneratorState,
    LEDState,
    ResponseData,
    TTSStream,
    TurnDecision,
    VAPResult,
    WordTimestamp,
)

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
    def add_user_message(self, text: str) -> int:
        """Append a user message to the current session.

        Args:
            text: The user's transcribed utterance (final ASR result).

        Returns:
            Message ID for later reference (e.g. update_message).
        """

    @abstractmethod
    def add_assistant_message(self, text: str) -> int:
        """Append an assistant message to the current session.

        Called with full response text on normal playback completion,
        or with truncated text on barge-in interruption.

        Args:
            text: The robot's spoken text (full or truncated).

        Returns:
            Message ID for later reference (e.g. update_message).
        """

    @abstractmethod
    def update_message(self, message_id: int, text: str) -> None:
        """Update the content of an existing message by ID.

        Used for barge-in truncation correction: an approximate
        truncation is saved first, then corrected when precise
        data becomes available.

        Args:
            message_id: ID returned by add_user_message/add_assistant_message.
            text: New content to replace the existing message content.

        Raises:
            HistoryError: If no message with the given ID exists.
        """

    @abstractmethod
    def get_messages(self) -> list[dict[str, Any]]:
        """Retrieve all conversation messages.

        Returns:
            List of message dicts in vendor-specific format (no internal IDs).
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
    """Text-to-speech synthesis interface.

    synthesize() returns a TTSStream that yields PCM audio chunks.
    After iteration, .timestamps and .audio are available on the stream.
    Thread-safe: concurrent synthesize() calls are independent.
    """

    @abstractmethod
    def synthesize(self, text: str) -> TTSStream:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize.

        Returns:
            TTSStream yielding PCM audio chunks. Iterate to receive audio.
            After iteration, access .audio, .timestamps, or .result.
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
    def send_stream_start(self) -> None:
        """Signal that audio streaming is about to begin."""

    @abstractmethod
    def send_audio(self, audio: bytes) -> None:
        """Send audio data for playback.

        Args:
            audio: Raw PCM audio bytes.
        """

    @abstractmethod
    def send_audio_end(self) -> None:
        """Signal that all audio data has been sent for the current stream."""

    @abstractmethod
    def send_stop(self) -> None:
        """Send a stop/interrupt signal to halt playback."""

    @abstractmethod
    def send_play_file(self, file_path: str) -> None:
        """Request the C++ process to play an audio file.

        Args:
            file_path: Path to the audio file (relative to C++ working dir).
        """

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


# ---------------------------------------------------------------------------
# VAP (Voice Activity Projection)
# ---------------------------------------------------------------------------


class IVAP(ABC):
    """Voice Activity Projection model wrapper.

    Maintains a rolling stereo audio buffer and runs periodic inference
    to estimate current/future voice activity probabilities.

    Returns cached result when no new inference is due.
    """

    @abstractmethod
    def feed_audio(
        self, user_audio: AudioFrame, robot_audio: AudioFrame | None = None
    ) -> VAPResult:
        """Feed one pipeline frame and return voice activity estimates.

        Args:
            user_audio: 30ms PCM chunk at 16kHz (pipeline AudioConfig rate).
            robot_audio: 30ms PCM chunk at TTS output sample rate (24kHz).
                None when the robot is not speaking. Wrapper resamples
                internally to 16kHz.

        Returns:
            VAPResult with p_now, p_fut, and user_is_speaking.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear the rolling buffer and internal state for a new turn."""


# ---------------------------------------------------------------------------
# TurnGPT
# ---------------------------------------------------------------------------


class ITurnGPT(ABC):
    """TurnGPT model wrapper for text-based turn-shift prediction.

    Accepts ``<ts>``-delimited dialog text and returns a turn-shift
    probability for the current (partial) turn.
    """

    @abstractmethod
    def predict(self, dialog_text: str) -> float:
        """Predict turn-shift probability for the given dialog.

        Args:
            dialog_text: Full conversation text with ``<ts>`` separators
                between completed turns. No trailing ``<ts>`` for the
                current in-progress turn.

        Returns:
            Turn-shift probability in [0, 1].
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for a new conversation."""


# ---------------------------------------------------------------------------
# TurnDetector
# ---------------------------------------------------------------------------


class ITurnDetector(ABC):
    """Combined turn-taking detector.

    Fuses VAP, TurnGPT, and timing heuristics into a single
    per-frame TurnDecision.
    """

    @abstractmethod
    def process_frame(
        self,
        user_audio: AudioFrame,
        asr_text: str,
        robot_audio: AudioFrame | None = None,
    ) -> TurnDecision:
        """Process one pipeline frame and return a turn decision.

        Args:
            user_audio: 30ms PCM chunk at 16kHz.
            asr_text: Current ASR transcription (interim or final).
            robot_audio: 30ms PCM chunk at TTS output sample rate (24kHz).
                None when the robot is not speaking.

        Returns:
            TurnDecision with at most one signal active.
        """

    @abstractmethod
    def notify_turn_complete(self, role: Literal["user", "robot"], text: str) -> None:
        """Inform the detector that a turn was completed.

        Called by Orchestrator after a user or robot turn is finalized.
        Used internally to maintain the ``<ts>``-delimited dialog context
        for TurnGPT predictions. Empty *text* is ignored (no-op).

        Args:
            role: ``"user"`` or ``"robot"``.
            text: Final text of the completed turn. For robot turns
                interrupted by barge-in, this should be the truncated
                text (what was actually spoken).
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset per-frame tracking state for a new turn.

        Clears frame counters, text-stability timers, and prepare flags.
        Does **not** clear the accumulated dialog context used by TurnGPT.
        """


# ---------------------------------------------------------------------------
# SpeechGenerator
# ---------------------------------------------------------------------------


class ISpeechGenerator(ABC):
    """Background speech generation pipeline.

    Chains ContextBuilder -> LLM -> TTS in a background thread.
    Supports cancellation, streaming audio output, and state inspection.

    State flow: IDLE → PREPARING → STREAMING → IDLE (normal)
                PREPARING → FAILED (LLM/TTS error, empty text)
                STREAMING → FAILED (TTS stream error mid-stream)
    """

    @property
    @abstractmethod
    def state(self) -> GeneratorState:
        """Current generator state."""

    @property
    @abstractmethod
    def stream_done(self) -> bool:
        """True when TTS stream is fully consumed.

        Check after poll_audio() returns None to distinguish
        empty queue from stream end.
        """

    @abstractmethod
    def prepare(self, current_text: str) -> None:
        """Start background generation for the given user text.

        If already PREPARING or STREAMING, cancels the current run and restarts.
        If FAILED, discards the previous result and starts fresh.

        Args:
            current_text: Current ASR transcription to generate a
                response for.
        """

    @abstractmethod
    def cancel(self) -> None:
        """Cancel any in-progress or completed preparation.

        Transitions state back to IDLE.
        """

    @abstractmethod
    def poll_audio(self) -> bytes | None:
        """Return next TTS audio chunk, or None if queue is empty.

        Use stream_done property to distinguish empty queue from stream end.
        """

    @abstractmethod
    def get_text(self) -> str:
        """Return the generated response text.

        Available once state is STREAMING.

        Raises:
            RuntimeError: If state is not STREAMING or later.
        """

    @abstractmethod
    def get_response_data(self) -> ResponseData:
        """Return full ResponseData after stream completes.

        Idempotent per run — callable multiple times until next prepare().
        Transitions state to IDLE on first call.

        Raises:
            RuntimeError: If stream is not done.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Shut down the background executor and release resources."""


# ---------------------------------------------------------------------------
# AudioInput
# ---------------------------------------------------------------------------


class IAudioInput(ABC):
    """Microphone capture interface.

    Runs on a separate daemon thread, pushing AudioFrame to a shared queue.
    """

    @abstractmethod
    def start(self) -> None:
        """Start capturing audio. Idempotent."""

    @abstractmethod
    def stop(self) -> None:
        """Stop capturing audio and release resources."""


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------


class ISessionManager(ABC):
    """Top-level state machine interface.

    Manages SLEEP → GREETING → ACTIVE → FAREWELL → SLEEP cycle.
    """

    @abstractmethod
    def run(self) -> None:
        """Run the session manager main loop."""

    @abstractmethod
    def shutdown(self) -> None:
        """Signal the session manager to shut down gracefully."""
