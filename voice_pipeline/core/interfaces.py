"""Module interfaces for the voice pipeline.

Only interfaces needed by the next implementation phase are defined here.
New interfaces are added just before their consuming phase begins.

Current: Phase 2 + Phase 3 + Phase 4 + Phase 5 + Phase 6 interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from voice_pipeline.core.types import (
    AudioFrame,
    CppEvent,
    GeneratorState,
    HistoryTurn,
    LEDState,
    LLMMetrics,
    LLMStream,
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

    Write-through model: each mutation is persisted immediately.
    Implementations: memory (testing), sqlite (production).
    """

    @abstractmethod
    def create_session(self, session_id: str, started_at: str) -> None:
        """Create a new session record.

        Args:
            session_id: Unique session identifier.
            started_at: Session start timestamp (UTC, '%Y-%m-%d %H:%M:%S').
        """

    @abstractmethod
    def end_session(self, session_id: str, ended_at: str) -> None:
        """Mark a session as ended.

        Args:
            session_id: Session to end.
            ended_at: Session end timestamp (UTC, '%Y-%m-%d %H:%M:%S').
        """

    @abstractmethod
    def load_session(self, session_id: str) -> list[tuple[int, int, dict[str, Any], int]]:
        """Load all messages for a session.

        Args:
            session_id: Session to load.

        Returns:
            List of (msg_id, turn_id, item, token_count) tuples,
            ordered by msg_id. Empty list if session not found.
        """

    @abstractmethod
    def append_message(
        self,
        session_id: str,
        msg_id: int,
        turn_id: int,
        item: dict[str, Any],
        token_count: int,
        metrics_json: str | None = None,
    ) -> None:
        """Append a message to the session (write-through).

        Args:
            session_id: Target session.
            msg_id: Sequential message identifier within session.
            turn_id: Turn group identifier.
            item: Message dict in Responses API input format.
            token_count: Pre-computed token count for context budgeting.
            metrics_json: JSON-serialized LLMMetrics, or None.
        """

    @abstractmethod
    def update_message(
        self,
        session_id: str,
        msg_id: int,
        item: dict[str, Any],
        token_count: int,
    ) -> None:
        """Update an existing message (e.g. barge-in truncation).

        Args:
            session_id: Target session.
            msg_id: Message to update.
            item: Replacement message dict.
            token_count: Updated token count.
        """

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """Delete all data for a session.

        Args:
            session_id: Session to delete.
        """


# ---------------------------------------------------------------------------
# ConversationHistory
# ---------------------------------------------------------------------------


class IConversationHistory(ABC):
    """Session-scoped conversation history store.

    Write-through: every mutation is persisted immediately via the
    storage backend. In-memory list is authoritative for reads.

    Thread-safe: writes from Orchestrator (main thread), reads from
    SpeechGenerator background thread (via ContextBuilder).

    Message dict schema follows the OpenAI Responses API input format.
    """

    @abstractmethod
    def new_session(self, session_id: str) -> None:
        """Start a new session, clearing any in-memory state.

        Args:
            session_id: Unique identifier for this conversation session.
        """

    @abstractmethod
    def add_user_message(self, text: str) -> int:
        """Append a user message. Auto-assigns turn_id.

        token_count is computed internally via token_counter.

        Args:
            text: The user's transcribed utterance.

        Returns:
            Message ID (msg_id) for later reference.
        """

    @abstractmethod
    def add_assistant_message(self, text: str, metrics: LLMMetrics | None = None) -> int:
        """Append an assistant text message. Auto-assigns turn_id.

        token_count: uses metrics.usage.output_tokens if available,
        falls back to token_counter.

        Args:
            text: The robot's spoken text (full or truncated).
            metrics: LLM call metrics. Stored as metrics_json in DB.

        Returns:
            Message ID (msg_id) for later reference.
        """

    @abstractmethod
    def add_message(
        self,
        item: dict[str, Any],
        turn_id: int | None = None,
        metrics: LLMMetrics | None = None,
    ) -> tuple[int, int]:
        """Append a message in Responses API input format.

        Low-level method for tool call items and other non-standard
        messages. Use add_user_message / add_assistant_message for
        simple text messages.

        Args:
            item: Message dict in Responses API input format.
            turn_id: Turn group ID. None to auto-assign a new turn.
            metrics: LLM call metrics, if this message was LLM-generated.

        Returns:
            Tuple of (msg_id, turn_id).
        """

    @abstractmethod
    def begin_turn(self) -> int:
        """Allocate a new turn_id for grouping multiple messages.

        Used for tool call turns where function_call, function_call_output,
        and assistant text must share the same turn_id.

        Returns:
            The allocated turn_id.
        """

    @abstractmethod
    def update_message(self, msg_id: int, text: str) -> None:
        """Update the text content of a message.

        Used for barge-in truncation correction. token_count is
        recomputed internally via token_counter.

        Args:
            msg_id: Message ID returned by add_* methods.
            text: New content to replace.

        Raises:
            HistoryError: If no message with the given ID exists.
        """

    @abstractmethod
    def get_messages(self) -> list[dict[str, Any]]:
        """Retrieve all messages as a flat list for LLM input.

        Returns:
            Ordered list of message dicts (no internal metadata).
        """

    @abstractmethod
    def get_turns(self) -> list[HistoryTurn]:
        """Retrieve messages grouped by turn for context budgeting.

        Each HistoryTurn is atomic: included or excluded as a whole.

        Returns:
            List of HistoryTurn with pre-computed token_count.
        """

    @abstractmethod
    def save(self) -> None:
        """Finalize the current session (sets ended_at).

        With write-through, individual messages are already persisted.
        """


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
    ) -> LLMStream:
        """Generate a streaming response from the given message history.

        Args:
            messages: List of message dicts in vendor-specific format.
            tools: Tool definitions. None uses config defaults.
                Empty list explicitly disables tools for this call.

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

    Threading: send_*() and poll_event() are called from the Orchestrator
    (main) thread. The implementation runs an internal receiver thread that
    reads WebSocket messages and enqueues CppEvents. poll_event() reads from
    a thread-safe queue. disconnect() must join the receiver thread before
    returning.
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

    @abstractmethod
    def close(self) -> None:
        """Release resources held by the detector."""


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

    @abstractmethod
    def close(self) -> None:
        """Stop animations and release hardware resources."""


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
        """Feed pipeline audio and return voice activity estimates.

        Args:
            user_audio: PCM audio at pipeline rate (16kHz). One or more
                concatenated 30ms frames when batch-draining.
            robot_audio: PCM audio at TTS output sample rate (24kHz).
                Length should match user_audio duration. None when the
                robot is not speaking. Wrapper resamples internally
                to 16kHz.

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
# Similarity
# ---------------------------------------------------------------------------


class ISimilarity(ABC):
    """Text similarity scorer.

    Used by TurnDetector to decide whether a new prepare() is needed
    when ASR text changes.
    """

    @abstractmethod
    def compare(self, a: str, b: str) -> float:
        """Return similarity score between *a* and *b* in [0.0, 1.0]."""


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
        frame_count: int = 1,
    ) -> TurnDecision:
        """Process one pipeline frame and return a turn decision.

        Args:
            user_audio: PCM audio at 16kHz. May be a single 30ms frame
                or multiple concatenated frames when batch-draining.
            asr_text: Current ASR transcription (interim or final).
            robot_audio: PCM chunk at TTS output sample rate (24kHz).
                None when the robot is not speaking.
            frame_count: Number of pipeline frames represented by this
                call. Used to advance internal timers correctly when
                multiple frames are processed in a single call.

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

    Threading: prepare() submits work to a ThreadPoolExecutor. The
    background thread updates internal state (guarded by Lock).
    state, stream_done, poll_audio(), get_text(), get_response_data()
    are polled from the Orchestrator (main) thread. cancel(), reset(),
    and shutdown() are called from the main thread only. Implementations
    must synchronize internal state with threading.Lock.
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

    @property
    @abstractmethod
    def input_text(self) -> str:
        """The user text passed to the most recent prepare() call.

        Set when prepare() is called, cleared on cancel() or reset().
        Used by Orchestrator to record what the LLM actually saw
        when saving to conversation history.
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
    def reset(self) -> None:
        """Cancel any running pipeline and reset state for the next session.

        Does not shut down the executor — the generator can be reused.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Permanently shut down the background executor.

        Call only at program exit. After this, prepare() will fail.
        """


# ---------------------------------------------------------------------------
# AudioInput
# ---------------------------------------------------------------------------


class IAudioInput(ABC):
    """Microphone capture interface.

    Runs on a separate daemon thread, pushing AudioFrame to a shared
    queue (queue.Queue[AudioFrame], injected via constructor).

    Threading: start() and stop() are called from the main thread.
    The capture thread pushes frames to the queue; the Orchestrator
    (main) thread consumes them. The ``error`` property is set by the
    capture thread and read by the main thread (use threading.Event
    or equivalent for safe cross-thread signalling).
    """

    @abstractmethod
    def start(self) -> None:
        """Start capturing audio. Idempotent."""

    @abstractmethod
    def stop(self) -> None:
        """Stop capturing audio and release resources."""

    @property
    @abstractmethod
    def error(self) -> Exception | None:
        """Return the captured error if the capture thread has died.

        Returns None if the thread is running normally.
        """


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
