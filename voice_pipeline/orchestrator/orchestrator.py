"""Orchestrator: ACTIVE mode frame-driven conversation loop."""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass

from voice_pipeline.core.config import AudioConfig, OrchestratorConfig, TTSConfig
from voice_pipeline.core.interfaces import (
    IASR,
    IConversationHistory,
    ICppBridge,
    ILEDController,
    ISpeechGenerator,
    ITurnDetector,
    IUtteranceTruncator,
)
from voice_pipeline.core.types import (
    AudioFrame,
    CppEventType,
    GeneratorState,
    LEDState,
    PlaybackState,
    ResponseData,
    TurnDecision,
)
from voice_pipeline.tts.utterance_truncator import DurationRatioTruncator

logger = logging.getLogger("voice_pipeline.orchestrator")


@dataclass
class _PendingTruncation:
    """Deferred truncation state for barge-in during streaming."""

    msg_id: int
    stop_position_sec: float


class Orchestrator:
    """ACTIVE mode conversation loop.

    Coordinates ASR, TurnDetector, SpeechGenerator, CppBridge,
    ConversationHistory, UtteranceTruncator, and LEDController.
    Runs a synchronous frame-driven loop driven by audio_queue.
    """

    def __init__(
        self,
        asr: IASR,
        turn_detector: ITurnDetector,
        speech_generator: ISpeechGenerator,
        cpp_bridge: ICppBridge,
        history: IConversationHistory,
        truncator: IUtteranceTruncator,
        led: ILEDController,
        config: OrchestratorConfig,
        tts_config: TTSConfig,
        audio_config: AudioConfig,
    ) -> None:
        self._asr = asr
        self._turn_detector = turn_detector
        self._generator = speech_generator
        self._bridge = cpp_bridge
        self._history = history
        self._truncator = truncator
        self._led = led
        self._config = config
        self._tts_config = tts_config
        self._audio_config = audio_config

        # External stop signal
        self._stop_event = threading.Event()

        # Internal state
        self._playback_state = PlaybackState.IDLE
        self._awaiting_response = False
        self._current_response: ResponseData | None = None
        self._sent_audio_buffer = bytearray()
        self._saved_user_text = ""
        self._last_asr_text = ""
        self._last_text_change_time = time.monotonic()
        self._playback_start_time = 0.0
        self._stop_pending_time = 0.0
        self._audio_end_sent = False
        self._pending_truncation: _PendingTruncation | None = None
        self._user_msg_id: int | None = None
        self._assistant_msg_id: int | None = None
        self._prepared_text = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, audio_queue: queue.Queue[AudioFrame]) -> None:
        """Run the conversation loop until exit keyword or timeout."""
        self._stop_event.clear()
        self._start_session()
        try:
            while not self._run_frame(audio_queue):
                pass
        finally:
            self._end_session()

    def request_stop(self) -> None:
        """Signal the orchestrator to stop at the next frame."""
        self._stop_event.set()

    def get_robot_audio_chunk(self) -> AudioFrame | None:
        """Extract a 30ms chunk from sent audio buffer at playback position.

        Used by Orchestrator to provide robot_audio to TurnDetector.
        Uses time-based position estimation from playback_started event.
        Returns None if not enough data at the current position.
        """
        if self._playback_state != PlaybackState.PLAYING:
            return None
        if self._playback_start_time == 0.0:
            return None

        sample_rate = self._tts_config.output_sample_rate
        sample_width = 2  # 16-bit PCM
        frame_ms = self._audio_config.frame_duration_ms

        elapsed = time.monotonic() - self._playback_start_time
        if elapsed < 0:
            return None
        frame_bytes = sample_rate * frame_ms * sample_width // 1000
        start = int(elapsed * sample_rate) * sample_width
        end = start + frame_bytes

        if end > len(self._sent_audio_buffer):
            return None
        return bytes(self._sent_audio_buffer[start:end])

    def _get_robot_audio_combined(self, frame_count: int) -> AudioFrame | None:
        """Extract N consecutive frames from sent audio buffer ending at playback position.

        Unlike get_robot_audio_chunk() which returns a single 30ms frame,
        this returns frame_count * 30ms of audio to match batch-drained
        user audio length.

        Returns None if not playing, not enough buffer, or batch_start < 0.
        """
        if self._playback_state != PlaybackState.PLAYING:
            return None
        if self._playback_start_time == 0.0:
            return None

        sample_rate = self._tts_config.output_sample_rate
        sample_width = 2  # 16-bit PCM
        frame_ms = self._audio_config.frame_duration_ms
        frame_bytes = sample_rate * frame_ms * sample_width // 1000

        elapsed = time.monotonic() - self._playback_start_time
        if elapsed < 0:
            return None

        # Sample-aligned current playback position
        current_start = int(elapsed * sample_rate) * sample_width
        # Back-track to cover the entire batch
        batch_start = current_start - frame_bytes * (frame_count - 1)

        if batch_start < 0:
            return None  # Playback just started, buffer doesn't cover full batch
        batch_end = current_start + frame_bytes

        if batch_end > len(self._sent_audio_buffer):
            return None
        return bytes(self._sent_audio_buffer[batch_start:batch_end])

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _start_session(self) -> None:
        # Reset internal state from any previous session
        self._playback_state = PlaybackState.IDLE
        self._awaiting_response = False
        self._current_response = None
        self._sent_audio_buffer = bytearray()
        self._saved_user_text = ""
        self._last_asr_text = ""
        self._playback_start_time = 0.0
        self._stop_pending_time = 0.0
        self._audio_end_sent = False
        self._pending_truncation = None
        self._user_msg_id = None
        self._assistant_msg_id = None
        self._prepared_text = ""

        self._asr.start()
        self._set_led(LEDState.LISTENING)
        self._last_text_change_time = time.monotonic()
        logger.info("Orchestrator session started")

    def _end_session(self) -> None:
        self._asr.stop()
        self._generator.reset()
        self._set_led(LEDState.OFF)
        self._pending_truncation = None
        logger.info("Orchestrator session ended")

    # ------------------------------------------------------------------
    # Frame loop
    # ------------------------------------------------------------------

    def _run_frame(self, audio_queue: queue.Queue[AudioFrame]) -> bool:
        """Process one frame (or batch of frames). Returns True to exit the loop."""
        # 0. Check external stop signal
        if self._stop_event.is_set():
            logger.info("External stop requested — exiting")
            return True

        # 1. Get audio frame + drain any backed-up frames
        frame = self._get_frame(audio_queue)
        if frame is not None:
            frames = [frame]
            self._drain_available_frames(audio_queue, frames)
        else:
            frames = []

        # 2. Feed all frames to ASR
        for f in frames:
            self._feed_asr(f)

        # 3. Get current text
        current_text = self._get_asr_text()

        # 4. Track text changes for timeout
        if current_text != self._last_asr_text:
            self._last_asr_text = current_text
            self._last_text_change_time = time.monotonic()

        # 5. Turn detection (only when we have frames)
        if frames:
            frame_count = len(frames)
            if frame_count > 1:
                logger.debug("Batch-drained %d frames", frame_count)
            # Concatenate all user audio for VAP buffer
            combined_audio: AudioFrame = b"".join(frames)
            robot_audio = (
                self._get_robot_audio_combined(frame_count)
                if frame_count > 1
                else self.get_robot_audio_chunk()
            )
            decision = self._process_turn_detector(
                combined_audio, current_text, robot_audio, frame_count
            )
            if decision is not None:
                if decision.turn_shift:
                    if self._handle_turn_shift(current_text):
                        return True
                elif decision.interrupt:
                    self._handle_interrupt()
                elif decision.prepare:
                    self._handle_prepare(current_text)

        # 6. Poll C++ events
        if self._poll_cpp_events():
            return True  # Bridge error → terminate

        # 7. Drain audio to bridge if PLAYING
        if self._playback_state == PlaybackState.PLAYING:
            self._drain_audio_to_bridge()

        # 8. Check generator completion if awaiting
        if self._awaiting_response:
            self._check_generator_completion()

        # 9. Check deferred truncation
        if self._pending_truncation is not None:
            self._check_deferred_truncation()

        # 10. STOP_PENDING watchdog
        if self._playback_state == PlaybackState.STOP_PENDING:
            self._check_stop_pending_watchdog()

        # 11. Session timeout
        return self._check_session_timeout()

    def _get_frame(self, audio_queue: queue.Queue[AudioFrame]) -> AudioFrame | None:
        try:
            return audio_queue.get(timeout=self._config.frame_timeout_sec)
        except queue.Empty:
            return None

    _MAX_BATCH_FRAMES = 10
    """Upper bound on frames drained per iteration to prevent timer spikes."""

    @staticmethod
    def _drain_available_frames(
        audio_queue: queue.Queue[AudioFrame], frames: list[AudioFrame]
    ) -> None:
        """Non-blocking drain of queued frames into *frames* (capped)."""
        while len(frames) < Orchestrator._MAX_BATCH_FRAMES:
            try:
                frames.append(audio_queue.get_nowait())
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    # ASR helpers
    # ------------------------------------------------------------------

    def _feed_asr(self, frame: AudioFrame) -> None:
        try:
            self._asr.feed_audio(frame)
        except Exception:
            logger.warning("ASR feed_audio error", exc_info=True)

    def _get_asr_text(self) -> str:
        try:
            return self._asr.get_text()
        except Exception:
            logger.warning("ASR get_text error", exc_info=True)
            return ""

    # ------------------------------------------------------------------
    # Turn detection
    # ------------------------------------------------------------------

    def _process_turn_detector(
        self,
        audio: AudioFrame,
        asr_text: str,
        robot_audio: AudioFrame | None,
        frame_count: int = 1,
    ) -> TurnDecision | None:
        try:
            return self._turn_detector.process_frame(audio, asr_text, robot_audio, frame_count)
        except Exception:
            logger.warning("TurnDetector error", exc_info=True)
            return None

    def _handle_turn_shift(self, text: str) -> bool:
        """Handle turn_shift decision. Returns True if session should end."""
        if self._check_exit_keyword(text):
            return True

        if self._generator.state == GeneratorState.STREAMING:
            self._begin_streaming(text)
        else:
            # Not ready yet — set awaiting
            self._awaiting_response = True
            self._saved_user_text = text
            self._set_led(LEDState.THINKING)
            # Start generation if not already preparing
            if self._generator.state == GeneratorState.IDLE:
                self._generator.prepare(text)
                self._prepared_text = text
        return False

    def _handle_prepare(self, text: str) -> None:
        """Cancel current generation and restart with new text."""
        if self._awaiting_response and self._saved_user_text:
            combined = self._saved_user_text + " " + text
        else:
            combined = text
        self._pending_truncation = None
        self._generator.prepare(combined)
        self._prepared_text = combined

    def _handle_interrupt(self) -> None:
        """Handle interrupt signal from TurnDetector."""
        if self._playback_state == PlaybackState.PLAYING:
            try:
                self._bridge.send_stop()
            except Exception:
                logger.warning("Bridge send_stop error", exc_info=True)
                # Treat as bridge failure — will be caught by poll
            self._playback_state = PlaybackState.STOP_PENDING
            self._stop_pending_time = time.monotonic()
        elif self._awaiting_response:
            self._generator.cancel()
            self._turn_detector.reset()
            self._awaiting_response = False
            self._saved_user_text = ""
            self._prepared_text = ""
            self._audio_end_sent = False
            self._set_led(LEDState.LISTENING)

    # ------------------------------------------------------------------
    # Streaming / playback
    # ------------------------------------------------------------------

    def _begin_streaming(self, text: str) -> None:
        """Start sending audio to bridge. Save user message to history."""
        # Use prepared text for history — matches what LLM actually saw
        final_text = self._prepared_text if self._prepared_text else text

        self._user_msg_id = self._history.add_user_message(final_text)
        self._turn_detector.notify_turn_complete("user", final_text)

        self._asr.reset()
        self._last_asr_text = ""

        self._awaiting_response = False
        self._saved_user_text = ""
        self._prepared_text = ""
        self._current_response = None
        self._sent_audio_buffer = bytearray()
        self._playback_start_time = 0.0
        self._audio_end_sent = False
        self._pending_truncation = None

        self._bridge.send_stream_start()
        self._drain_audio_to_bridge()
        self._playback_state = PlaybackState.PLAYING
        self._set_led(LEDState.SPEAKING)

    def _drain_audio_to_bridge(self) -> None:
        """Poll audio chunks from generator and send to bridge."""
        while True:
            chunk = self._generator.poll_audio()
            if chunk is None:
                break
            try:
                self._bridge.send_audio(chunk)
            except Exception:
                logger.warning("Bridge send_audio error", exc_info=True)
                return
            self._sent_audio_buffer.extend(chunk)

        # Check if stream is done
        if self._generator.stream_done and self._current_response is None:
            try:
                self._current_response = self._generator.get_response_data()
            except RuntimeError:
                logger.warning("Failed to get response data", exc_info=True)
            if not self._audio_end_sent:
                self._audio_end_sent = True
                try:
                    self._bridge.send_audio_end()
                except Exception:
                    logger.warning("Bridge send_audio_end error", exc_info=True)

    # ------------------------------------------------------------------
    # C++ events
    # ------------------------------------------------------------------

    def _poll_cpp_events(self) -> bool:
        """Poll and handle C++ events. Returns True on bridge error (terminate)."""
        try:
            while True:
                event = self._bridge.poll_event()
                if event is None:
                    break

                if event.event_type == CppEventType.PLAYBACK_STARTED:
                    if self._playback_state == PlaybackState.PLAYING:
                        self._playback_start_time = time.monotonic()

                elif event.event_type == CppEventType.PLAYBACK_COMPLETE:
                    if self._playback_state == PlaybackState.PLAYING:
                        self._on_playback_complete()
                    elif self._playback_state == PlaybackState.STOP_PENDING:
                        self._on_playback_interrupted()

        except Exception:
            logger.error("CppBridge error — terminating session", exc_info=True)
            return True
        return False

    def _on_playback_complete(self) -> None:
        """Normal playback completion — save full response to history."""
        text = self._get_response_text()
        if text:
            self._assistant_msg_id = self._history.add_assistant_message(text)
            self._turn_detector.notify_turn_complete("robot", text)

        self._turn_detector.reset()
        self._reset_playback_state()

    def _on_playback_interrupted(self) -> None:
        """Barge-in: truncate and save what was actually spoken.

        Stop position is estimated from the time between playback_started
        and stop being sent (time-based estimation).
        """
        if self._playback_start_time > 0.0:
            stop_pos = max(0.0, self._stop_pending_time - self._playback_start_time)
        else:
            stop_pos = 0.0
        text = self._get_response_text()

        if not text:
            self._turn_detector.reset()
            self._reset_playback_state()
            return

        if self._current_response is not None:
            # Case A or B: ResponseData available
            if self._current_response.has_timestamps:
                # Case A: precise timestamps
                truncated = self._truncator.truncate(
                    text, stop_pos, self._current_response.timestamps
                )
            else:
                # Case B: duration ratio
                total_dur = len(self._current_response.audio) / (
                    self._tts_config.output_sample_rate * 2
                )
                ratio_truncator = DurationRatioTruncator(total_dur)
                truncated = ratio_truncator.truncate(text, stop_pos, [])

            if truncated:
                self._assistant_msg_id = self._history.add_assistant_message(truncated)
                self._turn_detector.notify_turn_complete("robot", truncated)
        else:
            # Case C: stream not done — approximate now, correct later
            total_dur = len(self._sent_audio_buffer) / (self._tts_config.output_sample_rate * 2)
            ratio_truncator = DurationRatioTruncator(total_dur)
            truncated = ratio_truncator.truncate(text, stop_pos, [])

            if truncated:
                msg_id = self._history.add_assistant_message(truncated)
                self._assistant_msg_id = msg_id
                self._turn_detector.notify_turn_complete("robot", truncated)
                self._pending_truncation = _PendingTruncation(
                    msg_id=msg_id,
                    stop_position_sec=stop_pos,
                )

        self._turn_detector.reset()
        self._reset_playback_state()

    # ------------------------------------------------------------------
    # Deferred truncation
    # ------------------------------------------------------------------

    def _check_deferred_truncation(self) -> None:
        """Check if generator finished so we can correct the approximate truncation."""
        pending = self._pending_truncation
        if pending is None:
            return

        if self._generator.state == GeneratorState.FAILED:
            # Keep approximate truncation
            self._pending_truncation = None
            return

        if not self._generator.stream_done:
            return

        # Stream done — get precise data
        try:
            response_data = self._generator.get_response_data()
        except RuntimeError:
            self._pending_truncation = None
            return

        if response_data.has_timestamps:
            corrected = self._truncator.truncate(
                response_data.text,
                pending.stop_position_sec,
                response_data.timestamps,
            )
        else:
            total_dur = len(response_data.audio) / (self._tts_config.output_sample_rate * 2)
            ratio_truncator = DurationRatioTruncator(total_dur)
            corrected = ratio_truncator.truncate(response_data.text, pending.stop_position_sec, [])

        if corrected:
            try:
                self._history.update_message(pending.msg_id, corrected)
            except Exception:
                logger.warning("Failed to update message for deferred truncation", exc_info=True)

        self._pending_truncation = None

    # ------------------------------------------------------------------
    # Generator completion (awaiting_response)
    # ------------------------------------------------------------------

    def _check_generator_completion(self) -> None:
        """Check if the generator is ready while awaiting_response."""
        state = self._generator.state
        if state == GeneratorState.STREAMING:
            # Pass empty text — _begin_streaming uses _saved_user_text when awaiting
            self._begin_streaming("")
        elif state == GeneratorState.FAILED:
            logger.warning("Generator failed while awaiting — skipping turn")
            self._generator.reset()
            self._awaiting_response = False
            self._saved_user_text = ""
            self._prepared_text = ""
            self._turn_detector.reset()
            self._set_led(LEDState.LISTENING)

    # ------------------------------------------------------------------
    # STOP_PENDING watchdog
    # ------------------------------------------------------------------

    def _check_stop_pending_watchdog(self) -> None:
        elapsed = time.monotonic() - self._stop_pending_time
        if elapsed >= self._config.stop_pending_timeout_sec:
            logger.warning("STOP_PENDING watchdog timeout — forcing IDLE")
            self._reset_playback_state()

    # ------------------------------------------------------------------
    # Session timeout
    # ------------------------------------------------------------------

    def _check_session_timeout(self) -> bool:
        """Check for session timeout. Paused during PLAYING or awaiting."""
        if self._playback_state == PlaybackState.PLAYING or self._awaiting_response:
            self._last_text_change_time = time.monotonic()
            return False

        elapsed = time.monotonic() - self._last_text_change_time
        if elapsed >= self._config.session_timeout_sec:
            logger.info("Session timeout — exiting")
            return True
        return False

    # ------------------------------------------------------------------
    # Exit keyword
    # ------------------------------------------------------------------

    def _check_exit_keyword(self, text: str) -> bool:
        """Check if text contains an exit keyword (word boundary, case-insensitive)."""
        if not text:
            return False
        # Strip punctuation for matching
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        words = set(cleaned.split())
        return any(kw.lower() in words for kw in self._config.exit_keywords)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_response_text(self) -> str:
        """Get the current response text from generator or current_response."""
        if self._current_response is not None:
            return self._current_response.text
        try:
            return self._generator.get_text()
        except RuntimeError:
            return ""

    def _reset_playback_state(self) -> None:
        """Reset to IDLE after playback ends (complete or interrupted)."""
        self._playback_state = PlaybackState.IDLE
        self._current_response = None
        self._sent_audio_buffer = bytearray()
        self._playback_start_time = 0.0
        self._audio_end_sent = False
        self._set_led(LEDState.LISTENING)

    def _set_led(self, state: LEDState) -> None:
        try:
            self._led.set_state(state)
        except Exception:
            logger.warning("LED set_state error", exc_info=True)
