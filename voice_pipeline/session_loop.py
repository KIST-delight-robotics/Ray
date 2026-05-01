"""SessionLoop: ACTIVE mode frame-driven conversation loop."""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from voice_pipeline.audio.constants import FRAME_DURATION_MS
from voice_pipeline.core.interfaces import (
    IASR,
    IConversationHistory,
    ICppBridge,
    ILEDController,
    IMemoryStorage,
    ISpeechGenerator,
    ITurnDetector,
)
from voice_pipeline.core.types import (
    AudioFrame,
    CppEventType,
    GeneratorState,
    LEDState,
    PlaybackState,
    ResponseData,
    TokenCounter,
    TurnDecision,
)
from voice_pipeline.tts.utterance_truncator import truncate_by_ratio, truncate_by_timestamps

logger = logging.getLogger("voice_pipeline.session_loop")


@dataclass
class SessionComponents:
    """Per-session objects created by the session factory."""

    session_loop: SessionLoop
    history: IConversationHistory
    session_id: str


@dataclass
class _PendingTruncation:
    """Deferred truncation state for barge-in during streaming."""

    msg_id: int
    stop_position_sec: float


class SessionLoop:
    """ACTIVE mode conversation loop.

    Coordinates ASR, TurnDetector, SpeechGenerator, CppBridge,
    ConversationHistory, UtteranceTruncator, and LEDController.
    Runs a synchronous frame-driven loop driven by audio_queue.
    """

    _EXIT_KEYWORDS: tuple[str, ...] = ("bye", "goodbye")  # 대화 종료 트리거 단어 (대소문자 무시)
    _SESSION_TIMEOUT_SEC = 60.0  # 비활성 세션 자동 종료 시간 (초)
    _FRAME_TIMEOUT_SEC = 0.1  # 프레임 대기 timeout (초). ACTIVE 모드 audio_queue.get
    _STOP_PENDING_TIMEOUT_SEC = 5.0  # barge-in stop 후 재생 완료 ack 최대 대기 (초)
    _AUDIO_STARVATION_TIMEOUT_SEC = 5.0  # 오디오 프레임 단절 감지 timeout — 세션 종료 (초)
    _AWAITING_CANCEL_GRACE_SEC = 0.5  # turn_shift 직후 ASR finalization noise 무시 grace (초)
    _MAX_BATCH_FRAMES = 10  # 한 iteration 최대 drain 프레임 수. timer spike 방지

    def __init__(
        self,
        asr: IASR,
        turn_detector: ITurnDetector,
        speech_generator: ISpeechGenerator,
        cpp_bridge: ICppBridge,
        history: IConversationHistory,
        led: ILEDController,
        audio_queue: queue.Queue[AudioFrame],
        tts_sample_rate: int,
        memory_storage: IMemoryStorage | None = None,
        session_id: str | None = None,
        token_counter: TokenCounter | None = None,
        trace_store: object | None = None,
        shutdown_event: threading.Event | None = None,
    ) -> None:
        """Initialize the SessionLoop.

        Args:
            asr: ASR 인터페이스. 세션마다 start/stop, 프레임 피드·텍스트 조회.
            turn_detector: VAP+TurnGPT 기반 turn decision 제공자.
            speech_generator: ContextBuilder→LLM→TTS 오케스트레이션.
            cpp_bridge: C++ 오디오 재생 프로세스와의 WebSocket 브릿지.
            history: 세션 대화 이력. 세션마다 새로 생성됨.
            led: LED 상태 컨트롤러.
            audio_queue: AudioInput이 push하는 프레임 공유 큐 (ACTIVE 모드
                소비자).
            tts_sample_rate: TTS 출력 샘플레이트 (Hz). robot_audio 버퍼
                시간 기반 위치 추정에 사용.
            memory_storage: 장기 메모리 스토리지. ``None``이면 utterance
                저장 skip.
            session_id: 현재 세션 ID. memory utterance attach용.
                ``None``이면 utterance 저장 skip.
            token_counter: 토큰 카운터 콜러블. memory utterance 토큰 수
                계산용. ``None``이면 0.
            trace_store: pipeline latency trace store. ``None``이면
                trace 저장 skip.
            shutdown_event: 프로세스 전역 종료 시그널. ``None``이면
                ``request_stop()``으로만 중단 가능.
        """
        self._asr = asr
        self._turn_detector = turn_detector
        self._generator = speech_generator
        self._bridge = cpp_bridge
        self._history = history
        self._led = led
        self._audio_queue = audio_queue
        self._tts_sample_rate = tts_sample_rate
        self._memory_storage = memory_storage
        self._session_id = session_id
        self._token_counter = token_counter
        self._trace_store = trace_store
        self._shutdown_event = shutdown_event

        # External stop signal
        self._stop_event = threading.Event()

        # Internal state
        self._playback_state = PlaybackState.IDLE
        self._awaiting_response = False
        self._current_response: ResponseData | None = None
        self._sent_audio_buffer = bytearray()
        self._last_asr_text = ""
        self._last_text_change_time = time.monotonic()
        self._playback_start_time = 0.0
        self._stop_pending_time = 0.0
        self._audio_end_sent = False
        self._pending_truncation: _PendingTruncation | None = None
        self._user_msg_id: int | None = None
        self._assistant_msg_id: int | None = None
        self._last_frame_time = 0.0
        self._turn_shift_time = 0.0
        self._begin_streaming_time = 0.0
        self._speculative_attempts = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the conversation loop until exit keyword or timeout."""
        self._stop_event.clear()
        self._start_session()
        try:
            while not self._run_frame():
                pass
        finally:
            self._end_session()

    def request_stop(self) -> None:
        """Signal the session loop to stop at the next frame."""
        self._stop_event.set()

    def get_robot_audio_chunk(self) -> AudioFrame | None:
        """Extract a 30ms chunk from sent audio buffer at playback position.

        Used by SessionLoop to provide robot_audio to TurnDetector.
        Uses time-based position estimation from playback_started event.
        Returns None if not enough data at the current position.
        """
        if self._playback_state != PlaybackState.PLAYING:
            return None
        if self._playback_start_time == 0.0:
            return None

        sample_rate = self._tts_sample_rate
        sample_width = 2  # 16-bit PCM
        frame_ms = FRAME_DURATION_MS

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

        sample_rate = self._tts_sample_rate
        sample_width = 2  # 16-bit PCM
        frame_ms = FRAME_DURATION_MS
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
        self._last_asr_text = ""
        self._playback_start_time = 0.0
        self._stop_pending_time = 0.0
        self._audio_end_sent = False
        self._pending_truncation = None
        self._user_msg_id = None
        self._assistant_msg_id = None
        self._turn_shift_time = 0.0
        self._begin_streaming_time = 0.0
        self._speculative_attempts = 0

        self._asr.start()
        self._set_led(LEDState.IDLE)
        now = time.monotonic()
        self._last_text_change_time = now
        self._last_frame_time = now
        logger.info("SessionLoop started")

    def _end_session(self) -> None:
        self._asr.stop()
        self._generator.reset()
        self._set_led(LEDState.OFF)
        self._pending_truncation = None
        logger.info("SessionLoop ended")

    # ------------------------------------------------------------------
    # Frame loop
    # ------------------------------------------------------------------

    def _run_frame(self) -> bool:
        """Process one frame (or batch of frames). Returns True to exit the loop."""
        # 0. Check stop / shutdown signals
        if self._stop_event.is_set():
            logger.info("External stop requested — exiting")
            return True
        if self._shutdown_event is not None and self._shutdown_event.is_set():
            logger.info("Shutdown signal — exiting")
            return True

        # 1. Get audio frame + drain any backed-up frames
        frame = self._get_frame()
        if frame is not None:
            frames = [frame]
            self._drain_available_frames(frames)
            self._last_frame_time = time.monotonic()
        else:
            frames = []

        # 2. Feed all frames to ASR
        for f in frames:
            self._feed_asr(f)

        # 3. Get current text
        current_text = self._get_asr_text()

        # 4. Track text changes for timeout
        text_changed = current_text != self._last_asr_text
        if text_changed:
            self._last_asr_text = current_text
            self._last_text_change_time = time.monotonic()

        # 5. Cancel awaiting if user continues speaking after turn_shift
        if (
            self._awaiting_response
            and text_changed
            and time.monotonic() - self._turn_shift_time > self._AWAITING_CANCEL_GRACE_SEC
        ):
            logger.info("User continued speaking during awaiting — cancelling generation")
            self._save_trace("cancelled")
            self._generator.cancel()
            self._turn_detector.reset()
            self._awaiting_response = False
            self._audio_end_sent = False
            self._pending_truncation = None

        # 6. Turn detection (only when we have frames)
        if frames:
            frame_count = len(frames)
            if frame_count > 1:
                logger.debug("Batch-drained %d frames", frame_count)
            # Concatenate all user audio for VAP buffer
            combined_audio: AudioFrame = b"".join(frames)
            robot_audio = (
                self._get_robot_audio_combined(frame_count) if frame_count > 1 else self.get_robot_audio_chunk()
            )
            decision = self._process_turn_detector(combined_audio, current_text, robot_audio, frame_count)
            if decision is not None:
                if decision.turn_shift:
                    if self._handle_turn_shift(current_text):
                        return True
                elif decision.interrupt:
                    self._handle_interrupt()
                elif decision.prepare:
                    self._handle_prepare(current_text)

        # 7. Poll C++ events
        if self._poll_cpp_events():
            return True  # Bridge error → terminate

        # 8. Drain audio to bridge if PLAYING
        if self._playback_state == PlaybackState.PLAYING:
            self._drain_audio_to_bridge()

        # 9. Check generator completion if awaiting
        if self._awaiting_response:
            self._check_generator_completion()

        # 10. Check deferred truncation
        if self._pending_truncation is not None:
            self._check_deferred_truncation()

        # 11. STOP_PENDING watchdog
        if self._playback_state == PlaybackState.STOP_PENDING:
            self._check_stop_pending_watchdog()

        # 12. Audio starvation check
        if self._check_audio_starvation():
            return True

        # 13. Session timeout
        return self._check_session_timeout()

    def _get_frame(self) -> AudioFrame | None:
        try:
            return self._audio_queue.get(timeout=self._FRAME_TIMEOUT_SEC)
        except queue.Empty:
            return None

    def _drain_available_frames(self, frames: list[AudioFrame]) -> None:
        """Non-blocking drain of queued frames into *frames* (capped)."""
        while len(frames) < self._MAX_BATCH_FRAMES:
            try:
                frames.append(self._audio_queue.get_nowait())
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

        self._turn_shift_time = time.monotonic()
        if self._generator.state == GeneratorState.STREAMING:
            self._begin_streaming()
        else:
            # Not ready yet — set awaiting
            self._awaiting_response = True
            # Start generation if not already preparing
            if self._generator.state == GeneratorState.IDLE:
                self._speculative_attempts += 1
                self._generator.prepare(text)
        return False

    def _handle_prepare(self, text: str) -> None:
        """Start speculative generation with current text."""
        if self._awaiting_response:
            return
        self._speculative_attempts += 1
        self._pending_truncation = None
        self._generator.prepare(text)

    def _handle_interrupt(self) -> None:
        """Handle interrupt signal from TurnDetector."""
        if self._playback_state == PlaybackState.PLAYING:
            logger.info("Interrupt → send_stop (playback PLAYING)")
            try:
                self._bridge.send_stop()
            except Exception:
                logger.warning("Bridge send_stop error", exc_info=True)
                # Treat as bridge failure — will be caught by poll
            self._playback_state = PlaybackState.STOP_PENDING
            now = time.monotonic()
            self._stop_pending_time = now
            trace = self._generator.trace
            if trace is not None:
                trace.interrupt_ts = now
        elif self._awaiting_response:
            logger.info("Interrupt → cancel generator (awaiting_response)")
            self._save_trace("cancelled")
            self._generator.cancel()
            self._turn_detector.reset()
            self._awaiting_response = False
            self._audio_end_sent = False
        else:
            logger.debug("Interrupt ignored (state=%s)", self._playback_state.value)

    # ------------------------------------------------------------------
    # Streaming / playback
    # ------------------------------------------------------------------

    def _begin_streaming(self) -> None:
        """Start sending audio to bridge. Save user message to history."""
        user_text = self._generator.input_text
        if not user_text:
            logger.warning("_begin_streaming called with empty input_text — skipping")
            self._generator.reset()
            self._awaiting_response = False
            return

        self._user_msg_id = self._history.add_user_message(user_text)
        self._save_utterance("user", user_text)
        self._turn_detector.notify_turn_complete("user", user_text)

        self._asr.reset()
        self._last_asr_text = ""

        self._awaiting_response = False
        self._current_response = None
        self._sent_audio_buffer = bytearray()
        self._playback_start_time = 0.0
        self._audio_end_sent = False
        self._pending_truncation = None

        self._begin_streaming_time = time.monotonic()
        self._bridge.send_stream_start()
        self._drain_audio_to_bridge()
        self._playback_state = PlaybackState.PLAYING
        buf_sec = len(self._sent_audio_buffer) / (self._tts_sample_rate * 2)
        logger.info("begin_streaming: %.1fs audio buffered → PLAYING", buf_sec)

        trace = self._generator.trace
        if trace is not None:
            trace.session_id = self._session_id or ""
            trace.turn_shift_ts = self._turn_shift_time
            trace.begin_streaming_ts = self._begin_streaming_time

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
                        trace = self._generator.trace
                        if trace is not None:
                            trace.playback_started_ts = self._playback_start_time

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
        logger.info("Playback complete (normal)")
        text = self._get_response_text()
        if text:
            # Pass LLM metrics from the last LLM call (if available)
            metrics = None
            if self._current_response and self._current_response.metrics_list:
                metrics = self._current_response.metrics_list[-1]
            self._assistant_msg_id = self._history.add_assistant_message(text, metrics)
            self._save_utterance("assistant", text)
            self._turn_detector.notify_turn_complete("robot", text)

        self._save_trace("completed")
        self._turn_detector.reset()
        self._reset_playback_state()

    def _on_playback_interrupted(self) -> None:
        """Barge-in: truncate and save what was actually spoken.

        Stop position is estimated from the time between playback_started
        and stop being sent (time-based estimation).
        """
        now = time.monotonic()
        trace = self._generator.trace
        if trace is not None:
            trace.interrupt_ack_ts = now
        if self._playback_start_time > 0.0:
            stop_pos = max(0.0, self._stop_pending_time - self._playback_start_time)
        else:
            stop_pos = 0.0
        text = self._get_response_text()
        logger.info("Playback interrupted (barge-in): stop_pos=%.2fs full=%r", stop_pos, text)

        if not text:
            self._turn_detector.reset()
            self._reset_playback_state()
            return

        if self._current_response is not None:
            # Case A or B: ResponseData available
            if self._current_response.has_timestamps:
                truncated = truncate_by_timestamps(text, stop_pos, self._current_response.timestamps)
                trunc_method = "timestamps"
            else:
                total_dur = len(self._current_response.audio) / (self._tts_sample_rate * 2)
                truncated = truncate_by_ratio(text, stop_pos, total_dur)
                trunc_method = "ratio"

            if truncated:
                logger.info("Truncated (%s): %r", trunc_method, truncated)
                self._assistant_msg_id = self._history.add_assistant_message(truncated)
                self._save_utterance("assistant", truncated)
                self._turn_detector.notify_turn_complete("robot", truncated)
        else:
            # Case C: stream not done — approximate now, correct later
            total_dur = len(self._sent_audio_buffer) / (self._tts_sample_rate * 2)
            truncated = truncate_by_ratio(text, stop_pos, total_dur)
            trunc_method = "ratio-pending"

            if truncated:
                logger.info("Truncated (%s): %r", trunc_method, truncated)
                msg_id = self._history.add_assistant_message(truncated)
                self._assistant_msg_id = msg_id
                self._save_utterance("assistant", truncated)
                self._turn_detector.notify_turn_complete("robot", truncated)
                self._pending_truncation = _PendingTruncation(
                    msg_id=msg_id,
                    stop_position_sec=stop_pos,
                )

        self._save_trace("truncated")
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
            corrected = truncate_by_timestamps(
                response_data.text,
                pending.stop_position_sec,
                response_data.timestamps,
            )
        else:
            total_dur = len(response_data.audio) / (self._tts_sample_rate * 2)
            corrected = truncate_by_ratio(response_data.text, pending.stop_position_sec, total_dur)

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
            self._begin_streaming()
        elif state == GeneratorState.FAILED:
            logger.warning("Generator failed while awaiting — skipping turn")
            self._save_trace("cancelled")
            self._generator.reset()
            self._awaiting_response = False
            self._turn_detector.reset()

    # ------------------------------------------------------------------
    # STOP_PENDING watchdog
    # ------------------------------------------------------------------

    def _check_stop_pending_watchdog(self) -> None:
        elapsed = time.monotonic() - self._stop_pending_time
        if elapsed >= self._STOP_PENDING_TIMEOUT_SEC:
            logger.warning("STOP_PENDING watchdog timeout — forcing IDLE")
            self._turn_detector.reset()
            self._reset_playback_state()

    # ------------------------------------------------------------------
    # Audio starvation
    # ------------------------------------------------------------------

    def _check_audio_starvation(self) -> bool:
        """Terminate if no audio frames arrived for too long."""
        elapsed = time.monotonic() - self._last_frame_time
        if elapsed >= self._AUDIO_STARVATION_TIMEOUT_SEC:
            logger.error(
                "Audio starvation (%.1fs without frames) — terminating session",
                elapsed,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Session timeout
    # ------------------------------------------------------------------

    def _check_session_timeout(self) -> bool:
        """Check for session timeout. Paused during PLAYING or awaiting."""
        if self._playback_state == PlaybackState.PLAYING or self._awaiting_response:
            self._last_text_change_time = time.monotonic()
            return False

        elapsed = time.monotonic() - self._last_text_change_time
        if elapsed >= self._SESSION_TIMEOUT_SEC:
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
        return any(kw.lower() in words for kw in self._EXIT_KEYWORDS)

    # ------------------------------------------------------------------
    # Utterance storage (memory system)
    # ------------------------------------------------------------------

    def _save_utterance(self, role: str, text: str) -> None:
        """Store an utterance for later memory extraction."""
        if self._memory_storage is None or self._session_id is None:
            return
        if not text:
            return
        try:
            timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
            token_count = self._token_counter(text) if self._token_counter else 0
            self._memory_storage.add_utterance(self._session_id, role, text, timestamp, token_count)
        except Exception:
            logger.warning("Failed to save %s utterance", role, exc_info=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_trace(self, outcome: str) -> None:
        """Persist the current pipeline trace with the given outcome."""
        if self._trace_store is None:
            return
        trace = self._generator.trace
        if trace is None:
            return
        trace.outcome = outcome
        trace.speculative_attempts = self._speculative_attempts
        if self._user_msg_id is not None:
            trace.user_msg_id = self._user_msg_id
        if not trace.session_id:
            trace.session_id = self._session_id or ""
        try:
            self._trace_store.save(trace)
            if outcome != "cancelled":
                logger.info("Pipeline trace: %s", trace.summary())
        except Exception:
            logger.warning("Failed to save pipeline trace", exc_info=True)
        self._speculative_attempts = 0

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
        self._begin_streaming_time = 0.0
        self._audio_end_sent = False

    def _set_led(self, state: LEDState) -> None:
        try:
            self._led.set_state(state)
        except Exception:
            logger.warning("LED set_state error", exc_info=True)
