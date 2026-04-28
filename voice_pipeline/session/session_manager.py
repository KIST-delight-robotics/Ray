"""SessionManager: top-level state machine for the voice pipeline."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from voice_pipeline.core.interfaces import (
    IAudioInput,
    IConversationHistory,
    ICppBridge,
    ILEDController,
    ISessionManager,
    IWakewordDetector,
)
from voice_pipeline.core.types import AudioFrame, CppEventType, LEDState, SystemMode
from voice_pipeline.orchestrator.orchestrator import Orchestrator

logger = logging.getLogger("voice_pipeline.session")

_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass
class SessionComponents:
    """Per-session objects created by the session factory."""

    orchestrator: Orchestrator
    history: IConversationHistory
    session_id: str


class SessionManager(ISessionManager):
    """Top-level state machine: SLEEP → GREETING → ACTIVE → FAREWELL → SLEEP.

    Uses a session factory to create fresh per-session components,
    ensuring clean state isolation between conversations.
    """

    AUDIO_QUEUE_SIZE = 300  # 오디오 프레임 공유 큐 최대 크기 (30ms frame 기준 약 9초 buffer)
    _GREETING_TIMEOUT_SEC = 10.0  # greeting 재생 완료 최대 대기 시간 (초)
    _FAREWELL_TIMEOUT_SEC = 10.0  # farewell 재생 완료 최대 대기 시간 (초)
    _FRAME_TIMEOUT_SEC = 0.1  # SLEEP 모드 프레임 대기 timeout (초)
    _POLL_INTERVAL_SEC = 0.05  # greeting/farewell 재생 대기 polling 주기 (초)

    def __init__(
        self,
        audio_input: IAudioInput,
        wakeword: IWakewordDetector,
        session_factory: Callable[[], SessionComponents],
        cpp_bridge: ICppBridge,
        led: ILEDController,
        greeting_audio_path: str,
        farewell_audio_path: str,
        audio_queue: queue.Queue[AudioFrame] | None = None,
        on_session_end: Callable[[str, str], None] | None = None,
    ) -> None:
        """Initialize the session manager.

        Args:
            audio_input: 마이크 캡처 스레드 (``IAudioInput``).
            wakeword: 웨이크워드 감지기 (``IWakewordDetector``).
            session_factory: 세션 진입마다 호출되어 ``SessionComponents``
                (Orchestrator + ConversationHistory + session_id)를 새로 반환하는 팩토리.
            cpp_bridge: C++ 오디오 재생 프로세스 브릿지 (``ICppBridge``).
            led: LED 컨트롤러 (``ILEDController``).
            greeting_audio_path: greeting 오디오 파일 경로.
            farewell_audio_path: farewell 오디오 파일 경로.
            audio_queue: AudioInput과 공유하는 프레임 큐. ``None``이면
                ``AUDIO_QUEUE_SIZE`` 크기로 내부 생성.
            on_session_end: 세션 종료 시 호출되는 콜백
                ``(session_id, started_at) -> None``. ``None``이면 콜백 미실행.
        """
        self._audio_input = audio_input
        self._wakeword = wakeword
        self._session_factory = session_factory
        self._bridge = cpp_bridge
        self._led = led
        self._greeting_audio_path = greeting_audio_path
        self._farewell_audio_path = farewell_audio_path
        self._on_session_end = on_session_end

        self._audio_queue: queue.Queue[AudioFrame] = audio_queue or queue.Queue(maxsize=self.AUDIO_QUEUE_SIZE)
        self._shutdown_event = threading.Event()
        self._mode = SystemMode.SLEEP
        self._session_started = False

        self._session_lock = threading.Lock()
        self._current_orchestrator: Orchestrator | None = None
        self._current_history: IConversationHistory | None = None
        self._current_session_id: str | None = None
        self._session_started_at: str | None = None

    @property
    def audio_queue(self) -> queue.Queue[AudioFrame]:
        """The bounded audio queue shared with AudioInput."""
        return self._audio_queue

    def run(self) -> None:
        """Run the session manager main loop."""
        logger.info("SessionManager starting")
        self._bridge.connect()
        self._audio_input.start()

        try:
            while not self._shutdown_event.is_set():
                if self._mode == SystemMode.SLEEP:
                    self._run_sleep()
                elif self._mode == SystemMode.GREETING:
                    self._run_greeting()
                elif self._mode == SystemMode.ACTIVE:
                    self._run_active()
                elif self._mode == SystemMode.FAREWELL:
                    self._run_farewell()
        finally:
            self._audio_input.stop()
            self._bridge.disconnect()
            logger.info("SessionManager stopped")

    def shutdown(self) -> None:
        """Signal the session manager to shut down gracefully."""
        self._shutdown_event.set()
        with self._session_lock:
            if self._current_orchestrator is not None:
                self._current_orchestrator.request_stop()
            if self._session_started and self._current_history is not None:
                try:
                    self._current_history.save()
                except Exception:
                    logger.warning("History save error on shutdown", exc_info=True)
                self._trigger_session_end()
                self._session_started = False

    # ------------------------------------------------------------------
    # Mode implementations
    # ------------------------------------------------------------------

    def _run_sleep(self) -> None:
        """SLEEP mode: listen for wakeword."""
        self._led.set_state(LEDState.SLEEPING)

        while not self._shutdown_event.is_set():
            try:
                frame = self._audio_queue.get(timeout=self._FRAME_TIMEOUT_SEC)
            except queue.Empty:
                if self._audio_input.error is not None:
                    logger.error("Audio capture thread died: %s", self._audio_input.error)
                    raise self._audio_input.error from None
                continue

            if self._wakeword.feed_audio(frame):
                logger.info("Wakeword detected — transitioning to GREETING")
                self._mode = SystemMode.GREETING
                return

    def _run_greeting(self) -> None:
        """GREETING mode: send greeting, wait for playback completion."""
        try:
            self._bridge.connect()
        except Exception:
            logger.error("Bridge connect failed — returning to SLEEP", exc_info=True)
            self._mode = SystemMode.SLEEP
            return

        self._flush_bridge_events()
        self._led.set_state(LEDState.IDLE)

        try:
            self._bridge.send_play_file(self._greeting_audio_path)
        except Exception:
            logger.warning("Failed to send greeting", exc_info=True)

        deadline = time.monotonic() + self._GREETING_TIMEOUT_SEC
        while not self._shutdown_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning("Greeting timeout — proceeding to ACTIVE")
                break

            try:
                event = self._bridge.poll_event()
            except Exception:
                logger.warning("Bridge poll_event error during greeting", exc_info=True)
                break

            if event is not None and event.event_type == CppEventType.PLAYBACK_COMPLETE:
                break

            time.sleep(min(self._POLL_INTERVAL_SEC, remaining))

        self._mode = SystemMode.ACTIVE

    def _run_active(self) -> None:
        """ACTIVE mode: create session components, run orchestrator."""
        self._drain_audio_queue()

        try:
            components = self._session_factory()
        except Exception:
            logger.error("Session factory failed", exc_info=True)
            self._mode = SystemMode.SLEEP
            return

        with self._session_lock:
            self._current_orchestrator = components.orchestrator
            self._current_history = components.history
            self._current_session_id = components.session_id
            self._session_started_at = datetime.now(UTC).strftime(_TIMESTAMP_FORMAT)
            self._session_started = True

        self._current_history.new_session(components.session_id)
        logger.info("Session started: %s", components.session_id)

        try:
            self._current_orchestrator.run()
        except Exception:
            logger.error("Orchestrator run failed", exc_info=True)

        self._mode = SystemMode.FAREWELL

    def _run_farewell(self) -> None:
        """FAREWELL mode: send farewell, wait for playback, save history."""
        self._flush_bridge_events()

        try:
            self._bridge.send_play_file(self._farewell_audio_path)
        except Exception:
            logger.warning("Failed to send farewell", exc_info=True)

        deadline = time.monotonic() + self._FAREWELL_TIMEOUT_SEC
        while not self._shutdown_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning("Farewell timeout — proceeding to SLEEP")
                break

            try:
                event = self._bridge.poll_event()
            except Exception:
                logger.warning("Bridge poll_event error during farewell", exc_info=True)
                break

            if event is not None and event.event_type == CppEventType.PLAYBACK_COMPLETE:
                break

            time.sleep(min(self._POLL_INTERVAL_SEC, remaining))

        if self._session_started and self._current_history is not None:
            try:
                self._current_history.save()
            except Exception:
                logger.warning("History save error in farewell", exc_info=True)
            self._trigger_session_end()

        self._session_started = False
        self._drain_audio_queue()
        self._led.set_state(LEDState.SLEEPING)

        with self._session_lock:
            self._current_orchestrator = None
            self._current_history = None
            self._current_session_id = None
            self._session_started_at = None

        self._mode = SystemMode.SLEEP
        logger.info("Session ended — returning to SLEEP")

    # ------------------------------------------------------------------
    # Memory write trigger
    # ------------------------------------------------------------------

    def _trigger_session_end(self) -> None:
        """Invoke the on_session_end callback if configured."""
        if self._on_session_end is None:
            return
        session_id = self._current_session_id
        started_at = self._session_started_at
        if session_id is None or started_at is None:
            return
        try:
            self._on_session_end(session_id, started_at)
        except Exception:
            logger.warning("on_session_end callback failed", exc_info=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flush_bridge_events(self) -> None:
        """Drain all pending CppBridge events to avoid stale signals."""
        try:
            while True:
                event = self._bridge.poll_event()
                if event is None:
                    break
        except Exception:
            logger.debug("Error flushing bridge events", exc_info=True)

    def _drain_audio_queue(self) -> None:
        """Drain any stale frames from the audio queue."""
        while True:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
