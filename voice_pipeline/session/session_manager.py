"""SessionManager: top-level state machine for the voice pipeline."""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from voice_pipeline.core.config import SessionConfig
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


@dataclass
class SessionComponents:
    """Per-session objects created by the session factory."""

    orchestrator: Orchestrator
    history: IConversationHistory


class SessionManager(ISessionManager):
    """Top-level state machine: SLEEP → GREETING → ACTIVE → FAREWELL → SLEEP.

    Uses a session factory to create fresh per-session components,
    ensuring clean state isolation between conversations.
    """

    def __init__(
        self,
        audio_input: IAudioInput,
        wakeword: IWakewordDetector,
        session_factory: Callable[[], SessionComponents],
        cpp_bridge: ICppBridge,
        led: ILEDController,
        config: SessionConfig,
        greeting_audio_path: str,
        farewell_audio_path: str,
        audio_queue: queue.Queue[AudioFrame] | None = None,
    ) -> None:
        self._audio_input = audio_input
        self._wakeword = wakeword
        self._session_factory = session_factory
        self._bridge = cpp_bridge
        self._led = led
        self._config = config
        self._greeting_audio_path = greeting_audio_path
        self._farewell_audio_path = farewell_audio_path

        self._audio_queue: queue.Queue[AudioFrame] = audio_queue or queue.Queue(
            maxsize=config.audio_queue_size
        )
        self._shutdown_event = threading.Event()
        self._mode = SystemMode.SLEEP
        self._session_started = False

        self._session_lock = threading.Lock()
        self._current_orchestrator: Orchestrator | None = None
        self._current_history: IConversationHistory | None = None

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

    # ------------------------------------------------------------------
    # Mode implementations
    # ------------------------------------------------------------------

    def _run_sleep(self) -> None:
        """SLEEP mode: listen for wakeword."""
        self._led.set_state(LEDState.SLEEPING)

        while not self._shutdown_event.is_set():
            try:
                frame = self._audio_queue.get(timeout=self._config.frame_timeout_sec)
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
        self._flush_bridge_events()
        self._led.set_state(LEDState.LISTENING)

        try:
            self._bridge.send_play_file(self._greeting_audio_path)
        except Exception:
            logger.warning("Failed to send greeting", exc_info=True)

        deadline = time.monotonic() + self._config.greeting_timeout_sec
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

            time.sleep(min(0.05, remaining))

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

        session_id = str(uuid.uuid4())
        self._current_history.new_session(session_id)
        self._session_started = True
        logger.info("Session started: %s", session_id)

        try:
            self._current_orchestrator.run(self._audio_queue)
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

        deadline = time.monotonic() + self._config.farewell_timeout_sec
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

            time.sleep(min(0.05, remaining))

        if self._session_started and self._current_history is not None:
            try:
                self._current_history.save()
            except Exception:
                logger.warning("History save error in farewell", exc_info=True)

        self._session_started = False
        self._drain_audio_queue()
        self._led.set_state(LEDState.SLEEPING)

        with self._session_lock:
            self._current_orchestrator = None
            self._current_history = None

        self._mode = SystemMode.SLEEP
        logger.info("Session ended — returning to SLEEP")

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
