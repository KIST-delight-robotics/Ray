"""Tests for voice_pipeline.session.session_manager."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

from voice_pipeline.core.config import SessionConfig
from voice_pipeline.core.types import (
    AudioFrame,
    CppEvent,
    CppEventType,
    LEDState,
    SystemMode,
)
from voice_pipeline.session.session_manager import SessionManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_manager(
    *,
    greeting_timeout_sec: float = 0.1,
    farewell_timeout_sec: float = 0.1,
    frame_timeout_sec: float = 0.01,
    audio_queue_size: int = 300,
) -> tuple[SessionManager, dict[str, MagicMock]]:
    """Create a SessionManager with all dependencies mocked."""
    mocks = {
        "audio_input": MagicMock(),
        "wakeword": MagicMock(),
        "orchestrator": MagicMock(),
        "bridge": MagicMock(),
        "history": MagicMock(),
        "led": MagicMock(),
    }

    # Defaults
    mocks["wakeword"].feed_audio.return_value = False
    mocks["bridge"].poll_event.return_value = None
    mocks["orchestrator"].request_stop = MagicMock()

    config = SessionConfig(
        audio_queue_size=audio_queue_size,
        greeting_timeout_sec=greeting_timeout_sec,
        farewell_timeout_sec=farewell_timeout_sec,
        frame_timeout_sec=frame_timeout_sec,
    )

    sm = SessionManager(
        audio_input=mocks["audio_input"],
        wakeword=mocks["wakeword"],
        orchestrator=mocks["orchestrator"],
        cpp_bridge=mocks["bridge"],
        history=mocks["history"],
        led=mocks["led"],
        config=config,
    )

    return sm, mocks


def _frame() -> AudioFrame:
    return b"\x00" * 960


# ---------------------------------------------------------------------------
# Full cycle
# ---------------------------------------------------------------------------


class TestFullCycle:
    def test_sleep_greeting_active_farewell_sleep(self) -> None:
        """Full cycle: wakeword → greeting → active → farewell → sleep."""
        sm, mocks = _make_session_manager()

        # Wakeword detects on first feed
        mocks["wakeword"].feed_audio.return_value = True

        # Greeting: immediate PLAYBACK_COMPLETE
        greeting_event = CppEvent(CppEventType.PLAYBACK_COMPLETE)
        farewell_event = CppEvent(CppEventType.PLAYBACK_COMPLETE)
        # First poll_event call flushes (returns None), then greeting poll
        mocks["bridge"].poll_event.side_effect = [
            None,  # flush in greeting
            greeting_event,  # greeting done
            None,  # flush in farewell
            farewell_event,  # farewell done
        ]

        # Orchestrator.run returns immediately
        mocks["orchestrator"].run.return_value = None

        # Shutdown after one full cycle
        cycle_count = 0
        original_run_sleep = sm._run_sleep

        def _shutdown_on_second_sleep():
            nonlocal cycle_count
            cycle_count += 1
            if cycle_count >= 2:
                sm._shutdown_event.set()
                return
            original_run_sleep()

        sm._run_sleep = _shutdown_on_second_sleep

        # Push a frame for wakeword detection
        sm._audio_queue.put(_frame())

        sm.run()

        mocks["bridge"].connect.assert_called_once()
        mocks["audio_input"].start.assert_called_once()
        mocks["audio_input"].stop.assert_called_once()
        mocks["bridge"].send_play_file.assert_any_call(sm._config.greeting_audio_path)
        mocks["orchestrator"].run.assert_called_once_with(sm._audio_queue)
        mocks["bridge"].send_play_file.assert_any_call(sm._config.farewell_audio_path)
        mocks["history"].new_session.assert_called_once()
        mocks["history"].save.assert_called_once()


class TestGreeting:
    def test_greeting_timeout(self) -> None:
        """Greeting proceeds to ACTIVE even without PLAYBACK_COMPLETE."""
        sm, mocks = _make_session_manager(greeting_timeout_sec=0.01)

        # No PLAYBACK_COMPLETE ever
        mocks["bridge"].poll_event.return_value = None

        sm._run_greeting()

        assert sm._mode == SystemMode.ACTIVE
        mocks["bridge"].send_play_file.assert_called_once_with(sm._config.greeting_audio_path)

    def test_greeting_flushes_stale_events(self) -> None:
        """Stale events are flushed before sending greeting."""
        sm, mocks = _make_session_manager()

        stale = CppEvent(CppEventType.PLAYBACK_COMPLETE)
        fresh = CppEvent(CppEventType.PLAYBACK_COMPLETE)
        # flush returns stale then None; greeting poll returns fresh
        mocks["bridge"].poll_event.side_effect = [stale, None, fresh]

        sm._run_greeting()

        assert sm._mode == SystemMode.ACTIVE
        mocks["bridge"].send_play_file.assert_called_once_with(sm._config.greeting_audio_path)

    def test_greeting_bridge_error(self) -> None:
        """send_greeting error doesn't crash SessionManager."""
        sm, mocks = _make_session_manager(greeting_timeout_sec=0.01)

        mocks["bridge"].send_play_file.side_effect = RuntimeError("Bridge down")
        mocks["bridge"].poll_event.return_value = None

        sm._run_greeting()  # Should not raise
        assert sm._mode == SystemMode.ACTIVE


class TestFarewell:
    def test_farewell_timeout(self) -> None:
        """Farewell proceeds to SLEEP even without PLAYBACK_COMPLETE."""
        sm, mocks = _make_session_manager(farewell_timeout_sec=0.01)
        sm._session_started = True

        mocks["bridge"].poll_event.return_value = None

        sm._run_farewell()

        assert sm._mode == SystemMode.SLEEP
        mocks["history"].save.assert_called_once()

    def test_farewell_flushes_stale_events(self) -> None:
        """Stale events are flushed before sending farewell."""
        sm, mocks = _make_session_manager()
        sm._session_started = True

        stale = CppEvent(CppEventType.PLAYBACK_COMPLETE)
        fresh = CppEvent(CppEventType.PLAYBACK_COMPLETE)
        mocks["bridge"].poll_event.side_effect = [stale, None, fresh]

        sm._run_farewell()

        assert sm._mode == SystemMode.SLEEP
        mocks["bridge"].send_play_file.assert_called_once_with(sm._config.farewell_audio_path)

    def test_farewell_saves_history(self) -> None:
        """History is saved during farewell."""
        sm, mocks = _make_session_manager(farewell_timeout_sec=0.01)
        sm._session_started = True

        mocks["bridge"].poll_event.return_value = None

        sm._run_farewell()

        mocks["history"].save.assert_called_once()
        assert sm._session_started is False

    def test_farewell_poll_event_error(self) -> None:
        """poll_event error during farewell doesn't crash."""
        sm, mocks = _make_session_manager()
        sm._session_started = True

        # flush returns None, then poll_event raises
        mocks["bridge"].poll_event.side_effect = [None, RuntimeError("gone")]

        sm._run_farewell()  # Should not raise
        assert sm._mode == SystemMode.SLEEP


class TestSleep:
    def test_wakeword_transitions_to_greeting(self) -> None:
        """Wakeword detection transitions from SLEEP to GREETING."""
        sm, mocks = _make_session_manager()

        mocks["wakeword"].feed_audio.return_value = True
        sm._audio_queue.put(_frame())

        sm._run_sleep()

        assert sm._mode == SystemMode.GREETING

    def test_shutdown_during_sleep(self) -> None:
        """Shutdown during SLEEP exits immediately."""
        sm, mocks = _make_session_manager()

        sm._shutdown_event.set()
        sm._run_sleep()

        # Should exit without transitioning
        assert sm._mode == SystemMode.SLEEP


class TestActive:
    def test_drains_queue_before_orchestrator(self) -> None:
        """Audio queue is drained before orchestrator.run()."""
        sm, mocks = _make_session_manager()

        # Put stale frames
        for _ in range(5):
            sm._audio_queue.put(_frame())

        sm._run_active()

        assert sm._audio_queue.empty()
        mocks["orchestrator"].run.assert_called_once()

    def test_new_session_called_with_uuid(self) -> None:
        """history.new_session is called with a UUID string."""
        sm, mocks = _make_session_manager()

        sm._run_active()

        mocks["history"].new_session.assert_called_once()
        session_id = mocks["history"].new_session.call_args[0][0]
        # Should be a valid UUID string
        import uuid

        uuid.UUID(session_id)  # Raises if invalid


class TestShutdown:
    def test_shutdown_calls_request_stop(self) -> None:
        """shutdown() calls orchestrator.request_stop()."""
        sm, mocks = _make_session_manager()

        sm.shutdown()

        assert sm._shutdown_event.is_set()
        mocks["orchestrator"].request_stop.assert_called_once()

    def test_shutdown_saves_history_if_session_started(self) -> None:
        """shutdown() saves history when a session is active."""
        sm, mocks = _make_session_manager()
        sm._session_started = True

        sm.shutdown()

        mocks["history"].save.assert_called_once()

    def test_shutdown_no_save_if_no_session(self) -> None:
        """shutdown() doesn't save history when no session was started."""
        sm, mocks = _make_session_manager()
        sm._session_started = False

        sm.shutdown()

        mocks["history"].save.assert_not_called()

    def test_shutdown_during_active(self) -> None:
        """Shutdown during ACTIVE triggers orchestrator.request_stop."""
        sm, mocks = _make_session_manager()

        # Orchestrator.run blocks until shutdown
        def _blocking_run(q):
            while not sm._shutdown_event.is_set():
                time.sleep(0.01)

        mocks["orchestrator"].run.side_effect = _blocking_run

        t = threading.Thread(target=sm.run)
        # Push frame for wakeword
        sm._audio_queue.put(_frame())
        mocks["wakeword"].feed_audio.return_value = True
        # Greeting completes immediately
        mocks["bridge"].poll_event.side_effect = [None, CppEvent(CppEventType.PLAYBACK_COMPLETE)]

        t.start()
        time.sleep(0.1)  # Let it reach ACTIVE

        sm.shutdown()
        t.join(timeout=2.0)

        mocks["orchestrator"].request_stop.assert_called_once()


class TestLED:
    def test_sleep_sets_sleeping(self) -> None:
        """SLEEP mode sets LED to SLEEPING."""
        sm, mocks = _make_session_manager()

        # Exit immediately
        sm._shutdown_event.set()
        sm._run_sleep()

        mocks["led"].set_state.assert_called_with(LEDState.SLEEPING)

    def test_greeting_sets_listening(self) -> None:
        """GREETING mode sets LED to LISTENING."""
        sm, mocks = _make_session_manager(greeting_timeout_sec=0.01)
        mocks["bridge"].poll_event.return_value = None

        sm._run_greeting()

        led_calls = [c.args[0] for c in mocks["led"].set_state.call_args_list]
        assert LEDState.LISTENING in led_calls

    def test_farewell_sets_sleeping(self) -> None:
        """FAREWELL sets LED back to SLEEPING."""
        sm, mocks = _make_session_manager(farewell_timeout_sec=0.01)
        sm._session_started = True
        mocks["bridge"].poll_event.return_value = None

        sm._run_farewell()

        led_calls = [c.args[0] for c in mocks["led"].set_state.call_args_list]
        assert led_calls[-1] == LEDState.SLEEPING


class TestCppBridgeConnect:
    def test_connect_called_on_startup(self) -> None:
        """cpp_bridge.connect() is called at the start of run()."""
        sm, mocks = _make_session_manager()

        sm._shutdown_event.set()  # Exit immediately
        sm.run()

        mocks["bridge"].connect.assert_called_once()
