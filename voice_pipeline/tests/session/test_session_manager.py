"""Tests for voice_pipeline.session.session_manager."""

from __future__ import annotations

import threading
import time
import uuid
from unittest.mock import MagicMock

import pytest

from voice_pipeline.core.types import (
    AudioFrame,
    CppEvent,
    CppEventType,
    LEDState,
    SystemMode,
)
from voice_pipeline.session.session_manager import SessionComponents, SessionManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_manager(
    monkeypatch: pytest.MonkeyPatch,
    *,
    greeting_timeout_sec: float = 0.1,
    farewell_timeout_sec: float = 0.1,
    frame_timeout_sec: float = 0.01,
    audio_queue_size: int = 300,
    on_session_end: MagicMock | None = None,
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
    mocks["audio_input"].error = None
    mocks["wakeword"].feed_audio.return_value = False
    mocks["bridge"].poll_event.return_value = None
    mocks["orchestrator"].request_stop = MagicMock()
    mocks["orchestrator"].run = MagicMock()

    monkeypatch.setattr(SessionManager, "AUDIO_QUEUE_SIZE", audio_queue_size)
    monkeypatch.setattr(SessionManager, "_GREETING_TIMEOUT_SEC", greeting_timeout_sec)
    monkeypatch.setattr(SessionManager, "_FAREWELL_TIMEOUT_SEC", farewell_timeout_sec)
    monkeypatch.setattr(SessionManager, "_FRAME_TIMEOUT_SEC", frame_timeout_sec)

    def session_factory() -> SessionComponents:
        return SessionComponents(
            orchestrator=mocks["orchestrator"],
            history=mocks["history"],
            session_id=str(uuid.uuid4()),
        )

    sm = SessionManager(
        audio_input=mocks["audio_input"],
        wakeword=mocks["wakeword"],
        session_factory=session_factory,
        cpp_bridge=mocks["bridge"],
        led=mocks["led"],
        greeting_audio_path="assets/audio/greeting.wav",
        farewell_audio_path="assets/audio/farewell.wav",
        on_session_end=on_session_end,
    )

    return sm, mocks


def _frame() -> AudioFrame:
    return b"\x00" * 960


# ---------------------------------------------------------------------------
# Full cycle
# ---------------------------------------------------------------------------


class TestFullCycle:
    def test_sleep_greeting_active_farewell_sleep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Full cycle: wakeword → greeting → active → farewell → sleep."""
        sm, mocks = _make_session_manager(monkeypatch)

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

        assert mocks["bridge"].connect.call_count == 2  # startup + greeting reconnect
        mocks["audio_input"].start.assert_called_once()
        mocks["audio_input"].stop.assert_called_once()
        mocks["bridge"].send_play_file.assert_any_call(sm._greeting_audio_path)
        mocks["orchestrator"].run.assert_called_once_with()
        mocks["bridge"].send_play_file.assert_any_call(sm._farewell_audio_path)
        mocks["history"].new_session.assert_called_once()
        mocks["history"].save.assert_called_once()


class TestGreeting:
    def test_greeting_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Greeting proceeds to ACTIVE even without PLAYBACK_COMPLETE."""
        sm, mocks = _make_session_manager(monkeypatch, greeting_timeout_sec=0.01)

        # No PLAYBACK_COMPLETE ever
        mocks["bridge"].poll_event.return_value = None

        sm._run_greeting()

        assert sm._mode == SystemMode.ACTIVE
        mocks["bridge"].send_play_file.assert_called_once_with(sm._greeting_audio_path)

    def test_greeting_flushes_stale_events(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stale events are flushed before sending greeting."""
        sm, mocks = _make_session_manager(monkeypatch)

        stale = CppEvent(CppEventType.PLAYBACK_COMPLETE)
        fresh = CppEvent(CppEventType.PLAYBACK_COMPLETE)
        # flush returns stale then None; greeting poll returns fresh
        mocks["bridge"].poll_event.side_effect = [stale, None, fresh]

        sm._run_greeting()

        assert sm._mode == SystemMode.ACTIVE
        mocks["bridge"].send_play_file.assert_called_once_with(sm._greeting_audio_path)

    def test_greeting_bridge_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """send_greeting error doesn't crash SessionManager."""
        sm, mocks = _make_session_manager(monkeypatch, greeting_timeout_sec=0.01)

        mocks["bridge"].send_play_file.side_effect = RuntimeError("Bridge down")
        mocks["bridge"].poll_event.return_value = None

        sm._run_greeting()  # Should not raise
        assert sm._mode == SystemMode.ACTIVE


class TestFarewell:
    def test_farewell_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Farewell proceeds to SLEEP even without PLAYBACK_COMPLETE."""
        sm, mocks = _make_session_manager(monkeypatch, farewell_timeout_sec=0.01)
        sm._session_started = True
        sm._current_history = mocks["history"]

        mocks["bridge"].poll_event.return_value = None

        sm._run_farewell()

        assert sm._mode == SystemMode.SLEEP
        mocks["history"].save.assert_called_once()

    def test_farewell_flushes_stale_events(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stale events are flushed before sending farewell."""
        sm, mocks = _make_session_manager(monkeypatch)
        sm._session_started = True
        sm._current_history = mocks["history"]

        stale = CppEvent(CppEventType.PLAYBACK_COMPLETE)
        fresh = CppEvent(CppEventType.PLAYBACK_COMPLETE)
        mocks["bridge"].poll_event.side_effect = [stale, None, fresh]

        sm._run_farewell()

        assert sm._mode == SystemMode.SLEEP
        mocks["bridge"].send_play_file.assert_called_once_with(sm._farewell_audio_path)

    def test_farewell_saves_history(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """History is saved during farewell."""
        sm, mocks = _make_session_manager(monkeypatch, farewell_timeout_sec=0.01)
        sm._session_started = True
        sm._current_history = mocks["history"]

        mocks["bridge"].poll_event.return_value = None

        sm._run_farewell()

        mocks["history"].save.assert_called_once()
        assert sm._session_started is False

    def test_farewell_poll_event_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """poll_event error during farewell doesn't crash."""
        sm, mocks = _make_session_manager(monkeypatch)
        sm._session_started = True
        sm._current_history = mocks["history"]

        # flush returns None, then poll_event raises
        mocks["bridge"].poll_event.side_effect = [None, RuntimeError("gone")]

        sm._run_farewell()  # Should not raise
        assert sm._mode == SystemMode.SLEEP


class TestSleep:
    def test_wakeword_transitions_to_greeting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Wakeword detection transitions from SLEEP to GREETING."""
        sm, mocks = _make_session_manager(monkeypatch)

        mocks["wakeword"].feed_audio.return_value = True
        sm._audio_queue.put(_frame())

        sm._run_sleep()

        assert sm._mode == SystemMode.GREETING

    def test_shutdown_during_sleep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Shutdown during SLEEP exits immediately."""
        sm, mocks = _make_session_manager(monkeypatch)

        sm._shutdown_event.set()
        sm._run_sleep()

        # Should exit without transitioning
        assert sm._mode == SystemMode.SLEEP


class TestAudioInputError:
    def test_mic_failure_propagates_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mic capture thread death raises the stored error in SLEEP loop."""
        sm, mocks = _make_session_manager(monkeypatch)

        mic_error = OSError("No such device")
        mocks["audio_input"].error = mic_error

        with pytest.raises(OSError, match="No such device"):
            sm._run_sleep()


class TestActive:
    def test_drains_queue_before_orchestrator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Audio queue is drained before orchestrator.run()."""
        sm, mocks = _make_session_manager(monkeypatch)

        # Put stale frames
        for _ in range(5):
            sm._audio_queue.put(_frame())

        sm._run_active()

        assert sm._audio_queue.empty()
        mocks["orchestrator"].run.assert_called_once()

    def test_session_id_from_factory_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Factory-provided session_id is passed to history.new_session."""
        sm, mocks = _make_session_manager(monkeypatch)

        sm._run_active()

        mocks["history"].new_session.assert_called_once()
        session_id = mocks["history"].new_session.call_args[0][0]
        # Should be a valid UUID string
        uuid.UUID(session_id)  # Raises if invalid
        # SessionManager should store the same session_id
        assert sm._current_session_id == session_id

    def test_factory_failure_returns_to_sleep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Session factory exception → SLEEP, no crash."""
        sm, mocks = _make_session_manager(monkeypatch)

        # Replace factory with one that raises
        sm._session_factory = MagicMock(side_effect=RuntimeError("factory boom"))

        sm._run_active()

        assert sm._mode == SystemMode.SLEEP
        mocks["orchestrator"].run.assert_not_called()


class TestShutdown:
    def test_shutdown_calls_request_stop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """shutdown() calls orchestrator.request_stop() when orchestrator is set."""
        sm, mocks = _make_session_manager(monkeypatch)
        sm._current_orchestrator = mocks["orchestrator"]

        sm.shutdown()

        assert sm._shutdown_event.is_set()
        mocks["orchestrator"].request_stop.assert_called_once()

    def test_shutdown_saves_history_if_session_started(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """shutdown() saves history when a session is active."""
        sm, mocks = _make_session_manager(monkeypatch)
        sm._session_started = True
        sm._current_history = mocks["history"]

        sm.shutdown()

        mocks["history"].save.assert_called_once()

    def test_shutdown_no_save_if_no_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """shutdown() doesn't save history when no session was started."""
        sm, mocks = _make_session_manager(monkeypatch)
        sm._session_started = False

        sm.shutdown()

        mocks["history"].save.assert_not_called()

    def test_shutdown_during_active(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Shutdown during ACTIVE triggers orchestrator.request_stop."""
        sm, mocks = _make_session_manager(monkeypatch)

        # Orchestrator.run blocks until shutdown
        def _blocking_run():
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

    def test_shutdown_no_orchestrator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """shutdown() is safe when no orchestrator is set."""
        sm, mocks = _make_session_manager(monkeypatch)
        # _current_orchestrator is None by default

        sm.shutdown()  # Should not raise
        assert sm._shutdown_event.is_set()


class TestLED:
    def test_sleep_sets_sleeping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SLEEP mode sets LED to SLEEPING."""
        sm, mocks = _make_session_manager(monkeypatch)

        # Exit immediately
        sm._shutdown_event.set()
        sm._run_sleep()

        mocks["led"].set_state.assert_called_with(LEDState.SLEEPING)

    def test_greeting_sets_idle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GREETING mode sets LED to IDLE."""
        sm, mocks = _make_session_manager(monkeypatch, greeting_timeout_sec=0.01)
        mocks["bridge"].poll_event.return_value = None

        sm._run_greeting()

        led_calls = [c.args[0] for c in mocks["led"].set_state.call_args_list]
        assert LEDState.IDLE in led_calls

    def test_farewell_sets_sleeping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FAREWELL sets LED back to SLEEPING."""
        sm, mocks = _make_session_manager(monkeypatch, farewell_timeout_sec=0.01)
        sm._session_started = True
        sm._current_history = mocks["history"]
        mocks["bridge"].poll_event.return_value = None

        sm._run_farewell()

        led_calls = [c.args[0] for c in mocks["led"].set_state.call_args_list]
        assert led_calls[-1] == LEDState.SLEEPING


class TestCppBridgeConnect:
    def test_connect_called_on_startup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """cpp_bridge.connect() is called at the start of run()."""
        sm, mocks = _make_session_manager(monkeypatch)

        sm._shutdown_event.set()  # Exit immediately
        sm.run()

        mocks["bridge"].connect.assert_called_once()


class TestMultiSessionIsolation:
    def test_two_sessions_get_independent_components(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two consecutive sessions get independent history and orchestrator."""
        histories = []
        orchestrators = []

        def tracking_factory() -> SessionComponents:
            orch = MagicMock()
            orch.run = MagicMock()
            hist = MagicMock()
            histories.append(hist)
            orchestrators.append(orch)
            return SessionComponents(orchestrator=orch, history=hist, session_id=str(uuid.uuid4()))

        monkeypatch.setattr(SessionManager, "AUDIO_QUEUE_SIZE", 300)
        monkeypatch.setattr(SessionManager, "_GREETING_TIMEOUT_SEC", 0.01)
        monkeypatch.setattr(SessionManager, "_FAREWELL_TIMEOUT_SEC", 0.01)
        monkeypatch.setattr(SessionManager, "_FRAME_TIMEOUT_SEC", 0.01)

        bridge = MagicMock()
        bridge.poll_event.return_value = None

        sm = SessionManager(
            audio_input=MagicMock(),
            wakeword=MagicMock(),
            session_factory=tracking_factory,
            cpp_bridge=bridge,
            led=MagicMock(),
            greeting_audio_path="assets/audio/greeting.wav",
            farewell_audio_path="assets/audio/farewell.wav",
        )

        # Run two active sessions
        sm._run_active()
        sm._run_farewell()

        sm._run_active()
        sm._run_farewell()

        assert len(histories) == 2
        assert len(orchestrators) == 2
        assert histories[0] is not histories[1]
        assert orchestrators[0] is not orchestrators[1]


# ---------------------------------------------------------------------------
# Bridge reconnect in greeting
# ---------------------------------------------------------------------------


class TestGreetingReconnect:
    def test_greeting_reconnects_after_bridge_disconnect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If bridge is disconnected, _run_greeting() reconnects and proceeds."""
        sm, mocks = _make_session_manager(monkeypatch)

        greeting_event = CppEvent(CppEventType.PLAYBACK_COMPLETE)
        mocks["bridge"].poll_event.side_effect = [
            None,  # flush
            greeting_event,
        ]

        sm._run_greeting()

        mocks["bridge"].connect.assert_called_once()
        mocks["bridge"].send_play_file.assert_called_once_with(sm._greeting_audio_path)
        assert sm._mode == SystemMode.ACTIVE

    def test_greeting_reconnect_failure_returns_to_sleep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If reconnect fails, _run_greeting() returns to SLEEP without entering ACTIVE."""
        from voice_pipeline.bridge.exceptions import BridgeError

        sm, mocks = _make_session_manager(monkeypatch)
        mocks["bridge"].connect.side_effect = BridgeError("Connection refused")

        sm._run_greeting()

        assert sm._mode == SystemMode.SLEEP
        mocks["bridge"].send_play_file.assert_not_called()


# ---------------------------------------------------------------------------
# on_session_end callback
# ---------------------------------------------------------------------------


class TestOnSessionEnd:
    def test_callback_called_in_farewell(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """on_session_end is called during farewell with session_id and started_at."""
        callback = MagicMock()
        sm, mocks = _make_session_manager(monkeypatch, farewell_timeout_sec=0.01, on_session_end=callback)
        mocks["bridge"].poll_event.return_value = None

        # Simulate a session that went through _run_active
        sm._run_active()
        session_id = sm._current_session_id
        started_at = sm._session_started_at

        sm._run_farewell()

        callback.assert_called_once_with(session_id, started_at)

    def test_callback_error_doesnt_crash_farewell(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If on_session_end raises, farewell still completes."""
        callback = MagicMock(side_effect=RuntimeError("write failed"))
        sm, mocks = _make_session_manager(monkeypatch, farewell_timeout_sec=0.01, on_session_end=callback)
        mocks["bridge"].poll_event.return_value = None

        sm._run_active()
        sm._run_farewell()

        assert sm._mode == SystemMode.SLEEP
        callback.assert_called_once()

    def test_no_callback_backward_compat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without on_session_end, farewell works as before."""
        sm, mocks = _make_session_manager(monkeypatch, farewell_timeout_sec=0.01)
        mocks["bridge"].poll_event.return_value = None

        sm._run_active()
        sm._run_farewell()

        assert sm._mode == SystemMode.SLEEP
        mocks["history"].save.assert_called_once()

    def test_callback_called_on_shutdown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """shutdown() triggers on_session_end if a session is active."""
        callback = MagicMock()
        sm, mocks = _make_session_manager(monkeypatch, on_session_end=callback)

        # Simulate an active session
        sm._run_active()

        sm.shutdown()

        callback.assert_called_once()
        assert sm._shutdown_event.is_set()
