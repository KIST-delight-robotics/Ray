"""Unit tests for CppBridge.

All WebSocket I/O is mocked — no real server needed.
"""

from __future__ import annotations

import base64
import json
import time
from unittest.mock import MagicMock, patch

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.frames import Close

from voice_pipeline.adapters.cpp_bridge import CppBridge, CppEvent, CppEventType, _parse_event

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WS_CONNECT = "voice_pipeline.adapters.cpp_bridge.ws_connect"


def _make_close_exc(code: int = 1006, reason: str = "gone") -> ConnectionClosed:
    """Build a ConnectionClosed with a Close frame."""
    return ConnectionClosed(Close(code, reason), None)


def _connect_with_mock(bridge: CppBridge, mock_conn: MagicMock) -> CppBridge:
    """Connect bridge using mocked WebSocket."""
    with patch(_WS_CONNECT, return_value=mock_conn):
        bridge.connect()
    return bridge


# ===================================================================
# TestEventParsing — module-level _parse_event
# ===================================================================


class TestEventParsing:
    def test_playback_started(self) -> None:
        event = _parse_event('{"type": "playback_started"}')
        assert event == CppEvent(event_type=CppEventType.PLAYBACK_STARTED)

    def test_playback_complete(self) -> None:
        event = _parse_event('{"type": "playback_complete"}')
        assert event == CppEvent(event_type=CppEventType.PLAYBACK_COMPLETE)

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown event type"):
            _parse_event('{"type": "unknown_thing"}')

    def test_missing_type_raises(self) -> None:
        with pytest.raises(KeyError):
            _parse_event('{"data": 123}')

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            _parse_event("not json at all")

    def test_non_dict_json_raises(self) -> None:
        with pytest.raises((TypeError, KeyError)):
            _parse_event("[1, 2, 3]")

    def test_null_json_raises(self) -> None:
        with pytest.raises((TypeError, KeyError)):
            _parse_event("null")

    def test_bytes_input(self) -> None:
        raw = b'{"type": "playback_started"}'
        event = _parse_event(raw)
        assert event.event_type == CppEventType.PLAYBACK_STARTED


# ===================================================================
# TestIdleState — operations before connect
# ===================================================================


class TestIdleState:
    def test_send_audio_before_connect(self, make_bridge) -> None:
        bridge = make_bridge()
        with pytest.raises(RuntimeError, match="Not connected"):
            bridge.send_audio(b"\x00" * 100)

    def test_send_stop_before_connect(self, make_bridge) -> None:
        bridge = make_bridge()
        with pytest.raises(RuntimeError, match="Not connected"):
            bridge.send_stop()

    def test_send_stream_start_before_connect(self, make_bridge) -> None:
        bridge = make_bridge()
        with pytest.raises(RuntimeError, match="Not connected"):
            bridge.send_stream_start()

    def test_send_audio_end_before_connect(self, make_bridge) -> None:
        bridge = make_bridge()
        with pytest.raises(RuntimeError, match="Not connected"):
            bridge.send_audio_end()

    def test_send_play_file_before_connect(self, make_bridge) -> None:
        bridge = make_bridge()
        with pytest.raises(RuntimeError, match="Not connected"):
            bridge.send_play_file("test.wav")

    def test_poll_event_before_connect(self, make_bridge) -> None:
        bridge = make_bridge()
        assert bridge.poll_event() is None

    def test_disconnect_before_connect_is_noop(self, make_bridge) -> None:
        bridge = make_bridge()
        bridge.disconnect()  # should not raise


# ===================================================================
# TestConnectDisconnect
# ===================================================================


class TestConnectDisconnect:
    def test_connect_starts_receiver_thread(self, make_bridge, mock_conn: MagicMock) -> None:
        bridge = _connect_with_mock(make_bridge(), mock_conn)
        assert bridge._running.is_set()
        assert bridge._receiver_thread is not None
        assert bridge._receiver_thread.is_alive()
        bridge.disconnect()

    def test_disconnect_cleans_up(self, make_bridge, mock_conn: MagicMock) -> None:
        bridge = _connect_with_mock(make_bridge(), mock_conn)
        bridge.disconnect()
        assert not bridge._running.is_set()
        assert bridge._conn is None
        assert bridge._receiver_thread is None
        mock_conn.close.assert_called_once()

    def test_connect_retry_on_failure(self, make_bridge) -> None:
        mock_conn = MagicMock()
        mock_conn.recv = MagicMock(side_effect=TimeoutError)
        with patch(_WS_CONNECT, side_effect=[OSError("refused"), mock_conn]) as ws:
            bridge = make_bridge()
            bridge.connect()
            assert ws.call_count == 2
            assert bridge._running.is_set()
        bridge.disconnect()

    def test_all_retries_exhausted(self, make_bridge) -> None:
        with patch(_WS_CONNECT, side_effect=OSError("refused")):
            bridge = make_bridge()
            with pytest.raises(RuntimeError, match="Failed to connect"):
                bridge.connect()

    def test_connect_while_connected_is_noop(self, make_bridge, mock_conn: MagicMock) -> None:
        bridge = _connect_with_mock(make_bridge(), mock_conn)
        # Second connect should log warning, not create new connection
        with patch(_WS_CONNECT) as ws:
            bridge.connect()
            ws.assert_not_called()
        bridge.disconnect()

    def test_idempotent_disconnect(self, make_bridge, mock_conn: MagicMock) -> None:
        bridge = _connect_with_mock(make_bridge(), mock_conn)
        bridge.disconnect()
        bridge.disconnect()  # should not raise
        mock_conn.close.assert_called_once()

    def test_disconnect_after_receiver_failure(self, make_bridge) -> None:
        """disconnect() cleans up even after receiver thread already cleared _running."""
        mock_conn = MagicMock()
        mock_conn.recv = MagicMock(side_effect=_make_close_exc())
        bridge = _connect_with_mock(make_bridge(), mock_conn)

        # Wait for receiver to detect connection loss and clear _running
        deadline = time.monotonic() + 2.0
        while bridge._running.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not bridge._running.is_set()

        # disconnect() must still clean up _conn and _receiver_thread
        bridge.disconnect()
        assert bridge._conn is None
        assert bridge._receiver_thread is None

    def test_reconnect_clears_stale_state(self, make_bridge, mock_conn: MagicMock) -> None:
        bridge = _connect_with_mock(make_bridge(), mock_conn)
        # Inject a stale event
        bridge._event_queue.put(CppEvent(event_type=CppEventType.PLAYBACK_STARTED))
        bridge.disconnect()

        # Reconnect with fresh mock
        mock_conn2 = MagicMock()
        mock_conn2.recv = MagicMock(side_effect=TimeoutError)
        with patch(_WS_CONNECT, return_value=mock_conn2):
            bridge.connect()

        # Old event should be gone (fresh queue)
        assert bridge.poll_event() is None
        bridge.disconnect()


# ===================================================================
# TestSendMethods
# ===================================================================


class TestSendMethods:
    def test_send_audio_encodes_base64(self, make_bridge, mock_conn: MagicMock) -> None:
        bridge = _connect_with_mock(make_bridge(), mock_conn)
        audio = b"\x01\x02\x03\x04"
        bridge.send_audio(audio)

        sent = json.loads(mock_conn.send.call_args[0][0])
        assert sent["type"] == "audio"
        assert base64.b64decode(sent["data"]) == audio
        bridge.disconnect()

    def test_send_stop(self, make_bridge, mock_conn: MagicMock) -> None:
        bridge = _connect_with_mock(make_bridge(), mock_conn)
        bridge.send_stop()
        sent = json.loads(mock_conn.send.call_args[0][0])
        assert sent == {"type": "stop"}
        bridge.disconnect()

    def test_send_stream_start(self, make_bridge, mock_conn: MagicMock) -> None:
        bridge = _connect_with_mock(make_bridge(), mock_conn)
        bridge.send_stream_start()
        sent = json.loads(mock_conn.send.call_args[0][0])
        assert sent == {"type": "stream_start"}
        bridge.disconnect()

    def test_send_audio_end(self, make_bridge, mock_conn: MagicMock) -> None:
        bridge = _connect_with_mock(make_bridge(), mock_conn)
        bridge.send_audio_end()
        sent = json.loads(mock_conn.send.call_args[0][0])
        assert sent == {"type": "audio_end"}
        bridge.disconnect()

    def test_send_play_file(self, make_bridge, mock_conn: MagicMock) -> None:
        bridge = _connect_with_mock(make_bridge(), mock_conn)
        bridge.send_play_file("assets/audio/awake.wav")
        sent = json.loads(mock_conn.send.call_args[0][0])
        assert sent == {"type": "play_file", "file_path": "assets/audio/awake.wav"}
        bridge.disconnect()

    def test_connection_closed_during_send(self, make_bridge, mock_conn: MagicMock) -> None:
        bridge = _connect_with_mock(make_bridge(), mock_conn)
        mock_conn.send.side_effect = _make_close_exc()
        with pytest.raises(RuntimeError, match="Connection lost during send"):
            bridge.send_stop()
        bridge.disconnect()


# ===================================================================
# TestPollEvent
# ===================================================================


class TestPollEvent:
    def test_empty_returns_none(self, make_bridge, mock_conn: MagicMock) -> None:
        bridge = _connect_with_mock(make_bridge(), mock_conn)
        assert bridge.poll_event() is None
        bridge.disconnect()

    def test_queued_events_returned_fifo(self, make_bridge, mock_conn: MagicMock) -> None:
        bridge = _connect_with_mock(make_bridge(), mock_conn)
        e1 = CppEvent(event_type=CppEventType.PLAYBACK_STARTED)
        e2 = CppEvent(event_type=CppEventType.PLAYBACK_COMPLETE)
        bridge._event_queue.put(e1)
        bridge._event_queue.put(e2)
        assert bridge.poll_event() == e1
        assert bridge.poll_event() == e2
        assert bridge.poll_event() is None
        bridge.disconnect()


# ===================================================================
# TestReceiverThread
# ===================================================================


class TestReceiverThread:
    def test_events_enqueued(self, make_bridge) -> None:
        """Receiver thread parses JSON messages and enqueues CppEvents."""
        messages = [
            '{"type": "playback_started"}',
            '{"type": "playback_complete"}',
        ]
        call_count = 0

        def mock_recv(timeout: float = None) -> str:
            nonlocal call_count
            if call_count < len(messages):
                msg = messages[call_count]
                call_count += 1
                return msg
            raise TimeoutError

        mock_conn = MagicMock()
        mock_conn.recv = mock_recv
        bridge = _connect_with_mock(make_bridge(), mock_conn)

        # Give receiver thread time to process
        deadline = time.monotonic() + 2.0
        while bridge._event_queue.qsize() < 2 and time.monotonic() < deadline:
            time.sleep(0.01)

        assert bridge.poll_event() == CppEvent(event_type=CppEventType.PLAYBACK_STARTED)
        assert bridge.poll_event() == CppEvent(event_type=CppEventType.PLAYBACK_COMPLETE)
        bridge.disconnect()

    def test_connection_loss_stores_error(self, make_bridge) -> None:
        """ConnectionClosed in receiver stores RuntimeError."""
        mock_conn = MagicMock()
        mock_conn.recv = MagicMock(side_effect=_make_close_exc())
        bridge = _connect_with_mock(make_bridge(), mock_conn)

        # Wait for receiver to detect connection loss
        deadline = time.monotonic() + 2.0
        while bridge._running.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)

        assert not bridge._running.is_set()
        with pytest.raises(RuntimeError, match="Connection lost"):
            bridge.poll_event()
        bridge.disconnect()

    def test_unparseable_messages_skipped(self, make_bridge) -> None:
        """Bad JSON is skipped, valid messages still enqueued."""
        messages = [
            "not valid json",
            '{"type": "playback_complete"}',
        ]
        call_count = 0

        def mock_recv(timeout: float = None) -> str:
            nonlocal call_count
            if call_count < len(messages):
                msg = messages[call_count]
                call_count += 1
                return msg
            raise TimeoutError

        mock_conn = MagicMock()
        mock_conn.recv = mock_recv
        bridge = _connect_with_mock(make_bridge(), mock_conn)

        deadline = time.monotonic() + 2.0
        while bridge._event_queue.qsize() < 1 and time.monotonic() < deadline:
            time.sleep(0.01)

        assert bridge.poll_event() == CppEvent(event_type=CppEventType.PLAYBACK_COMPLETE)
        bridge.disconnect()

    def test_non_dict_json_skipped(self, make_bridge) -> None:
        """Non-dict JSON (list, null, number) is skipped without killing receiver."""
        messages = [
            "[1, 2, 3]",
            "null",
            "42",
            '{"type": "playback_started"}',
        ]
        call_count = 0

        def mock_recv(timeout: float = None) -> str:
            nonlocal call_count
            if call_count < len(messages):
                msg = messages[call_count]
                call_count += 1
                return msg
            raise TimeoutError

        mock_conn = MagicMock()
        mock_conn.recv = mock_recv
        bridge = _connect_with_mock(make_bridge(), mock_conn)

        deadline = time.monotonic() + 2.0
        while bridge._event_queue.qsize() < 1 and time.monotonic() < deadline:
            time.sleep(0.01)

        assert bridge.poll_event() == CppEvent(event_type=CppEventType.PLAYBACK_STARTED)
        assert bridge._running.is_set()  # receiver thread still alive
        bridge.disconnect()


# ===================================================================
# TestErrorPropagation
# ===================================================================


class TestErrorPropagation:
    def test_error_surfaced_via_poll_event(self, make_bridge) -> None:
        mock_conn = MagicMock()
        mock_conn.recv = MagicMock(side_effect=_make_close_exc())
        bridge = _connect_with_mock(make_bridge(), mock_conn)

        deadline = time.monotonic() + 2.0
        while bridge._running.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)

        with pytest.raises(RuntimeError):
            bridge.poll_event()
        bridge.disconnect()

    def test_error_surfaced_via_send(self, make_bridge) -> None:
        mock_conn = MagicMock()
        mock_conn.recv = MagicMock(side_effect=_make_close_exc())
        bridge = _connect_with_mock(make_bridge(), mock_conn)

        deadline = time.monotonic() + 2.0
        while bridge._running.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)

        with pytest.raises(RuntimeError):
            bridge.send_stop()
        bridge.disconnect()

    def test_error_cleared_after_raise(self, make_bridge) -> None:
        mock_conn = MagicMock()
        mock_conn.recv = MagicMock(side_effect=_make_close_exc())
        bridge = _connect_with_mock(make_bridge(), mock_conn)

        deadline = time.monotonic() + 2.0
        while bridge._running.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)

        with pytest.raises(RuntimeError):
            bridge.poll_event()
        # Second call should not raise (error cleared), but Not connected
        # since _running is cleared by connection loss
        with pytest.raises(RuntimeError, match="Not connected"):
            bridge.send_stop()
        bridge.disconnect()
