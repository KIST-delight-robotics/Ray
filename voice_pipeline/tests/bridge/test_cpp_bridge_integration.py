"""Integration tests for CppBridge using an in-process WebSocket server.

No external C++ process required — uses websockets.sync.server.
"""

from __future__ import annotations

import base64
import json
import threading
import time

import pytest
from websockets.sync.server import ServerConnection, serve

from voice_pipeline.bridge.cpp_bridge import CppBridge
from voice_pipeline.bridge.exceptions import BridgeError
from voice_pipeline.core.types import CppEventType

pytestmark = pytest.mark.requires_api

# ---------------------------------------------------------------------------
# Test server helpers
# ---------------------------------------------------------------------------

_HOST = "localhost"
_PORT = 19876


def _echo_handler(conn: ServerConnection) -> None:
    """Echo handler: receives JSON, sends back an event for each command."""
    for msg in conn:
        data = json.loads(msg)
        msg_type = data["type"]
        if msg_type == "audio":
            conn.send(json.dumps({"type": "playback_started"}))
        elif msg_type == "stop" or msg_type == "play_file":
            conn.send(json.dumps({"type": "playback_complete"}))


def _silent_handler(conn: ServerConnection) -> None:
    """Handler that receives but never responds."""
    for _ in conn:
        pass


# Tracks server-side connections so tests can close them explicitly.
_server_connections: list[ServerConnection] = []
_server_conn_lock = threading.Lock()


def _tracking_handler(conn: ServerConnection) -> None:
    """Handler that tracks the connection for explicit close in tests."""
    with _server_conn_lock:
        _server_connections.append(conn)
    for _ in conn:
        pass


def _collector_handler(collected: list[dict]):
    """Return a handler that collects all received messages."""

    def handler(conn: ServerConnection) -> None:
        for msg in conn:
            collected.append(json.loads(msg))

    return handler


@pytest.fixture
def echo_bridge(make_bridge):
    """Factory for echo test bridges (uses _PORT, longer connect/close timeouts)."""

    def _make(**overrides) -> CppBridge:
        return make_bridge(
            host=_HOST,
            port=_PORT,
            connect_timeout_sec=2.0,
            close_timeout_sec=2.0,
            **overrides,
        )

    return _make


@pytest.fixture
def echo_server():
    """Start an echo WebSocket server in a background thread."""
    server = serve(_echo_handler, _HOST, _PORT)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join(timeout=5.0)


@pytest.fixture
def silent_server():
    """Start a silent WebSocket server."""
    server = serve(_silent_handler, _HOST, _PORT)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_connect_disconnect(self, echo_bridge, echo_server) -> None:
        bridge = echo_bridge()
        bridge.connect()
        assert bridge._running.is_set()
        bridge.disconnect()
        assert not bridge._running.is_set()

    def test_connect_to_nonexistent_server(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(CppBridge, "_RECONNECT_ATTEMPTS", 1)
        monkeypatch.setattr(CppBridge, "_CONNECT_TIMEOUT_SEC", 1.0)
        monkeypatch.setattr(CppBridge, "_HOST", _HOST)
        monkeypatch.setattr(CppBridge, "_PORT", 19877)
        bridge = CppBridge()  # nothing listening
        with pytest.raises(BridgeError, match="Failed to connect"):
            bridge.connect()

    def test_disconnect_then_reconnect(self, echo_bridge, echo_server) -> None:
        bridge = echo_bridge()
        bridge.connect()
        bridge.disconnect()

        bridge.connect()
        assert bridge._running.is_set()
        bridge.disconnect()


class TestSendReceive:
    def test_server_receives_commands(self, echo_bridge) -> None:
        collected: list[dict] = []
        server = serve(_collector_handler(collected), _HOST, _PORT)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            bridge = echo_bridge()
            bridge.connect()

            bridge.send_stop()
            bridge.send_stream_start()
            bridge.send_play_file("test.wav")
            audio = b"\xaa\xbb\xcc"
            bridge.send_audio(audio)
            bridge.send_audio_end()

            # Give server time to collect
            time.sleep(0.2)
            bridge.disconnect()

            assert len(collected) == 5
            assert collected[0] == {"type": "stop"}
            assert collected[1] == {"type": "stream_start"}
            assert collected[2] == {"type": "play_file", "file_path": "test.wav"}
            assert collected[3]["type"] == "audio"
            assert base64.b64decode(collected[3]["data"]) == audio
            assert collected[4] == {"type": "audio_end"}
        finally:
            server.shutdown()
            thread.join(timeout=5.0)

    def test_round_trip_audio_to_event(self, echo_bridge, echo_server) -> None:
        bridge = echo_bridge()
        bridge.connect()
        bridge.send_audio(b"\x00" * 100)

        # Wait for echo event
        event = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            event = bridge.poll_event()
            if event is not None:
                break
            time.sleep(0.01)

        assert event is not None
        assert event.event_type == CppEventType.PLAYBACK_STARTED
        bridge.disconnect()

    def test_round_trip_stop_to_complete(self, echo_bridge, echo_server) -> None:
        bridge = echo_bridge()
        bridge.connect()
        bridge.send_stop()

        event = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            event = bridge.poll_event()
            if event is not None:
                break
            time.sleep(0.01)

        assert event is not None
        assert event.event_type == CppEventType.PLAYBACK_COMPLETE
        bridge.disconnect()


class TestServerFailure:
    def test_server_disconnect_detected(self, echo_bridge) -> None:
        """Explicit server-side close is detected as BridgeError."""
        _server_connections.clear()
        server = serve(_tracking_handler, _HOST, _PORT)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            bridge = echo_bridge()
            bridge.connect()

            # Wait for server to register the connection
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                with _server_conn_lock:
                    if _server_connections:
                        break
                time.sleep(0.01)

            # Close the server-side connection explicitly
            with _server_conn_lock:
                for sc in _server_connections:
                    sc.close()
                _server_connections.clear()

            # Receiver should detect connection loss
            deadline = time.monotonic() + 3.0
            while bridge._running.is_set() and time.monotonic() < deadline:
                time.sleep(0.01)

            with pytest.raises(BridgeError):
                bridge.poll_event()
            bridge.disconnect()
        finally:
            server.shutdown()
            thread.join(timeout=5.0)
