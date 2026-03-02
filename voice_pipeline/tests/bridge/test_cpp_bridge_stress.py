"""Stress tests for CppBridge.

Tests rapid lifecycle cycling, high-volume streaming, and concurrent access.
Uses an in-process WebSocket server.
"""

from __future__ import annotations

import json
import threading
import time

import pytest
from websockets.sync.server import ServerConnection, serve

from voice_pipeline.bridge.cpp_bridge import CppBridge
from voice_pipeline.core.config import CppBridgeConfig
from voice_pipeline.core.types import CppEventType

pytestmark = pytest.mark.requires_api

_HOST = "localhost"
_PORT = 19878


def _sink_handler(conn: ServerConnection) -> None:
    """Accept all messages, respond to nothing."""
    for _ in conn:
        pass


def _event_flood_handler(conn: ServerConnection) -> None:
    """Send a playback_position event for every message received."""
    for msg in conn:
        data = json.loads(msg)
        if data["type"] == "audio":
            conn.send(json.dumps({"type": "playback_position", "position_sec": 0.0}))


def _mass_event_handler(conn: ServerConnection) -> None:
    """On first message, flood 500 events back."""
    first = True
    for _msg in conn:
        if first:
            first = False
            for i in range(500):
                conn.send(json.dumps({"type": "playback_position", "position_sec": float(i)}))
            conn.send(json.dumps({"type": "playback_complete"}))


@pytest.fixture
def stress_config() -> CppBridgeConfig:
    return CppBridgeConfig(
        host=_HOST,
        port=_PORT,
        reconnect_attempts=2,
        recv_timeout_sec=0.1,
        connect_timeout_sec=2.0,
        close_timeout_sec=2.0,
    )


class TestRapidCycles:
    def test_rapid_connect_disconnect(self, stress_config: CppBridgeConfig) -> None:
        """10 rapid connect/disconnect cycles without leaking threads."""
        server = serve(_sink_handler, _HOST, _PORT)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            bridge = CppBridge(stress_config)
            for _ in range(10):
                bridge.connect()
                assert bridge._running.is_set()
                bridge.disconnect()
                assert not bridge._running.is_set()
        finally:
            server.shutdown()
            thread.join(timeout=5.0)


class TestHighVolume:
    def test_stream_1000_audio_chunks(self, stress_config: CppBridgeConfig) -> None:
        """Send 1000 audio chunks without errors."""
        server = serve(_sink_handler, _HOST, _PORT)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            bridge = CppBridge(stress_config)
            bridge.connect()
            chunk = b"\x00" * 960  # 30ms at 16kHz 16-bit mono
            for _ in range(1000):
                bridge.send_audio(chunk)
            bridge.disconnect()
        finally:
            server.shutdown()
            thread.join(timeout=5.0)

    def test_receive_500_events(self, stress_config: CppBridgeConfig) -> None:
        """Receive 500 events in quick succession."""
        server = serve(_mass_event_handler, _HOST, _PORT)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            bridge = CppBridge(stress_config)
            bridge.connect()

            # Trigger the flood
            bridge.send_audio(b"\x00" * 10)

            # Collect all events (500 position + 1 complete)
            events = []
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                event = bridge.poll_event()
                if event is not None:
                    events.append(event)
                    if event.event_type == CppEventType.PLAYBACK_COMPLETE:
                        break
                else:
                    time.sleep(0.01)

            assert len(events) == 501
            assert events[-1].event_type == CppEventType.PLAYBACK_COMPLETE
            bridge.disconnect()
        finally:
            server.shutdown()
            thread.join(timeout=5.0)


class TestConcurrency:
    def test_concurrent_send_and_poll(self, stress_config: CppBridgeConfig) -> None:
        """Concurrent send + poll from different threads — no deadlock."""
        server = serve(_event_flood_handler, _HOST, _PORT)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            bridge = CppBridge(stress_config)
            bridge.connect()

            errors: list[Exception] = []
            events_received = []
            stop = threading.Event()

            def sender() -> None:
                try:
                    for _ in range(200):
                        bridge.send_audio(b"\x00" * 100)
                        time.sleep(0.001)
                except Exception as exc:
                    errors.append(exc)
                finally:
                    stop.set()

            def poller() -> None:
                try:
                    while not stop.is_set():
                        event = bridge.poll_event()
                        if event is not None:
                            events_received.append(event)
                        time.sleep(0.001)
                    # Drain remaining
                    while True:
                        event = bridge.poll_event()
                        if event is None:
                            break
                        events_received.append(event)
                except Exception as exc:
                    errors.append(exc)

            t_send = threading.Thread(target=sender)
            t_poll = threading.Thread(target=poller)
            t_send.start()
            t_poll.start()
            t_send.join(timeout=10.0)
            t_poll.join(timeout=10.0)

            assert not errors, f"Errors during concurrent access: {errors}"
            assert len(events_received) > 0
            bridge.disconnect()
        finally:
            server.shutdown()
            thread.join(timeout=5.0)
