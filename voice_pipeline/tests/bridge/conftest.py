"""Shared fixtures and helpers for bridge tests."""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock

import pytest

from voice_pipeline.core.config import CppBridgeConfig


@pytest.fixture
def config() -> CppBridgeConfig:
    """Default CppBridgeConfig with fast timeouts for tests."""
    return CppBridgeConfig(
        host="localhost",
        port=18765,
        reconnect_attempts=2,
        recv_timeout_sec=0.1,
        connect_timeout_sec=1.0,
        close_timeout_sec=1.0,
    )


@pytest.fixture
def mock_conn() -> MagicMock:
    """A mock websockets ClientConnection."""
    conn = MagicMock()
    conn.close = MagicMock()
    conn.send = MagicMock()
    conn.recv = MagicMock(side_effect=TimeoutError)
    return conn


class FakeServer:
    """Minimal in-test WebSocket message collector.

    Used by unit tests to simulate C++ sending messages back.
    Not a real WebSocket server — just drives the mock's recv side effects.
    """

    def __init__(self) -> None:
        self.received: list[dict] = []
        self._responses: list[str] = []
        self._lock = threading.Lock()

    def queue_response(self, msg: dict) -> None:
        """Queue a JSON response to be returned by mock recv."""
        with self._lock:
            self._responses.append(json.dumps(msg))

    def pop_response(self) -> str | None:
        with self._lock:
            return self._responses.pop(0) if self._responses else None

    def capture_send(self, data: str) -> None:
        """Capture a sent JSON message."""
        self.received.append(json.loads(data))


@pytest.fixture
def fake_server() -> FakeServer:
    return FakeServer()
