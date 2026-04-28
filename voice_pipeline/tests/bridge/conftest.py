"""Shared fixtures and helpers for bridge tests."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from voice_pipeline.bridge.cpp_bridge import CppBridge


@pytest.fixture
def make_bridge(monkeypatch: pytest.MonkeyPatch) -> Callable[..., CppBridge]:
    """테스트용 fast-timeout CppBridge 생성.

    Test-wide class var defaults을 미리 설정한 뒤, 레거시 kwargs(host, port,
    reconnect_attempts, recv/connect/close_timeout_sec)는 class var
    monkeypatch로 변환해 적용한다.
    """

    monkeypatch.setattr(CppBridge, "_RECONNECT_ATTEMPTS", 2)
    monkeypatch.setattr(CppBridge, "_RECV_TIMEOUT_SEC", 0.1)
    monkeypatch.setattr(CppBridge, "_CONNECT_TIMEOUT_SEC", 1.0)
    monkeypatch.setattr(CppBridge, "_CLOSE_TIMEOUT_SEC", 1.0)
    monkeypatch.setattr(CppBridge, "_HOST", "localhost")
    monkeypatch.setattr(CppBridge, "_PORT", 18765)

    _CLASS_VAR_MAP = {
        "host": "_HOST",
        "port": "_PORT",
        "reconnect_attempts": "_RECONNECT_ATTEMPTS",
        "recv_timeout_sec": "_RECV_TIMEOUT_SEC",
        "connect_timeout_sec": "_CONNECT_TIMEOUT_SEC",
        "close_timeout_sec": "_CLOSE_TIMEOUT_SEC",
    }

    def _make(**overrides) -> CppBridge:
        for key, value in overrides.items():
            if key in _CLASS_VAR_MAP:
                monkeypatch.setattr(CppBridge, _CLASS_VAR_MAP[key], value)
            else:
                raise TypeError(f"Unknown override: {key}")
        return CppBridge()

    return _make


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
