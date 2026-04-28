"""WebSocket bridge to the C++ audio playback process."""

from __future__ import annotations

import base64
import json
import logging
import queue
import threading
import time

from websockets.exceptions import ConnectionClosed, WebSocketException
from websockets.sync.client import ClientConnection
from websockets.sync.client import connect as ws_connect

from voice_pipeline.bridge.exceptions import BridgeError
from voice_pipeline.core.interfaces import ICppBridge
from voice_pipeline.core.types import CppEvent, CppEventType

logger = logging.getLogger("voice_pipeline.bridge")

# ---------------------------------------------------------------------------
# Event parsing
# ---------------------------------------------------------------------------

_EVENT_TYPE_MAP: dict[str, CppEventType] = {e.value: e for e in CppEventType}


def _parse_event(raw: str | bytes) -> CppEvent:
    """Parse a JSON message into a CppEvent.

    Raises:
        ValueError: If the message is not valid JSON or has an unknown type.
        KeyError: If required fields are missing.
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    data = json.loads(raw)
    type_str = data["type"]
    event_type = _EVENT_TYPE_MAP.get(type_str)
    if event_type is None:
        raise ValueError(f"Unknown event type: {type_str!r}")
    return CppEvent(event_type=event_type)


# ---------------------------------------------------------------------------
# CppBridge
# ---------------------------------------------------------------------------


class CppBridge(ICppBridge):
    """WebSocket bridge to the C++ audio playback process.

    Threading model:
        The orchestrator thread calls connect(), disconnect(), send_*(), and
        poll_event().  A daemon receiver thread reads WebSocket messages and
        enqueues parsed CppEvents.  Each receiver thread gets its own stop
        event so stale threads cannot corrupt a newer connection's state.
    """

    _HOST = "localhost"  # C++ 프로세스 호스트 주소
    _PORT = 9200  # C++ 프로세스 WebSocket 포트
    _RECONNECT_ATTEMPTS = 3  # 연결 실패 시 재시도 횟수
    _RECV_TIMEOUT_SEC = 1.0  # 메시지 수신 polling 간격 (초)
    _CONNECT_TIMEOUT_SEC = 5.0  # 연결 수립 최대 대기 시간 (초)
    _CLOSE_TIMEOUT_SEC = 5.0  # 연결 종료 최대 대기 시간 (초)
    _RECONNECT_DELAY_SEC = 1.0  # 연결 재시도 사이 대기 시간 (초)
    _THREAD_JOIN_TIMEOUT_SEC = 5.0  # 수신 스레드 종료 대기 시간 (초)

    def __init__(self) -> None:
        self._conn: ClientConnection | None = None
        self._receiver_thread: threading.Thread | None = None
        self._receiver_stop: threading.Event | None = None
        self._event_queue: queue.Queue[CppEvent] = queue.Queue()
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._error: BridgeError | None = None

    # ------------------------------------------------------------------
    # ICppBridge lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Establish connection to the C++ process."""
        if self._running.is_set():
            logger.debug("connect() called while already connected — skipping")
            return

        # Clean up residual state from a previous failed connection
        self._cleanup()

        uri = f"ws://{self._HOST}:{self._PORT}"
        last_exc: Exception | None = None

        for attempt in range(1, self._RECONNECT_ATTEMPTS + 1):
            try:
                conn = ws_connect(
                    uri,
                    open_timeout=self._CONNECT_TIMEOUT_SEC,
                    close_timeout=self._CLOSE_TIMEOUT_SEC,
                    proxy=None,
                    ping_interval=None,
                    compression=None,
                )
                break
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Connection attempt %d/%d failed: %s",
                    attempt,
                    self._RECONNECT_ATTEMPTS,
                    exc,
                )
                if attempt < self._RECONNECT_ATTEMPTS:
                    time.sleep(self._RECONNECT_DELAY_SEC)
        else:
            raise BridgeError(f"Failed to connect to {uri} after {self._RECONNECT_ATTEMPTS} attempts") from last_exc

        # Fresh state for new connection
        self._conn = conn
        self._event_queue = queue.Queue()
        with self._lock:
            self._error = None
        stop = threading.Event()
        self._receiver_stop = stop
        self._running.set()
        self._receiver_thread = threading.Thread(
            target=self._receive_loop,
            args=(conn, stop),
            daemon=True,
        )
        self._receiver_thread.start()
        logger.info("Connected to %s", uri)

    def disconnect(self) -> None:
        """Close the connection to the C++ process."""
        if self._conn is None and self._receiver_thread is None:
            return
        self._running.clear()
        self._cleanup()
        with self._lock:
            self._error = None
        logger.info("Disconnected")

    # ------------------------------------------------------------------
    # ICppBridge send methods
    # ------------------------------------------------------------------

    def send_stream_start(self) -> None:
        """Signal that audio streaming is about to begin."""
        self._guard_connected()
        self._send_json({"type": "stream_start"})

    def send_audio(self, audio: bytes) -> None:
        """Send audio data for playback."""
        self._guard_connected()
        encoded = base64.b64encode(audio).decode("ascii")
        self._send_json({"type": "audio", "data": encoded})

    def send_audio_end(self) -> None:
        """Signal that all audio data has been sent for the current stream."""
        self._guard_connected()
        self._send_json({"type": "audio_end"})

    def send_stop(self) -> None:
        """Send a stop/interrupt signal to halt playback."""
        self._guard_connected()
        self._send_json({"type": "stop"})

    def send_play_file(self, file_path: str) -> None:
        """Request the C++ process to play an audio file."""
        self._guard_connected()
        self._send_json({"type": "play_file", "file_path": file_path})

    # ------------------------------------------------------------------
    # ICppBridge poll
    # ------------------------------------------------------------------

    def poll_event(self) -> CppEvent | None:
        """Poll for the next event from the C++ process."""
        self._check_error()
        try:
            return self._event_queue.get_nowait()
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _guard_connected(self) -> None:
        """Raise if not connected."""
        self._check_error()
        if not self._running.is_set():
            raise BridgeError("Not connected")

    def _send_json(self, msg: dict) -> None:
        """Serialize and send a JSON message."""
        try:
            self._conn.send(json.dumps(msg))  # type: ignore[union-attr]
        except ConnectionClosed as exc:
            raise BridgeError(f"Connection lost during send: {exc}") from exc
        except WebSocketException as exc:
            raise BridgeError(f"WebSocket error during send: {exc}") from exc

    def _cleanup(self) -> None:
        """Close connection and join receiver thread if they exist."""
        if self._receiver_stop is not None:
            self._receiver_stop.set()
            self._receiver_stop = None
        if self._conn is not None:
            try:
                self._conn.close(timeout=self._CLOSE_TIMEOUT_SEC)
            except Exception:
                logger.debug("Error closing WebSocket (suppressed)", exc_info=True)
            self._conn = None
        if self._receiver_thread is not None:
            self._receiver_thread.join(timeout=self._THREAD_JOIN_TIMEOUT_SEC)
            if self._receiver_thread.is_alive():
                logger.warning("Receiver thread did not exit within timeout")
            self._receiver_thread = None

    def _receive_loop(self, conn: ClientConnection, stop: threading.Event) -> None:
        """Read messages from the WebSocket and enqueue events (daemon thread)."""
        while not stop.is_set():
            try:
                raw = conn.recv(timeout=self._RECV_TIMEOUT_SEC)
            except TimeoutError:
                continue
            except ConnectionClosed as exc:
                if not stop.is_set():
                    with self._lock:
                        self._error = BridgeError(f"Connection lost: {exc}")
                    self._running.clear()
                return
            except Exception as exc:
                if not stop.is_set():
                    with self._lock:
                        self._error = BridgeError(f"Receiver error: {exc}")
                    self._running.clear()
                return

            try:
                event = _parse_event(raw)
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning("Skipping unparseable message: %s (%s)", raw, exc)
                continue
            self._event_queue.put(event)

    def _check_error(self) -> None:
        """Raise and clear any stored error from the receiver thread."""
        with self._lock:
            if self._error is not None:
                error = self._error
                self._error = None
                raise error
