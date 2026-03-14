"""Mock C++ audio playback server.

Replaces the real C++ process for testing the full Python pipeline.
Receives audio over WebSocket, plays it through speakers via PyAudio.

Protocol (JSON over WebSocket):

  Client → Server:
    {"type": "stream_start"}
    {"type": "audio", "data": "<base64-pcm>"}
    {"type": "audio_end"}
    {"type": "stop"}
    {"type": "play_file", "file_path": "..."}

  Server → Client:
    {"type": "playback_started"}
    {"type": "playback_complete"}

Usage:
    uv run python mock_cpp_server.py
    uv run python mock_cpp_server.py --port 8765 --sample-rate 24000
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import queue
import threading
import wave
from pathlib import Path

import pyaudio
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import ServerConnection, serve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("mock_cpp_server")

_STREAM_END = b""
_STOP = object()


class AudioPlayer:
    """Threaded audio player using PyAudio.

    Uses a stop event for immediate interruption — the playback thread
    checks the event between chunks, avoiding queue-order delays.
    """

    def __init__(self, sample_rate: int, channels: int = 1, sample_width: int = 2) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._sample_width = sample_width
        self._audio_queue: queue.Queue[bytes] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._playing = threading.Event()
        self._stopped = threading.Event()
        self._stopped.set()
        self._stop_event = threading.Event()
        self._end_of_stream = threading.Event()

    def start_stream(self) -> None:
        """Start a new playback stream."""
        if self._thread is not None and self._thread.is_alive():
            self.stop()

        self._audio_queue = queue.Queue()
        self._playing.clear()
        self._stopped.clear()
        self._stop_event.clear()
        self._end_of_stream.clear()
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()

    def feed(self, pcm_data: bytes) -> None:
        """Queue PCM data for playback."""
        self._audio_queue.put(pcm_data)

    def end_stream(self) -> None:
        """Signal that no more audio data will be sent."""
        self._end_of_stream.set()

    def stop(self) -> None:
        """Interrupt playback immediately."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def wait_started(self, timeout: float = 5.0) -> bool:
        """Wait until playback actually starts."""
        return self._playing.wait(timeout)

    def wait_done(self, timeout: float = 60.0) -> bool:
        """Wait until playback finishes."""
        return self._stopped.wait(timeout)

    def play_file(self, file_path: str) -> None:
        """Play a WAV file."""
        path = Path(file_path)
        if not path.exists():
            logger.warning("File not found: %s — simulating short playback", file_path)
            self._playing.set()
            self._stopped.set()
            return

        self.start_stream()
        try:
            with wave.open(str(path), "rb") as wf:
                chunk_size = 4096
                data = wf.readframes(chunk_size)
                while data:
                    self.feed(data)
                    data = wf.readframes(chunk_size)
        except Exception as exc:
            logger.error("Error reading WAV file %s: %s", file_path, exc)
        self.end_stream()

    def _playback_loop(self) -> None:
        """Thread target: play audio through speakers."""
        pa = pyaudio.PyAudio()
        stream = None
        try:
            stream = pa.open(
                format=pa.get_format_from_width(self._sample_width),
                channels=self._channels,
                rate=self._sample_rate,
                output=True,
                frames_per_buffer=1024,
            )
            self._playing.set()

            while not self._stop_event.is_set():
                try:
                    data = self._audio_queue.get(timeout=0.05)
                except queue.Empty:
                    if self._end_of_stream.is_set():
                        break
                    continue

                if self._stop_event.is_set():
                    break

                stream.write(data)

        except Exception as exc:
            logger.error("Playback error: %s", exc)
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            pa.terminate()
            self._stopped.set()


class MockCppServer:
    """WebSocket server that mimics the C++ audio playback process.

    Message receiving runs in a polling loop with short timeouts so that
    playback-completion responses can be sent between incoming messages.
    This allows ``stop`` commands to be processed while audio is playing.
    """

    def __init__(self, host: str, port: int, sample_rate: int) -> None:
        self._host = host
        self._port = port
        self._sample_rate = sample_rate

    def run(self) -> None:
        """Start the WebSocket server (blocking)."""
        logger.info("Mock C++ server starting on ws://%s:%d", self._host, self._port)
        logger.info("Audio output: %d Hz, 16-bit, mono", self._sample_rate)
        logger.info("Waiting for Python pipeline connection...")

        with serve(self._handle_client, self._host, self._port) as server:
            server.serve_forever()

    def _handle_client(self, ws: ServerConnection) -> None:
        """Handle one client connection."""
        logger.info("Client connected: %s", ws.remote_address)
        player = AudioPlayer(self._sample_rate)
        response_queue: queue.Queue[str] = queue.Queue()

        try:
            while True:
                self._flush_responses(ws, response_queue)

                try:
                    raw = ws.recv(timeout=0.05)
                except TimeoutError:
                    continue

                msg = json.loads(raw)
                msg_type = msg.get("type")

                if msg_type == "stream_start":
                    logger.info("[stream_start]")
                    player.start_stream()
                    player.wait_started(timeout=3.0)
                    ws.send(json.dumps({"type": "playback_started"}))
                    logger.info("  → playback_started")
                    self._monitor_completion(player, response_queue)

                elif msg_type == "audio":
                    player.feed(base64.b64decode(msg["data"]))

                elif msg_type == "audio_end":
                    logger.info("[audio_end] waiting for playback to finish...")
                    player.end_stream()

                elif msg_type == "stop":
                    logger.info("[stop] interrupting playback")
                    player.stop()
                    ws.send(json.dumps({"type": "playback_complete"}))
                    logger.info("  → playback_complete (after stop)")

                elif msg_type == "play_file":
                    file_path = msg.get("file_path", "")
                    logger.info("[play_file] %s", file_path)
                    ws.send(json.dumps({"type": "playback_started"}))
                    logger.info("  → playback_started")
                    player.play_file(file_path)
                    self._monitor_completion(player, response_queue)

                else:
                    logger.warning("Unknown message type: %s", msg_type)

        except ConnectionClosed:
            logger.info("Client disconnected")
        except Exception as exc:
            logger.info("Client error: %s", exc)
        finally:
            player.stop()
            logger.info("Client session ended")

    @staticmethod
    def _monitor_completion(player: AudioPlayer, rq: queue.Queue[str]) -> None:
        """Start a daemon thread that enqueues playback_complete when done."""

        def _wait() -> None:
            player.wait_done()
            rq.put(json.dumps({"type": "playback_complete"}))

        threading.Thread(target=_wait, daemon=True).start()

    @staticmethod
    def _flush_responses(ws: ServerConnection, rq: queue.Queue[str]) -> None:
        """Send all pending responses to the client."""
        while True:
            try:
                msg = rq.get_nowait()
            except queue.Empty:
                break
            ws.send(msg)
            logger.info("  → playback_complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock C++ audio playback server")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--sample-rate", type=int, default=24000, help="TTS output sample rate")
    args = parser.parse_args()

    server = MockCppServer(args.host, args.port, args.sample_rate)
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Server shutting down")


if __name__ == "__main__":
    main()
