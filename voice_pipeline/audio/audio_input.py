"""AudioInput: microphone capture on a daemon thread."""

from __future__ import annotations

import array
import logging
import queue
import threading

from voice_pipeline.audio.constants import (
    CHANNELS,
    FRAME_SIZE_SAMPLES,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
)
from voice_pipeline.audio.exceptions import AudioInputError
from voice_pipeline.core.interfaces import IAudioInput
from voice_pipeline.core.types import AudioFrame

logger = logging.getLogger("voice_pipeline.audio")


class AudioInput(IAudioInput):
    """Captures audio from a microphone via PyAudio on a daemon thread.

    Frames are pushed to the provided queue. If the queue is full,
    frames are dropped (never blocks the capture thread).
    """

    _THREAD_JOIN_TIMEOUT_SEC = 2.0  # 캡처 스레드 종료 대기 (초)
    _DEVICE_INDEX: int | None = None  # PyAudio 입력 디바이스 인덱스. None은 시스템 기본 장치
    _CAPTURE_CHANNELS: int | None = None  # 디바이스에서 캡처할 채널 수. None은 mono (ReSpeaker 6ch는 6)
    _EXTRACT_CHANNEL = 0  # 다중 채널 캡처 시 mono 추출에 사용할 채널 인덱스 (0-based)

    def __init__(self, audio_queue: queue.Queue[AudioFrame]) -> None:
        """Initialize capture state and ensure PyAudio is importable.

        Args:
            audio_queue: 캡처된 오디오 프레임을 push할 공유 큐.
        """
        self._queue = audio_queue

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

        # Lazy import PyAudio
        try:
            import pyaudio  # noqa: F401

            self._pyaudio_module = pyaudio
        except ImportError as exc:
            raise AudioInputError("PyAudio is not installed. Install it with: pip install pyaudio") from exc

    def start(self) -> None:
        """Start capturing audio. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._error = None
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop capturing audio and release resources."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._THREAD_JOIN_TIMEOUT_SEC)
            self._thread = None

    @property
    def error(self) -> Exception | None:
        """Return the captured error if the capture thread has died."""
        return self._error

    def _capture_loop(self) -> None:
        """Thread target: open stream, read frames, push to queue."""
        pa = None
        stream = None
        try:
            pa = self._pyaudio_module.PyAudio()

            capture_ch = self._CAPTURE_CHANNELS or CHANNELS
            need_extract = capture_ch != CHANNELS

            stream = pa.open(
                format=pa.get_format_from_width(SAMPLE_WIDTH),
                channels=capture_ch,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=self._DEVICE_INDEX,
                frames_per_buffer=FRAME_SIZE_SAMPLES,
            )

            if need_extract:
                ch_idx = self._EXTRACT_CHANNEL
                if ch_idx >= capture_ch:
                    raise AudioInputError(f"extract_channel ({ch_idx}) >= capture_channels ({capture_ch})")
                logger.info(
                    "Capturing %dch, extracting CH%d as mono",
                    capture_ch,
                    ch_idx,
                )

            while not self._stop_event.is_set():
                data = stream.read(FRAME_SIZE_SAMPLES, exception_on_overflow=False)
                if need_extract:
                    samples = array.array("h", data)
                    mono = samples[ch_idx::capture_ch]
                    data = mono.tobytes()
                try:
                    self._queue.put_nowait(data)
                except queue.Full:
                    logger.warning("Audio queue full — dropping frame")

        except Exception as exc:
            logger.error("AudioInput capture error: %s", exc, exc_info=True)
            self._error = exc
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    logger.debug("Error closing audio stream", exc_info=True)
            if pa is not None:
                try:
                    pa.terminate()
                except Exception:
                    logger.debug("Error terminating PyAudio", exc_info=True)
