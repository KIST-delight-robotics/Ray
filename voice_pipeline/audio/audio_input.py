"""AudioInput: microphone capture on a daemon thread."""

from __future__ import annotations

import array
import logging
import queue
import threading

from voice_pipeline.audio.exceptions import AudioInputError
from voice_pipeline.core.config import AudioConfig, AudioInputConfig
from voice_pipeline.core.interfaces import IAudioInput
from voice_pipeline.core.types import AudioFrame

logger = logging.getLogger("voice_pipeline.audio")


class AudioInput(IAudioInput):
    """Captures audio from a microphone via PyAudio on a daemon thread.

    Frames are pushed to the provided queue. If the queue is full,
    frames are dropped (never blocks the capture thread).
    """

    def __init__(
        self,
        audio_queue: queue.Queue[AudioFrame],
        audio_config: AudioConfig,
        config: AudioInputConfig,
    ) -> None:
        self._queue = audio_queue
        self._audio_config = audio_config
        self._config = config

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

        # Lazy import PyAudio
        try:
            import pyaudio  # noqa: F401

            self._pyaudio_module = pyaudio
        except ImportError as exc:
            raise AudioInputError(
                "PyAudio is not installed. Install it with: pip install pyaudio"
            ) from exc

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
            self._thread.join(timeout=2.0)
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
            frame_size = self._audio_config.frame_size_samples

            capture_ch = self._config.capture_channels or self._audio_config.channels
            need_extract = capture_ch != self._audio_config.channels

            stream = pa.open(
                format=pa.get_format_from_width(self._audio_config.sample_width),
                channels=capture_ch,
                rate=self._audio_config.sample_rate,
                input=True,
                input_device_index=self._config.device_index,
                frames_per_buffer=frame_size,
            )

            if need_extract:
                ch_idx = self._config.extract_channel
                if ch_idx >= capture_ch:
                    raise AudioInputError(
                        f"extract_channel ({ch_idx}) >= capture_channels ({capture_ch})"
                    )
                logger.info(
                    "Capturing %dch, extracting CH%d as mono", capture_ch, ch_idx,
                )

            while not self._stop_event.is_set():
                data = stream.read(frame_size, exception_on_overflow=False)
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
