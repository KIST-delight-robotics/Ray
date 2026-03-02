"""Google Cloud Speech-to-Text V1 streaming ASR implementation."""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
from collections.abc import Generator

from google.api_core.exceptions import GoogleAPICallError
from google.cloud import speech

from voice_pipeline.asr.exceptions import ASRError
from voice_pipeline.core.config import ASRConfig, AudioConfig
from voice_pipeline.core.interfaces import IASR
from voice_pipeline.core.types import AudioFrame

logger = logging.getLogger("voice_pipeline.asr")

_ENCODING_MAP: dict[int, speech.RecognitionConfig.AudioEncoding] = {
    2: speech.RecognitionConfig.AudioEncoding.LINEAR16,
}

_SAMPLE_RATE_MIN = 8000
_SAMPLE_RATE_MAX = 48000

_QUEUE_MAXSIZE = 300
_SENTINEL = b""


class GoogleCloudASR(IASR):
    """Streaming ASR using Google Cloud Speech-to-Text V1.

    Threading model:
        The orchestrator thread calls feed_audio() to enqueue frames and
        get_text() to read the latest transcript.  A daemon reader thread
        sends audio to gRPC and reads responses, updating the transcript
        under a lock.
    """

    def __init__(self, asr_config: ASRConfig, audio_config: AudioConfig) -> None:
        self._asr_config = asr_config
        self._audio_config = audio_config

        self._client: speech.SpeechClient | None = None
        self._audio_queue: queue.Queue[bytes] | None = None
        self._reader_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._final_transcript = ""
        self._interim_transcript = ""
        self._error: ASRError | None = None

    # ------------------------------------------------------------------
    # IASR lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the recognition session."""
        if self._running.is_set():
            logger.warning("start() called while already running — ignoring")
            return
        client = speech.SpeechClient()
        self._client = client
        try:
            self._start_stream()
        except Exception:
            client.transport.close()
            self._client = None
            raise

    def stop(self) -> None:
        """Stop the recognition session and release resources."""
        if not self._running.is_set():
            return
        self._running.clear()
        self._stop_stream()
        if self._client is not None:
            self._client.transport.close()
            self._client = None
        self._audio_queue = None
        with self._lock:
            self._final_transcript = ""
            self._interim_transcript = ""
            self._error = None

    def feed_audio(self, frame: AudioFrame) -> None:
        """Feed a single audio frame to the recognizer."""
        if not self._running.is_set():
            logger.warning("feed_audio() called while not running — ignoring")
            return
        self._check_error()
        try:
            self._audio_queue.put_nowait(frame)  # type: ignore[union-attr]
        except queue.Full:
            logger.warning("Audio queue full — dropping frame")

    def get_text(self) -> str:
        """Return the current transcription.

        Returns the accumulated final results concatenated with the current
        interim result (if any).
        """
        self._check_error()
        with self._lock:
            if self._interim_transcript:
                return self._final_transcript + self._interim_transcript
            return self._final_transcript

    def reset(self) -> None:
        """Reset recognizer state for the next turn."""
        if not self._running.is_set():
            logger.warning("reset() called while not running — ignoring")
            return
        self._stop_stream()
        with self._lock:
            self._final_transcript = ""
            self._interim_transcript = ""
            self._error = None
        self._start_stream()

    # ------------------------------------------------------------------
    # Internal streaming
    # ------------------------------------------------------------------

    def _start_stream(self) -> None:
        """Build gRPC config and start the reader thread."""
        sample_rate = self._audio_config.sample_rate
        if not (_SAMPLE_RATE_MIN <= sample_rate <= _SAMPLE_RATE_MAX):
            raise ASRError(
                f"sample_rate={sample_rate} outside valid range "
                f"[{_SAMPLE_RATE_MIN}, {_SAMPLE_RATE_MAX}]"
            )

        encoding = _ENCODING_MAP.get(self._audio_config.sample_width)
        if encoding is None:
            raise ASRError(
                f"Unsupported sample_width={self._audio_config.sample_width}; "
                f"supported: {sorted(_ENCODING_MAP.keys())}"
            )

        recognition_config = speech.RecognitionConfig(
            encoding=encoding,
            sample_rate_hertz=self._audio_config.sample_rate,
            language_code=self._asr_config.language_code,
            model=self._asr_config.model,
            audio_channel_count=self._audio_config.channels,
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=self._asr_config.interim_results,
        )

        self._audio_queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._running.set()
        self._reader_thread = threading.Thread(
            target=self._read_responses,
            args=(streaming_config,),
            daemon=True,
        )
        self._reader_thread.start()

    def _stop_stream(self) -> None:
        """Send sentinel and join the reader thread."""
        if self._audio_queue is not None:
            with contextlib.suppress(queue.Full):
                self._audio_queue.put_nowait(_SENTINEL)
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=5.0)
            if self._reader_thread.is_alive():
                logger.warning("Reader thread did not exit within timeout")
            self._reader_thread = None

    def _audio_generator(self) -> Generator[speech.StreamingRecognizeRequest, None, None]:
        """Yield audio requests from the queue until sentinel or shutdown."""
        while self._running.is_set():
            try:
                chunk = self._audio_queue.get(timeout=1.0)  # type: ignore[union-attr]
            except queue.Empty:
                continue
            if chunk == _SENTINEL:
                return
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    def _read_responses(self, streaming_config: speech.StreamingRecognitionConfig) -> None:
        """Read gRPC responses and update transcript (runs on reader thread)."""
        try:
            responses = self._client.streaming_recognize(  # type: ignore[union-attr]
                config=streaming_config,
                requests=self._audio_generator(),
            )
            for response in responses:
                if not self._running.is_set():
                    return
                for result in response.results:
                    if not result.alternatives:
                        continue
                    transcript = result.alternatives[0].transcript
                    with self._lock:
                        if result.is_final:
                            self._final_transcript += transcript
                            self._interim_transcript = ""
                        else:
                            self._interim_transcript = transcript
        except GoogleAPICallError as exc:
            with self._lock:
                self._error = ASRError(str(exc))
        except Exception as exc:
            with self._lock:
                self._error = ASRError(str(exc))

    # ------------------------------------------------------------------
    # Error propagation
    # ------------------------------------------------------------------

    def _check_error(self) -> None:
        """Raise and clear any stored error from the reader thread."""
        with self._lock:
            if self._error is not None:
                error = self._error
                self._error = None
                raise error
