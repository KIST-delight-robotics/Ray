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
from voice_pipeline.audio.constants import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH
from voice_pipeline.core.interfaces import IASR
from voice_pipeline.core.types import AudioFrame

logger = logging.getLogger("voice_pipeline.asr")


class GoogleCloudASR(IASR):
    """Streaming ASR using Google Cloud Speech-to-Text V1.

    Threading model:
        The orchestrator thread calls feed_audio() to enqueue frames and
        get_text() to read the latest transcript.  A daemon reader thread
        sends audio to gRPC and reads responses, updating the transcript
        under a lock.

    Args:
        language_code: BCP-47 언어 코드 (예: "en-US", "ko-KR")
    """

    _MODEL = "latest_long"  # Google STT 모델 (장시간 음성 인식)
    _QUEUE_MAXSIZE = 300  # 오디오 큐 최대 프레임 수
    _QUEUE_GET_TIMEOUT_SEC = 1.0  # 오디오 큐 poll 간격 (초)
    _THREAD_JOIN_TIMEOUT_SEC = 5.0  # reader 스레드 종료 대기 (초)
    _SENTINEL = b""  # 스트림 종료 신호
    _ENCODING_MAP: dict[int, speech.RecognitionConfig.AudioEncoding] = {
        2: speech.RecognitionConfig.AudioEncoding.LINEAR16,
    }

    def __init__(self, language_code: str = "en-US") -> None:
        self.language_code = language_code

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
        if not (8000 <= SAMPLE_RATE <= 48000):
            raise ASRError(f"sample_rate={SAMPLE_RATE} outside Google STT valid range [8000, 48000]")

        encoding = self._ENCODING_MAP.get(SAMPLE_WIDTH)
        if encoding is None:
            raise ASRError(f"Unsupported sample_width={SAMPLE_WIDTH}; supported: {sorted(self._ENCODING_MAP.keys())}")

        recognition_config = speech.RecognitionConfig(
            encoding=encoding,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=self.language_code,
            model=self._MODEL,
            audio_channel_count=CHANNELS,
        )
        # interim_results=True는 get_text()가 부분 transcript 반환에 의존하므로 고정.
        streaming_config = speech.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=True,
        )

        self._audio_queue = queue.Queue(maxsize=self._QUEUE_MAXSIZE)
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
                self._audio_queue.put_nowait(self._SENTINEL)
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=self._THREAD_JOIN_TIMEOUT_SEC)
            if self._reader_thread.is_alive():
                logger.warning("Reader thread did not exit within timeout")
            self._reader_thread = None

    def _audio_generator(self) -> Generator[speech.StreamingRecognizeRequest, None, None]:
        """Yield audio requests from the queue until sentinel or shutdown."""
        while self._running.is_set():
            try:
                chunk = self._audio_queue.get(timeout=self._QUEUE_GET_TIMEOUT_SEC)  # type: ignore[union-attr]
            except queue.Empty:
                continue
            if chunk == self._SENTINEL:
                return
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    def _read_responses(self, streaming_config: speech.StreamingRecognitionConfig) -> None:
        """Read gRPC responses and update transcript (runs on reader thread).

        A single response may contain multiple sequential results (e.g. a
        stable prefix + a speculative suffix).  We concatenate all interim
        parts so that ``get_text()`` always returns the best available text.
        """
        try:
            responses = self._client.streaming_recognize(  # type: ignore[union-attr]
                config=streaming_config,
                requests=self._audio_generator(),
            )
            for response in responses:
                if not self._running.is_set():
                    return
                new_finals: list[str] = []
                interim_parts: list[str] = []
                has_results = False
                for result in response.results:
                    if not result.alternatives:
                        continue
                    has_results = True
                    transcript = result.alternatives[0].transcript
                    if result.is_final:
                        new_finals.append(transcript)
                        interim_parts.clear()
                    else:
                        interim_parts.append(transcript)
                if not has_results:
                    continue
                with self._lock:
                    if new_finals:
                        self._final_transcript += "".join(new_finals)
                    self._interim_transcript = "".join(interim_parts)
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
