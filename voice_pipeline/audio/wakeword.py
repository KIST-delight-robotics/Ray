"""Wakeword detection using Silero VAD + Google Cloud STT keyword matching."""

from __future__ import annotations

import collections
import enum
import logging
import re
import struct

from google.cloud import speech

try:
    import torch
    from silero_vad import load_silero_vad
except ImportError:
    torch = None  # type: ignore[assignment]
    load_silero_vad = None  # type: ignore[assignment]

from voice_pipeline.audio.constants import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH
from voice_pipeline.audio.exceptions import WakewordError
from voice_pipeline.core.interfaces import IWakewordDetector
from voice_pipeline.core.types import AudioFrame

logger = logging.getLogger("voice_pipeline.audio")


class _State(enum.Enum):
    IDLE = "idle"
    SPEECH = "speech"
    TRAILING = "trailing"


class WakewordDetector(IWakewordDetector):
    """Wakeword detection via Silero VAD speech segmentation + Google STT keyword match.

    Architecture:
        1. Pipeline audio frames (480 samples / 30ms) are rechunked to 512 samples
           for Silero VAD.
        2. VAD produces a speech probability per chunk. A state machine tracks
           IDLE → SPEECH → TRAILING → IDLE transitions.
        3. When speech ends (trailing silence exceeds ``_SPEECH_PAD_MS``), accumulated
           PCM is sent to Google STT ``recognize()`` (non-streaming, synchronous).
        4. All transcript alternatives are checked for keyword matches using
           word-boundary regex.

    Error handling:
        - Initialization failures (model load, client creation) raise ``WakewordError``.
        - Transient VAD/STT/network errors log a warning and return ``False`` (fail closed).

    Threading:
        Not thread-safe. SessionManager calls ``feed_audio()`` from a single thread
        (the SLEEP loop). No locking is needed.

    Args:
        language_code: Google STT BCP-47 언어 코드.
    """

    # Wakeword 키워드
    _KEYWORDS: tuple[str, ...] = ("ray",)  # 감지할 트리거 단어 목록

    # Silero VAD 모델 입력 규격
    _VAD_CHUNK_SAMPLES = 512  # VAD 입력 청크 샘플 수
    _VAD_CHUNK_BYTES = _VAD_CHUNK_SAMPLES * 2  # 파생: 청크 바이트 수 (16-bit mono)
    _VAD_CHUNK_DURATION_MS = 32  # 파생: 청크 길이 (512 @ 16kHz)

    _VAD_THRESHOLD = 0.5  # VAD 음성 확률 임계값 (0.0~1.0)
    _MAX_SPEECH_DURATION_SEC = 3.0  # 이 시간 초과 시 강제 STT 인식 (초)
    _PRE_BUFFER_MS = 300  # 음성 시작 onset 캡처용 ring buffer 길이 (ms)
    _SPEECH_PAD_MS = 300  # 음성 종료 검출용 후행 침묵 길이 (ms)
    _MIN_SPEECH_DURATION_MS = 100  # 이 시간 미만 음성은 STT 인식 스킵 (ms)
    _STT_TIMEOUT_SEC = 5.0  # Google STT recognize() 응답 대기 시간 (초)
    _MAX_ALTERNATIVES = 5  # STT 응답에 요청할 대안 수

    def __init__(
        self,
        language_code: str = "en-US",
    ) -> None:
        self.language_code = language_code

        if torch is None or load_silero_vad is None:
            raise WakewordError(
                "torch and silero-vad are required for wakeword detection. Install with: uv sync --extra models-pytorch"
            )

        # Load Silero VAD model
        try:
            self._vad_model = load_silero_vad(onnx=False)
        except Exception as exc:
            raise WakewordError(f"Failed to load Silero VAD model: {exc}") from exc

        # Create Google STT client
        try:
            self._stt_client = speech.SpeechClient()
        except Exception as exc:
            raise WakewordError(f"Failed to create Google STT client: {exc}") from exc

        # Pre-build recognition config
        self._recognition_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=language_code,
            audio_channel_count=CHANNELS,
            max_alternatives=self._MAX_ALTERNATIVES,
            speech_contexts=[speech.SpeechContext(phrases=list(self._KEYWORDS))],
        )

        # Pre-compile keyword patterns
        self._keyword_patterns = [re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in self._KEYWORDS]

        # Bytes per second for duration calculations (accounts for channels + sample width)
        self._bytes_per_sec = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS

        # Pre-buffer: ring buffer of recent chunks for capturing speech onset
        pre_buffer_chunks = max(1, self._PRE_BUFFER_MS // self._VAD_CHUNK_DURATION_MS)
        self._pre_buffer: collections.deque[bytes] = collections.deque(
            maxlen=pre_buffer_chunks,
        )

        # State
        self._state = _State.IDLE
        self._vad_buffer = bytearray()  # residual bytes for rechunking
        self._speech_buffer = bytearray()  # accumulated PCM during speech
        self._silence_chunks = 0  # trailing silence chunk counter
        self._detected = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release resources (STT client transport).

        Safe to call multiple times.
        """
        if self._stt_client is not None:
            try:
                self._stt_client.transport.close()
            except Exception:
                logger.debug("Error closing STT client transport", exc_info=True)
            self._stt_client = None

    # ------------------------------------------------------------------
    # IWakewordDetector
    # ------------------------------------------------------------------

    def feed_audio(self, frame: AudioFrame) -> bool:
        """Feed an audio frame and check for wakeword detection."""
        self._vad_buffer.extend(frame)

        while len(self._vad_buffer) >= self._VAD_CHUNK_BYTES:
            chunk_bytes = bytes(self._vad_buffer[: self._VAD_CHUNK_BYTES])
            del self._vad_buffer[: self._VAD_CHUNK_BYTES]

            tensor = self._bytes_to_tensor(chunk_bytes)
            try:
                prob = self._vad_model(tensor, SAMPLE_RATE).item()
            except Exception:
                logger.warning("VAD inference failed, resetting", exc_info=True)
                self._reset()
                break
            self._process_vad(prob, chunk_bytes)

        result = self._detected
        self._detected = False
        return result

    # ------------------------------------------------------------------
    # VAD state machine
    # ------------------------------------------------------------------

    def _process_vad(self, prob: float, chunk_bytes: bytes) -> None:
        """Update state machine based on VAD probability."""
        if self._state is _State.IDLE:
            if prob > self._VAD_THRESHOLD:
                self._state = _State.SPEECH
                # Prepend pre-buffer to capture speech onset
                for buffered_chunk in self._pre_buffer:
                    self._speech_buffer.extend(buffered_chunk)
                self._pre_buffer.clear()
                self._speech_buffer.extend(chunk_bytes)
            else:
                self._pre_buffer.append(chunk_bytes)

        elif self._state is _State.SPEECH:
            self._speech_buffer.extend(chunk_bytes)
            if prob < self._VAD_THRESHOLD:
                self._state = _State.TRAILING
                self._silence_chunks = 1
                if self._SPEECH_PAD_MS <= self._VAD_CHUNK_DURATION_MS:
                    self._run_recognition()
                    return

        elif self._state is _State.TRAILING:
            self._speech_buffer.extend(chunk_bytes)
            if prob > self._VAD_THRESHOLD:
                self._state = _State.SPEECH
                self._silence_chunks = 0
            else:
                self._silence_chunks += 1
                silence_ms = self._silence_chunks * self._VAD_CHUNK_DURATION_MS
                if silence_ms >= self._SPEECH_PAD_MS:
                    self._run_recognition()
                    return

        # Safety cap: force recognition if speech is too long
        if self._state is not _State.IDLE:
            speech_duration_sec = len(self._speech_buffer) / self._bytes_per_sec
            if speech_duration_sec >= self._MAX_SPEECH_DURATION_SEC:
                self._run_recognition()

    # ------------------------------------------------------------------
    # STT recognition + keyword matching
    # ------------------------------------------------------------------

    def _run_recognition(self) -> None:
        """Send accumulated speech to Google STT and check for keywords."""
        speech_bytes = bytes(self._speech_buffer)
        speech_duration_ms = len(speech_bytes) * 1000 // self._bytes_per_sec

        if speech_duration_ms < self._MIN_SPEECH_DURATION_MS:
            logger.debug(
                "Speech too short (%d ms < %d ms), skipping recognition",
                speech_duration_ms,
                self._MIN_SPEECH_DURATION_MS,
            )
            self._reset()
            return

        try:
            audio = speech.RecognitionAudio(content=speech_bytes)
            response = self._stt_client.recognize(
                config=self._recognition_config,
                audio=audio,
                timeout=self._STT_TIMEOUT_SEC,
            )
        except Exception:
            logger.warning("Wakeword STT recognition failed", exc_info=True)
            self._reset()
            return

        # Check all alternatives for keyword matches
        for result in response.results:
            for alternative in result.alternatives:
                transcript = alternative.transcript
                logger.info("STT result: %r (confidence=%.2f)", transcript, alternative.confidence)
                for pattern in self._keyword_patterns:
                    if pattern.search(transcript):
                        logger.info(
                            "Wakeword detected in transcript: %r",
                            transcript,
                        )
                        self._detected = True
                        self._reset()
                        return

        if not response.results:
            logger.debug("STT returned no results")
        else:
            logger.debug("No wakeword match in %d results", len(response.results))
        self._reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        """Reset speech state for next detection cycle."""
        self._speech_buffer.clear()
        self._pre_buffer.clear()
        self._state = _State.IDLE
        self._silence_chunks = 0
        try:
            self._vad_model.reset_states()
        except Exception:
            logger.warning("VAD reset_states failed", exc_info=True)

    @staticmethod
    def _bytes_to_tensor(pcm_bytes: bytes) -> torch.Tensor:
        """Convert 16-bit mono PCM bytes to a float32 torch tensor."""
        samples = struct.unpack(f"<{len(pcm_bytes) // 2}h", pcm_bytes)
        # 16-bit signed PCM → float32 [-1, 1]: divide by 2^15
        return torch.tensor(samples, dtype=torch.float32) / 32768.0
