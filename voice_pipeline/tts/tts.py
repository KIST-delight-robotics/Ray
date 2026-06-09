"""OpenAI TTS streaming implementation."""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from typing import Any

import openai

from voice_pipeline.core.interfaces import ITTS
from voice_pipeline.core.types import TTSStream
from voice_pipeline.tts.exceptions import TTSError

logger = logging.getLogger("voice_pipeline.tts")


class OpenAITTS(ITTS):
    """TTS implementation using the OpenAI Audio API with streaming.

    Reads ``OPENAI_API_KEY`` from the environment. Streams raw PCM audio
    (24 kHz, 16-bit signed LE, mono) via ``with_streaming_response.create()``.

    The returned :class:`TTSStream` must be fully consumed or explicitly
    closed to release the underlying HTTP connection.
    """

    OUTPUT_SAMPLE_RATE: int = 24000  # OpenAI TTS API 고정 출력 샘플레이트 (Hz)

    _VOICE: str = "ash"  # OpenAI 음성 프리셋 (예: "ash", "alloy", "coral")
    _MODEL: str = "tts-1"  # OpenAI TTS 모델 (예: "tts-1", "tts-1-hd", "gpt-4o-mini-tts")
    _SPEED: float = 1.0  # 재생 속도 (0.25~4.0)
    _INSTRUCTIONS: str | None = None  # 음성 스타일 지시문 (`gpt-4o-mini-tts` 모델 전용)

    # `instructions` 인자를 지원하는 OpenAI TTS 모델
    _SUPPORTS_INSTRUCTIONS: frozenset[str] = frozenset({"gpt-4o-mini-tts"})

    _MAX_RETRIES = 2  # 합성 실패 시 자동 재시도 횟수
    _TIMEOUT_SEC = 5.0  # 합성 응답 대기 최대 시간 (초)
    _CHUNK_SIZE = 4096  # 스트리밍 오디오 버퍼 크기 (바이트)

    def __init__(self) -> None:
        self._client = openai.OpenAI(
            max_retries=self._MAX_RETRIES,
            timeout=self._TIMEOUT_SEC,
        )

    @property
    def output_sample_rate(self) -> int:
        return OpenAITTS.OUTPUT_SAMPLE_RATE

    @property
    def voice_id(self) -> str:
        return f"openai|{self._VOICE}|{self._MODEL}|{self._SPEED}|{self._INSTRUCTIONS or ''}"

    @property
    def model_name(self) -> str:
        return self._MODEL

    def synthesize(self, text: str) -> TTSStream:
        """Stream PCM audio from OpenAI TTS API.

        Args:
            text: Text to synthesize (max 4096 characters).

        Returns:
            TTSStream yielding PCM audio chunks.

        Raises:
            TTSError: On validation failure or API error.
        """
        self._validate_input(text)

        try:
            kwargs: dict[str, Any] = {
                "model": self._MODEL,
                "voice": self._VOICE,
                "input": text,
                "response_format": "pcm",
                "speed": self._SPEED,
            }
            if self._INSTRUCTIONS:
                if self._MODEL in self._SUPPORTS_INSTRUCTIONS:
                    kwargs["instructions"] = self._INSTRUCTIONS
                else:
                    logger.warning("instructions ignored for model %s", self._MODEL)

            response_cm = self._client.audio.speech.with_streaming_response.create(**kwargs)
        except openai.APITimeoutError as exc:
            logger.error("OpenAI TTS timeout after %.0fs: %s", self._TIMEOUT_SEC, exc)
            raise TTSError(f"TTS timeout ({self._TIMEOUT_SEC}s): {exc}") from exc
        except openai.OpenAIError as exc:
            logger.error("OpenAI TTS API error: %s", exc)
            raise TTSError(str(exc)) from exc

        # Enter CM eagerly so close_fn can always exit it safely.
        try:
            response = response_cm.__enter__()
        except Exception as exc:
            raise TTSError(str(exc)) from exc

        exited = [False]

        def safe_exit(*exc_info: object) -> None:
            if not exited[0]:
                exited[0] = True
                response_cm.__exit__(*exc_info)

        gen = _iter_chunks(response, safe_exit, self._CHUNK_SIZE)
        return TTSStream(gen, close_fn=lambda: _safe_close(safe_exit))

    def _validate_input(self, text: str) -> None:
        """Pre-validate input before API call."""
        if not text or not text.strip():
            raise TTSError("Text must not be empty or whitespace-only")
        if len(text) > 4096:
            raise TTSError(f"Text exceeds 4096 character limit ({len(text)} chars)")
        if not (0.25 <= self._SPEED <= 4.0):
            raise TTSError(f"Speed must be 0.25–4.0, got {self._SPEED}")


def _iter_chunks(response: Any, safe_exit: Callable[..., None], chunk_size: int) -> Generator[bytes, None, None]:
    """Yield audio chunks from an already-entered streaming response.

    Calls *safe_exit* exactly once on completion, error, or generator close.
    *safe_exit* is idempotent, so duplicate calls (e.g. from TTSStream.close_fn)
    are harmless.
    """
    try:
        for chunk in response.iter_bytes(chunk_size=chunk_size):  # noqa: UP028
            yield chunk
    except GeneratorExit:
        safe_exit(None, None, None)
        return
    except openai.OpenAIError as exc:
        safe_exit(type(exc), exc, exc.__traceback__)
        raise TTSError(str(exc)) from exc
    except Exception as exc:
        safe_exit(type(exc), exc, exc.__traceback__)
        raise TTSError(str(exc)) from exc
    else:
        safe_exit(None, None, None)


def _safe_close(safe_exit: Callable[..., None]) -> None:
    """Best-effort close via safe_exit (for TTSStream.close_fn)."""
    try:
        safe_exit(None, None, None)
    except Exception:
        logger.debug("Error closing CM (suppressed)", exc_info=True)
