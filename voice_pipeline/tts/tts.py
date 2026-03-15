"""OpenAI TTS streaming implementation."""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from typing import Any

import openai

from voice_pipeline.core.config import TTSConfig
from voice_pipeline.core.interfaces import ITTS
from voice_pipeline.core.types import TTSStream
from voice_pipeline.tts.exceptions import TTSError

logger = logging.getLogger("voice_pipeline.tts")

_SUPPORTS_INSTRUCTIONS: set[str] = {"gpt-4o-mini-tts"}


class OpenAITTS(ITTS):
    """TTS implementation using the OpenAI Audio API with streaming.

    Reads ``OPENAI_API_KEY`` from the environment. Streams raw PCM audio
    (24 kHz, 16-bit signed LE, mono) via ``with_streaming_response.create()``.

    The returned :class:`TTSStream` must be fully consumed or explicitly
    closed to release the underlying HTTP connection.
    """

    def __init__(self, config: TTSConfig) -> None:
        self._config = config
        self._client = openai.OpenAI(
            max_retries=config.max_retries,
            timeout=config.timeout_sec,
        )

    def synthesize(self, text: str) -> TTSStream:
        """Stream PCM audio from OpenAI TTS API.

        Args:
            text: Text to synthesize (max 4096 characters).

        Returns:
            TTSStream yielding PCM audio chunks.

        Raises:
            TTSError: On validation failure or API error.
        """
        _validate_input(text, self._config.speed)

        try:
            kwargs: dict[str, Any] = {
                "model": self._config.model,
                "voice": self._config.voice,
                "input": text,
                "response_format": "pcm",
                "speed": self._config.speed,
            }
            if self._config.instructions:
                if self._config.model in _SUPPORTS_INSTRUCTIONS:
                    kwargs["instructions"] = self._config.instructions
                else:
                    logger.warning("instructions ignored for model %s", self._config.model)

            response_cm = self._client.audio.speech.with_streaming_response.create(**kwargs)
        except openai.OpenAIError as exc:
            logger.warning("OpenAI TTS API error: %s", exc)
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

        gen = _iter_chunks(response, safe_exit)
        return TTSStream(gen, close_fn=lambda: _safe_close(safe_exit))



def _iter_chunks(response: Any, safe_exit: Callable[..., None]) -> Generator[bytes, None, None]:
    """Yield audio chunks from an already-entered streaming response.

    Calls *safe_exit* exactly once on completion, error, or generator close.
    *safe_exit* is idempotent, so duplicate calls (e.g. from TTSStream.close_fn)
    are harmless.
    """
    try:
        for chunk in response.iter_bytes(chunk_size=4096):  # noqa: UP028
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


def _validate_input(text: str, speed: float) -> None:
    """Pre-validate input before API call."""
    if not text or not text.strip():
        raise TTSError("Text must not be empty or whitespace-only")
    if len(text) > 4096:
        raise TTSError(f"Text exceeds 4096 character limit ({len(text)} chars)")
    if not (0.25 <= speed <= 4.0):
        raise TTSError(f"Speed must be 0.25–4.0, got {speed}")
