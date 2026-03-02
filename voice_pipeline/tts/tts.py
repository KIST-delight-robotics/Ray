"""OpenAI TTS streaming implementation."""

from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path
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

        gen = _iter_response(response_cm)
        return TTSStream(gen, close_fn=lambda: _close_cm(response_cm))

    def save_to_file(self, text: str, path: str | Path) -> None:
        """Non-streaming: synthesize and save as WAV file.

        Convenience method for testing and utility use. Not part of the
        ITTS interface.

        Args:
            text: Text to synthesize.
            path: Output file path.

        Raises:
            TTSError: On API error.
        """
        _validate_input(text, self._config.speed)

        try:
            kwargs: dict[str, Any] = {
                "model": self._config.model,
                "voice": self._config.voice,
                "input": text,
                "response_format": "wav",
                "speed": self._config.speed,
            }
            if self._config.instructions:
                if self._config.model in _SUPPORTS_INSTRUCTIONS:
                    kwargs["instructions"] = self._config.instructions
                else:
                    logger.warning("instructions ignored for model %s", self._config.model)

            response = self._client.audio.speech.create(**kwargs)
            response.write_to_file(path)
        except openai.OpenAIError as exc:
            logger.warning("OpenAI TTS API error: %s", exc)
            raise TTSError(str(exc)) from exc


def _iter_response(response_cm: Any) -> Generator[bytes, None, None]:
    """Enter the streaming response CM, yield audio chunks, exit exactly once."""
    response = response_cm.__enter__()
    try:
        for chunk in response.iter_bytes(chunk_size=4096):  # noqa: UP028
            yield chunk
    except GeneratorExit:
        response_cm.__exit__(None, None, None)
        return
    except openai.OpenAIError as exc:
        response_cm.__exit__(type(exc), exc, exc.__traceback__)
        raise TTSError(str(exc)) from exc
    except Exception as exc:
        response_cm.__exit__(type(exc), exc, exc.__traceback__)
        raise TTSError(str(exc)) from exc
    else:
        response_cm.__exit__(None, None, None)


def _close_cm(cm: Any) -> None:
    """Best-effort close of context manager (for TTSStream.close_fn)."""
    try:
        cm.__exit__(None, None, None)
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
