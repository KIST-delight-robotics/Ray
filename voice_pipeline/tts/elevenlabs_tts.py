"""ElevenLabs TTS streaming implementation with word timestamps."""

from __future__ import annotations

import base64
import json
import logging
import os
from collections.abc import Generator
from typing import Any

import httpx
from elevenlabs import ElevenLabs
from elevenlabs.core.api_error import ApiError
from elevenlabs.types import VoiceSettings

from voice_pipeline.core.interfaces import ITTS
from voice_pipeline.core.types import TTSStream, WordTimestamp
from voice_pipeline.tts.exceptions import TTSError

logger = logging.getLogger("voice_pipeline.tts")


class ElevenLabsTTS(ITTS):
    """TTS implementation using the ElevenLabs API with streaming + timestamps.

    Reads ``ELEVENLABS_API_KEY`` from the environment; raises :class:`TTSError`
    at construction if missing. Streams raw PCM audio (24 kHz, 16-bit signed
    LE, mono) via the ``stream_with_timestamps`` endpoint; word-level
    timestamps are aggregated from character alignment and exposed on the
    stream after full consumption.

    The returned :class:`TTSStream` must be fully consumed or explicitly
    closed to release the underlying HTTP connection.
    """

    OUTPUT_SAMPLE_RATE: int = 24000  # pcm_24000 — OpenAITTS와 동일 (downstream 무변경)

    _VOICE_ID: str = "EXAVITQu4vr4xnSDxMaL"  # Sarah — 임시 영어 default voice (추후 교체 예정)
    _MODEL: str = "eleven_flash_v2_5"  # 최저 지연 모델 (예: "eleven_turbo_v2_5", "eleven_multilingual_v2")
    _OUTPUT_FORMAT: str = "pcm_24000"  # 24kHz 16-bit signed LE mono (tier 제한 없음)
    _VOICE_SETTINGS: dict[str, float] | None = None  # 예: {"stability": 0.5}; None이면 voice 기본값

    _MAX_RETRIES = 2  # 합성 실패 시 자동 재시도 횟수 (request_options.max_retries)
    _TIMEOUT_SEC = 10.0  # httpx timeout — SDK 기본값 240s는 실시간 대화에 부적합
    _MAX_TEXT_LEN = 4096  # 보수적 입력 길이 상한 (OpenAITTS와 동일 동작)

    def __init__(self) -> None:
        # SDK는 키가 없어도 클라이언트를 생성하고 첫 요청에야 401을 내므로
        # (OpenAI SDK와 달리) 생성 시점에 직접 검증해 fail-fast.
        if not os.environ.get("ELEVENLABS_API_KEY"):
            raise TTSError("ELEVENLABS_API_KEY environment variable not set")
        self._client = ElevenLabs(timeout=self._TIMEOUT_SEC)

    @property
    def output_sample_rate(self) -> int:
        return ElevenLabsTTS.OUTPUT_SAMPLE_RATE

    @property
    def voice_id(self) -> str:
        settings = json.dumps(self._VOICE_SETTINGS, sort_keys=True) if self._VOICE_SETTINGS else ""
        return f"elevenlabs|{self._VOICE_ID}|{self._MODEL}|{settings}"

    @property
    def model_name(self) -> str:
        return self._MODEL

    def synthesize(self, text: str) -> TTSStream:
        """Stream PCM audio from the ElevenLabs TTS API.

        Args:
            text: Text to synthesize (max 4096 characters).

        Returns:
            TTSStream yielding PCM audio chunks. After full consumption,
            ``.timestamps`` holds word-level timestamps.

        Raises:
            TTSError: On validation failure or API error. The SDK issues the
                HTTP request lazily, so API errors surface during iteration
                rather than from this call.
        """
        self._validate_input(text)

        kwargs: dict[str, Any] = {
            "text": text,
            "model_id": self._MODEL,
            "output_format": self._OUTPUT_FORMAT,
            "request_options": {"max_retries": self._MAX_RETRIES},
        }
        if self._VOICE_SETTINGS:
            kwargs["voice_settings"] = VoiceSettings(**self._VOICE_SETTINGS)

        # Generator function — HTTP 요청과 에러는 첫 next()에서 발생.
        sdk_stream = self._client.text_to_speech.stream_with_timestamps(self._VOICE_ID, **kwargs)

        chars: list[str] = []
        starts: list[float] = []
        ends: list[float] = []
        gen = _iter_chunks(sdk_stream, chars, starts, ends)
        return TTSStream(
            gen,
            # gen이 시작 전이면 gen.close()가 finally에 도달하지 않으므로 SDK generator를 직접 닫는다.
            close_fn=sdk_stream.close,
            timestamps_fn=lambda: _alignment_to_word_timestamps(chars, starts, ends),
        )

    def _validate_input(self, text: str) -> None:
        """Pre-validate input before API call."""
        if not text or not text.strip():
            raise TTSError("Text must not be empty or whitespace-only")
        if len(text) > self._MAX_TEXT_LEN:
            raise TTSError(f"Text exceeds {self._MAX_TEXT_LEN} character limit ({len(text)} chars)")


def _iter_chunks(
    sdk_stream: Generator[Any, None, None],
    chars: list[str],
    starts: list[float],
    ends: list[float],
) -> Generator[bytes, None, None]:
    """Yield decoded PCM chunks, accumulating character alignment.

    Alignment is appended to the caller-owned lists so word timestamps can be
    aggregated after the stream ends (words may span chunk boundaries). The
    SDK generator is closed on every exit path, releasing the HTTP connection.
    ``GeneratorExit`` is not caught (BaseException) and propagates naturally.
    """
    try:
        for chunk in sdk_stream:
            alignment = chunk.alignment
            if alignment is not None:
                chars.extend(alignment.characters)
                starts.extend(alignment.character_start_times_seconds)
                ends.extend(alignment.character_end_times_seconds)
            audio = base64.b64decode(chunk.audio_base_64)
            if audio:  # 빈 오디오 chunk(alignment 전용)는 yield하지 않음
                yield audio
    except ApiError as exc:
        body = str(exc.body)[:200]
        logger.error("ElevenLabs TTS API error (status=%s): %s", exc.status_code, body)
        raise TTSError(f"ElevenLabs API error (status={exc.status_code}): {body}") from exc
    except httpx.TimeoutException as exc:
        logger.error("ElevenLabs TTS timeout: %s", exc)
        raise TTSError(f"TTS timeout: {exc}") from exc
    except Exception as exc:
        logger.error("ElevenLabs TTS error: %s", exc)
        raise TTSError(str(exc)) from exc
    finally:
        sdk_stream.close()


def _alignment_to_word_timestamps(
    chars: list[str],
    starts: list[float],
    ends: list[float],
) -> tuple[WordTimestamp, ...]:
    """Aggregate character alignment into whitespace-delimited word timestamps.

    Tokenization matches ``text.split()`` so the result feeds
    ``truncate_by_timestamps`` directly. Best-effort: anomalies (length
    mismatch, negative/inverted times) are truncated or clamped, never
    raised — a timestamps failure must not fail the turn.
    """
    if not (len(chars) == len(starts) == len(ends)):
        logger.warning(
            "Alignment length mismatch: %d chars / %d starts / %d ends",
            len(chars),
            len(starts),
            len(ends),
        )
    n = min(len(chars), len(starts), len(ends))

    words: list[WordTimestamp] = []
    buf: list[str] = []
    word_start = word_end = 0.0
    for i in range(n):
        ch = chars[i]
        if ch.isspace():
            if buf:
                words.append(_make_word(buf, word_start, word_end))
                buf = []
        else:
            if not buf:
                word_start = starts[i]
            buf.append(ch)
            word_end = ends[i]
    if buf:
        words.append(_make_word(buf, word_start, word_end))
    return tuple(words)


def _make_word(buf: list[str], start: float, end: float) -> WordTimestamp:
    """Build a WordTimestamp, clamping times to satisfy validation."""
    start = max(0.0, start)
    end = max(start, end)
    return WordTimestamp(word="".join(buf), start_sec=start, end_sec=end)
