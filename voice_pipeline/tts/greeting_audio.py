"""Pre-generate greeting/farewell audio files using TTS.

Ensures that greeting and farewell audio match the configured TTS voice.
Files are named with a voice_id+text hash to auto-invalidate when any
voice-affecting setting or text changes.
"""

from __future__ import annotations

import hashlib
import logging
import wave
from dataclasses import dataclass
from pathlib import Path

from voice_pipeline.core.interfaces import ITTS

logger = logging.getLogger("voice_pipeline.tts")

_AUDIO_DIR = "assets/audio"  # 생성 오디오 파일 저장 디렉토리 (C++ 작업 경로 기준)
_GREETING_TEXT = "Yes, how can I help you?"  # greeting 합성 텍스트
_FAREWELL_TEXT = "Talk to you next time!"  # farewell 합성 텍스트
_FALLBACK_GREETING_PATH = "assets/audio/greeting.wav"  # TTS 실패 시 사용할 greeting 파일
_FALLBACK_FAREWELL_PATH = "assets/audio/farewell.wav"  # TTS 실패 시 사용할 farewell 파일


@dataclass(frozen=True)
class GreetingAudioPaths:
    """Resolved paths for greeting and farewell audio files."""

    greeting: str
    farewell: str


def synthesize_to_wav(
    tts: ITTS,
    text: str,
    path: Path,
) -> None:
    """Synthesize *text* via TTS and write the result as a WAV file.

    Args:
        tts: TTS instance to synthesize audio (provides output_sample_rate).
        text: Text to synthesize.
        path: Output WAV file path.
    """
    stream = tts.synthesize(text)
    try:
        pcm_data = b"".join(stream)
    finally:
        stream.close()

    path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(tts.output_sample_rate)
        wf.writeframes(pcm_data)

    logger.info("Saved %s (%d bytes PCM)", path, len(pcm_data))


def _cache_key(tts: ITTS, text: str) -> str:
    """Short hash of voice_id + text. 같은 음성·텍스트면 같은 키."""
    source = f"{tts.voice_id}|{text}"
    return hashlib.sha256(source.encode()).hexdigest()[:8]


def ensure_greeting_audio(tts: ITTS) -> GreetingAudioPaths:
    """Ensure greeting/farewell WAV files exist, generating with TTS if needed.

    Derives filenames from a hash of the TTS voice_id and text, so that any
    voice or text change automatically triggers regeneration.
    Falls back to module-level fallback paths on TTS failure.

    Args:
        tts: TTS instance to synthesize audio.

    Returns:
        GreetingAudioPaths with resolved file paths.
    """
    base_dir = Path(_AUDIO_DIR)
    paths: dict[str, str] = {}

    items = (
        ("greeting", _GREETING_TEXT, _FALLBACK_GREETING_PATH),
        ("farewell", _FAREWELL_TEXT, _FALLBACK_FAREWELL_PATH),
    )

    for label, text, fallback in items:
        key = _cache_key(tts, text)
        filename = f"{label}_{key}.wav"
        path = base_dir / filename

        if path.exists():
            logger.debug("Audio cache hit: %s", path)
            paths[label] = str(path)
        else:
            try:
                logger.info("Generating %s audio: %s", label, path)
                synthesize_to_wav(tts, text, path)
                paths[label] = str(path)
            except Exception:
                logger.warning(
                    "Failed to generate %s audio, using fallback: %s",
                    label,
                    fallback,
                    exc_info=True,
                )
                paths[label] = fallback

    return GreetingAudioPaths(greeting=paths["greeting"], farewell=paths["farewell"])
