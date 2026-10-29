"""Pre-generate the greeting audio file using TTS.

Ensures that the greeting audio matches the configured TTS voice.
Files are named with a voice_id+text hash to auto-invalidate when any
voice-affecting setting or text changes. The returned path is a level-normalized
copy (see :mod:`~voice_pipeline.audio_level`) so that TTS 벤더가 바뀌어도
재생 크기는 GPT-Live 말소리에 맞는다.
"""

from __future__ import annotations

import hashlib
import logging
import wave
from pathlib import Path

from voice_pipeline.audio_level import ensure_normalized
from voice_pipeline.settings import VAR_DIR
from voice_pipeline.types import ITTS

logger = logging.getLogger("voice_pipeline.tts")

# TTS 로 생성한 인사 WAV 캐시 (C++ 작업 경로 기준). assets/audio 에는 추적되는 폴백만 둔다.
_AUDIO_DIR = f"{VAR_DIR}/audio"
_FALLBACK_GREETING_PATH = "assets/audio/greeting.wav"  # TTS 실패 시 사용할 greeting 파일


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


def ensure_greeting_audio(tts: ITTS, text: str) -> str:
    """Ensure the greeting WAV exists, generating it with TTS if needed.

    Derives the filename from a hash of the TTS voice_id and text, so that any
    voice or text change automatically triggers regeneration.
    Falls back to the tracked fallback file on TTS failure.

    Args:
        tts: TTS instance to synthesize audio.
        text: 인사 문구.

    Returns:
        Path of the level-normalized greeting WAV.
    """
    path = Path(_AUDIO_DIR) / f"greeting_{_cache_key(tts, text)}.wav"
    if path.exists():
        logger.debug("Audio cache hit: %s", path)
        return _leveled(str(path))
    try:
        logger.info("Generating greeting audio: %s", path)
        synthesize_to_wav(tts, text, path)
    except Exception:
        logger.warning("Failed to generate greeting audio, using fallback: %s", _FALLBACK_GREETING_PATH, exc_info=True)
        return _leveled(_FALLBACK_GREETING_PATH)
    return _leveled(str(path))


def _leveled(path: str) -> str:
    """출력 레벨을 맞춘 사본 경로. 실패하면 원본 그대로 — 소리가 큰 게 안 나오는 것보다 낫다."""
    try:
        return str(ensure_normalized(path, speech=True))
    except Exception:
        logger.warning("Level normalization failed, playing as is: %s", path, exc_info=True)
        return path
