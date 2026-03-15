"""Pre-generate greeting/farewell audio files using TTS.

Ensures that greeting and farewell audio match the configured TTS voice.
Files are named with a config+text hash to auto-invalidate when any
voice-affecting setting or text changes.
"""

from __future__ import annotations

import hashlib
import logging
import wave
from dataclasses import dataclass
from pathlib import Path

from voice_pipeline.core.config import GreetingAudioConfig, TTSConfig
from voice_pipeline.core.interfaces import ITTS

logger = logging.getLogger("voice_pipeline.tts")


@dataclass(frozen=True)
class GreetingAudioPaths:
    """Resolved paths for greeting and farewell audio files."""

    greeting: str
    farewell: str


def synthesize_to_wav(
    tts: ITTS,
    text: str,
    path: Path,
    sample_rate: int,
) -> None:
    """Synthesize *text* via TTS and write the result as a WAV file.

    Args:
        tts: TTS instance to synthesize audio.
        text: Text to synthesize.
        path: Output WAV file path.
        sample_rate: Sample rate of the PCM audio from TTS.
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
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)

    logger.info("Saved %s (%d bytes PCM)", path, len(pcm_data))


def _cache_key(tts_config: TTSConfig, text: str) -> str:
    """Short hash of voice-affecting settings + text."""
    source = (
        f"{tts_config.voice}|{tts_config.model}"
        f"|{tts_config.speed}|{tts_config.instructions}|{text}"
    )
    return hashlib.sha256(source.encode()).hexdigest()[:8]


def ensure_greeting_audio(
    tts: ITTS,
    tts_config: TTSConfig,
    greeting_config: GreetingAudioConfig,
) -> GreetingAudioPaths:
    """Ensure greeting/farewell WAV files exist, generating with TTS if needed.

    Derives filenames from a hash of voice-affecting TTS settings and text,
    so that any config or text change automatically triggers regeneration.
    Falls back to pre-recorded files on TTS failure.

    Args:
        tts: TTS instance to synthesize audio.
        tts_config: TTS configuration (voice, model, sample rate).
        greeting_config: Greeting audio configuration (texts, audio dir, fallbacks).

    Returns:
        GreetingAudioPaths with resolved file paths.
    """
    base_dir = Path(greeting_config.audio_dir)
    paths: dict[str, str] = {}

    items = (
        ("greeting", greeting_config.greeting_text, greeting_config.fallback_greeting_path),
        ("farewell", greeting_config.farewell_text, greeting_config.fallback_farewell_path),
    )

    for label, text, fallback in items:
        key = _cache_key(tts_config, text)
        filename = f"{label}_{key}.wav"
        path = base_dir / filename

        if path.exists():
            logger.debug("Audio cache hit: %s", path)
            paths[label] = str(path)
        else:
            try:
                logger.info("Generating %s audio: %s", label, path)
                synthesize_to_wav(tts, text, path, tts_config.output_sample_rate)
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
