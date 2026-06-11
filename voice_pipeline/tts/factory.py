"""TTS vendor factory."""

from __future__ import annotations

from typing import Literal

from voice_pipeline.core.interfaces import ITTS
from voice_pipeline.tts.elevenlabs_tts import ElevenLabsTTS
from voice_pipeline.tts.tts import OpenAITTS

_DEFAULT_VENDOR: Literal["openai", "elevenlabs"] = "elevenlabs"  # 기본 TTS vendor


def create_tts(vendor: Literal["openai", "elevenlabs"] = _DEFAULT_VENDOR) -> ITTS:
    """Factory: create an ITTS instance for *vendor*.

    Args:
        vendor: ``"openai"``이면 OpenAITTS, ``"elevenlabs"``이면 ElevenLabsTTS.

    Returns:
        Configured ITTS implementation.

    Raises:
        ValueError: On unknown vendor name.
    """
    if vendor == "openai":
        return OpenAITTS()
    elif vendor == "elevenlabs":
        return ElevenLabsTTS()
    else:
        raise ValueError(f"Unknown TTS vendor: {vendor!r}")
