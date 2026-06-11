"""TTS module."""

from voice_pipeline.tts.elevenlabs_tts import ElevenLabsTTS
from voice_pipeline.tts.exceptions import TTSError
from voice_pipeline.tts.factory import create_tts
from voice_pipeline.tts.openai_tts import OpenAITTS
from voice_pipeline.tts.utterance_truncator import truncate_by_ratio, truncate_by_timestamps

__all__ = [
    "ElevenLabsTTS",
    "OpenAITTS",
    "TTSError",
    "create_tts",
    "truncate_by_ratio",
    "truncate_by_timestamps",
]
