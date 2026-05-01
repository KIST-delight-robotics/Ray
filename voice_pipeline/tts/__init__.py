"""TTS module."""

from voice_pipeline.tts.exceptions import TTSError
from voice_pipeline.tts.tts import OpenAITTS
from voice_pipeline.tts.utterance_truncator import truncate_by_ratio, truncate_by_timestamps

__all__ = ["OpenAITTS", "TTSError", "truncate_by_ratio", "truncate_by_timestamps"]
