"""TTS module."""

from voice_pipeline.tts.exceptions import TTSError
from voice_pipeline.tts.tts import OpenAITTS
from voice_pipeline.tts.utterance_truncator import (
    DurationRatioTruncator,
    TimestampTruncator,
)

__all__ = ["DurationRatioTruncator", "OpenAITTS", "TTSError", "TimestampTruncator"]
