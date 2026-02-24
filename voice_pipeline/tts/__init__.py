"""TTS module."""

from voice_pipeline.tts.utterance_truncator import (
    DurationRatioTruncator,
    TimestampTruncator,
)

__all__ = ["DurationRatioTruncator", "TimestampTruncator"]
