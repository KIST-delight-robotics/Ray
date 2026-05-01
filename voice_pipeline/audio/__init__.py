"""Audio capture module."""

from voice_pipeline.audio.audio_input import AudioInput
from voice_pipeline.audio.exceptions import AudioInputError

__all__ = ["AudioInput", "AudioInputError"]
