"""Wakeword detection module."""

from voice_pipeline.wakeword.exceptions import WakewordError
from voice_pipeline.wakeword.wakeword import WakewordDetector

__all__ = ["WakewordDetector", "WakewordError"]
