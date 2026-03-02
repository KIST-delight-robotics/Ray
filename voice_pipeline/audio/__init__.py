"""Audio capture and wakeword detection."""

from voice_pipeline.audio.exceptions import WakewordError
from voice_pipeline.audio.wakeword import WakewordDetector

__all__ = ["WakewordDetector", "WakewordError"]
