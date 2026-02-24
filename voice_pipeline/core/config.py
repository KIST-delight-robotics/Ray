"""Dataclass-based configuration for the voice pipeline.

Only configs relevant to current and next implementation phases are defined.
New config sections are added as their modules are implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    """Audio capture settings shared across the pipeline.

    All audio-consuming modules (ASR, VAP, WakewordDetector) must agree
    on these parameters.
    """

    sample_rate: int = 16000
    channels: int = 1
    frame_duration_ms: int = 30

    @property
    def frame_size_samples(self) -> int:
        """Number of samples per frame."""
        return self.sample_rate * self.frame_duration_ms // 1000


@dataclass
class ConversationHistoryConfig:
    """Configuration for ConversationHistory and StorageBackend."""

    max_turns_in_context: int = 20
    storage_backend: str = "memory"
    storage_path: str = ""


@dataclass
class PipelineConfig:
    """Top-level configuration for the voice pipeline.

    Passed to SessionManager and distributed to modules during construction.
    New fields are added as modules are implemented in subsequent phases.
    """

    audio: AudioConfig = field(default_factory=AudioConfig)
    history: ConversationHistoryConfig = field(default_factory=ConversationHistoryConfig)
