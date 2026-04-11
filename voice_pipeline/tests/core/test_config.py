"""Tests for voice_pipeline.core.config."""

from voice_pipeline.core.config import (
    AudioConfig,
    PipelineConfig,
)


class TestAudioConfig:
    def test_frame_size_samples(self) -> None:
        cfg = AudioConfig(sample_rate=16000, frame_duration_ms=30)
        assert cfg.frame_size_samples == 480

    def test_frame_size_samples_custom(self) -> None:
        cfg = AudioConfig(sample_rate=8000, frame_duration_ms=20)
        assert cfg.frame_size_samples == 160


class TestPipelineConfig:
    def test_sub_configs_independent(self) -> None:
        cfg1 = PipelineConfig()
        cfg2 = PipelineConfig()
        cfg1.audio.sample_rate = 8000
        assert cfg2.audio.sample_rate == 16000
