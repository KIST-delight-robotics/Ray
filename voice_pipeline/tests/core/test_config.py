"""Tests for voice_pipeline.core.config."""

from voice_pipeline.core.config import AudioConfig, ConversationHistoryConfig, PipelineConfig


class TestAudioConfig:
    def test_defaults(self) -> None:
        cfg = AudioConfig()
        assert cfg.sample_rate == 16000
        assert cfg.channels == 1
        assert cfg.frame_duration_ms == 30

    def test_frame_size_samples(self) -> None:
        cfg = AudioConfig(sample_rate=16000, frame_duration_ms=30)
        assert cfg.frame_size_samples == 480

    def test_frame_size_samples_custom(self) -> None:
        cfg = AudioConfig(sample_rate=8000, frame_duration_ms=20)
        assert cfg.frame_size_samples == 160


class TestConversationHistoryConfig:
    def test_defaults(self) -> None:
        cfg = ConversationHistoryConfig()
        assert cfg.max_context_tokens == 4096
        assert cfg.storage_backend == "memory"
        assert cfg.storage_path == ""


class TestPipelineConfig:
    def test_default_construction(self) -> None:
        cfg = PipelineConfig()
        assert isinstance(cfg.audio, AudioConfig)
        assert isinstance(cfg.history, ConversationHistoryConfig)

    def test_sub_configs_independent(self) -> None:
        cfg1 = PipelineConfig()
        cfg2 = PipelineConfig()
        cfg1.audio.sample_rate = 8000
        assert cfg2.audio.sample_rate == 16000
