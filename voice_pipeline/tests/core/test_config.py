"""Tests for voice_pipeline.core.config."""

from voice_pipeline.core.config import (
    ASRConfig,
    AudioConfig,
    ConversationHistoryConfig,
    CppBridgeConfig,
    LEDConfig,
    LLMConfig,
    PipelineConfig,
    TTSConfig,
    WakewordConfig,
)


class TestAudioConfig:
    def test_defaults(self) -> None:
        cfg = AudioConfig()
        assert cfg.sample_rate == 16000
        assert cfg.channels == 1
        assert cfg.frame_duration_ms == 30
        assert cfg.sample_width == 2

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


class TestASRConfig:
    def test_defaults(self) -> None:
        cfg = ASRConfig()
        assert cfg.language_code == "en-US"
        assert cfg.model == "latest_long"
        assert cfg.interim_results is True


class TestLLMConfig:
    def test_defaults(self) -> None:
        cfg = LLMConfig()
        assert cfg.model == "gpt-4o"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 256
        assert cfg.max_retries == 2
        assert cfg.timeout_sec == 30.0


class TestTTSConfig:
    def test_defaults(self) -> None:
        cfg = TTSConfig()
        assert cfg.vendor == "openai"
        assert cfg.voice == "alloy"
        assert cfg.model == "tts-1"
        assert cfg.output_sample_rate == 24000


class TestCppBridgeConfig:
    def test_defaults(self) -> None:
        cfg = CppBridgeConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 8765


class TestWakewordConfig:
    def test_defaults(self) -> None:
        cfg = WakewordConfig()
        assert cfg.keywords == ("ray",)
        assert cfg.vad_threshold == 0.5


class TestLEDConfig:
    def test_defaults(self) -> None:
        cfg = LEDConfig()
        assert cfg.led_count == 12
        assert cfg.spi_device == "/dev/spidev0.0"
        assert cfg.brightness == 0.5
        assert cfg.noop is False


class TestPipelineConfig:
    def test_default_construction(self) -> None:
        cfg = PipelineConfig()
        assert isinstance(cfg.audio, AudioConfig)
        assert isinstance(cfg.history, ConversationHistoryConfig)
        assert isinstance(cfg.asr, ASRConfig)
        assert isinstance(cfg.llm, LLMConfig)
        assert isinstance(cfg.tts, TTSConfig)
        assert isinstance(cfg.cpp_bridge, CppBridgeConfig)
        assert isinstance(cfg.wakeword, WakewordConfig)
        assert isinstance(cfg.led, LEDConfig)

    def test_sub_configs_independent(self) -> None:
        cfg1 = PipelineConfig()
        cfg2 = PipelineConfig()
        cfg1.audio.sample_rate = 8000
        assert cfg2.audio.sample_rate == 16000
