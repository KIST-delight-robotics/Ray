"""Tests for voice_pipeline.core.config."""

from voice_pipeline.core.config import (
    ASRConfig,
    AudioConfig,
    ConversationHistoryConfig,
    CppBridgeConfig,
    LEDConfig,
    LLMConfig,
    PipelineConfig,
    SpeechGeneratorConfig,
    TTSConfig,
    TurnDetectorConfig,
    TurnGPTConfig,
    VAPConfig,
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
        assert cfg.storage_backend == "file"
        assert cfg.storage_path == "logs/sessions"


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
        assert cfg.voice == "ash"
        assert cfg.model == "tts-1"
        assert cfg.output_sample_rate == 24000
        assert cfg.speed == 1.0
        assert cfg.timeout_sec == 30.0
        assert cfg.max_retries == 2
        assert cfg.instructions is None


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
        assert cfg.bar_count == 8
        assert cfg.ring_count == 16
        assert cfg.spi_pin == 10
        assert cfg.brightness == 128


class TestVAPConfig:
    def test_defaults(self) -> None:
        cfg = VAPConfig()
        assert cfg.model_path
        assert cfg.context_sec == 20.0
        assert cfg.step_sec == 0.1
        assert cfg.tt_time == 0.5
        assert cfg.device == "cpu"
        assert cfg.vad_threshold == 0.5


class TestTurnGPTConfig:
    def test_defaults(self) -> None:
        cfg = TurnGPTConfig()
        assert cfg.checkpoint_path is None
        assert cfg.onnx_model_path
        assert cfg.tokenizer_path
        assert cfg.device == "cpu"
        assert cfg.max_context_tokens == 1024


class TestTurnDetectorConfig:
    def test_defaults(self) -> None:
        cfg = TurnDetectorConfig()
        assert cfg is not None


class TestSpeechGeneratorConfig:
    def test_defaults(self) -> None:
        cfg = SpeechGeneratorConfig()
        assert cfg.max_workers == 2


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
        assert isinstance(cfg.vap, VAPConfig)
        assert isinstance(cfg.turngpt, TurnGPTConfig)
        assert isinstance(cfg.turn_detector, TurnDetectorConfig)
        assert isinstance(cfg.speech_generator, SpeechGeneratorConfig)

    def test_sub_configs_independent(self) -> None:
        cfg1 = PipelineConfig()
        cfg2 = PipelineConfig()
        cfg1.audio.sample_rate = 8000
        assert cfg2.audio.sample_rate == 16000
