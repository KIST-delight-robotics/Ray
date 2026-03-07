"""Dataclass-based configuration for the voice pipeline.

Config sections are added as their modules are implemented.
Current: Phase 1–3 + Phase 4 (vap, turngpt, turn_detector, speech_generator).
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
    sample_width: int = 2

    @property
    def frame_size_samples(self) -> int:
        """Number of samples per frame."""
        return self.sample_rate * self.frame_duration_ms // 1000


@dataclass
class ConversationHistoryConfig:
    """Configuration for ConversationHistory and StorageBackend."""

    max_context_tokens: int = 4096
    storage_backend: str = "memory"
    storage_path: str = ""


@dataclass
class ASRConfig:
    """Configuration for the ASR module."""

    language_code: str = "en-US"
    model: str = "latest_long"
    interim_results: bool = True


@dataclass
class LLMConfig:
    """Configuration for the LLM module."""

    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 256
    max_retries: int = 2
    timeout_sec: float = 30.0


@dataclass
class TTSConfig:
    """Configuration for the TTS module."""

    vendor: str = "openai"
    voice: str = "alloy"
    model: str = "tts-1"
    output_sample_rate: int = 24000
    speed: float = 1.0
    timeout_sec: float = 30.0
    max_retries: int = 2
    instructions: str = ""


@dataclass
class CppBridgeConfig:
    """Configuration for the C++ bridge WebSocket connection."""

    host: str = "localhost"
    port: int = 8765
    reconnect_attempts: int = 3
    recv_timeout_sec: float = 1.0
    connect_timeout_sec: float = 5.0
    close_timeout_sec: float = 5.0


@dataclass
class WakewordConfig:
    """Configuration for wakeword detection."""

    keywords: tuple[str, ...] = ("ray",)
    vad_threshold: float = 0.5
    language_code: str = "en-US"
    speech_pad_ms: int = 300
    min_speech_duration_ms: int = 100
    max_speech_duration_sec: float = 3.0
    stt_timeout_sec: float = 5.0


@dataclass
class LEDConfig:
    """Configuration for the LED controller.

    Attributes:
        bar_count: Number of LEDs in the bar segment (indices 0..bar_count-1).
        ring_count: Number of LEDs in the ring segment (indices bar_count..bar_count+ring_count-1).
        spi_pin: SPI GPIO pin number (Pi 5 default: GPIO 10 = SPI0 MOSI).
        brightness: Global brightness 0-255.
    """

    bar_count: int = 8
    ring_count: int = 16
    spi_pin: int = 10
    brightness: int = 128


@dataclass
class VAPConfig:
    """Configuration for the VAP (Voice Activity Projection) wrapper.

    Attributes:
        model_path: Path to the VAP model state_dict file.
        context_sec: Rolling buffer duration in seconds.
        step_sec: Inference interval in seconds.
        tt_time: Turn-taking lookahead window in seconds for averaging.
        device: Torch device string.
        vad_threshold: Threshold for user_is_speaking derivation.
    """

    model_path: str = ""
    context_sec: float = 20.0
    step_sec: float = 0.1
    tt_time: float = 0.5
    device: str = "cpu"
    vad_threshold: float = 0.5


@dataclass
class TurnGPTConfig:
    """Configuration for the TurnGPT wrapper.

    Attributes:
        checkpoint_path: Path to the TurnGPT checkpoint file.
        device: Torch device string.
        max_context_tokens: Maximum token count before old turns are evicted.
            GPT-2 position limit is 1024. 0 = no limit.
    """

    checkpoint_path: str = ""
    device: str = "cpu"
    max_context_tokens: int = 1024


@dataclass
class TurnDetectorConfig:
    """Configuration for the combined TurnDetector.

    Fuses VAP (audio) and TurnGPT (text) signals with timing heuristics.

    Attributes:
        vap_user_threshold: p_now/p_fut below this means "favors robot".
        min_gap_time_sec: Sustained VAP robot-favor duration for turn-shift.
        turngpt_thresholds: Graduated (prob, timeout_sec) pairs for Path 2.
            Evaluated top-down; first matching prob triggers timeout.
        interrupt_user_threshold: p_now/p_fut above this means "favors user".
        prepare_turngpt_threshold: TurnGPT prob above this triggers prepare.
        prepare_timeout_sec: Time since last ASR change to trigger prepare.
        prepare_similarity_threshold: Skip prepare if text similarity >= this.
    """

    # --- VAP turn-shift thresholds (Path 1) ---
    vap_user_threshold: float = 0.5
    min_gap_time_sec: float = 0.5

    # --- TurnGPT graduated timeout (Path 2) ---
    turngpt_thresholds: tuple[tuple[float, float], ...] = (
        (0.3, 0.5),
        (0.2, 1.0),
        (0.1, 2.0),
        (0.0, 3.0),
    )

    # --- Interrupt detection (ROBOT_TURN) ---
    interrupt_user_threshold: float = 0.5

    # --- Prepare (speculative generation) ---
    prepare_turngpt_threshold: float = 0.2
    prepare_timeout_sec: float = 0.2
    prepare_similarity_threshold: float = 0.8


@dataclass
class SpeechGeneratorConfig:
    """Configuration for the SpeechGenerator.

    Attributes:
        max_workers: Thread pool size for background generation.
    """

    max_workers: int = 1


@dataclass
class PipelineConfig:
    """Top-level configuration for the voice pipeline.

    Passed to SessionManager and distributed to modules during construction.
    New fields are added as modules are implemented in subsequent phases.
    """

    audio: AudioConfig = field(default_factory=AudioConfig)
    history: ConversationHistoryConfig = field(default_factory=ConversationHistoryConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    cpp_bridge: CppBridgeConfig = field(default_factory=CppBridgeConfig)
    wakeword: WakewordConfig = field(default_factory=WakewordConfig)
    led: LEDConfig = field(default_factory=LEDConfig)
    vap: VAPConfig = field(default_factory=VAPConfig)
    turngpt: TurnGPTConfig = field(default_factory=TurnGPTConfig)
    turn_detector: TurnDetectorConfig = field(default_factory=TurnDetectorConfig)
    speech_generator: SpeechGeneratorConfig = field(default_factory=SpeechGeneratorConfig)
