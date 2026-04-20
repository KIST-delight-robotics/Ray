"""Dataclass-based configuration for the voice pipeline.

Config sections are added as their modules are implemented.
Current: Phase 1–3 + Phase 4 + Phase 5 + Phase 6 + Memory Phase 1–3 + Embedding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from voice_pipeline.core.exceptions import ConfigurationError


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

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ConfigurationError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.channels <= 0:
            raise ConfigurationError(f"channels must be positive, got {self.channels}")
        if self.frame_duration_ms <= 0:
            raise ConfigurationError(
                f"frame_duration_ms must be positive, got {self.frame_duration_ms}"
            )
        if self.sample_width <= 0:
            raise ConfigurationError(f"sample_width must be positive, got {self.sample_width}")

    @property
    def frame_size_samples(self) -> int:
        """Number of samples per frame."""
        return self.sample_rate * self.frame_duration_ms // 1000

    @property
    def frame_size_bytes(self) -> int:
        """Number of bytes per frame."""
        return self.frame_size_samples * self.sample_width * self.channels


@dataclass
class ConversationHistoryConfig:
    """Configuration for ConversationHistory and StorageBackend.

    Attributes:
        max_context_tokens: Total LLM context token budget.
        storage_backend: Persistence backend type.
        storage_path: Database file path.
        max_memory_tokens: Dedicated token budget for memory block (Block 4).
        max_profile_tokens: Dedicated token budget for profile block (Block 2).
        max_prev_session_tokens: Token budget for previous session summaries.
        previous_session_count: Number of recent sessions to load at session start.
    """

    max_context_tokens: int = 4096
    storage_backend: Literal["memory", "sqlite"] = "sqlite"
    storage_path: str = "data/ray.db"
    max_memory_tokens: int = 512
    max_profile_tokens: int = 256
    max_prev_session_tokens: int = 512
    previous_session_count: int = 3


@dataclass
class ASRConfig:
    """Configuration for the ASR module."""

    language_code: str = "en-US"
    model: str = "latest_long"
    interim_results: bool = True


@dataclass
class LLMConfig:
    """Configuration for the LLM module.

    Attributes:
        reasoning_effort: Reasoning effort level for reasoning models.
            None = omit the parameter (non-reasoning models like gpt-4o).
            Valid values depend on the model:
              gpt-5: "minimal", "low", "medium", "high"
              gpt-5.1: "none", "low", "medium", "high"
              gpt-5.4: "none", "low", "medium", "high", "xhigh"
        tools: Tool names to enable (resolved via llm.tools at startup).
            Available: "web_search".
    """

    model: str = "gpt-5.4"
    temperature: float = 0.7
    max_tokens: int = 256
    max_retries: int = 2
    timeout_sec: float = 30.0
    reasoning_effort: str | None = "none"
    tools: list[str] = field(default_factory=lambda: ["web_search"])


@dataclass
class TTSConfig:
    """Configuration for the TTS module."""

    vendor: str = "openai"
    voice: str = "ash"
    model: str = "tts-1"
    output_sample_rate: int = 24000
    speed: float = 1.0
    timeout_sec: float = 30.0
    max_retries: int = 2
    instructions: str | None = None


@dataclass
class CppBridgeConfig:
    """Configuration for the C++ bridge WebSocket connection."""

    host: str = "localhost"
    port: int = 9200
    reconnect_attempts: int = 3
    recv_timeout_sec: float = 1.0
    connect_timeout_sec: float = 5.0
    close_timeout_sec: float = 5.0

    def __post_init__(self) -> None:
        if not (1 <= self.port <= 65535):
            raise ConfigurationError(f"port must be in [1, 65535], got {self.port}")


@dataclass
class WakewordConfig:
    """Configuration for wakeword detection."""

    keywords: tuple[str, ...] = ("ray",)
    vad_threshold: float = 0.5
    language_code: str = "en-US"
    pre_buffer_ms: int = 300
    speech_pad_ms: int = 300
    min_speech_duration_ms: int = 100
    max_speech_duration_sec: float = 3.0
    stt_timeout_sec: float = 5.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.vad_threshold <= 1.0):
            raise ConfigurationError(f"vad_threshold must be in [0, 1], got {self.vad_threshold}")


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

    def __post_init__(self) -> None:
        if not (0 <= self.brightness <= 255):
            raise ConfigurationError(f"brightness must be in [0, 255], got {self.brightness}")


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

    model_path: str = (
        "external/VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt"
    )
    context_sec: float = 20.0
    step_sec: float = 0.1
    tt_time: float = 0.5
    device: str = "cpu"
    vad_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.context_sec <= 0:
            raise ConfigurationError(f"context_sec must be positive, got {self.context_sec}")
        if self.step_sec <= 0:
            raise ConfigurationError(f"step_sec must be positive, got {self.step_sec}")
        if not (0.0 <= self.vad_threshold <= 1.0):
            raise ConfigurationError(f"vad_threshold must be in [0, 1], got {self.vad_threshold}")


@dataclass
class MaAIVAPConfig:
    """Configuration for the MaAI VAP wrapper.

    Attributes:
        lang: Language code for MaAI model.
        frame_rate: VAP inference frame rate in Hz.
        context_len_sec: Encoder context length in seconds.
        vad_threshold: Threshold for user_is_speaking derivation.
        ort_threads: ONNX Runtime intra-op thread count.
        pt_threads: PyTorch intra-op thread count.
        encoder_onnx_path: Path to pre-exported encoder ONNX file.
        transformer_onnx_path: Path to pre-exported transformer ONNX file.
            If empty, falls back to PyTorch transformer via MaAI.
        use_torch_compile: Enable torch.compile for transformer (PyTorch mode only).
            Only applies when transformer_onnx_path is empty.
    """

    lang: str = "en"
    frame_rate: int = 10
    context_len_sec: float = 5.0
    vad_threshold: float = 0.5
    ort_threads: int = 1
    pt_threads: int = 1
    encoder_onnx_path: str = "models/maai/encoder_10hz_5s.onnx"
    transformer_onnx_path: str = "models/maai/transformer_en_5s.onnx"
    use_torch_compile: bool = True

    def __post_init__(self) -> None:
        if self.frame_rate <= 0:
            raise ConfigurationError(f"frame_rate must be positive, got {self.frame_rate}")
        if self.context_len_sec <= 0:
            raise ConfigurationError(
                f"context_len_sec must be positive, got {self.context_len_sec}"
            )
        if not (0.0 <= self.vad_threshold <= 1.0):
            raise ConfigurationError(f"vad_threshold must be in [0, 1], got {self.vad_threshold}")


@dataclass
class TurnGPTConfig:
    """Configuration for the TurnGPT wrapper.

    Attributes:
        checkpoint_path: Path to the TurnGPT checkpoint file (PyTorch mode).
            None when not using PyTorch mode.
        onnx_model_path: Path to ONNX model file. When set, uses ONNX Runtime
            instead of PyTorch for inference.
        tokenizer_path: Path to saved tokenizer directory (required for ONNX mode).
        device: Torch device string (PyTorch mode only).
        max_context_tokens: Maximum token count for the model input.
            GPT-2 position limit is 1024. 0 = no limit. Acts as a hard
            truncation safety net after turn-based eviction.
        keep_turns: Number of most recent completed turns to keep when the
            input exceeds max_context_tokens. The current incomplete turn
            is always kept. 0 = keep only the current incomplete turn.
        onnx_threads: Number of intra-op threads for ONNX Runtime.
    """

    checkpoint_path: str | None = None
    onnx_model_path: str = "models/turngpt/turngpt_v2_kvcache_int8.onnx"
    tokenizer_path: str = "models/turngpt/tokenizer"
    device: str = "cpu"
    max_context_tokens: int = 1024
    keep_turns: int = 2
    onnx_threads: int = 2


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

    # --- Similarity gate ---
    similarity_threshold: float = 0.8

    def __post_init__(self) -> None:
        if not (0.0 <= self.vap_user_threshold <= 1.0):
            raise ConfigurationError(
                f"vap_user_threshold must be in [0, 1], got {self.vap_user_threshold}"
            )
        if not (0.0 <= self.interrupt_user_threshold <= 1.0):
            raise ConfigurationError(
                f"interrupt_user_threshold must be in [0, 1], got {self.interrupt_user_threshold}"
            )
        if not (0.0 <= self.prepare_turngpt_threshold <= 1.0):
            raise ConfigurationError(
                f"prepare_turngpt_threshold must be in [0, 1], got "
                f"{self.prepare_turngpt_threshold}"
            )
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ConfigurationError(
                f"similarity_threshold must be in [0, 1], got {self.similarity_threshold}"
            )


@dataclass
class OrchestratorConfig:
    """Configuration for the Orchestrator conversation loop.

    Attributes:
        exit_keywords: Words that end the conversation (case-insensitive).
        session_timeout_sec: Inactivity timeout before auto-exit.
        frame_timeout_sec: audio_queue.get() timeout per frame.
        stop_pending_timeout_sec: Watchdog timeout for STOP_PENDING state.
        audio_starvation_timeout_sec: Terminate session if no audio frames
            arrive for this long. Detects AudioInput thread death regardless
            of playback or generation state.
        awaiting_cancel_grace_sec: Grace period after turn_shift before
            ASR text changes trigger awaiting cancellation. Filters out
            ASR finalization noise.
    """

    exit_keywords: tuple[str, ...] = ("bye", "goodbye")
    session_timeout_sec: float = 60.0
    frame_timeout_sec: float = 0.1
    stop_pending_timeout_sec: float = 5.0
    audio_starvation_timeout_sec: float = 5.0
    awaiting_cancel_grace_sec: float = 0.5


@dataclass
class SpeechGeneratorConfig:
    """Configuration for the SpeechGenerator.

    Attributes:
        max_workers: Thread pool size for background generation.
            Default 2 so a new prepare() run starts immediately on a
            separate worker while the cancelled run drains cooperatively.
        pipeline_mode: TTS pipeline mode. "full" collects all LLM text
            before TTS. "sentence" streams to TTS per sentence for
            lower first-audio latency.
        query_context_turns: Number of recent history turns concatenated
            with current STT text to form the memory retriever query.
        min_flush_words: Minimum word count before a detected sentence
            boundary triggers a TTS flush in sentence mode.  Short
            fragments (e.g. "Sure!") are accumulated with the next
            sentence to avoid tiny TTS calls.
    """

    max_workers: int = 2
    pipeline_mode: Literal["full", "sentence"] = "full"
    query_context_turns: int = 3
    min_flush_words: int = 4

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ConfigurationError(f"max_workers must be at least 1, got {self.max_workers}")
        if self.pipeline_mode not in ("full", "sentence"):
            raise ConfigurationError(
                f"pipeline_mode must be 'full' or 'sentence', got {self.pipeline_mode!r}"
            )
        if self.min_flush_words < 1:
            raise ConfigurationError(
                f"min_flush_words must be at least 1, got {self.min_flush_words}"
            )


@dataclass
class AudioInputConfig:
    """Configuration for the AudioInput module.

    Attributes:
        device_index: PyAudio device index. None = system default.
        capture_channels: Number of channels to capture from device.
            None = use AudioConfig.channels (default mono).
            Set to 6 for ReSpeaker 6ch firmware.
        extract_channel: Which channel to extract when capture_channels
            differs from AudioConfig.channels. Default 0 (first channel).
    """

    device_index: int | None = None
    capture_channels: int | None = None
    extract_channel: int = 0


@dataclass
class GreetingAudioConfig:
    """Configuration for pre-generating greeting/farewell audio via TTS.

    Attributes:
        audio_dir: Directory for generated audio files (relative to C++ working dir).
        greeting_text: Text to synthesize for greeting audio.
        farewell_text: Text to synthesize for farewell audio.
        fallback_greeting_path: Pre-recorded greeting file used when TTS generation fails.
        fallback_farewell_path: Pre-recorded farewell file used when TTS generation fails.
    """

    audio_dir: str = "assets/audio"
    greeting_text: str = "Yes, how can I help you?"
    farewell_text: str = "Talk to you next time!"
    fallback_greeting_path: str = "assets/audio/greeting.wav"
    fallback_farewell_path: str = "assets/audio/farewell.wav"


@dataclass
class SessionConfig:
    """Configuration for the SessionManager.

    Attributes:
        audio_queue_size: Bounded queue size for audio frames.
        greeting_timeout_sec: Max wait for greeting playback completion.
        farewell_timeout_sec: Max wait for farewell playback completion.
        frame_timeout_sec: Queue.get() timeout for audio frames.
    """

    audio_queue_size: int = 300
    greeting_timeout_sec: float = 10.0
    farewell_timeout_sec: float = 10.0
    frame_timeout_sec: float = 0.1


@dataclass
class MemoryConfig:
    """Configuration for the long-term memory system.

    Attributes:
        db_path: SQLite database path (shared with conversation history).
        embedding_model: Sentence-transformers model name for embeddings.
        embedding_backend: Embedding provider ('local' or 'api').
        embedding_dimension: Vector dimension of the embedding model.
        use_onnx: Use ONNX Runtime backend for local embedding model.
        max_memories: Maximum episodes injected into block 4 per turn.
        min_new_slots: Minimum slots reserved for new search results.
        retained_ttl: Turns a cited memory stays in the retained buffer.
        vector_top_k: Candidate count from vector search.
        bm25_top_k: Candidate count from BM25 search.
        rrf_k: RRF constant (original paper default 60).
        recency_half_life_days: Exponential decay half-life in days.
        salience_threshold: Minimum salience to include (0.0 = disabled).
        write_llm: LLM configuration for memory write (episode/profile extraction).
        write_max_input_tokens: Max tokens per episode extraction window.
            Session is processed in a single call when under this limit.
        write_window_overlap_ratio: Fraction of window tokens to overlap
            between adjacent windows (0.0–1.0).
        write_dedup_threshold: Cosine similarity threshold for candidate
            duplicate episode detection across windows.
        profile_max_content_tokens: Max tokens per profile slot content.
        profile_max_subtopics: Max subtopics per topic before reorganization.
    """

    db_path: str = "data/ray.db"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_backend: Literal["local", "api"] = "local"
    embedding_dimension: int = 384
    use_onnx: bool = False

    # Retrieval (Phase 2)
    max_memories: int = 10
    min_new_slots: int = 4
    retained_ttl: int = 3
    vector_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60
    recency_half_life_days: float = 30.0
    salience_threshold: float = 0.0

    # Write (Phase 3)
    write_llm: LLMConfig = field(
        default_factory=lambda: LLMConfig(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=4096,
            tools=[],
        )
    )
    write_max_input_tokens: int = 8000
    write_window_overlap_ratio: float = 0.25
    write_dedup_threshold: float = 0.8
    profile_max_content_tokens: int = 128
    profile_max_subtopics: int = 20

    def __post_init__(self) -> None:
        if self.embedding_dimension <= 0:
            raise ConfigurationError(
                f"embedding_dimension must be positive, got {self.embedding_dimension}"
            )
        if self.max_memories <= 0:
            raise ConfigurationError(f"max_memories must be positive, got {self.max_memories}")
        if not (0 < self.min_new_slots <= self.max_memories):
            raise ConfigurationError(
                f"min_new_slots must be in (0, max_memories], got {self.min_new_slots}"
            )
        if self.retained_ttl < 1:
            raise ConfigurationError(f"retained_ttl must be >= 1, got {self.retained_ttl}")
        if self.vector_top_k <= 0:
            raise ConfigurationError(f"vector_top_k must be positive, got {self.vector_top_k}")
        if self.bm25_top_k <= 0:
            raise ConfigurationError(f"bm25_top_k must be positive, got {self.bm25_top_k}")
        if self.rrf_k <= 0:
            raise ConfigurationError(f"rrf_k must be positive, got {self.rrf_k}")
        if self.recency_half_life_days <= 0:
            raise ConfigurationError(
                f"recency_half_life_days must be positive, got {self.recency_half_life_days}"
            )
        if not (0.0 <= self.salience_threshold <= 1.0):
            raise ConfigurationError(
                f"salience_threshold must be in [0, 1], got {self.salience_threshold}"
            )
        if self.write_max_input_tokens <= 0:
            raise ConfigurationError(
                f"write_max_input_tokens must be positive, got {self.write_max_input_tokens}"
            )
        if not (0.0 <= self.write_window_overlap_ratio < 1.0):
            raise ConfigurationError(
                f"write_window_overlap_ratio must be in [0, 1),"
                f" got {self.write_window_overlap_ratio}"
            )
        if not (0.0 < self.write_dedup_threshold <= 1.0):
            raise ConfigurationError(
                f"write_dedup_threshold must be in (0, 1], got {self.write_dedup_threshold}"
            )
        if self.profile_max_content_tokens <= 0:
            raise ConfigurationError(
                f"profile_max_content_tokens must be positive,"
                f" got {self.profile_max_content_tokens}"
            )
        if self.profile_max_subtopics <= 0:
            raise ConfigurationError(
                f"profile_max_subtopics must be positive, got {self.profile_max_subtopics}"
            )


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
    maai_vap: MaAIVAPConfig = field(default_factory=MaAIVAPConfig)
    turngpt: TurnGPTConfig = field(default_factory=TurnGPTConfig)
    turn_detector: TurnDetectorConfig = field(default_factory=TurnDetectorConfig)
    greeting_audio: GreetingAudioConfig = field(default_factory=GreetingAudioConfig)
    speech_generator: SpeechGeneratorConfig = field(default_factory=SpeechGeneratorConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    audio_input: AudioInputConfig = field(default_factory=AudioInputConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
