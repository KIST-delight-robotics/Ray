"""Pipeline execution sandbox for bug reproduction.

Provides building blocks to run the actual production pipeline with
controlled inputs.  Hardware-dependent components (mic, C++ bridge,
LED) are replaced with minimal stubs; everything else uses real
production code.

Usage (SpeechGenerator level)::

    from scripts.sandbox import (
        CaptureTTS, ObservableLLM, run_pipeline,
        setup_history, setup_memory,
    )
    from voice_pipeline.core.config import LLMConfig, SpeechGeneratorConfig
    from voice_pipeline.llm.llm import OpenAILLM
    from voice_pipeline.llm.prompts import DEFAULT_SYSTEM_PROMPT
    from voice_pipeline.llm.token_counter import create_token_counter
    from voice_pipeline.context.context_builder import ContextBuilder
    from voice_pipeline.generation.speech_generator import SpeechGenerator

    tc = create_token_counter("gpt-4o")
    history = setup_history(tc)
    tts = CaptureTTS()
    llm = ObservableLLM(OpenAILLM(LLMConfig(tools=[])))
    cb = ContextBuilder(history, ..., DEFAULT_SYSTEM_PROMPT, tc)
    gen = SpeechGenerator(cb, llm, tts, SpeechGeneratorConfig())

    result = run_pipeline(gen, "안녕하세요", tts=tts, llm=llm)
    print(result.tts_inputs, result.raw_llm_output)
    gen.shutdown()

Usage (Orchestrator level)::

    from scripts.sandbox import setup_orchestrator, run_orchestrator

    setup = setup_orchestrator(["hello how are you", "goodbye"])
    run_orchestrator(setup)
    print(setup.history.get_messages())
    print(setup.tts.calls)
    setup.cleanup()

Requires: OPENAI_API_KEY environment variable for real LLM calls.
"""

from __future__ import annotations

import contextlib
import queue
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from voice_pipeline.context.context_builder import ContextBuilder
from voice_pipeline.core.config import (
    AudioConfig,
    ConversationHistoryConfig,
    LLMConfig,
    MemoryConfig,
    OrchestratorConfig,
    SpeechGeneratorConfig,
    TTSConfig,
    TurnDetectorConfig,
)
from voice_pipeline.core.interfaces import (
    IASR,
    ILLM,
    ITTS,
    IVAP,
    ICppBridge,
    IEmbedder,
    ILEDController,
    IMemoryStorage,
    ITurnGPT,
)
from voice_pipeline.core.types import (
    AudioFrame,
    CppEvent,
    CppEventType,
    GeneratorState,
    LEDState,
    LLMResult,
    LLMStream,
    PipelineTrace,
    ResponseData,
    TTSStream,
    TurnDecision,
    VAPResult,
    WordTimestamp,
)
from voice_pipeline.generation.speech_generator import SpeechGenerator
from voice_pipeline.history.conversation_history import ConversationHistory
from voice_pipeline.history.storage_backend import MemoryStorageBackend
from voice_pipeline.llm.llm import OpenAILLM
from voice_pipeline.llm.prompts import DEFAULT_SYSTEM_PROMPT
from voice_pipeline.llm.token_counter import create_token_counter
from voice_pipeline.memory.retriever import MemoryRetriever
from voice_pipeline.memory.storage import SQLiteMemoryStorage
from voice_pipeline.memory.types import Episode, MemoryReadResult, Profile
from voice_pipeline.memory.vector_index import NumpyVectorIndex
from voice_pipeline.orchestrator.orchestrator import Orchestrator
from voice_pipeline.tts.utterance_truncator import TimestampTruncator
from voice_pipeline.turn_taking.async_turngpt import SyncTurnGPTAdapter
from voice_pipeline.turn_taking.turn_detector import TurnDetector

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AUDIO_CONFIG = AudioConfig(sample_rate=16000, channels=1, frame_duration_ms=30, sample_width=2)
_FRAME_BYTES = _AUDIO_CONFIG.sample_rate * _AUDIO_CONFIG.frame_duration_ms * _AUDIO_CONFIG.sample_width // 1000
_SILENCE_FRAME: AudioFrame = b"\x00" * _FRAME_BYTES

# Fast turn-detection config for sandbox (instant turn shift).
_FAST_TURN_DETECTOR_CONFIG = TurnDetectorConfig(
    vap_user_threshold=0.5,
    min_gap_time_sec=0.03,
    turngpt_thresholds=((0.3, 0.03), (0.0, 0.06)),
    interrupt_user_threshold=0.5,
    prepare_turngpt_threshold=0.2,
    prepare_timeout_sec=0.06,
)

_DEFAULT_ORCH_CONFIG = OrchestratorConfig(
    exit_keywords=("goodbye",),
    session_timeout_sec=30.0,
    frame_timeout_sec=0.05,
    stop_pending_timeout_sec=2.0,
)

# ---------------------------------------------------------------------------
# 1. Result types
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Observable outputs from a single SpeechGenerator pipeline run."""

    clean_text: str = ""
    tts_inputs: list[str] = field(default_factory=list)
    raw_llm_output: str = ""
    response_data: ResponseData | None = None
    trace: PipelineTrace | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# 2. Hardware stubs
# ---------------------------------------------------------------------------


class SandboxBridge(ICppBridge):
    """Simulates C++ audio playback process with immediate event responses."""

    def __init__(self) -> None:
        self._events: queue.Queue[CppEvent] = queue.Queue()
        self.audio_chunks: list[bytes] = []
        self.play_files: list[str] = []
        self.stream_start_count = 0
        self.audio_end_count = 0
        self.stop_count = 0

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def send_stream_start(self) -> None:
        self.stream_start_count += 1
        self._events.put(CppEvent(CppEventType.PLAYBACK_STARTED))

    def send_audio(self, audio: bytes) -> None:
        self.audio_chunks.append(audio)

    def send_audio_end(self) -> None:
        self.audio_end_count += 1
        self._events.put(CppEvent(CppEventType.PLAYBACK_COMPLETE))

    def send_stop(self) -> None:
        self.stop_count += 1
        self._events.put(CppEvent(CppEventType.PLAYBACK_COMPLETE))

    def send_play_file(self, file_path: str) -> None:
        self.play_files.append(file_path)
        self._events.put(CppEvent(CppEventType.PLAYBACK_COMPLETE))

    def poll_event(self) -> CppEvent | None:
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None


class NoOpLED(ILEDController):
    """No-op LED controller."""

    def set_state(self, state: LEDState) -> None:
        pass

    def close(self) -> None:
        pass


class ScriptedASR(IASR):
    """Returns predetermined text, advancing on reset().

    After ``feed_audio()`` is called at least once, ``get_text()``
    returns the current utterance.  ``reset()`` moves to the next one.
    """

    def __init__(self, utterances: list[str]) -> None:
        self._utterances = utterances
        self._idx = 0
        self._feed_count = 0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def feed_audio(self, frame: AudioFrame) -> None:
        self._feed_count += 1

    def get_text(self) -> str:
        if self._idx >= len(self._utterances):
            return ""
        if self._feed_count >= 1:
            return self._utterances[self._idx]
        return ""

    def reset(self) -> None:
        self._idx += 1
        self._feed_count = 0


class FakeVAP(IVAP):
    """Always returns robot-favoring probabilities for fast turn shift."""

    def feed_audio(
        self, user_audio: AudioFrame, robot_audio: AudioFrame | None = None
    ) -> VAPResult:
        return VAPResult(p_now=0.2, p_fut=0.2, user_is_speaking=False)

    def reset(self) -> None:
        pass


class FakeTurnGPT(ITurnGPT):
    """Returns a fixed turn-shift probability."""

    def predict(self, dialog_text: str) -> float:
        return 0.5

    def reset(self) -> None:
        pass


class StubEmbedder(IEmbedder):
    """Returns zero vectors.  Cosine similarity = 0.0 < threshold,
    so TurnDetector's similarity gate never blocks prepare()."""

    def __init__(self, dimension: int = 384) -> None:
        self._dim = dimension

    def embed(self, text: str) -> np.ndarray:
        return np.zeros(self._dim, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), self._dim), dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dim


class FrameFeeder:
    """Feeds silence frames to a queue on a background thread."""

    def __init__(self, audio_queue: queue.Queue[AudioFrame], interval: float = 0.005) -> None:
        self._queue = audio_queue
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            with contextlib.suppress(queue.Full):
                self._queue.put(_SILENCE_FRAME, timeout=0.01)
            self._stop.wait(self._interval)


# ---------------------------------------------------------------------------
# 3. Capture wrappers
# ---------------------------------------------------------------------------


class CaptureTTS(ITTS):
    """Records text inputs and produces minimal valid audio.

    Thread-safe: ``synthesize()`` may be called concurrently from the
    TTS executor in sentence mode.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls: list[str] = []

    def synthesize(self, text: str) -> TTSStream:
        with self._lock:
            self.calls.append(text)

        words = text.split()

        def _gen() -> Any:
            chunk = b"\x00" * 4800  # 100ms silence at 24kHz 16-bit mono
            for _ in range(3):
                yield chunk

        def _ts_fn() -> tuple[WordTimestamp, ...]:
            return tuple(
                WordTimestamp(word=w, start_sec=i * 0.1, end_sec=(i + 1) * 0.1)
                for i, w in enumerate(words)
            )

        return TTSStream(_gen(), timestamps_fn=_ts_fn)


class ObservableLLM(ILLM):
    """Wraps a real LLM and captures raw output text.

    ``self.calls`` accumulates the raw text from each ``generate()``
    invocation.  If the stream is cancelled (``close()``), no text
    is recorded for that call.
    """

    def __init__(self, real_llm: ILLM) -> None:
        self._real = real_llm
        self._lock = threading.Lock()
        self.calls: list[str] = []

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMStream:
        inner = self._real.generate(messages, tools, response_format)
        captured_chunks: list[str] = []
        obs = self

        def _tee() -> Any:
            for chunk in inner:
                captured_chunks.append(chunk)
                yield chunk
            # Normal completion — record raw text.
            with obs._lock:
                obs.calls.append("".join(captured_chunks))

        def _result_fn(_full_text: str) -> LLMResult:
            return inner.result

        return LLMStream(_tee(), close_fn=inner.close, result_fn=_result_fn)


# ---------------------------------------------------------------------------
# 4. Setup helpers
# ---------------------------------------------------------------------------


def setup_memory(
    episodes: list[Episode],
    embedder: IEmbedder,
    config: MemoryConfig | None = None,
) -> tuple[SQLiteMemoryStorage, MemoryRetriever, tempfile.TemporaryDirectory[str]]:
    """Create an isolated memory store pre-loaded with episodes.

    Uses real SQLiteMemoryStorage (FTS5 BM25) for production-identical
    retrieval.  Caller should call ``tmpdir.cleanup()`` when done.

    Returns:
        (storage, retriever, tmpdir)
    """
    tmpdir = tempfile.TemporaryDirectory()
    cfg = config or MemoryConfig(
        db_path=f"{tmpdir.name}/sandbox.db",
        embedding_dimension=embedder.dimension,
    )
    if cfg.db_path == MemoryConfig().db_path:
        # Override default path to avoid touching production DB.
        cfg = MemoryConfig(
            db_path=f"{tmpdir.name}/sandbox.db",
            embedding_dimension=cfg.embedding_dimension,
            max_memories=cfg.max_memories,
            min_new_slots=cfg.min_new_slots,
            retained_ttl=cfg.retained_ttl,
            vector_top_k=cfg.vector_top_k,
            bm25_top_k=cfg.bm25_top_k,
            rrf_k=cfg.rrf_k,
            recency_half_life_days=cfg.recency_half_life_days,
            salience_threshold=cfg.salience_threshold,
        )

    storage = SQLiteMemoryStorage(cfg)

    # Add episodes and compute embeddings.
    texts = [ep.text for ep in episodes]
    embeddings = embedder.embed_batch(texts)
    for ep, emb in zip(episodes, embeddings):
        eid = storage.add_episode(ep)
        if eid is not None:
            storage.update_episode_embedding(eid, emb)

    # Build vector index from stored data.
    vector_index = NumpyVectorIndex()
    ids, vecs = storage.load_all_embeddings()
    if ids:
        vector_index.load(ids, vecs)

    retriever = MemoryRetriever(storage, vector_index, embedder, cfg)
    return storage, retriever, tmpdir


def setup_history(
    token_counter: Any,
    turns: list[tuple[str, str]] | None = None,
) -> ConversationHistory:
    """Create an in-memory ConversationHistory, optionally pre-populated.

    Args:
        token_counter: ``Callable[[str], int]`` token counter.
        turns: Optional list of ``(role, text)`` pairs to pre-populate.
            role must be ``"user"`` or ``"assistant"``.
    """
    backend = MemoryStorageBackend()
    history = ConversationHistory(backend, token_counter)
    history.new_session("sandbox")
    if turns:
        for role, text in turns:
            if role == "user":
                history.add_user_message(text)
            elif role == "assistant":
                history.add_assistant_message(text)
    return history


# ---------------------------------------------------------------------------
# 5. Pipeline runner (SpeechGenerator level)
# ---------------------------------------------------------------------------


def run_pipeline(
    gen: SpeechGenerator,
    input_text: str,
    *,
    tts: CaptureTTS | None = None,
    llm: ObservableLLM | None = None,
    timeout: float = 60.0,
) -> PipelineResult:
    """Run a single SpeechGenerator pipeline execution and collect results.

    Calls ``gen.prepare(input_text)``, polls until completion or failure,
    and returns a :class:`PipelineResult`.

    The generator's lifecycle (``shutdown()``) is the caller's
    responsibility.
    """
    # Snapshot capture indices for slicing after run.
    tts_start = len(tts.calls) if tts else 0
    llm_start = len(llm.calls) if llm else 0

    gen.prepare(input_text)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = gen.state
        if state == GeneratorState.FAILED:
            return PipelineResult(
                error="Pipeline failed (GeneratorState.FAILED)",
                trace=gen.trace,
            )
        if gen.stream_done:
            break
        time.sleep(0.05)
    else:
        return PipelineResult(error=f"Timeout after {timeout}s", trace=gen.trace)

    # Collect results.  get_text() before get_response_data() because
    # get_response_data() transitions state to IDLE.
    clean_text = gen.get_text()
    trace = gen.trace
    try:
        response_data = gen.get_response_data()
    except RuntimeError:
        response_data = None

    tts_inputs = list(tts.calls[tts_start:]) if tts else []
    raw_llm_output = llm.calls[llm_start] if llm and len(llm.calls) > llm_start else ""

    return PipelineResult(
        clean_text=clean_text,
        tts_inputs=tts_inputs,
        raw_llm_output=raw_llm_output,
        response_data=response_data,
        trace=trace,
    )


# ---------------------------------------------------------------------------
# 6. Orchestrator level setup + runner
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorSetup:
    """Holds all wired components for an Orchestrator-level sandbox run."""

    orchestrator: Orchestrator
    history: ConversationHistory
    generator: SpeechGenerator
    bridge: SandboxBridge
    tts: CaptureTTS
    llm: ObservableLLM
    audio_queue: queue.Queue[AudioFrame]
    feeder: FrameFeeder
    memory_storage: IMemoryStorage | None = None
    retriever: MemoryRetriever | None = None
    _tmpdir: tempfile.TemporaryDirectory[str] | None = field(default=None, repr=False)

    def cleanup(self) -> None:
        """Shut down executor and clean up temp files."""
        self.generator.shutdown()
        if self._tmpdir is not None:
            self._tmpdir.cleanup()


def setup_orchestrator(
    asr_texts: list[str],
    *,
    llm: ObservableLLM | None = None,
    tts: CaptureTTS | None = None,
    episodes: list[Episode] | None = None,
    profiles: list[Profile] | None = None,
    history_turns: list[tuple[str, str]] | None = None,
    session_summaries: list[str] | None = None,
    system_prompt: str | None = None,
    embedder: IEmbedder | None = None,
    llm_config: LLMConfig | None = None,
    gen_config: SpeechGeneratorConfig | None = None,
    orch_config: OrchestratorConfig | None = None,
    memory_config: MemoryConfig | None = None,
) -> OrchestratorSetup:
    """Wire up a complete Orchestrator with real internal modules.

    Only hardware-dependent boundaries (ASR, Bridge, LED, VAP, TurnGPT)
    are stubbed.  Everything else uses production code.

    Args:
        asr_texts: Scripted ASR utterances.  Include an exit keyword
            (default ``"goodbye"``) as the last entry for clean exit.
    """
    # -- Stubs --
    asr = ScriptedASR(asr_texts)
    bridge = SandboxBridge()
    led = NoOpLED()
    vap = FakeVAP()
    turngpt_adapter = SyncTurnGPTAdapter(FakeTurnGPT())

    # -- Token counter --
    token_counter = create_token_counter("gpt-4o")

    # -- History --
    history = setup_history(token_counter, history_turns)

    # -- Memory (optional) --
    tmpdir: tempfile.TemporaryDirectory[str] | None = None
    retriever: MemoryRetriever | None = None
    memory_storage: IMemoryStorage | None = None
    td_embedder: IEmbedder

    if episodes:
        _embedder = embedder or _lazy_load_embedder()
        memory_storage, retriever, tmpdir = setup_memory(episodes, _embedder, memory_config)
        td_embedder = _embedder
    else:
        td_embedder = embedder or StubEmbedder()

    # -- Context builder --
    _profiles = profiles or []
    if memory_storage and _profiles == []:
        _profiles = list(memory_storage.get_all_profiles())

    history_config = ConversationHistoryConfig(max_context_tokens=4096, storage_backend="memory")
    cb = ContextBuilder(
        history,
        history_config,
        system_prompt or DEFAULT_SYSTEM_PROMPT,
        token_counter,
        profiles=_profiles or None,
        session_summaries=session_summaries,
    )

    # -- LLM + TTS --
    _tts = tts or CaptureTTS()
    _llm = llm or ObservableLLM(OpenAILLM(llm_config or LLMConfig(tools=[])))

    # -- SpeechGenerator --
    _gen_config = gen_config or SpeechGeneratorConfig()
    generator = SpeechGenerator(
        cb,
        _llm,
        _tts,
        _gen_config,
        retriever=retriever,
        history=history,
        exclude_session_ids={"sandbox"},
    )

    # -- TurnDetector --
    turn_detector = TurnDetector(
        vap,
        turngpt_adapter,
        td_embedder,
        _FAST_TURN_DETECTOR_CONFIG,
        _AUDIO_CONFIG,
    )

    # -- Orchestrator --
    _orch_config = orch_config or _DEFAULT_ORCH_CONFIG
    tts_config = TTSConfig(output_sample_rate=24000)
    truncator = TimestampTruncator()

    orchestrator = Orchestrator(
        asr=asr,
        turn_detector=turn_detector,
        speech_generator=generator,
        cpp_bridge=bridge,
        history=history,
        truncator=truncator,
        led=led,
        config=_orch_config,
        tts_config=tts_config,
        audio_config=_AUDIO_CONFIG,
        memory_storage=memory_storage,
        session_id="sandbox",
        token_counter=token_counter,
    )

    # -- Audio queue + feeder --
    audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=300)
    feeder = FrameFeeder(audio_queue)

    return OrchestratorSetup(
        orchestrator=orchestrator,
        history=history,
        generator=generator,
        bridge=bridge,
        tts=_tts,
        llm=_llm,
        audio_queue=audio_queue,
        feeder=feeder,
        memory_storage=memory_storage,
        retriever=retriever,
        _tmpdir=tmpdir,
    )


def run_orchestrator(setup: OrchestratorSetup) -> None:
    """Run the Orchestrator to completion.

    Starts the frame feeder, runs the orchestrator loop, then stops
    the feeder and resets the generator.

    After return, inspect ``setup.history``, ``setup.tts.calls``,
    ``setup.llm.calls``, ``setup.bridge.audio_chunks``, etc.
    """
    setup.feeder.start()
    try:
        setup.orchestrator.run(setup.audio_queue)
    finally:
        setup.feeder.stop()
        setup.generator.reset()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _lazy_load_embedder() -> IEmbedder:
    """Load SentenceTransformerEmbedder on demand."""
    from voice_pipeline.embedding.embedder import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder("all-MiniLM-L6-v2", expected_dimension=384)
