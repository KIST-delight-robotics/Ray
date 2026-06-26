"""Pipeline execution sandbox.

Provides a unified setup for running production pipeline modules with
real or stubbed components.  One ``setup_sandbox()`` assembles all
modules; separate runner functions execute different pipeline segments.

Usage (turn detection monitor — all real)::

    from scripts.sandbox import setup_sandbox, run_turn_monitor
    from voice_pipeline.audio.audio_input import AudioInput

    AudioInput._DEVICE_INDEX = 3
    AudioInput._CAPTURE_CHANNELS = 6
    AudioInput._EXTRACT_CHANNEL = 0
    setup = setup_sandbox()
    run_turn_monitor(setup)   # Ctrl+C to stop
    setup.cleanup()

Usage (generation pipeline — stub turn-taking to skip model loading)::

    from scripts.sandbox import (
        setup_sandbox, run_pipeline,
        ScriptedASR, FakeVAP, FakeTurnGPT, SandboxBridge,
    )

    setup = setup_sandbox(
        asr=ScriptedASR([]), vap=FakeVAP(), turngpt=FakeTurnGPT(),
        bridge=SandboxBridge(),
    )
    result = run_pipeline(setup, "안녕하세요")
    print(result.clean_text, result.trace.summary())
    setup.cleanup()

Usage (session loop with sound — mock server or C++ process required)::

    from scripts.sandbox import (
        setup_sandbox, run_session_loop,
        ScriptedASR, FakeVAP, FakeTurnGPT,
        apply_fast_turn_detector_config,
    )

    apply_fast_turn_detector_config()
    setup = setup_sandbox(
        asr=ScriptedASR(["hello", "goodbye"]),
        vap=FakeVAP(), turngpt=FakeTurnGPT(),
    )
    run_session_loop(setup)
    setup.cleanup()

Requires: OPENAI_API_KEY for LLM/TTS, GOOGLE_APPLICATION_CREDENTIALS
for real ASR, ONNX model files for VAP/TurnGPT, WebSocket server for
real CppBridge.
"""

from __future__ import annotations

import contextlib
import queue
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from voice_pipeline.audio.constants import FRAME_SIZE_BYTES
from voice_pipeline.core.interfaces import (
    IASR,
    ILLM,
    ITTS,
    IVAP,
    IAudioInput,
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
from voice_pipeline.memory.types import Episode, Profile
from voice_pipeline.memory.vector_index import NumpyVectorIndex
from voice_pipeline.session_loop import SessionLoop
from voice_pipeline.tts.openai_tts import OpenAITTS
from voice_pipeline.turn_taking.threaded_turngpt import SyncTurnGPTAdapter
from voice_pipeline.turn_taking.turn_detector import TurnDetector

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SILENCE_FRAME: AudioFrame = b"\x00" * FRAME_SIZE_BYTES


def apply_fast_turn_detector_config() -> None:
    """Override TurnDetector class vars for scripted ASR scenarios (instant turn shift).

    Call before ``setup_sandbox`` when using ScriptedASR with FakeVAP/FakeTurnGPT.
    """
    from voice_pipeline.turn_taking.turn_detector import TurnDetector

    TurnDetector._MIN_GAP_TIME_SEC = 0.03
    TurnDetector._TURNGPT_THRESHOLDS = ((0.3, 0.03), (0.0, 0.06))
    TurnDetector._PREPARE_TIMEOUT_SEC = 0.06


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


@dataclass
class TurnMonitorFrame:
    """Per-frame monitoring data from the turn detection monitor."""

    elapsed_sec: float
    asr_text: str
    vap_result: VAPResult
    turngpt_prob: float
    silence_sec: float
    vap_favor_sec: float
    decision: TurnDecision


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
    """Always reports robot-favoring probabilities for fast turn shift."""

    def feed_audio(self, user_audio: AudioFrame, robot_audio: AudioFrame | None = None) -> None:
        pass

    @property
    def latest_result(self) -> VAPResult:
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

    output_sample_rate: int = OpenAITTS.OUTPUT_SAMPLE_RATE
    voice_id: str = "sandbox|capture"

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
            return tuple(WordTimestamp(word=w, start_sec=i * 0.1, end_sec=(i + 1) * 0.1) for i, w in enumerate(words))

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


class ObservableVAP(IVAP):
    """Wraps a real VAP runtime and captures the latest result for monitoring.

    ``last_result`` snapshots ``latest_result`` on every ``feed_audio()`` call.
    """

    def __init__(self, real_vap: IVAP) -> None:
        self._real = real_vap
        self.last_result: VAPResult = VAPResult(p_now=0.5, p_fut=0.5, user_is_speaking=False)

    def feed_audio(self, user_audio: AudioFrame, robot_audio: AudioFrame | None = None) -> None:
        self._real.feed_audio(user_audio, robot_audio)
        self.last_result = self._real.latest_result

    @property
    def latest_result(self) -> VAPResult:
        return self._real.latest_result

    def reset(self) -> None:
        self._real.reset()

    def stop(self) -> None:
        stop = getattr(self._real, "stop", None)
        if stop is not None:
            stop()


# ---------------------------------------------------------------------------
# 4. Setup helpers
# ---------------------------------------------------------------------------


def setup_memory(
    episodes: list[Episode],
    embedder: IEmbedder,
) -> tuple[SQLiteMemoryStorage, MemoryRetriever, tempfile.TemporaryDirectory[str]]:
    """Create an isolated memory store pre-loaded with episodes.

    Uses real SQLiteMemoryStorage (FTS5 BM25) for production-identical
    retrieval.  Caller should call ``tmpdir.cleanup()`` when done.

    Override retriever tuning by setting class vars on ``MemoryRetriever``
    before calling (e.g. ``MemoryRetriever._MAX_MEMORIES = 5``).

    Returns:
        (storage, retriever, tmpdir)
    """
    tmpdir = tempfile.TemporaryDirectory()
    storage = SQLiteMemoryStorage(
        f"{tmpdir.name}/sandbox.db",
        dimension=embedder.dimension,
    )

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

    retriever = MemoryRetriever(storage, vector_index, embedder)
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
# 5. Module factories
# ---------------------------------------------------------------------------


def create_audio_input(audio_queue: queue.Queue[AudioFrame]) -> IAudioInput:
    """Create a real AudioInput with microphone capture.

    Requires PyAudio. Override device/channel by setting class vars on
    ``AudioInput`` before calling (e.g. ``AudioInput._DEVICE_INDEX = 3``).
    """
    from voice_pipeline.audio.audio_input import AudioInput

    return AudioInput(audio_queue)


def create_asr(*, language_code: str = "en-US") -> IASR:
    """Create a real Google Cloud ASR.

    Requires ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable.
    """
    from voice_pipeline.asr.asr import GoogleCloudASR

    return GoogleCloudASR(language_code=language_code)


def create_vap(
    *,
    tts_sample_rate: int = OpenAITTS.OUTPUT_SAMPLE_RATE,
) -> IVAP:
    """Create a real MaAI VAP runtime (ThreadedVAP wrapping MaAIVAPModel).

    Requires ONNX model files at configured paths. Override tuning values
    by setting class vars on ``MaAIVAPModel`` before calling (e.g.
    ``MaAIVAPModel._FRAME_RATE = 20``).
    """
    from voice_pipeline.turn_taking.maai_vap import MaAIVAPModel
    from voice_pipeline.turn_taking.threaded_vap import ThreadedVAP

    return ThreadedVAP(MaAIVAPModel(tts_sample_rate))


def create_turngpt() -> ITurnGPT:
    """Create a real TurnGPT (ONNX).

    Requires ONNX model and tokenizer files at configured paths. Override tuning values
    by setting class vars on ``TurnGPTWrapper`` before calling (e.g.
    ``TurnGPTWrapper._ONNX_THREADS = 4``).
    """
    from voice_pipeline.turn_taking.turngpt import TurnGPTWrapper

    return TurnGPTWrapper()


def create_tts(vendor: str | None = None) -> ITTS:
    """Create a real TTS via the vendor factory (production parity).

    Requires the vendor's API key env var (``ELEVENLABS_API_KEY`` or
    ``OPENAI_API_KEY``). *vendor*가 None이면 팩토리 기본 vendor 사용.
    """
    from voice_pipeline.tts.factory import create_tts as _create_tts

    return _create_tts(vendor) if vendor is not None else _create_tts()


def create_bridge() -> ICppBridge:
    """Create a real CppBridge (WebSocket).

    Connects to ``localhost:9200`` by default.  Start either the real
    C++ process or ``scripts/mock_cpp_server.py`` before calling
    ``bridge.connect()``.
    """
    from voice_pipeline.bridge.cpp_bridge import CppBridge

    return CppBridge()


# ---------------------------------------------------------------------------
# 6. Sandbox setup
# ---------------------------------------------------------------------------


@dataclass
class SandboxSetup:
    """Holds all wired components for sandbox execution."""

    audio_input: IAudioInput
    audio_queue: queue.Queue[AudioFrame]
    asr: IASR
    turn_detector: TurnDetector
    vap: ObservableVAP
    generator: SpeechGenerator
    llm: ILLM
    tts: ITTS
    bridge: ICppBridge
    history: ConversationHistory
    session_loop: SessionLoop
    memory_storage: IMemoryStorage | None = None
    retriever: MemoryRetriever | None = None
    _tmpdir: tempfile.TemporaryDirectory[str] | None = field(default=None, repr=False)

    def cleanup(self) -> None:
        """Shut down executor, stop the VAP thread, and clean up temp files."""
        self.generator.shutdown()
        stop = getattr(self.vap, "stop", None)
        if stop is not None:
            stop()
        if self._tmpdir is not None:
            self._tmpdir.cleanup()


def setup_sandbox(
    *,
    # Module overrides — pass an instance to replace the default.
    asr: IASR | None = None,
    vap: IVAP | None = None,
    turngpt: ITurnGPT | None = None,
    embedder: IEmbedder | None = None,
    llm: ILLM | None = None,
    tts: ITTS | None = None,
    bridge: ICppBridge | None = None,
    # Data
    episodes: list[Episode] | None = None,
    profiles: list[Profile] | None = None,
    history_turns: list[tuple[str, str]] | None = None,
    session_summaries: list[str] | None = None,
    system_prompt: str | None = None,
) -> SandboxSetup:
    """Wire all pipeline modules into a single sandbox setup.

    All parameters are optional.  Unspecified modules are created with
    real production defaults via ``create_*`` factories.  Pass stubs
    (e.g. ``ScriptedASR``, ``FakeVAP``) to skip hardware or model
    dependencies you don't need.

    The returned :class:`SandboxSetup` can be passed to any runner:

    - :func:`run_turn_monitor` — mic → ASR → TurnDetector loop
    - :func:`run_pipeline` — text → LLM → TTS
    - :func:`run_session_loop` — full SessionLoop frame loop
    """
    # -- Audio input --
    _audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=300)
    _audio_input = create_audio_input(_audio_queue)

    # -- ASR --
    _asr = asr or create_asr()

    # -- VAP + TurnGPT → TurnDetector --
    _raw_vap = vap or create_vap()
    _observable_vap = ObservableVAP(_raw_vap)
    _turngpt = turngpt or create_turngpt()
    _adapter = SyncTurnGPTAdapter(_turngpt)

    # -- Token counter --
    token_counter = create_token_counter("gpt-4o")

    # -- Memory (optional) --
    tmpdir: tempfile.TemporaryDirectory[str] | None = None
    retriever: MemoryRetriever | None = None
    memory_storage: IMemoryStorage | None = None
    _embedder: IEmbedder

    if episodes:
        _embedder = embedder or _lazy_load_embedder()
        memory_storage, retriever, tmpdir = setup_memory(episodes, _embedder)
    else:
        _embedder = embedder or StubEmbedder()

    _turn_detector = TurnDetector(
        _observable_vap,
        _adapter,
        _embedder,
    )

    # -- History --
    history = setup_history(token_counter, history_turns)

    # -- LLM + TTS + Bridge --
    _llm = llm or ObservableLLM(OpenAILLM(tools=[]))
    _tts = tts or create_tts()
    _bridge = bridge or create_bridge()

    # -- SpeechGenerator --
    generator = SpeechGenerator(
        _llm,
        _tts,
        history,
        token_counter,
        system_prompt or DEFAULT_SYSTEM_PROMPT,
        retriever=retriever,
        session_id="sandbox",
    )

    # -- SessionLoop --
    session_loop = SessionLoop(
        asr=_asr,
        turn_detector=_turn_detector,
        speech_generator=generator,
        cpp_bridge=_bridge,
        history=history,
        led=NoOpLED(),
        audio_queue=_audio_queue,
        tts_sample_rate=OpenAITTS.OUTPUT_SAMPLE_RATE,
        memory_storage=memory_storage,
        session_id="sandbox",
        token_counter=token_counter,
    )

    return SandboxSetup(
        audio_input=_audio_input,
        audio_queue=_audio_queue,
        asr=_asr,
        turn_detector=_turn_detector,
        vap=_observable_vap,
        generator=generator,
        llm=_llm,
        tts=_tts,
        bridge=_bridge,
        history=history,
        session_loop=session_loop,
        memory_storage=memory_storage,
        retriever=retriever,
        _tmpdir=tmpdir,
    )


# ---------------------------------------------------------------------------
# 7. Runners
# ---------------------------------------------------------------------------


def run_pipeline(
    setup: SandboxSetup,
    input_text: str,
    *,
    timeout: float = 60.0,
) -> PipelineResult:
    """Run a single SpeechGenerator pipeline execution and collect results.

    Calls ``setup.generator.prepare(input_text)``, polls until completion
    or failure, and returns a :class:`PipelineResult`.
    """
    gen = setup.generator
    tts = setup.tts
    llm = setup.llm

    # Snapshot capture indices (works if tts/llm have a ``calls`` attribute).
    tts_start = len(tts.calls) if hasattr(tts, "calls") else 0
    llm_start = len(llm.calls) if hasattr(llm, "calls") else 0

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

    # get_text() before get_response_data() because the latter transitions
    # state to IDLE.
    clean_text = gen.get_text()
    trace = gen.trace
    try:
        response_data = gen.get_response_data()
    except RuntimeError:
        response_data = None

    tts_inputs = list(tts.calls[tts_start:]) if hasattr(tts, "calls") else []
    raw_llm_output = llm.calls[llm_start] if hasattr(llm, "calls") and len(llm.calls) > llm_start else ""

    return PipelineResult(
        clean_text=clean_text,
        tts_inputs=tts_inputs,
        raw_llm_output=raw_llm_output,
        response_data=response_data,
        trace=trace,
    )


def run_turn_monitor(
    setup: SandboxSetup,
    *,
    callback: Callable[[TurnMonitorFrame], None] | None = None,
) -> None:
    """Run the turn detection monitor until Ctrl+C.

    Feeds microphone audio through ASR and TurnDetector, reporting
    per-frame monitoring data via *callback*.

    After each ``turn_shift``, resets ASR and TurnDetector and
    continues monitoring the next turn.
    """
    cb = callback or _default_turn_callback
    setup.audio_input.start()
    setup.asr.start()
    start = time.monotonic()
    try:
        while True:
            try:
                frame = setup.audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            setup.asr.feed_audio(frame)
            text = setup.asr.get_text()
            decision = setup.turn_detector.process_frame(frame, text)

            # Access TurnDetector internals for monitoring.
            td = setup.turn_detector
            frame_data = TurnMonitorFrame(
                elapsed_sec=time.monotonic() - start,
                asr_text=text,
                vap_result=setup.vap.last_result,
                turngpt_prob=td._turngpt_prob,  # noqa: SLF001
                silence_sec=td._silence_elapsed_sec,  # noqa: SLF001
                vap_favor_sec=td._vap_favor_robot_elapsed_sec,  # noqa: SLF001
                decision=decision,
            )
            cb(frame_data)

            if decision.turn_shift:
                setup.turn_detector.notify_turn_complete("user", text)
                setup.asr.reset()
                setup.turn_detector.reset()
    except KeyboardInterrupt:
        pass
    finally:
        setup.asr.stop()
        setup.audio_input.stop()


def run_session_loop(setup: SandboxSetup) -> None:
    """Run the SessionLoop to completion.

    Connects the bridge, starts audio input and ASR, then runs the
    session loop.  Works with both :class:`SandboxBridge`
    (no-op connect) and real :class:`CppBridge` (WebSocket).
    """
    setup.bridge.connect()
    setup.audio_input.start()
    setup.asr.start()
    try:
        setup.session_loop.run()
    finally:
        setup.asr.stop()
        setup.audio_input.stop()
        setup.generator.reset()
        setup.bridge.disconnect()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _default_turn_callback(frame: TurnMonitorFrame) -> None:
    """Default console output for turn monitor frames.

    Regular frames overwrite the same line (``\\r``).  Events
    (turn_shift, prepare) print on a new line.
    """
    vap = frame.vap_result
    speak = "speak" if vap.user_is_speaking else "     "
    decision_str = ""
    if frame.decision.turn_shift:
        decision_str = " -> TURN_SHIFT"
    elif frame.decision.prepare:
        decision_str = " -> PREPARE"

    line = (
        f"[{frame.elapsed_sec:6.1f}s] "
        f"VAP:{vap.p_now:.2f}/{vap.p_fut:.2f} {speak} "
        f"sil:{frame.silence_sec:.1f}s "
        f"tgpt:{frame.turngpt_prob:.2f}"
    )
    if frame.asr_text:
        line += f" | {frame.asr_text[:40]}"

    if decision_str:
        print(f"\r{line}{decision_str}")
        if frame.decision.turn_shift:
            print(f"--- turn shift: {frame.asr_text!r} ---")
    else:
        print(f"\r{line}", end="", flush=True)


def _lazy_load_embedder() -> IEmbedder:
    """Load SentenceTransformerEmbedder on demand."""
    from voice_pipeline.embedding.embedder import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder("all-MiniLM-L6-v2", expected_dimension=384)
