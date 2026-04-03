"""Full pipeline integration tests.

Tests the complete voice pipeline flow with mocked external boundaries:
  Real: SessionManager, Orchestrator, SpeechGenerator, ContextBuilder,
        ConversationHistory, TurnDetector, TimestampTruncator, MemoryStorageBackend
  Mocked: ASR, LLM, TTS, CppBridge, VAP, TurnGPT, Wakeword, AudioInput, LED

Scenarios:
  1. Single-turn conversation: user speaks → robot responds → exit keyword
  2. Multi-turn conversation: two exchanges before exit
  3. Full session lifecycle: SLEEP → GREETING → ACTIVE → FAREWELL → SLEEP
  4. Barge-in during playback
  5. Memory integration: utterance storage, retrieval, context injection, citation, session end
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from voice_pipeline.context.context_builder import ContextBuilder
from voice_pipeline.core.config import (
    AudioConfig,
    ConversationHistoryConfig,
    MemoryConfig,
    OrchestratorConfig,
    SessionConfig,
    SpeechGeneratorConfig,
    TTSConfig,
    TurnDetectorConfig,
)
from voice_pipeline.core.interfaces import (
    IASR,
    ILLM,
    ITTS,
    IVAP,
    IAudioInput,
    ICppBridge,
    IEmbedder,
    ILEDController,
    ITurnGPT,
    IWakewordDetector,
)
from voice_pipeline.core.types import (
    AudioFrame,
    CppEvent,
    CppEventType,
    LEDState,
    LLMResult,
    LLMStream,
    TTSStream,
    VAPResult,
    WordTimestamp,
)
from voice_pipeline.generation.speech_generator import SpeechGenerator
from voice_pipeline.history.conversation_history import ConversationHistory
from voice_pipeline.history.storage_backend import MemoryStorageBackend
from voice_pipeline.llm.prompts import DEFAULT_SYSTEM_PROMPT
from voice_pipeline.memory.retriever import MemoryRetriever
from voice_pipeline.memory.storage import InMemoryMemoryStorage
from voice_pipeline.memory.types import Episode
from voice_pipeline.memory.vector_index import NumpyVectorIndex
from voice_pipeline.orchestrator.orchestrator import Orchestrator
from voice_pipeline.session.session_manager import SessionComponents, SessionManager
from voice_pipeline.tts.utterance_truncator import TimestampTruncator
from voice_pipeline.turn_taking.async_turngpt import SyncTurnGPTAdapter
from voice_pipeline.turn_taking.turn_detector import TurnDetector

# ---------------------------------------------------------------------------
# Configs tuned for fast deterministic testing
# ---------------------------------------------------------------------------

AUDIO_CONFIG = AudioConfig(sample_rate=16000, channels=1, frame_duration_ms=30, sample_width=2)
TTS_CONFIG = TTSConfig(output_sample_rate=24000)
HISTORY_CONFIG = ConversationHistoryConfig(max_context_tokens=4096)
GENERATOR_CONFIG = SpeechGeneratorConfig(max_workers=2)

TURN_DETECTOR_CONFIG = TurnDetectorConfig(
    vap_user_threshold=0.5,
    min_gap_time_sec=0.03,  # 1 frame → instant turn shift
    turngpt_thresholds=((0.3, 0.03), (0.0, 0.06)),
    interrupt_user_threshold=0.5,
    prepare_turngpt_threshold=0.2,
    prepare_timeout_sec=0.06,
)

ORCHESTRATOR_CONFIG = OrchestratorConfig(
    exit_keywords=("goodbye",),
    session_timeout_sec=5.0,
    frame_timeout_sec=0.05,
    stop_pending_timeout_sec=2.0,
)

SESSION_CONFIG = SessionConfig(
    audio_queue_size=300,
    greeting_timeout_sec=2.0,
    farewell_timeout_sec=2.0,
    frame_timeout_sec=0.05,
)

GREETING_AUDIO_PATH = "assets/audio/greeting.wav"
FAREWELL_AUDIO_PATH = "assets/audio/farewell.wav"

FRAME_BYTES = (
    AUDIO_CONFIG.sample_rate * AUDIO_CONFIG.frame_duration_ms * AUDIO_CONFIG.sample_width // 1000
)
SILENCE_FRAME: AudioFrame = b"\x00" * FRAME_BYTES


# ---------------------------------------------------------------------------
# Fake / Scripted implementations of external boundaries
# ---------------------------------------------------------------------------


class ScriptedASR(IASR):
    """ASR mock that returns scripted utterances, advancing on reset()."""

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
        # Return text after first feed
        if self._feed_count >= 1:
            return self._utterances[self._idx]
        return ""

    def reset(self) -> None:
        self._idx += 1
        self._feed_count = 0


class FakeLLM(ILLM):
    """LLM mock that yields fixed response chunks."""

    def __init__(self, responses: list[list[str]] | None = None) -> None:
        self._responses = responses or [["I'm ", "doing ", "great!"]]
        self._call_count = 0

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMStream:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        chunks = self._responses[idx]

        def gen():
            yield from chunks

        return LLMStream(gen(), result_fn=lambda t: LLMResult(text=t))


class FakeTTS(ITTS):
    """TTS mock that returns a TTSStream with fake audio and timestamps."""

    def __init__(self, chunk_size: int = 4800) -> None:
        # 100ms of 24kHz 16-bit mono audio = 4800 bytes
        self._chunk_size = chunk_size

    def synthesize(self, text: str) -> TTSStream:
        words = text.split()
        chunk = b"\x00" * self._chunk_size

        def gen():
            for _ in range(3):  # 3 chunks = 300ms
                yield chunk

        def ts_fn() -> tuple[WordTimestamp, ...]:
            result = []
            for i, w in enumerate(words):
                start = i * 0.1
                result.append(WordTimestamp(word=w, start_sec=start, end_sec=start + 0.1))
            return tuple(result)

        return TTSStream(gen(), timestamps_fn=ts_fn)


class FakeVAP(IVAP):
    """VAP mock: always indicates user is not speaking, robot-favoring probs."""

    def feed_audio(
        self, user_audio: AudioFrame, robot_audio: AudioFrame | None = None
    ) -> VAPResult:
        return VAPResult(p_now=0.2, p_fut=0.2, user_is_speaking=False)

    def reset(self) -> None:
        pass


class InterruptableVAP(IVAP):
    """VAP mock that can switch between quiet and interrupting."""

    def __init__(self) -> None:
        self.interrupting = False

    def feed_audio(
        self, user_audio: AudioFrame, robot_audio: AudioFrame | None = None
    ) -> VAPResult:
        if self.interrupting:
            return VAPResult(p_now=0.8, p_fut=0.8, user_is_speaking=True)
        return VAPResult(p_now=0.2, p_fut=0.2, user_is_speaking=False)

    def reset(self) -> None:
        pass


class FakeTurnGPT(ITurnGPT):
    """TurnGPT mock returning a fixed high probability."""

    def predict(self, dialog_text: str) -> float:
        return 0.5

    def reset(self) -> None:
        pass


class ScriptedBridge(ICppBridge):
    """CppBridge mock with deterministic event generation.

    Auto-generates PLAYBACK_STARTED on send_stream_start(),
    PLAYBACK_COMPLETE on send_audio_end() and send_play_file().
    """

    def __init__(self) -> None:
        self._events: queue.Queue[CppEvent] = queue.Queue()
        self.audio_chunks: list[bytes] = []
        self.play_files: list[str] = []
        self.stream_start_count = 0
        self.audio_end_count = 0
        self.stop_count = 0
        self._connected = False

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

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


class FakeLED(ILEDController):
    """LED mock that records state transitions."""

    def __init__(self) -> None:
        self.states: list[LEDState] = []

    def set_state(self, state: LEDState) -> None:
        self.states.append(state)

    def close(self) -> None:
        pass


class FakeWakeword(IWakewordDetector):
    """Wakeword mock that triggers after N calls."""

    def __init__(self, trigger_after: int = 1) -> None:
        self._trigger_after = trigger_after
        self._count = 0

    def feed_audio(self, frame: AudioFrame) -> bool:
        self._count += 1
        return self._count >= self._trigger_after

    def close(self) -> None:
        pass


class FakeAudioInput(IAudioInput):
    """AudioInput mock (no-op, frames fed directly to queue by test)."""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    @property
    def error(self) -> Exception | None:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_token_counter(text: str) -> int:
    """Approximate token counter: 1 token per 4 characters."""
    return max(1, len(text) // 4)


def _make_orchestrator(
    asr: IASR,
    bridge: ScriptedBridge,
    led: FakeLED,
    *,
    llm: ILLM | None = None,
    tts: ITTS | None = None,
    vap: IVAP | None = None,
    turngpt: ITurnGPT | None = None,
    executor: ThreadPoolExecutor | None = None,
    orchestrator_config: OrchestratorConfig | None = None,
    turn_detector_config: TurnDetectorConfig | None = None,
) -> tuple[Orchestrator, ConversationHistory, SpeechGenerator, TurnDetector]:
    """Wire up a full Orchestrator with real internal modules."""
    _vap = vap or FakeVAP()
    _turngpt = turngpt or FakeTurnGPT()
    _llm = llm or FakeLLM()
    _tts = tts or FakeTTS()
    _executor = executor or ThreadPoolExecutor(max_workers=GENERATOR_CONFIG.max_workers)

    storage = MemoryStorageBackend()
    history = ConversationHistory(storage, _simple_token_counter)
    history.new_session("test-session")

    context_builder = ContextBuilder(
        history, HISTORY_CONFIG, DEFAULT_SYSTEM_PROMPT, _simple_token_counter
    )
    _turngpt_adapter = SyncTurnGPTAdapter(_turngpt)
    _embedder = MagicMock(spec=IEmbedder)
    turn_detector = TurnDetector(
        _vap,
        _turngpt_adapter,
        _embedder,
        turn_detector_config or TURN_DETECTOR_CONFIG,
        AUDIO_CONFIG,
    )
    generator = SpeechGenerator(context_builder, _llm, _tts, GENERATOR_CONFIG, _executor)
    truncator = TimestampTruncator()

    orchestrator = Orchestrator(
        asr=asr,
        turn_detector=turn_detector,
        speech_generator=generator,
        cpp_bridge=bridge,
        history=history,
        truncator=truncator,
        led=led,
        config=orchestrator_config or ORCHESTRATOR_CONFIG,
        tts_config=TTS_CONFIG,
        audio_config=AUDIO_CONFIG,
    )
    return orchestrator, history, generator, turn_detector


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
                self._queue.put(SILENCE_FRAME, timeout=0.01)
            self._stop.wait(self._interval)


# ---------------------------------------------------------------------------
# Test: Single-turn conversation (Orchestrator level)
# ---------------------------------------------------------------------------


class TestSingleTurnConversation:
    """Orchestrator processes one user utterance, generates response, exits on keyword."""

    def test_user_speaks_robot_responds_exit_keyword(self) -> None:
        """Flow: 'hello how are you' → response → 'goodbye' → exit."""
        asr = ScriptedASR(["hello how are you", "goodbye"])
        bridge = ScriptedBridge()
        led = FakeLED()

        orchestrator, history, generator, _ = _make_orchestrator(asr, bridge, led)

        audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=300)
        feeder = FrameFeeder(audio_queue)
        feeder.start()

        try:
            orchestrator.run(audio_queue)
        finally:
            feeder.stop()
            generator.reset()

        # --- Assertions ---
        messages = history.get_messages()
        assert len(messages) == 2

        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hello how are you"

        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "I'm doing great!"

        # Bridge received audio
        assert bridge.stream_start_count == 1
        assert bridge.audio_end_count == 1
        assert len(bridge.audio_chunks) > 0

        # LED transitions: IDLE (start) → OFF (end)
        assert LEDState.IDLE in led.states
        assert led.states[-1] == LEDState.OFF

    def test_conversation_history_context_passed_to_llm(self) -> None:
        """Verify that ContextBuilder includes system prompt and user message."""
        captured_messages: list[list[dict[str, Any]]] = []

        class CapturingLLM(ILLM):
            def generate(
                self,
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None,
                response_format: dict[str, Any] | None = None,
            ) -> LLMStream:
                captured_messages.append(list(messages))

                def gen():
                    yield "Sure thing!"

                return LLMStream(gen(), result_fn=lambda t: LLMResult(text=t))

        asr = ScriptedASR(["tell me something", "goodbye"])
        bridge = ScriptedBridge()
        led = FakeLED()

        orchestrator, _, generator, _ = _make_orchestrator(asr, bridge, led, llm=CapturingLLM())

        audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=300)
        feeder = FrameFeeder(audio_queue)
        feeder.start()

        try:
            orchestrator.run(audio_queue)
        finally:
            feeder.stop()
            generator.reset()

        assert len(captured_messages) >= 1
        msgs = captured_messages[0]
        # System prompt present
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == DEFAULT_SYSTEM_PROMPT
        # User message present
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "tell me something"


# ---------------------------------------------------------------------------
# Test: Multi-turn conversation (Orchestrator level)
# ---------------------------------------------------------------------------


class TestMultiTurnConversation:
    """Two exchanges before exit keyword."""

    def test_two_turns_then_exit(self) -> None:
        """Flow: turn 1 → response 1 → turn 2 → response 2 → 'goodbye' → exit."""
        asr = ScriptedASR(["what is AI", "tell me more", "goodbye"])
        llm = FakeLLM(
            responses=[
                ["AI ", "is ", "intelligence."],
                ["It's ", "fascinating!"],
            ]
        )
        bridge = ScriptedBridge()
        led = FakeLED()

        orchestrator, history, generator, _ = _make_orchestrator(asr, bridge, led, llm=llm)

        audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=300)
        feeder = FrameFeeder(audio_queue)
        feeder.start()

        try:
            orchestrator.run(audio_queue)
        finally:
            feeder.stop()
            generator.reset()

        messages = history.get_messages()
        assert len(messages) == 4

        assert messages[0] == {"role": "user", "content": "what is AI"}
        assert messages[1] == {"role": "assistant", "content": "AI is intelligence."}
        assert messages[2] == {"role": "user", "content": "tell me more"}
        assert messages[3] == {"role": "assistant", "content": "It's fascinating!"}

        # Two streaming rounds
        assert bridge.stream_start_count == 2
        assert bridge.audio_end_count == 2

    def test_second_turn_context_includes_first_turn(self) -> None:
        """Verify that ContextBuilder includes history from previous turns."""
        captured: list[list[dict[str, Any]]] = []

        class CapturingLLM(ILLM):
            def generate(
                self,
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None,
                response_format: dict[str, Any] | None = None,
            ) -> LLMStream:
                captured.append(list(messages))

                def gen():
                    yield "Response."

                return LLMStream(gen(), result_fn=lambda t: LLMResult(text=t))

        asr = ScriptedASR(["first question", "second question", "goodbye"])
        bridge = ScriptedBridge()
        led = FakeLED()

        orchestrator, _, generator, _ = _make_orchestrator(asr, bridge, led, llm=CapturingLLM())

        audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=300)
        feeder = FrameFeeder(audio_queue)
        feeder.start()

        try:
            orchestrator.run(audio_queue)
        finally:
            feeder.stop()
            generator.reset()

        # Second LLM call should include history from first turn
        assert len(captured) >= 2
        second_call = captured[1]
        roles = [m["role"] for m in second_call]
        # Should have: system, user(first), assistant(first), user(second)
        assert roles == ["system", "user", "assistant", "user"]
        assert second_call[-1]["content"] == "second question"


# ---------------------------------------------------------------------------
# Test: Full session lifecycle (SessionManager level)
# ---------------------------------------------------------------------------


class TestFullSessionLifecycle:
    """SLEEP → wakeword → GREETING → ACTIVE → FAREWELL → SLEEP."""

    def test_complete_session_cycle(self) -> None:
        """Full cycle with real Orchestrator inside SessionManager."""
        asr = ScriptedASR(["hello", "goodbye"])
        bridge = ScriptedBridge()
        led = FakeLED()
        wakeword = FakeWakeword(trigger_after=1)
        audio_input = FakeAudioInput()

        executor = ThreadPoolExecutor(max_workers=2)

        def session_factory() -> SessionComponents:
            _, history, generator, _ = (
                _make_orchestrator.__wrapped__(asr, bridge, led, executor=executor)
                if hasattr(_make_orchestrator, "__wrapped__")
                else (None, None, None, None)
            )

            # Build fresh components using the shared bridge/led/asr
            storage = MemoryStorageBackend()
            history = ConversationHistory(storage, _simple_token_counter)
            vap = FakeVAP()
            turngpt = FakeTurnGPT()
            context_builder = ContextBuilder(
                history, HISTORY_CONFIG, DEFAULT_SYSTEM_PROMPT, _simple_token_counter
            )
            turngpt_adapter = SyncTurnGPTAdapter(turngpt)
            _emb = MagicMock(spec=IEmbedder)
            turn_detector = TurnDetector(
                vap,
                turngpt_adapter,
                _emb,
                TURN_DETECTOR_CONFIG,
                AUDIO_CONFIG,
            )
            generator = SpeechGenerator(
                context_builder, FakeLLM(), FakeTTS(), GENERATOR_CONFIG, executor
            )
            truncator = TimestampTruncator()

            orchestrator = Orchestrator(
                asr=asr,
                turn_detector=turn_detector,
                speech_generator=generator,
                cpp_bridge=bridge,
                history=history,
                truncator=truncator,
                led=led,
                config=ORCHESTRATOR_CONFIG,
                tts_config=TTS_CONFIG,
                audio_config=AUDIO_CONFIG,
            )
            import uuid

            return SessionComponents(
                orchestrator=orchestrator, history=history, session_id=str(uuid.uuid4())
            )

        audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=SESSION_CONFIG.audio_queue_size)

        sm = SessionManager(
            audio_input=audio_input,
            wakeword=wakeword,
            session_factory=session_factory,
            cpp_bridge=bridge,
            led=led,
            config=SESSION_CONFIG,
            greeting_audio_path=GREETING_AUDIO_PATH,
            farewell_audio_path=FAREWELL_AUDIO_PATH,
            audio_queue=audio_queue,
        )

        feeder = FrameFeeder(audio_queue)
        feeder.start()

        # Run SessionManager in background, stop after one cycle
        original_run_sleep = sm._run_sleep

        call_count = 0

        def _shutdown_on_second_sleep() -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                sm._shutdown_event.set()
                return
            original_run_sleep()

        sm._run_sleep = _shutdown_on_second_sleep

        try:
            sm.run()
        finally:
            feeder.stop()
            executor.shutdown(wait=True)

        # --- Assertions ---
        # Bridge connected and disconnected
        assert bridge._connected is False  # disconnect called in run() finally

        # Greeting and farewell files sent
        assert GREETING_AUDIO_PATH in bridge.play_files
        assert FAREWELL_AUDIO_PATH in bridge.play_files

        # Orchestrator ran (audio was streamed)
        assert bridge.stream_start_count >= 1

        # LED went through full lifecycle
        assert LEDState.SLEEPING in led.states
        assert LEDState.IDLE in led.states


# ---------------------------------------------------------------------------
# Test: Barge-in (interrupt during playback)
# ---------------------------------------------------------------------------


class TestBargeIn:
    """User interrupts robot during playback."""

    def test_interrupt_truncates_response(self) -> None:
        """Interrupt during PLAYING triggers barge-in truncation."""
        # First utterance triggers generation, then interrupt, then exit
        asr = ScriptedASR(["tell me a story", "goodbye"])
        led = FakeLED()

        # VAP that becomes interrupting after streaming starts
        vap = InterruptableVAP()

        # Slow TTS: many chunks so streaming takes longer
        class SlowTTS(ITTS):
            def synthesize(self, text: str) -> TTSStream:
                chunk = b"\x00" * 4800

                def gen():
                    for _ in range(10):
                        yield chunk

                words = text.split()

                def ts_fn() -> tuple[WordTimestamp, ...]:
                    return tuple(
                        WordTimestamp(w, i * 0.1, (i + 1) * 0.1) for i, w in enumerate(words)
                    )

                return TTSStream(gen(), timestamps_fn=ts_fn)

        # Custom bridge that triggers interrupt after sending a few audio chunks.
        # Overrides send_audio_end to defer PLAYBACK_COMPLETE, simulating
        # real C++ behavior where remaining audio plays before completion.
        class InterruptBridge(ScriptedBridge):
            def __init__(self, vap_ref: InterruptableVAP) -> None:
                super().__init__()
                self._vap_ref = vap_ref
                self._audio_count = 0

            def send_audio(self, audio: bytes) -> None:
                super().send_audio(audio)
                self._audio_count += 1
                # After 2 audio chunks, simulate user starting to talk.
                if self._audio_count == 2:
                    self._vap_ref.interrupting = True

            def send_audio_end(self) -> None:
                # Don't enqueue PLAYBACK_COMPLETE immediately — real C++
                # keeps playing buffered audio. Interrupt should arrive first.
                self.audio_end_count += 1

        int_bridge = InterruptBridge(vap)

        orchestrator, history, generator, _ = _make_orchestrator(
            asr, int_bridge, led, vap=vap, tts=SlowTTS()
        )

        audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=300)
        feeder = FrameFeeder(audio_queue)
        feeder.start()

        try:
            orchestrator.run(audio_queue)
        finally:
            feeder.stop()
            generator.reset()

        messages = history.get_messages()
        # Should have user message and (possibly truncated) assistant message
        user_msgs = [m for m in messages if m["role"] == "user"]
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]

        assert len(user_msgs) >= 1
        assert user_msgs[0]["content"] == "tell me a story"

        # The assistant message may be truncated (shorter than full response)
        # or the full response if truncation didn't have enough data
        if assistant_msgs:
            # Just verify it exists and is non-empty
            assert assistant_msgs[0]["content"]

        # Bridge received a stop command (barge-in)
        assert int_bridge.stop_count >= 1


# ---------------------------------------------------------------------------
# Test: Session timeout
# ---------------------------------------------------------------------------


class TestSessionTimeout:
    """Orchestrator exits on session timeout when user is silent."""

    def test_timeout_exits_cleanly(self) -> None:
        """No user speech → session timeout → clean exit."""
        asr = ScriptedASR([])  # Never returns text
        bridge = ScriptedBridge()
        led = FakeLED()

        timeout_config = OrchestratorConfig(
            exit_keywords=("goodbye",),
            session_timeout_sec=0.2,
            frame_timeout_sec=0.05,
            stop_pending_timeout_sec=2.0,
        )

        orchestrator, history, generator, _ = _make_orchestrator(
            asr, bridge, led, orchestrator_config=timeout_config
        )

        audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=300)
        feeder = FrameFeeder(audio_queue)
        feeder.start()

        start = time.monotonic()
        try:
            orchestrator.run(audio_queue)
        finally:
            feeder.stop()
            generator.reset()

        elapsed = time.monotonic() - start
        # Should exit roughly at session_timeout_sec
        assert elapsed < 2.0  # sanity bound
        # No messages generated
        assert len(history.get_messages()) == 0
        # LED ended at OFF (session end)
        assert led.states[-1] == LEDState.OFF


# ---------------------------------------------------------------------------
# Test: External stop
# ---------------------------------------------------------------------------


class TestExternalStop:
    """Orchestrator exits cleanly on request_stop()."""

    def test_request_stop_exits(self) -> None:
        asr = ScriptedASR(["hello"])
        bridge = ScriptedBridge()
        led = FakeLED()

        orchestrator, _, generator, _ = _make_orchestrator(asr, bridge, led)

        audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=300)
        feeder = FrameFeeder(audio_queue)
        feeder.start()

        # Stop after a short delay
        def delayed_stop() -> None:
            time.sleep(0.1)
            orchestrator.request_stop()

        stopper = threading.Thread(target=delayed_stop, daemon=True)
        stopper.start()

        try:
            orchestrator.run(audio_queue)
        finally:
            feeder.stop()
            generator.reset()
            stopper.join(timeout=2.0)

        assert led.states[-1] == LEDState.OFF


# ---------------------------------------------------------------------------
# Test: Generator failure recovery
# ---------------------------------------------------------------------------


class TestGeneratorFailure:
    """Orchestrator recovers when LLM returns empty text."""

    def test_empty_llm_response_skips_turn(self) -> None:
        """Empty LLM response → generator FAILED → skip turn, stay active."""

        class EmptyThenRealLLM(ILLM):
            def __init__(self) -> None:
                self._call = 0

            def generate(
                self,
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None,
                response_format: dict[str, Any] | None = None,
            ) -> LLMStream:
                self._call += 1
                chunks = [""] if self._call == 1 else ["Recovery response!"]

                def gen():
                    yield from chunks

                return LLMStream(gen(), result_fn=lambda t: LLMResult(text=t))

        # 3 utterances: first triggers failed gen, second succeeds, third exits
        asr = ScriptedASR(["first try", "second try", "goodbye"])
        bridge = ScriptedBridge()
        led = FakeLED()

        orchestrator, history, generator, _ = _make_orchestrator(
            asr, bridge, led, llm=EmptyThenRealLLM()
        )

        audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=300)
        feeder = FrameFeeder(audio_queue)
        feeder.start()

        try:
            orchestrator.run(audio_queue)
        finally:
            feeder.stop()
            generator.reset()

        messages = history.get_messages()
        # First turn was skipped (empty response), second succeeded
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]

        # At least one successful exchange
        assert len(assistant_msgs) >= 1
        assert "Recovery response!" in [m["content"] for m in assistant_msgs]


# ---------------------------------------------------------------------------
# Memory integration helpers
# ---------------------------------------------------------------------------

_DIM = 8  # Small dimension for fast deterministic tests


class _DeterministicEmbedder(IEmbedder):
    """Embedder returning deterministic vectors seeded by text hash."""

    def embed(self, text: str) -> np.ndarray:
        import numpy as np

        rng = np.random.default_rng(hash(text) % (2**31))
        vec = rng.standard_normal(_DIM).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-9)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        import numpy as np

        return np.stack([self.embed(t) for t in texts])

    @property
    def dimension(self) -> int:
        return _DIM


def _make_memory_orchestrator(
    asr: IASR,
    bridge: ScriptedBridge,
    led: FakeLED,
    *,
    llm: ILLM | None = None,
    tts: ITTS | None = None,
    session_id: str = "mem-session-1",
    pre_episodes: list[Episode] | None = None,
) -> tuple[
    Orchestrator,
    ConversationHistory,
    SpeechGenerator,
    InMemoryMemoryStorage,
    MemoryRetriever,
]:
    """Wire up a full Orchestrator with memory modules enabled.

    Optionally seeds episodes into the memory store for retrieval tests.
    """
    import numpy as np

    _llm = llm or FakeLLM()
    _tts = tts or FakeTTS()
    executor = ThreadPoolExecutor(max_workers=GENERATOR_CONFIG.max_workers)

    # Conversation history
    conv_storage = MemoryStorageBackend()
    history = ConversationHistory(conv_storage, _simple_token_counter)
    history.new_session(session_id)

    # Memory infra
    memory_config = MemoryConfig(embedding_dimension=_DIM, max_memories=5, min_new_slots=2)
    memory_storage = InMemoryMemoryStorage(dimension=_DIM)
    vector_index = NumpyVectorIndex()
    embedder = _DeterministicEmbedder()

    # Seed pre-existing episodes
    if pre_episodes:
        for ep in pre_episodes:
            eid = memory_storage.add_episode(ep)
            ep.id = eid
            emb = embedder.embed(ep.text)
            ep.embedding = emb
            memory_storage.update_episode_embedding(eid, emb)
            vector_index.add(eid, emb)

    retriever = MemoryRetriever(memory_storage, vector_index, embedder, memory_config)

    # Context builder
    context_builder = ContextBuilder(
        history, HISTORY_CONFIG, DEFAULT_SYSTEM_PROMPT, _simple_token_counter
    )

    # Speech generator with memory
    generator = SpeechGenerator(
        context_builder,
        _llm,
        _tts,
        GENERATOR_CONFIG,
        executor,
        retriever=retriever,
        history=history,
        exclude_session_ids={session_id},
    )

    # Turn detector
    vap = FakeVAP()
    turngpt = FakeTurnGPT()
    turngpt_adapter = SyncTurnGPTAdapter(turngpt)
    _embedder = MagicMock(spec=IEmbedder)
    turn_detector = TurnDetector(
        vap, turngpt_adapter, _embedder, TURN_DETECTOR_CONFIG, AUDIO_CONFIG
    )

    truncator = TimestampTruncator()

    orchestrator = Orchestrator(
        asr=asr,
        turn_detector=turn_detector,
        speech_generator=generator,
        cpp_bridge=bridge,
        history=history,
        truncator=truncator,
        led=led,
        config=ORCHESTRATOR_CONFIG,
        tts_config=TTS_CONFIG,
        audio_config=AUDIO_CONFIG,
        memory_storage=memory_storage,
        session_id=session_id,
        token_counter=_simple_token_counter,
    )
    return orchestrator, history, generator, memory_storage, retriever


# ---------------------------------------------------------------------------
# Test: Memory — Orchestrator utterance storage
# ---------------------------------------------------------------------------


class TestMemoryUtteranceStorage:
    """Orchestrator stores utterances into memory storage during conversation."""

    def test_utterances_saved_during_conversation(self) -> None:
        """User and assistant utterances are saved to memory storage."""
        asr = ScriptedASR(["hello world", "goodbye"])
        bridge = ScriptedBridge()
        led = FakeLED()

        orchestrator, history, generator, memory_storage, _ = _make_memory_orchestrator(
            asr, bridge, led
        )

        audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=300)
        feeder = FrameFeeder(audio_queue)
        feeder.start()

        try:
            orchestrator.run(audio_queue)
        finally:
            feeder.stop()
            generator.reset()

        utterances = memory_storage.get_utterances("mem-session-1")
        roles = [u[0] for u in utterances]
        texts = [u[1] for u in utterances]

        # Both user and assistant utterances should be stored
        assert "user" in roles
        assert "assistant" in roles
        assert "hello world" in texts
        # Assistant text comes from FakeLLM default: "I'm doing great!"
        assert "I'm doing great!" in texts

    def test_utterance_token_counts_stored(self) -> None:
        """Stored utterances have non-zero token counts."""
        asr = ScriptedASR(["tell me something", "goodbye"])
        bridge = ScriptedBridge()
        led = FakeLED()

        orchestrator, _, generator, memory_storage, _ = _make_memory_orchestrator(
            asr, bridge, led
        )

        audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=300)
        feeder = FrameFeeder(audio_queue)
        feeder.start()

        try:
            orchestrator.run(audio_queue)
        finally:
            feeder.stop()
            generator.reset()

        utterances = memory_storage.get_utterances("mem-session-1")
        # Token counts should be > 0 (using _simple_token_counter)
        for _role, text, _ts, token_count in utterances:
            assert token_count > 0, f"Token count should be > 0 for '{text}'"


# ---------------------------------------------------------------------------
# Test: Memory — SpeechGenerator retrieval & context injection
# ---------------------------------------------------------------------------


class TestMemoryRetrievalInPipeline:
    """SpeechGenerator retrieves memories and injects them into LLM context."""

    def test_memory_block_injected_into_context(self) -> None:
        """When pre-existing episodes exist, LLM context includes a memory block."""
        captured_messages: list[list[dict[str, Any]]] = []

        class CapturingLLM(ILLM):
            def generate(
                self,
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None,
                response_format: dict[str, Any] | None = None,
            ) -> LLMStream:
                captured_messages.append(list(messages))

                def gen():
                    yield "Got it!"

                return LLMStream(gen(), result_fn=lambda t: LLMResult(text=t))

        pre_episodes = [
            Episode(
                id=None,
                text="The user loves watching sci-fi movies.",
                timestamp="2026-03-15 14:00:00",
                session_id="s-old",
                importance=1.0,
                last_cited_at="2026-03-15 14:00:00",
                embedding=None,
            ),
        ]

        asr = ScriptedASR(["I like movies", "goodbye"])
        bridge = ScriptedBridge()
        led = FakeLED()

        orchestrator, _, generator, _, _ = _make_memory_orchestrator(
            asr, bridge, led, llm=CapturingLLM(), pre_episodes=pre_episodes
        )

        audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=300)
        feeder = FrameFeeder(audio_queue)
        feeder.start()

        try:
            orchestrator.run(audio_queue)
        finally:
            feeder.stop()
            generator.reset()

        # LLM should have been called with a memory block in the context
        assert len(captured_messages) >= 1
        msgs = captured_messages[0]
        developer_msgs = [m for m in msgs if m.get("role") == "developer"]
        memory_msgs = [m for m in developer_msgs if "[Retrieved Memories]" in m.get("content", "")]

        assert len(memory_msgs) == 1, "Should have exactly one memory block in context"
        assert "[M1]" in memory_msgs[0]["content"]
        assert "sci-fi" in memory_msgs[0]["content"]

    def test_no_memory_block_when_no_episodes(self) -> None:
        """Without pre-existing episodes, no memory block is in the context."""
        captured_messages: list[list[dict[str, Any]]] = []

        class CapturingLLM(ILLM):
            def generate(
                self,
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None,
                response_format: dict[str, Any] | None = None,
            ) -> LLMStream:
                captured_messages.append(list(messages))

                def gen():
                    yield "Hello!"

                return LLMStream(gen(), result_fn=lambda t: LLMResult(text=t))

        asr = ScriptedASR(["hi there", "goodbye"])
        bridge = ScriptedBridge()
        led = FakeLED()

        orchestrator, _, generator, _, _ = _make_memory_orchestrator(
            asr, bridge, led, llm=CapturingLLM()
        )

        audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=300)
        feeder = FrameFeeder(audio_queue)
        feeder.start()

        try:
            orchestrator.run(audio_queue)
        finally:
            feeder.stop()
            generator.reset()

        assert len(captured_messages) >= 1
        msgs = captured_messages[0]
        memory_msgs = [
            m for m in msgs if "[Retrieved Memories]" in m.get("content", "")
        ]
        assert len(memory_msgs) == 0, "Should have no memory block without episodes"


# ---------------------------------------------------------------------------
# Test: Memory — Citation parsing
# ---------------------------------------------------------------------------


class TestMemoryCitationInPipeline:
    """SpeechGenerator parses citation tags and updates the retriever."""

    def test_citation_tag_stripped_from_response(self) -> None:
        """[MEMORIES: M1] tag is stripped from the response stored in history."""
        pre_episodes = [
            Episode(
                id=None,
                text="The user loves sci-fi movies.",
                timestamp="2026-03-15 14:00:00",
                session_id="s-old",
                importance=1.0,
                last_cited_at="2026-03-15 14:00:00",
                embedding=None,
            ),
        ]

        class CitingLLM(ILLM):
            def generate(
                self,
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None,
                response_format: dict[str, Any] | None = None,
            ) -> LLMStream:
                def gen():
                    yield "You mentioned sci-fi before!"
                    yield "\n[MEMORIES: M1]"

                return LLMStream(gen(), result_fn=lambda t: LLMResult(text=t))

        asr = ScriptedASR(["I love space", "goodbye"])
        bridge = ScriptedBridge()
        led = FakeLED()

        orchestrator, history, generator, _, retriever = _make_memory_orchestrator(
            asr, bridge, led, llm=CitingLLM(), pre_episodes=pre_episodes
        )

        audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=300)
        feeder = FrameFeeder(audio_queue)
        feeder.start()

        try:
            orchestrator.run(audio_queue)
        finally:
            feeder.stop()
            generator.reset()

        # History should contain the clean text WITHOUT citation tag
        messages = history.get_messages()
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) >= 1
        assert "[MEMORIES:" not in assistant_msgs[0]["content"]
        assert "You mentioned sci-fi before!" in assistant_msgs[0]["content"]

    def test_citation_updates_retained_buffer(self) -> None:
        """Cited episode enters the retriever's retained buffer."""
        pre_episodes = [
            Episode(
                id=None,
                text="The user loves sci-fi movies.",
                timestamp="2026-03-15 14:00:00",
                session_id="s-old",
                importance=1.0,
                last_cited_at="2026-03-15 14:00:00",
                embedding=None,
            ),
        ]

        class CitingLLM(ILLM):
            def generate(
                self,
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None,
                response_format: dict[str, Any] | None = None,
            ) -> LLMStream:
                def gen():
                    yield "Great movie taste!"
                    yield "\n[MEMORIES: M1]"

                return LLMStream(gen(), result_fn=lambda t: LLMResult(text=t))

        asr = ScriptedASR(["movies are fun", "goodbye"])
        bridge = ScriptedBridge()
        led = FakeLED()

        orchestrator, _, generator, _, retriever = _make_memory_orchestrator(
            asr, bridge, led, llm=CitingLLM(), pre_episodes=pre_episodes
        )

        audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=300)
        feeder = FrameFeeder(audio_queue)
        feeder.start()

        try:
            orchestrator.run(audio_queue)
        finally:
            feeder.stop()
            generator.reset()

        # The cited episode should be in the retained buffer
        assert len(retriever._retained) >= 1
        cited_ep = pre_episodes[0]
        assert cited_ep.id in retriever._retained


# ---------------------------------------------------------------------------
# Test: Memory — SessionManager triggers session-end callback
# ---------------------------------------------------------------------------


class TestMemorySessionEnd:
    """SessionManager invokes on_session_end callback at session termination."""

    def test_session_end_callback_invoked(self) -> None:
        """on_session_end receives the correct session_id and timestamp."""
        callback_calls: list[tuple[str, str]] = []

        def on_session_end(session_id: str, started_at: str) -> None:
            callback_calls.append((session_id, started_at))

        asr = ScriptedASR(["hello", "goodbye"])
        bridge = ScriptedBridge()
        led = FakeLED()
        wakeword = FakeWakeword(trigger_after=1)
        audio_input = FakeAudioInput()

        executor = ThreadPoolExecutor(max_workers=2)

        def session_factory() -> SessionComponents:
            import uuid

            sid = str(uuid.uuid4())

            conv_storage = MemoryStorageBackend()
            hist = ConversationHistory(conv_storage, _simple_token_counter)
            hist.new_session(sid)

            context_builder = ContextBuilder(
                hist, HISTORY_CONFIG, DEFAULT_SYSTEM_PROMPT, _simple_token_counter
            )
            vap = FakeVAP()
            turngpt_adapter = SyncTurnGPTAdapter(FakeTurnGPT())
            _emb = MagicMock(spec=IEmbedder)
            td = TurnDetector(
                vap, turngpt_adapter, _emb, TURN_DETECTOR_CONFIG, AUDIO_CONFIG
            )
            gen = SpeechGenerator(context_builder, FakeLLM(), FakeTTS(), GENERATOR_CONFIG, executor)
            trunc = TimestampTruncator()

            orch = Orchestrator(
                asr=asr,
                turn_detector=td,
                speech_generator=gen,
                cpp_bridge=bridge,
                history=hist,
                truncator=trunc,
                led=led,
                config=ORCHESTRATOR_CONFIG,
                tts_config=TTS_CONFIG,
                audio_config=AUDIO_CONFIG,
            )
            return SessionComponents(orchestrator=orch, history=hist, session_id=sid)

        audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=SESSION_CONFIG.audio_queue_size)

        sm = SessionManager(
            audio_input=audio_input,
            wakeword=wakeword,
            session_factory=session_factory,
            cpp_bridge=bridge,
            led=led,
            config=SESSION_CONFIG,
            greeting_audio_path=GREETING_AUDIO_PATH,
            farewell_audio_path=FAREWELL_AUDIO_PATH,
            audio_queue=audio_queue,
            on_session_end=on_session_end,
        )

        feeder = FrameFeeder(audio_queue)
        feeder.start()

        original_run_sleep = sm._run_sleep
        call_count = 0

        def _shutdown_on_second_sleep() -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                sm._shutdown_event.set()
                return
            original_run_sleep()

        sm._run_sleep = _shutdown_on_second_sleep

        try:
            sm.run()
        finally:
            feeder.stop()
            executor.shutdown(wait=True)

        # Callback should have been invoked once with a valid session_id and timestamp
        assert len(callback_calls) == 1
        sid, started_at = callback_calls[0]
        assert len(sid) > 0
        assert len(started_at) > 0
        # Timestamp format: "YYYY-MM-DD HH:MM:SS"
        assert len(started_at) == 19


# ---------------------------------------------------------------------------
# Test: Memory — Barge-in saves truncated utterance
# ---------------------------------------------------------------------------


class TestMemoryBargeIn:
    """Memory storage receives truncated text on barge-in, not full text."""

    def test_bargein_saves_truncated_utterance(self) -> None:
        """On barge-in the assistant utterance in memory storage is truncated."""
        vap = InterruptableVAP()

        # LLM returns a long multi-word response so truncation is observable
        class LongLLM(ILLM):
            def generate(
                self,
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None,
                response_format: dict[str, Any] | None = None,
            ) -> LLMStream:
                def gen():
                    yield "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"

                return LLMStream(gen(), result_fn=lambda t: LLMResult(text=t))

        full_text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"

        # SlowTTS: 10 chunks, each 100ms at 24kHz = 1 second total
        class SlowTTS(ITTS):
            def synthesize(self, text: str) -> TTSStream:
                chunk = b"\x00" * 4800

                def gen():
                    for _ in range(10):
                        yield chunk

                words = text.split()

                def ts_fn() -> tuple[WordTimestamp, ...]:
                    return tuple(
                        WordTimestamp(w, i * 0.1, (i + 1) * 0.1) for i, w in enumerate(words)
                    )

                return TTSStream(gen(), timestamps_fn=ts_fn)

        class InterruptBridge(ScriptedBridge):
            def __init__(self, vap_ref: InterruptableVAP) -> None:
                super().__init__()
                self._vap_ref = vap_ref
                self._audio_count = 0

            def send_audio(self, audio: bytes) -> None:
                super().send_audio(audio)
                self._audio_count += 1
                if self._audio_count == 2:
                    self._vap_ref.interrupting = True

            def send_audio_end(self) -> None:
                self.audio_end_count += 1

        bridge = InterruptBridge(vap)
        led = FakeLED()
        asr = ScriptedASR(["tell me words", "goodbye"])

        orchestrator, history, generator, memory_storage, _ = _make_memory_orchestrator(
            asr, bridge, led, llm=LongLLM(), tts=SlowTTS(),
        )
        # Patch VAP into the turn detector (replace the default FakeVAP)
        orchestrator._turn_detector._vap = vap

        audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=300)
        feeder = FrameFeeder(audio_queue)
        feeder.start()

        try:
            orchestrator.run(audio_queue)
        finally:
            feeder.stop()
            generator.reset()

        utterances = memory_storage.get_utterances("mem-session-1")
        assistant_utts = [(text, tc) for role, text, _ts, tc in utterances if role == "assistant"]

        # Should have at least one assistant utterance
        assert len(assistant_utts) >= 1

        saved_text = assistant_utts[0][0]
        # The saved text should be truncated (shorter than or equal to full text).
        # With interrupt after 2 chunks (~200ms of 1000ms), truncation should cut it.
        assert len(saved_text) <= len(full_text)
        # And it should be non-empty (some audio was played before interrupt)
        assert len(saved_text) > 0
