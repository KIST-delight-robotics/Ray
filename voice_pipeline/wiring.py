"""Shared component wiring for pipeline entry points.

프로덕션(``__main__``)과 eval(``evaluation.run``)이 같은 컴포넌트 그래프를 쓰도록
프로세스 수준 조립(모델·클라이언트·스토어)과 세션 수준 조립을 한 곳에 모은다.

규율: 이 모듈과 파이프라인 모듈에는 **중립적 주입점**(경로·토글·콜백 전달)만 둔다.
eval 전용 동작(측정·채점 로직)은 evaluation 패키지에 있어야 하며, 여기에 eval을
아는 분기를 넣지 않는다. 의존 방향은 ``evaluation → voice_pipeline`` 단방향.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import torch
from silero_vad import load_silero_vad

from voice_pipeline.asr.asr import GoogleCloudASR
from voice_pipeline.audio.audio_input import AudioInput
from voice_pipeline.audio.constants import SAMPLE_RATE
from voice_pipeline.bridge.cpp_bridge import CppBridge
from voice_pipeline.core.interfaces import ITTS, IStorageBackend
from voice_pipeline.core.types import AudioFrame
from voice_pipeline.embedding.embedder import create_embedder
from voice_pipeline.generation.speech_generator import SpeechGenerator
from voice_pipeline.history.conversation_history import ConversationHistory
from voice_pipeline.history.storage_backend import create_storage_backend
from voice_pipeline.led.led_controller import LEDController
from voice_pipeline.llm.llm import OpenAILLM
from voice_pipeline.llm.prompts import DEFAULT_SYSTEM_PROMPT
from voice_pipeline.llm.token_counter import TokenCounter, create_token_counter
from voice_pipeline.llm.tools import get_tools_token_cost
from voice_pipeline.memory.retriever import MemoryRetriever
from voice_pipeline.memory.storage import _DEFAULT_DB_PATH, _DEFAULT_DIMENSION, SQLiteMemoryStorage
from voice_pipeline.memory.vector_index import NumpyVectorIndex
from voice_pipeline.session_loop import SessionComponents, SessionLoop
from voice_pipeline.trace.openai_retry_handler import OpenAIRetryHandler
from voice_pipeline.trace.trace_store import SQLiteCallStore, SQLiteTraceStore
from voice_pipeline.trace.tracked_embedder import TrackedEmbedder
from voice_pipeline.trace.tracked_tts import TrackedTTS
from voice_pipeline.tts.factory import create_tts
from voice_pipeline.turn_taking.async_turngpt import AsyncTurnGPT
from voice_pipeline.turn_taking.async_vap import AsyncVAP
from voice_pipeline.turn_taking.maai_vap import MaAIVAPWrapper
from voice_pipeline.turn_taking.turn_detector import TurnDetector
from voice_pipeline.turn_taking.turngpt import TurnGPTWrapper

logger = logging.getLogger("voice_pipeline.wiring")

_AUDIO_QUEUE_SIZE = 300
_VAD_INFER_INTERVAL = 3  # 3프레임(90ms)마다 추론, 사이는 캐시 반환
_SILERO_CHUNK_BYTES = 512 * 2  # 512 samples × 16-bit


@dataclass
class ProcessComponents:
    """Process-level singletons shared across sessions.

    :func:`build_components`로 생성하고, 세션마다 :meth:`create_session`을 호출한다.
    수명 관리(start/stop/close)는 엔트리포인트 책임 — 이 클래스는 조립만 담당한다.
    """

    language_code: str
    asr: GoogleCloudASR
    llm: OpenAILLM
    raw_tts: ITTS
    tts: TrackedTTS
    vap: MaAIVAPWrapper
    turngpt: TurnGPTWrapper
    silero_vad_model: Any
    vad_fn: Callable[[AudioFrame], float]
    reset_vad: Callable[[], None]
    bridge: CppBridge
    led: LEDController
    storage: IStorageBackend
    executor: ThreadPoolExecutor
    token_counter: TokenCounter
    tools_token_cost: int
    embedder: TrackedEmbedder
    memory_storage: SQLiteMemoryStorage
    trace_store: SQLiteTraceStore
    call_store: SQLiteCallStore
    retry_handler: OpenAIRetryHandler
    vector_index: NumpyVectorIndex
    audio_queue: queue.Queue[AudioFrame]
    audio_input: AudioInput
    shutdown_event: threading.Event
    _prev_async: list[AsyncVAP | AsyncTurnGPT] = field(default_factory=list)

    def stop_async(self) -> None:
        """Stop the previous session's async model wrapper threads."""
        for wrapper in self._prev_async:
            wrapper.stop()
        self._prev_async.clear()

    def create_session(self, *, memory_enabled: bool = True, **session_loop_kwargs: Any) -> SessionComponents:
        """Assemble a fresh per-session component graph.

        Args:
            memory_enabled: False면 memory storage/retriever 없이 조립.
            **session_loop_kwargs: :class:`SessionLoop` 생성자로 그대로 전달되는
                선택 인자 (``on_turn_shift`` 등 콜백, ``disable_exit_keywords``,
                ``skip_generation``, ``record_path`` 등).

        Returns:
            SessionLoop과 세션 단위 컴포넌트를 담은 :class:`SessionComponents`.
        """
        self.stop_async()

        self.vap.reset()
        self.turngpt.reset()
        self.reset_vad()

        session_id = str(uuid.uuid4())
        self.embedder.session_id = session_id
        self.tts.session_id = session_id
        self.retry_handler.session_id = session_id

        async_vap = AsyncVAP(self.vap, call_store=self.call_store, session_id=session_id)
        async_turngpt = AsyncTurnGPT(self.turngpt, call_store=self.call_store, session_id=session_id)
        self._prev_async.extend([async_vap, async_turngpt])

        history = ConversationHistory(self.storage, self.token_counter)
        memory_storage = self.memory_storage if memory_enabled else None
        retriever = MemoryRetriever(self.memory_storage, self.vector_index, self.embedder) if memory_enabled else None
        turn_detector = TurnDetector(
            async_vap,
            async_turngpt,
            self.embedder,
            vad_fn=self.vad_fn,
            vad_reset_fn=self.reset_vad,
            call_store=self.call_store,
            session_id=session_id,
        )
        generator = SpeechGenerator(
            self.llm,
            self.tts,
            history,
            self.token_counter,
            DEFAULT_SYSTEM_PROMPT,
            self.executor,
            tools_token_cost=self.tools_token_cost,
            memory_storage=memory_storage,
            retriever=retriever,
            session_id=session_id,
        )
        session_loop = SessionLoop(
            asr=self.asr,
            turn_detector=turn_detector,
            speech_generator=generator,
            cpp_bridge=self.bridge,
            history=history,
            led=self.led,
            audio_queue=self.audio_queue,
            tts_sample_rate=self.tts.output_sample_rate,
            memory_storage=memory_storage,
            session_id=session_id,
            token_counter=self.token_counter,
            trace_store=self.trace_store,
            shutdown_event=self.shutdown_event,
            **session_loop_kwargs,
        )
        return SessionComponents(session_loop=session_loop, history=history, session_id=session_id)


def build_components(
    *,
    db_path: str = _DEFAULT_DB_PATH,
    led_enabled: bool | None = None,
    language_code: str = "en-US",
) -> ProcessComponents:
    """Build the process-level component graph shared by all sessions.

    Args:
        db_path: history/memory/trace/call 스토어가 공유하는 SQLite 경로.
            eval은 런별 격리 DB 경로를 전달한다.
        led_enabled: LED 하드웨어 구동 여부. ``None``이면 ``LED_ENABLED`` env로
            결정 (프로덕션 기본). eval은 ``False``를 전달한다.
        language_code: ASR 언어 코드.

    Returns:
        조립된 :class:`ProcessComponents`. ``audio_input.start()``/``bridge.connect()``
        등 수명 시작은 호출자가 수행한다.
    """
    asr = GoogleCloudASR(language_code=language_code)
    llm = OpenAILLM(model="gpt-5.4", temperature=0.7, reasoning_effort="none", max_tokens=256, tools=["web_search"])
    raw_tts = create_tts()
    vap = MaAIVAPWrapper(raw_tts.output_sample_rate)
    turngpt = TurnGPTWrapper()
    bridge = CppBridge()

    silero_vad_model = load_silero_vad(onnx=True)
    _vad_buf = bytearray()
    _vad_last_score = [0.0]
    _vad_call_count = [0]

    def vad_fn(frame: AudioFrame) -> float:
        _vad_buf.extend(frame)
        _vad_call_count[0] += 1
        if _vad_call_count[0] % _VAD_INFER_INTERVAL != 0:
            return _vad_last_score[0]
        while len(_vad_buf) >= _SILERO_CHUNK_BYTES:
            chunk = bytes(_vad_buf[:_SILERO_CHUNK_BYTES])
            del _vad_buf[:_SILERO_CHUNK_BYTES]
            samples = torch.frombuffer(bytearray(chunk), dtype=torch.int16).float() / 32768.0
            _vad_last_score[0] = silero_vad_model(samples, SAMPLE_RATE).item()
        return _vad_last_score[0]

    def reset_vad() -> None:
        # Silero LSTM 상태는 이전 오디오 이력을 유지하며 자연 회복되지 않음 —
        # 큰 발화 이력이 남으면 조용한 음성을 통째로 놓침. 세션 시작마다 초기화.
        silero_vad_model.reset_states()
        _vad_buf.clear()
        _vad_last_score[0] = 0.0
        _vad_call_count[0] = 0

    if led_enabled is None:
        # cpp/config.toml is C++-only, so the Python side reads LED_ENABLED directly.
        led_enabled = os.environ.get("LED_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")
    led = LEDController(enabled=led_enabled)

    storage = create_storage_backend("sqlite", db_path=db_path)
    executor = ThreadPoolExecutor(max_workers=SpeechGenerator.MAX_WORKERS)
    token_counter = create_token_counter(llm.model)
    tools_token_cost = get_tools_token_cost(llm.tools)

    memory_storage = SQLiteMemoryStorage(db_path)
    trace_store = SQLiteTraceStore(db_path)
    call_store = SQLiteCallStore(db_path)
    retry_handler = OpenAIRetryHandler(call_store)
    logging.getLogger("openai._base_client").addHandler(retry_handler)
    tts = TrackedTTS(raw_tts, call_store)
    embedder = TrackedEmbedder(create_embedder(expected_dimension=_DEFAULT_DIMENSION), call_store)

    vector_index = NumpyVectorIndex()
    ids, vectors = memory_storage.load_all_embeddings()
    if ids:
        vector_index.load(ids, vectors)

    audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=_AUDIO_QUEUE_SIZE)
    audio_input = AudioInput(audio_queue)

    return ProcessComponents(
        language_code=language_code,
        asr=asr,
        llm=llm,
        raw_tts=raw_tts,
        tts=tts,
        vap=vap,
        turngpt=turngpt,
        silero_vad_model=silero_vad_model,
        vad_fn=vad_fn,
        reset_vad=reset_vad,
        bridge=bridge,
        led=led,
        storage=storage,
        executor=executor,
        token_counter=token_counter,
        tools_token_cost=tools_token_cost,
        embedder=embedder,
        memory_storage=memory_storage,
        trace_store=trace_store,
        call_store=call_store,
        retry_handler=retry_handler,
        vector_index=vector_index,
        audio_queue=audio_queue,
        audio_input=audio_input,
        shutdown_event=threading.Event(),
    )
