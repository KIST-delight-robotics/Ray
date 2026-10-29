"""컴포넌트 조립 — 프로덕션(``__main__``)과 실험 하네스(``scripts/``)가 같은 그래프를 쓴다.

프로세스 수준(모델·클라이언트·스토어, ``build_components``)과 세션 수준(``create_session``)을
한 곳에 모은다.

규율: 이 모듈과 파이프라인 모듈에는 **중립적 주입점**(경로·토글·콜백)만 둔다. 실험 전용 동작은
하네스 쪽에 있어야 하며, 의존 방향은 ``scripts → voice_pipeline`` 단방향.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from silero_vad import load_silero_vad

from voice_pipeline import trace
from voice_pipeline.adapters.audio_input import AudioInput
from voice_pipeline.adapters.cpp_bridge import CppBridge
from voice_pipeline.adapters.embedder import create_embedder
from voice_pipeline.adapters.gpt_live import GPTLiveSession, LiveSessionConfig
from voice_pipeline.adapters.led import LEDController
from voice_pipeline.adapters.llm_openai import OpenAILLM
from voice_pipeline.adapters.token_counter import TokenCounter, create_token_counter
from voice_pipeline.device_settings import DeviceSettings
from voice_pipeline.engines.gpt_live.instructions import (
    LIVE_VOICE,
    build_backend_instructions,
    build_live_instructions,
)
from voice_pipeline.engines.gpt_live.loop import LiveSessionLoop
from voice_pipeline.engines.gpt_live.radio import (
    Radio,
    RadioJudge,
    RadioResumeJudge,
    ScriptLog,
    ScriptPicker,
    load_scripts,
)
from voice_pipeline.engines.gpt_live.songs import Song, load_song_catalog
from voice_pipeline.engines.gpt_live.tools import (
    DEFAULT_TOOLS,
    LOOP_TOOLS,
    RADIO_TOOLS,
    SEARCH_MEMORY_TOOL,
    make_device_settings_handlers,
    make_memory_search_handler,
    make_play_song_tool_def,
)
from voice_pipeline.history import ConversationHistory, SQLiteStorageBackend
from voice_pipeline.memory.retriever import MemoryRetriever
from voice_pipeline.memory.storage import _DEFAULT_DIMENSION, SQLiteMemoryStorage
from voice_pipeline.memory.vector_index import NumpyVectorIndex
from voice_pipeline.session_context import format_user_notes_block, load_recent_conversations
from voice_pipeline.settings import (
    BRIDGE_SAMPLE_RATE,
    DEFAULT_DB_PATH,
    DEVICE_SETTINGS_PATH,
    RADIO_SCRIPTS_DIR,
    SONG_CATALOG_PATH,
    VAR_DIR,
)
from voice_pipeline.trace import OpenAIRetryHandler, SQLiteCallStore, TrackedEmbedder
from voice_pipeline.types import AudioFrame

logger = logging.getLogger("voice_pipeline.wiring")

_AUDIO_QUEUE_SIZE = 300
_MIC_DUMP_DIR = f"{VAR_DIR}/log/mic"  # MIC_DUMP=1 일 때 마이크 전 채널 녹음 위치 (6ch + ray.wav 분당 ~13.4 MB)
_EXECUTOR_WORKERS = 2  # 툴 핸들러·라디오 판정기를 돌리는 스레드 풀 크기
_LIVE_BACKEND_MODEL = "gpt-5.4-mini"  # GPT-Live responses 위임 백엔드
# none 은 짧은 첫 턴에서 툴 선택이 미끄러진다(end_conversation 오발). low 는 지연 차이 없이 툴 선택이 맞는다
_LIVE_BACKEND_REASONING = "low"


@dataclass
class SessionComponents:
    """Per-session objects created by :meth:`ProcessComponents.create_session`."""

    session_loop: LiveSessionLoop
    history: ConversationHistory
    session_id: str


@dataclass
class ProcessComponents:
    """Process-level singletons shared across sessions.

    :func:`build_components`로 생성하고, 세션마다 :meth:`create_session`을 호출한다.
    수명 관리(start/stop/close)는 엔트리포인트 책임 — 이 클래스는 조립만 담당한다.
    """

    silero_vad_model: Any  # 웨이크워드가 쓴다
    bridge: CppBridge
    led: LEDController
    storage: SQLiteStorageBackend
    executor: ThreadPoolExecutor
    token_counter: TokenCounter
    embedder: TrackedEmbedder
    memory_storage: SQLiteMemoryStorage
    retry_handler: OpenAIRetryHandler
    vector_index: NumpyVectorIndex
    audio_queue: queue.Queue[AudioFrame]
    audio_input: AudioInput
    shutdown_event: threading.Event
    device_settings: DeviceSettings | None = None  # 볼륨·밝기 (live 툴). None 이면 툴 미노출
    song_catalog: dict[str, Song] = field(default_factory=dict)  # 재생 가능한 노래 (live 툴). 비면 툴 미노출
    radio_picker: ScriptPicker | None = None  # 라디오 대본 선택 (live 툴). None 이면 라디오 툴 미노출
    radio_judge: RadioJudge | None = None  # 라디오 끼어들기 판정기. radio_picker 와 함께 있다
    radio_resume_judge: RadioResumeJudge | None = None  # 멈춘 라디오의 복귀 판정기. None 이면 혼자 돌아가지 않는다

    def create_session(self, *, memory_enabled: bool = True, **session_loop_kwargs: Any) -> SessionComponents:
        """Assemble a fresh per-session component graph.

        Args:
            memory_enabled: False면 memory storage/retriever 없이 조립.
            **session_loop_kwargs: 세션 루프 생성자로 그대로 전달되는 선택 인자
                (``tool_handlers``, ``wait_for_playback_complete`` 등).

        Returns:
            세션 루프와 세션 단위 컴포넌트를 담은 :class:`SessionComponents`.
        """
        session_id = str(uuid.uuid4())
        trace.set_session(session_id)  # 이후 호출/턴 기록에 이 세션 ID가 찍힌다
        history = ConversationHistory(self.storage, self.token_counter)
        memory_storage = self.memory_storage if memory_enabled else None

        # 장기기억: 사용자 노트·최근 대화 원문은 시작 instructions 에(세션 중 재조립 불가), 에피소드 검색은
        # 백엔드 함수 툴로 — 트리거와 쿼리 생성을 백엔드 LLM 이 맡는다.
        now = datetime.now().astimezone()  # 모델에게 보이는 시각의 기준 — 시간대는 기기 시스템 설정
        tool_handlers: dict[str, Any] = dict(session_loop_kwargs.pop("tool_handlers", None) or {})
        if self.device_settings is not None:
            for name, handler in make_device_settings_handlers(self.device_settings).items():
                tool_handlers.setdefault(name, handler)
        if memory_enabled:
            notes = self.memory_storage.get_user_notes()
            recent = load_recent_conversations(self.memory_storage, session_id, self.token_counter, now=now)
            retriever = MemoryRetriever(self.memory_storage, self.vector_index, self.embedder)
            tool_handlers.setdefault(SEARCH_MEMORY_TOOL, make_memory_search_handler(retriever, now))
            instructions = build_live_instructions(format_user_notes_block(notes), recent, now=now)
        else:
            instructions = build_live_instructions(now=now)
        # 핸들러가 있는 툴만 백엔드에 노출한다 (end_conversation·노래 툴은 루프 내장)
        tools = tuple(t for t in DEFAULT_TOOLS if t["name"] in LOOP_TOOLS or t["name"] in tool_handlers)
        if (play_song := make_play_song_tool_def(self.song_catalog)) is not None:
            tools += (play_song,)
        radio = None
        if self.radio_picker is not None and self.radio_judge is not None:
            tools += RADIO_TOOLS
            self.radio_picker.new_session()
            radio = Radio(self.radio_picker, self.radio_judge, self.executor, self.radio_resume_judge)
        live = GPTLiveSession(
            LiveSessionConfig(
                instructions=instructions,
                backend_model=_LIVE_BACKEND_MODEL,
                backend_reasoning=_LIVE_BACKEND_REASONING,
                backend_instructions=build_backend_instructions(
                    self.song_catalog, radio.status if radio is not None else None
                ),
                tools=tools,
                voice=LIVE_VOICE,
                sample_rate=BRIDGE_SAMPLE_RATE,
            )
        )
        live_loop = LiveSessionLoop(
            live=live,
            cpp_bridge=self.bridge,
            history=history,
            led=self.led,
            audio_queue=self.audio_queue,
            memory_storage=memory_storage,
            session_id=session_id,
            token_counter=self.token_counter,
            shutdown_event=self.shutdown_event,
            tool_handlers=tool_handlers,
            executor=self.executor,
            song_catalog=self.song_catalog,
            radio=radio,
            **session_loop_kwargs,
        )
        return SessionComponents(session_loop=live_loop, history=history, session_id=session_id)


def build_components(
    *,
    db_path: str = DEFAULT_DB_PATH,
    device_settings_path: str = DEVICE_SETTINGS_PATH,
    led_enabled: bool | None = None,
    song_catalog_path: str = SONG_CATALOG_PATH,
    radio_scripts_dir: str = RADIO_SCRIPTS_DIR,
) -> ProcessComponents:
    """Build the process-level component graph shared by all sessions.

    Args:
        db_path: history/memory/trace/call 스토어가 공유하는 SQLite 경로.
            실험 하네스는 런별 격리 DB 경로를 전달한다.
        device_settings_path: 볼륨·밝기 단계를 저장하는 JSON 경로. 시작 시 읽어 하드웨어에 적용한다.
        led_enabled: LED 하드웨어 구동 여부. ``None``이면 ``LED_ENABLED`` env로
            결정 (프로덕션 기본). 실험 하네스는 ``False``를 전달한다.
        song_catalog_path: 노래 카탈로그(``songs.json``) 경로. 없거나 비면 노래 툴을 노출하지 않는다.
        radio_scripts_dir: 라디오 대본 폴더. 없거나 비면(또는 곡이 하나도 재생 불가면) 라디오 툴을 노출하지 않는다.

    Returns:
        조립된 :class:`ProcessComponents`. ``audio_input.start()``/``bridge.connect()``
        등 수명 시작은 호출자가 수행한다.
    """
    bridge = CppBridge()
    silero_vad_model = load_silero_vad(onnx=True)

    if led_enabled is None:
        # cpp/config.toml is C++-only, so the Python side reads LED_ENABLED directly.
        led_enabled = os.environ.get("LED_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")
    led = LEDController(enabled=led_enabled)
    device_settings = DeviceSettings(device_settings_path, led=led)
    device_settings.apply()  # 저장된 볼륨·밝기 복원. 볼륨 실패(PipeWire 미기동 등)는 경고만

    storage = SQLiteStorageBackend(db_path)
    executor = ThreadPoolExecutor(max_workers=_EXECUTOR_WORKERS)
    token_counter = create_token_counter(_LIVE_BACKEND_MODEL)

    memory_storage = SQLiteMemoryStorage(db_path)
    trace.install(SQLiteCallStore(db_path))
    retry_handler = OpenAIRetryHandler()
    logging.getLogger("openai._base_client").addHandler(retry_handler)
    # local_files_only: 부팅 시 HF 허브 왕복 생략 + 네트워크 미준비 상태에서도 기동.
    # 새 기기 첫 실행은 캐시가 없어 실패한다.
    embedder = TrackedEmbedder(create_embedder(expected_dimension=_DEFAULT_DIMENSION, local_files_only=True))

    vector_index = NumpyVectorIndex()
    ids, vectors = memory_storage.load_all_embeddings()
    if ids:
        vector_index.load(ids, vectors)

    audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=_AUDIO_QUEUE_SIZE)
    mic_dump = os.environ.get("MIC_DUMP", "0").strip().lower() in ("1", "true", "yes", "on")
    audio_input = AudioInput(audio_queue, dump_dir=_MIC_DUMP_DIR if mic_dump else None)
    song_catalog = load_song_catalog(song_catalog_path)
    scripts = load_scripts(radio_scripts_dir, song_catalog) if song_catalog else {}
    radio_picker = ScriptPicker(scripts, ScriptLog(db_path)) if scripts else None
    # 끼어든 말 판정 — 응답 지연이 판정 시점을 정하므로 추론 없이, 구조화 출력으로 짧게
    radio_judge = (
        RadioJudge(OpenAILLM(model="gpt-5.4-mini", temperature=0.0, reasoning_effort="none", max_tokens=120, tools=[]))
        if radio_picker is not None
        else None
    )
    # 멈춘 대화가 잠잠할 때 미리 부르므로 지연보다 판단이 중요하다 — 추론 low (추론 토큰도 출력 한도에 든다)
    radio_resume_judge = (
        RadioResumeJudge(OpenAILLM(model="gpt-5.4-mini", reasoning_effort="low", max_tokens=2000, tools=[]))
        if radio_picker is not None
        else None
    )

    return ProcessComponents(
        silero_vad_model=silero_vad_model,
        bridge=bridge,
        led=led,
        storage=storage,
        executor=executor,
        token_counter=token_counter,
        embedder=embedder,
        memory_storage=memory_storage,
        retry_handler=retry_handler,
        vector_index=vector_index,
        audio_queue=audio_queue,
        audio_input=audio_input,
        shutdown_event=threading.Event(),
        device_settings=device_settings,
        song_catalog=song_catalog,
        radio_picker=radio_picker,
        radio_judge=radio_judge,
        radio_resume_judge=radio_resume_judge,
    )
