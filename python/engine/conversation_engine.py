"""대화 상태 머신 엔진. 마이크/VAD/STT/LLM/TTS 파이프라인을 조율합니다."""

import math
import queue
import logging
import asyncio
import threading
from collections import deque

from audio import VADProcessor, SmartTurnProcessor, GoogleSTTStreamer, MicrophoneStream, find_input_device
from engine.session import ConversationManager
from llm import LLMManager
from tts import TTSManager
from engine.states import SleepState

from config import (
    SMART_TURN_MODEL_PATH,
    START_KEYWORD, END_KEYWORDS,
    ACTIVE_SESSION_TIMEOUT,
    AUDIO_CONFIG,
)

logger = logging.getLogger(__name__)


class ConversationEngine:
    def __init__(self, websocket, main_loop):
        logger.info("초기화 시작")
        self.websocket = websocket
        self.main_loop = main_loop

        # 1. 설정 로드
        self.sample_rate = AUDIO_CONFIG['SAMPLE_RATE']
        self.chunk_size = AUDIO_CONFIG['VAD_CHUNK_SIZE']

        self.start_keyword = START_KEYWORD
        self.end_keywords = END_KEYWORDS
        self.active_timeout = ACTIVE_SESSION_TIMEOUT

        # 2. 데이터 큐
        self.mic_queue = queue.Queue()
        self.stt_audio_queue = queue.Queue()
        self.stt_result_queue = queue.Queue()

        # 3. Pre-buffer (발화 감지 전 ~0.5초 오디오 보관)
        pre_buffer_len = math.ceil(self.sample_rate * 0.5 / self.chunk_size)
        self.stt_pre_buffer = deque(maxlen=pre_buffer_len)

        # 4. 오디오 프로세서
        self.vad_processor = VADProcessor(
            sample_rate=self.sample_rate,
            chunk_size=self.chunk_size,
            threshold=AUDIO_CONFIG['VAD_THRESHOLD'],
            consecutive_chunks=AUDIO_CONFIG['VAD_CONSECUTIVE_CHUNKS'],
            reset_interval=AUDIO_CONFIG['VAD_RESET_INTERVAL'],
        )
        self.smart_turn_processor = SmartTurnProcessor(SMART_TURN_MODEL_PATH)

        # 5. 매니저
        self.history_manager = ConversationManager()
        self.llm_manager = LLMManager(conversation_manager=self.history_manager)
        self.tts_manager = TTSManager(main_loop=self.main_loop, websocket=self.websocket)

        # 6. STT 스트리머
        self.stt_stop_event = threading.Event()
        self.stt_streamer = GoogleSTTStreamer(
            stt_result_queue=self.stt_result_queue,
            main_loop=self.main_loop,
            websocket=self.websocket,
            sample_rate=self.sample_rate,
            stt_audio_queue=self.stt_audio_queue,
            stt_stop_event=self.stt_stop_event,
        )

        # 7. 마이크
        self.mic_stream = MicrophoneStream(
            mic_audio_queue=self.mic_queue,
            sample_rate=self.sample_rate,
            chunk_size=self.chunk_size,
            channels=AUDIO_CONFIG['CHANNELS'],
            dtype=AUDIO_CONFIG['AUDIO_DTYPE'],
            device_idx=find_input_device(),
        )

        # 8. Turn ID 관리
        self._turn_id_counter = 0
        self._turn_id_lock = threading.Lock()

        # 9. 상태 머신
        self._current_state = SleepState(self)
        self._is_running = False
        self.robot_finished_speaking = False
        self._robot_finished_turn_id = 0

    # ------------------------------------------------------------------
    # Facade methods (states에서 호출)
    # ------------------------------------------------------------------

    def next_turn_id(self) -> int:
        """새로운 turn_id를 생성하여 반환합니다."""
        with self._turn_id_lock:
            self._turn_id_counter += 1
            return self._turn_id_counter

    def send_to_robot(self, message: str):
        """WebSocket으로 메시지 전송 (thread-safe)."""
        if self.websocket:
            asyncio.run_coroutine_threadsafe(
                self.websocket.send(message), self.main_loop
            )

    def clear_stt_audio_queue(self):
        """STT 오디오 큐의 모든 항목을 비웁니다."""
        while True:
            try:
                self.stt_audio_queue.get_nowait()
            except queue.Empty:
                break

    def clear_stt_result_queue(self):
        """STT 결과 큐의 모든 항목을 비웁니다."""
        while True:
            try:
                self.stt_result_queue.get_nowait()
            except queue.Empty:
                break

    def start_stt_session(self) -> threading.Thread:
        """STT 세션을 시작하고 스레드를 반환합니다."""
        self.stt_stop_event.clear()
        thread = threading.Thread(
            target=self.stt_streamer.run_stt_session,
            name="STTSessionThread",
            daemon=True,
        )
        thread.start()
        return thread

    def flush_pre_buffer_to(self, stt_queue: queue.Queue, audio_buffer: list):
        """Pre-buffer의 내용을 STT 큐와 오디오 버퍼로 이동시킵니다."""
        if self.stt_pre_buffer:
            for chunk in self.stt_pre_buffer:
                stt_queue.put(chunk)
                audio_buffer.append(chunk)
            self.stt_pre_buffer.clear()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_robot_finished(self, turn_id: int = 0):
        """C++로부터 말하기 종료 신호 수신."""
        logger.info(f"Robot finished speaking signal received (turn_id={turn_id})")
        self._robot_finished_turn_id = turn_id
        self.robot_finished_speaking = True

    async def start(self):
        logger.info("ConversationEngine 시작")
        self._is_running = True

        self.mic_stream.start()
        self._current_state.on_enter()

        try:
            await self._loop()
        except asyncio.CancelledError:
            logger.info("엔진 작업 취소됨")
        except KeyboardInterrupt:
            logger.info("키보드 인터럽트 감지")
        finally:
            self.stop()

    def stop(self):
        """엔진 종료 및 리소스 정리."""
        logger.info("ConversationEngine 종료 중...")
        self._is_running = False

        if self.mic_stream:
            self.mic_stream.stop()

        self.stt_stop_event.set()
        self.stt_audio_queue.put(None)
        self.llm_manager.cancel()
        self.tts_manager.stop()

        logger.info("종료 완료")

    async def _loop(self):
        """메인 오디오 처리 루프."""
        while self._is_running:
            await asyncio.sleep(0.01)

            try:
                chunk = self.mic_queue.get_nowait()
            except queue.Empty:
                chunk = None

            next_state = self._current_state.update(chunk)

            if next_state:
                self._transition(next_state)

    def _transition(self, new_state):
        prev_name = self._current_state.__class__.__name__
        next_name = new_state.__class__.__name__
        logger.info(f"상태 전이: {prev_name} -> {next_name}")

        self._current_state.on_exit()
        self._current_state = new_state
        self._current_state.on_enter()
