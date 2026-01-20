import time
import json
import queue
import logging
import threading
import numpy as np
from abc import ABC, abstractmethod

from led import led_set_ring
from config import (
    SMART_TURN_MODEL_PATH,
    TURN_END_SILENCE_CHUNKS,
    MAX_TURN_CHUNKS,
    SMART_TURN_GRACE_PERIOD,
    SMART_TURN_MAX_RETRIES,
    STT_WAIT_TIMEOUT_SECONDS,
)

# 기존 모듈 임포트 (가정)
# from audio_processor import VADProcessor, SmartTurnProcessor, GoogleSTTStreamer, ...

# ==================================================================================
# 0. 매니저 클래스 (LLM/TTS 스레드 관리용 - 새로 추가 필요)
# ==================================================================================
class LLMManager:
    """LLM 요청을 비동기로 처리"""
    def __init__(self):
        self.response_queue = queue.Queue()
        self._stop_event = threading.Event()

    def request_generation(self, text):
        self._stop_event.clear()
        threading.Thread(target=self._run_llm, args=(text,), daemon=True).start()

    def cancel(self):
        self._stop_event.set()

    def _run_llm(self, text):
        # 실제 LLM API 호출 로직
        # 스트리밍이라면 청크 단위로 put, 여기서는 텍스트 하나라고 가정
        time.sleep(1)  # (Mock) 생성 시간
        if not self._stop_event.is_set():
            self.response_queue.put(f"LLM 응답에 대한 처리: {text}")

class TTSManager:
    """TTS 생성 및 재생 관리"""
    def __init__(self):
        self.is_playing = False
        self.playback_started_event = threading.Event()

    def speak(self, text):
        self.is_playing = True
        self.playback_started_event.clear()
        threading.Thread(target=self._run_tts, args=(text,), daemon=True).start()

    def stop(self):
        self.is_playing = False
        # (Mock) 오디오 장치 중단 로직

    def _run_tts(self, text):
        # 1. TTS 생성 (Latency)
        time.sleep(0.5) 
        # 2. C++ 전송 및 재생 시작 신호 수신
        self.playback_started_event.set() # "이제 소리 납니다"
        
        # 3. 재생 중...
        for _ in range(50): # (Mock) 재생 루프
            if not self.is_playing: break
            time.sleep(0.1)
        
        self.is_playing = False

# ==================================================================================
# 1. State Interface
# ==================================================================================
class ConversationState(ABC):
    def __init__(self, engine):
        self.engine = engine

    @abstractmethod
    def on_enter(self):
        """상태 진입 시 1회 실행"""
        pass

    @abstractmethod
    def update(self, chunk: np.ndarray) -> 'ConversationState | None':
        """
        메인 루프에서 주기적으로 호출됨.
        - chunk: 마이크 입력 (VAD 분석용)
        - return: 상태 전이가 필요하면 State 객체 반환, 아니면 None
        """
        pass

    @abstractmethod
    def on_exit(self):
        """상태 탈출 시 1회 실행"""
        pass

# ==================================================================================
# 2. State Implementations
# ==================================================================================

class IdleState(ConversationState):
    def on_enter(self):
        logging.info("STATE: [Idle] 대기 시작")
        # LED: 노란색
        led_set_ring(233, 233, 50)

    def update(self, chunk):
        # 단순 VAD 감지
        self.engine.stt_pre_buffer.append(chunk)
        if self.engine.vad.process(chunk):
            logging.info("🗣️ 발화 시작 감지")
            return ListeningState(self.engine, is_interruption=False)
        return None

    def on_exit(self):
        pass


class ListeningState(ConversationState):
    def __init__(self, engine, is_interruption=False):
        super().__init__(engine)
        self.is_interruption = is_interruption

        # 현재 턴 오디오 버퍼 (SmartTurn용)
        self.audio_buffer = []

        # 턴 감지 관련 변수
        self.silent_chunks = 0
        self.turn_mode = "LISTENING" # LISTENING | GRACE
        self.grace_period_end_time = None

        # STT 스레드 핸들
        self.stt_thread = None

    def on_enter(self):
        logging.info(f"STATE: [Listening] (Interruption={self.is_interruption})")

        # LED: 대기와 동일 (노란색)
        led_set_ring(233, 233, 50)

        # 1. 큐 초기화 (이전 턴의 잔여 데이터 제거)
        with self.engine.stt_audio_queue.mutex:
            self.engine.stt_audio_queue.queue.clear()
        
        with self.engine.stt_result_queue.mutex:
            self.engine.stt_result_queue.queue.clear()

        # 2. STT 시작
        self.engine.stt_stop_event.clear()
        self.stt_thread = threading.Thread(
            target=self.engine.stt_streamer.run_stt_session,
            name="STTSessionThread",
            daemon=True
        )
        self.stt_thread.start()

        # 3. Pre-buffer 처리
        # Engine에 있는 버퍼를 털어서 STT 큐와 내 버퍼에 넣음
        if self.engine.stt_pre_buffer:
            for chunk in self.engine.stt_pre_buffer:
                self.engine.stt_audio_queue.put(chunk)
                self.audio_buffer.append(chunk)
            self.engine.stt_pre_buffer.clear() # 처리 했으니 비움
        
        # 4. 인터럽션 신호 전송
        if self.is_interruption and self.engine.websocket:
            pass # C++로 인터럽션 신호 전송

    def update(self, chunk):
        # 1. 오디오 데이터 공급
        self.engine.stt_audio_queue.put(chunk)
        self.audio_buffer.append(chunk)

        # 2. VAD 분석
        is_speech = self.engine.vad.process(chunk)

        if is_speech:
            self.silent_chunks = 0
            if self.turn_mode == "GRACE":
                logging.info("🔄 유예 시간 중 재발화 -> 계속 듣기")
                self.turn_mode = "LISTENING"
                self.grace_period_end_time = None
        else:
            self.silent_chunks += 1

        # 3. 턴 종료 판단
        # [Case A] 유예 시간 모드
        if self.turn_mode == "GRACE":
            if time.time() >= self.grace_period_end_time:
                logging.info("⏳ 유예 시간 종료 -> 턴 종료 확정")
                return SttResultWaitingState(self.engine, was_interruption=self.is_interruption)
            return None
        
        # [Case B] 일반 듣기 모드 (VAD 침묵 지속 시)
        if self.silent_chunks > TURN_END_SILENCE_CHUNKS:
            prediction = self._run_smart_turn()
            
            if prediction == 1: # [종료]
                logging.info("🤖 SmartTurn: 종료(1) 예측")
                return SttResultWaitingState(self.engine, was_interruption=self.is_interruption)
            
            elif prediction == 0: # [진행중]
                logging.info(f"🤖 SmartTurn: 진행중(0) 예측 -> 유예 진입 ({SMART_TURN_GRACE_PERIOD}s)")
                self.turn_mode = "GRACE"
                self.grace_period_end_time = time.time() + SMART_TURN_GRACE_PERIOD
                self.silent_chunks = 0 # 중복 체크 방지

        return None

    def _run_smart_turn(self):
        # State가 관리하는 audio_buffer 사용
        if not self.audio_buffer: return 0
        
        concatenated = np.concatenate([c.flatten() for c in self.audio_buffer])
        full_audio = concatenated.astype(np.float32) / 32768.0
        
        result = self.engine.smart_turn_processor.predict(full_audio)
        return result['prediction']

    def on_exit(self):
        logging.info("🛑 Listening 종료 -> STT 중단 신호")
        self.engine.stt_stop_event.set() 


class SttResultWaitingState(ConversationState):
    """STT 서버로부터 최종 결과를 기다리는 상태"""
    def __init__(self, engine, was_interruption):
        super().__init__(engine)
        self.was_interruption = was_interruption
        self.start_time = None

    def on_enter(self):
        logging.info("STATE: [Processing] STT 결과 대기")
        self.start_time = time.time()

    def update(self, chunk):
        # 오디오 청크는 무시
        
        # 1. STT 결과 큐 확인 (Non-blocking)
        try:
            result = self.engine.stt_result_queue.get_nowait()
            
            if result is None:
                # STT 실패 신호 수신 -> 즉시 실패 처리
                logging.info("STT 인식 실패(None) 수신")
                return self._handle_failure()

            # 정상 텍스트 수신
            self.engine.websocket.send(json.dumps({"type": "stt_done", "stt_done_time": int(time.time() * 1000)}))
            logging.info(f"📝 인식된 텍스트: {result}")
            return ThinkingState(self.engine, result)
            
        except queue.Empty:
            # 아직 결과가 도착하지 않음 -> 타임아웃 체크
            pass

        # 2. 타임아웃 처리
        # 네트워크 지연 등으로 STT 결과가 영원히 안 올 경우를 대비
        if time.time() - self.start_time > STT_WAIT_TIMEOUT_SECONDS:
            logging.warning(f"⚠️ STT 결과 대기 시간 초과 ({STT_WAIT_TIMEOUT_SECONDS}s)")
            return self._handle_failure()
                    
        return None # 계속 대기

    def _handle_failure(self):
        """결과 수신 실패(빈 값, 타임아웃) 시 분기 처리"""
        if self.was_interruption:
            # 인터럽션이었는데 실패함 -> "뭐라고 하셨죠?" 복구 시도
            logging.info("인터럽션 인식 실패 -> Hesitating(복구) 모드 진입")
            return HesitatingState(self.engine)
        else:
            # 그냥 혼자 말하다 멈춘 것 -> 무시하고 대기
            logging.info("단순 소음 또는 인식 실패 -> Idle 복귀")
            return IdleState(self.engine)

    def on_exit(self):
        pass


class HesitatingState(ConversationState):
    """
    인터럽션인 줄 알고 끊었는데, STT가 비었을 때.
    다시 물어볼지 대기하는 상태.
    """
    def on_enter(self):
        logging.info("STATE: [Hesitating] 눈치 보는 중...")
        # 1. "뭐라고 하셨죠?" 같은 멘트 생성 요청 (비동기)
        self.engine.llm.request_generation("방금 사용자가 말을 끊었는데 못 알아들었어. 다시 물어보는 짧은 멘트.")
        self.start_time = time.time()

    def update(self, chunk):
        # 1. 사용자가 다시 말하는지 감시 (VAD On)
        if self.engine.vad.process(chunk):
            logging.info("🗣️ 사용자가 다시 말함 -> 즉시 듣기")
            self.engine.llm.cancel() # 생성하던 거 취소
            return ListeningState(self.engine, is_interruption=True)

        # 2. 일정 시간 경과 대기
        if time.time() - self.start_time > 3.0:
            # 3. LLM이 "다시 말씀해 주세요" 멘트를 완성했는지 확인
            try:
                text = self.engine.llm.response_queue.get_nowait()
                # 멘트가 준비됐고, 사용자도 계속 조용하다면 -> 말하기
                return SpeakingState(self.engine, text)
            except queue.Empty:
                pass

        return None

    def on_exit(self):
        pass


class ThinkingState(ConversationState):
    """
    LLM 생성 ~ TTS 버퍼링 ~ 재생 시작 직전까지.
    ★ 끼어들기 불가 (VAD 무시)
    """
    def __init__(self, engine, query_text):
        super().__init__(engine)
        self.query_text = query_text
        self.step = "LLM" # LLM | TTS_BUFFER

    def on_enter(self):
        logging.info("STATE: [Thinking] 답변 생성 및 준비")
        self.engine.llm.request_generation(self.query_text)

    def update(self, chunk):
        # 오디오 청크 소비만 하고 반응 안 함 (끼어들기 불가)

        if self.step == "LLM":
            try:
                response = self.engine.llm.response_queue.get_nowait()
                logging.info(f"🤖 LLM 응답: {response}")
                # TTS 시작 요청
                self.engine.tts.speak(response)
                self.step = "TTS_BUFFER"
            except queue.Empty:
                pass
        
        elif self.step == "TTS_BUFFER":
            # C++ 등에서 "재생 시작되었습니다" 신호가 왔는지 확인
            if self.engine.tts.playback_started_event.is_set():
                logging.info("🔊 재생 시작됨 -> Speaking으로 전환")
                return SpeakingState(self.engine) # 텍스트는 이미 TTS 매니저가 가짐

        return None

    def on_exit(self):
        pass


class SpeakingState(ConversationState):
    """
    실제로 소리가 나고 있는 상태.
    ★ 끼어들기 가능 (VAD 감시)
    """
    def __init__(self, engine, text=None):
        super().__init__(engine)
        # text는 이미 TTS 매니저가 처리 중이므로 로깅용
    
    def on_enter(self):
        logging.info("STATE: [Speaking] 발화 중 (Barge-in On)")
        self.engine.vad.reset() # 내가 내는 소리에 반응 안 하도록 초기화

    def update(self, chunk):
        # 1. 끼어들기 감지
        if self.engine.vad.process(chunk):
            logging.info("⚡ 끼어들기 발생! -> 중단하고 듣기")
            self.engine.tts.stop()
            self.engine.llm.cancel() # 혹시 스트리밍 중이면 취소
            return ListeningState(self.engine, is_interruption=True)

        # 2. TTS 종료 확인
        if not self.engine.tts.is_playing:
            logging.info("✅ 발화 종료 -> Idle")
            return IdleState(self.engine)

        return None

    def on_exit(self):
        pass


# ==================================================================================
# 3. Context (Engine)
# ==================================================================================

class ConversationEngine:
    def __init__(self, config):
        # 1. 설정 및 큐
        self.config = config
        self.mic_queue = queue.Queue()
        self.stt_result_queue = queue.Queue()
        self.stt_pre_buffer = []
        self.current_turn_audio = []

        # 2. 모듈 초기화 (Stub)
        # self.vad = VADProcessor(...)
        # self.stt = GoogleSTTStreamer(...)
        # self.smart_turn = SmartTurnProcessor(...)
        
        # 3. 매니저 초기화
        self.llm = LLMManager()
        self.tts = TTSManager()

        # 4. 초기 상태
        self._current_state = IdleState(self)
        self._is_running = False

    def start(self):
        self._is_running = True
        self._current_state.on_enter()
        self._loop()

    def _loop(self):
        logging.info("🚀 엔진 루프 시작")
        while self._is_running:
            try:
                # 1. 마이크 입력 (Blocking w/ Timeout)
                # 타임아웃을 줘서 청크가 안 들어와도(ex: 종료 시그널) 루프가 돌 수 있게 함
                chunk = self.mic_queue.get(timeout=0.1)
            except queue.Empty:
                chunk = None # 데이터가 없어도 update는 호출해야 함 (타이머 로직 등)

            # 2. 상태 업데이트 (핵심)
            # chunk가 None이어도 상태 내부 로직(타이머, LLM 대기 등)은 돌아야 하므로 호출
            if chunk is not None:
                next_state = self._current_state.update(chunk)
            else:
                # 데이터 없을 때의 처리는 상태별로 다를 수 있으므로 빈 배열 등을 넘기거나
                # update 메서드 시그니처를 조절. 여기서는 편의상 update를 호출 안 하거나
                # 더미 데이터를 넘길 수 있음. (구현 디테일)
                next_state = None

            # 3. 상태 전이
            if next_state:
                self._transition(next_state)

    def _transition(self, new_state):
        prev_name = self._current_state.__class__.__name__
        next_name = new_state.__class__.__name__
        logging.info(f"🔄 전이: {prev_name} -> {next_name}")

        self._current_state.on_exit()
        self._current_state = new_state
        self._current_state.on_enter()

    # --- Helper Methods ---
    def start_stt(self):
        # STT 스레드 시작 로직
        pass

    def stop_stt(self):
        # STT 스레드 종료 로직
        pass
    
    def feed_stt(self, chunk):
        # STT 큐에 넣기
        pass

# ==================================================================================
# Main 실행 예시
# ==================================================================================
if __name__ == "__main__":
    engine = ConversationEngine(config={})
    # engine.start() # 실제 실행 시