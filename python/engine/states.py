"""대화 상태 머신의 상태 클래스 정의."""

from __future__ import annotations

import time
import queue
import logging
import threading
import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from hardware.led import led_set_dual
from hardware.led_animations import run_scanning_led_bar
from protocol import RobotMessage
from config import (
    TURN_END_SILENCE_CHUNKS,
    SMART_TURN_GRACE_PERIOD,
    STT_WAIT_TIMEOUT_SECONDS,
    SLEEP_FILE, AWAKE_FILE,
    START_KEYWORD, END_KEYWORDS,
    LED_COLOR_OFF, LED_COLOR_SLEEP, LED_COLOR_ACTIVE, LED_COLOR_LISTENING,
)
from llm.prompts import SYSTEM_PROMPT_RESP_ONLY

if TYPE_CHECKING:
    from engine.conversation_engine import ConversationEngine

logger = logging.getLogger(__name__)


# ==================================================================================
# State Interface
# ==================================================================================

class ConversationState(ABC):
    def __init__(self, engine: ConversationEngine):
        self.engine = engine

    @abstractmethod
    def on_enter(self):
        """상태 진입 시 1회 실행"""
        pass

    @abstractmethod
    def update(self, chunk: np.ndarray) -> ConversationState | None:
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
# State Implementations
# ==================================================================================

class SleepState(ConversationState):
    """시작 키워드만 기다리는 대기 상태."""

    def on_enter(self):
        logger.info("--- STATE: [Sleep] 시작 키워드 대기 중... ---")
        led_set_dual(bar_color=LED_COLOR_OFF, ring_color=LED_COLOR_SLEEP)
        self.engine.vad_processor.reset()
        self.engine.clear_stt_audio_queue()

    def update(self, chunk):
        if chunk is None:
            return None

        self.engine.stt_pre_buffer.append(chunk)

        if self.engine.vad_processor.process(chunk):
            logger.info("Sleep 중 발화 감지 -> 키워드 확인(Listening) 모드 진입")
            return ListeningState(self.engine, mode="WAKEWORD")

        return None

    def on_exit(self):
        pass


class IdleState(ConversationState):
    """활성 세션 중 사용자 입력 대기 상태."""

    def on_enter(self):
        logger.info("--- STATE: [Idle] 발화 대기 중... ---")
        led_set_dual(LED_COLOR_ACTIVE, LED_COLOR_ACTIVE)
        self.engine.vad_processor.reset()
        self.last_activity_time = time.time()

    def update(self, chunk):
        if chunk is not None:
            self.engine.stt_pre_buffer.append(chunk)

            if self.engine.vad_processor.process(chunk):
                logger.info("발화 시작 감지")
                return ListeningState(self.engine, is_interruption=False, mode="NORMAL")

        # 타임아웃 감지
        if time.time() - self.last_activity_time > self.engine.active_timeout:
            logger.info(f"{self.engine.active_timeout}초간 입력 없음 -> Sleep 전환")
            self.engine.send_to_robot(RobotMessage.play_audio(SLEEP_FILE))
            self.engine.history_manager.end_session()
            return SleepState(self.engine)

        return None

    def on_exit(self):
        pass


class ListeningState(ConversationState):
    """사용자 음성을 듣고 있는 상태."""

    def __init__(self, engine, is_interruption=False, mode="NORMAL"):
        super().__init__(engine)
        self.is_interruption = is_interruption
        self.mode = mode

        self.audio_buffer = []
        self.silent_chunks = 0
        self.turn_mode = "LISTENING"
        self.grace_period_end_time = None
        self.stt_thread = None

    def on_enter(self):
        logger.info(f"--- STATE: [Listening] 사용자 입력 받는 중... (Interruption={self.is_interruption}) ---")

        if self.mode == "NORMAL":
            led_set_dual(LED_COLOR_LISTENING, LED_COLOR_ACTIVE)

        # 큐 초기화
        self.engine.clear_stt_audio_queue()
        self.engine.clear_stt_result_queue()

        # STT 시작
        self.stt_thread = self.engine.start_stt_session()

        # Pre-buffer 처리
        self.engine.flush_pre_buffer_to(self.engine.stt_audio_queue, self.audio_buffer)

        # 인터럽션 처리
        if self.is_interruption:
            self.engine.send_to_robot(RobotMessage.user_interruption())

    def update(self, chunk):
        if chunk is None:
            return None

        self.engine.stt_audio_queue.put(chunk)
        self.audio_buffer.append(chunk)

        is_speech = self.engine.vad_processor.process(chunk)

        if is_speech:
            self.silent_chunks = 0
            if self.turn_mode == "GRACE":
                logger.info("유예 시간 중 재발화 -> 계속 듣기")
                self.turn_mode = "LISTENING"
                self.grace_period_end_time = None
        else:
            self.silent_chunks += 1

        # 턴 종료 판단
        if self.turn_mode == "GRACE":
            if time.time() >= self.grace_period_end_time:
                logger.info("유예 시간 종료 -> 턴 종료 확정")
                return self._finish_listening()
            return None

        if self.silent_chunks > TURN_END_SILENCE_CHUNKS:
            prediction = self._run_smart_turn()

            if prediction == 1:
                logger.info("SmartTurn: 종료(1) 예측")
                return self._finish_listening()
            elif prediction == 0:
                logger.info(f"SmartTurn: 진행중(0) 예측 -> 유예 진입 ({SMART_TURN_GRACE_PERIOD}s)")
                self.turn_mode = "GRACE"
                self.grace_period_end_time = time.time() + SMART_TURN_GRACE_PERIOD
                self.silent_chunks = 0

        return None

    def _run_smart_turn(self) -> int:
        if not self.audio_buffer:
            return 0
        concatenated = np.concatenate([c.flatten() for c in self.audio_buffer])
        full_audio = concatenated.astype(np.float32) / 32768.0
        result = self.engine.smart_turn_processor.predict(full_audio)
        return result['prediction']

    def _finish_listening(self):
        return SttResultWaitingState(
            self.engine,
            was_interruption=self.is_interruption,
            mode=self.mode
        )

    def on_exit(self):
        logger.info("Listening 종료 -> STT 중단 신호")
        self.engine.stt_stop_event.set()
        self.engine.stt_audio_queue.put(None)


class SttResultWaitingState(ConversationState):
    """STT 서버로부터 최종 결과를 기다리는 상태."""

    def __init__(self, engine, was_interruption, mode="NORMAL"):
        super().__init__(engine)
        self.was_interruption = was_interruption
        self.mode = mode
        self.start_time = 0.0

    def on_enter(self):
        logger.info("--- STATE: [SttResultWaiting] STT 결과 대기중... ---")
        self.start_time = time.time()
        if self.mode == "NORMAL":
            led_set_dual(LED_COLOR_ACTIVE, LED_COLOR_ACTIVE)

    def update(self, chunk):
        # STT 결과 큐 확인
        try:
            text = self.engine.stt_result_queue.get_nowait()

            if text is None:
                logger.info("STT 인식 실패(None) 수신")
                return self._handle_failure()

            logger.info(f"STT 결과: '{text}' (Mode={self.mode})")

            if self.mode == "WAKEWORD":
                if self.engine.start_keyword in text:
                    logger.info("시작 키워드 감지! -> Active 모드 시작")
                    self.engine.send_to_robot(RobotMessage.play_audio(AWAKE_FILE))
                    self.engine.history_manager.start_new_session(system_prompt=SYSTEM_PROMPT_RESP_ONLY)
                    return IdleState(self.engine)
                else:
                    logger.info("키워드 불일치 -> 다시 Sleep")
                    return SleepState(self.engine)
            else:
                # 종료 키워드 검사
                if any(kw in text for kw in self.engine.end_keywords):
                    logger.info(f"종료 키워드 감지: '{text}' -> Sleep 전환")
                    self.engine.send_to_robot(RobotMessage.play_audio(SLEEP_FILE))
                    self.engine.history_manager.end_session()
                    return SleepState(self.engine)

            # 일반 대화
            return ThinkingState(self.engine, text)

        except queue.Empty:
            pass

        # 타임아웃 처리
        if time.time() - self.start_time > STT_WAIT_TIMEOUT_SECONDS:
            logger.warning(f"STT 결과 대기 시간 초과 ({STT_WAIT_TIMEOUT_SECONDS}s)")
            return self._handle_failure()

        return None

    def _handle_failure(self):
        if self.mode == "WAKEWORD":
            logger.info("단순 소음 또는 인식 실패 -> Sleep 복귀")
            return SleepState(self.engine)
        if self.was_interruption:
            logger.info("인터럽션 인식 실패 -> Hesitating(복구) 모드 진입")
            return HesitatingState(self.engine)
        else:
            logger.info("단순 소음 또는 인식 실패 -> Idle 복귀")
            return IdleState(self.engine)

    def on_exit(self):
        pass


class HesitatingState(ConversationState):
    """인터럽션인 줄 알았는데 STT가 비었을 때 복구 시도 상태."""

    def __init__(self, engine):
        super().__init__(engine)
        self.start_time = 0.0
        self.has_llm_result = False
        self.generated_text = None
        self.turn_id = None

    def on_enter(self):
        self.turn_id = self.engine.next_turn_id()
        logger.info(f"--- STATE: [Hesitating] 눈치 보는 중... (turn_id={self.turn_id}) ---")
        self.start_time = time.time()
        self.engine.llm_manager.request_hesitation(self.turn_id)

    def update(self, chunk):
        # 사용자가 다시 말하는지 감시
        if chunk is not None:
            self.engine.stt_pre_buffer.append(chunk)
            if self.engine.vad_processor.process(chunk):
                logger.info("사용자가 다시 말함 -> 즉시 듣기")
                self.engine.llm_manager.cancel()
                return ListeningState(self.engine, is_interruption=True)

        # LLM 결과 확인
        if not self.has_llm_result:
            try:
                result_pkg = self.engine.llm_manager.response_queue.get_nowait()
                if result_pkg and result_pkg.get("text"):
                    self.generated_text = result_pkg["text"]
                    self.has_llm_result = True
                    logger.info(f"멘트 준비됨: {self.generated_text}")
            except queue.Empty:
                pass

        # 타임아웃 처리
        elapsed = time.time() - self.start_time

        if elapsed > 2.0:
            if self.has_llm_result:
                logger.info("침묵 지속 -> 복구 멘트 발화")
                self.engine.history_manager.add_message({"role": "assistant", "content": self.generated_text, "type": "message"})
                return ThinkingState(self.engine, pre_generated_text=self.generated_text)
            elif elapsed > 10.0:
                logger.info("너무 오래 걸림 -> 대기(Idle)로 복귀")
                self.engine.llm_manager.cancel()
                return IdleState(self.engine)

        return None

    def on_exit(self):
        pass


class ThinkingState(ConversationState):
    """LLM 생성 ~ TTS 버퍼링 ~ 재생 시작 직전까지."""

    def __init__(self, engine, query_text=None, pre_generated_text=None):
        super().__init__(engine)
        self.query_text = query_text
        self.pre_generated_text = pre_generated_text
        self.step = "LLM"
        self.led_task = None
        self.post_action = None
        self.turn_id = None

    def on_enter(self):
        self.turn_id = self.engine.next_turn_id()
        logger.info(f"--- STATE: [Thinking] 답변 생성 중... (turn_id={self.turn_id}) ---")
        if self.pre_generated_text:
            logger.info(f"미리 생성된 텍스트 사용: {self.pre_generated_text}")
            self.engine.tts_manager.speak(self.pre_generated_text, self.turn_id)
            self.step = "TTS_BUFFER"
            self.post_action = None
        else:
            if self.engine.main_loop:
                self.led_task = self.engine.main_loop.create_task(run_scanning_led_bar(*LED_COLOR_ACTIVE))
            self.engine.llm_manager.request_generation(self.query_text, self.turn_id)

    def update(self, chunk):
        # 끼어들기 감지
        if chunk is not None:
            self.engine.stt_pre_buffer.append(chunk)
            if self.engine.vad_processor.process(chunk):
                logger.info("Thinking 중 끼어들기 발생!")
                self.engine.tts_manager.stop()
                self.engine.llm_manager.cancel()
                return ListeningState(self.engine, is_interruption=True)

        if self.step == "LLM":
            try:
                result_pkg = self.engine.llm_manager.response_queue.get_nowait()

                if result_pkg:
                    self.text = result_pkg.get("text", "")
                    self.post_action = result_pkg.get("action")

                    if self.text:
                        logger.info(f"TTS 준비: {self.text[:30]}...")
                        self.engine.tts_manager.speak(self.text, self.turn_id)
                        self.step = "TTS_BUFFER"
                    else:
                        return IdleState(self.engine)
                else:
                    return IdleState(self.engine)
            except queue.Empty:
                pass

        elif self.step == "TTS_BUFFER":
            if self.engine.tts_manager.playback_started_event.is_set():
                return SpeakingState(self.engine, post_action=self.post_action, turn_id=self.turn_id)

        return None

    def on_exit(self):
        if not self.pre_generated_text:
            if self.led_task and not self.led_task.done():
                self.led_task.cancel()


class SpeakingState(ConversationState):
    """로봇이 TTS 응답을 말하고 있는 상태. 끼어들기 가능."""

    def __init__(self, engine, post_action=None, turn_id=None):
        super().__init__(engine)
        self.post_action = post_action
        self.turn_id = turn_id

    def on_enter(self):
        logger.info(f"--- STATE: [Speaking] 발화 중... (turn_id={self.turn_id}) ---")
        led_set_dual(LED_COLOR_ACTIVE, LED_COLOR_ACTIVE)
        self.engine.vad_processor.reset()
        self.engine.robot_finished_speaking = False

    def update(self, chunk):
        # 끼어들기 감지
        if chunk is not None:
            self.engine.stt_pre_buffer.append(chunk)
            if self.engine.vad_processor.process(chunk):
                logger.info("Speaking 중 끼어들기 발생!")
                self.engine.tts_manager.stop()
                self.engine.history_manager.add_message({"role": "system", "content": "끼어들기 감지. 발화 중단.", "type": "message"})
                return ListeningState(self.engine, is_interruption=True)

        # 로봇 동작 종료 확인
        if self.engine.robot_finished_speaking:
            received_turn_id = self.engine._robot_finished_turn_id
            if self.turn_id and received_turn_id != self.turn_id:
                logger.warning(f"Stale speaking_finished 무시 (received={received_turn_id}, expected={self.turn_id})")
                self.engine.robot_finished_speaking = False
                return None
            logger.info(f"로봇 발화 종료 (Signal Received, turn_id={received_turn_id})")
            self.engine.robot_finished_speaking = False

            if self.post_action:
                return MusicPreparingState(self.engine, self.post_action)
            return IdleState(self.engine)

        return None

    def on_exit(self):
        pass


class MusicPreparingState(ConversationState):
    """모션 생성 완료를 논블로킹으로 대기한 뒤, 음악 재생 명령을 전송한다."""

    MOTION_TIMEOUT = 60.0

    def __init__(self, engine, post_action):
        super().__init__(engine)
        self.post_action = post_action
        self.motion_thread = post_action.get("motion_thread")
        self.start_time = 0.0

    def on_enter(self):
        logger.info("--- STATE: [MusicPreparing] 모션 생성 완료 대기 중... ---")
        led_set_dual(LED_COLOR_ACTIVE, LED_COLOR_ACTIVE)
        self.engine.vad_processor.reset()
        self.start_time = time.time()

    def update(self, chunk):
        # 끼어들기 감지
        if chunk is not None:
            self.engine.stt_pre_buffer.append(chunk)
            if self.engine.vad_processor.process(chunk):
                logger.info("MusicPreparing 중 끼어들기 발생!")
                return ListeningState(self.engine, is_interruption=True)

        # 논블로킹 체크: 모션 스레드 완료 여부
        if self.motion_thread is None or not self.motion_thread.is_alive():
            audio_name = self.post_action.get("audio_name")
            music_turn_id = self.engine.next_turn_id()
            if audio_name:
                logger.info(f"모션 생성 완료. 음악 재생 명령 전송: {audio_name} (turn_id={music_turn_id})")
                self.engine.send_to_robot(RobotMessage.play_audio_csv(audio_name, music_turn_id))
            return MusicPlayingState(self.engine, turn_id=music_turn_id)

        # 타임아웃
        if time.time() - self.start_time > self.MOTION_TIMEOUT:
            logger.warning(f"모션 생성 타임아웃 ({self.MOTION_TIMEOUT}s) -> Idle 복귀")
            return IdleState(self.engine)

        return None

    def on_exit(self):
        pass


class MusicPlayingState(ConversationState):
    """C++에서 음악(CSV) 재생 중. speaking_finished 시그널을 대기한다."""

    PLAYBACK_TIMEOUT = 600.0  # 음악 재생 최대 대기 시간 (10분)

    def __init__(self, engine, turn_id):
        super().__init__(engine)
        self.turn_id = turn_id
        self.start_time = 0.0

    def on_enter(self):
        logger.info(f"--- STATE: [MusicPlaying] 음악 재생 중... (turn_id={self.turn_id}) ---")
        led_set_dual(LED_COLOR_ACTIVE, LED_COLOR_ACTIVE)
        self.engine.vad_processor.reset()
        self.engine.robot_finished_speaking = False
        self.start_time = time.time()

    def update(self, chunk):
        # 끼어들기 감지
        if chunk is not None:
            self.engine.stt_pre_buffer.append(chunk)
            if self.engine.vad_processor.process(chunk):
                logger.info("MusicPlaying 중 끼어들기 발생!")
                self.engine.send_to_robot(RobotMessage.user_interruption())
                return ListeningState(self.engine, is_interruption=True)

        # 재생 완료 확인
        if self.engine.robot_finished_speaking:
            received_turn_id = self.engine._robot_finished_turn_id
            if self.turn_id and received_turn_id != self.turn_id:
                logger.warning(f"Stale speaking_finished 무시 (received={received_turn_id}, expected={self.turn_id})")
                self.engine.robot_finished_speaking = False
                return None
            logger.info(f"음악 재생 종료 (Signal Received, turn_id={received_turn_id})")
            self.engine.robot_finished_speaking = False
            return IdleState(self.engine)

        # 타임아웃
        if time.time() - self.start_time > self.PLAYBACK_TIMEOUT:
            logger.warning(f"음악 재생 타임아웃 ({self.PLAYBACK_TIMEOUT}s) -> Idle 복귀")
            return IdleState(self.engine)

        return None

    def on_exit(self):
        pass
