from collections import deque
import math
import os
import sys
import time
import json
import queue
import base64
import logging
import asyncio
import threading
import numpy as np
from abc import ABC, abstractmethod

from led import led_set_ring
from conversation_manager import ConversationManager
from offline_motion import offline_motion_generation

from audio_processor import VADProcessor, SmartTurnProcessor, GoogleSTTStreamer, MicrophoneStream, find_input_device

from config import (
    SMART_TURN_MODEL_PATH,
    TURN_END_SILENCE_CHUNKS,
    MAX_TURN_CHUNKS,
    SMART_TURN_GRACE_PERIOD,
    STT_WAIT_TIMEOUT_SECONDS,
    SLEEP_FILE, AWAKE_FILE, ACTIVE_SESSION_TIMEOUT, START_KEYWORD, END_KEYWORDS,
    TTS_MODEL, VOICE, RESPONSES_MODEL, RESPONSES_PRESETS, AUDIO_CONFIG, ASSETS_DIR, OPENAI_API_KEY
)
from prompts import SYSTEM_PROMPT_RESP_ONLY

from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(threadName)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)


async def run_thinking_led_spin(r, g, b, speed=4.0, focus=10.0):
    """
    LLM 생각 중 표시를 위한 비동기 LED 애니메이션 (원형 회전).
    ThinkingState에서 asyncio.create_task로 실행됨.
    """
    if not strip:
        return
    
    ring_size = 8
    top_offset = 8
    bottom_offset = 16
    start_shift = 4 

    try:
        while True:
            t = time.time() * speed
            
            for i in range(ring_size):
                # 각도 및 파동 계산
                angle = ((i - start_shift) / ring_size) * 2 * math.pi
                wave = math.sin(t + angle)
                
                # 밝기 계산 (0 ~ 1)
                brightness = (wave + 1) / 2
                brightness = math.pow(brightness, focus)
                
                # 색상 적용
                cr = int(r * brightness)
                cg = int(g * brightness)
                cb = int(b * brightness)
                
                final_color = Color(cr, cg, cb)
                
                # 위/아래 링 동시 적용
                strip.set_pixel_color(top_offset + i, final_color)
                strip.set_pixel_color(bottom_offset + i, final_color)
            
            strip.show()
            await asyncio.sleep(0.02) # 약 50 FPS
            
    except asyncio.CancelledError:
        # 태스크 취소 시 해당 링 끄기 (또는 기본 색으로 복귀)
        # 여기서는 안전하게 끄는 것으로 처리
        for i in range(ring_size):
             strip.set_pixel_color(top_offset + i, Color(0,0,0))
             strip.set_pixel_color(bottom_offset + i, Color(0,0,0))
        strip.show()
        raise

# ==================================================================================
# LLM tools
# =================================================================================
import re

# 음악 재생
with open('assets/songs_db.json', 'r') as f:
    SONG_DB = json.load(f)

def normalize_string(input_str):
    return re.sub(r'\s+', '', input_str).lower()

song_candidates = []
for song in SONG_DB:
    song_processed = song.copy()
    song_processed['norm_title'] = normalize_string(song['title'])
    song_processed['norm_artist'] = normalize_string(song['artist'])
    song_candidates.append(song_processed)

def play_music(song_title: str = "", artist_name: str = ""):
    """
    LLM이 호출하는 함수
    사용자가 요청한 조건에 맞는 노래를 DB에서 검색하여 재생
    """
    target_title = normalize_string(song_title)
    target_artist = normalize_string(artist_name)

    candidates = song_candidates

    if song_title:
        candidates = [s for s in candidates if target_title in s['norm_title']]

    if artist_name:
        candidates = [s for s in candidates if target_artist in s['norm_artist']]

    if candidates:
        selected_song = candidates[0]
        logging.info(f"재생할 노래 찾음: '{selected_song['title']}' by {selected_song['artist']}")
        return selected_song['file_path'], f"Found and playing '{selected_song['title']}' by {selected_song['artist']}."
    else:
        logging.info("재생할 노래를 찾지 못함.")
        return None, "노래를 찾을 수 없습니다."


# ==================================================================================
# 0. 매니저 클래스 (LLM/TTS 스레드 관리용 - 새로 추가 필요)
# ==================================================================================
class LLMManager:
    def __init__(self, openai_api_key, conversation_manager, main_loop, websocket):
        self.client = OpenAI(api_key=openai_api_key)
        self.history_manager = conversation_manager
        self.main_loop = main_loop
        self.websocket = websocket
        
        # 결과 전달용 큐
        self.response_queue = queue.Queue()
        
        # 실행 제어용
        self._thread = None
        self._stop_event = threading.Event()
        self.current_request_id = 0

    def request_generation(self, user_text):
        """ThinkingState에서 호출: 답변 생성 요청"""
        self._stop_event.clear()
        self.current_request_id += 1
        request_id = self.current_request_id
        
        # 이전 큐 비우기
        with self.response_queue.mutex:
            self.response_queue.queue.clear()
            
        # 별도 스레드에서 실행
        self._thread = threading.Thread(
            target=self._run_generation,
            args=(user_text, request_id),
            name="LLMThread",
            daemon=True
        )
        self._thread.start()

    def cancel(self):
        """인터럽션 발생 시 호출: 작업 취소"""
        self._stop_event.set()
        self.current_request_id += 1 # 현재 작업 ID 무효화

    def _run_generation(self, user_text, request_id):
        try:
            # ID 검증
            if self.current_request_id != request_id: return

            # 1. 사용자 메시지 기록
            self.history_manager.add_message("user", user_text)
            current_log = self.history_manager.get_current_log()
            
            # 2. 도구 정의
            tools = [
                {
                    "type": "web_search",
                    "user_location": {"type": "approximate", "country": "KR"},
                },
                {
                    "type": "function",
                    "name": "play_music",
                    "description": "사용자가 요청한 노래를 검색하여 재생합니다. 저장된 DB에 있는 노래만 재생 가능합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "song_title": {"type": "string"},
                            "artist_name": {"type": "string"},
                        },
                        "required": ["song_title", "artist_name"] 
                    }
                }
            ]

            # 3. Responses API 호출 (1차)
            if self._stop_event.is_set() or self.current_request_id != request_id: return

            params = {
                **RESPONSES_PRESETS.get(RESPONSES_MODEL, {}),
                "input": current_log,
                "tools": tools,
            }
            response = self.client.responses.create(**params)
            
            final_text = ""
            music_action = None
            motion_thread = None

            # 4. Responses 결과 처리 루프
            for item in response.output:
                if self._stop_event.is_set() or self.current_request_id != request_id: return

                if item.type == "function_call":
                    logging.info(f"🧠 Function call: {item.name}")

                    if item.name == "play_music":
                        args = json.loads(item.arguments)
                        song_title = args.get("song_title", "")
                        artist_name = args.get("artist_name", "")

                        # (1) 노래 찾기
                        file_path, message = play_music(song_title, artist_name)
                        status = "failure"

                        if file_path:
                            status = "success"
                            audio_name = f"{song_title}_{artist_name}"
                            csv_path = os.path.join(ASSETS_DIR, "headMotion", f"{audio_name}.csv")

                            # (2) 모션 생성 (없을 경우) - 스레드로 분리하여 병렬 처리
                            if not os.path.exists(os.path.join(ASSETS_DIR, "headMotion", f"{audio_name}.csv")):
                                if not os.path.exists(csv_path):
                                    logging.info(f"⚙️ 모션 파일 없음. 생성 시작: {audio_name}")
                                    motion_thread = threading.Thread(
                                        target=offline_motion_generation,
                                        args=(audio_name,),
                                        name="MotionGenThread"
                                    )
                                    motion_thread.start()
                            
                            # 액션 정보 저장
                            music_action = {"audio_name": audio_name, "motion_thread": motion_thread}
                        
                        # (3) 결과 히스토리 업데이트 (재요청을 위해)
                        current_log_copy = current_log.copy()
                        current_log_copy.append(item)
                        current_log_copy.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json.dumps({"status": status, "message": message})
                        })

                        # (4) 2차 Responses API 호출 (결과 멘트 생성)
                        params["input"] = current_log_copy
                        response_2 = self.client.responses.create(**params)
                        final_text = response_2.output[0].content[0].text.strip()
                        break

                elif item.type == "message":
                    final_text = item.content[0].text.strip()
                    break


            # 5. 결과 반환
            if self._stop_event.is_set() or self.current_request_id != request_id: return

            logging.info(f"🧠 답변 생성 완료: {final_text}")
            result_package = {"text": final_text, "action": music_action}

            if self.current_request_id == request_id:
                self.response_queue.put(result_package)

        except Exception as e:
            logging.error(f"❌ LLM 처리 중 오류: {e}")
            if self.current_request_id == request_id:
                self.response_queue.put(None)

    def request_hesitation(self):
        """HesitatingState에서 호출: 복구 멘트 생성 요청"""
        self._stop_event.clear()
        self.current_request_id += 1
        request_id = self.current_request_id
        
        # 큐 비우기
        with self.response_queue.mutex:
            self.response_queue.queue.clear()
            
        self._thread = threading.Thread(
            target=self._run_hesitation,
            args=(request_id,),
            name="HesitationLLMThread",
            daemon=True
        )
        self._thread.start()

    def _run_hesitation(self, request_id):
        try:
            if self.current_request_id != request_id: return

            # 1. 현재 로그 가져오기 (원본)
            current_log = self.history_manager.get_current_log()
            
            # 2. 임시 로그 생성 (복사본에 시스템 메시지 추가)
            # 주의: 리스트를 얕은 복사(copy())해서 원본 history에는 영향 없게 함
            temp_log = current_log.copy()
            
            # 상황 설명 시스템 메시지 주입
            system_instruction = {
                "role": "system",
                "content": (
                    "상황: 사용자가 로봇의 말을 끊고 무언가 말하려 했으나, 로봇이 제대로 알아듣지 못했습니다(STT 실패/침묵). "
                    "지침: 사용자가 다시 말하도록 자연스럽게 유도하는 짧은 문장을 생성하세요. "
                    "예시: '죄송해요, 방금 말씀을 놓쳤어요.', '네? 다시 말씀해 주시겠어요?' '이어서 말해도 될까요?' "
                    "주의: 아주 짧고 정중하게, 15자 이내로."
                )
            }
            temp_log.append(system_instruction)

            # 3. Responses API 호출
            if self._stop_event.is_set() or self.current_request_id != request_id: return

            params = {
                **RESPONSES_PRESETS.get(RESPONSES_MODEL, {}),
                "model": RESPONSES_MODEL,
                "input": temp_log,
                # Hesitation에서는 도구(Tools) 사용 안 함 (단순 발화만)
            }

            response = self.client.responses.create(**params)
            
            final_text = response.output[0].content[0].text.strip()

            # 4. 결과 처리
            if not self._stop_event.is_set() and final_text and self.current_request_id == request_id:
                logging.info(f"🤔 복구 멘트 생성: {final_text}")
                
                # 여기서는 History에 추가하지 않음.
                # 나중에 SpeakingState로 넘어갈 때(확정될 때) 추가하거나,
                # 아니면 그냥 시스템 멘트니까 History에 안 남기는 게 깔끔할 수 있음.
                # (보통 "네?" 같은 추임새는 안 남기는 게 모델 성능에 좋음)
                
                # 결과 패키지 (Action 없음)
                result_package = {
                    "text": final_text,
                    "action": None,
                    "is_hesitation": True # 플래그 추가
                }
                self.response_queue.put(result_package)

        except Exception as e:
            logging.error(f"❌ Hesitation LLM 오류: {e}")
            if self.current_request_id == request_id:
                self.response_queue.put(None)


class TTSManager:
    def __init__(self, openai_api_key, main_loop, websocket):
        self.client = OpenAI(api_key=openai_api_key)
        self.main_loop = main_loop
        self.websocket = websocket
        
        self.is_playing = False
        self.playback_started_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread = None

    def speak(self, text):
        """TTS 스트리밍 시작 (ThinkingState 호출)"""
        self._stop_event.clear()
        self.playback_started_event.clear()
        self.is_playing = True
        
        self._thread = threading.Thread(
            target=self._run_tts,
            args=(text,),
            name="TTS_Thread",
            daemon=True
        )
        self._thread.start()

    def stop(self):
        """TTS 즉시 중단 (SpeakingState 인터럽션 호출)"""
        if self.is_playing:
            logging.info("🔇 TTS 중단 요청")
            self._stop_event.set() # 1. 루프 플래그 설정
            self.is_playing = False
            
            # 2. C++ 오디오 버퍼 클리어 명령 (선택 사항)
            # asyncio.run_coroutine_threadsafe(
            #     self.websocket.send(json.dumps({"type": "stop_audio"})),
            #     self.main_loop
            # )

    def _run_tts(self, text):
        try:
            # 1. 스트리밍 시작 알림 (C++ 모션 준비 등)
            if self.websocket:
                asyncio.run_coroutine_threadsafe(
                    self.websocket.send(json.dumps({"type": "responses_only"})),
                    self.main_loop
                )

            # 2. OpenAI TTS 호출 (Stream)
            with self.client.audio.speech.with_streaming_response.create(
                model=TTS_MODEL, voice=VOICE, input=text, response_format="pcm"
            ) as response:
                first_chunk = True

                # 3. 청크 전송 루프
                for chunk in response.iter_bytes(chunk_size=4096):
                    # 중단 요청 체크
                    if self._stop_event.is_set():
                        logging.info("🛑 TTS 스트리밍 루프 탈출")
                        break
                    
                    # 웹소켓 전송
                    if self.websocket:
                        asyncio.run_coroutine_threadsafe(
                            self.websocket.send(json.dumps({
                                "type": "responses_audio_chunk",
                                "data": base64.b64encode(chunk).decode('utf-8')
                            })),
                            self.main_loop
                        )

                    # 첫 청크 전송 시점에 '재생 시작' 간주
                    if first_chunk:
                        logging.info("🔊 TTS 첫 청크 전송 -> Playback Started")
                        self.playback_started_event.set()
                        first_chunk = False

            # 4. 스트리밍 완료 처리
            if not self._stop_event.is_set():
                if self.websocket:
                    asyncio.run_coroutine_threadsafe(
                        self.websocket.send(json.dumps({"type": "responses_stream_end"})),
                        self.main_loop
                    )

        except Exception as e:
            logging.error(f"❌ TTS 스트리밍 오류: {e}", exc_info=True)
        finally:
            self.is_playing = False
            # 혹시 에러나서 시작 이벤트가 안 켜졌으면, 무한 대기 방지를 위해 켜줌
            if not self.playback_started_event.is_set():
                self.playback_started_event.set()

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

class SleepState(ConversationState):
    """
    시작 키워드만 기다리는 대기 상태.
    """
    def on_enter(self):
        logging.info("STATE: [Sleep] 시작 키워드 대기 중... (ZZZ)")
        # LED Off or Dimmed
        self.engine.vad_processor.reset()
        
        # 큐 비우기
        with self.engine.stt_audio_queue.mutex:
            self.engine.stt_audio_queue.queue.clear()

    def update(self, chunk):
        if chunk is None: return None

        # 1. 버퍼링 (키워드 앞부분 잘림 방지)
        self.engine.stt_pre_buffer.append(chunk)

        # 2. VAD 감지
        if self.engine.vad_processor.process(chunk):
            logging.info("💤 Sleep 중 발화 감지 -> 키워드 확인(Listening) 모드 진입")
            
            return ListeningState(self.engine, mode="WAKEWORD")
            
        return None

    def on_exit(self):
        pass


class IdleState(ConversationState):
    def on_enter(self):
        logging.info("STATE: [Idle] 대기 시작")
        led_set_ring(233, 233, 50)

        # VAD 상태 리셋 (이전 잡음 영향 제거)
        self.engine.vad_processor.reset()
        self.last_activity_time = time.time()

    def update(self, chunk):
        if chunk is not None:
            self.engine.stt_pre_buffer.append(chunk)

            # 발화 감지
            if self.engine.vad_processor.process(chunk):
                logging.info("🗣️ 발화 시작 감지")
                return ListeningState(self.engine, is_interruption=False, mode="NORMAL")
    
        # 타임아웃 감지
        if time.time() - self.last_activity_time > self.engine.active_timeout:
            logging.info(f"⏰ {self.engine.active_timeout}초간 입력 없음 -> Sleep 전환")

            # 종료 사운드 재생
            if self.engine.websocket:
                asyncio.run_coroutine_threadsafe(
                    self.engine.websocket.send(json.dumps({"type": "play_audio", "file_to_play": str(SLEEP_FILE)})),
                    self.engine.main_loop
                )
            self.engine.history_manager.end_session()
            return SleepState(self.engine)
        
        return None

    def on_exit(self):
        pass


class ListeningState(ConversationState):
    def __init__(self, engine, is_interruption=False, mode="NORMAL"):
        super().__init__(engine)
        self.is_interruption = is_interruption
        self.mode = mode # NORMAL | WAKEWORD

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

        # LED
        if self.mode == "NORMAL":
            led_set_ring(233,233,50)

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
            # C++로 인터럽션 신호 전송
            asyncio.run_coroutine_threadsafe(
                self.engine.websocket.send(json.dumps({"type": "user_interruption"})),
                self.engine.main_loop
            )

    def update(self, chunk):
        if chunk is None: return None

        # 1. 오디오 데이터 공급
        self.engine.stt_audio_queue.put(chunk)
        self.audio_buffer.append(chunk)

        # 2. VAD 분석
        is_speech = self.engine.vad_processor.process(chunk)

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
                return self._finish_listening()
            return None
        
        # [Case B] 일반 듣기 모드 (VAD 침묵 지속 시)
        if self.silent_chunks > TURN_END_SILENCE_CHUNKS:
            prediction = self._run_smart_turn()
            
            if prediction == 1: # [종료]
                logging.info("🤖 SmartTurn: 종료(1) 예측")
                return self._finish_listening()
            
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
    
    def _finish_listening(self):
        """듣기 종료 후 다음 상태 결정"""
        return SttResultWaitingState(
            self.engine, 
            was_interruption=self.is_interruption,
            mode=self.mode
        )

    def on_exit(self):
        logging.info("🛑 Listening 종료 -> STT 중단 신호")
        self.engine.stt_stop_event.set()
        self.engine.stt_audio_queue.put(None)


class SttResultWaitingState(ConversationState):
    """STT 서버로부터 최종 결과를 기다리는 상태"""
    def __init__(self, engine, was_interruption, mode="NORMAL"):
        super().__init__(engine)
        self.was_interruption = was_interruption
        self.mode = mode # NORMAL | WAKEWORD

        self.start_time = 0.0

    def on_enter(self):
        logging.info("STATE: [SttResultWaiting] STT 결과 대기중...")
        self.start_time = time.time()

    def update(self, chunk):
        # 오디오 청크는 무시
        
        # 1. STT 결과 큐 확인 (Non-blocking)
        try:
            text = self.engine.stt_result_queue.get_nowait()
            
            if text is None:
                # STT 실패 신호 수신 -> 즉시 실패 처리
                logging.info("STT 인식 실패(None) 수신")
                return self._handle_failure()
            
            logging.info(f"📝 STT 결과: '{text}' (Mode={self.mode})")

            if self.mode == "WAKEWORD":
                if self.engine.start_keyword in text:
                    logging.info(f"✨ 시작 키워드 감지! -> Active 모드 시작")
                
                    # 1. 깨어남 사운드 재생
                    if self.engine.websocket:
                        asyncio.run_coroutine_threadsafe(
                            self.engine.websocket.send(json.dumps({"type": "play_audio", "file_to_play": str(AWAKE_FILE)})),
                            self.engine.main_loop
                        )
                    
                    # 2. 새 세션 생성
                    self.engine.history_manager.start_new_session(system_prompt=SYSTEM_PROMPT_RESP_ONLY)

                    return IdleState(self.engine)
                else:
                    logging.info("키워드 불일치 -> 다시 Sleep")
                    return SleepState(self.engine)
            
            else:
                # 종료 키워드 검사
                if any(kw in text for kw in self.engine.end_keywords):
                    logging.info(f"👋 종료 키워드 감지: '{text}' -> Sleep 전환")

                    # 종료 사운드 재생
                    if self.engine.websocket:
                        asyncio.run_coroutine_threadsafe(
                            self.engine.websocket.send(json.dumps({"type": "play_audio", "file_to_play": str(SLEEP_FILE)})),
                            self.engine.main_loop
                        )
                    self.engine.history_manager.end_session()
                    return SleepState(self.engine)
                
            # 일반 대화 --> ThinkingState로 진행
            return ThinkingState(self.engine, text)
            
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
        if self.mode == "WAKEWORD":
            logging.info("단순 소음 또는 인식 실패 -> Sleep 복귀")
            return SleepState(self.engine) 
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
    인터럽션인 줄 알았는데 STT가 비었을 때.
    잠시(예: 2~3초) 기다리며, 로봇이 "네?" 하고 되물을지 간을 보는 상태.
    """
    def __init__(self, engine):
        super().__init__(engine)
        self.start_time = 0.0
        self.has_llm_result = False
        self.generated_text = None

    def on_enter(self):
        logging.info("STATE: [Hesitating] 눈치 보는 중... (복구 멘트 생성 시작)")
        self.start_time = time.time()
        
        # 1. LLM에 "네?" 같은 복구 멘트 생성 요청
        self.engine.llm_manager.request_hesitation()

    def update(self, chunk):
        # 1. 사용자가 다시 말하는지 감시 (VAD On)
        if chunk is not None:
             self.engine.stt_pre_buffer.append(chunk) # 버퍼링 추가
             if self.engine.vad_processor.process(chunk):
                logging.info("🗣️ 사용자가 다시 말함 -> 즉시 듣기")
                
                # 생성 중이던 LLM 취소
                self.engine.llm_manager.cancel()

                return ListeningState(self.engine, is_interruption=True)
        
        # 2. LLM 결과 확인 (Non-blocking)
        if not self.has_llm_result:
            try:
                result_pkg = self.engine.llm_manager.response_queue.get_nowait()
                if result_pkg and result_pkg.get("text"):
                    self.generated_text = result_pkg["text"]
                    self.has_llm_result = True
                    logging.info(f"🤔 멘트 준비됨: {self.generated_text}")
            except queue.Empty:
                pass

        # 3. 눈치 보기 타임아웃 처리
        # 상황: 사용자가 조용함 + LLM 멘트도 준비됨 -> 말하기 시도
        elapsed = time.time() - self.start_time
        
        if elapsed > 2.0:
            if self.has_llm_result:
                # 멘트가 준비됐으면 -> Speaking으로 넘어가서 말함
                # 이때 History에 추가할지 말지는 정책 결정 (여기선 안 함)
                logging.info("⏳ 침묵 지속 -> 복구 멘트 발화")
                
                return ThinkingState(self.engine, pre_generated_text=self.generated_text)
            
            elif elapsed > 10.0:
                # 10초가 지났는데도 LLM이 안 나오거나 사용자가 조용하면
                logging.info("⏳ 너무 오래 걸림 -> 그냥 대기(Idle)로 복귀")
                self.engine.llm_manager.cancel()
                return IdleState(self.engine)

        return None

    def on_exit(self):
        pass


class ThinkingState(ConversationState):
    """
    LLM 생성 ~ TTS 버퍼링 ~ 재생 시작 직전까지.
    끼어들기 불가 (VAD 무시)
    """
    def __init__(self, engine, query_text=None, pre_generated_text=None):
        super().__init__(engine)
        self.query_text = query_text
        self.pre_generated_text = pre_generated_text
        self.step = "LLM" # LLM | TTS_BUFFER
        self.led_task = None

    def on_enter(self):
        logging.info("STATE: [Thinking] 답변 생성 및 준비")
        if self.pre_generated_text:
            # 1. 이미 텍스트가 있으면 LLM 생략하고 바로 TTS
            logging.info(f"🚀 미리 생성된 텍스트 사용: {self.pre_generated_text}")
            self.engine.tts_manager.speak(self.pre_generated_text)
            self.step = "TTS_BUFFER"
            self.post_action = None
        else:
            if self.engine.main_loop:
                # LED: Thinking Effect On
                self.led_task = self.engine.main_loop.create_task(run_thinking_led_spin(233, 233, 50))
            self.engine.llm_manager.request_generation(self.query_text)

    def update(self, chunk):
        # 오디오 청크 무시 (인터럽션 불가)

        if self.step == "LLM":
            try:
                result_pkg = self.engine.llm_manager.response_queue.get_nowait()

                if result_pkg:
                    self.text = result_pkg.get("text", "")
                    self.post_action = result_pkg.get("action")

                    if self.text:
                        logging.info(f"🤖 TTS 준비: {self.text[:30]}...")
                        self.engine.tts_manager.speak(self.text)
                        self.step = "TTS_BUFFER"
                    else:
                        return IdleState(self.engine)
                else:
                    return IdleState(self.engine) # 에러 처리
            except queue.Empty:
                pass
        
        # 2. TTS 재생 시작 대기
        elif self.step == "TTS_BUFFER":
            if self.engine.tts_manager.playback_started_event.is_set():
                return SpeakingState(
                    self.engine, 
                    post_action=self.post_action
                )

        return None

    def on_exit(self):
        if not self.pre_generated_text:
            # LED: Thinking Effect Off
            if self.led_task and not self.led_task.done():
                self.led_task.cancel()


class SpeakingState(ConversationState):
    """
    로봇이 말하고 있는 상태.
    끼어들기 가능 (VAD 감시)
    """
    def __init__(self, engine, post_action=None):
        super().__init__(engine)
        self.post_action = post_action
    
    def on_enter(self):
        logging.info("STATE: [Speaking] 발화 중...")
        # LED: 발화 중 이펙트
        led_set_ring(50, 50, 233)
        self.engine.vad_processor.reset()
        self.engine.robot_finished_speaking = False

    def update(self, chunk):
        # 1. 끼어들기 감지
        if chunk is not None:
            self.engine.stt_pre_buffer.append(chunk) # 버퍼링 추가
            if self.engine.vad_processor.process(chunk):
                logging.info("⚡ 끼어들기 발생!")
                self.engine.tts_manager.stop()
                return ListeningState(self.engine, is_interruption=True)

        # 2. 로봇 동작 종료 확인 (C++ 시그널)
        if self.engine.robot_finished_speaking:
            logging.info("✅ 로봇 발화 및 모션 종료 (Signal Received)")
            self.engine.robot_finished_speaking = False

            # 후속 액션(노래)이 있다면 처리
            if self.post_action:
                return self._handle_post_action()
            
            return IdleState(self.engine)

        return None
    
    def _handle_post_action(self):
        """노래 재생 전 모션 생성 확인 및 명령 전송"""
        logging.info("🎵 후속 액션(음악 재생) 준비 중...")
        motion_thread = self.post_action.get("motion_thread")

        # 1. 모션 생성 스레드가 있다면 완료 대기 (Join)
        if motion_thread and motion_thread.is_alive():
            logging.info("⚙️ 모션 생성 완료 대기 (Join)...")
            motion_thread.join()
            logging.info("⚙️ 모션 생성 완료.")

        # 2. 웹소켓으로 재생 명령 전송
        audio_name = self.post_action.get("audio_name")
        if audio_name:
            if self.engine.websocket:
                asyncio.run_coroutine_threadsafe(
                    self.engine.websocket.send(json.dumps({"type": "play_audio_csv", "audio_name": audio_name})),
                    self.engine.main_loop
                )
            logging.info(f"🚀 음악 재생 명령 전송: {audio_name}")

        # 3. 노래를 틀었으니 대기로 복귀
        # (만약 노래 끝날 때까지 기다려야 한다면 'MusicPlayingState' 같은 상태 필요)
        return IdleState(self.engine)

    def on_exit(self):
        pass


# ==================================================================================
# 3. Context (Engine)
# ==================================================================================

class ConversationEngine:
    def __init__(self, websocket, main_loop):
        logging.info("초기화 시작")
        self.websocket = websocket
        self.main_loop = main_loop # asyncio loop (for websocket thread-safety)

        # 1. 설정 로드
        self.sample_rate = AUDIO_CONFIG['SAMPLE_RATE']
        self.chunk_size = AUDIO_CONFIG['VAD_CHUNK_SIZE']

        # 키워드 설정
        self.start_keyword = START_KEYWORD
        self.end_keywords = END_KEYWORDS
        self.active_timeout = ACTIVE_SESSION_TIMEOUT

        # 2. 데이터 큐 초기화
        self.mic_queue = queue.Queue()
        self.stt_audio_queue = queue.Queue()
        self.stt_result_queue = queue.Queue()

        # 3. 버퍼 초기화
        # Pre-buffer: 발화 감지 전 0.5초 정도의 오디오를 저장 (deque로 자동 길이 관리)
        pre_buffer_len = math.ceil(self.sample_rate * 0.5 / self.chunk_size)
        self.stt_pre_buffer = deque(maxlen=pre_buffer_len)

        # 4. 도구(Tools) 초기화
        self.vad_processor = VADProcessor(
            sample_rate=self.sample_rate,
            chunk_size=self.chunk_size,
            threshold=AUDIO_CONFIG['VAD_THRESHOLD'],
            consecutive_chunks=AUDIO_CONFIG['VAD_CONSECUTIVE_CHUNKS'],
            reset_interval=AUDIO_CONFIG['VAD_RESET_INTERVAL']
        )
        
        self.smart_turn_processor = SmartTurnProcessor(SMART_TURN_MODEL_PATH)

        # 5. 매니저(Managers) 초기화
        self.history_manager = ConversationManager(openai_api_key=OPENAI_API_KEY)

        self.llm_manager = LLMManager(
            openai_api_key=OPENAI_API_KEY,
            conversation_manager=self.history_manager,
            main_loop=self.main_loop,
            websocket=self.websocket
        )
        
        self.tts_manager = TTSManager(
            openai_api_key=OPENAI_API_KEY,
            main_loop=self.main_loop,
            websocket=self.websocket
        )

        # 6. STT 스트리머 초기화
        self.stt_stop_event = threading.Event()
        self.stt_streamer = GoogleSTTStreamer(
            stt_result_queue=self.stt_result_queue,
            main_loop=self.main_loop,
            websocket=self.websocket,
            sample_rate=self.sample_rate,
            stt_audio_queue=self.stt_audio_queue,
            stt_stop_event=self.stt_stop_event
        )

        # 7. 마이크 스트림 초기화
        self.mic_stream = MicrophoneStream(
            mic_audio_queue=self.mic_queue,
            sample_rate=self.sample_rate,
            chunk_size=self.chunk_size,
            channels=AUDIO_CONFIG['CHANNELS'],
            dtype=AUDIO_CONFIG['AUDIO_DTYPE'],
            device_idx=find_input_device()
        )

        # 8. 상태 초기화
        self._current_state = SleepState(self)
        self._is_running = False
        self.robot_finished_speaking = False

    def on_robot_finished(self):
        """C++로부터 말하기 종료 신호 수신"""
        logging.info("🤖 Robot finished speaking signal received")
        self.robot_finished_speaking = True

    async def start(self):
        logging.info("🚀 ConversationEngine 시작")
        self._is_running = True

        # 마이크 캡처 시작 (Background Thread inside sounddevice)
        self.mic_stream.start()

        # 초기 상태 진입
        self._current_state.on_enter()
        
        try:
            await self._loop()
        except asyncio.CancelledError:
            logging.info("엔진 작업 취소됨")
        except KeyboardInterrupt:
            logging.info("키보드 인터럽트 감지")
        finally:
            self.stop()
    
    def stop(self):
        """엔진 종료 및 리소스 정리"""
        logging.info("🛑 ConversationEngine 종료 중...")
        self._is_running = False
        
        # 마이크 중지
        if self.mic_stream:
            self.mic_stream.stop()
        
        # 매니저/스레드 정리
        self.stt_stop_event.set()
        self.stt_audio_queue.put(None)
        self.llm_manager.cancel()
        self.tts_manager.stop()
        
        logging.info("✅ 종료 완료")

    async def _loop(self):
        """메인 오디오 처리 루프"""
        while self._is_running:
            await asyncio.sleep(0.01)

            try:
                # 1. 마이크 입력 (Blocking w/ Timeout)
                chunk = self.mic_queue.get_nowait()
            except queue.Empty:
                chunk = None # 데이터가 없어도 update는 호출해야 함 (타이머 로직 등)

            # 2. 현재 상태 업데이트
            next_state = self._current_state.update(chunk)

            # 3. 상태 전이
            if next_state:
                self._transition(next_state)

    def _transition(self, new_state):
        prev_name = self._current_state.__class__.__name__
        next_name = new_state.__class__.__name__
        logging.info(f"🔄 상태 전이: {prev_name} -> {next_name}")

        self._current_state.on_exit()
        self._current_state = new_state
        self._current_state.on_enter()

async def test():
    logging.info("test 시작")
    engine = ConversationEngine(websocket=None, main_loop=asyncio.get_running_loop())
    await engine.start()

if __name__ == "__main__":
    asyncio.run(test())