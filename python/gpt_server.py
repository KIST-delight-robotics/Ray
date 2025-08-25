import os
import sys
import time
import json
import queue
import asyncio
import logging
import threading
import math
from collections import deque

import websockets
import sounddevice as sd
import torch
import torchaudio
import numpy as np
from pathlib import Path
from openai import AsyncOpenAI
from google.cloud import speech
from google.api_core import exceptions

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

# --- 기본 설정 --- 
# OpenAI & Google Cloud 인증 정보는 환경 변수를 통해 설정해야 합니다.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- 경로 설정 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if PROJECT_ROOT not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config.prompts as prompts

ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_AUDIO_DIR = OUTPUT_DIR / "audio"
OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# 재생용 오디오 파일
AWAKE_FILE = ASSETS_DIR / "audio" / "awake.wav"
SLEEP_FILE = ASSETS_DIR / "audio" / "sleep.wav"

# --- 오디오 설정 ---
# Google STT 권장사항 및 VAD 모델과의 통일을 위해 16000Hz로 샘플레이트를 고정합니다.
SAMPLE_RATE = 16000
CHANNELS = 1
AUDIO_DTYPE = "int16"

# --- OpenAI 설정 ---
PROMPT = prompts.MONDAY_PROMPT
VOICE = "coral"

# ==================================================================================================
# 오디오 처리기 (VAD & STT 통합)
# ==================================================================================================

class AudioProcessor:
    """마이크 입력부터 VAD, STT까지 모든 오디오 처리를 전담하는 클래스"""

    def __init__(self, stt_result_queue: asyncio.Queue, main_loop: asyncio.AbstractEventLoop, websocket):
        # --- 상태 변수 ---
        self.stt_result_queue = stt_result_queue
        self.main_loop = main_loop
        self.websocket = websocket
        self.is_running = threading.Event()
        self.vad_active_flag = threading.Event() # VAD 감지 활성화 플래그
        self.vad_active_flag.set() # 초기 상태는 '활성화(set)'로 설정

        # --- 오디오 버퍼 ---
        self.audio_queue = queue.Queue() # 마이크 콜백에서 받은 원본 오디오가 쌓이는 곳
        # VAD 감지 전 오디오를 저장하는 롤링 버퍼 (numpy 배열 청크 저장)
        PRE_BUFFER_DURATION = 0.5  # 사전 버퍼링 시간 (초)
        self.VAD_CHUNK_SIZE = 512 # Silero VAD는 16kHz에서 512 샘플 크기를 사용
        pre_buffer_max_chunks = math.ceil(SAMPLE_RATE * PRE_BUFFER_DURATION / self.VAD_CHUNK_SIZE)
        self.stt_pre_buffer = deque(maxlen=pre_buffer_max_chunks)
        self.vad_buffer = torch.tensor([]) # VAD 처리를 위한 버퍼

        # --- VAD 설정 ---
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=True)
        self.vad_model = model
        self.VAD_THRESHOLD = 0.5  # 음성으로 판단할 확률 임계값
        self.VAD_CONSECUTIVE_CHUNKS = 3 # 연속으로 감지해야할 청크 수
        self.consecutive_speech_chunks = 0
        logging.info("✅ Silero VAD 초기화 완료")

        # --- STT 설정 ---
        self.stt_client = speech.SpeechClient()
        self.stt_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code="ko-KR",
        )
        self.stt_streaming_config = speech.StreamingRecognitionConfig(
            config=self.stt_config,
            interim_results=True,
            single_utterance=True,
        )
        logging.info("✅ Google STT 클라이언트 초기화 완료")

    def _audio_callback(self, indata, frames, time_info, status):
        """사운드디바이스 콜백. 원본 오디오를 큐에 넣기만 함."""
        if status:
            # ALSA 에러를 로그로만 남기고 계속 진행
            if status.input_overflow:
                logging.debug("Input overflow 발생 (일시적)")
            elif status.input_underflow:
                logging.debug("Input underflow 발생 (일시적)")
            else:
                logging.warning(f"[오디오 상태] {status}")
        
        try:
            self.audio_queue.put(indata.copy())
        except Exception as e:
            logging.debug(f"오디오 큐 저장 중 오류: {e}")

    def _stt_audio_generator(self, stt_should_stop=None):
        """STT API에 오디오를 공급하는 제너레이터. 사전 버퍼 -> 실시간 오디오 순으로 공급."""
        # 1. VAD가 감지되기 전까지 쌓아둔 사전 버퍼(pre-buffer)부터 보냄
        if self.stt_pre_buffer:
            combined_audio = np.concatenate(list(self.stt_pre_buffer))
            duration_sec = len(combined_audio) / SAMPLE_RATE
            yield speech.StreamingRecognizeRequest(audio_content=combined_audio.tobytes())
            logging.info(f"사전 버퍼 ({duration_sec:.2f}초) 전송 완료")
            self.stt_pre_buffer.clear() # 사전 버퍼는 전송 후 초기화

        # 2. 실시간으로 들어오는 오디오 전송
        while not self.vad_active_flag.is_set():
            # 타임아웃 신호가 있으면 즉시 중단
            if stt_should_stop and stt_should_stop.is_set():
                logging.info("타임아웃으로 인해 오디오 생성기 중단")
                break
                
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                yield speech.StreamingRecognizeRequest(audio_content=chunk.tobytes())
            except queue.Empty:
                if self.vad_active_flag.is_set():
                    break
                continue

    def _run_stt(self):
        """단일 STT 세션을 실행하고 결과를 반환. 이 함수는 동기적으로 실행됨."""
        
        first_response_timeout = 3.0  # 첫 응답 타임아웃 (초)
        start_time = time.time()
        has_received_first_response = threading.Event()
        stt_should_stop = threading.Event()
        
        def timeout_checker():
            """첫 응답 타임아웃을 실시간으로 체크하는 함수"""
            if not has_received_first_response.wait(timeout=first_response_timeout):
                logging.warning(f"STT 첫 응답 타임아웃 ({first_response_timeout}초) - 세션 종료")
                stt_should_stop.set()
        
        # 타임아웃 체커를 별도 스레드에서 실행
        timeout_thread = threading.Thread(target=timeout_checker, daemon=True)
        timeout_thread.start()
        
        try:
            responses = self.stt_client.streaming_recognize(self.stt_streaming_config, self._stt_audio_generator(stt_should_stop))
            
            for response in responses:
                # STT 중단 신호가 있으면 즉시 종료
                if stt_should_stop.is_set():
                    logging.info("타임아웃으로 인한 STT 세션 중단")
                    return
                
                # 첫 응답이 도착했음을 알림
                if not has_received_first_response.is_set():
                    has_received_first_response.set()
                    logging.info(f"STT 첫 응답 수신 (소요시간: {time.time() - start_time:.2f}초)")
                    
                    # c++에 인터럽션 신호 전송
                    asyncio.run_coroutine_threadsafe(
                        self.websocket.send(json.dumps({"type": "user_interruption"})),
                        self.main_loop
                    )

                if not response.results or not response.results[0].alternatives:
                    continue
                result = response.results[0]
                transcript = result.alternatives[0].transcript
                if result.is_final:
                    final_text = result.alternatives[0].transcript.strip()
                    logging.info(f"✅ STT 최종 결과: '{final_text}'")
                    # 메인 asyncio 루프로 결과를 안전하게 전송
                    if final_text: # 최종 텍스트가 있을 때만 큐에 넣음
                        self.main_loop.call_soon_threadsafe(self.stt_result_queue.put_nowait, final_text)
                    return
                else:
                    logging.info(f"✅ STT 중간 결과: '{transcript}'")
        except exceptions.DeadlineExceeded as e:
            logging.error(f"STT 세션 타임아웃(DeadlineExceeded): {e}")
        except Exception as e:
            logging.error(f"STT 세션 중 오류: {e}")
        finally:
            # 타임아웃 체커 스레드 정리
            has_received_first_response.set()  # 타임아웃 스레드가 대기 중이라면 깨워서 종료시킴
            self.stt_pre_buffer.clear() # 사전 버퍼 초기화
            self.vad_model.reset_states() # VAD 모델 상태 초기화
            self.vad_active_flag.set() # VAD 루프를 다시 시작하도록 신호
            logging.info("STT 세션 종료 및 VAD 감지 시작")

    def start(self):
        """오디오 처리 스레드의 메인 루프. 이 함수가 별도 스레드에서 실행됨."""
        self.is_running.set()
        logging.info("🎙️ 오디오 처리 스레드 시작...")

        device_idx = find_input_device()
        if device_idx is None: return

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=AUDIO_DTYPE,
                callback=self._audio_callback,
                device=device_idx,
                blocksize=self.VAD_CHUNK_SIZE
            ):
                while self.is_running.is_set():
                    # STT 실행중일 경우 대기
                    self.vad_active_flag.wait()
                    if not self.is_running.is_set(): break

                    try:
                        # 처리 전 큐 사이즈를 확인하여 처리가 밀리는지 파악
                        queue_size = self.audio_queue.qsize()
                        if queue_size > 1:
                            logging.warning(f"오디오 큐가 밀리고 있습니다. 현재 크기: {queue_size}")
                        
                        audio_chunk_int16 = self.audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    start_time = time.perf_counter()

                    # 사전 버퍼 저장 (롤링 버퍼)
                    self.stt_pre_buffer.append(audio_chunk_int16)

                    # VAD 처리를 위해 float32로 변환하고 버퍼에 추가
                    audio_chunk_float32 = audio_chunk_int16.astype(np.float32) / 32768.0
                    audio_tensor = torch.from_numpy(audio_chunk_float32.flatten())

                    if len(audio_tensor) == self.VAD_CHUNK_SIZE and len(self.vad_buffer) == 0:
                        speech_prob = self.vad_model(audio_tensor, SAMPLE_RATE).item()
                        if speech_prob > self.VAD_THRESHOLD:
                            self.consecutive_speech_chunks += 1
                        else:
                            self.consecutive_speech_chunks = 0
                    else:
                        logging.debug("예외 경로 실행: 오디오 청크 크기가 비정상이거나 버퍼가 남아있습니다.")
                        self.vad_buffer = torch.cat([self.vad_buffer, audio_tensor])
                        
                        # 버퍼에 VAD를 처리할 만큼의 데이터가 쌓였는지 확인.
                        while len(self.vad_buffer) >= self.VAD_CHUNK_SIZE:
                            vad_chunk = self.vad_buffer[:self.VAD_CHUNK_SIZE]
                            self.vad_buffer = self.vad_buffer[self.VAD_CHUNK_SIZE:]

                            speech_prob = self.vad_model(vad_chunk, SAMPLE_RATE).item()
                            if speech_prob > self.VAD_THRESHOLD:
                                self.consecutive_speech_chunks += 1
                            else:
                                self.consecutive_speech_chunks = 0
                            break
                    
                    # 연속적으로 음성이 감지되면 STT 세션을 시작.
                    if self.consecutive_speech_chunks >= self.VAD_CONSECUTIVE_CHUNKS:
                        self.vad_active_flag.clear() # VAD 루프를 '대기' 상태로 전환.
                        logging.info(f"🗣️ 음성 시작 감지! STT 시작.")
                        threading.Thread(target=self._run_stt).start()
                        
                        # STT 시작과 함께 VAD 관련 상태를 깨끗하게 초기화.
                        self.vad_buffer = torch.tensor([])
                        self.consecutive_speech_chunks = 0

                    processing_time_ms = (time.perf_counter() - start_time) * 1000
                    logging.debug(f"오디오 청크 처리 시간: {processing_time_ms:.2f}ms")

        except Exception as e:
            logging.error(f"오디오 처리 루프 중 치명적 오류: {e}", exc_info=True)

    def stop(self):
        """오디오 처리 스레드를 안전하게 종료."""
        self.is_running.clear()
        self.vad_active_flag.set() # 대기 상태의 스레드가 있다면 즉시 깨워서 종료되도록
        logging.info("오디오 처리 스레드 종료 신호 전송")



# ==================================================================================================
# 비동기 통신 및 메인 로직
# ==================================================================================================

async def handle_stt_results(stt_queue: asyncio.Queue, openai_connection, session_state, session_end_flag: asyncio.Event):
    """(태스크 A) STT 결과를 받아 OpenAI에 전송하는 역할"""
    while True:
        try:
            user_text = await stt_queue.get()
            if not user_text:
                continue

            # AI가 응답 중이었다면, 응답을 중단시킴
            if session_state['is_streaming_response']:
                try:
                    await openai_connection.response.cancel()
                    logging.info("기존 AI 응답을 중단했습니다.")
                    session_state['is_streaming_response'] = False
                except Exception as e:
                    logging.warning(f"응답 중단 중 오류: {e}")
            
            if any(kw in user_text for kw in ["종료", "쉬어"]):
                # 종료 키워드 감지 시 세션 종료, Sleep 모드로 전환
                logging.info(f"종료 키워드 감지: '{user_text}' - 세션을 종료합니다.")
                session_end_flag.set() 
                break  # STT 결과 처리 루프 종료
            else:
                # OpenAI에 사용자 메시지 전송 및 AI 응답 요청
                await openai_connection.conversation.item.create(
                    item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": user_text}]}
                )
                await openai_connection.response.create()
                logging.info(f"OpenAI에 사용자 메시지 '{user_text}' 전송 및 응답 요청")

        except asyncio.CancelledError:
            logging.info("STT 결과 처리 태스크가 취소되었습니다.")
            break
        except Exception as e:
            logging.error(f"STT 결과 처리 중 오류: {e}", exc_info=True)

async def handle_openai_responses(openai_connection, websocket, session_state, session_end_flag: asyncio.Event):
    """(태스크 B) OpenAI의 응답을 받아 C++ 클라이언트에 전송하는 역할"""
    try:
        async for event in openai_connection:
            if event.type == "response.created":
                session_state['is_streaming_response'] = True
                await websocket.send(json.dumps({"type": "gpt_stream_start"}))
            
            elif event.type == "response.audio.delta":
                await websocket.send(json.dumps({"type": "audio_chunk", "data": event.delta}))

            elif event.type == "response.done":
                session_state['is_streaming_response'] = False
                await websocket.send(json.dumps({"type": "gpt_stream_end"}))
                response = event.response.output[0].content[0].transcript
                logging.info(f"OpenAI 응답 완료: '{response}'")

    except asyncio.CancelledError:
        logging.info("OpenAI 응답 처리 태스크가 취소되었습니다.")
    except Exception as e:
        logging.error(f"OpenAI 응답 처리 중 오류: {e}", exc_info=True)
    finally:
        if not session_end_flag.set():
            session_end_flag.set()

async def realtime_session(websocket):
    """사용자 발화와 AI 응답을 동시에 처리하는 메인 세션 (Active 모드)"""
    logging.info("🤖 Realtime GPT 세션 시작...")
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    audio_processor = None
    audio_thread = None

    try:
        async with openai_client.beta.realtime.connect(model="gpt-4o-realtime-preview") as openai_connection:
            await openai_connection.session.update(session={
                "instructions": PROMPT,
                "voice": VOICE
            })

            stt_result_queue = asyncio.Queue()
            main_loop = asyncio.get_running_loop()
            session_state = {'is_streaming_response': False}
            session_end_flag = asyncio.Event()  # 세션 종료 신호를 위한 이벤트

            # 1. 오디오 처리기 생성 및 별도 스레드에서 실행
            audio_processor = AudioProcessor(stt_result_queue, main_loop, websocket)
            audio_thread = threading.Thread(target=audio_processor.start, daemon=True)
            audio_thread.start()

            # 2. STT 결과 처리와 OpenAI 응답 처리를 두 개의 태스크로 만들어 동시 실행
            task_a = asyncio.create_task(handle_stt_results(stt_result_queue, openai_connection, session_state, session_end_flag))
            task_b = asyncio.create_task(handle_openai_responses(openai_connection, websocket, session_state, session_end_flag))

            await session_end_flag.wait()

    except websockets.exceptions.ConnectionClosed:
        logging.warning("클라이언트 연결이 종료되었습니다.")
    except Exception as e:
        logging.error(f"Realtime GPT 세션 중 오류 발생: {e}", exc_info=True)
    finally:
        # audio_processor와 audio_thread 정리
        if audio_processor:
            audio_processor.stop()
        if audio_thread and audio_thread.is_alive():
            audio_thread.join(timeout=1.0)
        # 태스크 취소
        if 'task_a' in locals() and not task_a.done():
            task_a.cancel()
        if 'task_b' in locals() and not task_b.done():
            task_b.cancel()
        logging.info("🤖 Realtime GPT 세션 종료.")

# --- Sleep 모드 로직 ---
async def wakeword_detection_loop(websocket, keyword: str = "레이"):
    """'레이'라는 키워드를 감지할 때까지 VAD-STT 루프를 실행 (Sleep 모드)"""
    logging.info(f"💤 Sleep 모드 시작. '{keyword}' 호출 대기 중...")
    audio_processor = None
    audio_thread = None
    
    try:
        keyword_queue = asyncio.Queue()
        main_loop = asyncio.get_running_loop()

        audio_processor = AudioProcessor(keyword_queue, main_loop, websocket)
        audio_thread = threading.Thread(target=audio_processor.start, daemon=True)
        audio_thread.start()

        while True:
            stt_result = await keyword_queue.get()
            logging.info(f"[Sleep Mode] STT 결과: {stt_result}")
            if keyword in stt_result:
                logging.info(f"'{keyword}' 호출 감지! Active 모드로 전환합니다.")
                return # 호출이 감지되면 함수 종료 -> Active 모드로 전환
    
    except asyncio.CancelledError:
        logging.info("호출 감지 루프가 취소되었습니다.")
    except Exception as e:
        logging.error(f"호출 감지 루프 중 오류: {e}", exc_info=True)
    finally:
        if audio_processor:
            audio_processor.stop()
        if audio_thread and audio_thread.is_alive():
            audio_thread.join(timeout=1.0)
        logging.info("💤 Sleep 모드 종료.")


# --- 기존 헬퍼 함수들 ---
def find_input_device():
    try:
        devices = sd.query_devices()
        for idx, device in enumerate(devices):
            if device['max_input_channels'] > 0 and 'pipewire' in str(device['name']).lower():
                logging.info(f"🔍 발견된 입력 장치: [{idx}] {device['name']}")
                return idx
        logging.error("❌ 사용 가능한 오디오 입력 장치를 찾지 못했습니다.")
        return None
    except Exception as e:
        logging.error(f"장치 검색 중 오류: {e}")
        return None

# --- 메인 핸들러 및 서버 시작 ---
async def chat_handler(websocket):
    logging.info(f"✅ C++ 클라이언트 연결됨: {websocket.remote_address}")
    
    try:
        while True:
            # 1. Sleep 모드: 키워드 감지 대기
            await wakeword_detection_loop(websocket)

            # Sleep 모드 종료 후 기상 음성 재생
            await websocket.send(json.dumps({"type": "play_audio", "file_to_play": str(AWAKE_FILE)}))

            # 2. Active 모드: 실시간 대화 세션 진행
            await realtime_session(websocket)
            
            # Active 모드가 끝나면 다시 Sleep 모드로 돌아감
            await websocket.send(json.dumps({"type": "play_audio", "file_to_play": str(SLEEP_FILE)}))
            logging.info("Active 세션 종료. 다시 Sleep 모드로 전환합니다.")

    except websockets.exceptions.ConnectionClosed:
        logging.warning(f"🔌 C++ 클라이언트 연결 종료됨: {websocket.remote_address}")
    except Exception as e:
        logging.error(f"Chat 핸들러에서 예외 발생: {e}", exc_info=True)
    finally:
        logging.info(f"🔌 C++ 클라이언트 연결 핸들러 종료: {websocket.remote_address}")

async def main():
    logging.info("🚀 서버 초기화를 시작합니다...")
    server = await websockets.serve(chat_handler, "127.0.0.1", 5000)
    logging.info("🚀 통합 WebSocket 서버가 127.0.0.1:5000 에서 시작되었습니다.")
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("서버를 종료합니다.")