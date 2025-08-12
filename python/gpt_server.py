import os
import re
import time
import json
import queue
import asyncio
import logging
import threading
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
logging.basicConfig(level=logging.INFO, format='[python] [%(levelname)s] %(message)s')

# --- 기본 설정 --- 
# OpenAI & Google Cloud 인증 정보는 환경 변수를 통해 설정해야 합니다.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- 경로 설정 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_AUDIO_DIR = OUTPUT_DIR / "audio"
OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# --- 오디오 설정 ---
# Google STT 권장사항 및 VAD 모델과의 통일을 위해 16000Hz로 샘플레이트를 고정합니다.
SAMPLE_RATE = 16000
CHANNELS = 1
AUDIO_DTYPE = "int16"

# --- OpenAI 설정 ---
VOICE = "ash"

# ==================================================================================================
# 오디오 처리기 (VAD & STT 통합)
# ==================================================================================================

class AudioProcessor:
    """마이크 입력부터 VAD, STT까지 모든 오디오 처리를 전담하는 클래스"""

    def __init__(self, stt_result_queue: asyncio.Queue, main_loop: asyncio.AbstractEventLoop):
        # --- 상태 변수 ---
        self.stt_result_queue = stt_result_queue
        self.main_loop = main_loop
        self.is_running = threading.Event()
        self.stt_finished_event = threading.Event() # STT 세션의 완료/활성 상태를 관리하는 이벤트
        self.stt_finished_event.set() # 초기 상태는 '완료(set)'로 설정

        # --- 오디오 버퍼 ---
        self.audio_queue = queue.Queue() # 마이크 콜백에서 받은 원본 오디오가 쌓이는 곳
        # 0.5초 분량의 오디오를 바이트 단위로 저장하는 롤링 버퍼
        self.max_pre_buffer_bytes = int(SAMPLE_RATE * 0.5 * 2) # 0.5초 * 16000Hz * 2bytes(int16)
        self.stt_pre_buffer = b''
        self.vad_buffer = torch.tensor([]) # VAD 처리를 위한 버퍼

        # --- VAD 설정 ---
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=True)
        self.vad_model = model
        self.vad_iterator = utils[3](model) # VADIterator
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
            logging.warning(f"[오디오 상태] {status}")
        self.audio_queue.put(indata.copy())

    def _stt_audio_generator(self):
        """STT API에 오디오를 공급하는 제너레이터. 사전 버퍼 -> 실시간 오디오 순으로 공급."""
        # 1. VAD가 감지되기 전까지 쌓아둔 사전 버퍼(pre-buffer)부터 보냄
        if self.stt_pre_buffer:
            duration_sec = (len(self.stt_pre_buffer) / 2) / SAMPLE_RATE
            logging.info(f"사전 버퍼 ({duration_sec:.2f}초) 전송 중...")
            yield speech.StreamingRecognizeRequest(audio_content=self.stt_pre_buffer)
            self.stt_pre_buffer = b'' # 사전 버퍼는 전송 후 초기화

        # 2. 실시간으로 들어오는 오디오 전송
        while not self.stt_finished_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                yield speech.StreamingRecognizeRequest(audio_content=chunk.tobytes())
            except queue.Empty:
                if self.stt_finished_event.is_set():
                    break
                continue

    def _run_stt(self):
        """단일 STT 세션을 실행하고 결과를 반환. 이 함수는 동기적으로 실행됨."""
        try:
            responses = self.stt_client.streaming_recognize(self.stt_streaming_config, self._stt_audio_generator())
            for response in responses:
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
        except Exception as e:
            logging.error(f"STT 세션 중 오류: {e}")
        finally:
            self.vad_iterator.reset_states()
            self.stt_finished_event.set() # VAD 루프를 다시 시작하도록 신호
            logging.info("STT 세션 종료 및 VAD 상태 초기화")

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
                device=device_idx
            ):
                VAD_CHUNK_SIZE = 512 # Silero VAD는 16kHz에서 512 샘플 크기를 사용
                while self.is_running.is_set():
                    # STT 세션이 끝날 때까지 대기
                    self.stt_finished_event.wait()
                    if not self.is_running.is_set(): break # stop()이 호출되면 즉시 종료

                    # STT가 비활성화 상태일 때만 VAD를 위해 오디오 처리
                    try:
                        audio_chunk_int16 = self.audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    # 사전 버퍼 저장 (롤링 버퍼)
                    new_bytes = audio_chunk_int16.tobytes()
                    self.stt_pre_buffer = (self.stt_pre_buffer + new_bytes)[-self.max_pre_buffer_bytes:]

                    # VAD 처리를 위해 float32로 변환하고 버퍼에 추가
                    audio_chunk_float32 = audio_chunk_int16.astype(np.float32) / 32768.0
                    audio_tensor = torch.from_numpy(audio_chunk_float32.flatten())
                    self.vad_buffer = torch.cat([self.vad_buffer, audio_tensor])

                    # VAD 버퍼에 처리할 데이터가 충분한지 확인하고, 있다면 처리
                    while len(self.vad_buffer) >= VAD_CHUNK_SIZE:
                        vad_chunk = self.vad_buffer[:VAD_CHUNK_SIZE]
                        self.vad_buffer = self.vad_buffer[VAD_CHUNK_SIZE:]

                        speech_dict = self.vad_iterator(vad_chunk, return_seconds=True)

                        if speech_dict and 'start' in speech_dict:
                            self.stt_finished_event.clear() # STT 시작. VAD 루프를 '대기' 상태로 전환
                            logging.info("🗣️ 음성 시작 감지! STT 시작.")
                            threading.Thread(target=self._run_stt).start()
                            # STT가 이제 오디오를 직접 소비하므로 VAD 버퍼 초기화
                            self.vad_buffer = torch.tensor([])
                            break # VAD 처리 루프 종료, STT가 이제 활성화됨

        except Exception as e:
            logging.error(f"오디오 처리 루프 중 치명적 오류: {e}", exc_info=True)

    def stop(self):
        """오디오 처리 스레드를 안전하게 종료."""
        self.is_running.clear()
        self.stt_finished_event.set() # 대기 상태의 스레드가 있다면 즉시 깨워서 종료되도록
        logging.info("오디오 처리 스레드 종료 신호 전송")



# ==================================================================================================
# 비동기 통신 및 메인 로직
# ==================================================================================================

async def handle_stt_results(stt_queue: asyncio.Queue, openai_connection, session_state):
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

async def handle_openai_responses(openai_connection, websocket, session_state):
    """(태스크 B) OpenAI의 응답을 받아 C++ 클라이언트에 전송하는 역할"""
    try:
        async for event in openai_connection:
            if event.type == "response.created":
                session_state['is_streaming_response'] = True
                await websocket.send(json.dumps({"action": "gpt_stream_start"}))
            
            elif event.type == "response.audio.delta":
                await websocket.send(json.dumps({"action": "audio_chunk", "data": event.delta}))

            elif event.type == "response.done":
                session_state['is_streaming_response'] = False
                await websocket.send(json.dumps({"action": "gpt_stream_end"}))

    except asyncio.CancelledError:
        logging.info("OpenAI 응답 처리 태스크가 취소되었습니다.")
    except Exception as e:
        logging.error(f"OpenAI 응답 처리 중 오류: {e}", exc_info=True)

async def realtime_session(websocket):
    """사용자 발화와 AI 응답을 동시에 처리하는 메인 세션"""
    logging.info("🤖 Realtime GPT 세션 시작...")
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    audio_processor = None
    audio_thread = None

    try:
        async with openai_client.beta.realtime.connect(model="gpt-4o-mini-realtime-preview") as openai_connection:
            await openai_connection.session.update(session={
                "instructions": "너는 애니매트로닉스 로봇이야. 너의 이름은 레이야. 내가 물어보는 것들에 대해 잘 대답해줘",
                "voice": VOICE
            })

            stt_result_queue = asyncio.Queue()
            main_loop = asyncio.get_running_loop()
            session_state = {'is_streaming_response': False}

            # 1. 오디오 처리기 생성 및 별도 스레드에서 실행
            audio_processor = AudioProcessor(stt_result_queue, main_loop)
            audio_thread = threading.Thread(target=audio_processor.start, daemon=True)
            audio_thread.start()

            # 2. STT 결과 처리와 OpenAI 응답 처리를 두 개의 태스크로 만들어 동시 실행
            task_a = asyncio.create_task(handle_stt_results(stt_result_queue, openai_connection, session_state))
            task_b = asyncio.create_task(handle_openai_responses(openai_connection, websocket, session_state))

            # 두 태스크가 모두 종료될 때까지 대기
            await asyncio.gather(task_a, task_b)

    except websockets.exceptions.ConnectionClosed:
        logging.warning("클라이언트 연결이 종료되었습니다.")
    except Exception as e:
        logging.error(f"Realtime GPT 세션 중 오류 발생: {e}", exc_info=True)
    finally:
        if audio_processor:
            audio_processor.stop()
        if audio_thread and audio_thread.is_alive():
            audio_thread.join(timeout=1.0)
        logging.info("🤖 Realtime GPT 세션 종료.")

# --- 기존 헬퍼 함수들 (수정 없음) ---
def find_input_device():
    try:
        devices = sd.query_devices()
        for idx, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                device_name = str(device['name']).lower()
                if 'pipewire' in device_name:
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
    # 현재 구조에서는 'active' 모드만 존재한다고 가정하고 바로 realtime_session 시작
    # sleep/active 모드 전환이 필요하다면 이 부분에 해당 로직 추가 필요
    await realtime_session(websocket)
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