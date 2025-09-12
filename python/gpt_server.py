import os
import sys
import time
import json
import queue
import asyncio
import logging
import threading
import math
import base64
from collections import deque

import websockets
import sounddevice as sd
import torch
import numpy as np
from pathlib import Path
from openai import AsyncOpenAI
from google.cloud import speech
from google.api_core import exceptions

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

# --- 기본 설정 ---
# OpenAI 키 & Google Cloud 인증파일 경로 환경변수 등록 필요
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- 경로 설정 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if PROJECT_ROOT not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.prompts import SYSTEM_PROMPT, REALTIME_PROMPT

ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_AUDIO_DIR = OUTPUT_DIR / "audio"
OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# 재생용 오디오 파일
AWAKE_FILE = ASSETS_DIR / "audio" / "awake.wav"
SLEEP_FILE = ASSETS_DIR / "audio" / "sleep.wav"

# --- 오디오 설정 ---
SAMPLE_RATE = 16000
CHANNELS = 1
AUDIO_DTYPE = "int16"

# --- OpenAI 설정 ---
VOICE = "coral"
TTS_MODEL = "gpt-4o-mini-tts"

# --- 키워드 설정 ---
START_KEYWORD = "레이"
END_KEYWORDS = ["종료", "쉬어"]


def find_input_device():
    """오디오 입력 장치 검색"""
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

# ==================================================================================================
# 오디오 처리기 (VAD & STT 통합)
# ==================================================================================================

class AudioProcessor:
    """마이크 입력부터 VAD, STT까지 모든 오디오 처리를 전담하는 클래스"""

    def __init__(self, stt_result_queue: asyncio.Queue, main_loop: asyncio.AbstractEventLoop, websocket, adaptation_config=None):
        # --- 상태 변수 ---
        self.stt_result_queue = stt_result_queue
        self.main_loop = main_loop
        self.websocket = websocket
        self.is_running = threading.Event()
        self.vad_active_flag = threading.Event()
        self.vad_active_flag.set()

        # --- 오디오 버퍼 ---
        # 원본 오디오 버퍼
        self.audio_queue = queue.Queue()
        # STT 사전 버퍼
        PRE_BUFFER_DURATION = 0.3 # 사전 버퍼 길이 (초)
        self.VAD_CHUNK_SIZE = 512
        pre_buffer_max_chunks = math.ceil(SAMPLE_RATE * PRE_BUFFER_DURATION / self.VAD_CHUNK_SIZE)
        self.stt_pre_buffer = deque(maxlen=pre_buffer_max_chunks)
        # VAD 처리를 위한 버퍼
        self.vad_buffer = torch.tensor([])

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
            adaptation=adaptation_config
        )
        self.stt_streaming_config = speech.StreamingRecognitionConfig(
            config=self.stt_config,
            interim_results=True,
            single_utterance=True,
        )
        logging.info("✅ Google STT 클라이언트 초기화 완료")

    def _audio_callback(self, indata, frames, time_info, status):
        """사운드디바이스 콜백. 원본 오디오를 큐에 저장."""
        if status:
            logging.warning(f"[오디오 상태] {status}")

        try:
            self.audio_queue.put(indata.copy())
        except Exception as e:
            logging.debug(f"오디오 큐 저장 중 오류: {e}")

    def _stt_audio_generator(self, stt_stop_flag=None, inactivity_stop_flag=None):
        """STT API에 오디오를 공급하는 제너레이터. 사전 버퍼 -> 실시간 오디오 순으로 공급."""
        # 1. VAD가 감지되기 전까지 쌓아둔 사전 버퍼(pre-buffer) 전송
        if self.stt_pre_buffer:
            combined_audio = np.concatenate(list(self.stt_pre_buffer))
            duration_sec = len(combined_audio) / SAMPLE_RATE
            yield speech.StreamingRecognizeRequest(audio_content=combined_audio.tobytes())
            logging.info(f"사전 버퍼 ({duration_sec:.2f}초) 전송 완료")
            self.stt_pre_buffer.clear()

        # 2. 실시간으로 들어오는 오디오 전송
        while not self.vad_active_flag.is_set():
            # 타임아웃 신호가 있으면 즉시 중단
            if (stt_stop_flag and stt_stop_flag.is_set()) or (inactivity_stop_flag and inactivity_stop_flag.is_set()):
                logging.info("오디오 생성기 중단")
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
        
        FIRST_RESPONSE_TIMEOUT = 3.0  # 첫 응답 타임아웃 (초)
        first_response_event = threading.Event()
        stt_stop_flag = threading.Event()
        
        INACTIVITY_TIMEOUT = 3.0
        inactivity_stop_flag = threading.Event()
        last_response_time = time.time()
        inactivity_thread = None

        def timeout_checker():
            """첫 응답 타임아웃을 실시간으로 체크하는 함수"""
            if not first_response_event.wait(timeout=FIRST_RESPONSE_TIMEOUT):
                logging.warning(f"STT 첫 응답 타임아웃 - 세션 종료")
                stt_stop_flag.set()

        def inactivity_timeout_checker():
            """STT 응답이 없을 경우(비활성) 오디오 전송을 중단."""
            while not stt_stop_flag.is_set() and not inactivity_stop_flag.is_set():
                if time.time() - last_response_time > INACTIVITY_TIMEOUT:
                    logging.info(f"{INACTIVITY_TIMEOUT}초 동안 STT 응답이 없어 오디오 전송을 중단합니다.")
                    inactivity_stop_flag.set()
                    break
                time.sleep(0.1)
        
        # 타임아웃 체커를 별도 스레드에서 실행
        timeout_thread = threading.Thread(target=timeout_checker, daemon=True)
        timeout_thread.start()
        
        try:
            responses = self.stt_client.streaming_recognize(self.stt_streaming_config, self._stt_audio_generator(stt_stop_flag, inactivity_stop_flag))
            
            for response in responses:
                # STT 중단 신호가 있으면 즉시 종료
                if stt_stop_flag.is_set():
                    logging.info("타임아웃으로 인한 STT 세션 중단")
                    return
                
                last_response_time = time.time()

                # 첫 응답이 도착했음을 알림
                if not first_response_event.is_set():
                    first_response_event.set()
                    
                    inactivity_thread = threading.Thread(target=inactivity_timeout_checker, daemon=True)
                    inactivity_thread.start()
                    
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
                    # STT 완료시 메인 asyncio 루프로 결과 전송
                    if final_text:
                        self.main_loop.call_soon_threadsafe(self.stt_result_queue.put_nowait, final_text)
                    stt_completion_time = int(time.time() * 1000)
                    asyncio.run_coroutine_threadsafe(
                        self.websocket.send(json.dumps({"type": "stt_done", "stt_done_time": stt_completion_time})),
                        self.main_loop
                    )
                    return
                else:
                    logging.info(f"✅ STT 중간 결과: '{transcript}'")
                    
        except exceptions.DeadlineExceeded as e:
            logging.error(f"STT 세션 타임아웃(DeadlineExceeded): {e}")
        except Exception as e:
            logging.error(f"STT 세션 중 오류: {e}")
        finally:
            first_response_event.set()
            inactivity_stop_flag.set()
            self.stt_pre_buffer.clear()
            self.vad_model.reset_states()
            self.vad_active_flag.set()
            logging.info("STT 세션 종료 및 VAD 감지 시작")

    def _process_audio_for_vad(self, audio_chunk_int16):
        # float32로 변환 (Silero VAD 요구사항)
        audio_chunk_float32 = audio_chunk_int16.astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_chunk_float32.flatten())
        
        # 버퍼에 현재 청크 추가
        self.vad_buffer = torch.cat([self.vad_buffer, audio_tensor])
        
        # 충분한 데이터가 있으면 VAD 처리
        while len(self.vad_buffer) >= self.VAD_CHUNK_SIZE:
            # 청크 크기만큼 추출하여 처리
            vad_chunk = self.vad_buffer[:self.VAD_CHUNK_SIZE]
            self.vad_buffer = self.vad_buffer[self.VAD_CHUNK_SIZE:]
            
            # VAD 모델로 음성 확률 계산
            speech_prob = self.vad_model(vad_chunk, SAMPLE_RATE).item()
            
            # 임계값 이상이면 연속 카운터 증가, 아니면 리셋
            if speech_prob > self.VAD_THRESHOLD:
                self.consecutive_speech_chunks += 1
                logging.debug(f"VAD 음성 감지: {speech_prob:.2f}, 연속 청크: {self.consecutive_speech_chunks}")
            else:
                self.consecutive_speech_chunks = 0

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

                    # 사전 버퍼 저장
                    self.stt_pre_buffer.append(audio_chunk_int16)

                    # VAD 처리
                    self._process_audio_for_vad(audio_chunk_int16)

                    # 연속적으로 음성이 감지되면 STT 세션 시작.
                    if self.consecutive_speech_chunks >= self.VAD_CONSECUTIVE_CHUNKS:
                        self.vad_active_flag.clear() # VAD 루프를 대기 상태로 전환.
                        logging.info(f"🗣️ 음성 시작 감지! STT 시작.")
                        threading.Thread(target=self._run_stt).start()
                        
                        # STT 시작과 함께 VAD 관련 상태 초기화.
                        self.vad_buffer = torch.tensor([])
                        self.consecutive_speech_chunks = 0
        except Exception as e:
            logging.error(f"오디오 처리 루프 중 치명적 오류: {e}", exc_info=True)

    def stop(self):
        """오디오 처리 스레드를 안전하게 종료."""
        self.is_running.clear()
        self.vad_active_flag.set() # 대기 상태의 스레드가 있다면 즉시 깨워서 종료되도록
        logging.info("오디오 처리 스레드 종료 신호 전송")


# ==================================================================================================
# TTS 핸들러
# ==================================================================================================

async def handle_tts_stream(response_stream, client, websocket, conversation_log, responses_start_time=None):
    """Responses API의 텍스트 스트림을 받아 TTS 오디오 스트림으로 변환 후 전송"""
    await websocket.send(json.dumps({"type": "responses_stream_start"}))
    
    full_response_text = ""
    sentence_buffer = ""
    try:
        async for event in response_stream:
            if event.type == "response.output_text.delta":
                text_chunk = event.delta
                sentence_buffer += text_chunk
                full_response_text += text_chunk
                
                if any(p in sentence_buffer for p in ".?!"):
                    async with client.audio.speech.with_streaming_response.create(
                        model=TTS_MODEL,
                        voice=VOICE,
                        input=sentence_buffer,
                        response_format="pcm"
                    ) as tts_response:
                        async for audio_chunk in tts_response.iter_bytes(chunk_size=4096):
                            await websocket.send(json.dumps({
                                "type": "responses_audio_chunk", 
                                "data": base64.b64encode(audio_chunk).decode('utf-8')
                            }))
                    sentence_buffer = ""
            
            if event.type == "response.completed":
                if responses_start_time is not None:
                    message = f"(소요시간: {time.time() - responses_start_time:.2f}초)"
                else:
                    message = ""
                logging.info(f"OpenAI 응답 완료: '{full_response_text}' {message}")

        if sentence_buffer.strip():
            async with client.audio.speech.with_streaming_response.create(
                model=TTS_MODEL,
                voice=VOICE,
                input=sentence_buffer,
                response_format="pcm"
            ) as tts_response:
                async for audio_chunk in tts_response.iter_bytes(chunk_size=4096):
                    await websocket.send(json.dumps({
                        "type": "responses_audio_chunk", 
                        "data": base64.b64encode(audio_chunk).decode('utf-8')
                    }))

    except asyncio.CancelledError:
        logging.info("TTS 스트림 처리가 중단되었습니다.")
        raise # 인터럽션을 상위 루프에 전파
    finally:
        await websocket.send(json.dumps({"type": "responses_stream_end"}))
        if full_response_text:
            conversation_log.append({"role": "assistant", "content": full_response_text})
            logging.info(f"OpenAI 응답 완료: '{full_response_text}'")

async def handle_tts_oneshot(response_text, client, websocket):
    """전체 텍스트를 받아 한 번에 TTS 처리하는 함수"""
    try:
        await websocket.send(json.dumps({"type": "responses_stream_start"}))
        async with client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL,
            voice=VOICE,
            input=response_text,
            response_format="pcm"
        ) as tts_response:
            async for audio_chunk in tts_response.iter_bytes(chunk_size=4096):
                await websocket.send(json.dumps({
                    "type": "responses_audio_chunk", 
                    "data": base64.b64encode(audio_chunk).decode('utf-8')
                }))
        
    except asyncio.CancelledError:
        logging.info("TTS 처리가 중단되었습니다.")
        raise
    finally:
        await websocket.send(json.dumps({"type": "responses_stream_end"}))


# ==================================================================================================
# Unified API Pipeline (Realtime + Responses)
# ==================================================================================================

async def run_realtime_task(websocket, openai_connection, conversation_log, realtime_finished_event: asyncio.Event, item_ids_to_manage: list, user_text):
    """(Task 1) Realtime API를 호출하고 오디오를 스트리밍합니다."""
    logging.info("⚡️ Realtime Task 시작...")
    
    try:
        # 1. 이전 턴에서 생성된 모든 대화 아이템들을 삭제하여 세션을 초기화
        if item_ids_to_manage:
            logging.info(f"이전 Realtime 대화 아이템 {len(item_ids_to_manage)}개 삭제 중...")
            delete_tasks = [openai_connection.conversation.item.delete(item_id=item_id) for item_id in item_ids_to_manage]
            await asyncio.gather(*delete_tasks, return_exceptions=True) # 예외가 발생해도 계속 진행
            item_ids_to_manage.clear()
            logging.info("이전 아이템 삭제 완료.")

        # 2. 현재 대화 기록을 기반으로 새 아이템들을 생성
        await openai_connection.session.update(session={"instructions": REALTIME_PROMPT, "voice": VOICE})

        await openai_connection.conversation.item.create(
            item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": user_text}]}
        )

        # history_items = [entry for entry in conversation_log if entry['role'] != 'system']
        # for entry in history_items:
        #     item_to_create = {
        #         "type": "message",
        #         "role": entry['role'],
        #         "content": [{"type": "input_text" if entry['role'] == 'user' else "text", "text": entry['content']}]
        #     }
        #     await openai_connection.conversation.item.create(item=item_to_create)

        # 3. 응답 생성 시작
        realtime_start_time = time.time()
        await openai_connection.response.create()

        # 4. 오디오 스트림 처리
        async for event in openai_connection:
            if event.type == "conversation.item.created":
                item_ids_to_manage.append(event.item.id)
                # await openai_connection.conversation.item.retrieve(item_id=event.previous_item_id)

            elif event.type == "response.audio.delta":
                await websocket.send(json.dumps({"type": "realtime_audio_chunk", "data": event.delta}))

            elif event.type == "response.created":
                await websocket.send(json.dumps({"type": "realtime_stream_start"}))

            elif event.type == "response.done":
                await websocket.send(json.dumps({"type": "realtime_stream_end"}))
                logging.info(f"⚡️ Realtime API 답변 생성 완료: '{event.response.output[0].content[0].transcript}' (소요시간: {time.time() - realtime_start_time:.2f}초)")
                break

            elif event.type == "conversation.item.retrieved":
                logging.info(f"이전 대화 항목이 검색되었습니다: {event.item}")

            elif event.type == "error":
                logging.error(f"Realtime API 오류 이벤트: {event}")
    
    except asyncio.CancelledError:
        logging.info("⚡️ Realtime Task가 외부에서 중단되었습니다.")
    except Exception as e:
        logging.error(f"⚡️ Realtime Task 실행 중 오류: {e}", exc_info=True)
    finally:
        # 태스크가 정상적으로 끝나든, 취소되든 항상 이벤트를 설정하여 Responses Task의 대기를 해제
        realtime_finished_event.set()
        logging.info("⚡️ Realtime Task 종료.")


async def run_responses_task(websocket, openai_client, conversation_log, realtime_finished_event: asyncio.Event):
    """(Task 2) Responses API를 호출하고, Realtime 응답이 끝난 후 TTS를 스트리밍합니다."""
    
    logging.info("🧠 Responses Task 시작...")
    response_text = ""

    try:
        responses_start_time = time.time()
        # 1. Responses API로부터 텍스트 답변 생성
        response = await openai_client.responses.create(
            model="gpt-4.1",
            input=conversation_log,
            # reasoning={ "effort": "low" },
            # text={ "verbosity": "low" },
            # stream=True
        )
        response_text = response.output_text
        logging.info(f"🧠 Responses API 답변 생성 완료: '{response_text}' (소요시간: {time.time() - responses_start_time:.2f}초)")

        # 2. TTS 스트리밍
        logging.info("...Realtime 응답 완료. Responses API의 TTS를 시작합니다.")
        await handle_tts_oneshot(response_text, openai_client, websocket)

        # 3. 대화 기록 추가
        conversation_log.append({"role": "assistant", "content": response_text})

    except asyncio.CancelledError:
        logging.info("🧠 Responses Task가 외부에서 중단되었습니다.")
    except Exception as e:
        logging.error(f"🧠 Responses Task 실행 중 오류: {e}", exc_info=True)
    finally:
        logging.info("🧠 Responses Task 종료.")


async def unified_active_pipeline(websocket, conversation_log):
    """사용자 입력에 대해 Realtime API와 Responses API를 동시에 호출하여 순차적으로 응답하는 통합 파이프라인"""
    logging.info("🤖 Unified Active Pipeline 시작...")
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    audio_processor, audio_thread = None, None
    active_response_tasks = []

    stt_result_queue = asyncio.Queue()
    main_loop = asyncio.get_running_loop()

    # Active 모드에 진입할 때 Realtime API 연결을 한 번만 생성
    async with openai_client.beta.realtime.connect(model="gpt-4o-mini-realtime-preview") as openai_connection:
        realtime_item_ids_to_manage = []
        try:
            # 1. 오디오 처리기 시작
            audio_processor = AudioProcessor(stt_result_queue, main_loop, websocket)
            audio_thread = threading.Thread(target=audio_processor.start, daemon=True)
            audio_thread.start()

            # 2. 사용자 입력을 기다리는 메인 루프
            while True:
                user_text = await stt_result_queue.get()

                # 3. 새 입력이 들어오면, 이전의 모든 AI 응답 태스크를 즉시 중단
                if active_response_tasks:
                    logging.info(f"사용자 인터럽션 감지: '{user_text}'. 이전 응답 태스크를 중단합니다.")
                    for task in active_response_tasks:
                        task.cancel()
                    # 모든 태스크가 완전히 취소될 때까지 기다림
                    await asyncio.gather(*active_response_tasks, return_exceptions=True)
                    active_response_tasks = []

                # 4. 종료 키워드 확인
                if any(kw in user_text for kw in END_KEYWORDS):
                    await websocket.send(json.dumps({"type": "play_audio", "file_to_play": str(SLEEP_FILE)}))
                    logging.info(f"종료 키워드 감지: '{user_text}' - 세션을 종료합니다.")
                    break # Active Pipeline 종료

                # 5. 대화 기록에 사용자 메시지 추가
                conversation_log.append({"role": "user", "content": user_text})

                # 6. 두 API 태스크 간의 동기화를 위한 이벤트 생성
                realtime_finished_event = asyncio.Event()

                # 7. Realtime 및 Responses API 태스크를 생성하고 동시에 실행
                realtime_task = asyncio.create_task(
                    run_realtime_task(websocket, openai_connection, conversation_log, realtime_finished_event, realtime_item_ids_to_manage, user_text)
                )
                # await websocket.send(json.dumps({"type": "responses_only"}))
                responses_task = asyncio.create_task(
                    run_responses_task(websocket, openai_client, conversation_log, realtime_finished_event)
                )
                active_response_tasks = [responses_task, realtime_task]

        except Exception as e:
            logging.error(f"Unified Active Pipeline에서 오류 발생: {e}", exc_info=True)
        finally:
            # 파이프라인 종료 시 모든 리소스 정리
            if active_response_tasks:
                for task in active_response_tasks:
                    task.cancel()
                await asyncio.gather(*active_response_tasks, return_exceptions=True)

            # Active 모드 종료 시 서버에 남아있는 아이템들을 모두 정리
            if realtime_item_ids_to_manage:
                logging.info(f"Active 세션 종료. 남은 Realtime 아이템 {len(realtime_item_ids_to_manage)}개 정리 중...")
                delete_tasks = [openai_connection.conversation.item.delete(item_id=item_id) for item_id in realtime_item_ids_to_manage]
                await asyncio.gather(*delete_tasks, return_exceptions=True)
                logging.info("남은 아이템 정리 완료.")

            if audio_processor: audio_processor.stop()
            if audio_thread and audio_thread.is_alive(): audio_thread.join(timeout=1.0)
            logging.info("🤖 Unified Active Pipeline 종료.")


# ==================================================================================================
# Sleep 모드
# ==================================================================================================

async def wakeword_detection_loop(websocket):
    """START_KEYWORD를 감지할 때까지 VAD-STT 루프를 실행 (Sleep 모드)"""
    logging.info(f"💤 Sleep 모드 시작. '{START_KEYWORD}' 호출 대기 중...")
    audio_processor, audio_thread = None, None
    try:
        keyword_queue = asyncio.Queue()
        main_loop = asyncio.get_running_loop()

        # adaptation_client = speech.AdaptationClient()
        # parent = f"projects/{GOOGLE_CLOUD_PROJECT_ID}/locations/global"

        # phrase_set_response = adaptation_client.create_phrase_set(
        #     {
        #         "parent": parent,
        #         "phrase_set_id": "wakeup_keywords",
        #         "phrase_set": {
        #             "phrases": [{"value": START_KEYWORD}],
        #             "boost": 20.0  # boost 값으로 인식률 가중치 부여
        #         }
        #     }
        # )
        # phrase_set_name = phrase_set_response.name

        # adaptation_config = speech.SpeechAdaptation(phrase_set_references=[phrase_set_name])

        audio_processor = AudioProcessor(keyword_queue, main_loop, websocket)
        audio_thread = threading.Thread(target=audio_processor.start, daemon=True)
        audio_thread.start()

        while True:
            stt_result = await keyword_queue.get()
            logging.info(f"[Sleep Mode] STT 결과: {stt_result}")
            if START_KEYWORD in stt_result:
                return
    finally:
        if audio_processor: audio_processor.stop()
        if audio_thread and audio_thread.is_alive(): audio_thread.join(timeout=1.0)
        logging.info("💤 Sleep 모드 종료.")


# ==================================================================================================
# 메인 루프
# ==================================================================================================

async def chat_handler(websocket):
    logging.info(f"✅ C++ 클라이언트 연결됨: {websocket.remote_address}")
    
    conversation_log = []  # 대화 기록 저장용 리스트
    conversation_log.append({"role": "system", "content": SYSTEM_PROMPT})
    conversation_log.append({"role": "system", "content": "[start new chat]"})

    try:
        while True:
            # 1. Sleep 모드: 키워드 감지 대기
            await wakeword_detection_loop(websocket)

            # 2. Sleep 모드 종료 후 AWAKE 음성 재생
            await websocket.send(json.dumps({"type": "play_audio", "file_to_play": str(AWAKE_FILE)}))
            
            # 3. Active 모드
            await unified_active_pipeline(websocket, conversation_log)
            
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