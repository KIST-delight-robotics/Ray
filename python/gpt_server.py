import os
import re
import time
import wave
import json
import queue
import base64
import asyncio
import logging
import websockets
import sounddevice as sd
import torch
import torchaudio
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from openai import AsyncOpenAI
from google.cloud import speech, texttospeech
from google.api_core import exceptions

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='[python] [%(levelname)s] %(message)s')

# --- 설정 ---
# OpenAI API 키 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Google STT, TTS 인증 정보 설정
# GOOGLE_APPLICATION_CREDENTIALS 환경 변수에 Google Cloud API 키 파일 경로 설정해야 함.

# --- 디렉토리 설정 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "output"

# 경로 정의
ASSETS_AUDIO_DIR = ASSETS_DIR / "audio"
MUSIC_DIR = ASSETS_AUDIO_DIR / "music"
OUTPUT_AUDIO_DIR = OUTPUT_DIR / "audio"

# 디렉토리 생성 (없을 경우)
OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

COMMAND_CONFIG_FILE = str(CONFIG_DIR / "ray_conversation.json")
OUTPUT_GPT_FILE = str(OUTPUT_AUDIO_DIR / "output_gpt.wav")
OUTPUT_TTS_FILE = str(OUTPUT_AUDIO_DIR / "output_tts.wav")
AWAKE_FILE = str(ASSETS_AUDIO_DIR / "awake.wav")
FINISH_FILE = str(ASSETS_AUDIO_DIR / "finish.wav")

# Load command configuration
with open(COMMAND_CONFIG_FILE, encoding="utf-8") as f:
    cfg = json.load(f)["Kor"]

# 오디오 설정
CHANNELS = 1
SAMPLE_RATE = 24000
VOICE = "ash"

# 오디오 장치 설정 (직접 지정)
INPUT_DEVICE_INDEX = 7  # PipeWire를 통해 echo-cancel-source 접근

# --- 전역 변수 ---
STT_DONE_TIME = 0  # STT 완료 시간 (사용자 입력 완료 후 음성 출력까지의 시간 측정용)

# VAD 관련 전역 변수
vad_model = None
vad_sample_rate = 16000
vad_chunk_size = 512  # 32ms at 16kHz

@dataclass
class RealtimeSessionState:
    openai_connection: object
    is_streaming_response: bool = False
    current_response_id: str | None = None

# --- 오디오 장치 관리 ---
def find_pipewire_device():
    """PipeWire 장치를 동적으로 찾습니다."""
    try:
        devices = sd.query_devices()
        for idx, device in enumerate(devices):
            if device['max_input_channels'] > 0:  # type: ignore
                device_name = str(device['name']).lower()  # type: ignore
                # PipeWire 또는 echo-cancel 장치 찾기
                if 'pipewire' in device_name or ('echo' in device_name and 'cancel' in device_name):
                    logging.info(f"🔍 동적으로 발견된 장치: [{idx}] {device['name']}")  # type: ignore
                    return idx
        
        # 못 찾으면 설정값 사용
        logging.warning(f"동적 장치 검색 실패. 기본값 {INPUT_DEVICE_INDEX} 사용")
        return INPUT_DEVICE_INDEX
    except Exception as e:
        logging.error(f"장치 검색 중 오류: {e}")
        return INPUT_DEVICE_INDEX

def initialize_audio_device():
    """오디오 입력 장치를 검증합니다."""
    try:        
        # 동적으로 장치 찾기
        device_idx = find_pipewire_device()
        
        # 장치 확인
        try:
            device_info = sd.query_devices(device_idx)
            if device_info['max_input_channels'] > 0:  # type: ignore
                logging.info(f"✅ 입력 장치 확인됨: [{device_idx}] {device_info['name']}")  # type: ignore
                return True
            else:
                logging.error(f"❌ 장치 {device_idx}는 입력을 지원하지 않습니다.")
                return False
        except Exception as e:
            logging.error(f"❌ 장치 {device_idx}에 접근할 수 없습니다: {e}")
            return False
        
    except Exception as e:
        logging.error(f"❌ 오디오 장치 초기화 중 오류: {e}")
        return False

# get_input_device 함수 제거 - 직접 INPUT_DEVICE_INDEX 사용

# --- VAD 초기화 ---
def initialize_vad():
    """Silero VAD 모델을 초기화합니다."""
    global vad_model
    try:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',  # type: ignore
                                     model='silero_vad',
                                     force_reload=False,
                                     onnx=True)  # ONNX 런타임을 사용하여 CPU 성능을 최적화
        vad_model = model
        logging.info(f"✅ Silero VAD 모델 초기화 완료 - 타입: {type(vad_model)}")
        return True
    except Exception as e:
        logging.error(f"❌ VAD 모델 초기화 실패: {e}")
        return False
    
async def vad_loop(state: RealtimeSessionState):

    # VAD를 위한 오디오 설정
    audio_queue = asyncio.Queue()
    speech_start_counter = 0
    min_speech_chunks = 5  # 약 160ms (32ms * 5)
    
    loop = asyncio.get_running_loop()

    def audio_callback(indata, frames, time_info, status):
        if status:
            logging.warning(f"[오디오 상태] {status}")
        loop.call_soon_threadsafe(audio_queue.put_nowait, indata.copy())
    
    # 마이크 설정 - 동적으로 장치 검색
    device_idx = find_pipewire_device()
    
    try:
        with sd.InputStream(
            samplerate=vad_sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=vad_chunk_size,
            callback=audio_callback,
            device=device_idx
        ):
            logging.info("🎙️ VAD 기반 음성 감지 시작...")
            
            while True:
                try:
                    # 오디오 청크 가져오기
                    audio_chunk = await audio_queue.get()
                    
                    # VAD 모델이 초기화되었는지 확인
                    if vad_model is None:
                        logging.error("VAD 모델이 초기화되지 않았습니다.")
                        break
                    
                    # VAD 검사
                    audio_tensor = torch.from_numpy(audio_chunk.flatten())
                    with torch.no_grad():
                        speech_prob = vad_model(audio_tensor, vad_sample_rate).item()
                    
                    if speech_prob > 0.5:  # 음성 감지 임계값
                        speech_start_counter += 1
                        
                        # 충분한 음성 청크가 감지되면 즉시 응답 중단
                        if speech_start_counter >= min_speech_chunks:
                            logging.info("🗣️ 음성 감지! 응답 중단 및 STT 시작...")
                            
                            # 현재 응답이 진행 중이면 즉시 중단
                            if state.is_streaming_response and state.current_response_id:
                                try:
                                    await state.openai_connection.response.cancel()
                                    logging.info("기존 응답을 중단했습니다.")
                                    state.is_streaming_response = False
                                except Exception as e:
                                    logging.warning(f"응답 중단 중 오류: {e}")
                            
                            # STT 실행
                            user_text = await run_stt(timeout_sec=30.0)
                            speech_start_counter = 0  # 카운터 리셋
                            
                            if user_text.strip():
                                logging.info(f"사용자 발화 감지: '{user_text}'")
                                
                                try:
                                    # 새로운 대화 생성
                                    await state.openai_connection.conversation.item.create(
                                        item={
                                            "type": "message",
                                            "role": "user", 
                                            "content": [{"type": "input_text", "text": user_text}]
                                        }
                                    )
                                    logging.info("사용자 메시지를 대화에 추가했습니다.")
                                    # AI 응답 요청
                                    await state.openai_connection.response.create()
                                    logging.info("AI 응답을 요청했습니다.")

                                except Exception as e:
                                    logging.error(f"❌ OpenAI API 호출 중 오류: {e}")
                            else:
                                logging.warning("STT 결과가 비어있습니다.")
                    else:
                        # 음성이 감지되지 않으면 카운터 리셋
                        speech_start_counter = 0
                
                except queue.Empty:
                    continue
                except asyncio.CancelledError:
                    logging.info("VAD 루프가 취소되었습니다.")
                    break
                except Exception as e:
                    logging.error(f"VAD 처리 중 오류: {e}")
                    break
                    
    except Exception as e:
        logging.error(f"VAD 루프 중 오류: {e}")

# --- STT 기능 (기존 push_to_talk_app.py에서 가져옴) ---
async def run_stt(timeout_sec: float = 5.0):
    """
    timeout_sec 초 안에 최종 STT 결과(final_text)를 못 얻으면
    빈 문자열을 리턴하고 즉시 종료합니다.
    """
    global STT_DONE_TIME
    q_audio = queue.Queue()
    recording_done = asyncio.Event()

    def callback(indata, frames, time_info, status):
        if status:
            logging.warning(f"[오디오 상태] {status}")
        q_audio.put(bytes(indata))

    # 동적으로 장치 검색
    device_idx = find_pipewire_device()
    
    device_info = sd.query_devices(device_idx, 'input')
    stt_sample_rate = int(device_info['default_samplerate'])  # type: ignore

    stt_client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=stt_sample_rate,
        language_code="ko-KR",
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
        single_utterance=True,
    )

    def audio_generator():
        while not recording_done.is_set():
            chunk = q_audio.get()
            if chunk is None:
                break
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    final_text = ""

    try:
        with sd.InputStream(samplerate=stt_sample_rate, channels=CHANNELS, dtype="int16", callback=callback, device=device_idx):
            logging.info("🎙️ STT 시작: 말하세요...")
            responses = stt_client.streaming_recognize(streaming_config, audio_generator(), timeout=timeout_sec)

            for response in responses:
                if not response.results or not response.results[0].alternatives:
                    continue
                result = response.results[0]
                transcript = result.alternatives[0].transcript

                if result.is_final:
                    final_text += transcript
                    logging.info(f"[✅ 최종 인식 결과] {final_text}")
                    STT_DONE_TIME = time.time() * 1000
                    recording_done.set()
                    return final_text
                else:
                    logging.info(f"[📝 중간 인식] {transcript}")

    except exceptions.DeadlineExceeded:
        logging.warning(f"[STT] 타임아웃({timeout_sec}s) 발생.")
    except Exception as e:
        logging.error(f"STT 처리중 오류 발생: {e}")
    finally:
        logging.info("🎙️ STT 종료.")
        q_audio.put(None)
    
    return final_text

# --- TTS 기능 ---
async def run_tts(text, output_file=OUTPUT_TTS_FILE):
    """
    Google TTS를 사용하여 주어진 텍스트를 음성으로 변환하고 파일로 저장합니다.
    :param text: 변환할 텍스트
    :param output_file: 저장할 파일 경로
    :return: 저장된 파일 경로
    """
    logging.info(f"TTS 생성 요청: '{text}' -> {output_file}")
    tts_client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="ko-KR", ssml_gender=texttospeech.SsmlVoiceGender.MALE)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16, sample_rate_hertz=SAMPLE_RATE)
    
    response = await asyncio.to_thread(tts_client.synthesize_speech, input=synthesis_input, voice=voice, audio_config=audio_config)
    
    with open(output_file, "wb") as out:
        out.write(response.audio_content)
    logging.info(f"TTS 파일 저장 완료: {output_file}")
    return output_file

# --- 노래 파일 검색 기능 (C++ 로직과 동일) ---
def normalize_string(input_str):
    return re.sub(r'\s+', '', input_str).lower()

def find_music_file(user_text):
    normalized_input = normalize_string(user_text)
    music_files = list(Path(MUSIC_DIR).glob("*.wav"))
    
    best_match = {'song_path': None, 'type': 'none', 'match_length': 0}

    for f_path in music_files:
        try:
            title, artist = f_path.stem.split('_', 1)
            norm_title = normalize_string(title)
            norm_artist = normalize_string(artist)
            
            # 제목 우선 검색
            if norm_title in normalized_input:
                if best_match['type'] != 'title' or len(norm_title) > best_match['match_length']:
                    best_match = {'song_path': str(f_path), 'title': title, 'artist': artist, 'type': 'title', 'match_length': len(norm_title)}
            # 제목이 일치하지 않을 때만 가수 검색
            elif norm_artist in normalized_input and best_match['type'] != 'title':
                 if len(norm_artist) > best_match['match_length']:
                    best_match = {'song_path': str(f_path), 'title': title, 'artist': artist, 'type': 'artist', 'match_length': len(norm_artist)}
        except ValueError:
            logging.warning(f"파일명 형식 오류: {f_path.name}")
            continue
                
    return best_match if best_match['song_path'] else None

# --- GPT 응답 생성 ---
async def realtime_session(websocket):
    """
    VAD를 사용한 음성 감지와 OpenAI Realtime API 연동.
    사용자 발화를 감지하여 텍스트로 입력하고, AI 응답을 스트리밍합니다.
    """
    
    logging.info("🤖 Realtime GPT 세션 시작...")
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    vad_task = None
    try:
        async with openai_client.beta.realtime.connect(model="gpt-4o-mini-realtime-preview") as openai_connection:
            await openai_connection.session.update(session={
                "instructions": "너는 애니매트로닉스 로봇이야. 너의 이름은 레이야. 내가 물어보는 것들에 대해 잘 대답해줘",
                "voice": VOICE
            })

            session_state = RealtimeSessionState(openai_connection=openai_connection)

            vad_task = asyncio.create_task(vad_loop(session_state))

            accumulated_transcripts = {}
                        
            async for event in openai_connection:
                logging.info(f"📨 OpenAI 이벤트 수신: {event.type}")
                
                if event.type == "response.created":
                    session_state.current_response_id = event.response.id
                    session_state.is_streaming_response = True
                    logging.info("새로운 응답 스트림이 시작되었습니다.")
                    # 스트리밍 시작을 C++ 클라이언트에 알림
                    await websocket.send(json.dumps({"action": "gpt_stream_start"}))
                
                elif event.type == "response.audio.delta":
                    # 오디오 청크를 C++ 클라이언트로 전송
                    await websocket.send(json.dumps({
                        "action": "audio_chunk",
                        "data": event.delta
                    }))

                elif event.type == "response.audio_transcript.delta":
                    # AI 응답의 중간 텍스트
                    accumulated_transcripts[event.item_id] = accumulated_transcripts.get(event.item_id, "") + event.delta
                
                elif event.type == "response.done":
                    # AI 응답 오디오 스트림이 끝났을 때
                    final_text = accumulated_transcripts.get(event.response.id, "")
                    logging.info(f"[응답] {final_text}")
                    session_state.is_streaming_response = False
                    session_state.current_response_id = None
                    # 스트리밍 종료를 C++ 클라이언트에 알림
                    await websocket.send(json.dumps({"action": "gpt_stream_end"}))

    except websockets.exceptions.ConnectionClosed:
        logging.warning("스트리밍 중 클라이언트 연결이 종료되었습니다.")
    except Exception as e:
        logging.error(f"❌ Realtime GPT 세션 중 오류 발생: {e}", exc_info=True)
    finally:
        if vad_task and not vad_task.done():
            vad_task.cancel()
        try:
            await vad_task
        except asyncio.CancelledError:
            logging.info("VAD 태스크가 취소되었습니다.")

# --- 메인 핸들러 ---
async def chat_handler(websocket):
    global STT_DONE_TIME
    
    ray_mode = "sleep"

    logging.info(f"✅ C++ 클라이언트 연결됨.")
    
    try:
        while True:
            response_payload = None
            user_text = ""
            STT_DONE_TIME = 0

            # --- SLEEP 모드 처리 ---
            if ray_mode == "sleep":
                logging.info("💤 Sleep 모드 시작. '레이' 호출 대기 중...")
                user_text = await run_stt(timeout_sec=5)
                if "레이" in user_text:
                    logging.info("'레이' 호출 감지! Active 모드로 전환합니다.")
                    ray_mode = "active"
                    response_payload = {"action": "play_audio", "file_to_play": AWAKE_FILE, "stt_done_time": STT_DONE_TIME}
                else:
                    await asyncio.sleep(0.1)
                    continue
            
            # --- ACTIVE 모드 처리 ---
            elif ray_mode == "active":
                logging.info("⚡ Active 모드 시작. 실시간 대화를 시작합니다.")
                
                # realtime_session이 (연결 종료 등의 이유로) 완전히 끝날 때까지 여기서 대기합니다.
                await realtime_session(websocket)
                
                # realtime_session이 종료되었으므로, sleep 모드로 돌아갑니다.
                logging.info("Realtime 세션이 종료되었습니다. Sleep 모드로 전환합니다.")
                ray_mode = "sleep"
                response_payload = {"action": "play_audio", "file_to_play": FINISH_FILE}

            if response_payload:
                await websocket.send(json.dumps(response_payload))
                logging.info(f"C++ 클라이언트에 응답 전송: {response_payload['action']}")

    except websockets.exceptions.ConnectionClosed as e:
        logging.warning(f"ℹ️ C++ 클라이언트 연결이 종료되었습니다: {e}")
    except Exception as e:
        logging.error(f"❌ 핸들러 처리 중 치명적 오류 발생: {e}", exc_info=True)
    finally:
        logging.info(f"🔌 C++ 클라이언트 연결 핸들러 종료: {websocket.remote_address}")


# --- Google STT API 워밍업 ---
async def warm_up_stt_api():
    """
    최초 1회 실행합니다.
    Google STT API에 더미 요청을 보내 초기 연결 지연을 해소합니다.
    """
    logging.info("☁️ Google STT API 워밍업 시작...")
    start_time = time.time()
    try:
        def dummy_audio_generator():
            yield speech.StreamingRecognizeRequest(audio_content=b'\x00\x00')

        stt_client = speech.SpeechClient()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="ko-KR",
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config, single_utterance=True
        )

        def run_dummy_request():
            responses = stt_client.streaming_recognize(streaming_config, dummy_audio_generator())
            for _ in responses: pass
                
        await asyncio.to_thread(run_dummy_request)
        elapsed_time = time.time() - start_time
        logging.info("☁️ Google STT API 워밍업 완료. 소요 시간: {:.2f}초".format(elapsed_time))
    except Exception as e:
        logging.error(f"❌ STT API 워밍업 중 오류 발생: {e}")

async def main():
    logging.info("🚀 서버 초기화 시작: API 워밍업을 수행합니다...")
    try:
        # 오디오 장치 초기화
        if not initialize_audio_device():
            logging.error("오디오 장치 초기화에 실패했습니다. 서버를 종료합니다.")
            return
        
        # VAD 모델 초기화
        if not initialize_vad():
            logging.error("VAD 모델 초기화에 실패했습니다. 서버를 종료합니다.")
            return
        
        await warm_up_stt_api()
        server = await websockets.serve(chat_handler, "127.0.0.1", 5000)
        logging.info("🚀 통합 WebSocket 서버가 127.0.0.1:5000 에서 시작되었습니다.")
        await server.wait_closed()
    except Exception as e:
        logging.error(f"❌ 서버 실행 중 오류 발생: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass