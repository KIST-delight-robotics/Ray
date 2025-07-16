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
from pathlib import Path
from openai import AsyncOpenAI
from google.cloud import speech, texttospeech
from google.api_core import exceptions

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')

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
LOG_DIR = OUTPUT_DIR / "logs"

# 디렉토리 생성 (없을 경우)
OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

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

# --- 전역 변수 ---
openai_lock = asyncio.Lock()
# 시간 측정용 전역 변수
stt_first_attempt_flag = True  # STT 첫 시도 여부 플래그
STT_READY_TIME = 0  # STT 준비 시간 (프로그램 시작 후 음성 입력 대기까지의 시간 측정용)
STT_DONE_TIME = 0  # STT 완료 시간 (사용자 입력 완료 후 음성 출력까지의 시간 측정용)
GPT_RESPONSE_TIME = 0  # GPT 응답 시간 (GPT 응답 생성에 걸린 시간)
GPT_RESPONSE_TEXT = ""

# --- STT 기능 (기존 push_to_talk_app.py에서 가져옴) ---
async def run_stt(timeout_sec: float = 5.0):
    """
    timeout_sec 초 안에 최종 STT 결과(final_text)를 못 얻으면
    빈 문자열을 리턴하고 즉시 종료합니다.
    """
    global STT_READY_TIME, STT_DONE_TIME, stt_first_attempt_flag
    q_audio = queue.Queue()
    recording_done = asyncio.Event()

    def callback(indata, frames, time_info, status):
        if status:
            logging.warning(f"[오디오 상태] {status}")
        q_audio.put(bytes(indata))

    input_device_index = None
    for idx, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0:
            input_device_index = idx
            break

    if input_device_index is None:
        logging.error("[오류] 입력 가능한 마이크 장치를 찾을 수 없습니다.")
        return ""

    device_info = sd.query_devices(input_device_index, 'input')
    stt_sample_rate = int(device_info['default_samplerate'])

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
        with sd.InputStream(samplerate=stt_sample_rate, channels=CHANNELS, dtype="int16", callback=callback):
            logging.info("🎙️ STT 시작: 말하세요...")
            if stt_first_attempt_flag:
                STT_READY_TIME = time.time() * 1000
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

# --- OpenAI 세션 관리 ---
async def start_new_openai_session():
    """
    새로운 OpenAI 실시간 세션을 비동기적으로 설정하고,
    connection_manager와 connection 객체를 반환합니다.
    """
    logging.info("🤖 새로운 OpenAI 세션 연결 시작...")
    connection_manager = None
    try:
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        connection_manager = openai_client.beta.realtime.connect(model="gpt-4o-realtime-preview")
        openai_connection = await connection_manager.__aenter__()
        await openai_connection.session.update(session={
            "instructions": "너는 애니매트로닉스 로봇이야. 너의 이름은 레이야. 내가 물어보는 것들에 대해 잘 대답해줘",
            "voice": VOICE
        })
        logging.info("✅ 새로운 OpenAI 세션이 연결되었습니다.")
        return connection_manager, openai_connection
    except Exception as e:
        logging.error(f"❌ OpenAI 세션 시작 중 오류 발생: {e}")
        if connection_manager:
            await connection_manager.__aexit__(None, None, None)
        return None, None

async def end_current_openai_session(connection_manager):
    """
    현재 OpenAI 세션을 비동기적으로 종료합니다.
    """
    if not connection_manager:
        return
    logging.info("🔌 OpenAI 세션을 종료합니다.")
    try:
        await connection_manager.__aexit__(None, None, None)
        logging.info("✅ OpenAI 세션이 성공적으로 종료되었습니다.")
    except Exception as e:
        logging.error(f"❌ OpenAI 세션 종료 중 오류 발생: {e}")

# --- GPT 응답 생성 ---
async def generate_gpt_response_audio(user_text: str, openai_connection, log_file: str) -> str:
    """
    주어진 OpenAI 세션을 사용하여 사용자 텍스트에 대한 AI 음성 응답을 생성하고 파일로 저장합니다.
    """
    global GPT_RESPONSE_TIME, GPT_RESPONSE_TEXT
    if not openai_connection:
        logging.error("❌ GPT 요청 실패: OpenAI 세션이 유효하지 않습니다.")
        return None

    async with openai_lock:
        first_received = True
        start_time = time.time() * 1000
        logging.info(f"💬 GPT 대화 시작: {user_text}")
        log_conversation("user", user_text, log_file)
        await openai_connection.conversation.item.create(
            item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": user_text}]}
        )
        await openai_connection.response.create()
        with wave.open(OUTPUT_GPT_FILE, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            accumulated_transcripts = {}
            async for event in openai_connection:
                if event.type == "response.audio.delta":
                    if first_received:
                        logging.info(f"GPT 응답 시작 시간: {time.time() * 1000 - start_time}ms")
                        first_received = False
                    wf.writeframes(base64.b64decode(event.delta))
                elif event.type == "response.audio_transcript.delta":
                    accumulated_transcripts[event.item_id] = accumulated_transcripts.get(event.item_id, "") + event.delta
                elif event.type == "response.audio.done":
                    final_text = accumulated_transcripts.get(event.item_id, "")
                    logging.info(f"[응답] {final_text}")
                    GPT_RESPONSE_TIME = time.time() * 1000 - start_time
                    GPT_RESPONSE_TEXT = final_text
                    log_conversation("assistant", final_text, log_file)
                    break
    return OUTPUT_GPT_FILE

# --- 대화 로깅 ---
def log_conversation(role, text, log_file):
    """지정된 로그 파일에 대화를 기록합니다."""
    if not log_file:
        logging.warning("⚠️ 로그 파일 경로가 지정되지 않아 대화 로깅을 건너뜁니다.")
        return
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            log_entry = {"role": role, "content": text, "timestamp": time.time()}
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logging.error(f"❌ 로그 파일({log_file}) 작성 중 오류: {e}")

# --- 메인 핸들러 ---
async def chat_handler(websocket):
    global stt_first_attempt_flag, STT_READY_TIME, STT_DONE_TIME, GPT_RESPONSE_TIME, GPT_RESPONSE_TEXT
    
    ray_mode = "sleep"
    session_task = None
    log_file_path = None

    logging.info(f"✅ C++ 클라이언트 연결됨. 초기 모드: {ray_mode}")
    
    try:
        async for message in websocket:
            data = json.loads(message)
            if data.get("request") != "next_action":
                continue

            response_payload = None
            user_text = ""
            GPT_RESPONSE_TEXT = ""
            STT_DONE_TIME = 0

            # --- SLEEP 모드 처리 ---
            if ray_mode == "sleep":
                logging.info("💤 Sleep 모드 시작. '레이' 호출 대기 중...")
                user_text = await run_stt(timeout_sec=5)
                if "레이" in user_text:
                    logging.info("'레이' 호출 감지! Active 모드로 전환합니다.")
                    ray_mode = "active"
                    
                    response_payload = {"action": "play_audio", "file_to_play": AWAKE_FILE}

                    if session_task:
                        logging.warning("⚠️ 이전 세션 작업이 아직 완료되지 않았을 수 있습니다. 취소하고 새 작업을 시작합니다.")
                        session_task.cancel()

                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    log_file_path = str(LOG_DIR / f"conversation_{timestamp}.json")
                    logging.info(f"새 대화 세션 시작. 로그 파일: {log_file_path}")
                    session_task = asyncio.create_task(start_new_openai_session())
                else:
                    response_payload = {"action": "sleep"}
            
            # --- ACTIVE 모드 처리 ---
            elif ray_mode == "active":
                logging.info("⚡ Active 모드 시작. 사용자 질문 대기 중...")
                user_text = ""
                for attempt in range(3):
                    stt_first_attempt_flag = (attempt == 0)
                    text = await run_stt(timeout_sec=5.0)
                    if text:
                        user_text = text
                        break
                    logging.info(f"묵묵부답... ({attempt+1}/3)")

                is_quit_command = any(kw in user_text for kw in cfg["conditions"].get("QUIT_PROGRAM", []))
                if not user_text or is_quit_command:
                    logging.info("응답 없거나 종료 명령어 감지. Sleep 모드로 전환.")
                    ray_mode = "sleep"
                    response_payload = {"action": "play_audio", "file_to_play": FINISH_FILE}
                    
                    if session_task:
                        # 백그라운드에서 세션 종료 실행
                        async def end_session_safely(task):
                            try:
                                if task.done() and not task.cancelled():
                                    manager, _ = task.result()
                                    if manager:
                                        await end_current_openai_session(manager)
                                else:
                                    task.cancel()
                            except Exception as e:
                                logging.error(f"세션 종료 처리 중 오류: {e}")
                        asyncio.create_task(end_session_safely(session_task))
                    
                    session_task = None
                    log_file_path = None
                
                else: # 실제 대화 처리
                    if not session_task:
                        logging.error("비정상적인 상태: Active 모드이지만 세션 생성 작업이 없습니다. Sleep 모드로 강제 전환합니다.")
                        ray_mode = "sleep"
                        response_payload = {"action": "play_audio", "file_to_play": FINISH_FILE}
                    else:
                        try:
                            logging.info("OpenAI 세션 준비를 기다리는 중...")
                            connection_manager, openai_connection = await asyncio.wait_for(session_task, timeout=10.0)
                            
                            if not openai_connection:
                                raise ValueError("OpenAI 세션 연결에 실패했습니다.")

                            # 키워드 처리 (노래)
                            if any(kw in user_text for kw in cfg["conditions"].get("SING_A_SONG", [])):
                                logging.info("노래 명령어 감지.")
                                norm_input = normalize_string(user_text)
                                if norm_input in ["노래불러줘", "노래들려줘", "노래틀어줘"]:
                                    response_text = "네, 무슨 노래 불러줄까요?"
                                    file_to_play = await run_tts(response_text, OUTPUT_TTS_FILE)
                                    response_payload = {"action": "play_audio", "file_to_play": str(file_to_play)}
                                else:
                                    found_song_info = find_music_file(user_text)
                                    if found_song_info:
                                        title, artist = found_song_info['title'], found_song_info['artist']
                                        response_text = f"{title} 말씀이신가요? 지금 {title} by {artist}를 재생할게요."
                                        file_to_play = await run_tts(response_text, OUTPUT_TTS_FILE)
                                        response_payload = {"action": "play_music", "file_to_play": file_to_play, "title": title, "artist": artist}
                                    else:
                                        response_text = "말씀하신 곡은 목록에 없어요. 다시 말씀해 주세요!"
                                        file_to_play = await run_tts(response_text, OUTPUT_TTS_FILE)
                                        response_payload = {"action": "play_audio", "file_to_play": str(file_to_play)}
                            # 일반 대화 (GPT)
                            else:
                                file_to_play = await generate_gpt_response_audio(user_text, openai_connection, log_file_path)
                                response_payload = {
                                    "action": "play_audio",
                                    "file_to_play": str(file_to_play) if file_to_play else None,
                                    "stt_ready_time": STT_READY_TIME, "stt_done_time": STT_DONE_TIME,
                                    "gpt_response_time": GPT_RESPONSE_TIME,
                                    "user_text": user_text, "gpt_response_text": GPT_RESPONSE_TEXT
                                }

                        except (asyncio.TimeoutError, ValueError) as e:
                            logging.error(f"❌ 세션 준비 실패 ({type(e).__name__}). Sleep 모드로 전환합니다.")
                            ray_mode = "sleep"
                            response_payload = {"action": "play_audio", "file_to_play": FINISH_FILE}
                            if session_task:
                                session_task.cancel()
                            session_task = None
                        except Exception as e:
                            logging.error(f"❌ Active 모드 처리 중 오류: {e}. Sleep 모드로 전환합니다.", exc_info=True)
                            ray_mode = "sleep"
                            response_payload = {"action": "play_audio", "file_to_play": FINISH_FILE}
                            if session_task and session_task.done() and not session_task.cancelled():
                                manager, _ = session_task.result()
                                if manager:
                                    asyncio.create_task(end_current_openai_session(manager))
                            session_task = None

            if response_payload:
                await websocket.send(json.dumps(response_payload))
                logging.info(f"C++ 클라이언트에 응답 전송: {response_payload['action']}")

    except websockets.exceptions.ConnectionClosed as e:
        logging.warning(f"ℹ️ C++ 클라이언트 연결이 종료되었습니다: {e}")
        if session_task:
            logging.info("클라이언트 연결 종료로 인한 세션 정리 시작.")
            # 백그라운드에서 안전하게 종료
            async def final_cleanup(task):
                if task.done() and not task.cancelled():
                    try:
                        manager, _ = task.result()
                        if manager: await end_current_openai_session(manager)
                    except Exception as ex:
                        logging.error(f"최종 세션 정리 중 오류: {ex}")
                else:
                    task.cancel()
            asyncio.create_task(final_cleanup(session_task))
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