import asyncio
import base64
import wave
import json
import websockets
from openai import AsyncOpenAI
import logging
import sounddevice as sd
import queue
from google.cloud import speech, texttospeech
from google.api_core import exceptions
import os
import functools
import time
import re
from pathlib import Path

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

# 디렉토리 생성 (없을 경우)
OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

COMMAND_CONFIG_FILE = str(CONFIG_DIR / "ray_conversation.json")
LOG_FILE = str(OUTPUT_DIR / "conversation_log.json")
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
openai_connection = None
openai_lock = asyncio.Lock()
ray_mode = "sleep"
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

# --- GPT 응답 생성 ---
async def generate_gpt_response_audio(user_text: str) -> str:
    """
    GPT-4o-realtime API를 사용하여 사용자 텍스트에 대한 음성 응답을 생성하고 파일로 저장합니다.
    세션은 전역 `openai_connection`을 사용합니다.
    """
    global openai_connection, GPT_RESPONSE_TIME, GPT_RESPONSE_TEXT
    async with openai_lock:
        start_time = time.time() * 1000
        logging.info(f"💬 GPT 대화 시작: {user_text}")
        log_conversation("user", user_text)
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
                    wf.writeframes(base64.b64decode(event.delta))
                elif event.type == "response.audio_transcript.delta":
                    accumulated_transcripts[event.item_id] = accumulated_transcripts.get(event.item_id, "") + event.delta
                elif event.type == "response.audio.done":
                    final_text = accumulated_transcripts.get(event.item_id, "")
                    logging.info(f"[응답] {final_text}")
                    GPT_RESPONSE_TIME = time.time() * 1000 - start_time
                    GPT_RESPONSE_TEXT = final_text
                    log_conversation("assistant", final_text)
                    break
    return OUTPUT_GPT_FILE

# --- 대화 로깅 ---
def log_conversation(role, text):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            log_entry = {"role": role, "content": text}
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logging.error(f"❌ 로그 파일 작성 중 오류: {e}")

# --- 메인 핸들러 ---
async def chat_handler(websocket):
    global openai_connection, ray_mode, stt_first_attempt_flag, STT_READY_TIME, STT_DONE_TIME, GPT_RESPONSE_TIME, GPT_RESPONSE_TEXT
    if not openai_connection:
        logging.error("❌ OpenAI 전역 세션이 설정되지 않았습니다.")
        return

    logging.info(f"✅ C++ 클라이언트 연결됨. 현재 모드: {ray_mode}")
    
    try:
        # 클라이언트로부터 메시지를 기다림
        async for message in websocket:
            data = json.loads(message)
            if data.get("request") != "next_action":
                continue

            file_to_play = None
            user_text = ""
            GPT_RESPONSE_TEXT = ""
            STT_DONE_TIME = 0

            # --- SLEEP 모드 처리 ---
            if ray_mode == "sleep":
                logging.info("💤 Sleep 모드 시작. '레이' 호출 대기 중...")
                user_text = await run_stt(timeout_sec=5)
                if "레이" in user_text:
                    logging.info("'레이' 호출 감지!")
                    ray_mode = "active"
                    file_to_play = AWAKE_FILE
                else:
                    await websocket.send(json.dumps({"action": "sleep"}))
                    logging.info("레이 호출 없음. C++ 클라이언트에 sleep 유지 신호 전송.")
                    continue
            
            # --- ACTIVE 모드 처리 ---
            elif ray_mode == "active":
                logging.info("⚡ Active 모드 시작. 사용자 질문 대기 중...")
                user_text = ""
                for attempt in range(3):
                    if attempt == 0:
                        stt_first_attempt_flag = True
                    else:
                        stt_first_attempt_flag = False
                    text = await run_stt(timeout_sec=5.0)
                    if text:
                        user_text = text
                        break
                    logging.info(f"묵묵부답... ({attempt+1}/3)")

                if not user_text:
                    logging.info("응답 없음. Sleep 모드로 전환.")
                    ray_mode = "sleep"
                    file_to_play = FINISH_FILE

                # --- 키워드 처리 ---
                # 종료 명령어
                elif any(kw in user_text for kw in cfg["conditions"].get("QUIT_PROGRAM", [])):
                    logging.info("종료 명령어 감지. Sleep 모드로 전환.")
                    ray_mode = "sleep"
                    file_to_play = FINISH_FILE
                # 노래 명령어
                elif any(kw in user_text for kw in cfg["conditions"].get("SING_A_SONG", [])):
                    logging.info("노래 명령어 감지.")
                    norm_input = normalize_string(user_text)
                    if norm_input in ["노래불러줘", "노래들려줘", "노래틀어줘"]:
                        response_text = "네, 무슨 노래 불러줄까요?"
                        file_to_play = await run_tts(response_text, OUTPUT_TTS_FILE)
                    else:
                        found_song_info = find_music_file(user_text)
                        if found_song_info:
                            title = found_song_info['title']
                            artist = found_song_info['artist']
                            response_text = f"{title} 말씀이신가요? 지금 {title} by {artist}를 재생할게요."
                            file_to_play = await run_tts(response_text, OUTPUT_TTS_FILE)
                            await websocket.send(json.dumps({"action": "play_music", "file_to_play": file_to_play, "title": title, "artist": artist}))
                            logging.info(f"C++ 클라이언트에 재생 명령 전송: {found_song_info['song_path']}")
                            continue
                        else:
                            response_text = "말씀하신 곡은 목록에 없어요. 다시 말씀해 주세요!"
                            file_to_play = await run_tts(response_text, OUTPUT_TTS_FILE)

                # --- 일반 대화 처리 (GPT 호출) ---
                else:
                    file_to_play = await generate_gpt_response_audio(user_text)

            await websocket.send(json.dumps({
                "action": "play_audio",
                "file_to_play": str(file_to_play) if file_to_play else None,
                "stt_ready_time": STT_READY_TIME,
                "stt_done_time": STT_DONE_TIME,
                "gpt_response_time": GPT_RESPONSE_TIME,
                "user_text": user_text,
                "gpt_response_text": GPT_RESPONSE_TEXT
            }))
            logging.info(f"C++ 클라이언트에 재생 명령 전송: {file_to_play}")

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
        # 더미 오디오 데이터 생성
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
            # 요청이 실제로 전송되고 처리되도록 생성기를 소모합니다.
            for _ in responses:
                pass
                
        await asyncio.to_thread(run_dummy_request)

        elapsed_time = time.time() - start_time
        logging.info("☁️ Google STT API 워밍업 완료. 소요 시간: {:.2f}초".format(elapsed_time))
        return  # 성공 시 함수 종료
    except Exception as e:
        logging.error(f"❌ STT API 워밍업 중 오류 발생: {e}")

# --- OpenAI 연결 설정 ---
async def setup_openai_connection():
    """
    OpenAI 실시간 세션을 비동기적으로 설정하고, 전역 변수 openai_connection에 저장합니다.
    connection_manager를 반환하여 세션 종료 시 사용할 수 있도록 합니다.
    """
    global openai_connection
    logging.info("🤖 OpenAI 세션 연결 시작...")
    
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    connection_manager = openai_client.beta.realtime.connect(model="gpt-4o-realtime-preview")
    openai_connection = await connection_manager.__aenter__()
    await openai_connection.session.update(session={
        "instructions": "너는 애니매트로닉스 로봇이야. 너의 이름은 레이야. 내가 물어보는 것들에 대해 잘 대답해줘",
        "voice": VOICE
    })
    logging.info("✅ OpenAI 전역 세션이 연결되었습니다.")
    return connection_manager

async def main():
    logging.info("🚀 서버 초기화 시작: API 워밍업 및 연결을 수행합니다...")
    connection_manager = None
    try:
        # 시간이 소요되는 네트워크 작업들을 병렬로 실행
        results = await asyncio.gather(
            warm_up_stt_api(),
            setup_openai_connection(),
            return_exceptions=True
        )

        # gather 결과 처리
        stt_warmup_success = False
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"❌ 초기화 중 오류 발생: {result}", exc_info=result)
            elif hasattr(result, '__aexit__'): # OpenAI connection_manager 식별
                connection_manager = result
            else: # warm_up_stt_api는 성공 시 None을 반환하므로 성공으로 간주
                stt_warmup_success = True

        if not connection_manager:
            logging.error("❌ OpenAI 연결에 실패하여 서버를 시작할 수 없습니다.")
            return
        if not stt_warmup_success:
            logging.warning("⚠️ STT API 워밍업에 실패했지만 서버는 계속 실행됩니다.")

        # WebSocket 서버 시작
        server = await websockets.serve(chat_handler, "127.0.0.1", 5000)
        logging.info("🚀 통합 WebSocket 서버가 127.0.0.1:5000 에서 시작되었습니다.")
        await server.wait_closed()
    finally:
        if connection_manager:
            logging.info("🔌 OpenAI 전역 세션을 종료합니다.")
            await connection_manager.__aexit__(None, None, None)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass