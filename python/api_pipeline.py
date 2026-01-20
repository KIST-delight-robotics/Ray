import os
import json
import time
import wave
import base64
import asyncio
import logging
from typing import List, Dict, Any

import websockets
from openai import AsyncOpenAI

from config import (
    SLEEP_FILE, AWAKE_FILE, ACTIVE_SESSION_TIMEOUT, START_KEYWORD, END_KEYWORDS,
    TTS_MODEL, VOICE, RESPONSES_MODEL, RESPONSES_PRESETS, AUDIO_CONFIG, ASSETS_DIR
)
from audio_processor import AudioProcessor
from conversation_manager import ConversationManager
from offline_motion import offline_motion_generation

from rpi5_ws2812.ws2812 import Color
from led import led_set_ring, strip
import math

logger = logging.getLogger(__name__)


async def run_thinking_led_breathing():
    """
    LLM 생각 중 표시를 위한 비동기 LED 애니메이션 (숨쉬기 효과).
    """
    try:
        t = 0
        while True:
            # 파란색(Blue) 계열로 부드럽게 숨쉬는(Breathing) 효과
            # sin 함수를 이용해 최소 밝기(min_brightness) 이상에서 부드럽게 오르내림
            min_brightness = 20
            max_brightness = 255
            brightness = int(((math.sin(t) + 1) / 2) * (max_brightness - min_brightness) + min_brightness)

            # Blue channel만 사용 (R, G, B)
            led_set_ring(0, 0, brightness)
            
            t += 0.2
            await asyncio.sleep(0.05) # 50ms 대기 (취소 포인트)
            
    except asyncio.CancelledError:
        # 태스크 취소 시 LED 끄기
        led_set_ring(0, 0, 0)
        raise

async def run_thinking_led_spin(r, g, b, speed=4.0, focus=10.0):
    """
    LLM 생각 중 표시를 위한 비동기 LED 애니메이션 (원형 회전).
    """
    if not strip:
        return
    
    ring_size = 8
    top_offset = 8
    bottom_offset = 16
    start_shift = 4 # 12번 위치로 시작하기 위한 오프셋

    try:
        while True:
            t = time.time() * speed
            
            for i in range(ring_size):
                # 1. 각도 계산
                angle = ((i - start_shift) / ring_size) * 2 * math.pi
                
                # 2. 사인파 계산 (-1 ~ 1)
                wave = math.sin(t + angle)
                
                # 3. 밝기 변환 (0 ~ 1) 및 집중도(Focus) 적용
                brightness = (wave + 1) / 2
                brightness = math.pow(brightness, focus)
                
                # 4. 색상 적용
                cr = int(r * brightness)
                cg = int(g * brightness)
                cb = int(b * brightness)
                
                final_color = Color(cr, cg, cb)
                
                # 위/아래 링 동시 적용 (Batch Update)
                strip.set_pixel_color(top_offset + i, final_color)
                strip.set_pixel_color(bottom_offset + i, final_color)
            
            # 5. 한 번에 출력 (Efficient)
            strip.show()
            
            # 6. 비동기 대기 (Non-blocking)
            await asyncio.sleep(0.02) # 약 50 FPS
            
    except asyncio.CancelledError:
        # 태스크 취소 시 LED 끄기
        if strip:
            strip.clear()
            strip.show()
        raise

# ==================================================================================
# TTS 관련
# ==================================================================================

async def save_tts_to_file(response_text: str, client: AsyncOpenAI, filename: str = "output.mp3"):
    """텍스트를 받아 TTS 오디오 파일로 저장"""
    try:
        tts_start_time = time.time()
        # 파일 저장 경로의 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        async with client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL,
            voice=VOICE,
            input=response_text,
            # instructions="Speak in a positive tone.",
            response_format="wav"
        ) as tts_response:
            
            logging.info(f"💾 TTS 파일 저장 시작: {filename}")
            
            with open(filename, "wb") as f:
                async for audio_chunk in tts_response.iter_bytes(chunk_size=4096):
                    if audio_chunk:
                        f.write(audio_chunk)
        
        logging.info(f"✅ TTS 파일 '{filename}' 저장 완료 (소요시간: {time.time() - tts_start_time:.2f}초)")

    except asyncio.CancelledError:
        logging.info("🛑 TTS 처리가 중단되었습니다.")
        # 파일이 쓰다 만 상태라면 삭제하는 로직이 필요할 수 있습니다.
        if os.path.exists(filename):
            os.remove(filename)
        raise
    except Exception as e:
        logging.error(f"❌ TTS 저장 중 오류 발생: {e}")

async def handle_tts_stream(response_stream, client: AsyncOpenAI, websocket, conversation_log: List[Dict[str, Any]], responses_start_time=None):
    """(사용되지 않음 - 참고용) Responses API의 텍스트 스트림을 받아 TTS 오디오 스트림으로 변환 후 전송"""
    await websocket.send(json.dumps({"type": "responses_stream_start"}))
    
    full_response_text = ""
    sentence_buffer = ""
    try:
        async for event in response_stream:
            if event.type == "response.output_text.delta":
                text_chunk = event.delta
                sentence_buffer += text_chunk
                full_response_text += text_chunk
                
                if any(p in sentence_buffer for p in ".?!\n"):
                    async with client.audio.speech.with_streaming_response.create(
                        model=TTS_MODEL, voice=VOICE, input=sentence_buffer, response_format="pcm"
                    ) as tts_response:
                        async for audio_chunk in tts_response.iter_bytes(chunk_size=4096):
                            await websocket.send(json.dumps({
                                "type": "responses_audio_chunk", 
                                "data": base64.b64encode(audio_chunk).decode('utf-8')
                            }))
                    sentence_buffer = ""
            
            if event.type == "response.completed":
                message = f"(소요시간: {time.time() - responses_start_time:.2f}초)" if responses_start_time else ""
                logger.info(f"OpenAI 응답 완료: '{full_response_text}' {message}")

        if sentence_buffer.strip():
            async with client.audio.speech.with_streaming_response.create(
                model=TTS_MODEL, voice=VOICE, input=sentence_buffer, response_format="pcm"
            ) as tts_response:
                async for audio_chunk in tts_response.iter_bytes(chunk_size=4096):
                    await websocket.send(json.dumps({
                        "type": "responses_audio_chunk", 
                        "data": base64.b64encode(audio_chunk).decode('utf-8')
                    }))

    except asyncio.CancelledError:
        logger.info("TTS 스트림 처리가 중단되었습니다.")
        raise
    finally:
        await websocket.send(json.dumps({"type": "responses_stream_end"}))
        if full_response_text:
            conversation_log.append({"role": "assistant", "content": full_response_text})

async def handle_tts_oneshot(response_text: str, client: AsyncOpenAI, websocket, tts_start_event: asyncio.Event):
    """전체 텍스트를 받아 한 번에 TTS 처리하는 함수"""
    try:
        tts_streaming_start_time = time.time()

        await websocket.send(json.dumps({"type": "responses_only"}))
        async with client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL, voice=VOICE, input=response_text, response_format="pcm"
        ) as tts_response:
            first_chunk = True
            async for audio_chunk in tts_response.iter_bytes(chunk_size=4096):
                if first_chunk:
                    first_chunk = False
                    tts_start_event.set()
                    logger.info(f"TTS 스트리밍 시작... (소요시간: {time.time() - tts_streaming_start_time:.2f}초)")
                await websocket.send(json.dumps({
                    "type": "responses_audio_chunk",
                    "data": base64.b64encode(audio_chunk).decode('utf-8')
                }))
        logger.info(f"TTS 스트리밍 완료 (소요시간: {time.time() - tts_streaming_start_time:.2f}초)")
    except asyncio.CancelledError:
        logger.info("TTS 처리가 중단되었습니다.")
        raise
    finally:
        await websocket.send(json.dumps({"type": "responses_stream_end"}))
        if not tts_start_event.is_set():
            tts_start_event.set()


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
# LLM API Pipeline
# ==================================================================================

async def run_responses_task(openai_client: AsyncOpenAI, manager: ConversationManager):
    """Responses API를 호출하여 텍스트 응답과 수행할 액션을 반환합니다."""
    logger.info("🧠 Responses Task 시작...")
    responses_start_time = time.time()
    current_log = manager.get_current_log()

    response_text = ""
    music_action = None

    try:
        tools = [
            {
                "type": "web_search",
                "user_location": {"type": "approximate", "country": "KR"},
            },
            {
                "type": "function",
                "name": "play_music",
                "description": "사용자가 요청한 조건에 맞는 노래를 검색하여 재생합니다. 저장된 DB에 있는 노래만 재생 가능합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "song_title": {
                            "type": "string",
                            "description": "사용자가 요청한 노래 제목 (예: 밤편지)"
                        },
                        "artist_name": {
                            "type": "string",
                            "description": "사용자가 요청한 가수 이름 (예: 아이유)"
                        },
                    },
                    "required": ["song_title", "artist_name"] 
                }
            }
        ]

        params = {
            **RESPONSES_PRESETS.get(RESPONSES_MODEL, {}),
            "input": current_log,
            "tools": tools,
        }
        response = await openai_client.responses.create(**params)

        for item in response.output:
            if item.type == "function_call":
                logger.info(f"🧠 Function call: {item.name}")
                if item.name == "play_music":
                    args = json.loads(item.arguments)
                    song_title = args.get("song_title", "")
                    artist_name = args.get("artist_name", "")
                    file_path, message = play_music(song_title, artist_name)

                    if file_path:
                        audio_name = f"{song_title}_{artist_name}"
                        # assets/headMotion 폴더에 audio_name.csv 파일이 있는지 확인
                        if not os.path.exists(os.path.join(ASSETS_DIR, "headMotion", f"{audio_name}.csv")):
                            await asyncio.to_thread(offline_motion_generation, audio_name)
                        music_action = {"song_title": song_title, "artist_name": artist_name}
                    
                    # 함수 호출 결과 추가 및 재요청
                    current_log_copy = current_log.copy()
                    current_log_copy.append(item)
                    current_log_copy.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps({
                            "status": "success" if file_path else "failure",
                            "message": message
                        })
                    })

                    params = {
                        **RESPONSES_PRESETS.get(RESPONSES_MODEL, {}),
                        "input": current_log_copy,
                        "tools": tools,
                    }
                    response = await openai_client.responses.create(**params)
                    response_text = response.output[0].content[0].text.strip()
                    break

            elif item.type == "message":
                response_text = item.content[0].text.strip()
                break

        logger.info(f"🧠 답변 생성 완료: '{response_text}' (소요시간: {time.time() - responses_start_time:.2f}초)")
        return response_text, music_action

    except Exception as e:
        logger.error(f"🧠 Responses Task 실행 중 오류: {e}", exc_info=True)
        return None, None

async def run_tts_action_task(websocket, openai_client: AsyncOpenAI, response_text: str, music_action: dict, tts_start_event: asyncio.Event):
    """텍스트를 받아 TTS를 스트리밍하고 액션을 수행합니다. (취소 가능)"""
    try:
        # TTS 스트리밍
        if response_text:
            await handle_tts_oneshot(response_text, openai_client, websocket, tts_start_event)
        else:
            # 막히지 않도록 이벤트 설정
            tts_start_event.set()

        # 음악 재생 액션
        if music_action:
            song_title = music_action['song_title']
            artist_name = music_action['artist_name']
            await websocket.send(json.dumps({"type": "play_audio_csv", "audio_name": f"{song_title}_{artist_name}"}))

    except asyncio.CancelledError:
        logger.info("🔇 TTS/Action Task가 새 입력에 의해 중단되었습니다.")
        # TTS 중단 시그널 전송 등 추가 가능
    except Exception as e:
        logger.error(f"TTS/Action Task 실행 중 오류: {e}", exc_info=True)
        if not tts_start_event.is_set():
            tts_start_event.set()

async def unified_active_pipeline(websocket, openai_client: AsyncOpenAI, manager: ConversationManager):
    """사용자 입력에 대해 Responses API를 호출하여 답변을 생성하고 TTS 및 액션을 수행하는 통합 파이프라인"""
    logger.info("🤖 Unified Active Pipeline 시작...")

    current_tts_task = None
    stt_result_queue = asyncio.Queue()
    main_loop = asyncio.get_running_loop()

    try:
        with AudioProcessor(stt_result_queue, main_loop, websocket, config=AUDIO_CONFIG) as audio_processor:
            while True:
                try:
                    # 1. 사용자 입력 대기 (TTS가 실행 중이어도 듣고 있음)
                    user_text = await asyncio.wait_for(stt_result_queue.get(), timeout=ACTIVE_SESSION_TIMEOUT)

                    # 2. 입력 감지 시 말하고 있던 TTS 취소
                    if current_tts_task and not current_tts_task.done():
                        logger.info(f"사용자 인터럽션 감지: '{user_text}'. 이전 발화(TTS)를 중단합니다.")
                        current_tts_task.cancel()
                        await asyncio.gather(current_tts_task, return_exceptions=True)
                        current_tts_task = None

                    if any(kw in user_text for kw in END_KEYWORDS):
                        await websocket.send(json.dumps({"type": "play_audio", "file_to_play": str(SLEEP_FILE)}))
                        logger.info(f"종료 키워드 감지: '{user_text}' - 세션을 종료합니다.")
                        break

                    manager.add_message("user", user_text)

                    # 3. 답변 생성
                    # VAD/STT 일시 중단
                    audio_processor.pause_processing()

                    # LED로 생각 중 표시
                    thinking_led_task = asyncio.create_task(run_thinking_led_spin(50, 50, 233, speed=4.0, focus=10.0))

                    # LLM 응답 생성 (Block)
                    response_text, music_action = await run_responses_task(openai_client, manager)
                    

                    # 4. TTS 및 액션 수행 (취소 가능)
                    if response_text:
                        manager.add_message("assistant", response_text)
                        
                        tts_start_event = asyncio.Event()

                        current_tts_task = asyncio.create_task(
                            run_tts_action_task(websocket, openai_client, response_text, music_action, tts_start_event)
                        )

                        # TTS가 실제로 시작될 때까지 대기 (최대 5초)
                        try:
                            await asyncio.wait_for(tts_start_event.wait(), timeout=5.0)
                        except asyncio.TimeoutError:
                            logger.warning("⚠️ TTS 시작 이벤트 타임아웃. 그냥 진행합니다.")
                     
                        # "답변 생성 중 ~ TTS 준비 중" 사이에 들어온 모든 입력 삭제
                        ignored_count = 0
                        while not stt_result_queue.empty():
                            try:
                                stt_result_queue.get_nowait()
                                ignored_count += 1
                            except asyncio.QueueEmpty:
                                break
                        if ignored_count > 0:
                            logger.info(f"🧹 TTS 시작 전 들어온 {ignored_count}개의 입력을 무시했습니다.")
                        
                        # 로딩 LED 끄기
                        if not thinking_led_task.done():
                            thinking_led_task.cancel()
                            await asyncio.gather(thinking_led_task, return_exceptions=True)
                        led_set_ring(50, 50, 233)  # 답변 준비 완료 표시

                        # VAD/STT 재개
                        audio_processor.resume_processing()
                    else:
                        # 답변이 없는 경우 (오류 등) 바로 듣기 재개
                        audio_processor.resume_processing()
                    
                except asyncio.TimeoutError:
                    logger.info(f"⏰ {ACTIVE_SESSION_TIMEOUT}초 동안 입력이 없어 Active 세션을 종료합니다.")
                    await websocket.send(json.dumps({"type": "play_audio", "file_to_play": str(SLEEP_FILE)}))
                    break
    except Exception as e:
        logger.error(f"Unified Active Pipeline에서 오류 발생: {e}", exc_info=True)
    finally:
        if current_tts_task and not current_tts_task.done():
            current_tts_task.cancel()
        logger.info("🤖 Unified Active Pipeline 종료.")

# ==================================================================================
# Sleep 모드
# ==================================================================================

async def wakeword_detection_loop(websocket):
    """START_KEYWORD를 감지할 때까지 VAD-STT 루프를 실행 (Sleep 모드)"""
    logger.info(f"💤 Sleep 모드 시작. '{START_KEYWORD}' 호출 대기 중...")
    keyword_queue = asyncio.Queue()
    main_loop = asyncio.get_running_loop()

    try:
        with AudioProcessor(keyword_queue, main_loop, websocket, config=AUDIO_CONFIG) as audio_processor:
            while True:
                stt_result = await keyword_queue.get()
                logger.info(f"[Sleep Mode] STT 결과: {stt_result}")
                if START_KEYWORD in stt_result:
                    await websocket.send(json.dumps({"type": "play_audio", "file_to_play": str(AWAKE_FILE)}))
                    return
    except Exception as e:
        logger.error(f"Wakeword detection loop에서 오류 발생: {e}", exc_info=True)
    finally:
        logger.info("💤 Sleep 모드 종료.")