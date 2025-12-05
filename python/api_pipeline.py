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
    TTS_MODEL, VOICE, REALTIME_MODEL, RESPONSES_MODEL, AUDIO_CONFIG
)
from prompts import REALTIME_PROMPT
from audio_processor import AudioProcessor
from conversation_manager import ConversationManager

logger = logging.getLogger(__name__)

# ==================================================================================
# TTS 관련
# ==================================================================================

async def save_tts_to_file(response_text: str, client: AsyncOpenAI, filename: str = "output.mp3"):
    """텍스트를 받아 TTS 오디오 파일로 저장"""
    try:
        async with client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL,
            voice=VOICE,
            input=response_text,
            instructions="Speak in a positive tone.",
            response_format="wav" 
        ) as tts_response:
            
            logging.info(f"💾 TTS 파일 저장 시작: {filename}")
            
            with open(filename, "wb") as f:
                async for audio_chunk in tts_response.iter_bytes(chunk_size=4096):
                    if audio_chunk:
                        f.write(audio_chunk)
                        
        logging.info("✅ TTS 파일 저장 완료")

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

async def handle_tts_oneshot(response_text: str, client: AsyncOpenAI, websocket, realtime_start_event: asyncio.Event):
    """전체 텍스트를 받아 한 번에 TTS 처리하는 함수"""
    try:
        if realtime_start_event.is_set():
            await websocket.send(json.dumps({"type": "responses_stream_start"}))
        else:
            await websocket.send(json.dumps({"type": "responses_only"}))
        async with client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL, voice=VOICE, input=response_text, response_format="pcm"
        ) as tts_response:
            async for audio_chunk in tts_response.iter_bytes(chunk_size=4096):
                await websocket.send(json.dumps({
                    "type": "responses_audio_chunk", 
                    "data": base64.b64encode(audio_chunk).decode('utf-8')
                }))
    except asyncio.CancelledError:
        logger.info("TTS 처리가 중단되었습니다.")
        raise
    finally:
        await websocket.send(json.dumps({"type": "responses_stream_end"}))

# ==================================================================================
# Unified API Pipeline (Realtime + Responses)
# ==================================================================================

async def run_realtime_task(websocket, realtime_connection, item_ids_to_manage: list, user_text: str, realtime_start_event: asyncio.Event):
    """(Task 1) Realtime API를 호출하고 오디오를 스트리밍합니다."""
    logger.info("⚡️ Realtime Task 시작...")
    try:
        if item_ids_to_manage:
            logger.info(f"이전 Realtime 대화 아이템 {len(item_ids_to_manage)}개 삭제 중...")
            delete_tasks = [realtime_connection.conversation.item.delete(item_id=item_id) for item_id in item_ids_to_manage]
            await asyncio.gather(*delete_tasks, return_exceptions=True)
            item_ids_to_manage.clear()

        await realtime_connection.session.update(session={"type": "realtime", "instructions": REALTIME_PROMPT, "audio": {"output": {"voice": VOICE}}})
        await realtime_connection.conversation.item.create(
            item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": user_text}]}
        )

        realtime_start_time = time.time()
        await realtime_connection.response.create()

        with wave.open("output/audio/realtime.wav", "wb") as wf:
            wf.setnchannels(AUDIO_CONFIG['CHANNELS'])
            wf.setsampwidth(2)
            wf.setframerate(AUDIO_CONFIG['SAMPLE_RATE'])

            async for event in realtime_connection:

                if event.type == "conversation.item.added":
                    item_ids_to_manage.append(event.item.id)

                elif event.type == "response.output_audio.delta":
                    await websocket.send(json.dumps({"type": "realtime_audio_chunk", "data": event.delta}))
                    bytes_data = base64.b64decode(event.delta)
                    wf.writeframes(bytes_data)

                elif event.type == "response.created":
                    realtime_start_event.set()
                    await websocket.send(json.dumps({"type": "realtime_stream_start"}))

                elif event.type == "response.done":
                    await websocket.send(json.dumps({"type": "realtime_stream_end"}))
                    transcript = event.response.output[0].content[0].transcript if event.response.output[0].content[0].type != "text" else "[Realtime 응답 없음]"
                    logger.info(f"⚡️ Realtime API 답변 완료: '{transcript}' (소요시간: {time.time() - realtime_start_time:.2f}초)")
                    wf.close()
                    break

                elif event.type == "error":
                    logger.error(f"Realtime API 오류 이벤트: {event}")
    
    except asyncio.CancelledError:
        logger.info("⚡️ Realtime Task가 외부에서 중단되었습니다.")
    except Exception as e:
        logger.error(f"⚡️ Realtime Task 실행 중 오류: {e}", exc_info=True)
    finally:
        logger.info("⚡️ Realtime Task 종료.")

async def run_responses_task(websocket, openai_client: AsyncOpenAI, manager: ConversationManager, realtime_start_event: asyncio.Event):
    """(Task 2) Responses API를 호출하고, TTS를 스트리밍합니다."""
    logger.info("🧠 Responses Task 시작...")
    responses_start_time = time.time()
    current_log = manager.get_current_log()

    try:
        response = await openai_client.responses.create(
            model=RESPONSES_MODEL,
            input=current_log,
            tools=[
                {
                    "type": "web_search",
                    "user_location": {
                        "type": "approximate",
                        "country": "KR",
                    }
                }
            ],
            reasoning={"effort": "none"},
            text = {"verbosity": "low"},
        )
        # logging.info(f"🧠 Responses Query: \n{response}")
        # response_id = response.id

        # response_item = await openai_client.responses.input_items.list(response_id)
        # print(response_item.data)

        response_text = response.output_text.strip()
        logger.info(f"🧠 Responses API 답변 생성 완료: '{response_text}' (소요시간: {time.time() - responses_start_time:.2f}초)")

        await handle_tts_oneshot(response_text, openai_client, websocket, realtime_start_event)
        manager.add_message("assistant", response_text)

    except asyncio.CancelledError:
        logger.info("🧠 Responses Task가 외부에서 중단되었습니다.")
    except Exception as e:
        logger.error(f"🧠 Responses Task 실행 중 오류: {e}", exc_info=True)
    finally:
        logger.info("🧠 Responses Task 종료.")

async def unified_active_pipeline(websocket, openai_client: AsyncOpenAI, manager: ConversationManager):
    """사용자 입력에 대해 Realtime API와 Responses API를 동시에 호출하여 순차적으로 응답하는 통합 파이프라인"""
    logger.info("🤖 Unified Active Pipeline 시작...")
    active_response_tasks = []
    stt_result_queue = asyncio.Queue()
    main_loop = asyncio.get_running_loop()

    async with openai_client.realtime.connect(model=REALTIME_MODEL) as realtime_connection:
        realtime_item_ids_to_manage = []
        try:
            with AudioProcessor(stt_result_queue, main_loop, websocket, config=AUDIO_CONFIG) as audio_processor:
                while True:
                    try:
                        user_text = await asyncio.wait_for(stt_result_queue.get(), timeout=ACTIVE_SESSION_TIMEOUT)
                        realtime_start_event = asyncio.Event()

                        if active_response_tasks:
                            logger.info(f"사용자 인터럽션 감지: '{user_text}'. 이전 응답 태스크를 중단합니다.")
                            for task in active_response_tasks: task.cancel()
                            await asyncio.gather(*active_response_tasks, return_exceptions=True)
                            active_response_tasks = []

                        if any(kw in user_text for kw in END_KEYWORDS):
                            await websocket.send(json.dumps({"type": "play_audio", "file_to_play": str(SLEEP_FILE)}))
                            logger.info(f"종료 키워드 감지: '{user_text}' - 세션을 종료합니다.")
                            break

                        manager.add_message("user", user_text)

                        realtime_task = asyncio.create_task(
                            run_realtime_task(websocket, realtime_connection, realtime_item_ids_to_manage, user_text, realtime_start_event)
                        )
                        responses_task = asyncio.create_task(
                            run_responses_task(websocket, openai_client, manager, realtime_start_event)
                        )
                        active_response_tasks = [responses_task, realtime_task]
                        
                    except asyncio.TimeoutError:
                        logger.info(f"⏰ {ACTIVE_SESSION_TIMEOUT}초 동안 입력이 없어 Active 세션을 종료합니다.")
                        await websocket.send(json.dumps({"type": "play_audio", "file_to_play": str(SLEEP_FILE)}))
                        break
        except Exception as e:
            logger.error(f"Unified Active Pipeline에서 오류 발생: {e}", exc_info=True)
        finally:
            if active_response_tasks:
                for task in active_response_tasks: task.cancel()
                await asyncio.gather(*active_response_tasks, return_exceptions=True)
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
                # 테스트용 코드
                # await asyncio.sleep(1)
                # await websocket.send(json.dumps({"type": "play_audio", "file_to_play": "test_audio.wav"}))
                # await websocket.send(json.dumps({"type": "play_music", "title": "가까운 듯 먼 그대여", "artist": "카더가든"}))
                # return
    except Exception as e:
        logger.error(f"Wakeword detection loop에서 오류 발생: {e}", exc_info=True)
    finally:
        logger.info("💤 Sleep 모드 종료.")