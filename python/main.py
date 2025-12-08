import asyncio
import logging
import socket
import websockets
import json
from openai import AsyncOpenAI

from config import OPENAI_API_KEY, AWAKE_FILE, SLEEP_FILE, AWAKE_FILE_SCRIPT, SLEEP_FILE_SCRIPT
from prompts import SYSTEM_PROMPT
from conversation_manager import ConversationManager
from api_pipeline import unified_active_pipeline, wakeword_detection_loop, save_tts_to_file
from led import led_set_ring, led_set_bar, led_clear

async def main_logic_loop(websocket):
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    conversation_manager = ConversationManager(client=openai_client)

    if not AWAKE_FILE.exists():
        logging.info(f"음성 파일 생성 중: {AWAKE_FILE}")
        await save_tts_to_file(AWAKE_FILE_SCRIPT, openai_client, AWAKE_FILE)

    if not SLEEP_FILE.exists():
        logging.info(f"음성 파일 생성 중: {SLEEP_FILE}")
        await save_tts_to_file(SLEEP_FILE_SCRIPT, openai_client, SLEEP_FILE)
        
    try:
        while True:
            # 1. Sleep 모드: 키워드 감지 대기
            await wakeword_detection_loop(websocket)
            
            # 2. 새 세션 시작
            conversation_manager.start_new_session(system_prompt=SYSTEM_PROMPT)
            
            # 3. Active 모드 실행
            await unified_active_pipeline(websocket, openai_client, conversation_manager)

            # 4. Active 모드 종료 후 세션 정리
            await conversation_manager.end_session()

            logging.info("Active 세션 종료. 다시 Sleep 모드로 전환합니다.")
    except Exception as e:
        logging.error(f"메인 로직 루프 에러: {e}", exc_info=True)

async def background_listener(websocket):
    """백그라운드에서 클라이언트로부터 메시지를 수신합니다."""
    try:
        async for message in websocket:
            data = json.loads(message)
            cmd = data.get("cmd")
            if cmd == "led_all":
                r, g, b = data["r"], data["g"], data["b"]
                led_set_ring(r, g, b)
            
            if cmd == "led_ring":
                r, g, b = data["r"], data["g"], data["b"]
                led_set_ring(r, g, b)
                logging.info(f"LED on")
            
            if cmd == "led_bar":
                r, g, b = data["r"], data["g"], data["b"]
                led_set_bar(r, g, b)
    except websockets.exceptions.ConnectionClosed:
        logging.warning("Listener: 연결 종료")
    except Exception as e:
        logging.error(f"Listener 에러: {e}", exc_info=True)

async def chat_handler(websocket):
    """웹소켓 클라이언트 연결을 처리하고 전체 대화 사이클을 관리합니다."""
    logging.info(f"✅ C++ 클라이언트 연결됨: {websocket.remote_address}")
    
    listener_task = asyncio.create_task(background_listener(websocket))
    main_logic_task = asyncio.create_task(main_logic_loop(websocket))

    done, pending = await asyncio.wait(
        [listener_task, main_logic_task],
        return_when=asyncio.FIRST_COMPLETED
    )

    for task in pending:
        task.cancel()
    
    logging.info(f"🔌 C++ 클라이언트 연결 핸들러 종료: {websocket.remote_address}")

    # except websockets.exceptions.ConnectionClosed:
    #     logging.warning(f"🔌 C++ 클라이언트 연결 종료됨: {websocket.remote_address}")
    # except Exception as e:
    #     logging.error(f"Chat 핸들러에서 예외 발생: {e}", exc_info=True)
    # finally:
    #     logging.info(f"🔌 C++ 클라이언트 연결 핸들러 종료: {websocket.remote_address}")

async def main():
    """서버를 시작하고 로깅을 설정합니다."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%H:%M:%S',
        force=True
    )
    
    server = await websockets.serve(chat_handler, "127.0.0.1", 5000, family=socket.AF_INET)
    logging.info("🚀 통합 WebSocket 서버가 127.0.0.1:5000 에서 시작되었습니다.")
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        led_clear()
        logging.info("서버를 종료합니다.")