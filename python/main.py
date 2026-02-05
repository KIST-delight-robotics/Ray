import asyncio
import logging
import socket
import websockets
import json
import signal
import sys
import atexit
from openai import AsyncOpenAI

from config import OPENAI_API_KEY, AWAKE_FILE, SLEEP_FILE, AWAKE_FILE_SCRIPT, SLEEP_FILE_SCRIPT, RAG_PERSIST_DIR
from prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_RESP_ONLY
from conversation_manager import ConversationManager
from api_pipeline import save_tts_to_file
from led import led_set_ring, led_set_bar, led_clear
from state_manager import ConversationEngine
from rag import init_db

# 전역 엔진 변수 (Listening Loop에서 접근)
conversation_engine = None

# 종료 처리 함수
def shutdown_handler(signum=None, frame=None):
    """프로그램 종료 시 실행될 정리 함수"""
    logging.info(f"종료 신호 감지: {signum if signum else 'Normal Exit'}. 정리 작업을 시작합니다.")
    led_clear()
    # 이미 종료 중이 아니라면 강제 종료
    if signum is not None:
        sys.exit(0)

atexit.register(shutdown_handler) # 정상 종료시 실행
signal.signal(signal.SIGTERM, shutdown_handler) # kill 신호시 실행


async def main_logic_loop(websocket):
    global conversation_engine
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    if not AWAKE_FILE.exists():
        logging.info(f"음성 파일 생성 중: {AWAKE_FILE}")
        await save_tts_to_file(AWAKE_FILE_SCRIPT, openai_client, AWAKE_FILE)

    if not SLEEP_FILE.exists():
        logging.info(f"음성 파일 생성 중: {SLEEP_FILE}")
        await save_tts_to_file(SLEEP_FILE_SCRIPT, openai_client, SLEEP_FILE)
    
    # RAG DB 초기화
    try:
        logging.info("📚 RAG DB 초기화 중...")
        init_db(str(RAG_PERSIST_DIR), OPENAI_API_KEY)
        logging.info("✅ RAG DB 준비 완료!")
    except Exception as e:
        logging.warning(f"⚠️ RAG DB 초기화 실패 (RAG 기능 비활성화): {e}")
        
    try:
        logging.info("🚀 StateMachine 엔진 초기화 및 시작")
        conversation_engine = ConversationEngine(websocket, asyncio.get_running_loop())
        await conversation_engine.start()

    except (asyncio.CancelledError, SystemExit, KeyboardInterrupt):
        logging.info("메인 로직: 종료 신호를 감지하여 루프를 멈춥니다.")
        if conversation_engine:
            conversation_engine.stop()
        return
    
    except Exception as e:
        logging.error(f"메인 로직 루프 에러: {e}", exc_info=True)
        if conversation_engine:
            conversation_engine.stop()

async def background_listener(websocket):
    """백그라운드에서 클라이언트로부터 메시지를 수신합니다."""
    global conversation_engine
    try:
        async for message in websocket:
            data = json.loads(message)
            
            # C++ 로봇 동작 완료 신호 ("speaking_finished")
            if data.get("type") == "speaking_finished":
                logging.info("Signal: speaking_finished received from C++")
                if conversation_engine:
                    conversation_engine.on_robot_finished()
                continue

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

async def main():
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%H:%M:%S',
        force=True
    )
    
    # 웹소켓 서버 시작
    server = await websockets.serve(chat_handler, "127.0.0.1", 5000, family=socket.AF_INET)
    logging.info("🚀 통합 WebSocket 서버가 127.0.0.1:5000 에서 시작되었습니다.")
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        shutdown_handler()