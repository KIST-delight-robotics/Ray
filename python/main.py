import asyncio
import logging
import websockets
from openai import AsyncOpenAI

from config import OPENAI_API_KEY
from prompts import SYSTEM_PROMPT
from conversation_manager import ConversationManager
from api_pipeline import unified_active_pipeline, wakeword_detection_loop

async def chat_handler(websocket):
    """웹소켓 클라이언트 연결을 처리하고 전체 대화 사이클을 관리합니다."""
    logging.info(f"✅ C++ 클라이언트 연결됨: {websocket.remote_address}")
    
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    conversation_manager = ConversationManager(client=openai_client)

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

    except websockets.exceptions.ConnectionClosed:
        logging.warning(f"🔌 C++ 클라이언트 연결 종료됨: {websocket.remote_address}")
    except Exception as e:
        logging.error(f"Chat 핸들러에서 예외 발생: {e}", exc_info=True)
    finally:
        logging.info(f"🔌 C++ 클라이언트 연결 핸들러 종료: {websocket.remote_address}")

async def main():
    """서버를 시작하고 로깅을 설정합니다."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    server = await websockets.serve(chat_handler, "127.0.0.1", 5000)
    logging.info("🚀 통합 WebSocket 서버가 127.0.0.1:5000 에서 시작되었습니다.")
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("서버를 종료합니다.")