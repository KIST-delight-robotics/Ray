import os
import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any

from openai import OpenAI

from config import OUTPUT_LOG_DIR
from prompts import SUMMARY_PROMPT_TEMPLATE

# 모듈 수준 로거 설정
logger = logging.getLogger(__name__)


class ConversationManager:
    """대화 세션의 상태, 기록, 요약을 관리하는 클래스."""

    def __init__(self, openai_api_key):
        """
        ConversationManager를 초기화합니다.

        Args:
            client (OpenAI): OpenAI API와 통신하기 위한 클라이언트.
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.session_id: str | None = None
        self.session_start_time: datetime | None = None
        self.current_conversation_log: List[Dict[str, Any]] = []

    def start_new_session(self, system_prompt: str):
        """
        새로운 대화 세션을 시작하고 초기 컨텍스트를 설정합니다.
        """
        self.session_id = str(uuid.uuid4())
        self.session_start_time = datetime.now()
        self.current_conversation_log = self._create_initial_context(system_prompt, num_recent=10)
        logger.info(f"새로운 세션을 시작합니다. (ID: {self.session_id})")

    def add_message(self, role: str, content: str):
        """
        대화 기록에 새로운 메시지를 추가합니다.

        Args:
            role (str): 메시지 발신자 역할 ('user' 또는 'assistant').
            content (str): 메시지 내용.
        """
        message = {"role": role, "content": content}
        self.current_conversation_log.append(message)

    def get_current_log(self) -> List[Dict[str, Any]]:
        """
        현재 대화 기록을 반환합니다.
        """
        return self.current_conversation_log

    def end_session(self):
        """
        현재 세션을 종료하고, 대화 내용을 요약하여 파일로 저장합니다.
        """
        # 시스템 프롬프트를 제외하고 2개 이상의 메시지가 있어야 유의미한 대화로 간주
        if len(self.current_conversation_log) < 3:
            logger.info("저장할 대화 기록이 충분하지 않아 세션 저장을 건너뜁니다.")
            self._reset_session()
            return

        summary = self._summarize_session()

        session_data = {
            "session_id": self.session_id,
            "start_time": self.session_start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "summary": summary,
            "full_log": self.current_conversation_log
        }

        timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
        filepath = OUTPUT_LOG_DIR / f"{timestamp}_{self.session_id[:8]}.json"

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            logger.info(f"📋 세션 기록 저장 완료: {filepath}")
        except Exception as e:
            logger.error(f"📋 세션 기록 저장 중 오류 발생: {e}")
        finally:
            self._reset_session()

    def _reset_session(self):
        """현재 세션 정보를 초기화합니다."""
        self.session_id = None
        self.session_start_time = None
        self.current_conversation_log = []

    def _summarize_session(self) -> str:
        """OpenAI API를 호출하여 현재 세션의 대화를 요약합니다."""
        log_for_summary = [msg for msg in self.current_conversation_log if msg.get("role") != "system"]
        if not log_for_summary:
            return "요약할 대화 내용이 없습니다."

        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in log_for_summary])
        prompt = SUMMARY_PROMPT_TEMPLATE.format(conversation_text=conversation_text)

        try:
            logger.info("📋 세션 요약 API 호출...")
            responses = self.client.responses.create(
                model="gpt-4.1-mini",
                input=[{"role": "user", "content": prompt}],
            )
            summary = responses.output_text
            logger.info(f"📋 세션 요약 완료:\n{summary}")
            return summary
        except Exception as e:
            logger.error(f"📋 세션 요약 중 오류 발생: {e}")
            return "[오류] 요약 생성에 실패했습니다."

    def _create_initial_context(self, system_prompt: str, num_recent: int = 2) -> List[Dict[str, Any]]:
        """새 세션 시작 시 초기 컨텍스트(시스템 프롬프트 + 이전 대화 요약)를 구성합니다."""
        initial_context = [{"role": "system", "content": system_prompt}]
        
        recent_summaries = self._load_recent_summaries_from_files(num_recent)
        if recent_summaries:
            summary_text = "\n\n---\n\n".join(recent_summaries)
            initial_context.append({
                "role": "system",
                "content": f"## 참고: 과거 대화 요약\n{summary_text}"
            })
            logger.info(f"최근 대화 요약 {len(recent_summaries)}개를 컨텍스트에 추가했습니다.")

        initial_context.append({"role": "system", "content": "[새 대화 시작]"})
        return initial_context

    def _load_recent_summaries_from_files(self, num_to_load: int) -> List[str]:
        """로그 폴더에서 가장 최근의 요약 파일을 찾아 내용을 반환합니다."""
        try:
            history_files = sorted(OUTPUT_LOG_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
            
            summaries = []
            for filepath in history_files[:num_to_load]:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    summary_with_time = f"[{data.get('start_time')[:10]}]\n{data.get('summary', '')}"
                    summaries.append(summary_with_time)
            return summaries
        except Exception as e:
            logger.error(f"최근 요약 로드 중 오류: {e}")
            return []

# ==================================================================================
# 단독 실행 및 테스트를 위한 예제 코드
# ==================================================================================
async def main_test():
    """ConversationManager 클래스의 기능을 테스트하는 메인 함수."""
    
    # --- 로깅 설정 (테스트용) ---
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    # --- OpenAI 클라이언트 초기화 ---
    # config.py에서 API 키를 로드했다고 가정
    from config import OPENAI_API_KEY
    from prompts import SYSTEM_PROMPT
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY가 설정되지 않았습니다. 테스트를 종료합니다.")
        return
        
    
    # --- ConversationManager 사용 ---
    manager = ConversationManager(openai_api_key=OPENAI_API_KEY)

    # 1. 새 세션 시작
    manager.start_new_session(system_prompt=SYSTEM_PROMPT)
    print("\n--- 초기 컨텍스트 ---")
    print(json.dumps(manager.get_current_log(), indent=2, ensure_ascii=False))

    # 2. 대화 시뮬레이션
    manager.add_message("user", "오늘 날씨 어때?")
    manager.add_message("assistant", "오늘은 전국적으로 맑고 화창한 날씨가 예상됩니다! 외출하기 좋은 날이에요.")
    manager.add_message("user", "좋아. 그럼 근처 공원 산책이나 가야겠다. 고마워!")
    
    print("\n--- 대화 기록 추가 후 ---")
    print(json.dumps(manager.get_current_log(), indent=2, ensure_ascii=False))

    # 3. 세션 종료 및 요약/저장
    print("\n--- 세션 종료 중 ---")
    await manager.end_session()

    # 4. 세션 종료 후 상태 확인
    print("\n--- 세션 종료 후 상태 ---")
    print(f"현재 대화 기록: {manager.get_current_log()}")
    print(f"세션 ID: {manager.session_id}")


if __name__ == '__main__':
    import asyncio
    # `output/logs` 디렉토리가 있는지 확인하고 없으면 생성
    if not OUTPUT_LOG_DIR.exists():
        print(f"'{OUTPUT_LOG_DIR}' 디렉토리를 생성합니다.")
        OUTPUT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        
    asyncio.run(main_test())