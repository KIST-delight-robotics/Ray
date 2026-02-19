import os
import json
import uuid
import logging
import threading
from datetime import datetime
from typing import List, Dict, Any

from config import create_openai_client, OUTPUT_LOG_DIR, RESPONSES_PRESETS, SUMMARY_MODEL
from llm.prompts import SUMMARY_PROMPT_TEMPLATE

# 모듈 수준 로거 설정
logger = logging.getLogger(__name__)


class ConversationManager:
    """대화 세션의 상태, 기록, 요약을 관리하는 클래스."""

    def __init__(self):
        """ConversationManager를 초기화합니다."""
        self.client = create_openai_client()
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

    def add_message(self, response_item):
        """
        대화 기록에 새로운 메시지를 추가합니다.
        """
        # OpenAI SDK의 Response 객체를 통째로 받아서 저장
        if hasattr(response_item, 'model_dump'):
            message_dict = response_item.model_dump()
            self.current_conversation_log.append(message_dict)

        elif isinstance(response_item, dict):
            self.current_conversation_log.append(response_item)
        else:
            logger.error(f"저장할 수 없는 메시지 타입입니다: {type(response_item)}")

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
        
        # 1. 세션 정보 스냅샷
        log_snapshot = self.current_conversation_log.copy()
        session_id_snapshot = self.session_id
        start_time_snapshot = self.session_start_time

        # 2. 세션 초기화 (다음 세션 준비)
        self._reset_session()

        # 3. 별도 스레드에서 요약 생성
        threading.Thread(
            target=self._run_background_summary,
            args=(log_snapshot, session_id_snapshot, start_time_snapshot),
            name="SessionSummaryThread",
            daemon=True
        ).start()

    def _run_background_summary(self, log_data, sess_id, start_dt):
        """백그라운드 스레드에서 실행되는 실제 요약 및 저장 로직"""
        try:
            # 요약 생성 (API 호출 - 오래 걸림)
            summary = self._summarize_log(log_data) # 내부 메서드 호출 방식 변경 필요

            session_data = {
                "session_id": sess_id,
                "start_time": start_dt.isoformat(),
                "end_time": datetime.now().isoformat(),
                "summary": summary,
                "full_log": log_data
            }

            timestamp = start_dt.strftime("%Y%m%d_%H%M%S")
            filepath = OUTPUT_LOG_DIR / f"{timestamp}_{sess_id[:8]}.json"

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            logger.info(f"📋 [Background] 세션 기록 저장 완료: {filepath}")
            
        except Exception as e:
            logger.error(f"📋 [Background] 세션 저장 중 오류: {e}")

    def _reset_session(self):
        """현재 세션 정보를 초기화합니다."""
        self.session_id = None
        self.session_start_time = None
        self.current_conversation_log = []

    def _summarize_log(self, log_data: List[Dict[str, Any]]) -> str:
        """주어진 로그 리스트를 기반으로 요약 API를 호출합니다."""
        log_for_summary = [msg for msg in log_data if msg.get("type") == "message"]
        if not log_for_summary:
            return "요약할 대화 내용이 없습니다."

        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in log_for_summary])
        print(conversation_text)
        prompt = SUMMARY_PROMPT_TEMPLATE.format(conversation_text=conversation_text)

        try:
            logger.info("📋 [Thread] 세션 요약 API 호출...")
            
            params = {
                **RESPONSES_PRESETS.get(SUMMARY_MODEL, {}),
                "model": SUMMARY_MODEL,
                "input": prompt,
            }
            response = self.client.responses.create(**params)

            summary = ""
            if response.output:
                for item in response.output:
                    if item.type == "message" and item.content:
                        summary = item.content[0].text.strip()
                        break

            logger.info(f"📋 [Thread] 세션 요약 완료:\n{summary}")
            return summary
        except Exception as e:
            logger.error(f"📋 [Thread] 세션 요약 중 오류 발생: {e}")
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
        """로그 폴더에서 가장 최근의 요약 파일을 찾아 내용을 반환합니다. (오류 파일 건너뜀)"""
        # 전체 try-except 제거 -> 개별 파일 try-except로 변경
        
        history_files = sorted(OUTPUT_LOG_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
        
        summaries = []
        
        # 파일 목록을 순회하며 정상적인 파일만 골라냄
        for filepath in history_files:
            # 목표 개수를 채웠으면 중단
            if len(summaries) >= num_to_load:
                break

            try:
                # 파일 크기가 0이면 건너뜀 (생성 직후 상태)
                if os.path.getsize(filepath) == 0:
                    continue

                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 필수 키가 없으면 건너뜀 (잘못된 형식)
                    if "summary" not in data or "start_time" not in data:
                        continue
                        
                    summary_with_time = f"[{data.get('start_time')[:10]}]\n{data.get('summary', '')}"
                    summaries.append(summary_with_time)

            except (json.JSONDecodeError, OSError) as e:
                # 특정 파일 읽기 실패 시 로그만 남기고 다음 파일 시도
                logger.warning(f"⚠️ 이전 기록 로드 실패(건너뜀) - {filepath.name}: {e}")
                continue
            except Exception as e:
                logger.error(f"⚠️ 알 수 없는 오류(건너뜀) - {filepath.name}: {e}")
                continue

        return summaries

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

    from llm.prompts import SYSTEM_PROMPT

    # --- ConversationManager 사용 ---
    manager = ConversationManager()

    # 1. 새 세션 시작
    manager.start_new_session(system_prompt=SYSTEM_PROMPT)
    print("\n--- 초기 컨텍스트 ---")
    print(json.dumps(manager.get_current_log(), indent=2, ensure_ascii=False))

    # 2. 대화 시뮬레이션
    manager.add_message({"role": "user", "content": "오늘 날씨 어때?"})
    manager.add_message({"role": "assistant", "content": "오늘은 전국적으로 맑고 화창한 날씨가 예상됩니다! 외출하기 좋은 날이에요."})
    manager.add_message({"role": "user", "content": "좋아. 그럼 근처 공원 산책이나 가야겠다. 고마워!"})
    
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