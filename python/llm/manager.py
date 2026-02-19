"""LLM 응답 생성 매니저. 별도 스레드에서 OpenAI API를 호출합니다."""

import time
import queue
import logging
import threading

from config import create_openai_client, RESPONSES_MODEL, RESPONSES_PRESETS
from llm.tools import TOOL_SCHEMAS, handle_play_music, handle_consult_archive

logger = logging.getLogger(__name__)


class LLMManager:
    def __init__(self, conversation_manager):
        self.client = create_openai_client()
        self.history_manager = conversation_manager

        self.response_queue = queue.Queue()

        self._thread = None
        self._stop_event = threading.Event()
        self._id_lock = threading.Lock()
        self._active_turn_id = 0

    def request_generation(self, user_text: str, turn_id: int):
        """ThinkingState에서 호출: 답변 생성 요청"""
        self._stop_event.clear()
        with self._id_lock:
            self._active_turn_id = turn_id

        _drain_queue(self.response_queue)

        self._thread = threading.Thread(
            target=self._run_generation,
            args=(user_text, turn_id),
            name=f"LLMThread-{turn_id}",
            daemon=True
        )
        self._thread.start()

    def cancel(self):
        """인터럽션 발생 시 호출: 작업 취소"""
        self._stop_event.set()

    def _is_cancelled(self, turn_id: int) -> bool:
        return self._stop_event.is_set() or self._active_turn_id != turn_id

    def _run_generation(self, user_text: str, turn_id: int):
        try:
            llm_start_time = time.time()
            if self._is_cancelled(turn_id):
                return

            # 1. 사용자 메시지 기록
            self.history_manager.add_message({"role": "user", "content": user_text, "type": "message"})
            current_log = self.history_manager.get_current_log()

            # 2. Responses API 호출 (1차)
            if self._is_cancelled(turn_id):
                return

            params = {
                **RESPONSES_PRESETS.get(RESPONSES_MODEL, {}),
                "input": current_log,
                "tools": TOOL_SCHEMAS,
            }
            response = self.client.responses.create(**params)

            final_text = ""
            music_action = None

            # 3. 결과 처리
            for item in response.output:
                if self._is_cancelled(turn_id):
                    return

                if item.type == "message":
                    final_text = item.content[0].text.strip()
                    break

                elif item.type == "function_call":
                    logger.info(f"Function call: {item.name}")

                    if item.name == "play_music":
                        result = handle_play_music(item, self.history_manager, current_log)
                        music_action = result["music_action"]

                        # 2차 API 호출 (결과 멘트 생성)
                        response_2 = self.client.responses.create(**params)
                        if response_2.output:
                            for resp_item in response_2.output:
                                if resp_item.type == "message" and resp_item.content:
                                    final_text = resp_item.content[0].text.strip()
                                    break
                        break

                    elif item.name == "consult_archive":
                        temp_log = handle_consult_archive(item, current_log)

                        params_with_context = {
                            **RESPONSES_PRESETS.get(RESPONSES_MODEL, {}),
                            "input": temp_log,
                            "tools": TOOL_SCHEMAS,
                        }
                        response_2 = self.client.responses.create(**params_with_context)

                        if response_2.output:
                            for resp_item in response_2.output:
                                if resp_item.type == "message" and resp_item.content:
                                    final_text = resp_item.content[0].text.strip()
                                    break
                        break

            # 4. 결과 반환
            if self._is_cancelled(turn_id):
                return

            if final_text:
                self.history_manager.add_message({"role": "assistant", "content": final_text, "type": "message"})

            logger.info(f"답변 생성 완료: {final_text} (소요 시간: {time.time() - llm_start_time:.2f}초)")

            result_package = {"text": final_text, "action": music_action}

            if not self._is_cancelled(turn_id):
                self.response_queue.put(result_package)

        except Exception as e:
            logger.error(f"LLM 처리 중 오류: {e}")
            if not self._is_cancelled(turn_id):
                self.response_queue.put(None)

    def request_hesitation(self, turn_id: int):
        """HesitatingState에서 호출: 복구 멘트 생성 요청"""
        self._stop_event.clear()
        with self._id_lock:
            self._active_turn_id = turn_id

        _drain_queue(self.response_queue)

        self._thread = threading.Thread(
            target=self._run_hesitation,
            args=(turn_id,),
            name=f"HesitationLLMThread-{turn_id}",
            daemon=True
        )
        self._thread.start()

    def _run_hesitation(self, turn_id: int):
        try:
            if self._is_cancelled(turn_id):
                return

            current_log = self.history_manager.get_current_log()

            # 임시 로그에 복구 안내 시스템 메시지 추가
            temp_log = current_log.copy()
            temp_log.append({
                "role": "system",
                "content": (
                    "상황: 사용자가 로봇의 말을 끊고 무언가 말하려 했으나, 로봇이 제대로 알아듣지 못했습니다(STT 실패/침묵). 이후 약 3초간 사용자의 추가 발화가 없습니다."
                    "지침: 상황에 맞게, 사용자가 다시 말하도록 자연스럽게 유도하는 짧은 문장을 생성하거나 침묵 상태를 인지하고 적절히 대응하는 문장을 생성하세요."
                    "예시: '죄송해요, 방금 말씀을 놓쳤어요.', '혹시 무언가 말씀을 하셨나요?' '이어서 말해도 될까요?' "
                    "주의: 너무 길지 않게, 간결하고 자연스럽게 상황에 맞게 답변하세요."
                )
            })

            if self._is_cancelled(turn_id):
                return

            params = {
                **RESPONSES_PRESETS.get(RESPONSES_MODEL, {}),
                "model": RESPONSES_MODEL,
                "input": temp_log,
            }
            response = self.client.responses.create(**params)

            final_text = ""
            if response.output:
                for item in response.output:
                    if item.type == "message" and item.content:
                        final_text = item.content[0].text.strip()
                        break

            if not self._stop_event.is_set() and final_text and not self._is_cancelled(turn_id):
                logger.info(f"복구 멘트 생성: {final_text}")
                result_package = {
                    "text": final_text,
                    "action": None,
                    "is_hesitation": True
                }
                self.response_queue.put(result_package)

        except Exception as e:
            logger.error(f"Hesitation LLM 오류: {e}")
            if not self._is_cancelled(turn_id):
                self.response_queue.put(None)


def _drain_queue(q: queue.Queue) -> None:
    """큐의 모든 항목을 비웁니다."""
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break
