"""LLM 도구(Function Calling) 스키마 정의 및 핸들러."""

import os
import json
import logging
import threading

from tools.music import play_music
from tools.offline_motion import offline_motion_generation
from rag import search_archive
from config import ASSETS_DIR, RAG_TOP_K

logger = logging.getLogger(__name__)

# ==================================================================================
# 도구 스키마 정의
# ==================================================================================

TOOL_SCHEMAS = [
    {
        "type": "web_search",
        "user_location": {"type": "approximate", "country": "KR"},
    },
    {
        "type": "function",
        "name": "play_music",
        "description": "사용자가 요청한 노래를 검색하여 재생합니다. 저장된 DB에 있는 노래만 재생 가능합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "song_title": {"type": "string"},
                "artist_name": {"type": "string"},
            },
            "required": ["song_title", "artist_name"]
        }
    },
    {
        "type": "function",
        "name": "consult_archive",
        "description": "영화/음악에 대한 정보를 찾거나, 사용자의 기분/상황에 맞는 작품을 연상할 때 사용합니다. 사실 확인, 위로, 공감, 추천이 필요할 때 적극적으로 사용하세요.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색할 키워드 또는 문장 (예: '비 오는 날의 우울함', '헤어질 결심 해석')"
                },
                "intent": {
                    "type": "string",
                    "enum": ["fact", "vibe", "critique"],
                    "description": "fact=사실정보(감독/출연진), vibe=분위기/추천, critique=평론/해석"
                }
            },
            "required": ["query", "intent"]
        }
    }
]


# ==================================================================================
# 도구 핸들러
# ==================================================================================

def handle_play_music(item, history_manager, current_log) -> dict:
    """
    play_music 도구 호출을 처리합니다.

    Returns:
        {
            "function_call_output": dict,  # 2차 API 호출에 사용할 결과
            "music_action": dict | None,   # 음악 재생 정보 (없으면 None)
        }
    """
    history_manager.add_message(item)
    args = json.loads(item.arguments)
    song_title = args.get("song_title", "")
    artist_name = args.get("artist_name", "")

    file_path, message = play_music(song_title, artist_name)
    status = "failure"
    music_action = None
    motion_thread = None

    if file_path:
        status = "success"
        audio_name = f"{song_title}_{artist_name}"
        csv_path = os.path.join(ASSETS_DIR, "headMotion", f"{audio_name}.csv")

        # 모션 파일이 없으면 별도 스레드에서 생성
        if not os.path.exists(csv_path):
            logger.info(f"모션 파일 없음. 생성 시작: {audio_name}")
            motion_thread = threading.Thread(
                target=offline_motion_generation,
                args=(audio_name,),
                name=f"MotionGenThread-{audio_name}"
            )
            motion_thread.start()

        music_action = {"audio_name": audio_name, "motion_thread": motion_thread}

    function_call_output = {
        "type": "function_call_output",
        "call_id": item.call_id,
        "output": json.dumps({"status": status, "message": message})
    }
    history_manager.add_message(function_call_output)

    return {
        "function_call_output": function_call_output,
        "music_action": music_action,
    }


def handle_consult_archive(item, current_log) -> list:
    """
    consult_archive 도구 호출을 처리합니다.
    RAG 검색을 수행하고, 2차 API 호출에 사용할 임시 로그를 반환합니다.

    Returns:
        임시 대화 로그 (list) - 원본 로그 + function_call + function_call_output
    """
    args = json.loads(item.arguments)
    query = args.get("query", "")
    intent = args.get("intent", "vibe")

    logger.info(f"RAG 검색: query='{query}', intent='{intent}'")

    search_result = search_archive(query, intent, top_k=RAG_TOP_K)

    # 임시 로그 생성 (휘발성 - history에 저장하지 않음)
    temp_log = current_log.copy()
    temp_log.append({
        "type": "function_call",
        "name": item.name,
        "call_id": item.call_id,
        "arguments": item.arguments
    })
    temp_log.append({
        "type": "function_call_output",
        "call_id": item.call_id,
        "output": search_result
    })

    return temp_log
