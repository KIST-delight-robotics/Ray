"""GPT-Live 백엔드 함수 툴 — 정의(Responses API function 스키마)와 핸들러.

대화 모델(gpt-live-1)은 툴을 직접 갖지 못하고 responses 위임 백엔드만 함수를 부른다. 백엔드가 호출하면
:class:`~voice_pipeline.engines.gpt_live.loop.LiveSessionLoop` 가 여기 핸들러를 executor 에서 돌려 결과를
제출한다. 스키마와 그것을 읽는 핸들러를 한 파일에 둬 인자 이름이 어긋나지 않게 한다.

툴:
- ``end_conversation``: 종료. 핸들러는 루프 내장(종료 시퀀스와 얽힘).
- ``search_memory``: 장기기억 에피소드 검색 → :func:`make_memory_search_handler`.
- ``adjust_volume`` / ``set_brightness`` / ``get_device_settings``: 볼륨·밝기
  → :func:`make_device_settings_handlers` (상태·저장은 :mod:`voice_pipeline.device_settings`).
- ``play_song``: 노래 재생 → :func:`make_play_song_tool_def`. ``stop_song`` 은 재생 중에만 백엔드에 주어지는 툴
  (:data:`STOP_SONG_TOOL_DEF`, 루프가 백엔드를 교체할 때 사용). 둘 다 핸들러는 루프 내장.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from voice_pipeline.device_settings import (
    BRIGHTNESS_LEVELS,
    VOLUME_STEPS,
    DeviceSettings,
)
from voice_pipeline.engines.gpt_live.songs import Song, format_song_list
from voice_pipeline.memory.retriever import MemoryRetriever

ToolHandler = Callable[[str], str]  # arguments(JSON 문자열) → output(JSON 문자열)

END_CONVERSATION_TOOL = "end_conversation"
SEARCH_MEMORY_TOOL = "search_memory"

END_CONVERSATION_TOOL_DEF: dict[str, Any] = {
    "type": "function",
    "name": END_CONVERSATION_TOOL,
    "description": (
        "End the current conversation session. Call this when the user says goodbye or clearly wants to stop talking."
    ),
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    "strict": True,
}

SEARCH_MEMORY_TOOL_DEF: dict[str, Any] = {
    "type": "function",
    "name": SEARCH_MEMORY_TOOL,
    "description": (
        "Search Ray's long-term memory of earlier conversations with the user. "
        "Returns episodes (third-person notes with dates) that match the query. "
        "Use it for anything the user told Ray in a past session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "What to look for, in the user's language. "
                    "Include names, topics and time hints from the conversation."
                ),
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": True,
}

ADJUST_VOLUME_TOOL = "adjust_volume"
SET_BRIGHTNESS_TOOL = "set_brightness"
GET_DEVICE_SETTINGS_TOOL = "get_device_settings"

ADJUST_VOLUME_TOOL_DEF: dict[str, Any] = {
    "type": "function",
    "name": ADJUST_VOLUME_TOOL,
    "description": (
        f"Turn Ray's speaker volume up or down by a number of steps ({VOLUME_STEPS} steps total, no mute). "
        "Use steps=1 for an ordinary request and steps=2 when the user asks for a big change. "
        "If the volume is already at the limit, the result says so instead of failing."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["up", "down"], "description": "Louder or quieter."},
            "steps": {
                "type": "integer",
                "description": "How many steps to move. 1 for a normal request, 2 for 'a lot'.",
            },
        },
        "required": ["direction", "steps"],
        "additionalProperties": False,
    },
    "strict": True,
}

SET_BRIGHTNESS_TOOL_DEF: dict[str, Any] = {
    "type": "function",
    "name": SET_BRIGHTNESS_TOOL,
    "description": (
        "Set the brightness of Ray's LED lights to one of the fixed levels. 'off' turns the lights off. "
        "For 'brighter' or 'dimmer' pick the level next to the current one; if the current level is not known "
        "from the conversation, call get_device_settings first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "level": {
                "type": "string",
                "enum": list(BRIGHTNESS_LEVELS),
                "description": "Target brightness level.",
            }
        },
        "required": ["level"],
        "additionalProperties": False,
    },
    "strict": True,
}

GET_DEVICE_SETTINGS_TOOL_DEF: dict[str, Any] = {
    "type": "function",
    "name": GET_DEVICE_SETTINGS_TOOL,
    "description": (
        "Read Ray's current speaker volume step and LED brightness level without changing them. "
        "Use it when the user asks how loud or how bright Ray is, or when a relative change needs the current "
        "level and it is not already known from the conversation."
    ),
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    "strict": True,
}

DEVICE_TOOLS: tuple[dict[str, Any], ...] = (
    ADJUST_VOLUME_TOOL_DEF,
    SET_BRIGHTNESS_TOOL_DEF,
    GET_DEVICE_SETTINGS_TOOL_DEF,
)

ToolHandler = Callable[[str], str]  # arguments(JSON 문자열) → output(JSON 문자열). live_session 의 것과 같은 모양

DEFAULT_TOOLS: tuple[dict[str, Any], ...] = (END_CONVERSATION_TOOL_DEF, SEARCH_MEMORY_TOOL_DEF, *DEVICE_TOOLS)

PLAY_SONG_TOOL = "play_song"
STOP_SONG_TOOL = "stop_song"
# 핸들러 없이 루프가 직접 처리하는 툴 — wiring 은 이 이름들을 핸들러 유무와 무관하게 노출한다
LOOP_TOOLS: frozenset[str] = frozenset({END_CONVERSATION_TOOL, PLAY_SONG_TOOL, STOP_SONG_TOOL})

STOP_SONG_TOOL_DEF: dict[str, Any] = {
    "type": "function",
    "name": STOP_SONG_TOOL,
    "description": "Stop the song that is playing.",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    "strict": True,
}


def make_play_song_tool_def(catalog: Mapping[str, Song]) -> dict[str, Any] | None:
    """``play_song`` 정의 (곡 키 enum, 설명에 곡 목록). 카탈로그가 비면 None."""
    if not catalog:
        return None
    return {
        "type": "function",
        "name": PLAY_SONG_TOOL,
        "description": "Play a song stored on Ray. Pass the song id from this list:\n" + format_song_list(catalog),
        "parameters": {
            "type": "object",
            "properties": {
                "song": {"type": "string", "enum": sorted(catalog), "description": "Song id from the list."},
            },
            "required": ["song"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def make_memory_search_handler(retriever: MemoryRetriever, exclude_session_ids: set[str]) -> ToolHandler:
    """``search_memory`` 툴 핸들러. 검색 결과를 ``{"memories": [{"text", "date"}, …]}`` JSON 으로 돌려준다.

    백엔드가 결과를 읽고 현재 질문에 맞는 것만 골라 답을 만들므로 여기서는 선별하지 않는다(상한은
    retriever 의 것). 인용 갱신은 하지 않는다 — 대화 모델 출력에 인용 태그가 없고, 턴 단위 retained
    buffer 도 이 엔진에서는 의미가 없다.

    Args:
        retriever: 세션 단위 retriever. 핸들러는 executor 스레드에서 불리지만 한 번에 하나씩이다.
        exclude_session_ids: 검색에서 제외할 세션 — 현재 세션과 instructions 의 최근 세션 블록에 포함된 세션.
    """

    def handler(arguments: str) -> str:
        args = json.loads(arguments or "{}")
        query = str(args.get("query", "")).strip()
        if not query:
            return json.dumps({"memories": [], "error": "empty query"})
        result = retriever.retrieve(query, exclude_session_ids)
        memories = [{"text": ep.text, "date": ep.timestamp[:10]} for ep in result.episodes]
        return json.dumps({"memories": memories}, ensure_ascii=False)

    return handler


def make_device_settings_handlers(settings: DeviceSettings) -> dict[str, ToolHandler]:
    """볼륨·밝기 툴 핸들러. 세션 루프의 ``tool_handlers`` 에 그대로 넣는다.

    Args:
        settings: 프로세스 수명의 기기 설정. 핸들러는 executor 스레드에서 한 번에 하나씩 불린다.
    """

    def adjust_volume(arguments: str) -> str:
        args = json.loads(arguments or "{}")
        return json.dumps(settings.adjust_volume(str(args.get("direction", "")), int(args.get("steps", 1))))

    def set_brightness(arguments: str) -> str:
        args = json.loads(arguments or "{}")
        return json.dumps(settings.set_brightness(str(args.get("level", ""))))

    def get_device_settings(_arguments: str) -> str:
        return json.dumps(settings.status())

    return {
        ADJUST_VOLUME_TOOL: adjust_volume,
        SET_BRIGHTNESS_TOOL: set_brightness,
        GET_DEVICE_SETTINGS_TOOL: get_device_settings,
    }
