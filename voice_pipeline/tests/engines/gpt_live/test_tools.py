"""Tests for voice_pipeline.engines.gpt_live.tools — 스키마·핸들러 이름 일치, JSON in/out 계약."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from voice_pipeline.adapters.led import LEDController
from voice_pipeline.device_settings import (
    BRIGHTNESS_DEFAULT,
    BRIGHTNESS_LEVELS,
    VOLUME_STEPS,
    DeviceSettings,
)
from voice_pipeline.engines.gpt_live.songs import Song
from voice_pipeline.engines.gpt_live.tools import (
    ADJUST_VOLUME_TOOL,
    DEFAULT_TOOLS,
    DEVICE_TOOLS,
    END_CONVERSATION_TOOL,
    GET_DEVICE_SETTINGS_TOOL,
    LOOP_TOOLS,
    PLAY_SONG_TOOL,
    SEARCH_MEMORY_TOOL,
    SET_BRIGHTNESS_TOOL,
    STOP_SONG_TOOL,
    STOP_SONG_TOOL_DEF,
    make_device_settings_handlers,
    make_memory_search_handler,
    make_play_song_tool_def,
)
from voice_pipeline.memory.retriever import MemoryRetriever
from voice_pipeline.memory.types import Episode, MemoryReadResult


def _settings(tmp_path: Path, volume_calls: list[int] | None = None) -> DeviceSettings:
    led = Mock(spec=LEDController)
    calls = volume_calls if volume_calls is not None else []
    return DeviceSettings(tmp_path / "device_settings.json", led=led, set_volume=calls.append)


class TestToolDefinitions:
    def test_default_tools_cover_every_handler(self, tmp_path: Path) -> None:
        names = {t["name"] for t in DEFAULT_TOOLS}
        assert names == {END_CONVERSATION_TOOL, SEARCH_MEMORY_TOOL, *(t["name"] for t in DEVICE_TOOLS)}
        assert set(make_device_settings_handlers(_settings(tmp_path))) == {t["name"] for t in DEVICE_TOOLS}

    def test_strict_schemas_require_every_property(self) -> None:
        for t in DEFAULT_TOOLS:  # strict 모드: 모든 속성이 required 여야 한다
            assert t["strict"] is True
            assert set(t["parameters"].get("required", [])) == set(t["parameters"]["properties"])

    def test_device_schema_params_match_handler_args(self) -> None:
        by_name = {t["name"]: t for t in DEVICE_TOOLS}
        assert set(by_name[ADJUST_VOLUME_TOOL]["parameters"]["properties"]) == {"direction", "steps"}
        assert by_name[SET_BRIGHTNESS_TOOL]["parameters"]["properties"]["level"]["enum"] == list(BRIGHTNESS_LEVELS)
        assert by_name[GET_DEVICE_SETTINGS_TOOL]["parameters"]["properties"] == {}


class TestDeviceSettingsHandlers:
    def test_volume_handler_round_trip(self, tmp_path: Path) -> None:
        handlers = make_device_settings_handlers(_settings(tmp_path))
        out = json.loads(handlers[ADJUST_VOLUME_TOOL](json.dumps({"direction": "down", "steps": 1})))
        assert out["level"] == VOLUME_STEPS - 1
        assert out["moved"] == 1

    def test_brightness_handler_round_trip(self, tmp_path: Path) -> None:
        handlers = make_device_settings_handlers(_settings(tmp_path))
        out = json.loads(handlers[SET_BRIGHTNESS_TOOL](json.dumps({"level": "medium"})))
        assert out == {"level": "medium", "previous": BRIGHTNESS_DEFAULT, "levels": list(BRIGHTNESS_LEVELS)}

    def test_status_handler_reports_both_without_changing(self, tmp_path: Path) -> None:
        (tmp_path / "device_settings.json").write_text(json.dumps({"volume_step": 3, "brightness": "medium"}))
        volume_calls: list[int] = []
        settings = _settings(tmp_path, volume_calls)
        out = json.loads(make_device_settings_handlers(settings)[GET_DEVICE_SETTINGS_TOOL]("{}"))
        assert out == {
            "volume": {"level": 3, "max": VOLUME_STEPS},
            "brightness": {"level": "medium", "levels": list(BRIGHTNESS_LEVELS)},
        }
        assert volume_calls == []
        settings._led.set_brightness.assert_not_called()  # type: ignore[attr-defined]

    def test_handler_propagates_bad_args_as_exception(self, tmp_path: Path) -> None:
        # LiveSessionLoop 가 예외를 {"error": ...} 출력으로 바꾼다
        with pytest.raises(ValueError):
            make_device_settings_handlers(_settings(tmp_path))[ADJUST_VOLUME_TOOL]("{}")


class TestMemorySearchHandler:
    def test_search_handler_returns_memories_json(self) -> None:
        retriever = MagicMock(spec=MemoryRetriever)
        ep = Episode(7, "User cried watching Interstellar.", "2026-03-15 20:00:00", "s-1", 1.0, "2026-03-15 20:00:00")
        retriever.retrieve.return_value = MemoryReadResult(episodes=[ep], scores=[0.5], index_to_id={1: 7})
        handler = make_memory_search_handler(retriever, {"cur", "s-recent"})

        out = json.loads(handler(json.dumps({"query": "인터스텔라"})))

        retriever.retrieve.assert_called_once_with("인터스텔라", {"cur", "s-recent"})
        assert out == {"memories": [{"text": "User cried watching Interstellar.", "date": "2026-03-15"}]}

    def test_search_handler_empty_query_does_not_search(self) -> None:
        retriever = MagicMock(spec=MemoryRetriever)
        handler = make_memory_search_handler(retriever, set())
        out = json.loads(handler(json.dumps({"query": "  "})))
        assert out["memories"] == [] and "error" in out
        retriever.retrieve.assert_not_called()


class TestSongToolDefinitions:
    _CATALOG = {
        "IAM": Song("IAM", "I AM", "IVE", ("아이엠",), ("아이브",)),
        "Butter_BTS": Song("Butter_BTS", "Butter", "BTS"),
    }

    def test_play_song_enum_is_catalog_keys_and_description_lists_songs(self) -> None:
        play, stop = make_play_song_tool_def(self._CATALOG), STOP_SONG_TOOL_DEF
        assert play is not None and play["name"] == PLAY_SONG_TOOL and stop["name"] == STOP_SONG_TOOL
        assert play["parameters"]["properties"]["song"]["enum"] == ["Butter_BTS", "IAM"]
        assert play["description"].endswith("IAM: I AM — IVE (아이엠, 아이브)")
        for t in (play, stop):  # strict 모드 계약은 다른 툴과 같다
            assert t["strict"] is True
            assert set(t["parameters"].get("required", [])) == set(t["parameters"]["properties"])

    def test_empty_catalog_yields_no_play_tool(self) -> None:
        assert make_play_song_tool_def({}) is None

    def test_song_tools_are_loop_builtin(self) -> None:
        assert {PLAY_SONG_TOOL, STOP_SONG_TOOL, END_CONVERSATION_TOOL} <= LOOP_TOOLS
