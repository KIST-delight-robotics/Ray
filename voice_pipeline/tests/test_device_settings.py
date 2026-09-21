"""DeviceSettings 단위 테스트 — 단계 이동·경계·영속화·툴 핸들러 JSON 계약."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from voice_pipeline.adapters.led import LEDController
from voice_pipeline.device_settings import (
    ADJUST_VOLUME_TOOL,
    BRIGHTNESS_DEFAULT,
    BRIGHTNESS_LEVELS,
    DEVICE_TOOLS,
    GET_DEVICE_SETTINGS_TOOL,
    SET_BRIGHTNESS_TOOL,
    VOLUME_DEFAULT_STEP,
    VOLUME_PERCENT_PER_STEP,
    VOLUME_STEPS,
    DeviceSettings,
)


@pytest.fixture
def led() -> Mock:
    return Mock(spec=LEDController)


@pytest.fixture
def volume_calls() -> list[int]:
    return []


def _make(tmp_path: Path, led: Mock, volume_calls: list[int], *, fail_volume: bool = False) -> DeviceSettings:
    def set_volume(percent: int) -> None:
        if fail_volume:
            raise RuntimeError("wpctl down")
        volume_calls.append(percent)

    return DeviceSettings(tmp_path / "device_settings.json", led=led, set_volume=set_volume)


# ---------------------------------------------------------------------------
# Defaults / apply
# ---------------------------------------------------------------------------


class TestStartup:
    def test_defaults_when_file_missing(self, tmp_path: Path, led: Mock, volume_calls: list[int]) -> None:
        ds = _make(tmp_path, led, volume_calls)
        assert ds.volume_step == VOLUME_DEFAULT_STEP
        assert ds.brightness == BRIGHTNESS_DEFAULT
        assert not (tmp_path / "device_settings.json").exists()  # 변경 전에는 파일을 만들지 않는다

    def test_apply_pushes_saved_values_to_hardware(self, tmp_path: Path, led: Mock, volume_calls: list[int]) -> None:
        (tmp_path / "device_settings.json").write_text(json.dumps({"volume_step": 4, "brightness": "low"}))
        ds = _make(tmp_path, led, volume_calls)
        ds.apply()
        led.set_brightness.assert_called_once_with(BRIGHTNESS_LEVELS["low"])
        assert volume_calls == [4 * VOLUME_PERCENT_PER_STEP]

    def test_apply_survives_volume_failure(self, tmp_path: Path, led: Mock, volume_calls: list[int]) -> None:
        ds = _make(tmp_path, led, volume_calls, fail_volume=True)
        ds.apply()  # 경고만, 예외 없음
        led.set_brightness.assert_called_once_with(BRIGHTNESS_LEVELS[BRIGHTNESS_DEFAULT])

    @pytest.mark.parametrize(
        "payload",
        [
            "not json",
            json.dumps({"volume_step": 0, "brightness": "purple"}),
            json.dumps([1, 2]),
            json.dumps({"volume_step": "7"}),
        ],
    )
    def test_invalid_file_falls_back_to_defaults(
        self, tmp_path: Path, led: Mock, volume_calls: list[int], payload: str
    ) -> None:
        (tmp_path / "device_settings.json").write_text(payload)
        ds = _make(tmp_path, led, volume_calls)
        assert (ds.volume_step, ds.brightness) == (VOLUME_DEFAULT_STEP, BRIGHTNESS_DEFAULT)


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


class TestVolume:
    def test_up_at_max_reports_limit(self, tmp_path: Path, led: Mock, volume_calls: list[int]) -> None:
        ds = _make(tmp_path, led, volume_calls)  # 기본 = 최대
        result = ds.adjust_volume("up", 1)
        assert result == {"level": VOLUME_STEPS, "max": VOLUME_STEPS, "moved": 0, "at_limit": "max"}
        assert volume_calls == []
        assert not (tmp_path / "device_settings.json").exists()

    def test_down_two_steps_applies_and_persists(self, tmp_path: Path, led: Mock, volume_calls: list[int]) -> None:
        ds = _make(tmp_path, led, volume_calls)
        result = ds.adjust_volume("down", 2)
        assert result["level"] == VOLUME_STEPS - 2
        assert result["moved"] == 2
        assert result["at_limit"] is None
        assert volume_calls == [(VOLUME_STEPS - 2) * VOLUME_PERCENT_PER_STEP]
        reloaded = _make(tmp_path, led, [])
        assert reloaded.volume_step == VOLUME_STEPS - 2

    def test_partial_move_when_not_enough_steps(self, tmp_path: Path, led: Mock, volume_calls: list[int]) -> None:
        (tmp_path / "device_settings.json").write_text(json.dumps({"volume_step": 2, "brightness": "high"}))
        ds = _make(tmp_path, led, volume_calls)
        result = ds.adjust_volume("down", 3)
        assert result == {"level": 1, "max": VOLUME_STEPS, "moved": 1, "at_limit": "min"}
        assert volume_calls == [1 * VOLUME_PERCENT_PER_STEP]

    def test_steps_below_one_is_treated_as_one(self, tmp_path: Path, led: Mock, volume_calls: list[int]) -> None:
        (tmp_path / "device_settings.json").write_text(json.dumps({"volume_step": 5, "brightness": "high"}))
        ds = _make(tmp_path, led, volume_calls)
        assert ds.adjust_volume("up", 0)["level"] == 6

    def test_invalid_direction_raises(self, tmp_path: Path, led: Mock, volume_calls: list[int]) -> None:
        ds = _make(tmp_path, led, volume_calls)
        with pytest.raises(ValueError):
            ds.adjust_volume("sideways", 1)

    def test_hardware_failure_leaves_state_unchanged(self, tmp_path: Path, led: Mock, volume_calls: list[int]) -> None:
        ds = _make(tmp_path, led, volume_calls, fail_volume=True)
        with pytest.raises(RuntimeError):
            ds.adjust_volume("down", 1)
        assert ds.volume_step == VOLUME_DEFAULT_STEP
        assert not (tmp_path / "device_settings.json").exists()


# ---------------------------------------------------------------------------
# Brightness
# ---------------------------------------------------------------------------


class TestBrightness:
    def test_set_applies_and_persists(self, tmp_path: Path, led: Mock, volume_calls: list[int]) -> None:
        ds = _make(tmp_path, led, volume_calls)
        result = ds.set_brightness("off")
        assert result["level"] == "off"
        assert result["previous"] == BRIGHTNESS_DEFAULT
        assert result["levels"] == list(BRIGHTNESS_LEVELS)
        led.set_brightness.assert_called_once_with(0.0)
        assert _make(tmp_path, led, []).brightness == "off"

    def test_same_level_reapplies_without_save(self, tmp_path: Path, led: Mock, volume_calls: list[int]) -> None:
        ds = _make(tmp_path, led, volume_calls)
        ds.set_brightness(BRIGHTNESS_DEFAULT)
        led.set_brightness.assert_called_once()
        assert not (tmp_path / "device_settings.json").exists()

    def test_unknown_level_raises(self, tmp_path: Path, led: Mock, volume_calls: list[int]) -> None:
        ds = _make(tmp_path, led, volume_calls)
        with pytest.raises(ValueError):
            ds.set_brightness("blinding")
        led.set_brightness.assert_not_called()


# ---------------------------------------------------------------------------
# Tool handlers — JSON in / JSON out, 스키마와 이름이 맞는지
# ---------------------------------------------------------------------------


class TestToolHandlers:
    def test_handler_names_match_tool_defs(self, tmp_path: Path, led: Mock, volume_calls: list[int]) -> None:
        ds = _make(tmp_path, led, volume_calls)
        expected = {ADJUST_VOLUME_TOOL, SET_BRIGHTNESS_TOOL, GET_DEVICE_SETTINGS_TOOL}
        assert set(ds.tool_handlers()) == {t["name"] for t in DEVICE_TOOLS} == expected

    def test_schema_params_match_handler_args(self) -> None:
        by_name = {t["name"]: t for t in DEVICE_TOOLS}
        assert set(by_name[ADJUST_VOLUME_TOOL]["parameters"]["properties"]) == {"direction", "steps"}
        assert by_name[SET_BRIGHTNESS_TOOL]["parameters"]["properties"]["level"]["enum"] == list(BRIGHTNESS_LEVELS)
        for t in DEVICE_TOOLS:  # strict 모드: 모든 속성이 required 여야 한다
            assert set(t["parameters"].get("required", [])) == set(t["parameters"]["properties"])

    def test_volume_handler_round_trip(self, tmp_path: Path, led: Mock, volume_calls: list[int]) -> None:
        ds = _make(tmp_path, led, volume_calls)
        out = json.loads(ds.tool_handlers()[ADJUST_VOLUME_TOOL](json.dumps({"direction": "down", "steps": 1})))
        assert out["level"] == VOLUME_STEPS - 1
        assert out["moved"] == 1

    def test_brightness_handler_round_trip(self, tmp_path: Path, led: Mock, volume_calls: list[int]) -> None:
        ds = _make(tmp_path, led, volume_calls)
        out = json.loads(ds.tool_handlers()[SET_BRIGHTNESS_TOOL](json.dumps({"level": "medium"})))
        assert out == {"level": "medium", "previous": BRIGHTNESS_DEFAULT, "levels": list(BRIGHTNESS_LEVELS)}

    def test_status_handler_reports_both_without_changing(
        self, tmp_path: Path, led: Mock, volume_calls: list[int]
    ) -> None:
        (tmp_path / "device_settings.json").write_text(json.dumps({"volume_step": 3, "brightness": "medium"}))
        ds = _make(tmp_path, led, volume_calls)
        out = json.loads(ds.tool_handlers()[GET_DEVICE_SETTINGS_TOOL]("{}"))
        assert out == {
            "volume": {"level": 3, "max": VOLUME_STEPS},
            "brightness": {"level": "medium", "levels": list(BRIGHTNESS_LEVELS)},
        }
        assert volume_calls == []
        led.set_brightness.assert_not_called()

    def test_handler_propagates_bad_args_as_exception(self, tmp_path: Path, led: Mock, volume_calls: list[int]) -> None:
        # LiveSessionLoop 가 예외를 {"error": ...} 출력으로 바꾼다
        ds = _make(tmp_path, led, volume_calls)
        with pytest.raises(ValueError):
            ds.tool_handlers()[ADJUST_VOLUME_TOOL]("{}")
