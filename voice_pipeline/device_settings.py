"""사용자가 말로 바꾸는 기기 설정 — 스피커 볼륨(단계), LED 밝기(단계).

GPT-Live 백엔드 함수 툴 ``adjust_volume`` / ``set_brightness`` / ``get_device_settings`` 의 정의와 핸들러, 값의 영속화
(``var/device_settings.json``), 프로세스 시작 시 재적용을 한곳에 둔다. 툴 스키마와 그것을 읽는 핸들러가
같은 파일에 있어 인자 이름이 어긋나지 않는다.

스마트 스피커처럼 단순하게 간다. 볼륨은 올리기/내리기만 있고 끄기는 없다(최소 단계가 곧 가장 작은 소리).
밝기는 단계 이름 하나로 정한다(끄기 포함). 끝에 닿으면 실패가 아니라 가능한 만큼만 움직이고 결과에
``at_limit`` 을 표시해, 백엔드가 "최대예요/최소예요" 라고 안내하게 한다.

적용 경로: 볼륨은 PipeWire 기본 싱크(:mod:`~voice_pipeline.adapters.system_volume`), 밝기는
:meth:`~voice_pipeline.adapters.led.LEDController.set_brightness`. 핸들러는 세션 루프의 executor 스레드에서
한 번에 하나씩 불리고, :meth:`DeviceSettings.apply` 는 시작 시 메인 스레드에서 부르므로 락으로 직렬화한다.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from voice_pipeline.adapters.led import LEDController
from voice_pipeline.adapters.system_volume import set_sink_volume

logger = logging.getLogger("voice_pipeline.device_settings")

# 볼륨: 1..VOLUME_STEPS 단계, 단계 n = n × VOLUME_PERCENT_PER_STEP %. 최소 단계도 소리가 나야 하므로 0% 는 없다.
VOLUME_STEPS = 10
VOLUME_PERCENT_PER_STEP = 10
VOLUME_DEFAULT_STEP = 10  # 설정 파일이 없을 때. 도입 전 동작(싱크 100%)과 같게 둔다

# 밝기: 단계 이름 → LED 전체 밝기 (0.0~1.0). 값은 기기에서 보고 조정할 것.
BRIGHTNESS_LEVELS: dict[str, float] = {"off": 0.0, "low": 0.3, "medium": 0.65, "high": 1.0}
BRIGHTNESS_DEFAULT = "high"  # 설정 파일이 없을 때. 도입 전 동작(LEDController._BRIGHTNESS = 1.0)과 같게 둔다

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


class DeviceSettings:
    """볼륨 단계와 밝기 단계를 보관·적용·저장한다. 프로세스 수명 객체.

    Args:
        path: 설정 JSON 파일 경로 (``var/`` 아래). 없거나 손상되면 기본값으로 시작하고 첫 변경 때 만든다.
        led: 밝기를 적용할 LED 컨트롤러. 스트립 인수 전에 불러도 값이 보관된다.
        set_volume: 볼륨 퍼센트(0~100)를 하드웨어에 적용하는 호출자. 기본은 PipeWire 기본 싱크. 테스트용 주입점.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        led: LEDController,
        set_volume: Callable[[int], None] = set_sink_volume,
    ) -> None:
        self._path = Path(path)
        self._led = led
        self._set_volume = set_volume
        self._lock = threading.Lock()
        self.volume_step, self.brightness = self._load()

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def apply(self) -> None:
        """저장된 값을 하드웨어에 적용한다 (프로세스 시작 시 한 번).

        볼륨 적용 실패는 경고만 남긴다 — PipeWire 가 아직 안 떠 있거나 개발 PC 인 경우이고, 기동을 막을 일이 아니다.
        """
        with self._lock:
            self._led.set_brightness(BRIGHTNESS_LEVELS[self.brightness])
            try:
                self._set_volume(self._percent(self.volume_step))
            except Exception:
                logger.warning("Volume apply failed at startup — leaving sink volume as is", exc_info=True)
        logger.info(
            "Device settings applied: volume %d/%d, brightness %s",
            self.volume_step,
            VOLUME_STEPS,
            self.brightness,
        )

    # ------------------------------------------------------------------
    # Operations (tool handlers call these)
    # ------------------------------------------------------------------

    def adjust_volume(self, direction: str, steps: int = 1) -> dict[str, Any]:
        """볼륨을 ``direction`` 으로 ``steps`` 칸 움직인다. 칸이 모자라면 가능한 만큼만.

        Returns:
            ``level``(현재 단계), ``max``, ``moved``(실제 움직인 칸), ``at_limit``(``"max"``/``"min"``/None).

        Raises:
            ValueError: direction 이 up/down 이 아닐 때.
            RuntimeError: 하드웨어 적용 실패. 상태는 바뀌지 않는다.
        """
        if direction not in ("up", "down"):
            raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
        steps = max(1, int(steps))
        with self._lock:
            current = self.volume_step
            target = current + steps if direction == "up" else current - steps
            target = max(1, min(VOLUME_STEPS, target))
            moved = abs(target - current)
            if moved:
                self._set_volume(self._percent(target))
                self.volume_step = target
                self._save()
            at_limit = "max" if target == VOLUME_STEPS else "min" if target == 1 else None
        logger.info("Volume %s %d → %d/%d (moved %d)", direction, current, target, VOLUME_STEPS, moved)
        return {"level": target, "max": VOLUME_STEPS, "moved": moved, "at_limit": at_limit}

    def set_brightness(self, level: str) -> dict[str, Any]:
        """LED 밝기를 단계 ``level`` 로 맞춘다.

        Returns:
            ``level``(적용된 단계), ``previous``(이전 단계), ``levels``(단계 목록, 어두운 것부터).

        Raises:
            ValueError: 모르는 단계 이름.
        """
        if level not in BRIGHTNESS_LEVELS:
            raise ValueError(f"level must be one of {list(BRIGHTNESS_LEVELS)}, got {level!r}")
        with self._lock:
            previous = self.brightness
            self._led.set_brightness(BRIGHTNESS_LEVELS[level])
            if level != previous:
                self.brightness = level
                self._save()
        logger.info("Brightness %s → %s", previous, level)
        return {"level": level, "previous": previous, "levels": list(BRIGHTNESS_LEVELS)}

    def status(self) -> dict[str, Any]:
        """현재 상태. ``volume``(level/max), ``brightness``(level/levels)."""
        with self._lock:
            return {
                "volume": {"level": self.volume_step, "max": VOLUME_STEPS},
                "brightness": {"level": self.brightness, "levels": list(BRIGHTNESS_LEVELS)},
            }

    def tool_handlers(self) -> dict[str, ToolHandler]:
        """백엔드 함수 툴 이름 → 핸들러. 세션 루프의 ``tool_handlers`` 에 그대로 넣는다."""

        def adjust_volume(arguments: str) -> str:
            args = json.loads(arguments or "{}")
            return json.dumps(self.adjust_volume(str(args.get("direction", "")), int(args.get("steps", 1))))

        def set_brightness(arguments: str) -> str:
            args = json.loads(arguments or "{}")
            return json.dumps(self.set_brightness(str(args.get("level", ""))))

        def get_device_settings(_arguments: str) -> str:
            return json.dumps(self.status())

        return {
            ADJUST_VOLUME_TOOL: adjust_volume,
            SET_BRIGHTNESS_TOOL: set_brightness,
            GET_DEVICE_SETTINGS_TOOL: get_device_settings,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _percent(step: int) -> int:
        return step * VOLUME_PERCENT_PER_STEP

    def _load(self) -> tuple[int, str]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return VOLUME_DEFAULT_STEP, BRIGHTNESS_DEFAULT
        except (OSError, ValueError):
            logger.warning("Device settings file unreadable — using defaults: %s", self._path, exc_info=True)
            return VOLUME_DEFAULT_STEP, BRIGHTNESS_DEFAULT

        volume = data.get("volume_step", VOLUME_DEFAULT_STEP) if isinstance(data, dict) else VOLUME_DEFAULT_STEP
        brightness = data.get("brightness", BRIGHTNESS_DEFAULT) if isinstance(data, dict) else BRIGHTNESS_DEFAULT
        if not isinstance(volume, int) or not 1 <= volume <= VOLUME_STEPS:
            logger.warning("Invalid volume_step %r in %s — using default", volume, self._path)
            volume = VOLUME_DEFAULT_STEP
        if brightness not in BRIGHTNESS_LEVELS:
            logger.warning("Invalid brightness %r in %s — using default", brightness, self._path)
            brightness = BRIGHTNESS_DEFAULT
        return volume, brightness

    def _save(self) -> None:
        """원자적으로 쓴다 (임시 파일 → rename). 실패해도 메모리 상태는 유지하고 경고만."""
        payload = {"volume_step": self.volume_step, "brightness": self.brightness}
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            tmp.replace(self._path)
        except OSError:
            logger.warning("Device settings save failed: %s", self._path, exc_info=True)
