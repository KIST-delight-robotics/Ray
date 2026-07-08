"""Mock backend — verify the whole command path with no hardware.

Why this exists: the bulb isn't here yet, but the "middle box" logic (command →
signal) still needs to be built and proven correct. ``MockBackend`` implements
the exact same :class:`MatterLightBackend` contract as the real driver, so the
CLI, the controller, and (later) the LLM tool call all run for real against it.

State is persisted to a small JSON file in the OS temp dir, keyed by node id, so
that separate CLI invocations see each other's effect (``on`` then ``status``
reports "on"). Delete the file, or call :meth:`reset`, to start clean.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from matter_platform_led.exceptions import MatterNotCommissionedError
from matter_platform_led.interface import LightStatus, MatterLightBackend

logger = logging.getLogger("matter_platform_led.mock")


class MockBackend(MatterLightBackend):
    """In-memory/file-backed fake bulb that logs every action."""

    def __init__(self, node_id: str, endpoint_id: int, state_file: str | Path | None = None) -> None:
        self._node_id = node_id
        self._endpoint_id = endpoint_id
        if state_file is not None:
            self._state_file = Path(state_file)
        else:
            safe_node = node_id.replace("/", "_")
            self._state_file = Path(tempfile.gettempdir()) / f"matter_led_mock_{safe_node}.json"

    # -- persistence -------------------------------------------------------

    def _load(self) -> dict:
        if not self._state_file.is_file():
            return {"commissioned": False, "on": False}
        try:
            return json.loads(self._state_file.read_text())
        except (OSError, json.JSONDecodeError):
            return {"commissioned": False, "on": False}

    def _save(self, state: dict) -> None:
        self._state_file.write_text(json.dumps(state))

    def _require_commissioned(self, state: dict) -> None:
        if not state.get("commissioned", False):
            raise MatterNotCommissionedError(f"mock node {self._node_id} not commissioned — run `commission` first")

    # -- backend contract --------------------------------------------------

    def commission(self, pairing_code: str, ssid: str | None, password: str | None) -> None:
        mode = "wifi(ble)" if ssid else "on-network"
        logger.info(
            "MOCK commission node=%s endpoint=%s code=%s mode=%s ssid=%s",
            self._node_id,
            self._endpoint_id,
            pairing_code,
            mode,
            ssid or "-",
        )
        self._save({"commissioned": True, "on": False})

    def turn_on(self) -> None:
        state = self._load()
        self._require_commissioned(state)
        state["on"] = True
        self._save(state)
        logger.info("MOCK turn_on node=%s endpoint=%s -> ON", self._node_id, self._endpoint_id)

    def turn_off(self) -> None:
        state = self._load()
        self._require_commissioned(state)
        state["on"] = False
        self._save(state)
        logger.info("MOCK turn_off node=%s endpoint=%s -> OFF", self._node_id, self._endpoint_id)

    def toggle(self) -> None:
        state = self._load()
        self._require_commissioned(state)
        state["on"] = not state.get("on", False)
        self._save(state)
        logger.info(
            "MOCK toggle node=%s endpoint=%s -> %s",
            self._node_id,
            self._endpoint_id,
            "ON" if state["on"] else "OFF",
        )

    def open_commissioning_window(self, timeout_sec: int = 180) -> str:
        state = self._load()
        self._require_commissioned(state)
        code = "MT:MOCK-SHARE-CODE"
        logger.info(
            "MOCK open_commissioning_window node=%s timeout=%ss -> %s",
            self._node_id,
            timeout_sec,
            code,
        )
        return code

    def read_status(self) -> LightStatus:
        state = self._load()
        self._require_commissioned(state)
        return LightStatus(on=bool(state.get("on", False)), reachable=True)

    def reset(self) -> None:
        """Forget commissioning and state (delete the persisted file)."""
        self._state_file.unlink(missing_ok=True)
        logger.info("MOCK reset node=%s (%s)", self._node_id, self._state_file)
