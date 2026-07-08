"""Matter platform LED control — standalone command → on/off signal layer.

This package is the "middle box": it turns an already-decided command
("turn the light on/off") into a real Matter On/Off signal to a WiFi bulb.

It deliberately does NOT include:
    * the LLM deciding *when* to send the command (wired later), or
    * a hard dependency on real bulb hardware (swap the backend when ready).

Callers depend only on :class:`MatterLedController` (the high-level middle box)
and :class:`MatterLightBackend` (the swappable driver interface).
"""

from __future__ import annotations

from matter_platform_led.config import MatterConfig, load_config
from matter_platform_led.controller import MatterLedController, build_backend
from matter_platform_led.exceptions import (
    MatterCommissionError,
    MatterError,
    MatterNotCommissionedError,
)
from matter_platform_led.interface import LightStatus, MatterLightBackend

__all__ = [
    "LightStatus",
    "MatterCommissionError",
    "MatterConfig",
    "MatterError",
    "MatterLedController",
    "MatterLightBackend",
    "MatterNotCommissionedError",
    "build_backend",
    "load_config",
]
