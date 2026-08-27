"""Keep LED unit tests off the real strip and the OS_LED daemon.

On a robot Pi both ``rpi5_ws2812`` and the ``os-led-display`` daemon are present,
so an unpatched ``LEDController`` would open ``/dev/spidev0.0`` and send ACQUIRE
to ``/run/os-led.sock`` — the test run visibly blanks the robot's LEDs (seen in the
daemon journal as ``client acquired`` / ``client released`` bursts).

Every test therefore starts in noop mode: no SPI driver, and an arbiter client whose
``acquire``/``release`` are no-ops. Tests that need a driver patch ``_WS2812SpiDriver``
themselves with a ``MagicMock`` class; the inner patch wins and the arbiter stays mocked.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _no_led_hardware() -> Iterator[None]:
    with (
        patch("voice_pipeline.led.led_controller._WS2812SpiDriver", None),
        patch("voice_pipeline.led.led_controller.OSLedArbiterClient", MagicMock()),
    ):
        yield
