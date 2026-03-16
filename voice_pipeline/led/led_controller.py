"""LED controller with optional hardware driver and animation thread."""

from __future__ import annotations

import logging
import threading
from typing import Any

from voice_pipeline.core.config import LEDConfig
from voice_pipeline.core.interfaces import ILEDController
from voice_pipeline.core.types import LEDState
from voice_pipeline.led.animations import RGB, BreathingAnimation, LEDAnimation, StaticAnimation
from voice_pipeline.led.exceptions import LEDError

logger = logging.getLogger("voice_pipeline.led")

# ---------------------------------------------------------------------------
# Optional hardware import
# ---------------------------------------------------------------------------

_WS2812SpiDriver: type | None = None
_Color: type | None = None

try:
    from rpi5_ws2812.ws2812 import Color as _Color  # type: ignore[no-redef]
    from rpi5_ws2812.ws2812 import WS2812SpiDriver as _WS2812SpiDriver  # type: ignore[no-redef]
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Default animation map (placeholder colors)
# ---------------------------------------------------------------------------

_OFF: RGB = (0, 0, 0)
_BASE: RGB = (233, 233, 50)

_DEFAULT_ANIMATIONS: dict[LEDState, LEDAnimation] = {
    LEDState.OFF: StaticAnimation(bar_color=_OFF, ring_color=_OFF),
    LEDState.SLEEPING: BreathingAnimation(color=_BASE),
    LEDState.IDLE: StaticAnimation(bar_color=_BASE, ring_color=_BASE),
}


# ---------------------------------------------------------------------------
# LEDController
# ---------------------------------------------------------------------------


class LEDController(ILEDController):
    """LED display controller with background animation thread.

    Hardware:
        Uses ``rpi5_ws2812.WS2812SpiDriver`` when available. Falls back to
        logging-only mode when the driver is not installed (development/CI).

    Threading:
        A daemon thread runs the animation loop. ``set_state()`` is thread-safe
        and swaps the active animation under a lock. ``close()`` stops the thread.

    Args:
        config: LED configuration (counts, SPI pin, brightness).
        animations: Optional custom animation map. Defaults to built-in
            static-color placeholders.
    """

    def __init__(
        self,
        config: LEDConfig,
        animations: dict[LEDState, LEDAnimation] | None = None,
    ) -> None:
        self._config = config
        self._bar_count = config.bar_count
        self._ring_count = config.ring_count
        self._led_count = config.bar_count + config.ring_count
        if animations is not None:
            self._animations = dict(animations)
        else:
            self._animations = dict(_DEFAULT_ANIMATIONS)

        self._lock = threading.Lock()
        self._state = LEDState.OFF
        self._tick = 0
        self._stop_event = threading.Event()
        self._state_changed = threading.Event()

        # Hardware strip (None = noop fallback)
        self._strip: Any = None
        self._init_strip(config)

        # Start animation thread
        self._thread = threading.Thread(
            target=self._animation_loop,
            name="led-animation",
            daemon=True,
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Hardware init
    # ------------------------------------------------------------------

    def _init_strip(self, config: LEDConfig) -> None:
        if _WS2812SpiDriver is None:
            logger.info("rpi5_ws2812 not available — LED controller running in noop mode")
            return
        try:
            driver = _WS2812SpiDriver(
                spi_bus=0,
                spi_device=0,
                led_count=self._led_count,
            )
            self._strip = driver.get_strip()
            self._strip.set_brightness(config.brightness / 255.0)
            logger.info(
                "LED strip initialized: %d LEDs (bar=%d, ring=%d), brightness=%d",
                self._led_count,
                self._bar_count,
                self._ring_count,
                config.brightness,
            )
        except Exception as exc:
            raise LEDError(f"Failed to initialize LED strip: {exc}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_state(self, state: LEDState) -> None:
        """Set the LED display to the given state.

        Swaps the active animation and resets the tick counter.
        Thread-safe.
        """
        with self._lock:
            if state == self._state:
                return
            self._state = state
            self._tick = 0
            anim = self._animations.get(state)
            if anim is not None:
                anim.reset()
            else:
                logger.warning("No animation registered for state %s", state)
        # Wake the animation thread so it picks up the new state immediately
        self._state_changed.set()
        logger.debug("LED state → %s", state.value)

    def close(self) -> None:
        """Stop the animation thread and turn off LEDs."""
        self._stop_event.set()
        self._state_changed.set()  # wake thread if sleeping
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            logger.warning("LED animation thread did not exit within timeout")
        self._apply_off()
        logger.debug("LED controller closed")

    # ------------------------------------------------------------------
    # Animation loop (runs on daemon thread)
    # ------------------------------------------------------------------

    def _animation_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                anim = self._animations.get(self._state)
                tick = self._tick
                self._tick += 1

            if anim is None:
                self._apply_frame(self._off_frame())
                self._wait(0.1)
                continue

            try:
                frame = anim.render(tick, self._bar_count, self._ring_count)
                self._apply_frame(frame)
            except Exception:
                logger.debug("Animation render error (suppressed)", exc_info=True)

            self._wait(anim.frame_interval_sec)

    def _wait(self, seconds: float) -> None:
        """Sleep for *seconds*, but wake early on state change or stop."""
        self._state_changed.clear()
        self._state_changed.wait(timeout=seconds)

    def _off_frame(self) -> list[RGB]:
        return [_OFF] * self._led_count

    # ------------------------------------------------------------------
    # Strip helpers
    # ------------------------------------------------------------------

    def _apply_frame(self, frame: list[tuple[int, int, int]]) -> None:
        if self._strip is None:
            return
        for i, (r, g, b) in enumerate(frame):
            self._strip.set_pixel_color(i, _Color(r, g, b))
        self._strip.show()

    def _apply_off(self) -> None:
        if self._strip is None:
            return
        try:
            self._strip.set_all_pixels(_Color(0, 0, 0))
            self._strip.show()
        except Exception:
            logger.debug("Error turning off LEDs (suppressed)", exc_info=True)
