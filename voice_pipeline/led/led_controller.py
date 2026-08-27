"""LED controller with optional hardware driver and animation thread."""

from __future__ import annotations

import logging
import threading
from typing import Any

from voice_pipeline.core.interfaces import ILEDController
from voice_pipeline.core.types import LEDState
from voice_pipeline.led.animations import RGB, BreathingAnimation, LEDAnimation, StaticAnimation
from voice_pipeline.led.arbiter_client import OSLedArbiterClient
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
# LEDController
# ---------------------------------------------------------------------------


class LEDController(ILEDController):
    """LED display controller with background animation thread.

    Hardware:
        Uses ``rpi5_ws2812.WS2812SpiDriver`` when available. Falls back to
        logging-only (noop) mode when the driver is not installed
        (development/CI) or when constructed with ``enabled=False`` (e.g. no
        LED hardware connected).

    Threading:
        A daemon thread runs the animation loop. ``set_state()`` is thread-safe
        and swaps the active animation under a lock. ``close()`` stops the thread.

    Strip ownership:
        The strip is shared with the OS_LED boot daemon, which keeps drawing the
        boot animation until RAY borrows it. Borrowing happens lazily on the
        first ``set_state()`` — not in ``__init__`` — because construction runs
        before model loading (~50 s of it). Acquiring at construction time would
        blank the strip for that whole stretch, since the boot daemon stops
        drawing the moment we take the token and we have nothing to show yet.
    """

    _BAR_COUNT = 8  # 바 세그먼트 LED 개수
    _RING_COUNT = 16  # 링 세그먼트 LED 개수
    _LED_COUNT = _BAR_COUNT + _RING_COUNT  # 전체 LED 개수
    _BRIGHTNESS = 1.0  # LED 전체 밝기 (0.0=꺼짐, 1.0=최대)
    _NOOP_SLEEP_SEC = 0.1  # 애니메이션 없을 때 스레드 폴링 간격 (초)
    _CLOSE_JOIN_TIMEOUT_SEC = 2.0  # close 시 애니메이션 스레드 종료 대기 (초)

    # 상태별 애니메이션 맵 (단색 플레이스홀더)
    _ANIMATIONS: dict[LEDState, LEDAnimation] = {
        LEDState.OFF: StaticAnimation(bar_color=(0, 0, 0), ring_color=(0, 0, 0)),
        LEDState.SLEEPING: BreathingAnimation(color=(233, 233, 50)),
        LEDState.IDLE: StaticAnimation(bar_color=(233, 233, 50), ring_color=(233, 233, 50)),
    }

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._animations = dict(self._ANIMATIONS)
        self._brightness = self._BRIGHTNESS

        self._lock = threading.Lock()
        self._state = LEDState.OFF
        self._tick = 0
        self._stop_event = threading.Event()
        self._state_changed = threading.Event()

        # Hardware strip (None = noop fallback). When a real strip is used we
        # first borrow it from the OS_LED rainbow daemon (shared SPI bus).
        self._strip: Any = None
        self._driver: Any = None
        self._arbiter = OSLedArbiterClient()
        # Borrowed on the first set_state() — see "Strip ownership" above.
        self._init_lock = threading.Lock()
        self._strip_init_done = False

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

    def _ensure_strip(self) -> None:
        """Borrow and open the strip once, on first use.

        A failure here degrades to noop mode instead of propagating: the boot
        daemon keeps the strip and stays visible, which is a better outcome than
        killing the pipeline over an indicator light. (When this ran in
        ``__init__`` the same failure aborted startup.)
        """
        if self._strip_init_done:
            return
        with self._init_lock:
            if self._strip_init_done:
                return
            self._strip_init_done = True
            try:
                self._init_strip()
            except LEDError:
                logger.error("LED strip init failed — continuing in noop mode", exc_info=True)

    def _init_strip(self) -> None:
        if not self._enabled:
            logger.info("LED disabled (enabled=False) — LED controller running in noop mode")
            return
        if _WS2812SpiDriver is None:
            logger.info("rpi5_ws2812 not available — LED controller running in noop mode")
            return
        # Borrow the shared strip from the OS_LED rainbow daemon before opening
        # SPI, so the two processes never drive the bus at the same time.
        self._arbiter.acquire()
        try:
            driver = _WS2812SpiDriver(
                spi_bus=0,
                spi_device=0,
                led_count=self._LED_COUNT,
            )
            self._driver = driver
            self._strip = driver.get_strip()
            self._strip.set_brightness(self._brightness)
            logger.info(
                "LED strip initialized: %d LEDs (bar=%d, ring=%d), brightness=%.2f",
                self._LED_COUNT,
                self._BAR_COUNT,
                self._RING_COUNT,
                self._brightness,
            )
        except Exception as exc:
            self._arbiter.release()  # hand the strip back to the rainbow daemon
            raise LEDError(f"Failed to initialize LED strip: {exc}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_state(self, state: LEDState) -> None:
        """Set the LED display to the given state.

        Swaps the active animation and resets the tick counter.
        Thread-safe.

        The first call also borrows the strip from the OS_LED boot daemon, which
        is what makes the boot animation hand over directly to the RAY pattern.
        """
        self._ensure_strip()
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
        self._thread.join(timeout=self._CLOSE_JOIN_TIMEOUT_SEC)
        if self._thread.is_alive():
            logger.warning("LED animation thread did not exit within timeout")
        self._apply_off()
        # Fully close our SPI device BEFORE releasing the token, so no RAY-side
        # write can overlap the daemon's rainbow fade-in on the shared bus.
        self._close_strip()
        # Hand the strip back: the daemon fades the rainbow back in.
        self._arbiter.release()
        logger.debug("LED controller closed")

    def _close_strip(self) -> None:
        """Close the underlying SPI device (rpi5_ws2812 exposes no public close)."""
        if self._driver is None:
            return
        try:
            self._driver._device.close()
        except Exception:
            logger.debug("Error closing SPI device (suppressed)", exc_info=True)
        self._strip = None
        self._driver = None

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
                self._wait(self._NOOP_SLEEP_SEC)
                continue

            try:
                frame = anim.render(tick, self._BAR_COUNT, self._RING_COUNT)
                self._apply_frame(frame)
            except Exception:
                logger.debug("Animation render error (suppressed)", exc_info=True)

            self._wait(anim.frame_interval_sec)

    def _wait(self, seconds: float) -> None:
        """Sleep for *seconds*, but wake early on state change or stop."""
        self._state_changed.clear()
        self._state_changed.wait(timeout=seconds)

    def _off_frame(self) -> list[RGB]:
        return [(0, 0, 0)] * self._LED_COUNT

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
