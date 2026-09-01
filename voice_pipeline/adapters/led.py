"""LED 제어 어댑터.

- ``LEDController``: 상태(``LEDState``)별 애니메이션을 전용 스레드에서 렌더링. WS2812 드라이버
  (``rpi5_ws2812``)가 없거나 초기화에 실패하면 no-op으로 동작한다.
- ``StaticAnimation`` / ``BreathingAnimation``: 상태별 애니메이션 구현.
- ``OSLedArbiterClient``: OS LED arbiter 유닉스 소켓으로 제어권을 요청/반납.

하드웨어 셋업: docs/modules/led.md
"""

from __future__ import annotations

import contextlib
import enum
import logging
import math
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("voice_pipeline.led")


class LEDState(enum.Enum):
    """LED display states triggered by the pipeline.

    Implementations map these states to specific colors/animations.
    """

    OFF = "off"
    SLEEPING = "sleeping"
    IDLE = "idle"


# RGB tuple type alias
RGB = tuple[int, int, int]


@runtime_checkable
class LEDAnimation(Protocol):
    """Protocol for LED animations.

    Each LEDState maps to an animation. The controller calls render() on every
    tick of the animation thread, at a rate determined by frame_interval_sec.
    """

    @property
    def frame_interval_sec(self) -> float:
        """Seconds between render ticks."""
        ...

    def reset(self) -> None:
        """Called when this animation becomes the active animation (state entry)."""
        ...

    def render(self, tick: int, bar_count: int, ring_count: int) -> list[RGB]:
        """Produce one frame of LED colors.

        Args:
            tick: Monotonically increasing frame counter (reset to 0 on state entry).
            bar_count: Number of bar LEDs (first segment).
            ring_count: Number of ring LEDs (second segment).

        Returns:
            List of (R, G, B) tuples, length bar_count + ring_count.
        """
        ...


class StaticAnimation:
    """Fixed-color animation: bar and ring each hold a constant color.

    Useful as a baseline implementation and placeholder for states that
    don't need dynamic effects.

    Args:
        bar_color: 바 세그먼트 색상.
        ring_color: 링 세그먼트 색상.
    """

    _FRAME_INTERVAL_SEC = 0.1  # 렌더 틱 간격 (초)

    def __init__(
        self,
        bar_color: RGB = (0, 0, 0),
        ring_color: RGB = (0, 0, 0),
    ) -> None:
        self._bar_color = bar_color
        self._ring_color = ring_color

    @property
    def frame_interval_sec(self) -> float:
        return self._FRAME_INTERVAL_SEC

    def reset(self) -> None:
        pass

    def render(self, tick: int, bar_count: int, ring_count: int) -> list[RGB]:
        return [self._bar_color] * bar_count + [self._ring_color] * ring_count


class BreathingAnimation:
    """Smooth breathing (fade in/out) animation using a sine curve.

    Brightness oscillates between ``_MIN_BRIGHTNESS`` and 1.0 over
    ``_CYCLE_SEC`` seconds, applied to the base color for ring LEDs.
    Bar LEDs remain off.

    Args:
        color: 링 세그먼트 기본 색상.
    """

    _CYCLE_SEC = 4.0  # 페이드 한 주기 시간 (초)
    _MIN_BRIGHTNESS = 0.15  # 페이드 최소 밝기 (0.0~1.0)
    _FRAME_INTERVAL_SEC = 0.03  # 렌더 틱 간격 (초)

    def __init__(
        self,
        color: RGB = (233, 233, 50),
    ) -> None:
        self._color = color

    @property
    def frame_interval_sec(self) -> float:
        return self._FRAME_INTERVAL_SEC

    def reset(self) -> None:
        pass

    def render(self, tick: int, bar_count: int, ring_count: int) -> list[RGB]:
        t = tick * self._FRAME_INTERVAL_SEC
        # sine oscillates 0→1→0 over _CYCLE_SEC
        phase = (math.sin(2 * math.pi * t / self._CYCLE_SEC - math.pi / 2) + 1) / 2
        brightness = self._MIN_BRIGHTNESS + (1.0 - self._MIN_BRIGHTNESS) * phase
        r = int(self._color[0] * brightness)
        g = int(self._color[1] * brightness)
        b = int(self._color[2] * brightness)
        pixel: RGB = (r, g, b)
        return [(0, 0, 0)] * bar_count + [pixel] * ring_count


CONTROL_SOCK = "/run/os-led.sock"
_CONNECT_TIMEOUT_S = 1.0
_GRANT_TIMEOUT_S = 3.0
# OS_LED 데몬은 system 서비스(After=multi-user.target, 부팅 ~+23s)이고 RAY는 user 매니저
# 아래라 서로 순서를 걸 수 없다 — 부팅 직후엔 소켓이 아직 없을 수 있다. 그 창에서 standalone으로
# 폴백하면 두 프로세스가 같은 WS2812 라인을 동시에 써서 깜박이므로, 소켓을 기다린다.
# 얼마나 기다릴지는 데몬이 이 기기에 설치돼 있는지로 가른다:
#   설치됨(로봇 Pi)   → 부팅 기동 시각을 덮는 긴 창
#   미설치(개발기)    → 빠른 폴백, standalone이 의도된 모드
# 유닛 파일 존재가 설치 마커다 (world-readable이라 유저 세션에서 접근 가능).
_DAEMON_UNIT = "/etc/systemd/system/os-led-display.service"
_CONNECT_RETRY_S = 5.0
_CONNECT_RETRY_INSTALLED_S = 30.0
_CONNECT_RETRY_INTERVAL_S = 0.25


class OSLedArbiterClient:
    """Borrows the WS2812 strip from the OS_LED daemon (노란 호흡 담당)."""

    def __init__(self, sock_path: str = CONTROL_SOCK) -> None:
        self._sock_path = sock_path
        self._conn: socket.socket | None = None

    def acquire(self) -> None:
        """Borrow the strip from the OS_LED daemon.

        Blocks until the daemon has faded out and stopped driving SPI, so RAY
        can take over without interleaved frames. A missing/unreachable daemon
        is treated as "standalone" — RAY proceeds to drive the strip directly.
        """
        if self._conn is not None:
            return
        conn = self._connect_with_retry()
        if conn is None:
            return

        try:
            conn.sendall(b"ACQUIRE\n")
            conn.settimeout(_GRANT_TIMEOUT_S)
            resp = conn.recv(32)
        except OSError as exc:
            logger.warning("OS_LED arbiter handshake failed (%s) — standalone", exc)
            conn.close()
            return

        if b"GRANTED" not in resp:
            logger.warning("OS_LED arbiter did not grant — proceeding anyway")
        conn.settimeout(None)
        self._conn = conn
        logger.info("OS_LED strip acquired from OS_LED daemon")

    def _connect_with_retry(self) -> socket.socket | None:
        """Connect to the arbiter socket, retrying while it is merely absent.

        Returns None once the retry window is exhausted (daemon not installed or
        not up), which the caller treats as standalone mode.
        """
        installed = os.path.exists(_DAEMON_UNIT)
        window = _CONNECT_RETRY_INSTALLED_S if installed else _CONNECT_RETRY_S
        deadline = time.monotonic() + window
        attempt = 0
        while True:
            attempt += 1
            try:
                conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                conn.settimeout(_CONNECT_TIMEOUT_S)
                conn.connect(self._sock_path)
                if attempt > 1:
                    logger.info("OS_LED arbiter reached after %d attempts", attempt)
                return conn
            except (FileNotFoundError, ConnectionRefusedError):
                if attempt == 1 and installed:
                    logger.info("OS_LED daemon installed but socket absent — waiting up to %.0fs", window)
                if time.monotonic() >= deadline:
                    if installed:
                        logger.warning(
                            "OS_LED daemon installed but never came up within %.0fs — "
                            "driving strip standalone (expect contention if it starts later)",
                            window,
                        )
                    else:
                        logger.info("OS_LED arbiter not present — driving strip standalone")
                    return None
                time.sleep(_CONNECT_RETRY_INTERVAL_S)
            except OSError as exc:
                logger.warning("OS_LED arbiter connect failed (%s) — standalone", exc)
                return None

    def release(self) -> None:
        """Return the strip — the daemon resumes its yellow breathing."""
        if self._conn is None:
            return
        with contextlib.suppress(OSError):
            self._conn.sendall(b"RELEASE\n")
        with contextlib.suppress(OSError):
            self._conn.close()
        self._conn = None
        logger.info("OS_LED strip released back to OS_LED daemon")


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


class LEDController:
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
    _TRANSITION_SEC = 0.3  # 상태 전환 크로스페이드 시간 (초)
    _TRANSITION_STEPS = 10  # 크로스페이드 보간 스텝 수
    # 하부 LED (RP1 하드웨어 PWM 채널0 = GPIO12): 링 세그먼트 밝기를 매 프레임 duty로 미러.
    # 모든 출력이 _apply_frame을 통과하므로 호흡·정적·크로스페이드가 자동으로 따라온다.
    # 실측(2026-09-01, unit4): duty 10~100% 전 구간 선형 조광 확인 — 문턱 매핑 불필요.
    # sysfs가 없으면(setup 미실행 기기) noop.
    _LOWER_LED_PWM_DIR = "/sys/class/pwm/pwmchip0/pwm0"
    _LOWER_LED_MAX_PCT = 100  # 링 최대 밝기일 때의 duty (%) — 하부 밝기 상한

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
        self._last_frame: list[RGB] | None = None  # 마지막으로 실제 쓴 프레임 (크로스페이드 기점)
        self._lower_led_period: int | None = None  # pwm0 주기(ns). None = 하부 LED 미러 비활성
        self._lower_led_duty = -1  # 마지막으로 쓴 duty (동일 값 재쓰기 방지)

        # Hardware strip (None = noop fallback). When a real strip is used we
        # first borrow it from the OS_LED daemon (shared SPI bus).
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
            except RuntimeError:
                logger.error("LED strip init failed — continuing in noop mode", exc_info=True)

    def _init_strip(self) -> None:
        if not self._enabled:
            logger.info("LED disabled (enabled=False) — LED controller running in noop mode")
            return
        if _WS2812SpiDriver is None:
            logger.info("rpi5_ws2812 not available — LED controller running in noop mode")
            return
        # Borrow the shared strip from the OS_LED daemon before opening
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
            # 인수 직후 SLEEPING 첫 프레임을 즉시 그린다: SPI 오픈 시 라인 글리치로
            # 스트립이 임의 색(흰 반짝 등)을 래치해도 바로 덮어써지고, 애니메이션
            # 스레드가 돌기 전까지 공백이 없다. OS_LED 데몬도 같은 프레임으로 페이드해
            # 파킹하므로(handoff_frame) 부팅 호흡 → RAY 호흡이 끊김 없이 이어진다.
            self._apply_frame(BreathingAnimation().render(0, self._BAR_COUNT, self._RING_COUNT))
            # 하부 LED 미러 활성화 — period를 캐시하면 이후 _apply_frame마다 duty만 쓴다.
            # 로딩 중에는 스트립 인수 전이라 하부도 꺼져 있다.
            try:
                self._lower_led_period = int((Path(self._LOWER_LED_PWM_DIR) / "period").read_text())
                logger.info("Lower LED mirror enabled (period=%dns)", self._lower_led_period)
            except OSError:
                self._lower_led_period = None
                logger.debug("Lower LED PWM unavailable (suppressed)", exc_info=True)
            logger.info(
                "LED strip initialized: %d LEDs (bar=%d, ring=%d), brightness=%.2f",
                self._LED_COUNT,
                self._BAR_COUNT,
                self._RING_COUNT,
                self._brightness,
            )
        except Exception as exc:
            self._arbiter.release()  # hand the strip back to the OS_LED daemon
            raise RuntimeError(f"Failed to initialize LED strip: {exc}") from exc

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
        # 즉시 소등하면 종료가 "툭 꺼짐"으로 보인다 — 마지막 프레임에서 검정으로 페이드.
        if self._strip is not None and self._last_frame is not None:
            try:
                self._crossfade(self._last_frame, self._off_frame())
            except Exception:
                logger.debug("Close fade error (suppressed)", exc_info=True)
        self._apply_off()
        self._mirror_lower_led(self._off_frame())
        # Fully close our SPI device BEFORE releasing the token, so no RAY-side
        # write can overlap the daemon's breathing resume on the shared bus.
        self._close_strip()
        # Hand the strip back: the daemon resumes its yellow breathing.
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
        prev_state: LEDState | None = None
        while not self._stop_event.is_set():
            with self._lock:
                state = self._state
                anim = self._animations.get(state)
                tick = self._tick
                self._tick += 1

            if anim is None:
                self._apply_frame(self._off_frame())
                self._wait(self._NOOP_SLEEP_SEC)
                continue

            try:
                frame = anim.render(tick, self._BAR_COUNT, self._RING_COUNT)
                # 상태가 바뀐 첫 프레임은 즉시 점프 대신 이전 화면에서 크로스페이드
                # (예: 호흡 → 대화 정적 최대 밝기). 부팅 첫 프레임(prev_state None)은
                # OS_LED 데몬이 이미 같은 프레임으로 페이드해 파킹했으므로 제외.
                if prev_state is not None and state != prev_state and self._last_frame is not None:
                    self._crossfade(
                        self._last_frame,
                        frame,
                        abort=lambda st=state: self._stop_event.is_set() or self._state != st,
                    )
                prev_state = state
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
        self._last_frame = frame
        self._mirror_lower_led(frame)

    def _crossfade(self, start: list[RGB], end: list[RGB], abort=None) -> None:
        """start → end 프레임을 _TRANSITION_SEC에 걸쳐 선형 보간해 쓴다.

        상태 전환이 즉시 점프로 보이지 않게 하는 용도. ``abort``가 True를
        반환하면 중단한다 (전환 중 또 다른 상태 변경 대응).
        """
        step_dt = self._TRANSITION_SEC / self._TRANSITION_STEPS
        for i in range(1, self._TRANSITION_STEPS + 1):
            if abort is not None and abort():
                return
            t = i / self._TRANSITION_STEPS
            frame = [
                (
                    int(s0[0] + (e0[0] - s0[0]) * t),
                    int(s0[1] + (e0[1] - s0[1]) * t),
                    int(s0[2] + (e0[2] - s0[2]) * t),
                )
                for s0, e0 in zip(start, end, strict=True)
            ]
            self._apply_frame(frame)
            time.sleep(step_dt)

    def _apply_off(self) -> None:
        if self._strip is None:
            return
        try:
            self._strip.set_all_pixels(_Color(0, 0, 0))
            self._strip.show()
        except Exception:
            logger.debug("Error turning off LEDs (suppressed)", exc_info=True)

    def _mirror_lower_led(self, frame: list[RGB]) -> None:
        """링 세그먼트 밝기를 하부 LED(PWM0) duty로 미러. 미러 비활성이면 noop."""
        if self._lower_led_period is None:
            return
        ring = frame[self._BAR_COUNT :]
        level = max((max(px) for px in ring), default=0) / 255
        duty = int(self._lower_led_period * self._LOWER_LED_MAX_PCT * level) // 100
        if duty == self._lower_led_duty:
            return
        try:
            duty_file = Path(self._LOWER_LED_PWM_DIR) / "duty_cycle"
            if self._lower_led_duty < 0 and duty > 0:
                # 첫 점등 램프: 인수 페이드(0.7초)는 데몬이 그려서 미러가 볼 수 없어
                # 하부가 꺼진 채 시작한다 — 합류 순간 "팍" 켜지지 않게 0 → 목표로
                # 0.3초에 걸쳐 올린다. 애니메이션 스레드에서 실행되므로 프레임과 충돌 없음.
                for i in range(1, 11):
                    duty_file.write_text(str(duty * i // 10))
                    time.sleep(0.03)
            else:
                duty_file.write_text(str(duty))
            self._lower_led_duty = duty
        except OSError:
            logger.debug("Lower LED PWM write failed (suppressed)", exc_info=True)
            self._lower_led_period = None  # 반복 실패 방지 — 미러 비활성화
