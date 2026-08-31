"""Unit tests for the LED controller module.

All tests run in noop fallback mode: an autouse fixture patches the SPI driver to
None and mocks the OS_LED arbiter client, so the suite is safe to run on a robot Pi
where both are really present. Tests needing a driver patch ``_DRIVER_PATH`` with a mock.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from voice_pipeline.adapters.led import (
    BreathingAnimation,
    LEDAnimation,
    LEDController,
    LEDState,
    StaticAnimation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DRIVER_PATH = "voice_pipeline.adapters.led._WS2812SpiDriver"


def _make_controller() -> LEDController:
    """Create a controller in noop mode (no hardware)."""
    return LEDController()


@pytest.fixture(autouse=True)
def _no_led_hardware():
    """Keep these tests off the real strip and the OS_LED daemon.

    On a robot Pi both rpi5_ws2812 and the os-led-display daemon are present, so an
    unpatched LEDController would open /dev/spidev0.0 and send ACQUIRE to
    /run/os-led.sock — the test run visibly blanks the robot's LEDs. Every test
    starts in noop mode; tests needing a driver patch _DRIVER_PATH themselves
    (the inner patch wins and the arbiter stays mocked).
    """
    with (
        patch(_DRIVER_PATH, None),
        patch("voice_pipeline.adapters.led.OSLedArbiterClient", MagicMock()),
    ):
        yield


# ===================================================================
# StaticAnimation
# ===================================================================


class TestStaticAnimation:
    def test_render_returns_correct_length(self) -> None:
        anim = StaticAnimation(bar_color=(10, 20, 30), ring_color=(40, 50, 60))
        frame = anim.render(0, bar_count=8, ring_count=16)
        assert len(frame) == 24

    def test_render_bar_and_ring_colors(self) -> None:
        bar = (255, 0, 0)
        ring = (0, 0, 255)
        anim = StaticAnimation(bar_color=bar, ring_color=ring)
        frame = anim.render(0, bar_count=3, ring_count=2)
        assert frame == [bar, bar, bar, ring, ring]

    def test_render_same_across_ticks(self) -> None:
        anim = StaticAnimation(bar_color=(1, 2, 3), ring_color=(4, 5, 6))
        f0 = anim.render(0, 2, 2)
        f1 = anim.render(100, 2, 2)
        assert f0 == f1

    def test_default_colors_are_black(self) -> None:
        anim = StaticAnimation()
        frame = anim.render(0, 1, 1)
        assert frame == [(0, 0, 0), (0, 0, 0)]

    def test_frame_interval_default(self) -> None:
        anim = StaticAnimation()
        assert anim.frame_interval_sec == 0.1

    def test_frame_interval_class_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(StaticAnimation, "_FRAME_INTERVAL_SEC", 0.5)
        anim = StaticAnimation()
        assert anim.frame_interval_sec == 0.5

    def test_reset_is_noop(self) -> None:
        anim = StaticAnimation(bar_color=(1, 2, 3))
        anim.reset()  # should not raise

    def test_satisfies_protocol(self) -> None:
        assert isinstance(StaticAnimation(), LEDAnimation)


# ===================================================================
# BreathingAnimation
# ===================================================================


class TestBreathingAnimation:
    def test_render_returns_correct_length(self) -> None:
        anim = BreathingAnimation()
        frame = anim.render(0, bar_count=8, ring_count=16)
        assert len(frame) == 24

    def test_bar_leds_are_off(self) -> None:
        anim = BreathingAnimation()
        frame = anim.render(5, bar_count=4, ring_count=8)
        for pixel in frame[:4]:
            assert pixel == (0, 0, 0)

    def test_ring_brightness_varies_with_tick(self) -> None:
        anim = BreathingAnimation(color=(233, 233, 50))
        frames = [anim.render(t, 0, 1)[0] for t in range(200)]
        # Not all frames should be the same — brightness changes
        unique = set(frames)
        assert len(unique) > 1

    def test_min_brightness_floor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(BreathingAnimation, "_MIN_BRIGHTNESS", 0.2)
        anim = BreathingAnimation(color=(100, 100, 100))
        # tick=0 → phase at minimum (sin starts at -pi/2 → phase=0)
        frame = anim.render(0, 0, 1)
        r, g, b = frame[0]
        assert r == 20  # 100 * 0.2
        assert g == 20
        assert b == 20

    def test_frame_interval_default(self) -> None:
        anim = BreathingAnimation()
        assert anim.frame_interval_sec == 0.03

    def test_satisfies_protocol(self) -> None:
        assert isinstance(BreathingAnimation(), LEDAnimation)


# ===================================================================
# LEDController — noop mode (no hardware driver)
# ===================================================================


class TestControllerNoop:
    def test_starts_in_off_state(self) -> None:
        ctrl = _make_controller()
        try:
            assert ctrl._state == LEDState.OFF
        finally:
            ctrl.close()

    def test_set_state_changes_state(self) -> None:
        ctrl = _make_controller()
        try:
            ctrl.set_state(LEDState.IDLE)
            assert ctrl._state == LEDState.IDLE
        finally:
            ctrl.close()

    def test_set_state_same_is_noop(self) -> None:
        """Setting the same state twice should not reset tick."""
        ctrl = _make_controller()
        try:
            ctrl.set_state(LEDState.IDLE)
            # Let a few ticks accumulate
            time.sleep(0.15)
            tick_before = ctrl._tick
            ctrl.set_state(LEDState.IDLE)
            # Tick should not have been reset to 0 by the second call
            assert ctrl._tick >= tick_before
        finally:
            ctrl.close()

    def test_set_state_resets_tick(self) -> None:
        ctrl = _make_controller()
        try:
            ctrl.set_state(LEDState.IDLE)
            time.sleep(0.5)
            assert ctrl._tick > 0
            ctrl.set_state(LEDState.SLEEPING)
            # Tick resets on state change; may have incremented slightly
            # but should be much smaller than before
            assert ctrl._tick < 5
        finally:
            ctrl.close()

    def test_close_stops_thread(self) -> None:
        ctrl = _make_controller()
        ctrl.close()
        assert not ctrl._thread.is_alive()

    def test_close_idempotent(self) -> None:
        ctrl = _make_controller()
        ctrl.close()
        ctrl.close()  # should not raise

    def test_all_states_accepted(self) -> None:
        ctrl = _make_controller()
        try:
            for state in LEDState:
                ctrl.set_state(state)
                assert ctrl._state == state
        finally:
            ctrl.close()

    def test_strip_is_none_without_hardware(self) -> None:
        with patch(_DRIVER_PATH, None):
            ctrl = _make_controller()
            try:
                assert ctrl._strip is None
            finally:
                ctrl.close()

    def test_animation_thread_is_daemon(self) -> None:
        ctrl = _make_controller()
        try:
            assert ctrl._thread.daemon
        finally:
            ctrl.close()


# ===================================================================
# LEDController — animation reset on state change
# ===================================================================


class TestAnimationReset:
    def test_reset_called_on_state_change(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_anim = MagicMock(spec=["reset", "render", "frame_interval_sec"])
        mock_anim.frame_interval_sec = 0.05
        mock_anim.render.return_value = [(0, 0, 0)] * 24

        animations = {state: StaticAnimation() for state in LEDState}
        animations[LEDState.IDLE] = mock_anim
        monkeypatch.setattr(LEDController, "_ANIMATIONS", animations)

        ctrl = _make_controller()
        try:
            ctrl.set_state(LEDState.IDLE)
            mock_anim.reset.assert_called_once()
        finally:
            ctrl.close()

    def test_render_called_after_state_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_anim = MagicMock(spec=["reset", "render", "frame_interval_sec"])
        mock_anim.frame_interval_sec = 0.02
        mock_anim.render.return_value = [(0, 0, 0)] * 24

        animations = {state: StaticAnimation() for state in LEDState}
        animations[LEDState.SLEEPING] = mock_anim
        monkeypatch.setattr(LEDController, "_ANIMATIONS", animations)

        ctrl = _make_controller()
        try:
            ctrl.set_state(LEDState.SLEEPING)
            time.sleep(0.5)
            assert mock_anim.render.call_count > 0
        finally:
            ctrl.close()

    def test_reset_called_each_transition(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_anim = MagicMock(spec=["reset", "render", "frame_interval_sec"])
        mock_anim.frame_interval_sec = 0.05
        mock_anim.render.return_value = [(0, 0, 0)] * 24

        animations = {state: StaticAnimation() for state in LEDState}
        animations[LEDState.IDLE] = mock_anim
        monkeypatch.setattr(LEDController, "_ANIMATIONS", animations)

        ctrl = _make_controller()
        try:
            ctrl.set_state(LEDState.IDLE)
            ctrl.set_state(LEDState.SLEEPING)
            ctrl.set_state(LEDState.IDLE)
            assert mock_anim.reset.call_count == 2
        finally:
            ctrl.close()


# ===================================================================
# LEDController — custom animations
# ===================================================================


class TestCustomAnimations:
    def test_custom_animation_map(self, monkeypatch: pytest.MonkeyPatch) -> None:
        custom = StaticAnimation(bar_color=(99, 99, 99), ring_color=(11, 11, 11))
        animations = {state: StaticAnimation() for state in LEDState}
        animations[LEDState.SLEEPING] = custom
        monkeypatch.setattr(LEDController, "_ANIMATIONS", animations)

        ctrl = _make_controller()
        try:
            ctrl.set_state(LEDState.SLEEPING)
            anim = ctrl._animations[LEDState.SLEEPING]
            assert anim is custom
        finally:
            ctrl.close()


# ===================================================================
# LEDController — hardware init error
# ===================================================================


class TestHardwareInit:
    def test_init_strip_raises_led_error(self) -> None:
        mock_driver_cls = MagicMock(side_effect=RuntimeError("SPI fail"))
        ctrl = _make_controller()
        try:
            with (
                patch(_DRIVER_PATH, mock_driver_cls),
                pytest.raises(RuntimeError, match="Failed to initialize LED strip"),
            ):
                ctrl._init_strip()
        finally:
            ctrl.close()

    def test_construction_does_not_touch_hardware(self) -> None:
        """The strip is borrowed lazily, so __init__ must not open the driver.

        Acquiring at construction time blanks the shared strip for the whole
        model-loading stretch (the OS_LED daemon stops drawing once we hold the
        token), which is why init moved to the first set_state().
        """
        mock_driver_cls = MagicMock()
        with patch(_DRIVER_PATH, mock_driver_cls):
            ctrl = _make_controller()
            try:
                mock_driver_cls.assert_not_called()
                ctrl.set_state(LEDState.SLEEPING)
                mock_driver_cls.assert_called_once()
            finally:
                ctrl.close()

    def test_lazy_init_failure_degrades_to_noop(self) -> None:
        """A failing strip must not kill the pipeline — set_state still works."""
        mock_driver_cls = MagicMock(side_effect=RuntimeError("SPI fail"))
        with patch(_DRIVER_PATH, mock_driver_cls):
            ctrl = _make_controller()
            try:
                ctrl.set_state(LEDState.SLEEPING)  # must not raise
                assert ctrl._strip is None
                ctrl.set_state(LEDState.IDLE)
                mock_driver_cls.assert_called_once()  # not retried every call
            finally:
                ctrl.close()


# ===================================================================
# LEDController — missing animation fallback
# ===================================================================


class TestMissingAnimation:
    def test_missing_animation_does_not_crash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """State with no registered animation should not raise."""
        # Only register OFF, leave IDLE unmapped
        animations = {LEDState.OFF: StaticAnimation()}
        monkeypatch.setattr(LEDController, "_ANIMATIONS", animations)

        ctrl = _make_controller()
        try:
            ctrl.set_state(LEDState.IDLE)
            time.sleep(0.15)  # let animation loop run a few ticks
            assert ctrl._state == LEDState.IDLE
        finally:
            ctrl.close()


# ===================================================================
# LEDController — render error resilience
# ===================================================================


class TestRenderError:
    def test_render_exception_suppressed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A render() that raises should not kill the animation thread."""
        bad_anim = MagicMock(spec=["reset", "render", "frame_interval_sec"])
        bad_anim.frame_interval_sec = 0.02
        bad_anim.render.side_effect = RuntimeError("render boom")

        animations = {state: StaticAnimation() for state in LEDState}
        animations[LEDState.IDLE] = bad_anim
        monkeypatch.setattr(LEDController, "_ANIMATIONS", animations)

        ctrl = _make_controller()
        try:
            ctrl.set_state(LEDState.IDLE)
            time.sleep(0.5)
            # Thread should still be alive despite render errors
            assert ctrl._thread.is_alive()
            # Should have retried render multiple times
            assert bad_anim.render.call_count > 1
        finally:
            ctrl.close()


# ===================================================================
# LEDController — state change responsiveness
# ===================================================================


class TestStateChangeResponsiveness:
    def test_set_state_wakes_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """set_state() should wake the animation thread, not wait for sleep to expire."""
        # Mock animations with controllable tick intervals. StaticAnimation은 tick이
        # 클래스 레벨 고정이라 두 인스턴스에 서로 다른 값을 주입할 수 없어 mock 사용.
        slow = MagicMock(spec=["reset", "render", "frame_interval_sec"])
        slow.frame_interval_sec = 1.0
        slow.render.return_value = [(0, 0, 0)] * 24
        fast = MagicMock(spec=["reset", "render", "frame_interval_sec"])
        fast.frame_interval_sec = 0.02
        fast.render.return_value = [(1, 2, 3)] * 24

        animations = {state: StaticAnimation() for state in LEDState}
        animations[LEDState.SLEEPING] = slow
        animations[LEDState.IDLE] = fast
        monkeypatch.setattr(LEDController, "_ANIMATIONS", animations)

        ctrl = _make_controller()
        try:
            ctrl.set_state(LEDState.SLEEPING)
            time.sleep(0.1)  # let it enter the slow sleep

            ctrl.set_state(LEDState.IDLE)
            time.sleep(0.5)
            # Thread should have picked up IDLE quickly, not after 1s
            assert ctrl._tick > 0
        finally:
            ctrl.close()
