"""Unit tests for the LED controller module.

Hardware driver is never present in CI — all tests exercise noop fallback mode.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from voice_pipeline.core.config import LEDConfig
from voice_pipeline.core.types import LEDState
from voice_pipeline.led.animations import LEDAnimation, StaticAnimation
from voice_pipeline.led.exceptions import LEDError
from voice_pipeline.led.led_controller import LEDController

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DRIVER_PATH = "voice_pipeline.led.led_controller._WS2812SpiDriver"


def _make_controller(
    config: LEDConfig | None = None,
    animations: dict[LEDState, LEDAnimation] | None = None,
) -> LEDController:
    """Create a controller in noop mode (no hardware)."""
    return LEDController(config or LEDConfig(), animations)


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

    def test_frame_interval_custom(self) -> None:
        anim = StaticAnimation(frame_interval_sec=0.5)
        assert anim.frame_interval_sec == 0.5

    def test_reset_is_noop(self) -> None:
        anim = StaticAnimation(bar_color=(1, 2, 3))
        anim.reset()  # should not raise

    def test_satisfies_protocol(self) -> None:
        assert isinstance(StaticAnimation(), LEDAnimation)


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
            ctrl.set_state(LEDState.LISTENING)
            assert ctrl._state == LEDState.LISTENING
        finally:
            ctrl.close()

    def test_set_state_same_is_noop(self) -> None:
        """Setting the same state twice should not reset tick."""
        ctrl = _make_controller()
        try:
            ctrl.set_state(LEDState.LISTENING)
            # Let a few ticks accumulate
            time.sleep(0.15)
            tick_before = ctrl._tick
            ctrl.set_state(LEDState.LISTENING)
            # Tick should not have been reset to 0 by the second call
            assert ctrl._tick >= tick_before
        finally:
            ctrl.close()

    def test_set_state_resets_tick(self) -> None:
        ctrl = _make_controller()
        try:
            ctrl.set_state(LEDState.LISTENING)
            time.sleep(0.15)
            assert ctrl._tick > 0
            ctrl.set_state(LEDState.THINKING)
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
    def test_reset_called_on_state_change(self) -> None:
        mock_anim = MagicMock(spec=["reset", "render", "frame_interval_sec"])
        mock_anim.frame_interval_sec = 0.05
        mock_anim.render.return_value = [(0, 0, 0)] * 24

        animations = {state: StaticAnimation() for state in LEDState}
        animations[LEDState.LISTENING] = mock_anim

        ctrl = _make_controller(animations=animations)
        try:
            ctrl.set_state(LEDState.LISTENING)
            mock_anim.reset.assert_called_once()
        finally:
            ctrl.close()

    def test_render_called_after_state_set(self) -> None:
        mock_anim = MagicMock(spec=["reset", "render", "frame_interval_sec"])
        mock_anim.frame_interval_sec = 0.02
        mock_anim.render.return_value = [(0, 0, 0)] * 24

        animations = {state: StaticAnimation() for state in LEDState}
        animations[LEDState.SPEAKING] = mock_anim

        ctrl = _make_controller(animations=animations)
        try:
            ctrl.set_state(LEDState.SPEAKING)
            time.sleep(0.15)
            assert mock_anim.render.call_count > 0
        finally:
            ctrl.close()

    def test_reset_called_each_transition(self) -> None:
        mock_anim = MagicMock(spec=["reset", "render", "frame_interval_sec"])
        mock_anim.frame_interval_sec = 0.05
        mock_anim.render.return_value = [(0, 0, 0)] * 24

        animations = {state: StaticAnimation() for state in LEDState}
        animations[LEDState.LISTENING] = mock_anim

        ctrl = _make_controller(animations=animations)
        try:
            ctrl.set_state(LEDState.LISTENING)
            ctrl.set_state(LEDState.THINKING)
            ctrl.set_state(LEDState.LISTENING)
            assert mock_anim.reset.call_count == 2
        finally:
            ctrl.close()


# ===================================================================
# LEDController — custom animations
# ===================================================================


class TestCustomAnimations:
    def test_custom_animation_map(self) -> None:
        custom = StaticAnimation(bar_color=(99, 99, 99), ring_color=(11, 11, 11))
        animations = {state: StaticAnimation() for state in LEDState}
        animations[LEDState.THINKING] = custom

        ctrl = _make_controller(animations=animations)
        try:
            ctrl.set_state(LEDState.THINKING)
            anim = ctrl._animations[LEDState.THINKING]
            assert anim is custom
        finally:
            ctrl.close()


# ===================================================================
# LEDController — hardware init error
# ===================================================================


class TestHardwareInit:
    def test_hardware_init_error_raises_led_error(self) -> None:
        mock_driver_cls = MagicMock(side_effect=RuntimeError("SPI fail"))
        with patch(_DRIVER_PATH, mock_driver_cls):
            with pytest.raises(LEDError, match="Failed to initialize LED strip"):
                _make_controller()


# ===================================================================
# LEDController — missing animation fallback
# ===================================================================


class TestMissingAnimation:
    def test_missing_animation_does_not_crash(self) -> None:
        """State with no registered animation should not raise."""
        # Only register OFF, leave LISTENING unmapped
        animations = {LEDState.OFF: StaticAnimation()}
        ctrl = _make_controller(animations=animations)
        try:
            ctrl.set_state(LEDState.LISTENING)
            time.sleep(0.15)  # let animation loop run a few ticks
            assert ctrl._state == LEDState.LISTENING
        finally:
            ctrl.close()


# ===================================================================
# LEDController — render error resilience
# ===================================================================


class TestRenderError:
    def test_render_exception_suppressed(self) -> None:
        """A render() that raises should not kill the animation thread."""
        bad_anim = MagicMock(spec=["reset", "render", "frame_interval_sec"])
        bad_anim.frame_interval_sec = 0.02
        bad_anim.render.side_effect = RuntimeError("render boom")

        animations = {state: StaticAnimation() for state in LEDState}
        animations[LEDState.LISTENING] = bad_anim

        ctrl = _make_controller(animations=animations)
        try:
            ctrl.set_state(LEDState.LISTENING)
            time.sleep(0.15)
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
    def test_set_state_wakes_thread(self) -> None:
        """set_state() should wake the animation thread, not wait for sleep to expire."""
        # Use a slow animation (1s interval) for the initial state
        slow = StaticAnimation(frame_interval_sec=1.0)
        fast = StaticAnimation(bar_color=(1, 2, 3), frame_interval_sec=0.02)

        animations = {state: StaticAnimation() for state in LEDState}
        animations[LEDState.SLEEPING] = slow
        animations[LEDState.LISTENING] = fast

        ctrl = _make_controller(animations=animations)
        try:
            ctrl.set_state(LEDState.SLEEPING)
            time.sleep(0.05)  # let it enter the slow sleep

            ctrl.set_state(LEDState.LISTENING)
            time.sleep(0.1)
            # Thread should have picked up LISTENING quickly, not after 1s
            assert ctrl._tick > 0
        finally:
            ctrl.close()


# ===================================================================
# LEDConfig
# ===================================================================


class TestLEDConfig:
    def test_defaults(self) -> None:
        cfg = LEDConfig()
        assert cfg.bar_count == 8
        assert cfg.ring_count == 16
        assert cfg.spi_pin == 10
        assert cfg.brightness == 128

    def test_custom_values(self) -> None:
        cfg = LEDConfig(bar_count=4, ring_count=8, spi_pin=12, brightness=50)
        assert cfg.bar_count == 4
        assert cfg.ring_count == 8
        assert cfg.spi_pin == 12
        assert cfg.brightness == 50
