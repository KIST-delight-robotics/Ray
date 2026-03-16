"""LED animation protocol and built-in implementations."""

from __future__ import annotations

import math

from typing import Protocol, runtime_checkable

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
    """

    def __init__(
        self,
        bar_color: RGB = (0, 0, 0),
        ring_color: RGB = (0, 0, 0),
        *,
        frame_interval_sec: float = 0.1,
    ) -> None:
        self._bar_color = bar_color
        self._ring_color = ring_color
        self._frame_interval_sec = frame_interval_sec

    @property
    def frame_interval_sec(self) -> float:
        return self._frame_interval_sec

    def reset(self) -> None:
        pass

    def render(self, tick: int, bar_count: int, ring_count: int) -> list[RGB]:
        return [self._bar_color] * bar_count + [self._ring_color] * ring_count


class BreathingAnimation:
    """Smooth breathing (fade in/out) animation using a sine curve.

    Brightness oscillates between ``min_brightness`` and 1.0 over
    ``cycle_sec`` seconds, applied to the base color for ring LEDs.
    Bar LEDs remain off.
    """

    def __init__(
        self,
        color: RGB = (233, 233, 50),
        *,
        cycle_sec: float = 4.0,
        min_brightness: float = 0.15,
        frame_interval_sec: float = 0.03,
    ) -> None:
        self._color = color
        self._cycle_sec = cycle_sec
        self._min_brightness = min_brightness
        self._frame_interval_sec = frame_interval_sec

    @property
    def frame_interval_sec(self) -> float:
        return self._frame_interval_sec

    def reset(self) -> None:
        pass

    def render(self, tick: int, bar_count: int, ring_count: int) -> list[RGB]:
        t = tick * self._frame_interval_sec
        # sine oscillates 0→1→0 over cycle_sec
        phase = (math.sin(2 * math.pi * t / self._cycle_sec - math.pi / 2) + 1) / 2
        brightness = self._min_brightness + (1.0 - self._min_brightness) * phase
        r = int(self._color[0] * brightness)
        g = int(self._color[1] * brightness)
        b = int(self._color[2] * brightness)
        pixel: RGB = (r, g, b)
        return [(0, 0, 0)] * bar_count + [pixel] * ring_count
