"""LED animation protocol and built-in implementations."""

from __future__ import annotations

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
