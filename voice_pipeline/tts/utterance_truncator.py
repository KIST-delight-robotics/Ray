"""Barge-in text truncation.

Two strategies:
- truncate_by_timestamps: uses word-level timestamps for precision.
- truncate_by_ratio: estimates from total audio duration ratio.
"""

from __future__ import annotations

import math

from voice_pipeline.core.types import WordTimestamp


def truncate_by_timestamps(
    text: str,
    stop_position_sec: float,
    timestamps: list[WordTimestamp],
) -> str:
    """Return the portion of text spoken before the stop point using word timestamps."""
    if not timestamps:
        return ""
    spoken = [ts.word for ts in timestamps if ts.start_sec < stop_position_sec]
    return " ".join(spoken)


def truncate_by_ratio(
    text: str,
    stop_position_sec: float,
    total_duration_sec: float,
) -> str:
    """Return the estimated portion of text spoken before the stop point using duration ratio."""
    if not text:
        return ""
    if total_duration_sec <= 0 or stop_position_sec >= total_duration_sec:
        return text
    ratio = stop_position_sec / total_duration_sec
    words = text.split()
    count = math.ceil(ratio * len(words))
    return " ".join(words[:count])
