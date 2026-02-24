"""Barge-in text truncation strategies.

Two implementations of IUtteranceTruncator:
- TimestampTruncator: uses word-level timestamps for precision.
- DurationRatioTruncator: estimates from total audio duration ratio.
"""

from __future__ import annotations

import math

from voice_pipeline.core.interfaces import IUtteranceTruncator
from voice_pipeline.core.types import WordTimestamp


class TimestampTruncator(IUtteranceTruncator):
    """Truncates text using word-level timestamps from TTS.

    Collects all words whose start time is before the stop position.
    """

    def truncate(
        self,
        text: str,
        stop_position_sec: float,
        timestamps: list[WordTimestamp],
    ) -> str:
        """Return the portion of text spoken before the stop point."""
        if not timestamps:
            return ""
        spoken = [ts.word for ts in timestamps if ts.start_sec < stop_position_sec]
        return " ".join(spoken)


class DurationRatioTruncator(IUtteranceTruncator):
    """Truncates text by estimating from the audio duration ratio.

    Used when word-level timestamps are not available.
    Ignores the timestamps parameter entirely.
    """

    def __init__(self, total_duration_sec: float) -> None:
        self._total_duration_sec = total_duration_sec

    def truncate(
        self,
        text: str,
        stop_position_sec: float,
        timestamps: list[WordTimestamp],
    ) -> str:
        """Return the estimated portion of text spoken before the stop point."""
        if not text:
            return ""
        if self._total_duration_sec <= 0 or stop_position_sec >= self._total_duration_sec:
            return text
        ratio = stop_position_sec / self._total_duration_sec
        words = text.split()
        count = math.ceil(ratio * len(words))
        return " ".join(words[:count])
