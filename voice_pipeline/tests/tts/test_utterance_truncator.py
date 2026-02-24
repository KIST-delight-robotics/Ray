"""Tests for voice_pipeline.tts.utterance_truncator."""

from voice_pipeline.core.types import WordTimestamp
from voice_pipeline.tts.utterance_truncator import (
    DurationRatioTruncator,
    TimestampTruncator,
)


class TestTimestampTruncator:
    def _make_timestamps(self) -> list[WordTimestamp]:
        return [
            WordTimestamp(word="hello", start_sec=0.0, end_sec=0.5),
            WordTimestamp(word="world", start_sec=0.5, end_sec=1.0),
            WordTimestamp(word="how", start_sec=1.0, end_sec=1.3),
            WordTimestamp(word="are", start_sec=1.3, end_sec=1.6),
            WordTimestamp(word="you", start_sec=1.6, end_sec=2.0),
        ]

    def test_truncate_midway(self) -> None:
        t = TimestampTruncator()
        # stop at 1.2: "hello"(0.0), "world"(0.5), "how"(1.0) all start before 1.2
        result = t.truncate("hello world how are you", 1.2, self._make_timestamps())
        assert result == "hello world how"

    def test_truncate_all_words(self) -> None:
        t = TimestampTruncator()
        result = t.truncate("hello world how are you", 3.0, self._make_timestamps())
        assert result == "hello world how are you"

    def test_truncate_no_words(self) -> None:
        t = TimestampTruncator()
        result = t.truncate("hello world", 0.0, self._make_timestamps())
        assert result == ""

    def test_truncate_empty_timestamps(self) -> None:
        t = TimestampTruncator()
        result = t.truncate("hello world", 1.0, [])
        assert result == ""

    def test_truncate_exact_boundary(self) -> None:
        t = TimestampTruncator()
        # stop at exactly 0.5 — "world" starts at 0.5, should NOT be included
        result = t.truncate("hello world how are you", 0.5, self._make_timestamps())
        assert result == "hello"


class TestDurationRatioTruncator:
    def test_truncate_half(self) -> None:
        t = DurationRatioTruncator(total_duration_sec=2.0)
        result = t.truncate("one two three four", 1.0, [])
        # ratio = 0.5, ceil(0.5 * 4) = 2 words
        assert result == "one two"

    def test_truncate_beyond_end(self) -> None:
        t = DurationRatioTruncator(total_duration_sec=2.0)
        result = t.truncate("hello world", 3.0, [])
        assert result == "hello world"

    def test_truncate_zero_duration(self) -> None:
        t = DurationRatioTruncator(total_duration_sec=0.0)
        result = t.truncate("hello world", 1.0, [])
        assert result == "hello world"

    def test_truncate_empty_text(self) -> None:
        t = DurationRatioTruncator(total_duration_sec=2.0)
        result = t.truncate("", 1.0, [])
        assert result == ""

    def test_truncate_small_ratio(self) -> None:
        t = DurationRatioTruncator(total_duration_sec=10.0)
        # ratio = 0.1, ceil(0.1 * 4) = 1 word
        result = t.truncate("one two three four", 1.0, [])
        assert result == "one"

    def test_ignores_timestamps(self) -> None:
        ts = [WordTimestamp(word="hello", start_sec=0.0, end_sec=0.5)]
        t = DurationRatioTruncator(total_duration_sec=2.0)
        result = t.truncate("hello world", 1.0, ts)
        # Should use ratio, not timestamps
        assert result == "hello"

    def test_negative_duration_returns_full(self) -> None:
        t = DurationRatioTruncator(total_duration_sec=-1.0)
        result = t.truncate("hello world", 1.0, [])
        assert result == "hello world"
