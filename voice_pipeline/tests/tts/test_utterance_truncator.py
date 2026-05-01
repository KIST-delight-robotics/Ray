"""Tests for voice_pipeline.tts.utterance_truncator."""

from voice_pipeline.core.types import WordTimestamp
from voice_pipeline.tts.utterance_truncator import truncate_by_ratio, truncate_by_timestamps


class TestTruncateByTimestamps:
    def _make_timestamps(self) -> list[WordTimestamp]:
        return [
            WordTimestamp(word="hello", start_sec=0.0, end_sec=0.5),
            WordTimestamp(word="world", start_sec=0.5, end_sec=1.0),
            WordTimestamp(word="how", start_sec=1.0, end_sec=1.3),
            WordTimestamp(word="are", start_sec=1.3, end_sec=1.6),
            WordTimestamp(word="you", start_sec=1.6, end_sec=2.0),
        ]

    def test_truncate_midway(self) -> None:
        result = truncate_by_timestamps("hello world how are you", 1.2, self._make_timestamps())
        assert result == "hello world how"

    def test_truncate_all_words(self) -> None:
        result = truncate_by_timestamps("hello world how are you", 3.0, self._make_timestamps())
        assert result == "hello world how are you"

    def test_truncate_no_words(self) -> None:
        result = truncate_by_timestamps("hello world", 0.0, self._make_timestamps())
        assert result == ""

    def test_truncate_empty_timestamps(self) -> None:
        result = truncate_by_timestamps("hello world", 1.0, [])
        assert result == ""

    def test_truncate_exact_boundary(self) -> None:
        result = truncate_by_timestamps("hello world how are you", 0.5, self._make_timestamps())
        assert result == "hello"


class TestTruncateByRatio:
    def test_truncate_half(self) -> None:
        result = truncate_by_ratio("one two three four", 1.0, 2.0)
        assert result == "one two"

    def test_truncate_beyond_end(self) -> None:
        result = truncate_by_ratio("hello world", 3.0, 2.0)
        assert result == "hello world"

    def test_truncate_zero_duration(self) -> None:
        result = truncate_by_ratio("hello world", 1.0, 0.0)
        assert result == "hello world"

    def test_truncate_empty_text(self) -> None:
        result = truncate_by_ratio("", 1.0, 2.0)
        assert result == ""

    def test_truncate_small_ratio(self) -> None:
        result = truncate_by_ratio("one two three four", 1.0, 10.0)
        assert result == "one"

    def test_negative_duration_returns_full(self) -> None:
        result = truncate_by_ratio("hello world", 1.0, -1.0)
        assert result == "hello world"
