"""Tests for voice_pipeline.engines.gpt_live.dj — 소재 로드, 재생 기록, 선택 규칙."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from voice_pipeline.engines.gpt_live.dj import Segment, SegmentLog, SegmentPicker, load_segments
from voice_pipeline.engines.gpt_live.songs import Song

SONGS = {"IAM": Song("IAM", "I AM", "IVE"), "snowman": Song("snowman", "눈사람", "정승환")}


def _write(dir_: Path, name: str, song: str = "IAM", topic: str = "주제", notes: str = "- 노트") -> None:
    (dir_ / f"{name}.md").write_text(f"---\nsong: {song}\ntopic: {topic}\n---\n\n{notes}\n", encoding="utf-8")


class TestLoadSegments:
    def test_loads_frontmatter_and_notes(self, tmp_path: Path) -> None:
        _write(tmp_path, "a", notes="- 첫 줄\n- 둘째 줄  # 주석 아님")
        segs = load_segments(tmp_path, SONGS)
        assert list(segs) == ["a"]
        seg = segs["a"]
        assert seg.topic == "주제" and seg.song is SONGS["IAM"] and seg.notes == "- 첫 줄\n- 둘째 줄  # 주석 아님"
        assert seg.song_label == "I AM — IVE"

    def test_missing_dir_or_empty_catalog_gives_nothing(self, tmp_path: Path) -> None:
        assert load_segments(tmp_path / "nope", SONGS) == {}
        _write(tmp_path, "a")
        assert load_segments(tmp_path, {}) == {}

    def test_unplayable_song_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        _write(tmp_path, "a", song="Nope")
        _write(tmp_path, "b")
        assert list(load_segments(tmp_path, SONGS)) == ["b"]

    @pytest.mark.parametrize(
        "text", ["no frontmatter", "---\nsong: IAM\n---\n\n- 노트\n", "---\nsong: IAM\ntopic: t\n---\n\n"]
    )
    def test_bad_file_raises(self, tmp_path: Path, text: str) -> None:
        (tmp_path / "bad.md").write_text(text, encoding="utf-8")
        with pytest.raises(RuntimeError, match="DJ segment load failed"):
            load_segments(tmp_path, SONGS)


class TestSegmentLog:
    def test_played_keys_oldest_first_by_last_play(self) -> None:
        log = SegmentLog(":memory:")
        assert log.played_keys() == []
        for key in ("a", "b", "a"):
            log.record(key)
        assert log.played_keys() == ["b", "a"]  # a 는 가장 최근에 다시 했으므로 뒤로


def _segments(*keys: str) -> dict[str, Segment]:
    return {k: Segment(k, f"topic {k}", SONGS["IAM"], "- n") for k in keys}


class TestSegmentPicker:
    def test_picks_unplayed_at_random_then_oldest(self) -> None:
        log = SegmentLog(":memory:")
        log.record("a")  # 전에 한 것
        picker = SegmentPicker(_segments("a", "b", "c"), log, rng=random.Random(1))
        first, second = picker.pick(), picker.pick()
        assert {first.key, second.key} == {"b", "c"}  # 한 적 없는 것부터, 세션 안에서 중복 없이
        assert picker.pick().key == "a"  # 전부 했으면 오래된 순
        assert log.played_keys()[-1] == "a"  # 고르면 기록된다

    def test_session_wraps_around_when_all_used(self) -> None:
        picker = SegmentPicker(_segments("a"), SegmentLog(":memory:"))
        assert picker.pick().key == "a"
        assert picker.pick().key == "a"  # 소재가 하나뿐이면 다시 그것

    def test_new_session_forgets_session_usage_but_not_log(self) -> None:
        log = SegmentLog(":memory:")
        picker = SegmentPicker(_segments("a", "b"), log, rng=random.Random(0))
        first = picker.pick()
        picker.new_session()
        assert picker.pick().key != first.key  # 로그에 남은 건 여전히 피한다

    def test_empty_segments_gives_none(self) -> None:
        assert SegmentPicker({}, SegmentLog(":memory:")).pick() is None
