"""Tests for voice_pipeline.engines.gpt_live.dj — 소재 로드, 재생 기록, 선택 규칙."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from voice_pipeline.engines.gpt_live.dj import (
    Segment,
    SegmentLog,
    SegmentPicker,
    ends_with_closing_line,
    load_segments,
)
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


DRAWER = Song("Drawer_10cm", "서랍", "10CM", aliases=("Drawer",), artist_aliases=("십센치",))


class TestClosingLine:
    @pytest.mark.parametrize(
        "text",
        [
            "그런 마음을 담은 곡이 있어요.십센치의 서랍, 들려드릴게요",  # 마침표 없음 (2026-10-06 실험)
            "…이야기죠. 십센치의 서랍, 들려드릴게요.",
            "십센치의서랍, 들려드릴게요.",  # 전사 띄어쓰기 누락
            "10CM의 ‘서랍’, 들려드릴게요!",  # 원래 표기·따옴표
            "십센치의 Drawer, 들려 드릴게요.",  # 제목 별칭, 동사 띄어쓰기
        ],
    )
    def test_ending_with_the_closing_line(self, text: str) -> None:
        assert ends_with_closing_line(text, DRAWER)

    @pytest.mark.parametrize(
        "text",
        [
            "십센치의 서랍, 들려드릴게요. 노래가 시작되면 잠시 쉬어 가세요.",  # 문구 뒤에 말을 붙였다
            "오늘은 서랍 이야기를 들려드릴게요.",  # 소재 단어 + 동사만
            "이어서 들려드릴 곡은 십센치의 '서랍'이에요.",  # 소개했지만 문구가 아니다
            "박효신의 야생화, 들려드릴게요.",  # 다른 곡
            "",
        ],
    )
    def test_not_the_closing_line(self, text: str) -> None:
        assert not ends_with_closing_line(text, DRAWER)

    def test_song_without_artist_needs_only_the_title(self) -> None:
        song = Song("S", "제목만", "")
        assert ends_with_closing_line("제목만, 들려드릴게요.", song)
        assert not ends_with_closing_line("들려드릴게요.", song)
