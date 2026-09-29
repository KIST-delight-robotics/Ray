"""Tests for voice_pipeline.engines.gpt_live.stories — 대본 로드, 후보 검색, 기록, 검색→시작 검증."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import pytest

from voice_pipeline.engines.gpt_live.songs import Song
from voice_pipeline.engines.gpt_live.stories import (
    Story,
    StoryLog,
    StorySelector,
    load_stories,
    search_stories,
)

CATALOG = {
    "snowman": Song("snowman", "눈사람", "정승환"),
    "IAM": Song("IAM", "I AM", "IVE"),
}


def _write(root: Path, key: str, *, song: str | None = None, summary: str = "요약", tags: str = "[겨울, 이별]") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    front = [f"song: {song or key}", "status: draft", f"summary: {summary}", f"tags: {tags}", "link_type: 정서"]
    body = (
        "## 곡 프로필\n- 핵심 정서: x\n\n## 도입부\n첫 문장.\n\n> (지침)\n\n"
        "## 본문\n이어지는 이야기.\n\n마지막 이야기와 노래 제안.\n> (play_song.)\n"
    )
    path = root / f"{key}.md"
    path.write_text("---\n" + "\n".join(front) + "\n---\n\n" + body, encoding="utf-8")
    return path


def _story(key: str, *, summary: str = "", tags: tuple[str, ...] = (), title: str = "") -> Story:
    return Story(key=key, song=key, summary=summary or key, tags=tags, body="본문", song_title=title)


class TestLoadStories:
    def test_frontmatter_fields_and_raw_body(self, tmp_path: Path) -> None:
        _write(tmp_path, "snowman", summary="보내주는 사람", tags="[영화, 이별, 겨울]")
        stories = load_stories(tmp_path, CATALOG)
        s = stories["snowman"]
        assert (s.song, s.summary, s.tags) == ("snowman", "보내주는 사람", ("영화", "이별", "겨울"))
        assert s.song_title == "눈사람" and s.song_artist == "정승환"
        assert s.body.startswith("## 곡 프로필") and "## 본문" in s.body and "---" not in s.body
        assert s.describe() == {
            "id": "snowman",
            "summary": "보내주는 사람",
            "tags": ["영화", "이별", "겨울"],
            "song": "눈사람 — 정승환",
        }

    def test_story_for_unplayable_song_is_skipped_with_warning(self, tmp_path: Path, caplog) -> None:
        _write(tmp_path, "snowman")
        _write(tmp_path, "ghost", song="NotInCatalog")
        with caplog.at_level(logging.WARNING, logger="voice_pipeline.stories"):
            stories = load_stories(tmp_path, CATALOG)
        assert set(stories) == {"snowman"}
        assert "ghost" in caplog.text and "NotInCatalog" in caplog.text

    def test_missing_dir_gives_empty(self, tmp_path: Path) -> None:
        assert load_stories(tmp_path / "nope", CATALOG) == {}

    def test_bad_frontmatter_raises(self, tmp_path: Path) -> None:
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / "x.md").write_text("no frontmatter\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="frontmatter"):
            load_stories(tmp_path, CATALOG)

    def test_missing_summary_raises(self, tmp_path: Path) -> None:
        _write(tmp_path, "snowman", summary="")
        with pytest.raises(RuntimeError, match="summary"):
            load_stories(tmp_path, CATALOG)

    def test_status_does_not_affect_loading(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "snowman")
        path.write_text(path.read_text(encoding="utf-8").replace("status: draft", "status: whatever"), encoding="utf-8")
        assert set(load_stories(tmp_path, CATALOG)) == {"snowman"}


class TestSearchStories:
    STORIES = {
        k: v
        for k, v in (
            ("a", _story("a", tags=("겨울", "이별"), title="눈사람")),
            ("b", _story("b", tags=("여름", "축제"))),
            ("c", _story("c", tags=("겨울", "기다림"))),
            ("d", _story("d", tags=("철학",))),
        )
    }

    def test_empty_query_returns_random_subset_of_untold(self) -> None:
        found = search_stories(self.STORIES, "", played=["b"], limit=2, rng=random.Random(1))
        assert len(found) == 2 and all(s.key != "b" for s in found)

    def test_keyword_matches_come_first_then_random_fill(self) -> None:
        found = search_stories(self.STORIES, "겨울 얘기", limit=3, rng=random.Random(0))
        assert {s.key for s in found[:2]} == {"a", "c"} and len(found) == 3

    def test_song_title_in_query_matches(self) -> None:
        found = search_stories(self.STORIES, "눈사람 이야기 해줘", limit=1)
        assert [s.key for s in found] == ["a"]

    def test_all_told_falls_back_to_oldest_first(self) -> None:
        found = search_stories(self.STORIES, "", played=["c", "a", "d", "b"], limit=2, rng=random.Random(0))
        assert {s.key for s in found} == {"c", "a"}

    def test_limit_caps_matches(self) -> None:
        assert len(search_stories(self.STORIES, "겨울", limit=1)) == 1


class TestStoryLog:
    def test_played_keys_ordered_by_last_play(self) -> None:
        log = StoryLog(":memory:")
        log.record("a")
        log.record("b")
        log.record("a")  # a 를 다시 → b 가 더 오래됨
        assert log.played_keys() == ["b", "a"]

    def test_empty(self) -> None:
        assert StoryLog(":memory:").played_keys() == []


class TestStorySelector:
    def test_begin_accepts_only_last_search_and_records(self) -> None:
        log = StoryLog(":memory:")
        sel = StorySelector(TestSearchStories.STORIES, log, limit=2, rng=random.Random(0))
        assert sel.begin("a") is None  # 검색 전
        found = sel.search("")
        key = found[0].key
        other = next(k for k in TestSearchStories.STORIES if k not in {s.key for s in found})
        assert sel.begin(other) is None
        assert sel.begin(key).key == key
        assert log.played_keys() == [key]
        assert sel.begin(key) is None  # 한 번 쓰면 다시 검색해야 한다

    def test_search_excludes_told(self) -> None:
        log = StoryLog(":memory:")
        sel = StorySelector(TestSearchStories.STORIES, log, limit=4)
        sel.begin(sel.search("")[0].key)
        told = log.played_keys()[0]
        assert told not in {s.key for s in sel.search("")}

    def test_replacement_excludes_current_even_when_all_stories_were_told(self) -> None:
        log = StoryLog(":memory:")
        stories = TestSearchStories.STORIES
        for key in stories:
            log.record(key)
        sel = StorySelector(stories, log, limit=4)
        found = sel.search("겨울", exclude=("a",))
        assert {s.key for s in found} == {"b", "c", "d"}
        assert sel.begin("a") is None
        assert sel.begin(found[0].key) is not None
        log.close()

    def test_only_current_story_leaves_no_replacement_or_stale_selection(self) -> None:
        log = StoryLog(":memory:")
        sel = StorySelector({"a": _story("a")}, log)
        sel.search("")
        assert sel.search("", exclude=("a",)) == []
        assert sel.begin("a") is None
        log.close()
