"""Tests for the cascade block formatters (context_builder + session_context)."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from voice_pipeline.engines.cascade.context_builder import (
    format_carryover_block,
    format_memory_block,
    parse_citation_tag,
)
from voice_pipeline.memory.types import Episode, MemoryReadResult, Profile
from voice_pipeline.session_context import format_profile_block, format_session_summary_block

_NOW = datetime(2026, 3, 30, 9, 0, tzinfo=ZoneInfo("Asia/Seoul"))  # = 2026-03-30 00:00 UTC


def _ep(
    text: str = "User likes sci-fi.",
    timestamp: str = "2026-03-15 14:00:00",
    ep_id: int = 1,
) -> Episode:
    return Episode(
        id=ep_id,
        text=text,
        timestamp=timestamp,
        session_id="s1",
        importance=1.0,
        last_cited_at=timestamp,
    )


def _prof(topic: str, sub_topic: str, content: str) -> Profile:
    return Profile(id=1, topic=topic, sub_topic=sub_topic, content=content, updated_at="2026-03-15 14:00:00")


# ---------------------------------------------------------------------------
# format_profile_block
# ---------------------------------------------------------------------------


class TestFormatProfileBlock:
    def test_empty(self) -> None:
        assert format_profile_block([]) == ""

    def test_single_profile(self) -> None:
        result = format_profile_block([_prof("basic_info", "name", "Alice")])
        assert "[User Profile]" in result
        assert "basic_info::name: Alice" in result

    def test_sorted_by_topic_and_subtopic(self) -> None:
        profiles = [
            _prof("interest", "music", "Jazz"),
            _prof("basic_info", "name", "Alice"),
            _prof("interest", "movie", "SF"),
        ]
        result = format_profile_block(profiles)
        lines = result.strip().split("\n")
        # Header + 3 profiles
        assert len(lines) == 4
        # basic_info comes before interest
        assert "basic_info" in lines[1]
        # movie before music within interest
        assert "movie" in lines[2]
        assert "music" in lines[3]


# ---------------------------------------------------------------------------
# format_session_summary_block
# ---------------------------------------------------------------------------


class TestFormatSessionSummaryBlock:
    def test_with_episodes(self) -> None:
        episodes = [
            _ep("User talked about Dune.", "2026-03-28 14:00:00"),
            _ep("User mentioned moving to a new apartment.", "2026-03-28 14:05:00"),
        ]
        result = format_session_summary_block("2026-03-28 14:00:00", episodes, _NOW)
        assert result.startswith("[Session — 2026-03-28 (Sat) 23:00, 2 days ago]\n")
        assert "- User talked about Dune." in result
        assert "- User mentioned moving" in result

    def test_no_episodes(self) -> None:
        result = format_session_summary_block("2026-03-28 14:00:00", [], _NOW)
        assert "(no summary available)" in result


# ---------------------------------------------------------------------------
# format_carryover_block
# ---------------------------------------------------------------------------


class TestFormatCarryoverBlock:
    def test_without_summary(self) -> None:
        result = format_carryover_block("2026-03-28 21:30:00", _NOW)
        assert result == "[Previous session — 2026-03-29 (Sun) 06:30, yesterday]"

    def test_with_summary(self) -> None:
        result = format_carryover_block("2026-03-28 21:30:00", _NOW, "User planned a trip to Busan.")
        assert result.startswith("[Previous session — 2026-03-29 (Sun) 06:30, yesterday]\n")
        assert "Earlier in that session: User planned a trip to Busan." in result


# ---------------------------------------------------------------------------
# format_memory_block
# ---------------------------------------------------------------------------


class TestFormatMemoryBlock:
    def test_empty(self) -> None:
        result = format_memory_block(MemoryReadResult([], [], {}), _NOW)
        assert result == ""

    def test_single_episode(self) -> None:
        ep = _ep("User cried watching Interstellar.", "2026-03-15 14:00:00", ep_id=42)
        mem = MemoryReadResult([ep], [0.9], {1: 42})
        result = format_memory_block(mem, _NOW)
        assert "[Retrieved Memories]" in result
        assert "[M1] User cried watching Interstellar. (2026-03-15 (Sun) 23:00, 15 days ago)" in result

    def test_multiple_episodes(self) -> None:
        eps = [
            _ep("Episode one.", "2026-03-15 14:00:00", ep_id=1),
            _ep("Episode two.", "2026-03-20 10:00:00", ep_id=2),
        ]
        mem = MemoryReadResult(eps, [0.9, 0.8], {1: 1, 2: 2})
        result = format_memory_block(mem, _NOW)
        assert "[M1]" in result
        assert "[M2]" in result
        assert "(2026-03-15 (Sun) 23:00, 15 days ago)" in result
        assert "(2026-03-20 (Fri) 19:00, 10 days ago)" in result


# ---------------------------------------------------------------------------
# parse_citation_tag
# ---------------------------------------------------------------------------


class TestParseCitationTag:
    def test_single_citation(self) -> None:
        text = "Great movie!\n[MEMORIES: M1]"
        clean, cited = parse_citation_tag(text)
        assert clean == "Great movie!"
        assert cited == [1]

    def test_multiple_citations(self) -> None:
        text = "I remember that.\n[MEMORIES: M1, M2, M3]"
        clean, cited = parse_citation_tag(text)
        assert clean == "I remember that."
        assert cited == [1, 2, 3]

    def test_no_citation(self) -> None:
        text = "Just a normal response."
        clean, cited = parse_citation_tag(text)
        assert clean == "Just a normal response."
        assert cited == []

    def test_trailing_whitespace(self) -> None:
        text = "Response text\n[MEMORIES: M5]  \n"
        clean, cited = parse_citation_tag(text)
        assert clean == "Response text"
        assert cited == [5]

    def test_citation_with_spaces(self) -> None:
        text = "Text\n[MEMORIES:  M1 ,  M10 ]"
        clean, cited = parse_citation_tag(text)
        assert clean == "Text"
        assert cited == [1, 10]

    def test_tag_in_middle_does_not_match(self) -> None:
        text = "Before [MEMORIES: M1] after this."
        clean, cited = parse_citation_tag(text)
        assert clean == text
        assert cited == []

    def test_malformed_tag_no_m_prefix(self) -> None:
        text = "Text\n[MEMORIES: 1, 2]"
        clean, cited = parse_citation_tag(text)
        # Does not match the regex pattern (requires M prefix)
        assert clean == text
        assert cited == []
