"""Tests for voice_pipeline.context.formatters."""

from __future__ import annotations

from voice_pipeline.context.formatters import (
    format_memory_block,
    format_profile_block,
    format_session_summary_block,
    parse_citation_tag,
)
from voice_pipeline.memory.types import Episode, MemoryReadResult, Profile


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
    return Profile(
        id=1, topic=topic, sub_topic=sub_topic, content=content, updated_at="2026-03-15 14:00:00"
    )


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
        result = format_session_summary_block("2026-03-28 14:00:00", episodes)
        assert "[2026-03-28 14:00 session]" in result
        assert "- User talked about Dune." in result
        assert "- User mentioned moving" in result

    def test_no_episodes(self) -> None:
        result = format_session_summary_block("2026-03-28 14:00:00", [])
        assert "(no summary available)" in result


# ---------------------------------------------------------------------------
# format_memory_block
# ---------------------------------------------------------------------------


class TestFormatMemoryBlock:
    def test_empty(self) -> None:
        result = format_memory_block(MemoryReadResult([], [], {}))
        assert result == ""

    def test_single_episode(self) -> None:
        ep = _ep("User cried watching Interstellar.", "2026-03-15 14:00:00", ep_id=42)
        mem = MemoryReadResult([ep], [0.9], {1: 42})
        result = format_memory_block(mem)
        assert "[Retrieved Memories]" in result
        assert "[M1] User cried watching Interstellar. (2026-03-15)" in result

    def test_multiple_episodes(self) -> None:
        eps = [
            _ep("Episode one.", "2026-03-15 14:00:00", ep_id=1),
            _ep("Episode two.", "2026-03-20 10:00:00", ep_id=2),
        ]
        mem = MemoryReadResult(eps, [0.9, 0.8], {1: 1, 2: 2})
        result = format_memory_block(mem)
        assert "[M1]" in result
        assert "[M2]" in result
        assert "(2026-03-15)" in result
        assert "(2026-03-20)" in result


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
