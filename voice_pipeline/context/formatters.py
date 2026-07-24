"""Block formatters and citation parsing for LLM context assembly."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voice_pipeline.memory.types import Episode, MemoryReadResult, Profile

logger = logging.getLogger("voice_pipeline.context")

# Match "[MEMORIES: M1, M2, ...]" at the end of text (with optional trailing whitespace)
_CITATION_RE = re.compile(r"\[MEMORIES:\s*(M\d+(?:\s*,\s*M\d+)*)\s*\]\s*$")

# Markdown link [text](url) — including optional wrapping parentheses
_MD_LINK_WITH_PARENS_RE = re.compile(r"\(\[[^\]]*\]\([^)]*\)\)")
_MD_LINK_RE = re.compile(r"\[[^\]]*\]\([^)]*\)")


# ---------------------------------------------------------------------------
# Block 2: Profile
# ---------------------------------------------------------------------------


def format_profile_block(profiles: list[Profile]) -> str:
    """Format user profiles for LLM context injection (Block 2).

    Output example::

        [User Profile]
        basic_info::name: Alice
        interest::movie: SF, especially Nolan
    """
    if not profiles:
        return ""
    # Sort by (topic, sub_topic) for stable ordering
    sorted_profiles = sorted(profiles, key=lambda p: (p.topic, p.sub_topic))
    lines = ["[User Profile]"]
    for p in sorted_profiles:
        lines.append(f"{p.topic}::{p.sub_topic}: {p.content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Block 3: Previous session summaries
# ---------------------------------------------------------------------------


def format_session_summary_block(
    started_at: str,
    episodes: list[Episode],
) -> str:
    """Format a single previous session's episodes as a summary block.

    Args:
        started_at: Session start timestamp (UTC, '%Y-%m-%d %H:%M:%S').
        episodes: Episodes extracted from the session.

    Output example::

        [2026-03-28 14:00 session]
        - User talked about watching Dune 2 over the weekend.
        - User said the Interstellar OST is their favorite.
    """
    display_time = started_at[:16] if len(started_at) >= 16 else started_at
    header = f"[{display_time} session]"
    if not episodes:
        return f"{header}\n(no summary available)"
    lines = [header]
    for ep in episodes:
        lines.append(f"- {ep.text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Previous-session carryover
# ---------------------------------------------------------------------------


def format_carryover_block(started_at: str, summary_text: str | None = None) -> str:
    """Header opening the previous session's raw turns carried into context.

    Args:
        started_at: Previous session start timestamp (UTC, '%Y-%m-%d %H:%M:%S').
        summary_text: Persisted rolling summary covering the earlier part of
            that session, or None if it never summarized.

    Output example::

        [Previous session — 2026-03-28 21:30]
        Earlier in that session: User asked about weekend plans; Ray suggested
        a movie night. ...
    """
    display_time = started_at[:16] if len(started_at) >= 16 else started_at
    header = f"[Previous session — {display_time}]"
    if summary_text:
        return f"{header}\nEarlier in that session: {summary_text}"
    return header


def format_session_boundary(started_at: str) -> str:
    """Marker separating carried-over previous-session turns from the current session.

    Args:
        started_at: Current session start timestamp (UTC, '%Y-%m-%d %H:%M:%S').

    Output example::

        [New session — 2026-03-29 09:15]
    """
    display_time = started_at[:16] if len(started_at) >= 16 else started_at
    return f"[New session — {display_time}]"


# ---------------------------------------------------------------------------
# In-session history summary
# ---------------------------------------------------------------------------


def format_history_summary_block(summary_text: str) -> str:
    """Format the rolling summary of earlier turns in the current session.

    Shown in place of the turns it covers, right before the live history.

    Output example::

        [Earlier in this conversation]
        User asked about weekend plans; Ray suggested a movie night. ...
    """
    return f"[Earlier in this conversation]\n{summary_text}"


# ---------------------------------------------------------------------------
# Block 4: Retrieved memories
# ---------------------------------------------------------------------------


def format_memory_block(memory_result: MemoryReadResult) -> str:
    """Format retrieved memories for LLM context injection (Block 4).

    Output example::

        [Retrieved Memories]
        [M1] User cried watching Interstellar on a rainy day. (2026-03-15)
        [M2] User said Dune 2 was better than the original. (2026-03-20)
    """
    if not memory_result.episodes:
        return ""
    lines = ["[Retrieved Memories]"]
    for i, ep in enumerate(memory_result.episodes, 1):
        date = ep.timestamp[:10] if len(ep.timestamp) >= 10 else ep.timestamp
        lines.append(f"[M{i}] {ep.text} ({date})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Citation parsing
# ---------------------------------------------------------------------------


def strip_urls(text: str) -> str:
    """Remove markdown links from text.

    Handles both ``([text](url))`` and ``[text](url)`` forms.
    """
    text = _MD_LINK_WITH_PARENS_RE.sub("", text)
    text = _MD_LINK_RE.sub("", text)
    return re.sub(r"  +", " ", text).strip()


def parse_citation_tag(text: str) -> tuple[str, list[int]]:
    """Parse ``[MEMORIES: M1, M2]`` from the end of LLM output.

    Args:
        text: Raw LLM response text.

    Returns:
        Tuple of (clean_text, cited_indices) where cited_indices are
        1-based integers (e.g. [1, 3] for M1, M3). If no tag is found,
        returns (text, []).
    """
    match = _CITATION_RE.search(text)
    if not match:
        return (text, [])

    clean = text[: match.start()].rstrip()
    raw_indices = match.group(1).split(",")
    cited: list[int] = []
    for token in raw_indices:
        token = token.strip().lstrip("Mm")
        try:
            cited.append(int(token))
        except ValueError:
            continue
    return (clean, cited)
