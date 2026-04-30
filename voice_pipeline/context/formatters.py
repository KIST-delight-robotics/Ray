"""Block formatters, citation parsing, and session context loading for LLM context assembly."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from voice_pipeline.core.interfaces import IMemoryStorage, IStorageBackend

if TYPE_CHECKING:
    from voice_pipeline.memory.types import Episode, MemoryReadResult, Profile

logger = logging.getLogger("voice_pipeline.context")

# Match "[MEMORIES: M1, M2, ...]" at the end of text (with optional trailing whitespace)
_CITATION_RE = re.compile(r"\[MEMORIES:\s*(M\d+(?:\s*,\s*M\d+)*)\s*\]\s*$")


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


def format_raw_transcript_block(
    started_at: str,
    utterances: list[tuple[str, str, str, int]],
) -> str:
    """Format raw utterances as a session block (fallback when no episodes).

    Used for sessions not yet processed by MemoryWriter.

    Args:
        started_at: Session start timestamp (UTC, '%Y-%m-%d %H:%M:%S').
        utterances: List of (role, text, timestamp, token_count) tuples.

    Output example::

        [2026-03-28 21:30 session]
        User: What time is it?
        Ray: It's 9:30.
    """
    display_time = started_at[:16] if len(started_at) >= 16 else started_at
    header = f"[{display_time} session]"
    lines = [header]
    for role, text, _ts, _tc in utterances:
        label = "User" if role == "user" else "Ray"
        lines.append(f"{label}: {text}")
    return "\n".join(lines)


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


# ---------------------------------------------------------------------------
# Session context loading
# ---------------------------------------------------------------------------


def load_session_context(
    memory_storage: IMemoryStorage,
    storage: IStorageBackend,
    session_id: str,
    recent_count: int,
) -> tuple[list[Profile], list[str], set[str]]:
    """Load previous session context for LLM injection.

    Returns:
        (profiles, session_summaries, exclude_session_ids)
    """
    profiles = memory_storage.get_all_profiles()
    recent = storage.get_recent_sessions(recent_count, exclude_session_id=session_id)
    recent_session_ids = [s[0] for s in recent]
    session_episodes = memory_storage.get_episodes_by_session_ids(recent_session_ids)
    processed_ids = memory_storage.get_processed_session_ids(recent_session_ids)

    session_summaries: list[str] = []
    for sid, started_at, _ended_at in recent:
        episodes = session_episodes.get(sid, [])
        if episodes:
            session_summaries.append(format_session_summary_block(started_at, episodes))
        elif sid in processed_ids:
            continue
        else:
            utterances = memory_storage.get_utterances(sid)
            if utterances:
                session_summaries.append(format_raw_transcript_block(started_at, utterances))

    exclude_session_ids = {session_id} | set(recent_session_ids)
    return profiles, session_summaries, exclude_session_ids
