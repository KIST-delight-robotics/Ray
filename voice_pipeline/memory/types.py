"""Data types for the long-term memory system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

NoteKind = Literal["profile", "preference"]
NOTE_KINDS: tuple[NoteKind, ...] = ("profile", "preference")


@dataclass
class Episode:
    """A single episodic memory extracted from a conversation session.

    Attributes:
        id: Database primary key. None for episodes not yet persisted.
        text: Third-person narrative describing the episode.
        timestamp: When the episode occurred (UTC, '%Y-%m-%d %H:%M:%S').
        session_id: Reference to the source session.
        importance: Qualitative importance judged by LLM, in [0, 1].
        last_cited_at: Last time this memory was cited in conversation
            (UTC, '%Y-%m-%d %H:%M:%S'). Initial value equals timestamp.
        citation_count: Number of times this memory has been cited.
            Reserved for future use (reinforcement signal).
        embedding: Dense vector for semantic search. None if not yet computed.
    """

    id: int | None
    text: str
    timestamp: str
    session_id: str
    importance: float
    last_cited_at: str
    citation_count: int = 0
    embedding: np.ndarray | None = None


@dataclass
class UserNote:
    """One line of what Ray always knows about the user (Block 2).

    Attributes:
        id: Database primary key. None for notes not yet persisted.
        kind: ``"profile"`` — who the user is; ``"preference"`` — how the
            user wants Ray to talk, written as an instruction to Ray.
        text: One short sentence.
        session_id: Session whose update last wrote this note.
        updated_at: Last update timestamp (UTC, '%Y-%m-%d %H:%M:%S').
    """

    id: int | None
    kind: NoteKind
    text: str
    session_id: str
    updated_at: str


@dataclass
class MemoryReadResult:
    """Result of a memory retrieval query.

    Attributes:
        episodes: Ranked episodes for block 4 injection
            (retained first, then new search results).
        scores: Salience scores, 1:1 with episodes.
        index_to_id: Mapping from 1-based display index (M1, M2, ...)
            to episode database ID. Used for citation resolution.
    """

    episodes: list[Episode]
    scores: list[float]
    index_to_id: dict[int, int]
