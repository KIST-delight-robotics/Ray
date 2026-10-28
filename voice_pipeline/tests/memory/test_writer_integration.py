"""Integration tests for MemoryWriter with real OpenAI LLM.

Requires OPENAI_API_KEY and a local embedding model.
"""

from __future__ import annotations

import pytest

from voice_pipeline.adapters.embedder import SentenceTransformerEmbedder
from voice_pipeline.adapters.llm_openai import OpenAILLM
from voice_pipeline.memory.storage import SQLiteMemoryStorage
from voice_pipeline.memory.types import UserNote
from voice_pipeline.memory.vector_index import NumpyVectorIndex
from voice_pipeline.memory.writer import MemoryWriter
from voice_pipeline.tests.memory.conftest import (
    CONVERSATION_MOVIE,
    CONVERSATION_PERSONAL,
    CONVERSATION_TRIVIAL,
    populate_utterances,
)
from voice_pipeline.types import ILLM

pytestmark = pytest.mark.requires_api


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_writer(
    storage: SQLiteMemoryStorage,
    index: NumpyVectorIndex,
    embedder: SentenceTransformerEmbedder,
    llm: ILLM,
) -> MemoryWriter:
    return MemoryWriter(storage, index, embedder, llm)


# ---------------------------------------------------------------------------
# Episode extraction
# ---------------------------------------------------------------------------


class TestEpisodeExtraction:
    """Episode extraction with real gpt-4o-mini."""

    def test_extracts_episodes_from_korean(
        self,
        memory_db: SQLiteMemoryStorage,
        vector_index: NumpyVectorIndex,
        shared_embedder: SentenceTransformerEmbedder,
        write_llm: OpenAILLM,
    ) -> None:
        """A meaningful Korean conversation yields at least one episode."""
        populate_utterances(memory_db, "s1", CONVERSATION_MOVIE)
        writer = _make_writer(memory_db, vector_index, shared_embedder, write_llm)

        episodes = writer.process_session("s1", "2026-04-01 10:00:00")

        assert len(episodes) >= 1, "Should extract at least one episode"
        for ep in episodes:
            assert ep.id is not None
            assert len(ep.text) > 10
            assert ep.session_id == "s1"
            assert ep.timestamp == "2026-04-01 10:00:00"
            assert ep.importance == 1.0

    def test_episode_is_third_person(
        self,
        memory_db: SQLiteMemoryStorage,
        vector_index: NumpyVectorIndex,
        shared_embedder: SentenceTransformerEmbedder,
        write_llm: OpenAILLM,
    ) -> None:
        """Episode text should not contain raw timestamps or role labels."""
        populate_utterances(memory_db, "s1", CONVERSATION_PERSONAL)
        writer = _make_writer(memory_db, vector_index, shared_embedder, write_llm)

        episodes = writer.process_session("s1", "2026-04-01 11:00:00")
        assert len(episodes) >= 1

        for ep in episodes:
            # Should not contain timestamp patterns from the transcript
            assert "[2026-" not in ep.text
            assert "2026-04-01" not in ep.text
            # Should not contain raw role labels
            assert "assistant:" not in ep.text.lower()

    def test_trivial_conversation_yields_empty(
        self,
        memory_db: SQLiteMemoryStorage,
        vector_index: NumpyVectorIndex,
        shared_embedder: SentenceTransformerEmbedder,
        write_llm: OpenAILLM,
    ) -> None:
        """A trivial greeting exchange yields zero episodes."""
        populate_utterances(memory_db, "s1", CONVERSATION_TRIVIAL)
        writer = _make_writer(memory_db, vector_index, shared_embedder, write_llm)

        episodes = writer.process_session("s1", "2026-04-01 13:00:00")
        # Trivial conversation: too few utterances (MemoryWriter._MIN_UTTERANCES = 2)
        # or LLM returns empty episodes list
        assert len(episodes) == 0

    def test_episodes_have_embeddings(
        self,
        memory_db: SQLiteMemoryStorage,
        vector_index: NumpyVectorIndex,
        shared_embedder: SentenceTransformerEmbedder,
        write_llm: OpenAILLM,
    ) -> None:
        """Extracted episodes have embeddings stored."""
        populate_utterances(memory_db, "s1", CONVERSATION_MOVIE)
        writer = _make_writer(memory_db, vector_index, shared_embedder, write_llm)

        episodes = writer.process_session("s1", "2026-04-01 10:00:00")
        assert len(episodes) >= 1

        for ep in episodes:
            assert ep.embedding is not None
            assert ep.embedding.shape == (384,)
            # Verify vector index also has the embedding
            assert len(vector_index) >= 1


# ---------------------------------------------------------------------------
# User notes
# ---------------------------------------------------------------------------


class TestUserNotes:
    """User-notes update with real LLM."""

    def test_identity_facts_become_profile_notes_but_tastes_do_not(
        self,
        memory_db: SQLiteMemoryStorage,
        vector_index: NumpyVectorIndex,
        shared_embedder: SentenceTransformerEmbedder,
        write_llm: OpenAILLM,
    ) -> None:
        """Seoul/programmer are identity facts; liking jazz is a taste and stays out."""
        populate_utterances(memory_db, "s1", CONVERSATION_PERSONAL)
        writer = _make_writer(memory_db, vector_index, shared_embedder, write_llm)

        writer.process_session("s1", "2026-04-01 11:00:00")

        notes = memory_db.get_user_notes()
        assert any(n.kind == "profile" for n in notes)
        assert not any(n.kind == "preference" for n in notes)
        assert not any("jazz" in n.text.lower() or "Evans" in n.text for n in notes)

    def test_unrelated_existing_note_is_kept(
        self,
        memory_db: SQLiteMemoryStorage,
        vector_index: NumpyVectorIndex,
        shared_embedder: SentenceTransformerEmbedder,
        write_llm: OpenAILLM,
    ) -> None:
        """A conversation that does not touch an existing note leaves it as is."""
        memory_db.add_user_note(UserNote(None, "preference", "Speak in 반말.", "s0", "2026-03-01 10:00:00"))

        populate_utterances(memory_db, "s1", CONVERSATION_PERSONAL)
        writer = _make_writer(memory_db, vector_index, shared_embedder, write_llm)
        writer.process_session("s1", "2026-04-01 11:00:00")

        assert "Speak in 반말." in [n.text for n in memory_db.get_user_notes()]


# ---------------------------------------------------------------------------
# Session processing status
# ---------------------------------------------------------------------------


class TestSessionProcessing:
    """Session processing lifecycle."""

    def test_session_marked_processed(
        self,
        memory_db: SQLiteMemoryStorage,
        vector_index: NumpyVectorIndex,
        shared_embedder: SentenceTransformerEmbedder,
        write_llm: OpenAILLM,
    ) -> None:
        """Session is marked as processed after write completes."""
        populate_utterances(memory_db, "s1", CONVERSATION_MOVIE)
        writer = _make_writer(memory_db, vector_index, shared_embedder, write_llm)

        writer.process_session("s1", "2026-04-01 10:00:00")

        processed = memory_db.get_processed_session_ids(["s1"])
        assert "s1" in processed

    def test_processed_session_detectable_by_caller(
        self,
        memory_db: SQLiteMemoryStorage,
        vector_index: NumpyVectorIndex,
        shared_embedder: SentenceTransformerEmbedder,
        write_llm: OpenAILLM,
    ) -> None:
        """After processing, the session ID is detectable via get_processed_session_ids.

        The writer marks sessions but does not self-skip — callers should
        check get_processed_session_ids() before calling process_session().
        """
        populate_utterances(memory_db, "s1", CONVERSATION_MOVIE)
        writer = _make_writer(memory_db, vector_index, shared_embedder, write_llm)

        # Not processed yet
        assert "s1" not in memory_db.get_processed_session_ids(["s1"])

        writer.process_session("s1", "2026-04-01 10:00:00")

        # Now marked — caller can skip
        assert "s1" in memory_db.get_processed_session_ids(["s1"])
