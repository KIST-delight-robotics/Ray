"""Integration tests for MemoryWriter with real OpenAI LLM.

Requires OPENAI_API_KEY and a local embedding model.
"""

from __future__ import annotations

from typing import Any

import pytest

from voice_pipeline.core.config import MemoryConfig
from voice_pipeline.core.interfaces import ILLM
from voice_pipeline.core.types import TokenCounter
from voice_pipeline.embedding.embedder import SentenceTransformerEmbedder
from voice_pipeline.llm.llm import OpenAILLM
from voice_pipeline.memory.prompts import PROFILE_SCHEMA
from voice_pipeline.memory.storage import SQLiteMemoryStorage
from voice_pipeline.memory.types import Profile
from voice_pipeline.memory.vector_index import NumpyVectorIndex
from voice_pipeline.memory.writer import MemoryWriter

from voice_pipeline.tests.memory.conftest import (
    CONVERSATION_COOKING,
    CONVERSATION_MOVIE,
    CONVERSATION_PERSONAL,
    CONVERSATION_TRIVIAL,
    populate_utterances,
)

pytestmark = pytest.mark.requires_api


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_TOPICS = set(PROFILE_SCHEMA.keys())


def _make_writer(
    storage: SQLiteMemoryStorage,
    index: NumpyVectorIndex,
    embedder: SentenceTransformerEmbedder,
    llm: ILLM,
    config: MemoryConfig,
    token_counter: TokenCounter,
) -> MemoryWriter:
    return MemoryWriter(storage, index, embedder, llm, config, token_counter)


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
        memory_config: MemoryConfig,
        token_counter: TokenCounter,
    ) -> None:
        """A meaningful Korean conversation yields at least one episode."""
        populate_utterances(memory_db, "s1", CONVERSATION_MOVIE)
        writer = _make_writer(memory_db, vector_index, shared_embedder, write_llm, memory_config, token_counter)

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
        memory_config: MemoryConfig,
        token_counter: TokenCounter,
    ) -> None:
        """Episode text should not contain raw timestamps or role labels."""
        populate_utterances(memory_db, "s1", CONVERSATION_PERSONAL)
        writer = _make_writer(memory_db, vector_index, shared_embedder, write_llm, memory_config, token_counter)

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
        memory_config: MemoryConfig,
        token_counter: TokenCounter,
    ) -> None:
        """A trivial greeting exchange yields zero episodes."""
        populate_utterances(memory_db, "s1", CONVERSATION_TRIVIAL)
        writer = _make_writer(memory_db, vector_index, shared_embedder, write_llm, memory_config, token_counter)

        episodes = writer.process_session("s1", "2026-04-01 13:00:00")
        # Trivial conversation: too few utterances (_MIN_UTTERANCES = 2) or
        # LLM returns empty episodes list
        assert len(episodes) == 0

    def test_episodes_have_embeddings(
        self,
        memory_db: SQLiteMemoryStorage,
        vector_index: NumpyVectorIndex,
        shared_embedder: SentenceTransformerEmbedder,
        write_llm: OpenAILLM,
        memory_config: MemoryConfig,
        token_counter: TokenCounter,
    ) -> None:
        """Extracted episodes have embeddings stored."""
        populate_utterances(memory_db, "s1", CONVERSATION_MOVIE)
        writer = _make_writer(memory_db, vector_index, shared_embedder, write_llm, memory_config, token_counter)

        episodes = writer.process_session("s1", "2026-04-01 10:00:00")
        assert len(episodes) >= 1

        for ep in episodes:
            assert ep.embedding is not None
            assert ep.embedding.shape == (384,)
            # Verify vector index also has the embedding
            assert len(vector_index) >= 1


# ---------------------------------------------------------------------------
# Profile extraction
# ---------------------------------------------------------------------------


class TestProfileExtraction:
    """Profile extraction and merge with real LLM."""

    def test_extracts_valid_profile_topics(
        self,
        memory_db: SQLiteMemoryStorage,
        vector_index: NumpyVectorIndex,
        shared_embedder: SentenceTransformerEmbedder,
        write_llm: OpenAILLM,
        memory_config: MemoryConfig,
        token_counter: TokenCounter,
    ) -> None:
        """Extracted profiles have valid topics from PROFILE_SCHEMA."""
        # CONVERSATION_PERSONAL mentions: jazz (interest), Seoul (basic_info), programmer (basic_info)
        populate_utterances(memory_db, "s1", CONVERSATION_PERSONAL)
        writer = _make_writer(memory_db, vector_index, shared_embedder, write_llm, memory_config, token_counter)

        writer.process_session("s1", "2026-04-01 11:00:00")

        profiles = memory_db.get_all_profiles()
        # LLM should extract at least some profile facts
        if profiles:
            for p in profiles:
                assert p.topic in _VALID_TOPICS, f"Unknown topic: {p.topic}"
                assert len(p.sub_topic) > 0
                assert len(p.content) > 0

    def test_profile_merge_preserves_existing(
        self,
        memory_db: SQLiteMemoryStorage,
        vector_index: NumpyVectorIndex,
        shared_embedder: SentenceTransformerEmbedder,
        write_llm: OpenAILLM,
        memory_config: MemoryConfig,
        token_counter: TokenCounter,
    ) -> None:
        """Merging new facts does not delete existing profiles."""
        # Pre-populate an existing profile
        memory_db.upsert_profile(
            Profile(
                id=None,
                topic="interest",
                sub_topic="movie",
                content="인터스텔라를 좋아함",
                updated_at="2026-03-01 10:00:00",
            )
        )
        profiles_before = memory_db.get_all_profiles()

        # Process a conversation that mentions music (new topic)
        populate_utterances(memory_db, "s1", CONVERSATION_PERSONAL)
        writer = _make_writer(memory_db, vector_index, shared_embedder, write_llm, memory_config, token_counter)
        writer.process_session("s1", "2026-04-01 11:00:00")

        profiles_after = memory_db.get_all_profiles()
        # The existing movie profile should still be present
        movie_profiles = [p for p in profiles_after if p.sub_topic == "movie"]
        assert len(movie_profiles) >= 1, "Existing movie profile should not be deleted"
        # Total profiles should not decrease
        assert len(profiles_after) >= len(profiles_before)


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
        memory_config: MemoryConfig,
        token_counter: TokenCounter,
    ) -> None:
        """Session is marked as processed after write completes."""
        populate_utterances(memory_db, "s1", CONVERSATION_MOVIE)
        writer = _make_writer(memory_db, vector_index, shared_embedder, write_llm, memory_config, token_counter)

        writer.process_session("s1", "2026-04-01 10:00:00")

        processed = memory_db.get_processed_session_ids(["s1"])
        assert "s1" in processed

    def test_processed_session_detectable_by_caller(
        self,
        memory_db: SQLiteMemoryStorage,
        vector_index: NumpyVectorIndex,
        shared_embedder: SentenceTransformerEmbedder,
        write_llm: OpenAILLM,
        memory_config: MemoryConfig,
        token_counter: TokenCounter,
    ) -> None:
        """After processing, the session ID is detectable via get_processed_session_ids.

        The writer marks sessions but does not self-skip — callers should
        check get_processed_session_ids() before calling process_session().
        """
        populate_utterances(memory_db, "s1", CONVERSATION_MOVIE)
        writer = _make_writer(memory_db, vector_index, shared_embedder, write_llm, memory_config, token_counter)

        # Not processed yet
        assert "s1" not in memory_db.get_processed_session_ids(["s1"])

        writer.process_session("s1", "2026-04-01 10:00:00")

        # Now marked — caller can skip
        assert "s1" in memory_db.get_processed_session_ids(["s1"])
