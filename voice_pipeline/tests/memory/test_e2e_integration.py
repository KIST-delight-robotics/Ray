"""End-to-end integration tests: write → retrieve → format → citation.

Exercises the full memory pipeline with real LLM and embedding model.
Requires OPENAI_API_KEY.
"""

from __future__ import annotations

import pytest

from voice_pipeline.context.formatters import format_memory_block, parse_citation_tag
from voice_pipeline.core.config import MemoryConfig
from voice_pipeline.core.types import TokenCounter
from voice_pipeline.embedding.embedder import SentenceTransformerEmbedder
from voice_pipeline.llm.llm import OpenAILLM
from voice_pipeline.memory.retriever import MemoryRetriever
from voice_pipeline.memory.storage import SQLiteMemoryStorage
from voice_pipeline.memory.vector_index import NumpyVectorIndex
from voice_pipeline.memory.writer import MemoryWriter

from voice_pipeline.tests.memory.conftest import (
    CONVERSATION_COOKING,
    CONVERSATION_MOVIE,
    populate_utterances,
)

pytestmark = pytest.mark.requires_api


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_session(
    storage: SQLiteMemoryStorage,
    index: NumpyVectorIndex,
    embedder: SentenceTransformerEmbedder,
    llm: OpenAILLM,
    config: MemoryConfig,
    token_counter: TokenCounter,
    session_id: str,
    timestamp: str,
    conversation: list[tuple[str, str, str, int]],
):
    populate_utterances(storage, session_id, conversation)
    writer = MemoryWriter(storage, index, embedder, llm, config, token_counter)
    return writer.process_session(session_id, timestamp)


# ---------------------------------------------------------------------------
# Write → Retrieve
# ---------------------------------------------------------------------------


class TestWriteThenRetrieve:
    """Full cycle: store utterances → extract episodes → embed → retrieve."""

    def test_full_cycle(
        self,
        memory_db: SQLiteMemoryStorage,
        vector_index: NumpyVectorIndex,
        shared_embedder: SentenceTransformerEmbedder,
        write_llm: OpenAILLM,
        memory_config: MemoryConfig,
        token_counter: TokenCounter,
    ) -> None:
        """Written episodes are retrievable by a relevant query."""
        episodes = _write_session(
            memory_db, vector_index, shared_embedder, write_llm,
            memory_config, token_counter,
            "s-movie", "2026-04-01 10:00:00", CONVERSATION_MOVIE,
        )
        assert len(episodes) >= 1

        retriever = MemoryRetriever(memory_db, vector_index, shared_embedder, memory_config)
        result = retriever.retrieve("인터스텔라 영화", set())

        assert len(result.episodes) >= 1
        assert len(result.scores) == len(result.episodes)
        assert all(s > 0 for s in result.scores)
        # Index mapping is well-formed
        for idx, eid in result.index_to_id.items():
            assert isinstance(idx, int) and idx >= 1
            assert isinstance(eid, int) and eid > 0

    def test_two_sessions_relevance(
        self,
        memory_db: SQLiteMemoryStorage,
        vector_index: NumpyVectorIndex,
        shared_embedder: SentenceTransformerEmbedder,
        write_llm: OpenAILLM,
        memory_config: MemoryConfig,
        token_counter: TokenCounter,
    ) -> None:
        """Movie query ranks movie-session episodes above cooking-session episodes."""
        movie_eps = _write_session(
            memory_db, vector_index, shared_embedder, write_llm,
            memory_config, token_counter,
            "s-movie", "2026-04-01 10:00:00", CONVERSATION_MOVIE,
        )
        cook_eps = _write_session(
            memory_db, vector_index, shared_embedder, write_llm,
            memory_config, token_counter,
            "s-cook", "2026-04-01 12:00:00", CONVERSATION_COOKING,
        )
        assert len(movie_eps) >= 1 and len(cook_eps) >= 1

        retriever = MemoryRetriever(memory_db, vector_index, shared_embedder, memory_config)
        result = retriever.retrieve("영화를 봤어", set())

        if len(result.episodes) >= 2:
            movie_scores = [
                result.scores[i]
                for i, ep in enumerate(result.episodes)
                if ep.session_id == "s-movie"
            ]
            cook_scores = [
                result.scores[i]
                for i, ep in enumerate(result.episodes)
                if ep.session_id == "s-cook"
            ]
            if movie_scores and cook_scores:
                assert max(movie_scores) > max(cook_scores), (
                    "Best movie score should exceed best cooking score for a movie query"
                )


# ---------------------------------------------------------------------------
# Context formatting & citation
# ---------------------------------------------------------------------------


class TestContextIntegration:
    """format_memory_block and citation parsing with real data."""

    def test_memory_block_format(
        self,
        memory_db: SQLiteMemoryStorage,
        vector_index: NumpyVectorIndex,
        shared_embedder: SentenceTransformerEmbedder,
        write_llm: OpenAILLM,
        memory_config: MemoryConfig,
        token_counter: TokenCounter,
    ) -> None:
        """format_memory_block produces [M1] tags from real retrieval results."""
        _write_session(
            memory_db, vector_index, shared_embedder, write_llm,
            memory_config, token_counter,
            "s1", "2026-04-01 10:00:00", CONVERSATION_MOVIE,
        )

        retriever = MemoryRetriever(memory_db, vector_index, shared_embedder, memory_config)
        result = retriever.retrieve("영화", set())
        assert len(result.episodes) >= 1

        formatted = format_memory_block(result)
        assert "[Retrieved Memories]" in formatted
        assert "[M1]" in formatted
        assert "(2026-04-01)" in formatted

    def test_citation_roundtrip(
        self,
        memory_db: SQLiteMemoryStorage,
        vector_index: NumpyVectorIndex,
        shared_embedder: SentenceTransformerEmbedder,
        write_llm: OpenAILLM,
        memory_config: MemoryConfig,
        token_counter: TokenCounter,
    ) -> None:
        """Full citation roundtrip: format → simulated LLM response → parse → update."""
        _write_session(
            memory_db, vector_index, shared_embedder, write_llm,
            memory_config, token_counter,
            "s1", "2026-04-01 10:00:00", CONVERSATION_MOVIE,
        )

        retriever = MemoryRetriever(memory_db, vector_index, shared_embedder, memory_config)
        result = retriever.retrieve("영화", set())
        assert len(result.episodes) >= 1

        # Simulate LLM response with citation tag
        simulated_response = "좋은 영화였죠!\n[MEMORIES: M1]"
        clean_text, cited_indices = parse_citation_tag(simulated_response)

        assert clean_text == "좋은 영화였죠!"
        assert cited_indices == [1]

        # Resolve display index to database ID
        db_id = result.index_to_id.get(1)
        assert db_id is not None

        # Update citations
        retriever.update_citations(cited_indices)

        # Verify retained buffer was updated
        assert db_id in retriever._retained
        assert retriever._retained[db_id].ttl == memory_config.retained_ttl
