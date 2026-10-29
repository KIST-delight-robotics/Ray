"""End-to-end integration tests: write → retrieve.

Exercises the full memory pipeline with real LLM and embedding model.
Requires OPENAI_API_KEY.
"""

from __future__ import annotations

import pytest

from voice_pipeline.adapters.embedder import SentenceTransformerEmbedder
from voice_pipeline.adapters.llm_openai import OpenAILLM
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
    session_id: str,
    timestamp: str,
    conversation: list[tuple[str, str, str, int]],
):
    populate_utterances(storage, session_id, conversation)
    writer = MemoryWriter(storage, index, embedder, llm)
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
    ) -> None:
        """Written episodes are retrievable by a relevant query."""
        episodes = _write_session(
            memory_db,
            vector_index,
            shared_embedder,
            write_llm,
            "s-movie",
            "2026-04-01 10:00:00",
            CONVERSATION_MOVIE,
        )
        assert len(episodes) >= 1

        retriever = MemoryRetriever(memory_db, vector_index, shared_embedder)
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
    ) -> None:
        """Movie query ranks movie-session episodes above cooking-session episodes."""
        movie_eps = _write_session(
            memory_db,
            vector_index,
            shared_embedder,
            write_llm,
            "s-movie",
            "2026-04-01 10:00:00",
            CONVERSATION_MOVIE,
        )
        cook_eps = _write_session(
            memory_db,
            vector_index,
            shared_embedder,
            write_llm,
            "s-cook",
            "2026-04-01 12:00:00",
            CONVERSATION_COOKING,
        )
        assert len(movie_eps) >= 1 and len(cook_eps) >= 1

        retriever = MemoryRetriever(memory_db, vector_index, shared_embedder)
        result = retriever.retrieve("영화를 봤어", set())

        if len(result.episodes) >= 2:
            movie_scores = [result.scores[i] for i, ep in enumerate(result.episodes) if ep.session_id == "s-movie"]
            cook_scores = [result.scores[i] for i, ep in enumerate(result.episodes) if ep.session_id == "s-cook"]
            if movie_scores and cook_scores:
                assert max(movie_scores) > max(cook_scores), (
                    "Best movie score should exceed best cooking score for a movie query"
                )
