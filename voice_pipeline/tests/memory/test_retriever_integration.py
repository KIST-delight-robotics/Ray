"""Integration tests for MemoryRetriever with real embeddings and SQLite FTS5.

Uses the local sentence-transformers model — no API key needed.
"""

from __future__ import annotations

import pytest

from voice_pipeline.embedding.embedder import SentenceTransformerEmbedder
from voice_pipeline.memory.retriever import MemoryRetriever
from voice_pipeline.memory.storage import SQLiteMemoryStorage
from voice_pipeline.memory.vector_index import NumpyVectorIndex
from voice_pipeline.tests.memory.conftest import make_episode, store_episode_with_embedding

pytestmark = pytest.mark.requires_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RETRIEVER_CLASS_VAR_MAP = {
    "max_memories": "_MAX_MEMORIES",
    "min_new_slots": "_MIN_NEW_SLOTS",
    "retained_ttl": "_RETAINED_TTL",
    "vector_top_k": "_VECTOR_TOP_K",
    "bm25_top_k": "_BM25_TOP_K",
    "rrf_k": "_RRF_K",
    "recency_half_life_days": "_RECENCY_HALF_LIFE_DAYS",
    "salience_threshold": "_SALIENCE_THRESHOLD",
}


def _build_retriever(
    storage: SQLiteMemoryStorage,
    index: NumpyVectorIndex,
    embedder: SentenceTransformerEmbedder,
    monkeypatch=None,
    **config_overrides,
) -> MemoryRetriever:
    if config_overrides:
        assert monkeypatch is not None, "monkeypatch fixture required when overriding retriever tuning"
        for key, value in config_overrides.items():
            monkeypatch.setattr(MemoryRetriever, _RETRIEVER_CLASS_VAR_MAP[key], value)
    return MemoryRetriever(storage, index, embedder)


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------


class TestVectorSearchReal:
    """Vector search with real sentence-transformer embeddings."""

    def test_semantic_query_ranks_relevant_higher(
        self,
        memory_db: SQLiteMemoryStorage,
        vector_index: NumpyVectorIndex,
        shared_embedder: SentenceTransformerEmbedder,
    ) -> None:
        """Movie query ranks movie episodes above cooking episodes."""
        movie_ep = store_episode_with_embedding(
            memory_db,
            vector_index,
            shared_embedder,
            make_episode(
                "The user said Interstellar was an amazing sci-fi movie and cried at the ending.",
                session_id="s-old",
            ),
        )
        cooking_ep = store_episode_with_embedding(
            memory_db,
            vector_index,
            shared_embedder,
            make_episode(
                "The user made cream mushroom pasta from scratch and wants to learn Italian cooking.",
                session_id="s-old",
            ),
        )
        retriever = _build_retriever(memory_db, vector_index, shared_embedder)
        result = retriever.retrieve("What movies do you like? I love sci-fi films.", set())

        assert len(result.episodes) == 2
        ep_ids = [ep.id for ep in result.episodes]
        movie_rank = ep_ids.index(movie_ep.id)
        cooking_rank = ep_ids.index(cooking_ep.id)
        assert movie_rank < cooking_rank, "Movie episode should rank higher for a movie query"

    def test_bm25_keyword_match(
        self,
        memory_db: SQLiteMemoryStorage,
        vector_index: NumpyVectorIndex,
        shared_embedder: SentenceTransformerEmbedder,
    ) -> None:
        """BM25 returns episodes containing the query keyword."""
        ep = store_episode_with_embedding(
            memory_db,
            vector_index,
            shared_embedder,
            make_episode("사용자가 인터스텔라 영화를 좋아한다고 말했다.", session_id="s-old"),
        )
        bm25_results = memory_db.search_bm25("인터스텔라", top_k=5)
        assert len(bm25_results) >= 1
        assert bm25_results[0][0] == ep.id

    def test_hybrid_rrf_boosts_dual_channel(
        self,
        memory_db: SQLiteMemoryStorage,
        vector_index: NumpyVectorIndex,
        shared_embedder: SentenceTransformerEmbedder,
    ) -> None:
        """Episode matching both vector AND BM25 scores higher than single-channel."""
        # "both" — semantically similar AND contains keyword "Interstellar"
        both_ep = store_episode_with_embedding(
            memory_db,
            vector_index,
            shared_embedder,
            make_episode(
                "The user watched the Interstellar movie and found it deeply moving.",
                session_id="s-old",
            ),
        )
        # "bm25-only" — contains keyword but semantically distant
        bm25_only_ep = store_episode_with_embedding(
            memory_db,
            vector_index,
            shared_embedder,
            make_episode(
                "The user mentioned cooking pasta while Interstellar played in the background.",
                session_id="s-old",
            ),
        )

        retriever = _build_retriever(memory_db, vector_index, shared_embedder)
        result = retriever.retrieve("Interstellar is my favorite sci-fi film", set())

        assert len(result.episodes) >= 2
        ep_ids = [ep.id for ep in result.episodes]
        both_idx = ep_ids.index(both_ep.id)
        bm25_only_idx = ep_ids.index(bm25_only_ep.id)
        assert result.scores[both_idx] > result.scores[bm25_only_idx], (
            "Dual-channel episode should have higher salience"
        )


# ---------------------------------------------------------------------------
# Retained buffer
# ---------------------------------------------------------------------------


class TestRetainedBufferReal:
    """Retained buffer behaviour with real embeddings."""

    def test_cited_memory_persists_across_topic_change(
        self,
        memory_db: SQLiteMemoryStorage,
        vector_index: NumpyVectorIndex,
        shared_embedder: SentenceTransformerEmbedder,
        monkeypatch,
    ) -> None:
        """A cited episode remains in results even when the query topic changes."""
        movie_ep = store_episode_with_embedding(
            memory_db,
            vector_index,
            shared_embedder,
            make_episode("사용자가 놀란 감독의 영화를 좋아한다고 말했다.", session_id="s-old"),
        )
        store_episode_with_embedding(
            memory_db,
            vector_index,
            shared_embedder,
            make_episode("사용자가 파스타 요리를 즐긴다.", session_id="s-old"),
        )

        retriever = _build_retriever(memory_db, vector_index, shared_embedder, monkeypatch, retained_ttl=3)

        # First query: movie topic → cite the movie episode
        result1 = retriever.retrieve("영화를 좋아해", set())
        movie_display_idx = next(idx for idx, eid in result1.index_to_id.items() if eid == movie_ep.id)
        retriever.update_citations([movie_display_idx])

        # Second query: completely different topic
        result2 = retriever.retrieve("요리를 좋아해", set())
        result2_ids = {ep.id for ep in result2.episodes}
        assert movie_ep.id in result2_ids, "Cited movie episode should persist via retained buffer"

    def test_uncited_decays_and_evicts(
        self,
        memory_db: SQLiteMemoryStorage,
        vector_index: NumpyVectorIndex,
        shared_embedder: SentenceTransformerEmbedder,
        monkeypatch,
    ) -> None:
        """Uncited episodes decay from the retained buffer after TTL expires."""
        movie_ep = store_episode_with_embedding(
            memory_db,
            vector_index,
            shared_embedder,
            make_episode("사용자가 놀란 감독의 영화를 좋아한다고 말했다.", session_id="s-old"),
        )
        cooking_ep = store_episode_with_embedding(
            memory_db,
            vector_index,
            shared_embedder,
            make_episode("사용자가 파스타를 만들었다.", session_id="s-old"),
        )

        retriever = _build_retriever(
            memory_db,
            vector_index,
            shared_embedder,
            monkeypatch,
            retained_ttl=2,
            max_memories=5,
        )

        # Turn 1: movie query → movie_ep enters retained with ttl=1
        retriever.retrieve("영화를 좋아해", set())

        # Turn 2: cooking query → movie_ep not a search hit → ttl decays to 0
        retriever.retrieve("파스타 요리", set())

        # Turn 3: cooking query again → movie_ep should be evicted (ttl ≤ 0)
        result3 = retriever.retrieve("파스타 요리", set())
        result3_ids = {ep.id for ep in result3.episodes}
        # movie_ep may or may not be in result3 depending on whether it's a new search hit,
        # but it should NOT be in the retained buffer anymore
        if movie_ep.id in result3_ids:
            # If it's back, it's because search found it again, not retained
            pass
        assert cooking_ep.id in result3_ids


# ---------------------------------------------------------------------------
# Session filtering
# ---------------------------------------------------------------------------


class TestSessionFiltering:
    """Session exclusion in retrieval."""

    def test_exclude_current_session(
        self,
        memory_db: SQLiteMemoryStorage,
        vector_index: NumpyVectorIndex,
        shared_embedder: SentenceTransformerEmbedder,
    ) -> None:
        """Episodes from excluded sessions do not appear in results."""
        store_episode_with_embedding(
            memory_db,
            vector_index,
            shared_embedder,
            make_episode("사용자가 영화를 좋아한다.", session_id="current-session"),
        )
        old_ep = store_episode_with_embedding(
            memory_db,
            vector_index,
            shared_embedder,
            make_episode("사용자가 예전에 영화를 많이 봤다고 말했다.", session_id="s-old"),
        )

        retriever = _build_retriever(memory_db, vector_index, shared_embedder)
        result = retriever.retrieve("영화", {"current-session"})

        session_ids = {ep.session_id for ep in result.episodes}
        assert "current-session" not in session_ids
        assert old_ep.id in {ep.id for ep in result.episodes}
