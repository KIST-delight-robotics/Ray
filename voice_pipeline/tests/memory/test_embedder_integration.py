"""Integration tests for embedding model with Korean text.

Requires a local sentence-transformers model (all-MiniLM-L6-v2).
No API key needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from voice_pipeline.embedding.embedder import SentenceTransformerEmbedder

pytestmark = pytest.mark.requires_model


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class TestSentenceTransformerKorean:
    """Verify embedding model handles Korean text correctly."""

    def test_embed_returns_correct_shape(
        self, shared_embedder: SentenceTransformerEmbedder
    ) -> None:
        """Korean text produces a 384-dim float32 vector."""
        vec = shared_embedder.embed("어제 인터스텔라를 봤는데 정말 감동적이었어")
        assert vec.shape == (384,)
        assert vec.dtype == np.float32
        assert not np.any(np.isnan(vec))

    def test_semantic_similarity_ordering(
        self, shared_embedder: SentenceTransformerEmbedder
    ) -> None:
        """Semantically similar Korean texts have higher cosine similarity."""
        v_movie1 = shared_embedder.embed("인터스텔라 영화가 정말 감동적이었어")
        v_movie2 = shared_embedder.embed("놀란 감독의 SF 영화를 좋아해")
        v_cooking = shared_embedder.embed("오늘 파스타를 만들어서 먹었어")

        sim_movies = _cosine_sim(v_movie1, v_movie2)
        sim_cross = _cosine_sim(v_movie1, v_cooking)
        assert sim_movies > sim_cross, (
            f"Movie-movie similarity ({sim_movies:.3f}) should exceed "
            f"movie-cooking similarity ({sim_cross:.3f})"
        )

    def test_batch_matches_individual(self, shared_embedder: SentenceTransformerEmbedder) -> None:
        """embed_batch produces the same vectors as individual embed calls."""
        texts = ["영화를 봤어", "음악을 들었어", "요리를 했어"]
        batch = shared_embedder.embed_batch(texts)
        assert batch.shape == (3, 384)
        for i, t in enumerate(texts):
            individual = shared_embedder.embed(t)
            np.testing.assert_allclose(batch[i], individual, atol=1e-5)

    def test_mixed_language(self, shared_embedder: SentenceTransformerEmbedder) -> None:
        """Korean-English mixed text produces a valid vector."""
        vec = shared_embedder.embed("나는 Interstellar을 봤어, it was amazing")
        assert vec.shape == (384,)
        assert not np.any(np.isnan(vec))

    def test_empty_string(self, shared_embedder: SentenceTransformerEmbedder) -> None:
        """Empty string produces a valid (non-NaN) vector."""
        vec = shared_embedder.embed("")
        assert vec.shape == (384,)
        assert not np.any(np.isnan(vec))
