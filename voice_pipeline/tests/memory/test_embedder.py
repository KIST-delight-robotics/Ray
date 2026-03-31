"""Tests for IEmbedder implementations."""

from __future__ import annotations

import numpy as np
import pytest

from voice_pipeline.core.config import MemoryConfig
from voice_pipeline.memory.embedder import SentenceTransformerEmbedder


@pytest.mark.requires_model
class TestSentenceTransformerEmbedder:
    """Integration tests requiring the actual sentence-transformers model."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self._config = MemoryConfig(
            embedding_model="all-MiniLM-L6-v2",
            embedding_dimension=384,
        )

    def test_embed_returns_correct_shape(self) -> None:
        embedder = SentenceTransformerEmbedder(self._config)
        vec = embedder.embed("Hello world")
        assert vec.shape == (384,)
        assert vec.dtype == np.float32

    def test_embed_batch_returns_correct_shape(self) -> None:
        embedder = SentenceTransformerEmbedder(self._config)
        texts = ["Hello world", "Goodbye world", "Test sentence"]
        vecs = embedder.embed_batch(texts)
        assert vecs.shape == (3, 384)
        assert vecs.dtype == np.float32

    def test_similar_texts_have_higher_similarity(self) -> None:
        embedder = SentenceTransformerEmbedder(self._config)
        v1 = embedder.embed("I love watching sci-fi movies")
        v2 = embedder.embed("Science fiction films are my favorite")
        v3 = embedder.embed("I enjoy cooking pasta")

        sim_12 = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        sim_13 = float(np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3)))
        assert sim_12 > sim_13

    def test_dimension_property(self) -> None:
        embedder = SentenceTransformerEmbedder(self._config)
        assert embedder.dimension == 384
