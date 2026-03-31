"""Tests for IEmbedder implementations."""

from __future__ import annotations

import numpy as np
import pytest

from voice_pipeline.embedding.embedder import SentenceTransformerEmbedder, create_embedder


@pytest.mark.requires_model
class TestSentenceTransformerEmbedder:
    """Integration tests requiring the actual sentence-transformers model."""

    @pytest.fixture(autouse=True, scope="class")
    def _shared_embedder(self, request: pytest.FixtureRequest) -> None:
        request.cls.embedder = SentenceTransformerEmbedder(
            "all-MiniLM-L6-v2",
            expected_dimension=384,
        )

    def test_embed_returns_correct_shape(self) -> None:
        vec = self.embedder.embed("Hello world")
        assert vec.shape == (384,)
        assert vec.dtype == np.float32

    def test_embed_batch_returns_correct_shape(self) -> None:
        texts = ["Hello world", "Goodbye world", "Test sentence"]
        vecs = self.embedder.embed_batch(texts)
        assert vecs.shape == (3, 384)
        assert vecs.dtype == np.float32

    def test_similar_texts_have_higher_similarity(self) -> None:
        v1 = self.embedder.embed("I love watching sci-fi movies")
        v2 = self.embedder.embed("Science fiction films are my favorite")
        v3 = self.embedder.embed("I enjoy cooking pasta")

        sim_12 = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        sim_13 = float(np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3)))
        assert sim_12 > sim_13

    def test_dimension_property(self) -> None:
        assert self.embedder.dimension == 384

    def test_auto_detect_dimension(self) -> None:
        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        assert embedder.dimension == 384

    def test_wrong_dimension_raises(self) -> None:
        from voice_pipeline.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            SentenceTransformerEmbedder("all-MiniLM-L6-v2", expected_dimension=999)


class TestCreateEmbedder:
    """Unit tests for the create_embedder factory."""

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown embedding backend"):
            create_embedder("some-model", "unknown_backend")  # type: ignore[arg-type]

    def test_api_dimension_unknown_before_embed(self) -> None:
        """API embedder without dimension raises on .dimension access before embed."""
        from voice_pipeline.embedding.embedder import OpenAIEmbedder

        embedder = OpenAIEmbedder.__new__(OpenAIEmbedder)
        embedder._dimension = None
        with pytest.raises(RuntimeError, match="Dimension unknown"):
            _ = embedder.dimension
