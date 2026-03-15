"""Text similarity implementations.

Provides local (sentence-transformers) and API (OpenAI) embedding-based
similarity scoring. Used by TurnDetector to decide whether ASR text
has changed enough to warrant a new prepare().
"""

from __future__ import annotations

import logging

import numpy as np

from voice_pipeline.core.config import SimilarityConfig
from voice_pipeline.core.interfaces import ISimilarity

logger = logging.getLogger("voice_pipeline.core.similarity")


class LocalEmbeddingSimilarity(ISimilarity):
    """Embedding similarity using a local sentence-transformers model."""

    def __init__(self, config: SimilarityConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for local embedding similarity. "
                "Install with: uv add sentence-transformers"
            ) from exc
        backend = "onnx" if config.use_onnx else "torch"
        self._model = SentenceTransformer(config.model, backend=backend)
        logger.info("Loaded similarity model: %s (backend=%s)", config.model, backend)

    def compare(self, a: str, b: str) -> float:
        """Return cosine similarity between sentence embeddings of *a* and *b*."""
        embeddings = self._model.encode([a, b], show_progress_bar=False)
        return float(self._model.similarity(embeddings[0], embeddings[1]).item())


class APIEmbeddingSimilarity(ISimilarity):
    """Embedding similarity using the OpenAI embeddings API."""

    def __init__(self, config: SimilarityConfig) -> None:
        self._model = config.model
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "openai is required for API embedding similarity."
            ) from exc
        self._client = openai.OpenAI()

    def compare(self, a: str, b: str) -> float:
        """Return cosine similarity between OpenAI embeddings of *a* and *b*."""
        response = self._client.embeddings.create(input=[a, b], model=self._model)
        vec_a = np.array(response.data[0].embedding)
        vec_b = np.array(response.data[1].embedding)
        return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def create_similarity(config: SimilarityConfig) -> ISimilarity:
    """Factory: create an ISimilarity instance from config."""
    if config.backend == "local":
        return LocalEmbeddingSimilarity(config)
    elif config.backend == "api":
        return APIEmbeddingSimilarity(config)
    else:
        raise ValueError(f"Unknown similarity backend: {config.backend!r}")
