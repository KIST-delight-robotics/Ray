"""Text similarity implementations.

Provides embedding-based and difflib-based similarity scoring.
Used by TurnDetector to decide whether ASR text has changed enough
to warrant a new prepare().
"""

from __future__ import annotations

import difflib
import logging

import numpy as np

from voice_pipeline.core.config import SimilarityConfig
from voice_pipeline.core.interfaces import IEmbedder, ISimilarity

logger = logging.getLogger("voice_pipeline.similarity")


class EmbeddingSimilarity(ISimilarity):
    """Cosine similarity using an IEmbedder.

    Delegates text encoding to the injected embedder, then computes
    cosine similarity between the resulting vectors.
    """

    def __init__(self, embedder: IEmbedder) -> None:
        self._embedder = embedder

    def compare(self, a: str, b: str) -> float:
        """Return cosine similarity between embeddings of *a* and *b*."""
        vecs = self._embedder.embed_batch([a, b])
        vec_a, vec_b = vecs[0], vecs[1]
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


class DiffLibSimilarity(ISimilarity):
    """Character-level similarity using stdlib difflib.

    Zero-dependency fallback when sentence-transformers is not installed.
    Uses SequenceMatcher ratio which returns [0.0, 1.0].
    Less accurate than embedding similarity (misses paraphrases) but
    sufficient for detecting near-identical ASR text changes.
    """

    def compare(self, a: str, b: str) -> float:
        """Return SequenceMatcher ratio between *a* and *b*."""
        return difflib.SequenceMatcher(None, a, b).ratio()


def create_similarity(
    config: SimilarityConfig,
    embedder: IEmbedder | None = None,
) -> ISimilarity:
    """Factory: create an ISimilarity instance from config.

    Args:
        config: Similarity configuration.
        embedder: Optional pre-created IEmbedder. When provided, used
            directly (enables sharing an embedder instance across modules).
            When None, an embedder is created internally from config.

    When ``backend="local"`` and sentence-transformers is not installed,
    falls back to DiffLibSimilarity with a warning.
    """
    if config.backend == "difflib":
        return DiffLibSimilarity()

    if config.backend in ("local", "api"):
        if embedder is not None:
            return EmbeddingSimilarity(embedder)
        try:
            from voice_pipeline.embedding.embedder import create_embedder

            emb = create_embedder(
                model=config.model,
                backend=config.backend,
                use_onnx=config.use_onnx,
            )
            return EmbeddingSimilarity(emb)
        except ImportError:
            logger.warning("Embedding library not available — falling back to difflib similarity.")
            return DiffLibSimilarity()

    raise ValueError(f"Unknown similarity backend: {config.backend!r}")
