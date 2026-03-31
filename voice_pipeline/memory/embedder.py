"""Embedding provider interface and implementations.

Converts text to dense vectors for semantic search.
Internal to the memory module — not exposed cross-module.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from voice_pipeline.core.config import MemoryConfig
from voice_pipeline.core.exceptions import ConfigurationError

logger = logging.getLogger("voice_pipeline.memory")


class IEmbedder(ABC):
    """Text embedding provider.

    Implementations may use local models or external APIs.
    """

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text.

        Args:
            text: Input text.

        Returns:
            1-D float32 array of shape (dimension,).
        """

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts in a single call.

        Args:
            texts: List of input texts.

        Returns:
            2-D float32 array of shape (len(texts), dimension).
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""


class SentenceTransformerEmbedder(IEmbedder):
    """Embedding via sentence-transformers (local model).

    Uses the same model family as the similarity module.
    """

    def __init__(self, config: MemoryConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: uv sync --extra similarity"
            ) from exc
        backend = "onnx" if config.use_onnx else "torch"
        self._model = SentenceTransformer(config.embedding_model, backend=backend)
        actual_dim = self._model.get_sentence_embedding_dimension()
        if actual_dim != config.embedding_dimension:
            raise ConfigurationError(
                f"Embedding model dimension ({actual_dim}) does not match "
                f"config embedding_dimension ({config.embedding_dimension})"
            )
        self._dimension = config.embedding_dimension
        logger.info(
            "Loaded embedding model: %s (backend=%s, dim=%d)",
            config.embedding_model,
            backend,
            self._dimension,
        )

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text."""
        vec = self._model.encode(text, show_progress_bar=False)
        return np.asarray(vec, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts."""
        vecs = self._model.encode(texts, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        return self._dimension
