"""Embedding provider implementations.

Converts text to dense vectors for semantic search.
Shared across modules (memory, similarity, etc.).
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from voice_pipeline.core.exceptions import ConfigurationError
from voice_pipeline.core.interfaces import IEmbedder

logger = logging.getLogger("voice_pipeline.embedding")


class SentenceTransformerEmbedder(IEmbedder):
    """Embedding via sentence-transformers (local model).

    Supports both PyTorch and ONNX Runtime backends.
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        *,
        use_onnx: bool = False,
        expected_dimension: int | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for local embeddings. Install with: uv sync"
            ) from exc
        backend = "onnx" if use_onnx else "torch"
        self._model = SentenceTransformer(model, backend=backend)
        actual_dim = self._model.get_sentence_embedding_dimension()
        if expected_dimension is not None and actual_dim != expected_dimension:
            raise ConfigurationError(
                f"Embedding model dimension ({actual_dim}) does not match "
                f"expected_dimension ({expected_dimension})"
            )
        self._dimension = actual_dim
        logger.info(
            "Loaded embedding model: %s (backend=%s, dim=%d)",
            model,
            backend,
            self._dimension,
        )

    def embed(self, text: str) -> np.ndarray:
        vec = self._model.encode(text, show_progress_bar=False)
        return np.asarray(vec, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIEmbedder(IEmbedder):
    """Embedding via the OpenAI embeddings API.

    If *dimension* is not provided, it is auto-detected from the first
    embed call.  Accessing :attr:`dimension` before any embed raises
    ``RuntimeError``.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        dimension: int | None = None,
    ) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ImportError("openai is required for API embeddings.") from exc
        self._client = openai.OpenAI()
        self._model = model
        self._dimension: int | None = dimension

    def embed(self, text: str) -> np.ndarray:
        response = self._client.embeddings.create(input=[text], model=self._model)
        vec = np.asarray(response.data[0].embedding, dtype=np.float32)
        if self._dimension is None:
            self._dimension = vec.shape[0]
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        response = self._client.embeddings.create(input=texts, model=self._model)
        vecs = np.asarray([d.embedding for d in response.data], dtype=np.float32)
        if self._dimension is None and vecs.ndim == 2 and vecs.shape[0] > 0:
            self._dimension = vecs.shape[1]
        return vecs

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError(
                "Dimension unknown — call embed() first or provide dimension at construction"
            )
        return self._dimension


def create_embedder(
    model: str,
    backend: Literal["local", "api"] = "local",
    *,
    use_onnx: bool = False,
    expected_dimension: int | None = None,
) -> IEmbedder:
    """Factory: create an IEmbedder instance.

    Args:
        model: Model name (sentence-transformers or OpenAI model).
        backend: ``"local"`` for sentence-transformers, ``"api"`` for OpenAI.
        use_onnx: Use ONNX Runtime backend (local only).
        expected_dimension: If provided, validate model dimension matches.
            For ``"api"`` backend, auto-detected from first call if omitted.

    Returns:
        Configured IEmbedder instance.
    """
    if backend == "local":
        return SentenceTransformerEmbedder(
            model,
            use_onnx=use_onnx,
            expected_dimension=expected_dimension,
        )
    elif backend == "api":
        return OpenAIEmbedder(model, dimension=expected_dimension)
    else:
        raise ValueError(f"Unknown embedding backend: {backend!r}")
