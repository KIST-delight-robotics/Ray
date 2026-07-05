"""Vector index interface and implementations.

In-memory index for dense vector similarity search.
Internal to the memory module — not exposed cross-module.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger("voice_pipeline.memory")


class IVectorIndex(ABC):
    """In-memory vector index for similarity search.

    Implementations may use exact search (numpy) or approximate
    nearest neighbors (hnswlib, etc.).
    """

    @abstractmethod
    def add(self, id: int, vector: np.ndarray) -> None:
        """Add or update a vector.

        Args:
            id: Episode database ID.
            vector: 1-D float32 vector.
        """

    @abstractmethod
    def remove(self, id: int) -> None:
        """Remove a vector by ID. No-op if ID not found.

        Args:
            id: Episode database ID to remove.
        """

    @abstractmethod
    def search(self, query: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Find the most similar vectors.

        Args:
            query: 1-D float32 query vector.
            top_k: Maximum number of results.

        Returns:
            List of (id, similarity) tuples, sorted by similarity
            descending. Similarity is cosine similarity in [−1, 1].
            Implementations may omit results below a relevance floor,
            so fewer than ``top_k`` results can be returned even when
            the index holds more vectors.
        """

    @abstractmethod
    def load(self, ids: list[int], vectors: np.ndarray) -> None:
        """Bulk-load vectors, replacing any existing index contents.

        Called at startup to populate from DB.

        Args:
            ids: Episode IDs, length N.
            vectors: 2-D float32 array of shape (N, dimension).
        """

    @abstractmethod
    def __len__(self) -> int:
        """Number of vectors in the index."""


class NumpyVectorIndex(IVectorIndex):
    """Exact cosine similarity search using numpy.

    Suitable for small-to-medium collections (< ~10k vectors).
    All vectors are held in a single (N, dim) matrix for
    efficient batch cosine similarity via matrix multiplication.

    Thread-safe: all public methods are guarded by a lock so that
    concurrent ``search()`` (retriever thread) and ``add()``
    (write-executor thread) do not corrupt shared state.
    """

    _MIN_COSINE_SIMILARITY = 0.2  # 검색 결과 관련성 하한 — 이 값 미만은 top_k에 들어도 제외

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ids: list[int] = []
        self._matrix: np.ndarray | None = None  # shape: (N, dim), L2-normalized

    def add(self, id: int, vector: np.ndarray) -> None:
        """Add or update a vector."""
        normed = self._normalize(vector.reshape(1, -1))
        with self._lock:
            try:
                idx = self._ids.index(id)
                assert self._matrix is not None
                self._matrix[idx] = normed[0]
            except ValueError:
                self._ids.append(id)
                if self._matrix is None:
                    self._matrix = normed
                else:
                    self._matrix = np.vstack([self._matrix, normed])

    def remove(self, id: int) -> None:
        """Remove a vector by ID."""
        with self._lock:
            try:
                idx = self._ids.index(id)
            except ValueError:
                return
            self._ids.pop(idx)
            assert self._matrix is not None
            self._matrix = np.delete(self._matrix, idx, axis=0)
            if len(self._ids) == 0:
                self._matrix = None

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Cosine similarity search. Results below ``_MIN_COSINE_SIMILARITY`` are dropped."""
        with self._lock:
            if self._matrix is None or len(self._ids) == 0:
                return []
            q = self._normalize(query.reshape(1, -1))  # (1, dim)
            scores = (self._matrix @ q.T).flatten()  # (N,)
            k = min(top_k, len(self._ids))
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            return [(self._ids[i], float(scores[i])) for i in top_indices if scores[i] >= self._MIN_COSINE_SIMILARITY]

    def load(self, ids: list[int], vectors: np.ndarray) -> None:
        """Bulk-load vectors from DB."""
        with self._lock:
            if len(ids) == 0:
                self._ids = []
                self._matrix = None
                return
            self._ids = list(ids)
            self._matrix = self._normalize(np.asarray(vectors, dtype=np.float32))
            logger.info("Vector index loaded: %d vectors", len(self._ids))

    def __len__(self) -> int:
        """Number of vectors in the index."""
        with self._lock:
            return len(self._ids)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize rows. Near-zero vectors are left as-is."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return vectors / norms
