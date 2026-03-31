"""Tests for NumpyVectorIndex."""

from __future__ import annotations

import numpy as np
import pytest

from voice_pipeline.memory.vector_index import NumpyVectorIndex


class TestNumpyVectorIndex:
    """Unit tests for NumpyVectorIndex."""

    def _make_index(self) -> NumpyVectorIndex:
        return NumpyVectorIndex()

    def _random_vec(self, dim: int = 4, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.standard_normal(dim).astype(np.float32)

    # --- Basic operations ---

    def test_empty_index(self) -> None:
        idx = self._make_index()
        assert len(idx) == 0
        assert idx.search(self._random_vec(), top_k=5) == []

    def test_add_and_search(self) -> None:
        idx = self._make_index()
        v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        idx.add(1, v1)
        idx.add(2, v2)
        assert len(idx) == 2

        # Query close to v1
        results = idx.search(np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32), top_k=2)
        assert len(results) == 2
        assert results[0][0] == 1  # v1 should be top hit

    def test_add_updates_existing(self) -> None:
        idx = self._make_index()
        v_old = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v_new = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        idx.add(1, v_old)
        idx.add(1, v_new)
        assert len(idx) == 1

        query = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        results = idx.search(query, top_k=1)
        assert results[0][0] == 1
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_remove(self) -> None:
        idx = self._make_index()
        idx.add(1, self._random_vec(seed=1))
        idx.add(2, self._random_vec(seed=2))
        idx.remove(1)
        assert len(idx) == 1
        results = idx.search(self._random_vec(seed=1), top_k=5)
        assert all(r[0] != 1 for r in results)

    def test_remove_nonexistent_is_noop(self) -> None:
        idx = self._make_index()
        idx.add(1, self._random_vec(seed=1))
        idx.remove(999)
        assert len(idx) == 1

    def test_remove_last_empties_index(self) -> None:
        idx = self._make_index()
        idx.add(1, self._random_vec(seed=1))
        idx.remove(1)
        assert len(idx) == 0
        assert idx.search(self._random_vec(), top_k=5) == []

    # --- Bulk load ---

    def test_load(self) -> None:
        idx = self._make_index()
        ids = [10, 20, 30]
        vecs = np.eye(3, 4, dtype=np.float32)
        idx.load(ids, vecs)
        assert len(idx) == 3

        results = idx.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), top_k=1)
        assert results[0][0] == 10

    def test_load_replaces_existing(self) -> None:
        idx = self._make_index()
        idx.add(1, self._random_vec(seed=1))
        idx.load([10], np.eye(1, 4, dtype=np.float32))
        assert len(idx) == 1
        results = idx.search(self._random_vec(), top_k=5)
        assert results[0][0] == 10

    def test_load_empty(self) -> None:
        idx = self._make_index()
        idx.add(1, self._random_vec(seed=1))
        idx.load([], np.empty((0, 4), dtype=np.float32))
        assert len(idx) == 0

    # --- Search behavior ---

    def test_top_k_limits_results(self) -> None:
        idx = self._make_index()
        for i in range(10):
            idx.add(i, self._random_vec(seed=i))
        results = idx.search(self._random_vec(seed=0), top_k=3)
        assert len(results) == 3

    def test_search_returns_descending_similarity(self) -> None:
        idx = self._make_index()
        for i in range(5):
            idx.add(i, self._random_vec(seed=i))
        results = idx.search(self._random_vec(seed=0), top_k=5)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_cosine_similarity_values(self) -> None:
        idx = self._make_index()
        v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        idx.add(1, v)
        # Same vector → similarity 1.0
        results = idx.search(v, top_k=1)
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)
        # Orthogonal → similarity 0.0
        orth = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        idx.add(2, orth)
        results = idx.search(v, top_k=2)
        id_to_score = dict(results)
        assert id_to_score[2] == pytest.approx(0.0, abs=1e-5)
