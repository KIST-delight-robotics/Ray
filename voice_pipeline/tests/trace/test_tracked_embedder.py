"""Tests for TrackedEmbedder wrapper."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from voice_pipeline.core.interfaces import IEmbedder
from voice_pipeline.trace.trace_store import InMemoryCallStore
from voice_pipeline.trace.tracked_embedder import TrackedEmbedder


class _FakeEmbedder(IEmbedder):
    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    def embed(self, text: str) -> np.ndarray:
        return np.ones(self._dim, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), self._dim), dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "fake-model"


class TestTrackedEmbedder:
    @pytest.fixture()
    def setup(self) -> tuple[TrackedEmbedder, InMemoryCallStore, _FakeEmbedder]:
        inner = _FakeEmbedder()
        store = InMemoryCallStore()
        wrapper = TrackedEmbedder(inner, store)
        wrapper.session_id = "sess-1"
        return wrapper, store, inner

    def test_embed_passthrough(self, setup: tuple[TrackedEmbedder, InMemoryCallStore, _FakeEmbedder]) -> None:
        wrapper, _, _ = setup
        result = wrapper.embed("hello")
        assert result.shape == (4,)
        assert result.dtype == np.float32

    def test_embed_batch_passthrough(self, setup: tuple[TrackedEmbedder, InMemoryCallStore, _FakeEmbedder]) -> None:
        wrapper, _, _ = setup
        result = wrapper.embed_batch(["a", "b", "c"])
        assert result.shape == (3, 4)

    def test_embed_records_call(self, setup: tuple[TrackedEmbedder, InMemoryCallStore, _FakeEmbedder]) -> None:
        wrapper, store, _ = setup
        wrapper.embed("test")
        assert len(store.records) == 1
        rec = store.records[0]
        assert rec.session_id == "sess-1"
        assert rec.module == "embedder"
        assert rec.operation == "embed"
        assert rec.model == "fake-model"
        assert rec.status == "ok"
        assert rec.elapsed_ms >= 0
        assert rec.metadata is None

    def test_embed_batch_records_metadata(
        self, setup: tuple[TrackedEmbedder, InMemoryCallStore, _FakeEmbedder]
    ) -> None:
        wrapper, store, _ = setup
        wrapper.embed_batch(["a", "b"])
        assert len(store.records) == 1
        rec = store.records[0]
        assert rec.operation == "embed_batch"
        meta = json.loads(rec.metadata)
        assert meta["count"] == 2

    def test_dimension_delegates(self, setup: tuple[TrackedEmbedder, InMemoryCallStore, _FakeEmbedder]) -> None:
        wrapper, _, _ = setup
        assert wrapper.dimension == 4

    def test_model_name_delegates(self, setup: tuple[TrackedEmbedder, InMemoryCallStore, _FakeEmbedder]) -> None:
        wrapper, _, _ = setup
        assert wrapper.model_name == "fake-model"

    def test_store_error_swallowed(self) -> None:
        inner = _FakeEmbedder()
        store = MagicMock()
        store.record.side_effect = RuntimeError("db error")
        wrapper = TrackedEmbedder(inner, store)
        result = wrapper.embed("test")
        assert result.shape == (4,)

    def test_session_id_mutable(self) -> None:
        inner = _FakeEmbedder()
        store = InMemoryCallStore()
        wrapper = TrackedEmbedder(inner, store)
        wrapper.session_id = "s1"
        wrapper.embed("a")
        wrapper.session_id = "s2"
        wrapper.embed("b")
        assert store.records[0].session_id == "s1"
        assert store.records[1].session_id == "s2"
