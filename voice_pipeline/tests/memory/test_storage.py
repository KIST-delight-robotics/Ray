"""Tests for IMemoryStorage implementations.

Uses a mixin pattern so the same test suite runs against both
InMemoryMemoryStorage and SQLiteMemoryStorage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pytest

from voice_pipeline.core.interfaces import IMemoryStorage
from voice_pipeline.memory.types import Episode, Profile


def _make_episode(
    text: str = "The user mentioned they love sci-fi movies.",
    timestamp: str = "2026-03-15 14:00:00",
    session_id: str = "s1",
    importance: float = 0.7,
    last_cited_at: str | None = None,
    citation_count: int = 0,
    embedding: np.ndarray | None = None,
) -> Episode:
    return Episode(
        id=None,
        text=text,
        timestamp=timestamp,
        session_id=session_id,
        importance=importance,
        last_cited_at=last_cited_at or timestamp,
        citation_count=citation_count,
        embedding=embedding,
    )


def _make_profile(
    topic: str = "interest",
    sub_topic: str = "movie",
    content: str = "Loves sci-fi movies",
    updated_at: str = "2026-03-15 14:00:00",
) -> Profile:
    return Profile(
        id=None,
        topic=topic,
        sub_topic=sub_topic,
        content=content,
        updated_at=updated_at,
    )


class _StorageTests(ABC):
    """Shared tests for all IMemoryStorage implementations."""

    @abstractmethod
    def make_storage(self) -> IMemoryStorage:
        """Create a fresh storage instance."""

    # --- Episode CRUD ---

    def test_add_and_get_episode(self) -> None:
        s = self.make_storage()
        ep = _make_episode()
        eid = s.add_episode(ep)
        assert eid is not None and eid > 0

        loaded = s.get_episode(eid)
        assert loaded is not None
        assert loaded.id == eid
        assert loaded.text == ep.text
        assert loaded.timestamp == ep.timestamp
        assert loaded.session_id == ep.session_id
        assert loaded.importance == pytest.approx(ep.importance)
        assert loaded.last_cited_at == ep.last_cited_at
        assert loaded.citation_count == 0

    def test_episode_citation_count_roundtrip(self) -> None:
        s = self.make_storage()
        ep = _make_episode(citation_count=5)
        eid = s.add_episode(ep)
        loaded = s.get_episode(eid)
        assert loaded is not None
        assert loaded.citation_count == 5

    def test_get_episode_not_found(self) -> None:
        s = self.make_storage()
        assert s.get_episode(999) is None

    def test_get_episodes_by_ids(self) -> None:
        s = self.make_storage()
        e1 = s.add_episode(_make_episode(text="Episode one"))
        e2 = s.add_episode(_make_episode(text="Episode two"))
        s.add_episode(_make_episode(text="Episode three"))

        results = s.get_episodes_by_ids([e1, e2])
        assert len(results) == 2
        texts = {ep.text for ep in results}
        assert texts == {"Episode one", "Episode two"}

    def test_get_episodes_by_ids_empty(self) -> None:
        s = self.make_storage()
        assert s.get_episodes_by_ids([]) == []

    def test_get_episodes_by_ids_missing_skipped(self) -> None:
        s = self.make_storage()
        e1 = s.add_episode(_make_episode(text="Episode one"))
        results = s.get_episodes_by_ids([e1, 999])
        assert len(results) == 1

    def test_update_episode_cited(self) -> None:
        s = self.make_storage()
        eid = s.add_episode(_make_episode(last_cited_at="2026-03-15 14:00:00"))
        s.update_episode_cited(eid, "2026-03-20 10:00:00")

        loaded = s.get_episode(eid)
        assert loaded is not None
        assert loaded.last_cited_at == "2026-03-20 10:00:00"

    def test_episode_with_embedding(self) -> None:
        s = self.make_storage()
        emb = np.random.default_rng(42).standard_normal(384).astype(np.float32)
        eid = s.add_episode(_make_episode(embedding=emb))

        loaded = s.get_episode(eid)
        assert loaded is not None
        assert loaded.embedding is not None
        np.testing.assert_array_almost_equal(loaded.embedding, emb, decimal=5)

    def test_episode_without_embedding(self) -> None:
        s = self.make_storage()
        eid = s.add_episode(_make_episode(embedding=None))
        loaded = s.get_episode(eid)
        assert loaded is not None
        assert loaded.embedding is None

    # --- BM25 search ---

    def test_search_bm25_basic(self) -> None:
        s = self.make_storage()
        s.add_episode(_make_episode(text="The user loves science fiction movies"))
        s.add_episode(_make_episode(text="The user enjoys cooking Italian food"))
        s.add_episode(_make_episode(text="The user watched a new movie last night"))

        results = s.search_bm25("movie", top_k=5)
        assert len(results) >= 1
        # All results should have positive scores
        for _, score in results:
            assert score > 0

    def test_search_bm25_no_match(self) -> None:
        s = self.make_storage()
        s.add_episode(_make_episode(text="The user loves cooking"))
        results = s.search_bm25("xyzzyzzy", top_k=5)
        assert len(results) == 0

    def test_search_bm25_top_k(self) -> None:
        s = self.make_storage()
        for i in range(10):
            s.add_episode(_make_episode(text=f"Movie number {i} is great"))
        results = s.search_bm25("movie", top_k=3)
        assert len(results) <= 3

    def test_search_bm25_special_characters(self) -> None:
        s = self.make_storage()
        s.add_episode(_make_episode(text="The user likes sci-fi movies"))
        s.add_episode(_make_episode(text="The user said hello world"))
        # Hyphen: should not crash, should still return results
        results = s.search_bm25("sci-fi", top_k=5)
        assert len(results) >= 1
        # Quotes: should not crash
        results = s.search_bm25('"quoted"', top_k=5)
        assert isinstance(results, list)
        # FTS5 operator "NOT" treated as literal word, not operator.
        # No document contains literal "NOT", so 0 results is correct.
        # If NOT were an FTS5 operator, it would exclude "hello" docs
        # and return the sci-fi episode — that must NOT happen.
        results = s.search_bm25("NOT hello", top_k=5)
        result_ids = {r[0] for r in results}
        sci_fi_ep = [r for r in s.get_episodes_by_ids(list(result_ids)) if "sci-fi" in r.text]
        assert len(sci_fi_ep) == 0  # NOT was not interpreted as operator

    def test_search_bm25_empty_query(self) -> None:
        s = self.make_storage()
        s.add_episode(_make_episode(text="Some episode text"))
        assert s.search_bm25("", top_k=5) == []
        assert s.search_bm25("   ", top_k=5) == []

    # --- Episode embedding update ---

    def test_update_episode_embedding(self) -> None:
        s = self.make_storage()
        eid = s.add_episode(_make_episode(embedding=None))
        assert eid is not None

        emb = np.random.default_rng(42).standard_normal(384).astype(np.float32)
        s.update_episode_embedding(eid, emb)

        loaded = s.get_episode(eid)
        assert loaded is not None
        assert loaded.embedding is not None
        np.testing.assert_array_almost_equal(loaded.embedding, emb, decimal=5)

    # --- Profile CRUD ---

    def test_upsert_and_get_profile(self) -> None:
        s = self.make_storage()
        p = _make_profile()
        pid = s.upsert_profile(p)
        assert pid is not None and pid > 0

        profiles = s.get_all_profiles()
        assert len(profiles) == 1
        assert profiles[0].id == pid
        assert profiles[0].topic == p.topic
        assert profiles[0].sub_topic == p.sub_topic
        assert profiles[0].content == p.content

    def test_upsert_profile_update(self) -> None:
        s = self.make_storage()
        pid = s.upsert_profile(_make_profile(content="Old content"))

        updated = Profile(
            id=pid,
            topic="interest",
            sub_topic="movie",
            content="New content",
            updated_at="2026-03-20 10:00:00",
        )
        returned_id = s.upsert_profile(updated)
        assert returned_id == pid

        profiles = s.get_all_profiles()
        assert len(profiles) == 1
        assert profiles[0].content == "New content"

    def test_delete_profile(self) -> None:
        s = self.make_storage()
        pid = s.upsert_profile(_make_profile())
        s.delete_profile(pid)
        assert s.get_all_profiles() == []

    def test_delete_profile_nonexistent(self) -> None:
        s = self.make_storage()
        s.delete_profile(999)  # should not raise

    # --- Utterance ---

    def test_add_and_get_utterances(self) -> None:
        s = self.make_storage()
        s.add_utterance("s1", "user", "Hello", "2026-03-15 14:00:00", token_count=3)
        s.add_utterance("s1", "assistant", "Hi there!", "2026-03-15 14:00:01", token_count=5)
        s.add_utterance("s2", "user", "Other session", "2026-03-15 15:00:00")

        utts = s.get_utterances("s1")
        assert len(utts) == 2
        assert utts[0] == ("user", "Hello", "2026-03-15 14:00:00", 3)
        assert utts[1] == ("assistant", "Hi there!", "2026-03-15 14:00:01", 5)

    def test_get_utterances_empty(self) -> None:
        s = self.make_storage()
        assert s.get_utterances("nonexistent") == []

    # --- Embeddings load ---

    def test_load_all_embeddings(self) -> None:
        s = self.make_storage()
        emb1 = np.ones(384, dtype=np.float32)
        emb2 = np.ones(384, dtype=np.float32) * 2
        s.add_episode(_make_episode(text="Ep 1", embedding=emb1))
        s.add_episode(_make_episode(text="Ep 2", embedding=emb2))
        s.add_episode(_make_episode(text="Ep 3", embedding=None))

        ids, vectors = s.load_all_embeddings()
        assert len(ids) == 2
        assert vectors.shape == (2, 384)

    def test_load_all_embeddings_empty(self) -> None:
        s = self.make_storage()
        ids, vectors = s.load_all_embeddings()
        assert ids == []
        assert vectors.shape[0] == 0


# ---------------------------------------------------------------------------
# Concrete test classes
# ---------------------------------------------------------------------------


class TestInMemoryMemoryStorage(_StorageTests):
    def make_storage(self) -> IMemoryStorage:
        from voice_pipeline.memory.storage import InMemoryMemoryStorage

        return InMemoryMemoryStorage(dimension=384)


class TestSQLiteMemoryStorage(_StorageTests):
    @pytest.fixture(autouse=True)
    def _setup_db(self, tmp_path: object) -> None:
        self._db_path = str(tmp_path / "test_memory.db")  # type: ignore[operator]

    def make_storage(self) -> IMemoryStorage:
        from voice_pipeline.core.config import MemoryConfig
        from voice_pipeline.memory.storage import SQLiteMemoryStorage

        config = MemoryConfig(db_path=self._db_path)
        return SQLiteMemoryStorage(config)


class TestSQLiteMemoryStoragePersistence:
    """Tests that verify data persists across separate connections."""

    @pytest.fixture(autouse=True)
    def _setup_db(self, tmp_path: object) -> None:
        self._db_path = str(tmp_path / "test_memory.db")  # type: ignore[operator]

    def _make_storage(self) -> IMemoryStorage:
        from voice_pipeline.core.config import MemoryConfig
        from voice_pipeline.memory.storage import SQLiteMemoryStorage

        return SQLiteMemoryStorage(MemoryConfig(db_path=self._db_path))

    def test_episode_persists(self) -> None:
        s1 = self._make_storage()
        eid = s1.add_episode(_make_episode(text="Persistent episode"))
        s1.close()

        s2 = self._make_storage()
        loaded = s2.get_episode(eid)
        assert loaded is not None
        assert loaded.text == "Persistent episode"
        s2.close()

    def test_profile_persists(self) -> None:
        s1 = self._make_storage()
        s1.upsert_profile(_make_profile(content="Persistent profile"))
        s1.close()

        s2 = self._make_storage()
        profiles = s2.get_all_profiles()
        assert len(profiles) == 1
        assert profiles[0].content == "Persistent profile"
        s2.close()

    def test_utterance_persists(self) -> None:
        s1 = self._make_storage()
        s1.add_utterance("s1", "user", "Persistent utterance", "2026-03-15 14:00:00", 10)
        s1.close()

        s2 = self._make_storage()
        utts = s2.get_utterances("s1")
        assert len(utts) == 1
        assert utts[0][1] == "Persistent utterance"
        assert utts[0][3] == 10
        s2.close()

    def test_embedding_persists(self) -> None:
        s1 = self._make_storage()
        emb = np.random.default_rng(42).standard_normal(384).astype(np.float32)
        s1.add_episode(_make_episode(embedding=emb))
        s1.close()

        s2 = self._make_storage()
        ids, vectors = s2.load_all_embeddings()
        assert len(ids) == 1
        np.testing.assert_array_almost_equal(vectors[0], emb, decimal=5)
        s2.close()
