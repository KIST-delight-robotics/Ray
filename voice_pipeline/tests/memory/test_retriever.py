"""Tests for MemoryRetriever."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import numpy as np

from voice_pipeline.core.config import MemoryConfig
from voice_pipeline.core.interfaces import IEmbedder
from voice_pipeline.memory.retriever import MemoryRetriever
from voice_pipeline.memory.storage import InMemoryMemoryStorage
from voice_pipeline.memory.types import Episode
from voice_pipeline.memory.vector_index import NumpyVectorIndex

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 4


def _make_episode(
    text: str = "The user mentioned they love sci-fi movies.",
    timestamp: str = "2026-03-15 14:00:00",
    session_id: str = "s-old",
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


def _vec(*values: float) -> np.ndarray:
    """Create a float32 vector (zero-padded to _DIM)."""
    v = np.zeros(_DIM, dtype=np.float32)
    for i, val in enumerate(values):
        v[i] = val
    return v


class _FakeEmbedder(IEmbedder):
    """Returns a fixed vector for any input."""

    def __init__(self, vector: np.ndarray | None = None) -> None:
        self._vector = vector if vector is not None else _vec(1.0, 0.0)

    def embed(self, text: str) -> np.ndarray:
        return self._vector.copy()

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector.copy() for _ in texts])

    @property
    def dimension(self) -> int:
        return _DIM


def _make_config(**overrides: object) -> MemoryConfig:
    defaults = dict(
        embedding_dimension=_DIM,
        max_memories=5,
        min_new_slots=2,
        retained_ttl=3,
        vector_top_k=10,
        bm25_top_k=10,
        rrf_k=60,
        recency_half_life_days=30.0,
        salience_threshold=0.0,
    )
    defaults.update(overrides)
    return MemoryConfig(**defaults)  # type: ignore[arg-type]


def _fixed_now() -> datetime:
    return datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)


_NOW_PATH = "voice_pipeline.memory.retriever.datetime"


def _setup(
    episodes: list[Episode] | None = None,
    query_vec: np.ndarray | None = None,
    config: MemoryConfig | None = None,
) -> tuple[MemoryRetriever, InMemoryMemoryStorage, NumpyVectorIndex]:
    """Create a retriever with pre-populated storage and vector index."""
    cfg = config or _make_config()
    storage = InMemoryMemoryStorage(dimension=_DIM)
    index = NumpyVectorIndex()
    embedder = _FakeEmbedder(query_vec)

    if episodes:
        for ep in episodes:
            eid = storage.add_episode(ep)
            if ep.embedding is not None and eid is not None:
                index.add(eid, ep.embedding)

    retriever = MemoryRetriever(storage, index, embedder, cfg)
    return retriever, storage, index


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------


class TestRRF:
    def test_both_channels_contribute(self) -> None:
        """Episode in both channels gets higher RRF than single-channel."""
        v_common = _vec(1.0, 0.0)
        v_vec_only = _vec(0.9, 0.1)

        eps = [
            _make_episode(text="common episode", embedding=v_common),
            _make_episode(text="vector only episode", embedding=v_vec_only),
        ]
        retriever, storage, index = _setup(eps, query_vec=_vec(1.0, 0.0))

        # common appears in both vector (high sim) and bm25 (matches "common")
        # vec_only appears only in vector
        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result = retriever.retrieve("common episode", set())

        assert len(result.episodes) >= 2
        # common should rank higher
        common_idx = next(i for i, ep in enumerate(result.episodes) if "common" in ep.text)
        vec_only_idx = next(i for i, ep in enumerate(result.episodes) if "vector only" in ep.text)
        assert result.scores[common_idx] > result.scores[vec_only_idx]

    def test_vector_only(self) -> None:
        """Episode found only by vector search still gets a score."""
        ep = _make_episode(text="xyz unique text", embedding=_vec(1.0, 0.0))
        retriever, _, _ = _setup([ep], query_vec=_vec(1.0, 0.0))

        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result = retriever.retrieve("totally different words", set())

        assert len(result.episodes) == 1

    def test_bm25_only(self) -> None:
        """Episode found only by BM25 (no embedding) still gets a score."""
        ep = _make_episode(text="the user likes pizza", embedding=None)
        retriever, _, _ = _setup([ep], query_vec=_vec(1.0, 0.0))

        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result = retriever.retrieve("pizza", set())

        assert len(result.episodes) == 1
        assert "pizza" in result.episodes[0].text

    def test_both_empty(self) -> None:
        """No results from either channel → empty result."""
        retriever, _, _ = _setup([], query_vec=_vec(1.0, 0.0))

        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result = retriever.retrieve("anything", set())

        assert result.episodes == []
        assert result.scores == []
        assert result.index_to_id == {}


# ---------------------------------------------------------------------------
# Salience
# ---------------------------------------------------------------------------


class TestSalience:
    def test_recent_episode_scores_higher(self) -> None:
        """More recent episode gets higher salience (same importance/RRF)."""
        v = _vec(1.0, 0.0)
        eps = [
            _make_episode(
                text="recent event scifi",
                timestamp="2026-03-30 12:00:00",
                importance=0.7,
                embedding=v.copy(),
            ),
            _make_episode(
                text="old event scifi",
                timestamp="2026-01-01 12:00:00",
                importance=0.7,
                embedding=v.copy(),
            ),
        ]
        retriever, _, _ = _setup(eps, query_vec=_vec(1.0, 0.0))

        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result = retriever.retrieve("scifi", set())

        recent_idx = next(i for i, ep in enumerate(result.episodes) if "recent" in ep.text)
        old_idx = next(i for i, ep in enumerate(result.episodes) if "old" in ep.text)
        assert result.scores[recent_idx] > result.scores[old_idx]

    def test_important_episode_scores_higher(self) -> None:
        """Higher importance → higher salience (same recency/RRF)."""
        v = _vec(1.0, 0.0)
        eps = [
            _make_episode(
                text="important event scifi",
                importance=1.0,
                embedding=v.copy(),
            ),
            _make_episode(
                text="trivial event scifi",
                importance=0.1,
                embedding=v.copy(),
            ),
        ]
        retriever, _, _ = _setup(eps, query_vec=_vec(1.0, 0.0))

        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result = retriever.retrieve("scifi", set())

        imp_idx = next(i for i, ep in enumerate(result.episodes) if "important" in ep.text)
        triv_idx = next(i for i, ep in enumerate(result.episodes) if "trivial" in ep.text)
        assert result.scores[imp_idx] > result.scores[triv_idx]

    def test_half_life_decay(self) -> None:
        """At exactly half_life days, recency_decay ≈ 0.5."""
        v = _vec(1.0, 0.0)
        half_life = 30.0
        # Episode timestamped exactly half_life days before now
        ep = _make_episode(
            text="half life test",
            timestamp="2026-03-01 12:00:00",
            importance=1.0,
            embedding=v,
        )
        cfg = _make_config(recency_half_life_days=half_life)
        retriever, _, _ = _setup([ep], query_vec=_vec(1.0, 0.0), config=cfg)

        now = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)
        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.strptime = datetime.strptime
            result = retriever.retrieve("half life test", set())

        # Also compute score for a "fresh" episode at now
        ep_fresh = _make_episode(
            text="fresh test episode",
            timestamp="2026-03-31 12:00:00",
            importance=1.0,
            embedding=v.copy(),
        )
        retriever2, _, _ = _setup([ep_fresh], query_vec=_vec(1.0, 0.0), config=cfg)
        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.strptime = datetime.strptime
            result_fresh = retriever2.retrieve("fresh test episode", set())

        # Score ratio should be ~0.5 (half-life decay)
        ratio = result.scores[0] / result_fresh.scores[0]
        assert 0.4 < ratio < 0.6


# ---------------------------------------------------------------------------
# Session filtering
# ---------------------------------------------------------------------------


class TestFiltering:
    def test_excluded_sessions_filtered(self) -> None:
        """Episodes from exclude_session_ids are not returned."""
        v = _vec(1.0, 0.0)
        eps = [
            _make_episode(
                text="current session memory",
                session_id="s-current",
                embedding=v.copy(),
            ),
            _make_episode(
                text="summary session memory",
                session_id="s-summary",
                embedding=v.copy(),
            ),
            _make_episode(
                text="old session memory",
                session_id="s-old",
                embedding=v.copy(),
            ),
        ]
        retriever, _, _ = _setup(eps, query_vec=_vec(1.0, 0.0))

        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result = retriever.retrieve("memory", {"s-current", "s-summary"})

        session_ids = {ep.session_id for ep in result.episodes}
        assert "s-current" not in session_ids
        assert "s-summary" not in session_ids
        assert "s-old" in session_ids

    def test_other_sessions_included(self) -> None:
        """Episodes from non-excluded sessions appear."""
        v = _vec(1.0, 0.0)
        ep = _make_episode(text="other session memory", session_id="s-other", embedding=v)
        retriever, _, _ = _setup([ep], query_vec=_vec(1.0, 0.0))

        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result = retriever.retrieve("memory", {"s-current"})

        assert len(result.episodes) == 1
        assert result.episodes[0].session_id == "s-other"


# ---------------------------------------------------------------------------
# Slot allocation
# ---------------------------------------------------------------------------


class TestSlotAllocation:
    def test_max_memories_limits_output(self) -> None:
        """No more than max_memories episodes returned."""
        v = _vec(1.0, 0.0)
        cfg = _make_config(max_memories=3, min_new_slots=1)
        eps = [_make_episode(text=f"episode {i} scifi", embedding=v.copy()) for i in range(10)]
        retriever, _, _ = _setup(eps, query_vec=_vec(1.0, 0.0), config=cfg)

        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result = retriever.retrieve("scifi", set())

        assert len(result.episodes) <= 3

    def test_min_new_slots_guaranteed(self) -> None:
        """Even with retained entries, at least min_new_slots new results appear."""
        v = _vec(1.0, 0.0)
        cfg = _make_config(max_memories=5, min_new_slots=2, retained_ttl=3)

        eps = [_make_episode(text=f"episode {i} scifi", embedding=v.copy()) for i in range(10)]
        retriever, _, _ = _setup(eps, query_vec=_vec(1.0, 0.0), config=cfg)

        # First retrieve — all are new
        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result1 = retriever.retrieve("scifi", set())

        # Cite all to fill retained buffer
        retriever.update_citations(list(result1.index_to_id.keys()))

        # Second retrieve — retained should leave room for min_new_slots
        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            retriever.retrieve("scifi", set())

        # Structural check: entries with high TTL (carried from cited)
        # are capped at max_retained, the rest entered as new (ttl=1)
        carried = sum(1 for e in retriever._retained.values() if e.ttl == cfg.retained_ttl)
        new_entries = sum(1 for e in retriever._retained.values() if e.ttl == 1)
        assert carried <= 3  # max_retained = 5 - 2
        assert new_entries >= 2  # min_new_slots

    def test_retained_overflow_evicts_by_ttl_then_salience(self) -> None:
        """When retained exceeds max, lowest TTL (then salience) evicted."""
        v = _vec(1.0, 0.0)
        cfg = _make_config(max_memories=4, min_new_slots=1, retained_ttl=3)

        eps = [_make_episode(text=f"episode {i} scifi", embedding=v.copy()) for i in range(6)]
        retriever, _, _ = _setup(eps, query_vec=_vec(1.0, 0.0), config=cfg)

        # First retrieve: get 4 episodes
        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result1 = retriever.retrieve("scifi", set())

        # Cite all → all get TTL=3
        retriever.update_citations(list(result1.index_to_id.keys()))

        # Max retained = 4 - 1 = 3, but we have 4 retained → one must be evicted
        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result2 = retriever.retrieve("scifi", set())

        assert len(result2.episodes) <= 4


# ---------------------------------------------------------------------------
# Retained Buffer TTL
# ---------------------------------------------------------------------------


class TestRetainedBuffer:
    def _make_retriever_with_episodes(
        self,
    ) -> tuple[MemoryRetriever, InMemoryMemoryStorage, NumpyVectorIndex]:
        v = _vec(1.0, 0.0)
        cfg = _make_config(max_memories=5, min_new_slots=2, retained_ttl=3)
        eps = [
            _make_episode(text="target episode scifi", embedding=v.copy()),
            _make_episode(text="other episode scifi", embedding=v.copy()),
        ]
        return _setup(eps, query_vec=_vec(1.0, 0.0), config=cfg)

    def test_new_entry_gets_ttl_1(self) -> None:
        """New search result enters retained with TTL=1."""
        retriever, _, _ = self._make_retriever_with_episodes()

        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result = retriever.retrieve("scifi", set())

        # All entries should be in retained with ttl=1
        for eid in result.index_to_id.values():
            assert retriever._retained[eid].ttl == 1

    def test_cited_resets_ttl(self) -> None:
        """After update_citations(), TTL resets to retained_ttl."""
        retriever, _, _ = self._make_retriever_with_episodes()

        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result = retriever.retrieve("scifi", set())

        retriever.update_citations([1])  # cite first episode

        eid = result.index_to_id[1]
        assert retriever._retained[eid].ttl == 3

    def test_search_hit_keeps_ttl(self) -> None:
        """Retained entry found in search results: TTL unchanged."""
        retriever, _, _ = self._make_retriever_with_episodes()

        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result1 = retriever.retrieve("scifi", set())

        retriever.update_citations([1])
        eid = result1.index_to_id[1]
        ttl_before = retriever._retained[eid].ttl
        assert ttl_before == 3

        # Second retrieve — same query, episode is still a search hit
        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            retriever.retrieve("scifi", set())

        assert retriever._retained[eid].ttl == ttl_before

    def test_search_miss_decrements_ttl(self) -> None:
        """Retained entry NOT in search results: TTL -= 1."""
        v_target = _vec(1.0, 0.0)
        v_distract = _vec(0.0, 1.0)
        cfg = _make_config(
            max_memories=5,
            min_new_slots=2,
            retained_ttl=3,
            vector_top_k=3,
            bm25_top_k=3,
        )

        # Target + distractors aligned to the second query direction
        target = _make_episode(text="target episode scifi", embedding=v_target)
        distractors = [
            _make_episode(text=f"distractor {i} movie", embedding=v_distract.copy())
            for i in range(5)
        ]
        retriever, _, _ = _setup(
            [target] + distractors,
            query_vec=v_target,
            config=cfg,
        )

        # Retrieve with v_target → target found
        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result = retriever.retrieve("scifi", set())

        # Find and cite the target
        target_idx = next(
            idx
            for idx, eid in result.index_to_id.items()
            if any(ep.id == eid and "target" in ep.text for ep in result.episodes)
        )
        retriever.update_citations([target_idx])
        eid = result.index_to_id[target_idx]
        assert retriever._retained[eid].ttl == 3

        # Switch query direction → target falls out of top_k
        retriever._embedder = _FakeEmbedder(v_distract)

        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            retriever.retrieve("movie", set())

        assert retriever._retained[eid].ttl == 2

    def test_ttl_zero_evicts(self) -> None:
        """Entry with TTL=0 is evicted from retained buffer."""
        v_target = _vec(1.0, 0.0)
        v_distract = _vec(0.0, 1.0)
        cfg = _make_config(
            max_memories=5,
            min_new_slots=2,
            retained_ttl=1,
            vector_top_k=3,
            bm25_top_k=3,
        )

        target = _make_episode(text="target episode scifi", embedding=v_target)
        distractors = [
            _make_episode(text=f"distractor {i} movie", embedding=v_distract.copy())
            for i in range(5)
        ]
        retriever, _, _ = _setup(
            [target] + distractors,
            query_vec=v_target,
            config=cfg,
        )

        # Retrieve and cite → TTL = 1
        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result = retriever.retrieve("scifi", set())

        target_idx = next(
            idx
            for idx, eid in result.index_to_id.items()
            if any(ep.id == eid and "target" in ep.text for ep in result.episodes)
        )
        retriever.update_citations([target_idx])
        eid = result.index_to_id[target_idx]
        assert retriever._retained[eid].ttl == 1

        # Miss → TTL 0 → evict
        retriever._embedder = _FakeEmbedder(v_distract)

        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            retriever.retrieve("movie", set())

        assert eid not in retriever._retained

    def test_cited_then_misses_survive_ttl_turns(self) -> None:
        """Cited entry survives retained_ttl turns of misses before eviction."""
        v_target = _vec(1.0, 0.0)
        v_distract = _vec(0.0, 1.0)
        cfg = _make_config(
            max_memories=5,
            min_new_slots=2,
            retained_ttl=3,
            vector_top_k=3,
            bm25_top_k=3,
        )

        target = _make_episode(text="target episode scifi", embedding=v_target)
        distractors = [
            _make_episode(text=f"distractor {i} movie", embedding=v_distract.copy())
            for i in range(5)
        ]
        retriever, _, _ = _setup(
            [target] + distractors,
            query_vec=v_target,
            config=cfg,
        )

        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result = retriever.retrieve("scifi", set())

        target_idx = next(
            idx
            for idx, eid in result.index_to_id.items()
            if any(ep.id == eid and "target" in ep.text for ep in result.episodes)
        )
        retriever.update_citations([target_idx])
        eid = result.index_to_id[target_idx]

        # Switch to orthogonal direction for misses
        retriever._embedder = _FakeEmbedder(v_distract)

        # 3 miss turns: TTL 3 → 2 → 1 → 0 (evicted on 3rd)
        for turn in range(3):
            with patch(_NOW_PATH) as mock_dt:
                mock_dt.now.return_value = _fixed_now()
                mock_dt.strptime = datetime.strptime
                retriever.retrieve("movie", set())

            if turn < 2:
                assert eid in retriever._retained, f"Should survive turn {turn}"
            else:
                assert eid not in retriever._retained, "Should be evicted on turn 2"


# ---------------------------------------------------------------------------
# update_citations()
# ---------------------------------------------------------------------------


class TestUpdateCitations:
    def test_updates_db_last_cited_at(self) -> None:
        """update_citations() calls storage.update_episode_cited()."""
        v = _vec(1.0, 0.0)
        ep = _make_episode(
            text="cited episode scifi",
            timestamp="2026-03-15 14:00:00",
            embedding=v,
        )
        retriever, storage, _ = _setup([ep], query_vec=v)

        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result = retriever.retrieve("scifi", set())

        eid = result.index_to_id[1]
        retriever.update_citations([1])

        updated = storage.get_episode(eid)
        assert updated is not None
        assert updated.last_cited_at != "2026-03-15 14:00:00"

    def test_invalid_index_ignored(self) -> None:
        """Invalid citation index is silently skipped."""
        v = _vec(1.0, 0.0)
        ep = _make_episode(text="some episode scifi", embedding=v)
        retriever, _, _ = _setup([ep], query_vec=v)

        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            retriever.retrieve("scifi", set())

        # Should not raise
        retriever.update_citations([99])

    def test_empty_list_noop(self) -> None:
        """Empty citation list is a no-op."""
        v = _vec(1.0, 0.0)
        ep = _make_episode(text="some episode scifi", embedding=v)
        retriever, storage, _ = _setup([ep], query_vec=v)

        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result = retriever.retrieve("scifi", set())

        eid = result.index_to_id[1]
        original = storage.get_episode(eid)
        assert original is not None
        original_cited = original.last_cited_at

        retriever.update_citations([])

        still = storage.get_episode(eid)
        assert still is not None
        assert still.last_cited_at == original_cited


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


class TestResultStructure:
    def test_index_to_id_mapping(self) -> None:
        """index_to_id correctly maps 1-based indices to DB IDs."""
        v = _vec(1.0, 0.0)
        eps = [_make_episode(text=f"ep {i} scifi", embedding=v.copy()) for i in range(3)]
        retriever, _, _ = _setup(eps, query_vec=v)

        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result = retriever.retrieve("scifi", set())

        assert len(result.index_to_id) == len(result.episodes)
        for i, ep in enumerate(result.episodes):
            assert result.index_to_id[i + 1] == ep.id

    def test_retained_before_new(self) -> None:
        """Retained episodes appear before new ones in the result."""
        v = _vec(1.0, 0.0)
        cfg = _make_config(max_memories=5, min_new_slots=2, retained_ttl=3)
        eps = [_make_episode(text=f"ep {i} scifi", embedding=v.copy()) for i in range(5)]
        retriever, _, _ = _setup(eps, query_vec=v, config=cfg)

        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result1 = retriever.retrieve("scifi", set())

        # Cite first two
        retriever.update_citations([1, 2])
        retained_ids = {result1.index_to_id[1], result1.index_to_id[2]}

        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result2 = retriever.retrieve("scifi", set())

        retained_positions = [i for i, ep in enumerate(result2.episodes) if ep.id in retained_ids]
        new_positions = [i for i, ep in enumerate(result2.episodes) if ep.id not in retained_ids]
        if retained_positions and new_positions:
            assert max(retained_positions) < min(new_positions)

    def test_empty_database_returns_empty(self) -> None:
        """No episodes → empty result."""
        retriever, _, _ = _setup([], query_vec=_vec(1.0, 0.0))

        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result = retriever.retrieve("anything", set())

        assert result.episodes == []
        assert result.scores == []
        assert result.index_to_id == {}

    def test_scores_parallel_episodes(self) -> None:
        """scores list is same length as episodes list."""
        v = _vec(1.0, 0.0)
        eps = [_make_episode(text=f"ep {i} scifi", embedding=v.copy()) for i in range(3)]
        retriever, _, _ = _setup(eps, query_vec=v)

        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result = retriever.retrieve("scifi", set())

        assert len(result.scores) == len(result.episodes)


# ---------------------------------------------------------------------------
# Full lifecycle
# ---------------------------------------------------------------------------


class TestFullCycle:
    def test_retrieve_cite_retrieve(self) -> None:
        """Full cycle: add episodes → retrieve → cite → retrieve again."""
        v = _vec(1.0, 0.0)
        cfg = _make_config(max_memories=3, min_new_slots=1, retained_ttl=3)

        eps = [
            _make_episode(text="liked interstellar scifi", embedding=v.copy(), importance=0.9),
            _make_episode(text="enjoyed dune scifi", embedding=v.copy(), importance=0.8),
            _make_episode(text="watched matrix scifi", embedding=v.copy(), importance=0.6),
        ]
        retriever, storage, _ = _setup(eps, query_vec=v, config=cfg)

        # Turn 1: retrieve
        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            r1 = retriever.retrieve("scifi movies", set())

        assert len(r1.episodes) <= 3
        assert len(r1.index_to_id) == len(r1.episodes)

        # Cite first episode
        retriever.update_citations([1])

        # Turn 2: retrieve again
        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            r2 = retriever.retrieve("scifi movies", set())

        # Cited episode should still be present
        cited_eid = r1.index_to_id[1]
        assert cited_eid in r2.index_to_id.values()

        # DB should have updated last_cited_at
        updated_ep = storage.get_episode(cited_eid)
        assert updated_ep is not None
        assert updated_ep.last_cited_at != eps[0].timestamp


# ---------------------------------------------------------------------------
# Edge cases (from code review)
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_embed_failure_returns_empty_on_fresh_retriever(self) -> None:
        """If embedder raises on fresh retriever, result is empty."""

        class _FailingEmbedder(IEmbedder):
            def embed(self, text: str) -> np.ndarray:
                raise RuntimeError("embed failed")

            def embed_batch(self, texts: list[str]) -> np.ndarray:
                raise RuntimeError("embed failed")

            @property
            def dimension(self) -> int:
                return _DIM

        cfg = _make_config()
        storage = InMemoryMemoryStorage(dimension=_DIM)
        index = NumpyVectorIndex()
        retriever = MemoryRetriever(storage, index, _FailingEmbedder(), cfg)

        result = retriever.retrieve("anything", set())

        assert result.episodes == []
        assert result.scores == []
        assert result.index_to_id == {}

    def test_embed_failure_preserves_retained(self) -> None:
        """If embedder raises with retained entries, they still appear."""
        v = _vec(1.0, 0.0)
        ep = _make_episode(text="retained episode scifi", embedding=v)
        retriever, _, _ = _setup([ep], query_vec=v)

        # Populate retained buffer
        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result1 = retriever.retrieve("scifi", set())

        retriever.update_citations([1])
        eid = result1.index_to_id[1]

        # Break the embedder
        class _FailingEmbedder(IEmbedder):
            def embed(self, text: str) -> np.ndarray:
                raise RuntimeError("embed failed")

            def embed_batch(self, texts: list[str]) -> np.ndarray:
                raise RuntimeError("embed failed")

            @property
            def dimension(self) -> int:
                return _DIM

        retriever._embedder = _FailingEmbedder()

        result2 = retriever.retrieve("anything", set())

        # Retained entry should still be returned
        assert eid in result2.index_to_id.values()

    def test_zero_importance_produces_zero_salience(self) -> None:
        """Episode with importance=0.0 gets salience=0.0."""
        v = _vec(1.0, 0.0)
        ep = _make_episode(
            text="zero importance scifi",
            importance=0.0,
            embedding=v,
        )
        retriever, _, _ = _setup([ep], query_vec=v)

        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result = retriever.retrieve("scifi", set())

        assert len(result.episodes) == 1
        assert result.scores[0] == 0.0

    def test_retained_only_path(self) -> None:
        """When search returns nothing, retained entries still appear."""
        v_target = _vec(1.0, 0.0)
        v_distract = _vec(0.0, 1.0)
        cfg = _make_config(
            max_memories=5,
            min_new_slots=2,
            retained_ttl=3,
            vector_top_k=3,
            bm25_top_k=3,
        )

        target = _make_episode(text="target episode scifi", embedding=v_target)
        distractors = [
            _make_episode(text=f"distractor {i} movie", embedding=v_distract.copy())
            for i in range(5)
        ]
        retriever, _, _ = _setup(
            [target] + distractors,
            query_vec=v_target,
            config=cfg,
        )

        # First retrieve — target found and cited
        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result = retriever.retrieve("scifi", set())

        target_idx = next(
            idx
            for idx, eid in result.index_to_id.items()
            if any(ep.id == eid and "target" in ep.text for ep in result.episodes)
        )
        retriever.update_citations([target_idx])
        eid = result.index_to_id[target_idx]

        # Switch to query that returns nothing (no vector or BM25 hits)
        retriever._embedder = _FakeEmbedder(v_distract)

        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            # "xyznonexistent" won't match any text via BM25
            # distractors fill vector top_k, target falls out
            result2 = retriever.retrieve("xyznonexistent", set())

        # Target should still appear via retained buffer
        assert eid in result2.index_to_id.values()

    def test_citation_updates_cached_last_cited_at(self) -> None:
        """update_citations refreshes cached episode's last_cited_at."""
        v = _vec(1.0, 0.0)
        ep = _make_episode(
            text="cited episode scifi",
            timestamp="2026-03-15 14:00:00",
            embedding=v,
        )
        retriever, _, _ = _setup([ep], query_vec=v)

        with patch(_NOW_PATH) as mock_dt:
            mock_dt.now.return_value = _fixed_now()
            mock_dt.strptime = datetime.strptime
            result = retriever.retrieve("scifi", set())

        eid = result.index_to_id[1]
        old_cited = retriever._retained[eid].episode.last_cited_at
        retriever.update_citations([1])

        assert retriever._retained[eid].episode.last_cited_at != old_cited
