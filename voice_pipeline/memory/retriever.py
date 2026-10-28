"""Memory retrieval pipeline with hybrid search and retained buffer."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from voice_pipeline.memory.storage import SQLiteMemoryStorage
from voice_pipeline.memory.types import Episode, MemoryReadResult
from voice_pipeline.memory.vector_index import NumpyVectorIndex
from voice_pipeline.types import IEmbedder

logger = logging.getLogger("voice_pipeline.memory")

_LN2 = math.log(2)
_SECONDS_PER_DAY = 86400.0


@dataclass
class _RetainedEntry:
    """Internal state for a memory held in the retained buffer."""

    episode: Episode
    salience: float
    ttl: int


class MemoryRetriever:
    """Hybrid search retriever with RRF fusion and retained buffer.

    Searches episodes via vector similarity and BM25, fuses results
    with Reciprocal Rank Fusion, ranks by salience, and maintains a
    retained buffer that protects cited memories across turns.

    Thread safety: both ``retrieve()`` and ``update_citations()`` must
    be called from the same thread (the orchestrator main loop).
    """

    _MAX_MEMORIES = 10  # 턴당 Block 4 주입 최대 에피소드 수
    _MIN_NEW_SLOTS = 4  # 새 검색 결과에 확보할 최소 슬롯 수
    _RETAINED_TTL = 3  # 인용된 메모리가 retained 버퍼에 머무는 턴 수
    _VECTOR_TOP_K = 20  # 벡터 검색 후보 수
    _BM25_TOP_K = 20  # BM25 검색 후보 수
    _RRF_K = 60  # RRF 융합 상수 (원 논문 default 60)
    _RECENCY_HALF_LIFE_DAYS = 30.0  # 시간 감쇠 반감기 (일)
    _SALIENCE_THRESHOLD = 0.0  # salience 최소 기준 (0.0 = 비활성화)
    _RANGE_MAX_MEMORIES = 20  # 주제 없는 기간 검색이 돌려줄 최대 에피소드 수 (최근 것부터)
    _RANGE_OUTSIDE_EXTRA = 3  # 주제 있는 기간 검색에 덧붙이는 범위 밖 결과 수 — 날짜를 잘못 기억한 경우

    def __init__(
        self,
        storage: SQLiteMemoryStorage,
        vector_index: NumpyVectorIndex,
        embedder: IEmbedder,
        *,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        """
        Args:
            storage: Episode storage.
            vector_index: Vector search index over episode embeddings.
            embedder: Query embedder.
            now_fn: Clock override for recency decay (중립 주입점 — 과거 시점
                대화를 다루는 평가에서 "현재"를 고정할 때 사용). ``None``이면
                ``datetime.now(UTC)``.
        """
        self._storage = storage
        self._vector_index = vector_index
        self._embedder = embedder
        self._now_fn: Callable[[], datetime] = now_fn or (lambda: datetime.now(UTC))

        self._retained: dict[int, _RetainedEntry] = {}
        self._last_index_to_id: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, exclude_session_ids: set[str]) -> MemoryReadResult:
        now = self._now_fn()

        # 1. Embed query
        try:
            query_vec = self._embedder.embed(query)
        except Exception:
            logger.warning("Failed to embed query, returning retained only", exc_info=True)
            self._decay_retained_ttl(set())
            return self._build_result_retained_only()

        # 2. Search (sequential — both are fast)
        vector_results = self._vector_index.search(query_vec, self._VECTOR_TOP_K)
        bm25_results = self._storage.search_bm25(query, self._BM25_TOP_K)

        # 3. RRF fusion
        rrf_scores = self._compute_rrf(vector_results, bm25_results)
        if not rrf_scores:
            self._decay_retained_ttl(set())
            return self._build_result_retained_only()

        # 4. Load episodes
        episodes_by_id = {
            ep.id: ep for ep in self._storage.get_episodes_by_ids(list(rrf_scores.keys())) if ep.id is not None
        }

        # 5. Session filter
        for eid in list(episodes_by_id):
            if episodes_by_id[eid].session_id in exclude_session_ids:
                del episodes_by_id[eid]
                rrf_scores.pop(eid, None)

        # 6. Salience
        salience: dict[int, float] = {}
        for eid, ep in episodes_by_id.items():
            salience[eid] = self._compute_salience(rrf_scores[eid], ep, now)

        # 7. Threshold filter
        if self._SALIENCE_THRESHOLD > 0:
            salience = {eid: s for eid, s in salience.items() if s >= self._SALIENCE_THRESHOLD}

        # Threshold-filtered episodes are treated as search misses for
        # TTL purposes — no point retaining something with negligible salience.
        search_hit_ids = set(salience.keys())

        # 8. Update retained TTL + refresh salience for search hits
        self._decay_retained_ttl(search_hit_ids)
        for eid in search_hit_ids:
            if eid in self._retained:
                self._retained[eid].salience = salience[eid]

        # 9. Slot allocation
        retained_list = self._cap_retained()
        retained_ids = {entry.episode.id for entry in retained_list}

        # 9b. New results (not already retained)
        new_candidates = [(eid, salience[eid]) for eid in salience if eid not in retained_ids]
        new_candidates.sort(key=lambda x: x[1], reverse=True)

        new_slots = self._MAX_MEMORIES - len(retained_list)
        new_selected = new_candidates[:new_slots]

        # 10. New entries join retained with ttl=1
        # NOTE: The cached Episode object is a snapshot from this turn's
        # storage load. If Phase 3 mutates episodes between turns, the
        # retained buffer will serve stale data until the entry is evicted
        # and re-fetched via search.
        for eid, sal in new_selected:
            ep = episodes_by_id[eid]
            self._retained[eid] = _RetainedEntry(episode=ep, salience=sal, ttl=1)

        # 11. Build result: retained first, then new
        episodes: list[Episode] = []
        scores: list[float] = []

        for entry in retained_list:
            episodes.append(entry.episode)
            scores.append(entry.salience)

        for eid, sal in new_selected:
            episodes.append(episodes_by_id[eid])
            scores.append(sal)

        index_to_id: dict[int, int] = {}
        for i, ep in enumerate(episodes):
            assert ep.id is not None
            index_to_id[i + 1] = ep.id

        self._last_index_to_id = index_to_id

        return MemoryReadResult(episodes=episodes, scores=scores, index_to_id=index_to_id)

    def update_citations(self, cited_indices: list[int]) -> None:
        if not cited_indices:
            return

        now_str = self._now_fn().strftime("%Y-%m-%d %H:%M:%S")

        for idx in cited_indices:
            eid = self._last_index_to_id.get(idx)
            if eid is None:
                logger.warning("Citation index %d not in index_to_id, skipping", idx)
                continue

            if eid in self._retained:
                self._retained[eid].ttl = self._RETAINED_TTL
                self._retained[eid].episode.last_cited_at = now_str

            self._storage.update_episode_cited(eid, now_str)

    def retrieve_in_range(self, query: str, start: str, end: str, exclude_session_ids: set[str]) -> list[Episode]:
        """Episodes with ``start <= timestamp < end`` (UTC) — the time-bounded search.

        Without a query: the range's episodes in time order, the newest ``_RANGE_MAX_MEMORIES``.
        With a query: the range's episodes ranked by similarity — the range is filtered before
        ranking, since :meth:`retrieve`'s global top-K can miss them — followed by up to
        ``_RANGE_OUTSIDE_EXTRA`` unrestricted results from outside the range. Users misremember
        dates; every result carries its date, so the caller can tell which is which.
        """
        in_range = [
            ep for ep in self._storage.get_episodes_between(start, end) if ep.session_id not in exclude_session_ids
        ]
        if not query:
            return in_range[-self._RANGE_MAX_MEMORIES :]
        try:
            query_vec = self._embedder.embed(query)
        except Exception:
            logger.warning("Failed to embed query, returning range episodes in time order", exc_info=True)
            return in_range[-self._MAX_MEMORIES :]

        ranked = sorted(in_range, key=lambda ep: _cosine(query_vec, ep.embedding), reverse=True)
        in_range_ids = {ep.id for ep in in_range}
        outside = [ep for ep in self.retrieve(query, exclude_session_ids).episodes if ep.id not in in_range_ids]
        return ranked[: self._MAX_MEMORIES] + outside[: self._RANGE_OUTSIDE_EXTRA]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_rrf(
        self,
        vector_results: list[tuple[int, float]],
        bm25_results: list[tuple[int, float]],
    ) -> dict[int, float]:
        """Reciprocal Rank Fusion over two result lists.

        Uses ``1 / (k + rank + 1)`` where rank is 0-based, equivalent
        to the standard formula ``1 / (k + r)`` with 1-based rank r.
        """
        k = self._RRF_K
        scores: dict[int, float] = {}
        for rank, (eid, _score) in enumerate(vector_results):
            scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + rank + 1)
        for rank, (eid, _score) in enumerate(bm25_results):
            scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + rank + 1)
        return scores

    def _compute_salience(self, rrf_score: float, episode: Episode, now: datetime) -> float:
        """salience = rrf_score × recency_decay × importance."""
        dt = datetime.strptime(episode.last_cited_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        days = max((now - dt).total_seconds() / _SECONDS_PER_DAY, 0.0)
        recency_decay = math.exp(-_LN2 * days / self._RECENCY_HALF_LIFE_DAYS)
        return rrf_score * recency_decay * episode.importance

    def _decay_retained_ttl(self, search_hit_ids: set[int]) -> None:
        """Decrement TTL for retained entries not in current search results."""
        to_evict: list[int] = []
        for eid, entry in self._retained.items():
            if eid not in search_hit_ids:
                entry.ttl -= 1
                if entry.ttl <= 0:
                    to_evict.append(eid)
        for eid in to_evict:
            del self._retained[eid]

    def _cap_retained(self) -> list[_RetainedEntry]:
        """Sort retained entries and evict overflow beyond max_retained."""
        max_retained = self._MAX_MEMORIES - self._MIN_NEW_SLOTS
        retained_list = sorted(
            self._retained.values(),
            key=lambda e: (e.ttl, e.salience),
            reverse=True,
        )
        if len(retained_list) > max_retained:
            for entry in retained_list[max_retained:]:
                assert entry.episode.id is not None
                del self._retained[entry.episode.id]
            retained_list = retained_list[:max_retained]
        return retained_list

    def _build_result_retained_only(self) -> MemoryReadResult:
        """Build a result containing only retained entries (no new search hits).

        Scores are the salience values stored at entry/refresh time — not
        recomputed for the current turn since there is no RRF score to use.
        """
        if not self._retained:
            self._last_index_to_id = {}
            return MemoryReadResult(episodes=[], scores=[], index_to_id={})

        retained_list = self._cap_retained()

        episodes: list[Episode] = []
        scores: list[float] = []
        index_to_id: dict[int, int] = {}

        for i, entry in enumerate(retained_list):
            assert entry.episode.id is not None
            episodes.append(entry.episode)
            scores.append(entry.salience)
            index_to_id[i + 1] = entry.episode.id

        self._last_index_to_id = index_to_id
        return MemoryReadResult(episodes=episodes, scores=scores, index_to_id=index_to_id)


def _cosine(query: np.ndarray, embedding: np.ndarray | None) -> float:
    """Cosine similarity; an episode without an embedding ranks last."""
    if embedding is None:
        return -1.0
    return float(np.dot(query, embedding) / (np.linalg.norm(query) * np.linalg.norm(embedding) + 1e-9))
