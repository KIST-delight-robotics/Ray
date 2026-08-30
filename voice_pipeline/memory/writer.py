"""Memory write pipeline: extract episodes and profiles from session utterances.

Pure processing module — receives session data, performs LLM-based extraction,
and stores results. Does not manage triggers or scheduling (Phase 4).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np

from voice_pipeline.memory.prompts import (
    EPISODE_DEDUP_SCHEMA,
    EPISODE_EXTRACTION_SCHEMA,
    PROFILE_EXTRACTION_SCHEMA,
    PROFILE_MERGE_SCHEMA,
    PROFILE_SCHEMA,
    build_episode_dedup_messages,
    build_episode_extraction_messages,
    build_profile_extraction_messages,
    build_profile_merge_messages,
    format_profile_schema,
)
from voice_pipeline.memory.storage import SQLiteMemoryStorage
from voice_pipeline.memory.types import Episode, Profile
from voice_pipeline.memory.vector_index import NumpyVectorIndex
from voice_pipeline.types import ILLM, IEmbedder, TokenCounter

logger = logging.getLogger("voice_pipeline.memory")


class MemoryWriter:
    """Extracts episodes and profiles from session utterances.

    Thread safety: designed to be called from a single thread.
    """

    _MIN_UTTERANCES = 2  # 에피소드 추출 시도에 필요한 최소 utterance 수
    _WRITE_MAX_INPUT_TOKENS = 8000  # 에피소드 추출 윈도우 최대 토큰 수 (초과 시 분할)
    _WRITE_WINDOW_OVERLAP_RATIO = 0.25  # 인접 윈도우 overlap 비율 (0.0–1.0)
    _WRITE_DEDUP_THRESHOLD = 0.8  # 중복 판정 코사인 유사도 임계값
    _PROFILE_MAX_CONTENT_TOKENS = 128  # 프로필 슬롯 content 최대 토큰 수
    _PROFILE_CONTENT_WARN_RATIO = 0.7  # content 토큰이 예산의 몇 배를 넘으면 요약 경고 표시

    def __init__(
        self,
        storage: SQLiteMemoryStorage,
        vector_index: NumpyVectorIndex,
        embedder: IEmbedder,
        llm: ILLM,
        token_counter: TokenCounter,
        *,
        episode_system_prompt: str | None = None,
    ) -> None:
        """
        Args:
            storage: Episode/profile persistence backend.
            vector_index: Vector index updated with new episode embeddings.
            embedder: Episode embedder.
            llm: Extraction LLM.
            token_counter: Token counter for windowing and profile budgets.
            episode_system_prompt: 에피소드 추출 시스템 프롬프트 오버라이드
                (중립 주입점 — 평가에서 프롬프트 변형 실험용). ``None``이면
                기본 프롬프트.
        """
        self._storage = storage
        self._vector_index = vector_index
        self._embedder = embedder
        self._llm = llm
        self._token_counter = token_counter
        self._episode_system_prompt = episode_system_prompt
        self._profile_schema_text = format_profile_schema()

    def process_session(self, session_id: str, session_timestamp: str) -> list[Episode]:
        """Process a completed session: extract episodes and profiles.

        Args:
            session_id: Session to process.
            session_timestamp: Session start time (UTC, '%Y-%m-%d %H:%M:%S').
                Used as the episode timestamp.

        Returns:
            List of extracted episodes (with IDs assigned). Empty on failure
            or if the session is trivial.
        """
        try:
            return self._process(session_id, session_timestamp)
        except Exception:
            logger.error("Memory write failed for session %s", session_id, exc_info=True)
            return []

    def _process(self, session_id: str, session_timestamp: str) -> list[Episode]:
        t0 = time.monotonic()

        utterances = self._storage.get_utterances(session_id)
        if len(utterances) < self._MIN_UTTERANCES:
            logger.debug(
                "Session %s too short (%d utterances), skipping",
                session_id,
                len(utterances),
            )
            return []

        # 1. Episode extraction
        episodes = self._extract_episodes(utterances, session_id, session_timestamp)
        if not episodes:
            logger.info("No episodes extracted from session %s", session_id)
            duration_ms = (time.monotonic() - t0) * 1000
            self._storage.mark_session_processed(
                session_id,
                duration_ms=duration_ms,
                episode_count=0,
            )
            return []

        # 2. Store episodes + embeddings
        stored = self._store_episodes(episodes)
        if not stored:
            duration_ms = (time.monotonic() - t0) * 1000
            self._storage.mark_session_processed(
                session_id,
                duration_ms=duration_ms,
                episode_count=0,
            )
            return []

        # 3. Profile extraction from episodes
        facts = self._extract_profile_facts(stored)

        # 4. Profile merge
        if facts:
            self._merge_profiles(facts, session_timestamp)

        duration_ms = (time.monotonic() - t0) * 1000
        self._storage.mark_session_processed(
            session_id,
            duration_ms=duration_ms,
            episode_count=len(stored),
        )
        logger.info(
            "Session %s processed: %d episodes in %.0fms",
            session_id,
            len(stored),
            duration_ms,
        )
        return stored

    # ------------------------------------------------------------------
    # Episode extraction
    # ------------------------------------------------------------------

    def _extract_episodes(
        self,
        utterances: list[tuple[str, str, str, int]],
        session_id: str,
        session_timestamp: str,
    ) -> list[Episode]:
        """Extract episodes from utterances, with windowing if needed."""
        total_tokens = sum(tc for _, _, _, tc in utterances)
        max_tokens = self._WRITE_MAX_INPUT_TOKENS

        windows = [utterances] if total_tokens <= max_tokens else self._split_into_windows(utterances, max_tokens)

        all_episodes: list[list[Episode]] = []
        for window in windows:
            episodes = self._extract_episodes_from_window(window, session_id, session_timestamp)
            all_episodes.append(episodes)

        if len(all_episodes) <= 1:
            return all_episodes[0] if all_episodes else []

        return self._deduplicate_episodes(all_episodes)

    def _extract_episodes_from_window(
        self,
        utterances: list[tuple[str, str, str, int]],
        session_id: str,
        session_timestamp: str,
    ) -> list[Episode]:
        """Extract episodes from a single window of utterances."""
        session_date = session_timestamp[:10]  # YYYY-MM-DD
        messages = build_episode_extraction_messages(
            utterances, session_date, system_prompt=self._episode_system_prompt
        )

        text = self._call_llm(messages, EPISODE_EXTRACTION_SCHEMA)
        if text is None:
            return []

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse episode extraction JSON")
            return []

        episodes: list[Episode] = []
        for item in data.get("episodes", []):
            ep_text = item.get("text", "").strip()
            if not ep_text:
                continue
            episodes.append(
                Episode(
                    id=None,
                    text=ep_text,
                    timestamp=session_timestamp,
                    session_id=session_id,
                    importance=1.0,
                    last_cited_at=session_timestamp,
                    citation_count=0,
                    embedding=None,
                )
            )
        return episodes

    # ------------------------------------------------------------------
    # Episode storage
    # ------------------------------------------------------------------

    def _store_episodes(self, episodes: list[Episode]) -> list[Episode]:
        """Persist episodes, generate embeddings, update vector index."""
        stored: list[Episode] = []
        for ep in episodes:
            eid = self._storage.add_episode(ep)
            if eid is not None:
                ep.id = eid
                stored.append(ep)
            else:
                logger.warning("Failed to store episode: %s", ep.text[:50])

        if not stored:
            return []

        # Batch embedding
        try:
            texts = [ep.text for ep in stored]
            embeddings = self._embedder.embed_batch(texts)
            for ep, emb in zip(stored, embeddings, strict=True):
                self._storage.update_episode_embedding(ep.id, emb)
                self._vector_index.add(ep.id, emb)
                ep.embedding = emb
        except Exception:
            logger.warning(
                "Embedding generation failed; episodes stored without embeddings",
                exc_info=True,
            )

        return stored

    # ------------------------------------------------------------------
    # Profile extraction
    # ------------------------------------------------------------------

    def _extract_profile_facts(self, episodes: list[Episode]) -> list[dict[str, str]]:
        """Extract profile facts from episodes."""
        messages = build_profile_extraction_messages(episodes, self._profile_schema_text)

        text = self._call_llm(messages, PROFILE_EXTRACTION_SCHEMA)
        if text is None:
            return []

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse profile extraction JSON")
            return []

        valid_topics = set(PROFILE_SCHEMA.keys())
        facts: list[dict[str, str]] = []
        for item in data.get("facts", []):
            topic = item.get("topic", "")
            if topic not in valid_topics:
                logger.debug("Dropping fact with unknown topic: %s", topic)
                continue
            sub_topic = item.get("sub_topic", "").strip()
            content = item.get("content", "").strip()
            if sub_topic and content:
                facts.append({"topic": topic, "sub_topic": sub_topic, "content": content})

        return facts

    # ------------------------------------------------------------------
    # Profile merge
    # ------------------------------------------------------------------

    def _merge_profiles(self, new_facts: list[dict[str, str]], timestamp: str) -> None:
        """Merge extracted facts into existing profile slots."""
        existing = self._storage.get_all_profiles()
        messages = build_profile_merge_messages(
            existing,
            new_facts,
            self._PROFILE_MAX_CONTENT_TOKENS,
            self._PROFILE_CONTENT_WARN_RATIO,
            token_counter=self._token_counter,
        )

        text = self._call_llm(messages, PROFILE_MERGE_SCHEMA)
        if text is None:
            return

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse profile merge JSON")
            return

        # Build (topic, sub_topic) → Profile lookup
        slot_map: dict[tuple[str, str], Profile] = {}
        for p in existing:
            slot_map[(p.topic, p.sub_topic)] = p

        for action_item in data.get("actions", []):
            try:
                self._execute_merge_action(action_item, slot_map, new_facts, timestamp)
            except Exception:
                logger.warning(
                    "Failed to execute merge action: %s",
                    action_item,
                    exc_info=True,
                )

    def _execute_merge_action(
        self,
        action_item: dict[str, Any],
        slot_map: dict[tuple[str, str], Profile],
        new_facts: list[dict[str, str]],
        timestamp: str,
    ) -> None:
        """Execute a single merge action (APPEND/UPDATE/ABORT)."""
        action = action_item.get("action")
        fact_idx = action_item.get("fact_index", 0)
        new_content = action_item.get("new_content")

        if action == "ABORT":
            return

        # Validate fact index
        if not (1 <= fact_idx <= len(new_facts)):
            logger.warning("Invalid fact_index %d", fact_idx)
            return

        fact = new_facts[fact_idx - 1]
        key = (fact["topic"], fact["sub_topic"])

        if action == "APPEND":
            content = new_content or fact["content"]
            self._storage.upsert_profile(
                Profile(
                    id=None,
                    topic=fact["topic"],
                    sub_topic=fact["sub_topic"],
                    content=content,
                    updated_at=timestamp,
                )
            )

        elif action == "UPDATE":
            existing_profile = slot_map.get(key)
            if existing_profile is None:
                logger.warning(
                    "UPDATE for non-existent slot %s::%s, treating as APPEND",
                    fact["topic"],
                    fact["sub_topic"],
                )
                content = new_content or fact["content"]
                self._storage.upsert_profile(
                    Profile(
                        id=None,
                        topic=fact["topic"],
                        sub_topic=fact["sub_topic"],
                        content=content,
                        updated_at=timestamp,
                    )
                )
                return
            if not new_content:
                logger.warning(
                    "UPDATE without new_content for %s::%s",
                    fact["topic"],
                    fact["sub_topic"],
                )
                return
            self._storage.upsert_profile(
                Profile(
                    id=existing_profile.id,
                    topic=existing_profile.topic,
                    sub_topic=existing_profile.sub_topic,
                    content=new_content,
                    updated_at=timestamp,
                )
            )

    # ------------------------------------------------------------------
    # Windowing
    # ------------------------------------------------------------------

    def _split_into_windows(
        self,
        utterances: list[tuple[str, str, str, int]],
        max_tokens: int,
    ) -> list[list[tuple[str, str, str, int]]]:
        """Split utterances into overlapping windows based on token count."""
        overlap_tokens = int(max_tokens * self._WRITE_WINDOW_OVERLAP_RATIO)
        windows: list[list[tuple[str, str, str, int]]] = []
        current: list[tuple[str, str, str, int]] = []
        current_tokens = 0

        for utt in utterances:
            utt_tokens = utt[3]  # token_count
            # If adding this utterance exceeds limit and we have content, split
            if current_tokens + utt_tokens > max_tokens and current:
                windows.append(current)
                # Keep utterances from the end that fit within overlap_tokens
                overlap_utts: list[tuple[str, str, str, int]] = []
                overlap_sum = 0
                for u in reversed(current):
                    if overlap_sum + u[3] > overlap_tokens:
                        break
                    overlap_utts.append(u)
                    overlap_sum += u[3]
                overlap_utts.reverse()
                current = overlap_utts
                current_tokens = overlap_sum

            current.append(utt)
            current_tokens += utt_tokens

        if current:
            windows.append(current)

        return windows

    def _deduplicate_episodes(self, episodes_per_window: list[list[Episode]]) -> list[Episode]:
        """Deduplicate episodes across windows using embedding similarity + LLM.

        Processes candidates sequentially: after each merge/discard, the
        affected result embedding is updated so subsequent candidates
        compare against the current state.
        """
        if not episodes_per_window:
            return []

        result = list(episodes_per_window[0])
        if not result:
            for window_episodes in episodes_per_window[1:]:
                result.extend(window_episodes)
            return result

        result_embeddings = list(self._embedder.embed_batch([ep.text for ep in result]))
        threshold = self._WRITE_DEDUP_THRESHOLD

        for window_episodes in episodes_per_window[1:]:
            if not window_episodes:
                continue

            for candidate in window_episodes:
                cand_emb = self._embedder.embed(candidate.text)

                # Find best match in current result
                best_sim = 0.0
                best_ri = -1
                for ri, exist_emb in enumerate(result_embeddings):
                    sim = float(
                        np.dot(cand_emb, exist_emb) / (np.linalg.norm(cand_emb) * np.linalg.norm(exist_emb) + 1e-9)
                    )
                    if sim > best_sim:
                        best_sim = sim
                        best_ri = ri

                if best_sim < threshold:
                    result.append(candidate)
                    result_embeddings.append(cand_emb)
                    continue

                # LLM decides how to resolve
                action = self._resolve_dedup_pair(result[best_ri].text, candidate.text)
                act = action.get("action", "KEEP_BOTH")

                if act == "MERGE" and action.get("merged"):
                    result[best_ri] = Episode(
                        id=result[best_ri].id,
                        text=action["merged"],
                        timestamp=result[best_ri].timestamp,
                        session_id=result[best_ri].session_id,
                        importance=result[best_ri].importance,
                        last_cited_at=result[best_ri].last_cited_at,
                        citation_count=result[best_ri].citation_count,
                        embedding=None,
                    )
                    result_embeddings[best_ri] = self._embedder.embed(action["merged"])
                elif act == "DISCARD_FIRST":
                    result[best_ri] = candidate
                    result_embeddings[best_ri] = cand_emb
                elif act == "KEEP_BOTH":
                    result.append(candidate)
                    result_embeddings.append(cand_emb)
                # DISCARD_SECOND: keep result[best_ri] as-is

        return result

    def _resolve_dedup_pair(self, existing: str, candidate: str) -> dict[str, Any]:
        """Call LLM to resolve a single duplicate episode pair.

        On failure, falls back to keeping the longer episode.
        """
        messages = build_episode_dedup_messages(existing, candidate)
        text = self._call_llm(messages, EPISODE_DEDUP_SCHEMA)
        if text is None:
            return self._dedup_fallback(existing, candidate)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse episode dedup JSON")
            return self._dedup_fallback(existing, candidate)

        action = data.get("action")
        if action not in ("MERGE", "KEEP_BOTH", "DISCARD_FIRST", "DISCARD_SECOND"):
            return self._dedup_fallback(existing, candidate)

        return data

    @staticmethod
    def _dedup_fallback(existing: str, candidate: str) -> dict[str, Any]:
        """Fallback when LLM dedup fails: keep the longer episode."""
        return {"action": "DISCARD_FIRST" if len(candidate) > len(existing) else "DISCARD_SECOND"}

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------

    def _call_llm(self, messages: list[dict[str, Any]], response_format: dict[str, Any]) -> str | None:
        """Call LLM and return the full response text. None on failure."""
        try:
            stream = self._llm.generate(messages, tools=[], response_format=response_format)
            # Consume the stream to get the full text
            for _ in stream:
                pass
            return stream.text
        except Exception:
            logger.warning("LLM call failed during memory write", exc_info=True)
            return None
