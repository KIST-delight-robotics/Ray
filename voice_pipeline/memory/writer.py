"""Memory write pipeline: extract episodes and profiles from session utterances.

Pure processing module — receives session data, performs LLM-based extraction,
and stores results. Does not manage triggers or scheduling (Phase 4).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from voice_pipeline.core.config import MemoryConfig
from voice_pipeline.core.interfaces import ILLM, IEmbedder, IMemoryStorage
from voice_pipeline.core.types import TokenCounter
from voice_pipeline.memory.prompts import (
    EPISODE_EXTRACTION_SCHEMA,
    PROFILE_EXTRACTION_SCHEMA,
    PROFILE_MERGE_SCHEMA,
    PROFILE_SCHEMA,
    build_episode_extraction_messages,
    build_profile_extraction_messages,
    build_profile_merge_messages,
    format_profile_schema,
)
from voice_pipeline.memory.types import Episode, Profile
from voice_pipeline.memory.vector_index import IVectorIndex

logger = logging.getLogger("voice_pipeline.memory")

# Minimum utterances required to attempt extraction.
_MIN_UTTERANCES = 2


class MemoryWriter:
    """Extracts episodes and profiles from session utterances.

    Thread safety: designed to be called from a single thread.
    """

    def __init__(
        self,
        storage: IMemoryStorage,
        vector_index: IVectorIndex,
        embedder: IEmbedder,
        llm: ILLM,
        config: MemoryConfig,
        token_counter: TokenCounter,
    ) -> None:
        self._storage = storage
        self._vector_index = vector_index
        self._embedder = embedder
        self._llm = llm
        self._config = config
        self._token_counter = token_counter
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
        utterances = self._storage.get_utterances(session_id)
        if len(utterances) < _MIN_UTTERANCES:
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
            return []

        # 2. Store episodes + embeddings
        stored = self._store_episodes(episodes)
        if not stored:
            return []

        # 3. Profile extraction from episodes
        facts = self._extract_profile_facts(stored)

        # 4. Profile merge
        if facts:
            self._merge_profiles(facts, session_timestamp)

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
        max_tokens = self._config.write_max_input_tokens

        if total_tokens <= max_tokens:
            windows = [utterances]
        else:
            windows = self._split_into_windows(utterances, max_tokens)

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
        messages = build_episode_extraction_messages(utterances, session_date)

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
            self._config.profile_max_content_tokens,
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
        overlap = self._config.write_window_overlap_turns
        windows: list[list[tuple[str, str, str, int]]] = []
        current: list[tuple[str, str, str, int]] = []
        current_tokens = 0

        for utt in utterances:
            utt_tokens = utt[3]  # token_count
            # If adding this utterance exceeds limit and we have content, split
            if current_tokens + utt_tokens > max_tokens and current:
                windows.append(current)
                # Start new window with overlap from end of previous
                if overlap > 0 and len(current) >= overlap:
                    current = list(current[-overlap:])
                    current_tokens = sum(u[3] for u in current)
                else:
                    current = []
                    current_tokens = 0

            current.append(utt)
            current_tokens += utt_tokens

        if current:
            windows.append(current)

        return windows

    def _deduplicate_episodes(self, episodes_per_window: list[list[Episode]]) -> list[Episode]:
        """Deduplicate episodes across windows using text similarity."""
        if not episodes_per_window:
            return []

        result = list(episodes_per_window[0])

        for window_episodes in episodes_per_window[1:]:
            for candidate in window_episodes:
                if not any(
                    _text_similarity(candidate.text, existing.text) > 0.85 for existing in result
                ):
                    result.append(candidate)

        return result

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------

    def _call_llm(
        self, messages: list[dict[str, Any]], response_format: dict[str, Any]
    ) -> str | None:
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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _text_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity for deduplication."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
