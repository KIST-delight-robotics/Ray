"""Memory write pipeline: extract episodes and update user notes from session utterances.

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
    USER_NOTES_UPDATE_SCHEMA,
    build_episode_dedup_messages,
    build_episode_extraction_messages,
    build_user_notes_messages,
    label_user_notes,
)
from voice_pipeline.memory.storage import EVENT_ROLE, RADIO_PAUSED, RADIO_ROLE, RADIO_START, SQLiteMemoryStorage
from voice_pipeline.memory.types import NOTE_KINDS, Episode, UserNote
from voice_pipeline.memory.vector_index import NumpyVectorIndex
from voice_pipeline.types import ILLM, IEmbedder

logger = logging.getLogger("voice_pipeline.memory")

Utterance = tuple[str, str, str, int]  # (role, text, timestamp, token_count)


def select_utterances(rows: list[Utterance]) -> list[Utterance]:
    """추출 입력 — 사건 줄을 빼고, 사용자가 말을 걸지 않은(paused 없는) 대본의 낭독을 뺀다.

    대본 하나는 ``radio_start`` 부터 다음 ``radio_start`` 까지다. 곡을 틀기로 한 사건(``radio_song``)은 끝으로 보지
    않는다 — 낭독 세그먼트는 닫힐 때 저장되어 곡 사건보다 늦게 남기도 한다. 사용자 말은 어느 대본에서든 남긴다.
    """
    talked: set[int] = set()  # paused 가 있었던 대본 번호
    script_of: list[int | None] = []
    script: int | None = None
    for role, text, _, _ in rows:
        if role == EVENT_ROLE:
            name = text.split(" ", 1)[0]
            if name == RADIO_START:
                script = len(script_of)
            elif name == RADIO_PAUSED and script is not None:
                talked.add(script)
        script_of.append(script)
    return [
        row
        for row, n in zip(rows, script_of, strict=True)
        if row[0] != EVENT_ROLE and not (row[0] == RADIO_ROLE and n is not None and n not in talked)
    ]


class MemoryWriter:
    """Extracts episodes and updates user notes from session utterances.

    Thread safety: designed to be called from a single thread.
    """

    _MIN_UTTERANCES = 2  # 추출 시도에 필요한 최소 utterance 수
    _WRITE_MAX_INPUT_TOKENS = 8000  # 에피소드 추출 윈도우 최대 토큰 수 (초과 시 분할)
    _WRITE_WINDOW_OVERLAP_RATIO = 0.25  # 인접 윈도우 overlap 비율 (0.0–1.0)
    _WRITE_DEDUP_THRESHOLD = 0.8  # 중복 판정 코사인 유사도 임계값

    def __init__(
        self,
        storage: SQLiteMemoryStorage,
        vector_index: NumpyVectorIndex,
        embedder: IEmbedder,
        llm: ILLM,
        *,
        episode_system_prompt: str | None = None,
    ) -> None:
        """
        Args:
            storage: Episode/user-note persistence backend.
            vector_index: Vector index updated with new episode embeddings.
            embedder: Episode embedder.
            llm: Extraction LLM.
            episode_system_prompt: 에피소드 추출 시스템 프롬프트 오버라이드
                (중립 주입점 — 평가에서 프롬프트 변형 실험용). ``None``이면
                기본 프롬프트.
        """
        self._storage = storage
        self._vector_index = vector_index
        self._embedder = embedder
        self._llm = llm
        self._episode_system_prompt = episode_system_prompt

    def process_session(self, session_id: str, session_timestamp: str) -> list[Episode]:
        """Process a completed session: extract episodes and update user notes.

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

        utterances = select_utterances(self._storage.get_utterances(session_id))
        if len(utterances) < self._MIN_UTTERANCES:
            logger.debug(
                "Session %s too short (%d utterances), skipping",
                session_id,
                len(utterances),
            )
            return []

        # 1. Episode extraction + storage (embeddings, vector index)
        episodes = self._extract_episodes(utterances, session_id, session_timestamp)
        if not episodes:
            logger.info("No episodes extracted from session %s", session_id)
        stored = self._store_episodes(episodes)

        # 2. User notes from the transcript — 에피소드가 없어도 돈다: "앞으로 반말해" 한마디 세션은
        #    사소하다고 에피소드가 0개이기 쉽다. 에피소드 요약에선 라디오 맥락 같은 출처도 사라진다
        self._update_user_notes(utterances, session_id, session_timestamp)

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
    # User notes
    # ------------------------------------------------------------------

    def _update_user_notes(self, utterances: list[Utterance], session_id: str, timestamp: str) -> None:
        """Apply the add/replace/delete ops one LLM call returns for the current notes and the transcript."""
        labeled = label_user_notes(self._storage.get_user_notes())
        text = self._call_llm(build_user_notes_messages(labeled, utterances), USER_NOTES_UPDATE_SCHEMA)
        if text is None:
            return

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse user notes update JSON")
            return

        for op in data.get("ops", []):
            try:
                self._apply_user_note_op(op, labeled, session_id, timestamp)
            except Exception:
                logger.warning("Failed to apply user note op: %s", op, exc_info=True)

    def _apply_user_note_op(
        self,
        op: dict[str, Any],
        labeled: dict[str, UserNote],
        session_id: str,
        timestamp: str,
    ) -> None:
        """Apply one op. Ops naming an unknown item or missing their text are logged and skipped."""
        action = op.get("op")
        text = (op.get("text") or "").strip()

        if action == "add":
            kind = op.get("list")
            if kind not in NOTE_KINDS or not text:
                logger.warning("Skipping invalid user note add: %s", op)
                return
            self._storage.add_user_note(UserNote(None, kind, text, session_id, timestamp))
            logger.info("User note added (%s): %s", kind, text)
            return

        note = labeled.get(op.get("target") or "")
        if note is None or note.id is None:
            logger.warning("Skipping user note op on unknown item: %s", op)
            return
        if action == "replace" and text:
            self._storage.update_user_note(note.id, text, session_id, timestamp)
            logger.info("User note replaced (%s): %s -> %s", note.kind, note.text, text)
        elif action == "delete":
            self._storage.delete_user_note(note.id)
            logger.info("User note deleted (%s): %s", note.kind, note.text)
        else:
            logger.warning("Skipping invalid user note op: %s", op)

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
