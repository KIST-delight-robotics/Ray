"""Unit tests for MemoryWriter."""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

import numpy as np

from voice_pipeline.memory.storage import (
    EVENT_ROLE,
    RADIO_PAUSED,
    RADIO_RESUME,
    RADIO_ROLE,
    RADIO_SONG,
    RADIO_START,
    SQLiteMemoryStorage,
)
from voice_pipeline.memory.types import UserNote
from voice_pipeline.memory.vector_index import NumpyVectorIndex
from voice_pipeline.memory.writer import MemoryWriter, select_utterances
from voice_pipeline.types import ILLM, IEmbedder, LLMResult, LLMStream

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

_DIM = 384
_TIMESTAMP = "2026-04-01 10:00:00"
_SESSION_ID = "test-session-1"


class FakeLLM(ILLM):
    """LLM that returns pre-configured JSON responses in order."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.messages_log: list[list[dict[str, Any]]] = []

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMStream:
        self.messages_log.append(messages)
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        text = self._responses[idx]

        def gen() -> Generator[str, None, None]:
            yield text

        return LLMStream(gen(), result_fn=lambda t: LLMResult(text=t))


class FakeEmbedder(IEmbedder):
    """Embedder that returns fixed vectors."""

    def embed(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(hash(text) % (2**31))
        return rng.standard_normal(_DIM).astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts])

    @property
    def dimension(self) -> int:
        return _DIM

    @property
    def model_name(self) -> str:
        return "fake"


class ConstantEmbedder(IEmbedder):
    """Embedder returning identical vectors — cosine similarity always 1.0."""

    def embed(self, text: str) -> np.ndarray:
        return np.ones(_DIM, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), _DIM), dtype=np.float32)

    @property
    def dimension(self) -> int:
        return _DIM

    @property
    def model_name(self) -> str:
        return "constant"


_WRITER_CLASS_VAR_MAP = {
    "write_max_input_tokens": "_WRITE_MAX_INPUT_TOKENS",
    "write_window_overlap_ratio": "_WRITE_WINDOW_OVERLAP_RATIO",
    "write_dedup_threshold": "_WRITE_DEDUP_THRESHOLD",
}


def _apply_writer_overrides(monkeypatch, **overrides: Any) -> None:
    """Translate legacy MemoryConfig kwargs into MemoryWriter class var monkeypatches."""
    for key, value in overrides.items():
        monkeypatch.setattr(MemoryWriter, _WRITER_CLASS_VAR_MAP[key], value)


def _make_writer(
    llm_responses: list[str],
    storage: SQLiteMemoryStorage | None = None,
    embedder: IEmbedder | None = None,
) -> tuple[MemoryWriter, SQLiteMemoryStorage, NumpyVectorIndex, FakeLLM]:
    st = storage or SQLiteMemoryStorage(":memory:", dimension=_DIM)
    vi = NumpyVectorIndex()
    emb = embedder or FakeEmbedder()
    llm = FakeLLM(llm_responses)
    writer = MemoryWriter(st, vi, emb, llm)
    return writer, st, vi, llm


def _add_utterances(
    storage: SQLiteMemoryStorage,
    session_id: str = _SESSION_ID,
    count: int = 4,
) -> None:
    """Add sample utterances to storage."""
    ts = "2026-04-01 10:00:0"
    utts = [
        ("user", "I watched Interstellar last night and it was amazing.", f"{ts}0", 10),
        ("assistant", "That's a great movie! What did you like?", f"{ts}5", 10),
        ("user", "The space scenes were incredible. I cried.", f"{ts}9", 10),
        ("assistant", "The ending is really emotional.", f"{ts}9", 6),
    ]
    for role, text, ts, tc in utts[:count]:
        storage.add_utterance(session_id, role, text, ts, tc)


def _episode_response(*texts: str) -> str:
    """Build a JSON response for episode extraction."""
    return json.dumps({"episodes": [{"text": t} for t in texts]})


def _notes_response(*ops: tuple[str, str, str | None, str | None]) -> str:
    """Build a JSON response for the user-notes update: (op, list, target, text) per op."""
    return json.dumps({"ops": [{"op": o, "list": k, "target": t, "text": x} for o, k, t, x in ops]})


def _note(kind: str, text: str) -> UserNote:
    return UserNote(None, kind, text, "s-old", "2026-03-01 10:00:00")


def _dedup_response(action: str, merged: str | None = None) -> str:
    """Build a JSON response for episode deduplication (single pair)."""
    return json.dumps({"action": action, "merged": merged})


# ---------------------------------------------------------------------------
# Episode extraction tests
# ---------------------------------------------------------------------------


class TestEpisodeExtraction:
    def test_process_session_extracts_episodes(self) -> None:
        ep_json = _episode_response(
            "The user watched Interstellar and was deeply moved by it.",
            "The user cried at the ending of the movie.",
        )
        notes_json = _notes_response()
        writer, storage, vi, _ = _make_writer([ep_json, notes_json])
        _add_utterances(storage)

        episodes = writer.process_session(_SESSION_ID, _TIMESTAMP)

        assert len(episodes) == 2
        assert episodes[0].text == "The user watched Interstellar and was deeply moved by it."
        assert episodes[0].timestamp == _TIMESTAMP
        assert episodes[0].session_id == _SESSION_ID
        assert episodes[0].importance == 1.0
        assert episodes[0].citation_count == 0
        assert episodes[0].id is not None

    def test_process_session_empty_utterances(self) -> None:
        writer, storage, _, _ = _make_writer(["should not be called"])
        # No utterances added
        episodes = writer.process_session(_SESSION_ID, _TIMESTAMP)
        assert episodes == []

    def test_process_session_single_utterance(self) -> None:
        writer, storage, _, _ = _make_writer(["should not be called"])
        _add_utterances(storage, count=1)
        episodes = writer.process_session(_SESSION_ID, _TIMESTAMP)
        assert episodes == []

    def test_empty_extraction_result(self) -> None:
        ep_json = _episode_response()  # empty episodes
        writer, storage, _, _ = _make_writer([ep_json])
        _add_utterances(storage)

        episodes = writer.process_session(_SESSION_ID, _TIMESTAMP)
        assert episodes == []

    def test_session_marked_processed_after_extraction(self) -> None:
        ep_json = _episode_response("An episode.")
        notes_json = _notes_response()
        writer, storage, _, _ = _make_writer([ep_json, notes_json])
        _add_utterances(storage)

        writer.process_session(_SESSION_ID, _TIMESTAMP)

        assert _SESSION_ID in storage.get_processed_session_ids([_SESSION_ID])

    def test_session_marked_processed_on_empty_extraction(self) -> None:
        """Even if 0 episodes extracted, session is marked as processed."""
        ep_json = _episode_response()  # empty
        writer, storage, _, _ = _make_writer([ep_json])
        _add_utterances(storage)

        writer.process_session(_SESSION_ID, _TIMESTAMP)

        assert _SESSION_ID in storage.get_processed_session_ids([_SESSION_ID])

    def test_session_not_marked_when_too_short(self) -> None:
        """Sessions below MIN_UTTERANCES are not marked as processed."""
        writer, storage, _, _ = _make_writer(["should not be called"])
        _add_utterances(storage, count=1)

        writer.process_session(_SESSION_ID, _TIMESTAMP)

        assert _SESSION_ID not in storage.get_processed_session_ids([_SESSION_ID])

    def test_event_rows_do_not_count_as_utterances(self) -> None:
        writer, storage, _, llm = _make_writer(["should not be called"])
        _add_utterances(storage, count=1)
        storage.add_utterance(_SESSION_ID, EVENT_ROLE, f"{RADIO_START} drawer", "2026-04-01 10:00:01", 0)
        storage.add_utterance(_SESSION_ID, EVENT_ROLE, RADIO_PAUSED, "2026-04-01 10:00:02", 0)

        assert writer.process_session(_SESSION_ID, _TIMESTAMP) == []
        assert llm.messages_log == []

    def test_radio_talk_the_user_talked_into_is_marked_in_the_transcript(self) -> None:
        writer, storage, _, llm = _make_writer([_episode_response()])
        ts = "2026-04-01 10:00:0"
        storage.add_utterance(_SESSION_ID, "user", "라디오 해줘", f"{ts}0", 3)
        storage.add_utterance(_SESSION_ID, EVENT_ROLE, f"{RADIO_START} drawer", f"{ts}1", 0)
        storage.add_utterance(_SESSION_ID, RADIO_ROLE, "오늘은 책상 서랍 이야기예요.", f"{ts}2", 5)
        storage.add_utterance(_SESSION_ID, "user", "서랍에 뭐 넣어 둬?", f"{ts}3", 4)
        storage.add_utterance(_SESSION_ID, EVENT_ROLE, RADIO_PAUSED, f"{ts}4", 0)

        writer.process_session(_SESSION_ID, _TIMESTAMP)

        transcript = llm.messages_log[0][-1]["content"]
        assert RADIO_START not in transcript
        assert "Assistant (radio talk): 오늘은 책상 서랍 이야기예요." in transcript

    def test_a_session_of_only_listening_is_skipped(self) -> None:
        writer, storage, _, llm = _make_writer(["should not be called"])
        ts = "2026-04-01 10:00:0"
        storage.add_utterance(_SESSION_ID, "user", "라디오 해줘", f"{ts}0", 3)
        storage.add_utterance(_SESSION_ID, EVENT_ROLE, f"{RADIO_START} drawer", f"{ts}1", 0)
        storage.add_utterance(_SESSION_ID, RADIO_ROLE, "오늘은 책상 서랍 이야기예요.", f"{ts}2", 5)
        storage.add_utterance(_SESSION_ID, EVENT_ROLE, f"{RADIO_SONG} IAM", f"{ts}3", 0)

        assert writer.process_session(_SESSION_ID, _TIMESTAMP) == []
        assert llm.messages_log == []


class TestSelectUtterances:
    @staticmethod
    def _rows(*rows: tuple[str, str]) -> list[tuple[str, str, str, int]]:
        return [(role, text, f"2026-04-01 10:00:{i:02d}", 1) for i, (role, text) in enumerate(rows)]

    def test_drops_the_talk_of_a_script_without_a_pause_and_keeps_user_speech(self) -> None:
        rows = self._rows(
            ("user", "라디오 해줘"),
            (EVENT_ROLE, f"{RADIO_START} a"),
            (RADIO_ROLE, "에이 낭독"),
            ("user", "음"),  # on 인 채로 지나간 말
            (EVENT_ROLE, f"{RADIO_SONG} s"),
            (RADIO_ROLE, "에이 마지막 낭독"),  # 곡 사건보다 늦게 저장된 세그먼트도 그 대본이다
        )
        assert [r[1] for r in select_utterances(rows)] == ["라디오 해줘", "음"]

    def test_keeps_the_whole_talk_of_a_script_with_a_pause(self) -> None:
        rows = self._rows(
            (EVENT_ROLE, f"{RADIO_START} a"),
            (RADIO_ROLE, "에이 앞 낭독"),
            ("user", "그건 왜 그래?"),
            (EVENT_ROLE, RADIO_PAUSED),
            ("assistant", "대답"),
            (EVENT_ROLE, RADIO_RESUME),
            (RADIO_ROLE, "에이 뒤 낭독"),
            (EVENT_ROLE, f"{RADIO_SONG} s"),
            (EVENT_ROLE, f"{RADIO_START} b"),
            (RADIO_ROLE, "비 낭독"),
        )
        assert [r[1] for r in select_utterances(rows)] == ["에이 앞 낭독", "그건 왜 그래?", "대답", "에이 뒤 낭독"]

    def test_a_new_script_started_while_paused_is_its_own_unit(self) -> None:
        rows = self._rows(
            (EVENT_ROLE, f"{RADIO_START} a"),
            (RADIO_ROLE, "에이 낭독"),
            (EVENT_ROLE, RADIO_PAUSED),
            ("user", "다른 이야기 해 줘"),
            (EVENT_ROLE, f"{RADIO_START} b"),
            (RADIO_ROLE, "비 낭독"),
        )
        assert [r[1] for r in select_utterances(rows)] == ["에이 낭독", "다른 이야기 해 줘"]

    def test_episodes_have_embeddings(self) -> None:
        ep_json = _episode_response("The user loves sci-fi movies.")
        notes_json = _notes_response()
        writer, storage, vi, _ = _make_writer([ep_json, notes_json])
        _add_utterances(storage)

        episodes = writer.process_session(_SESSION_ID, _TIMESTAMP)

        assert len(episodes) == 1
        assert episodes[0].embedding is not None
        assert episodes[0].embedding.shape == (_DIM,)
        assert len(vi) == 1

    def test_episodes_stored_in_storage(self) -> None:
        ep_json = _episode_response("A memorable episode.")
        notes_json = _notes_response()
        writer, storage, _, _ = _make_writer([ep_json, notes_json])
        _add_utterances(storage)

        episodes = writer.process_session(_SESSION_ID, _TIMESTAMP)

        loaded = storage.get_episode(episodes[0].id)
        assert loaded is not None
        assert loaded.text == "A memorable episode."


# ---------------------------------------------------------------------------
# User notes
# ---------------------------------------------------------------------------


class TestUserNotes:
    def test_update_sees_current_lists_and_the_transcript(self) -> None:
        writer, storage, _, llm = _make_writer([_episode_response("The user loves Interstellar."), _notes_response()])
        storage.add_user_note(_note("profile", "The user's name is 재헌."))
        _add_utterances(storage)

        writer.process_session(_SESSION_ID, _TIMESTAMP)

        update_input = llm.messages_log[1][-1]["content"]
        assert "Profile:\nP1. The user's name is 재헌.\n\nPreferences:\n(empty)" in update_input
        assert "User: I watched Interstellar last night and it was amazing." in update_input
        assert "The user loves Interstellar." not in update_input  # 에피소드가 아니라 원문을 본다

    def test_add_stores_notes_with_their_session(self) -> None:
        notes_json = _notes_response(
            ("add", "profile", None, "The user is a graduate student."),
            ("add", "preference", None, "Speak in 반말."),
        )
        writer, storage, _, _ = _make_writer([_episode_response("Episode."), notes_json])
        _add_utterances(storage)

        writer.process_session(_SESSION_ID, _TIMESTAMP)

        notes = storage.get_user_notes()
        assert [(n.kind, n.text) for n in notes] == [
            ("profile", "The user is a graduate student."),
            ("preference", "Speak in 반말."),
        ]
        assert all(n.session_id == _SESSION_ID and n.updated_at == _TIMESTAMP for n in notes)

    def test_replace_and_delete_address_ids_numbered_per_list(self) -> None:
        notes_json = _notes_response(
            ("replace", "profile", "P2", "The user works at KIST."),
            ("delete", "preference", "R1", None),
        )
        writer, storage, _, _ = _make_writer([_episode_response("Episode."), notes_json])
        storage.add_user_note(_note("profile", "The user's name is 재헌."))  # P1
        storage.add_user_note(_note("preference", "Speak in 반말."))  # R1
        storage.add_user_note(_note("profile", "The user is a student."))  # P2
        _add_utterances(storage)

        writer.process_session(_SESSION_ID, _TIMESTAMP)

        notes = storage.get_user_notes()
        assert [(n.kind, n.text) for n in notes] == [
            ("profile", "The user's name is 재헌."),
            ("profile", "The user works at KIST."),
        ]
        assert notes[0].session_id == "s-old"  # 손대지 않은 줄은 그대로
        assert notes[1].session_id == _SESSION_ID

    def test_invalid_ops_are_skipped(self) -> None:
        notes_json = _notes_response(
            ("replace", "profile", "P9", "No such item."),
            ("replace", "profile", "P1", None),
            ("add", "profile", None, "  "),
            ("add", "hobby", None, "Not a list."),
            ("add", "profile", None, "The user lives in Seoul."),
        )
        writer, storage, _, _ = _make_writer([_episode_response("Episode."), notes_json])
        storage.add_user_note(_note("profile", "The user's name is 재헌."))
        _add_utterances(storage)

        writer.process_session(_SESSION_ID, _TIMESTAMP)

        assert [n.text for n in storage.get_user_notes()] == [
            "The user's name is 재헌.",
            "The user lives in Seoul.",
        ]

    def test_update_runs_even_without_episodes(self) -> None:
        """사소하다고 에피소드가 0개인 세션에도 "앞으로 반말해" 같은 선호가 있을 수 있다."""
        notes_json = _notes_response(("add", "preference", None, "Speak in 반말."))
        writer, storage, _, llm = _make_writer([_episode_response(), notes_json])
        _add_utterances(storage)

        episodes = writer.process_session(_SESSION_ID, _TIMESTAMP)

        assert episodes == []
        assert llm._call_count == 2
        assert [n.text for n in storage.get_user_notes()] == ["Speak in 반말."]
        assert _SESSION_ID in storage.get_processed_session_ids([_SESSION_ID])


# ---------------------------------------------------------------------------
# Window tests
# ---------------------------------------------------------------------------


class TestWindowing:
    def test_short_session_single_window(self) -> None:
        ep_json = _episode_response("Short session episode.")
        notes_json = _notes_response()
        writer, storage, _, llm = _make_writer([ep_json, notes_json])
        _add_utterances(storage)

        writer.process_session(_SESSION_ID, _TIMESTAMP)

        # Episode extraction + profile extraction = 2 calls
        assert llm._call_count == 2

    def test_long_session_windowed(self, monkeypatch) -> None:
        """Session exceeding token limit should be split into windows."""
        ep_json_1 = _episode_response("Episode from window 1.")
        ep_json_2 = _episode_response("Episode from window 2.")
        dedup_json = _dedup_response("KEEP_BOTH")
        notes_json = _notes_response()
        _apply_writer_overrides(
            monkeypatch,
            write_max_input_tokens=25,
            write_window_overlap_ratio=0.0,
            write_dedup_threshold=0.8,
        )
        writer, storage, _, llm = _make_writer(
            [ep_json_1, ep_json_2, dedup_json, notes_json],
            embedder=ConstantEmbedder(),
        )
        _add_utterances(storage)

        episodes = writer.process_session(_SESSION_ID, _TIMESTAMP)

        assert len(episodes) == 2
        # 2 extraction + 1 dedup + 1 profile = 4
        assert llm._call_count == 4

    def test_dedup_merge(self, monkeypatch) -> None:
        """LLM MERGE action combines two episodes into one."""
        ep_json_1 = _episode_response("The user watched Interstellar.")
        ep_json_2 = _episode_response(
            "The user watched Interstellar and cried.",
            "The user cried at the ending.",
        )
        # Sequential: candidate 1 → MERGE, candidate 2 → KEEP_BOTH
        merge_json = _dedup_response("MERGE", "The user watched Interstellar and was deeply moved to tears.")
        keep_json = _dedup_response("KEEP_BOTH")
        notes_json = _notes_response()
        _apply_writer_overrides(
            monkeypatch,
            write_max_input_tokens=25,
            write_window_overlap_ratio=0.0,
            write_dedup_threshold=0.8,
        )
        writer, storage, _, _ = _make_writer(
            [ep_json_1, ep_json_2, merge_json, keep_json, notes_json],
            embedder=ConstantEmbedder(),
        )
        _add_utterances(storage)

        episodes = writer.process_session(_SESSION_ID, _TIMESTAMP)

        texts = [ep.text for ep in episodes]
        assert len(texts) == 2
        assert "The user watched Interstellar and was deeply moved to tears." in texts
        assert "The user cried at the ending." in texts

    def test_dedup_discard_first(self, monkeypatch) -> None:
        """LLM DISCARD_FIRST replaces the first episode with the second."""
        ep_json_1 = _episode_response("The user likes sci-fi.")
        ep_json_2 = _episode_response("The user loves sci-fi movies, especially Nolan films.")
        dedup_json = _dedup_response("DISCARD_FIRST")
        notes_json = _notes_response()
        _apply_writer_overrides(
            monkeypatch,
            write_max_input_tokens=25,
            write_window_overlap_ratio=0.0,
            write_dedup_threshold=0.8,
        )
        writer, storage, _, _ = _make_writer(
            [ep_json_1, ep_json_2, dedup_json, notes_json],
            embedder=ConstantEmbedder(),
        )
        _add_utterances(storage)

        episodes = writer.process_session(_SESSION_ID, _TIMESTAMP)

        assert len(episodes) == 1
        assert episodes[0].text == "The user loves sci-fi movies, especially Nolan films."

    def test_dedup_discard_second(self, monkeypatch) -> None:
        """LLM DISCARD_SECOND keeps only the first episode."""
        ep_json_1 = _episode_response("The user loves sci-fi movies, especially Nolan films.")
        ep_json_2 = _episode_response("The user likes sci-fi.")
        dedup_json = _dedup_response("DISCARD_SECOND")
        notes_json = _notes_response()
        _apply_writer_overrides(
            monkeypatch,
            write_max_input_tokens=25,
            write_window_overlap_ratio=0.0,
            write_dedup_threshold=0.8,
        )
        writer, storage, _, _ = _make_writer(
            [ep_json_1, ep_json_2, dedup_json, notes_json],
            embedder=ConstantEmbedder(),
        )
        _add_utterances(storage)

        episodes = writer.process_session(_SESSION_ID, _TIMESTAMP)

        assert len(episodes) == 1
        assert episodes[0].text == "The user loves sci-fi movies, especially Nolan films."

    def test_dedup_fallback_on_llm_failure(self, monkeypatch) -> None:
        """When dedup LLM fails, keep the longer episode."""
        ep_json_1 = _episode_response("Short.")
        ep_json_2 = _episode_response("This is a longer and more detailed episode.")
        notes_json = _notes_response()
        _apply_writer_overrides(
            monkeypatch,
            write_max_input_tokens=25,
            write_window_overlap_ratio=0.0,
            write_dedup_threshold=0.8,
        )
        # 2 extraction calls succeed, dedup call returns bad JSON, profile call
        writer, storage, _, _ = _make_writer(
            [ep_json_1, ep_json_2, "bad json", notes_json],
            embedder=ConstantEmbedder(),
        )
        _add_utterances(storage)

        episodes = writer.process_session(_SESSION_ID, _TIMESTAMP)

        assert len(episodes) == 1
        assert episodes[0].text == "This is a longer and more detailed episode."

    def test_high_threshold_skips_dedup(self, monkeypatch) -> None:
        """When no embeddings exceed threshold, no dedup LLM call is made."""
        ep_json_1 = _episode_response("Episode about movies.")
        ep_json_2 = _episode_response("Episode about cooking.")
        notes_json = _notes_response()
        # Random embeddings have near-zero cosine similarity, threshold=0.99 skips all
        _apply_writer_overrides(
            monkeypatch,
            write_max_input_tokens=25,
            write_window_overlap_ratio=0.0,
            write_dedup_threshold=0.99,
        )
        writer, storage, _, llm = _make_writer([ep_json_1, ep_json_2, notes_json])
        _add_utterances(storage)

        episodes = writer.process_session(_SESSION_ID, _TIMESTAMP)

        assert len(episodes) == 2
        # 2 extraction + 1 profile = 3 (no dedup call)
        assert llm._call_count == 3

    def test_overlap_ratio_based(self, monkeypatch) -> None:
        """Window overlap uses token ratio, not turn count."""
        # Verify windowing mechanics — dedup skipped via high threshold.
        # max=25, overlap_ratio=0.4 → overlap of 10 tokens
        # Utterances: 10+10+10+6=36.
        #   Window 1: [10,10]=20. Overlap: last 10 tokens → [10].
        #   Window 2: overlap[10]+[10]=20. Overlap: last 10 → [10].
        #   Window 3: overlap[10]+[6]=16.
        # → 3 windows, 3 extraction calls + 1 profile call.
        ep_json_1 = _episode_response("Window 1 episode.")
        ep_json_2 = _episode_response("Window 2 episode.")
        ep_json_3 = _episode_response("Window 3 episode.")
        notes_json = _notes_response()
        _apply_writer_overrides(
            monkeypatch,
            write_max_input_tokens=25,
            write_window_overlap_ratio=0.4,
            write_dedup_threshold=0.99,
        )
        writer, storage, _, llm = _make_writer([ep_json_1, ep_json_2, ep_json_3, notes_json])
        _add_utterances(storage)

        episodes = writer.process_session(_SESSION_ID, _TIMESTAMP)

        # 3 extraction + 0 dedup (random embeddings < 0.99) + 1 profile = 4
        assert llm._call_count == 4
        assert len(episodes) == 3


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_llm_failure_graceful(self) -> None:
        """LLM failure should not raise, should return empty."""

        class FailingLLM(ILLM):
            def generate(self, messages, tools=None, response_format=None):
                raise RuntimeError("LLM down")

        storage = SQLiteMemoryStorage(":memory:", dimension=_DIM)
        vi = NumpyVectorIndex()
        emb = FakeEmbedder()
        writer = MemoryWriter(storage, vi, emb, FailingLLM())
        _add_utterances(storage)

        episodes = writer.process_session(_SESSION_ID, _TIMESTAMP)
        assert episodes == []

    def test_invalid_json_response(self) -> None:
        """Malformed JSON should be handled gracefully."""
        writer, storage, _, _ = _make_writer(["not valid json"])
        _add_utterances(storage)

        episodes = writer.process_session(_SESSION_ID, _TIMESTAMP)
        assert episodes == []

    def test_user_notes_failure_episodes_still_stored(self) -> None:
        """A failed user-notes update should not affect already-stored episodes."""
        ep_json = _episode_response("A good episode.")
        # User-notes update returns bad JSON
        writer, storage, _, _ = _make_writer([ep_json, "bad json"])
        _add_utterances(storage)

        episodes = writer.process_session(_SESSION_ID, _TIMESTAMP)

        assert len(episodes) == 1
        assert storage.get_episode(episodes[0].id) is not None


class TestEpisodePromptOverride:
    """episode_system_prompt 주입점 (평가에서 프롬프트 변형 실험용)."""

    def test_override_replaces_system_prompt(self) -> None:
        st = SQLiteMemoryStorage(":memory:", dimension=_DIM)
        llm = FakeLLM(['{"episodes": []}'])
        writer = MemoryWriter(
            st,
            NumpyVectorIndex(),
            FakeEmbedder(),
            llm,
            episode_system_prompt="CUSTOM EXTRACTION PROMPT",
        )
        _add_utterances(st)

        writer.process_session(_SESSION_ID, _TIMESTAMP)

        assert llm.messages_log[0][0]["role"] == "system"
        assert llm.messages_log[0][0]["content"] == "CUSTOM EXTRACTION PROMPT"

    def test_default_prompt_used_when_not_overridden(self) -> None:
        from voice_pipeline.memory.prompts import _EPISODE_SYSTEM

        writer, storage, _, llm = _make_writer(['{"episodes": []}'])
        _add_utterances(storage)

        writer.process_session(_SESSION_ID, _TIMESTAMP)

        assert llm.messages_log[0][0]["content"] == _EPISODE_SYSTEM
