"""Unit tests for MemoryWriter."""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

import numpy as np
import pytest

from voice_pipeline.core.config import MemoryConfig
from voice_pipeline.core.interfaces import ILLM, IEmbedder
from voice_pipeline.core.types import LLMResult, LLMStream
from voice_pipeline.memory.storage import InMemoryMemoryStorage
from voice_pipeline.memory.types import Profile
from voice_pipeline.memory.vector_index import NumpyVectorIndex
from voice_pipeline.memory.writer import MemoryWriter, _text_similarity

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


def _make_config(**overrides: Any) -> MemoryConfig:
    defaults = {"embedding_dimension": _DIM}
    defaults.update(overrides)
    return MemoryConfig(**defaults)


def _make_writer(
    llm_responses: list[str],
    config: MemoryConfig | None = None,
    storage: InMemoryMemoryStorage | None = None,
) -> tuple[MemoryWriter, InMemoryMemoryStorage, NumpyVectorIndex, FakeLLM]:
    cfg = config or _make_config()
    st = storage or InMemoryMemoryStorage(dimension=_DIM)
    vi = NumpyVectorIndex()
    emb = FakeEmbedder()
    llm = FakeLLM(llm_responses)
    counter = lambda text: len(text.split())  # noqa: E731
    writer = MemoryWriter(st, vi, emb, llm, cfg, counter)
    return writer, st, vi, llm


def _add_utterances(
    storage: InMemoryMemoryStorage,
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


def _profile_facts_response(*facts: tuple[str, str, str]) -> str:
    """Build a JSON response for profile extraction."""
    return json.dumps({"facts": [{"topic": t, "sub_topic": s, "content": c} for t, s, c in facts]})


def _merge_response(*actions: tuple[str, int, str | None]) -> str:
    """Build a JSON response for profile merge."""
    return json.dumps(
        {
            "actions": [
                {
                    "action": a,
                    "fact_index": fi,
                    "new_content": nc,
                }
                for a, fi, nc in actions
            ]
        }
    )


# ---------------------------------------------------------------------------
# Episode extraction tests
# ---------------------------------------------------------------------------


class TestEpisodeExtraction:
    def test_process_session_extracts_episodes(self) -> None:
        ep_json = _episode_response(
            "The user watched Interstellar and was deeply moved by it.",
            "The user cried at the ending of the movie.",
        )
        facts_json = _profile_facts_response()
        writer, storage, vi, _ = _make_writer([ep_json, facts_json])
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

    def test_episodes_have_embeddings(self) -> None:
        ep_json = _episode_response("The user loves sci-fi movies.")
        facts_json = _profile_facts_response()
        writer, storage, vi, _ = _make_writer([ep_json, facts_json])
        _add_utterances(storage)

        episodes = writer.process_session(_SESSION_ID, _TIMESTAMP)

        assert len(episodes) == 1
        assert episodes[0].embedding is not None
        assert episodes[0].embedding.shape == (_DIM,)
        assert len(vi) == 1

    def test_episodes_stored_in_storage(self) -> None:
        ep_json = _episode_response("A memorable episode.")
        facts_json = _profile_facts_response()
        writer, storage, _, _ = _make_writer([ep_json, facts_json])
        _add_utterances(storage)

        episodes = writer.process_session(_SESSION_ID, _TIMESTAMP)

        loaded = storage.get_episode(episodes[0].id)
        assert loaded is not None
        assert loaded.text == "A memorable episode."


# ---------------------------------------------------------------------------
# Profile extraction and merge tests
# ---------------------------------------------------------------------------


class TestProfileMerge:
    def test_profile_append(self) -> None:
        ep_json = _episode_response("The user loves Interstellar.")
        facts_json = _profile_facts_response(
            ("interest", "movie", "Loves Interstellar"),
        )
        merge_json = _merge_response(
            ("APPEND", 1, "Loves Interstellar"),
        )
        writer, storage, _, _ = _make_writer([ep_json, facts_json, merge_json])
        _add_utterances(storage)

        writer.process_session(_SESSION_ID, _TIMESTAMP)

        profiles = storage.get_all_profiles()
        assert len(profiles) == 1
        assert profiles[0].topic == "interest"
        assert profiles[0].sub_topic == "movie"
        assert profiles[0].content == "Loves Interstellar"

    def test_profile_update(self) -> None:
        ep_json = _episode_response("The user mentioned Tenet.")
        facts_json = _profile_facts_response(
            ("interest", "movie", "Also likes Tenet"),
        )
        merge_json = _merge_response(
            ("UPDATE", 1, "Loves Interstellar, Tenet"),
        )
        writer, storage, _, _ = _make_writer([ep_json, facts_json, merge_json])
        _add_utterances(storage)

        # Pre-populate existing profile
        storage.upsert_profile(
            Profile(
                id=None,
                topic="interest",
                sub_topic="movie",
                content="Loves Interstellar",
                updated_at="2026-03-01 10:00:00",
            )
        )

        writer.process_session(_SESSION_ID, _TIMESTAMP)

        profiles = storage.get_all_profiles()
        assert len(profiles) == 1
        assert profiles[0].content == "Loves Interstellar, Tenet"

    def test_profile_abort(self) -> None:
        ep_json = _episode_response("The user mentioned movies.")
        facts_json = _profile_facts_response(
            ("interest", "movie", "Likes movies"),
        )
        merge_json = _merge_response(
            ("ABORT", 1, None),
        )
        writer, storage, _, _ = _make_writer([ep_json, facts_json, merge_json])
        _add_utterances(storage)

        writer.process_session(_SESSION_ID, _TIMESTAMP)

        profiles = storage.get_all_profiles()
        assert len(profiles) == 0

    def test_mixed_merge_actions(self) -> None:
        ep_json = _episode_response("User talked about movies and music.")
        facts_json = _profile_facts_response(
            ("interest", "movie", "Likes sci-fi"),
            ("interest", "music", "Enjoys jazz"),
            ("interest", "book", "Trivial mention"),
        )
        merge_json = _merge_response(
            ("UPDATE", 1, "Sci-fi and action movies"),
            ("APPEND", 2, "Enjoys jazz"),
            ("ABORT", 3, None),
        )
        writer, storage, _, _ = _make_writer([ep_json, facts_json, merge_json])
        _add_utterances(storage)

        # Pre-populate
        storage.upsert_profile(
            Profile(
                id=None,
                topic="interest",
                sub_topic="movie",
                content="Action movies",
                updated_at="2026-03-01 10:00:00",
            )
        )

        writer.process_session(_SESSION_ID, _TIMESTAMP)

        profiles = storage.get_all_profiles()
        assert len(profiles) == 2
        topics = {(p.topic, p.sub_topic): p.content for p in profiles}
        assert topics[("interest", "movie")] == "Sci-fi and action movies"
        assert topics[("interest", "music")] == "Enjoys jazz"

    def test_unknown_topic_dropped(self) -> None:
        ep_json = _episode_response("The user said something.")
        facts_json = _profile_facts_response(
            ("unknown_topic", "sub", "content"),
            ("interest", "movie", "Valid fact"),
        )
        merge_json = _merge_response(
            ("APPEND", 1, "Valid fact"),
        )
        writer, storage, _, _ = _make_writer([ep_json, facts_json, merge_json])
        _add_utterances(storage)

        writer.process_session(_SESSION_ID, _TIMESTAMP)

        profiles = storage.get_all_profiles()
        assert len(profiles) == 1
        assert profiles[0].content == "Valid fact"

    def test_no_episodes_skips_profile(self) -> None:
        """If no episodes extracted, profile extraction should not happen."""
        ep_json = _episode_response()  # empty
        writer, _, _, llm = _make_writer([ep_json])
        _add_utterances(writer._storage)

        writer.process_session(_SESSION_ID, _TIMESTAMP)

        # Only episode extraction call, no profile calls
        assert llm._call_count == 1


# ---------------------------------------------------------------------------
# Window tests
# ---------------------------------------------------------------------------


class TestWindowing:
    def test_short_session_single_window(self) -> None:
        ep_json = _episode_response("Short session episode.")
        facts_json = _profile_facts_response()
        writer, storage, _, llm = _make_writer([ep_json, facts_json])
        _add_utterances(storage)

        writer.process_session(_SESSION_ID, _TIMESTAMP)

        # Episode extraction + profile extraction = 2 calls
        assert llm._call_count == 2

    def test_long_session_windowed(self) -> None:
        """Session exceeding token limit should be split into windows."""
        ep_json_1 = _episode_response("Episode from window 1.")
        ep_json_2 = _episode_response("Episode from window 2.")
        facts_json = _profile_facts_response()
        # max=25, overlap=0 → 4 utterances (10+10+10+6=36) split into 2 windows
        config = _make_config(write_max_input_tokens=25, write_window_overlap_turns=0)
        writer, storage, _, llm = _make_writer([ep_json_1, ep_json_2, facts_json], config=config)
        _add_utterances(storage)

        episodes = writer.process_session(_SESSION_ID, _TIMESTAMP)

        # Should have episodes from both windows
        assert len(episodes) == 2
        # 2 episode extraction calls + 1 profile call = 3
        assert llm._call_count == 3

    def test_deduplication(self) -> None:
        """Near-duplicate episodes across windows should be deduplicated."""
        ep_json_1 = _episode_response(
            "The user watched Interstellar and loved it.",
        )
        ep_json_2 = _episode_response(
            "The user watched Interstellar and really loved it.",  # near-duplicate
            "The user cried at the ending.",  # unique
        )
        facts_json = _profile_facts_response()
        config = _make_config(write_max_input_tokens=25, write_window_overlap_turns=0)
        writer, storage, _, _ = _make_writer([ep_json_1, ep_json_2, facts_json], config=config)
        _add_utterances(storage)

        episodes = writer.process_session(_SESSION_ID, _TIMESTAMP)

        texts = [ep.text for ep in episodes]
        # Near-duplicate dropped, unique one kept
        assert len(texts) == 2
        assert "The user cried at the ending." in texts


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_llm_failure_graceful(self) -> None:
        """LLM failure should not raise, should return empty."""

        class FailingLLM(ILLM):
            def generate(self, messages, tools=None, response_format=None):
                raise RuntimeError("LLM down")

        cfg = _make_config()
        storage = InMemoryMemoryStorage(dimension=_DIM)
        vi = NumpyVectorIndex()
        emb = FakeEmbedder()
        counter = lambda text: len(text.split())  # noqa: E731
        writer = MemoryWriter(storage, vi, emb, FailingLLM(), cfg, counter)
        _add_utterances(storage)

        episodes = writer.process_session(_SESSION_ID, _TIMESTAMP)
        assert episodes == []

    def test_invalid_json_response(self) -> None:
        """Malformed JSON should be handled gracefully."""
        writer, storage, _, _ = _make_writer(["not valid json"])
        _add_utterances(storage)

        episodes = writer.process_session(_SESSION_ID, _TIMESTAMP)
        assert episodes == []

    def test_profile_extraction_failure_episodes_still_stored(self) -> None:
        """Profile failure should not affect already-stored episodes."""
        ep_json = _episode_response("A good episode.")
        # Profile extraction returns bad JSON
        writer, storage, _, _ = _make_writer([ep_json, "bad json"])
        _add_utterances(storage)

        episodes = writer.process_session(_SESSION_ID, _TIMESTAMP)

        assert len(episodes) == 1
        assert storage.get_episode(episodes[0].id) is not None


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


class TestTextSimilarity:
    def test_identical_texts(self) -> None:
        assert _text_similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_different_texts(self) -> None:
        assert _text_similarity("hello world", "foo bar") == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        sim = _text_similarity("the user loves movies", "the user enjoys movies")
        assert 0.4 < sim < 0.9

    def test_empty_text(self) -> None:
        assert _text_similarity("", "hello") == pytest.approx(0.0)
        assert _text_similarity("", "") == pytest.approx(0.0)
