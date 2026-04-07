"""Shared fixtures and test data for memory integration tests."""

from __future__ import annotations

import os

import pytest

from voice_pipeline.core.config import LLMConfig, MemoryConfig
from voice_pipeline.embedding.embedder import SentenceTransformerEmbedder
from voice_pipeline.llm.llm import OpenAILLM
from voice_pipeline.llm.token_counter import create_token_counter
from voice_pipeline.memory.storage import SQLiteMemoryStorage
from voice_pipeline.memory.types import Episode
from voice_pipeline.memory.vector_index import NumpyVectorIndex

# ---------------------------------------------------------------------------
# Korean conversation data
# ---------------------------------------------------------------------------

CONVERSATION_MOVIE: list[tuple[str, str, str, int]] = [
    ("user", "어제 인터스텔라를 다시 봤는데 정말 감동적이었어.", "2026-04-01 10:00:00", 15),
    (
        "assistant",
        "인터스텔라 정말 좋은 영화죠! 어떤 장면이 가장 인상 깊었나요?",
        "2026-04-01 10:00:05",
        15,
    ),
    (
        "user",
        "우주 장면이 너무 아름다웠고, 마지막에 눈물이 났어. 놀란 감독 영화를 다 좋아해.",
        "2026-04-01 10:00:10",
        16,
    ),
    (
        "assistant",
        "놀란 감독의 영화는 정말 명작이 많죠. 다크나이트도 좋아하시나요?",
        "2026-04-01 10:00:15",
        14,
    ),
    (
        "user",
        "응, 다크나이트도 좋아하지. 히스 레저의 조커 연기가 최고였어.",
        "2026-04-01 10:00:20",
        14,
    ),
    ("assistant", "히스 레저의 조커는 정말 전설적이죠!", "2026-04-01 10:00:25", 10),
]

CONVERSATION_PERSONAL: list[tuple[str, str, str, int]] = [
    ("user", "요즘 재즈 음악에 빠졌어. 특히 빌 에반스가 좋아.", "2026-04-01 11:00:00", 14),
    (
        "assistant",
        "빌 에반스는 정말 훌륭한 재즈 피아니스트죠! 어떤 앨범을 들으셨어요?",
        "2026-04-01 11:00:05",
        14,
    ),
    (
        "user",
        "Waltz for Debby를 매일 듣고 있어. 나는 서울에 살고 있고, 프로그래머로 일하고 있어.",
        "2026-04-01 11:00:10",
        18,
    ),
    (
        "assistant",
        "서울에서 프로그래머로 일하시는군요! 재즈를 들으며 코딩하면 집중이 잘 되겠네요.",
        "2026-04-01 11:00:15",
        16,
    ),
    ("user", "맞아, 카페에서 재즈 들으면서 코딩하는 걸 좋아해.", "2026-04-01 11:00:20", 12),
    ("assistant", "그런 분위기 정말 좋죠!", "2026-04-01 11:00:25", 6),
]

CONVERSATION_COOKING: list[tuple[str, str, str, int]] = [
    ("user", "오늘 파스타를 직접 만들었는데 맛있게 됐어.", "2026-04-01 12:00:00", 12),
    (
        "assistant",
        "직접 만든 파스타, 대단하시네요! 어떤 소스를 사용하셨어요?",
        "2026-04-01 12:00:05",
        14,
    ),
    ("user", "크림 소스에 버섯을 넣었어. 이탈리안 요리를 배우고 싶어.", "2026-04-01 12:00:10", 14),
    (
        "assistant",
        "크림 버섯 파스타 정말 맛있겠네요! 이탈리안 요리 배우시면 더 다양한 요리를 할 수 있겠어요.",  # noqa: E501
        "2026-04-01 12:00:15",
        18,
    ),
    ("user", "응, 특히 리소토랑 라자냐를 만들어보고 싶어.", "2026-04-01 12:00:20", 10),
    ("assistant", "둘 다 만들면 정말 뿌듯하실 거예요!", "2026-04-01 12:00:25", 8),
]

CONVERSATION_TRIVIAL: list[tuple[str, str, str, int]] = [
    ("user", "안녕", "2026-04-01 13:00:00", 2),
    ("assistant", "안녕하세요!", "2026-04-01 13:00:05", 3),
]

# ---------------------------------------------------------------------------
# Session-scoped fixtures (expensive resources, loaded once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


@pytest.fixture(scope="session")
def shared_embedder() -> SentenceTransformerEmbedder:
    """Load embedding model once per test session."""
    return SentenceTransformerEmbedder("all-MiniLM-L6-v2", expected_dimension=384)


@pytest.fixture(scope="session")
def write_llm(openai_api_key: str) -> OpenAILLM:
    """Session-scoped LLM for writer integration tests."""
    cfg = LLMConfig(model="gpt-4o-mini", temperature=0.0, max_tokens=4096, tools=[])
    return OpenAILLM(cfg)


@pytest.fixture(scope="session")
def token_counter():
    return create_token_counter("gpt-4o-mini")


# ---------------------------------------------------------------------------
# Function-scoped fixtures (fresh per test)
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_config(tmp_path) -> MemoryConfig:
    return MemoryConfig(db_path=str(tmp_path / "test_memory.db"), embedding_dimension=384)


@pytest.fixture
def memory_db(memory_config: MemoryConfig) -> SQLiteMemoryStorage:
    storage = SQLiteMemoryStorage(memory_config)
    yield storage
    storage.close()


@pytest.fixture
def vector_index() -> NumpyVectorIndex:
    return NumpyVectorIndex()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def populate_utterances(
    storage: SQLiteMemoryStorage,
    session_id: str,
    conversation: list[tuple[str, str, str, int]],
) -> None:
    """Add a conversation's utterances to storage."""
    for role, text, ts, tc in conversation:
        storage.add_utterance(session_id, role, text, ts, tc)


def store_episode_with_embedding(
    storage: SQLiteMemoryStorage,
    index: NumpyVectorIndex,
    embedder: SentenceTransformerEmbedder,
    episode: Episode,
) -> Episode:
    """Persist an episode with its embedding in storage and vector index.

    Returns the episode with id and embedding set.
    """
    eid = storage.add_episode(episode)
    assert eid is not None
    episode.id = eid

    emb = embedder.embed(episode.text)
    episode.embedding = emb
    storage.update_episode_embedding(eid, emb)
    index.add(eid, emb)

    return episode


def make_episode(
    text: str,
    session_id: str = "s-old",
    timestamp: str = "2026-03-15 14:00:00",
    importance: float = 1.0,
) -> Episode:
    """Create an Episode with sensible defaults (id=None, no embedding)."""
    return Episode(
        id=None,
        text=text,
        timestamp=timestamp,
        session_id=session_id,
        importance=importance,
        last_cited_at=timestamp,
        citation_count=0,
        embedding=None,
    )
