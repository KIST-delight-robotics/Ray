"""Ingest phase: replay LoCoMo sessions through the production memory write pipeline.

듀얼 인제스트 — 대화당 두 관점(각 화자를 user로 매핑)으로 두 번 인제스트해서
관점별 DB를 만든다. 추출 프롬프트가 user 중심이라 한 관점만으로는 상대 화자
기억이 빠지는데, 프로덕션 프롬프트를 수정하지 않고 양쪽을 커버하기 위한 방식.

재실행 시 ``processed_sessions``를 확인해 이미 처리된 세션은 건너뛴다 (자연 resume).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from evaluation.memory_bench.common import (
    DBS_DIRNAME,
    DEFAULT_WRITER_MODEL,
    LockedEmbedder,
    db_path,
    session_db_id,
    update_config,
)
from evaluation.memory_bench.dataset import Conversation, load_locomo
from voice_pipeline.core.interfaces import IEmbedder
from voice_pipeline.embedding.embedder import create_embedder
from voice_pipeline.llm.llm import OpenAILLM
from voice_pipeline.llm.token_counter import TokenCounter, create_token_counter
from voice_pipeline.memory.retriever import MemoryRetriever
from voice_pipeline.memory.storage import _DEFAULT_DIMENSION, SQLiteMemoryStorage
from voice_pipeline.memory.vector_index import NumpyVectorIndex
from voice_pipeline.memory.writer import MemoryWriter

logger = logging.getLogger("eval.memory_bench")

_RETRIEVER_CONSTANT_NAMES = (
    "_MAX_MEMORIES",
    "_MIN_NEW_SLOTS",
    "_RETAINED_TTL",
    "_VECTOR_TOP_K",
    "_BM25_TOP_K",
    "_RRF_K",
    "_RECENCY_HALF_LIFE_DAYS",
    "_SALIENCE_THRESHOLD",
)
_WRITER_CONSTANT_NAMES = (
    "_MIN_UTTERANCES",
    "_WRITE_MAX_INPUT_TOKENS",
    "_WRITE_WINDOW_OVERLAP_RATIO",
    "_WRITE_DEDUP_THRESHOLD",
    "_PROFILE_MAX_CONTENT_TOKENS",
)


def ingest_run(
    data_path: str,
    run_dir: Path,
    sample_ids: list[str] | None = None,
    workers: int = 4,
    writer_model: str = DEFAULT_WRITER_MODEL,
) -> None:
    """Run the ingest phase for all (conversation, perspective) units.

    Args:
        data_path: Path to ``locomo10.json``.
        run_dir: Run directory; DBs are written under ``<run_dir>/dbs/``.
        sample_ids: 지정 시 해당 대화만 인제스트.
        workers: Concurrent (conversation, perspective) units.
        writer_model: LLM for episode/profile extraction.
    """
    conversations = load_locomo(data_path, sample_ids)
    if not conversations:
        raise ValueError(f"No conversations matched {sample_ids!r} in {data_path}")

    (run_dir / DBS_DIRNAME).mkdir(parents=True, exist_ok=True)

    embedder = LockedEmbedder(create_embedder(expected_dimension=_DEFAULT_DIMENSION))
    token_counter = create_token_counter(writer_model)

    units = [(conv, speaker) for conv in conversations for speaker in conv.speakers]
    logger.info("Ingesting %d conversations (%d units) with %d workers", len(conversations), len(units), workers)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_ingest_unit, conv, speaker, run_dir, embedder, token_counter, writer_model): (
                conv.sample_id,
                speaker,
            )
            for conv, speaker in units
        }
        for future, (sample_id, speaker) in futures.items():
            episode_count = future.result()
            logger.info("Ingested %s [user=%s]: %d episodes", sample_id, speaker, episode_count)

    update_config(
        run_dir,
        "ingest",
        {
            "data_path": str(data_path),
            "writer_model": writer_model,
            "embedder": "all-MiniLM-L6-v2 (local)",
            "sample_ids": [c.sample_id for c in conversations],
            "retriever_constants": {n: getattr(MemoryRetriever, n) for n in _RETRIEVER_CONSTANT_NAMES},
            "writer_constants": {n: getattr(MemoryWriter, n) for n in _WRITER_CONSTANT_NAMES},
            "completed_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


def _ingest_unit(
    conv: Conversation,
    user_speaker: str,
    run_dir: Path,
    embedder: IEmbedder,
    token_counter: TokenCounter,
    writer_model: str,
) -> int:
    """Ingest one conversation from one speaker's perspective. Returns episode count."""
    storage = SQLiteMemoryStorage(str(db_path(run_dir, conv.sample_id, user_speaker)))
    try:
        vector_index = NumpyVectorIndex()
        ids, vectors = storage.load_all_embeddings()
        if ids:
            vector_index.load(ids, vectors)

        llm = OpenAILLM(model=writer_model, temperature=0.0, reasoning_effort=None, max_tokens=4096, tools=[])
        writer = MemoryWriter(storage, vector_index, embedder, llm, token_counter)

        session_ids = [session_db_id(conv.sample_id, s.index) for s in conv.sessions]
        processed = storage.get_processed_session_ids(session_ids)

        episode_count = 0
        for session in conv.sessions:
            sid = session_db_id(conv.sample_id, session.index)
            if sid in processed:
                continue
            # 이전 실행이 utterance 저장 후 중단됐을 수 있으므로 중복 삽입 방지.
            if not storage.get_utterances(sid):
                for turn in session.turns:
                    role = "user" if turn.speaker == user_speaker else "assistant"
                    storage.add_utterance(sid, role, turn.text, session.timestamp, token_counter(turn.text))
            episode_count += len(writer.process_session(sid, session.timestamp))
        return episode_count
    finally:
        storage.close()
