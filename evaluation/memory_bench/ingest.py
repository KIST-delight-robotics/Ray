"""Ingest phase: replay benchmark sessions through the production memory write pipeline.

대화당 ``Conversation.perspectives``의 각 관점(user로 매핑할 화자)마다 별도 DB를
만든다. LoCoMo(user–user)는 듀얼 인제스트 — 추출 프롬프트가 user 중심이라 한
관점만으로는 상대 화자 기억이 빠지는데, 프로덕션 프롬프트를 수정하지 않고 양쪽을
커버하기 위한 방식. LongMemEval(user–assistant)은 user 관점 하나만 인제스트한다.

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
from evaluation.memory_bench.datasets import load_dataset
from evaluation.memory_bench.datasets.longmemeval import sample_per_type as lme_sample_per_type
from evaluation.memory_bench.types import Conversation
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
    dataset: str = "locomo",
    sample_ids: list[str] | None = None,
    per_type: int | None = None,
    workers: int = 4,
    writer_model: str = DEFAULT_WRITER_MODEL,
    episode_prompt_file: str | None = None,
) -> None:
    """Run the ingest phase for all (conversation, perspective) units.

    Args:
        data_path: Path to the dataset JSON.
        run_dir: Run directory; DBs are written under ``<run_dir>/dbs/``.
        dataset: 데이터셋 이름 (``locomo`` | ``longmemeval``).
        sample_ids: 지정 시 해당 대화/문항만 인제스트.
        per_type: 유형별 결정론적 샘플 수 (LongMemEval 전용 — 문항당 히스토리가
            독립이라 인제스트 비용이 문항 수에 비례하므로 서브셋 실행용).
        workers: Concurrent (conversation, perspective) units.
        writer_model: LLM for episode/profile extraction.
        episode_prompt_file: 에피소드 추출 시스템 프롬프트 변형 파일 (실험용).
            ``None``이면 프로덕션 프롬프트. 적용된 전문은 config에 기록된다.
    """
    conversations = load_dataset(dataset, data_path, sample_ids)
    if per_type is not None:
        if dataset != "longmemeval":
            raise ValueError("--sample-per-type은 longmemeval 전용입니다 (locomo는 --conversations 사용)")
        conversations = lme_sample_per_type(conversations, per_type)
    if not conversations:
        raise ValueError(f"No conversations matched {sample_ids!r} in {data_path}")

    episode_prompt = Path(episode_prompt_file).read_text().strip() if episode_prompt_file else None

    (run_dir / DBS_DIRNAME).mkdir(parents=True, exist_ok=True)

    embedder = LockedEmbedder(create_embedder(expected_dimension=_DEFAULT_DIMENSION))
    token_counter = create_token_counter(writer_model)

    units = [(conv, perspective) for conv in conversations for perspective in conv.perspectives]
    logger.info("Ingesting %d conversations (%d units) with %d workers", len(conversations), len(units), workers)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _ingest_unit, conv, perspective, run_dir, embedder, token_counter, writer_model, episode_prompt
            ): (conv.sample_id, perspective)
            for conv, perspective in units
        }
        for future, (sample_id, perspective) in futures.items():
            episode_count = future.result()
            logger.info("Ingested %s [user=%s]: %d episodes", sample_id, perspective, episode_count)

    update_config(
        run_dir,
        "ingest",
        {
            "dataset": dataset,
            "data_path": str(data_path),
            "writer_model": writer_model,
            "embedder": "all-MiniLM-L6-v2 (local)",
            "sample_ids": [c.sample_id for c in conversations],
            "episode_prompt_file": episode_prompt_file,
            "episode_prompt": episode_prompt,  # None = 프로덕션 기본 프롬프트
            "retriever_constants": {n: getattr(MemoryRetriever, n) for n in _RETRIEVER_CONSTANT_NAMES},
            "writer_constants": {n: getattr(MemoryWriter, n) for n in _WRITER_CONSTANT_NAMES},
            "completed_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


def _ingest_unit(
    conv: Conversation,
    perspective: str,
    run_dir: Path,
    embedder: IEmbedder,
    token_counter: TokenCounter,
    writer_model: str,
    episode_prompt: str | None,
) -> int:
    """Ingest one conversation from one perspective. Returns episode count."""
    storage = SQLiteMemoryStorage(str(db_path(run_dir, conv.sample_id, perspective)))
    try:
        vector_index = NumpyVectorIndex()
        ids, vectors = storage.load_all_embeddings()
        if ids:
            vector_index.load(ids, vectors)

        llm = OpenAILLM(model=writer_model, temperature=0.0, reasoning_effort=None, max_tokens=4096, tools=[])
        writer = MemoryWriter(storage, vector_index, embedder, llm, token_counter, episode_system_prompt=episode_prompt)

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
                    role = "user" if turn.speaker == perspective else "assistant"
                    storage.add_utterance(sid, role, turn.text, session.timestamp, token_counter(turn.text))
            episode_count += len(writer.process_session(sid, session.timestamp))
        return episode_count
    finally:
        storage.close()
