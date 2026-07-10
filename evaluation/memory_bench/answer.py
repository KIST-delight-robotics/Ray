"""Answer phase: retrieve memories per question and generate benchmark answers.

질문마다 새 :class:`MemoryRetriever`를 만든다 — retained buffer가 세션(대화)
상태라서 재사용하면 질문 간 검색 결과가 오염된다. ``update_citations()``는
호출하지 않으므로 DB는 읽기 전용으로 유지된다.

"현재 시각"은 마지막 세션 다음 날로 고정한다(``now_fn`` 주입). LoCoMo 세션은
수개월에 걸치므로 실제 현재 시각을 쓰면 recency decay가 전체 에피소드를
사멸시킨다.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

from evaluation.memory_bench.common import (
    ANSWERS_FILENAME,
    DEFAULT_ANSWER_MODEL,
    JsonlWriter,
    LockedEmbedder,
    db_path,
    load_config,
    read_answers,
    session_db_id,
    update_config,
)
from evaluation.memory_bench.dataset import Conversation, QAItem, load_locomo
from evaluation.memory_bench.prompts import (
    _ANSWER_SYSTEM,
    build_answer_messages,
    format_memories,
    format_profile,
)
from voice_pipeline.core.interfaces import IEmbedder
from voice_pipeline.embedding.embedder import create_embedder
from voice_pipeline.llm.llm import OpenAILLM
from voice_pipeline.memory.retriever import MemoryRetriever
from voice_pipeline.memory.storage import _DEFAULT_DIMENSION, SQLiteMemoryStorage
from voice_pipeline.memory.vector_index import NumpyVectorIndex

logger = logging.getLogger("eval.memory_bench")


class _Perspective:
    """Open storage + vector index + profiles for one speaker's DB."""

    def __init__(self, run_dir: Path, sample_id: str, speaker: str) -> None:
        self.speaker = speaker
        self.storage = SQLiteMemoryStorage(str(db_path(run_dir, sample_id, speaker)))
        self.vector_index = NumpyVectorIndex()
        ids, vectors = self.storage.load_all_embeddings()
        if ids:
            self.vector_index.load(ids, vectors)
        self.profiles = self.storage.get_all_profiles()

    def close(self) -> None:
        self.storage.close()


def answer_run(
    run_dir: Path,
    data_path: str | None = None,
    sample_ids: list[str] | None = None,
    workers: int = 8,
    answer_model: str = DEFAULT_ANSWER_MODEL,
    half_life_days: float | None = None,
) -> None:
    """Answer all benchmark questions using the ingested memory DBs.

    이미 ``answers.jsonl``에 있는 (sample_id, qa_index)는 건너뛴다 (resume).

    Args:
        half_life_days: 실험용 recency decay 반감기 오버라이드. 프로세스 전역
            클래스 변수를 바꾸므로 벤치 프로세스 안에서만 사용할 것. 적용값은
            config의 answer 섹션에 기록된다.
    """
    config = load_config(run_dir)
    if data_path is None:
        data_path = config.get("ingest", {}).get("data_path")
        if not data_path:
            raise ValueError("data_path not given and not found in config.json — run ingest first")

    if half_life_days is not None:
        MemoryRetriever._RECENCY_HALF_LIFE_DAYS = half_life_days
        logger.info("Recency half-life overridden to %.1f days", half_life_days)

    conversations = load_locomo(data_path, sample_ids)
    done = {(r["sample_id"], r["qa_index"]) for r in read_answers(run_dir)}

    embedder = LockedEmbedder(create_embedder(expected_dimension=_DEFAULT_DIMENSION))
    llm = OpenAILLM(model=answer_model, temperature=0.0, reasoning_effort=None, max_tokens=128, tools=[])
    writer = JsonlWriter(run_dir / ANSWERS_FILENAME)

    total = 0
    for conv in conversations:
        # DB가 없는 대화를 열면 SQLiteMemoryStorage가 빈 DB를 새로 만들어
        # "기억 없음" 답변이 조용히 생산된다 — 인제스트된 대화만 진행.
        missing = [s for s in conv.speakers if not db_path(run_dir, conv.sample_id, s).exists()]
        if missing:
            logger.warning("%s: skipping — no ingested DB for %s (run ingest first)", conv.sample_id, missing)
            continue

        pending = [qa for qa in conv.qa if (conv.sample_id, qa.qa_index) not in done]
        if not pending:
            logger.info("%s: all %d questions already answered", conv.sample_id, len(conv.qa))
            continue

        perspectives = [_Perspective(run_dir, conv.sample_id, s) for s in conv.speakers]
        try:
            # 마지막 세션 다음 날을 "현재"로 고정.
            now_dt = max(s.dt for s in conv.sessions).replace(tzinfo=UTC) + timedelta(days=1)
            logger.info("%s: answering %d questions (now=%s)", conv.sample_id, len(pending), now_dt.date())
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(_answer_one, conv, qa, perspectives, embedder, llm, now_dt, writer) for qa in pending
                ]
                for future in futures:
                    future.result()
            total += len(pending)
        finally:
            for p in perspectives:
                p.close()

    update_config(
        run_dir,
        "answer",
        {
            "answer_model": answer_model,
            "recency_half_life_days": MemoryRetriever._RECENCY_HALF_LIFE_DAYS,
            "answer_system_prompt": _ANSWER_SYSTEM,  # 적용 시점의 템플릿 전문 (재현용)
            "answered_this_run": total,
            "completed_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    logger.info("Answer phase done: %d questions answered this run", total)


def _answer_one(
    conv: Conversation,
    qa: QAItem,
    perspectives: list[_Perspective],
    embedder: IEmbedder,
    llm: OpenAILLM,
    now_dt: datetime,
    writer: JsonlWriter,
) -> None:
    retrieved: dict[str, list[dict[str, object]]] = {}
    blocks: dict[str, tuple[str, str]] = {}
    for p in perspectives:
        retriever = MemoryRetriever(p.storage, p.vector_index, embedder, now_fn=lambda now_dt=now_dt: now_dt)
        result = retriever.retrieve(qa.question, set())
        retrieved[p.speaker] = [
            {
                "episode_id": ep.id,
                "session_id": ep.session_id,
                "timestamp": ep.timestamp,
                "score": score,
                "text": ep.text,
            }
            for ep, score in zip(result.episodes, result.scores, strict=True)
        ]
        blocks[p.speaker] = (format_memories(result.episodes), format_profile(p.profiles))

    speaker_a, speaker_b = conv.speakers
    messages = build_answer_messages(
        qa.question,
        speaker_a,
        speaker_b,
        memories_a=blocks[speaker_a][0],
        profile_a=blocks[speaker_a][1],
        memories_b=blocks[speaker_b][0],
        profile_b=blocks[speaker_b][1],
    )
    stream = llm.generate(messages, tools=[])
    for _ in stream:
        pass
    answer_text = stream.text.strip()

    writer.append(
        {
            "sample_id": conv.sample_id,
            "qa_index": qa.qa_index,
            "question": qa.question,
            "gold_answer": qa.answer,
            "adversarial_answer": qa.adversarial_answer,
            "category": qa.category,
            "evidence": qa.evidence,
            "evidence_sessions": [session_db_id(conv.sample_id, i) for i in qa.evidence_session_indices],
            "speakers": list(conv.speakers),
            "answer": answer_text,
            "retrieved": retrieved,
        }
    )
