"""Answer phase: retrieve memories per question and generate benchmark answers.

질문마다 새 :class:`MemoryRetriever`를 만든다 — retained buffer가 세션(대화)
상태라서 재사용하면 질문 간 검색 결과가 오염된다. ``update_citations()``는
호출하지 않으므로 DB는 읽기 전용으로 유지된다.

"현재 시각"은 질문 시점(``QAItem.question_date``, LongMemEval)이 있으면 그것으로,
없으면 마지막 세션 다음 날(LoCoMo)로 고정한다(``now_fn`` 주입). 과거 대화에
실제 현재 시각을 쓰면 recency decay가 전체 에피소드를 사멸시킨다.

답변 프롬프트는 데이터셋 소유(``datasets/<name>.py``) — 화자 구조가 달라
(2인 대화 vs user–assistant) 공용 템플릿로 묶으면 양쪽 다 나빠진다.
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
    UsageTrackingLLM,
    db_path,
    load_config,
    read_answers,
    session_db_id,
    update_config,
)
from evaluation.memory_bench.datasets import AnswerBuilder, get_answer_builder, get_answer_system, load_dataset
from evaluation.memory_bench.production_answer import (
    PRODUCTION_MAX_TOKENS,
    build_production_messages,
    parse_production_answer,
)
from evaluation.memory_bench.prompts import format_memories, format_profile
from evaluation.memory_bench.types import TIMESTAMP_FORMAT, Conversation, QAItem
from voice_pipeline.core.interfaces import ILLM, IEmbedder
from voice_pipeline.embedding.embedder import create_embedder
from voice_pipeline.llm.llm import OpenAILLM
from voice_pipeline.llm.prompts import DEFAULT_SYSTEM_PROMPT
from voice_pipeline.memory.retriever import MemoryRetriever
from voice_pipeline.memory.storage import _DEFAULT_DIMENSION, SQLiteMemoryStorage
from voice_pipeline.memory.types import MemoryReadResult, Profile
from voice_pipeline.memory.vector_index import NumpyVectorIndex

logger = logging.getLogger("eval.memory_bench")

ANSWER_STYLES = ("bench", "production")


class _Perspective:
    """Open storage + vector index + profiles for one perspective's DB."""

    def __init__(self, run_dir: Path, sample_id: str, perspective: str) -> None:
        self.perspective = perspective
        self.storage = SQLiteMemoryStorage(str(db_path(run_dir, sample_id, perspective)))
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
    answer_style: str = "bench",
    no_memory: bool = False,
) -> None:
    """Answer all benchmark questions using the ingested memory DBs.

    데이터셋 종류와 대상 문항은 run config의 ingest 섹션에서 읽는다.
    이미 ``answers.jsonl``에 있는 (sample_id, qa_index)는 건너뛴다 (resume).

    Args:
        half_life_days: 실험용 recency decay 반감기 오버라이드. 프로세스 전역
            클래스 변수를 바꾸므로 벤치 프로세스 안에서만 사용할 것. 적용값은
            config의 answer 섹션에 기록된다.
        answer_style: ``bench``(데이터셋별 슬림 QA 프롬프트, 진단축) 또는
            ``production``(실제 레이 프롬프트·Block 조립, 제품축 — longmemeval 전용).
        no_memory: 검색·프로필 없이 답변 (오염 검사용 ablation — 기억 없이도
            점수가 나오면 모델이 벤치 정답을 암기했다는 신호).
    """
    if answer_style not in ANSWER_STYLES:
        raise ValueError(f"Unknown answer style: {answer_style!r} (expected one of {ANSWER_STYLES})")

    ingest_config = load_config(run_dir).get("ingest", {})
    dataset = ingest_config.get("dataset", "locomo")
    if answer_style == "production" and dataset != "longmemeval":
        raise ValueError("production 스타일은 longmemeval 전용입니다 (레이는 LoCoMo 대화의 참여자가 아님)")
    if data_path is None:
        data_path = ingest_config.get("data_path")
        if not data_path:
            raise ValueError("data_path not given and not found in config.json — run ingest first")
    if sample_ids is None:
        # 인제스트된 문항만 순회 (샘플링 런에서 전체 데이터셋 순회 방지).
        sample_ids = ingest_config.get("sample_ids")

    if half_life_days is not None:
        MemoryRetriever._RECENCY_HALF_LIFE_DAYS = half_life_days
        logger.info("Recency half-life overridden to %.1f days", half_life_days)

    conversations = load_dataset(dataset, data_path, sample_ids)
    build_messages = get_answer_builder(dataset)
    done = {(r["sample_id"], r["qa_index"]) for r in read_answers(run_dir)}

    embedder = LockedEmbedder(create_embedder(expected_dimension=_DEFAULT_DIMENSION))
    max_tokens = PRODUCTION_MAX_TOKENS if answer_style == "production" else 128
    llm = UsageTrackingLLM(
        OpenAILLM(model=answer_model, temperature=0.0, reasoning_effort=None, max_tokens=max_tokens, tools=[])
    )
    writer = JsonlWriter(run_dir / ANSWERS_FILENAME)

    total = 0
    for conv in conversations:
        # DB가 없는 대화를 열면 SQLiteMemoryStorage가 빈 DB를 새로 만들어
        # "기억 없음" 답변이 조용히 생산된다 — 인제스트된 대화만 진행.
        missing = [p for p in conv.perspectives if not db_path(run_dir, conv.sample_id, p).exists()]
        if missing and not no_memory:
            logger.warning("%s: skipping — no ingested DB for %s (run ingest first)", conv.sample_id, missing)
            continue

        pending = [qa for qa in conv.qa if (conv.sample_id, qa.qa_index) not in done]
        if not pending:
            logger.debug("%s: all %d questions already answered", conv.sample_id, len(conv.qa))
            continue

        perspectives = [] if no_memory else [_Perspective(run_dir, conv.sample_id, p) for p in conv.perspectives]
        try:
            fallback_now = max(s.dt for s in conv.sessions).replace(tzinfo=UTC) + timedelta(days=1)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        _answer_one,
                        conv,
                        qa,
                        perspectives,
                        embedder,
                        llm,
                        build_messages,
                        answer_style,
                        fallback_now,
                        writer,
                    )
                    for qa in pending
                ]
                for future in futures:
                    future.result()
            total += len(pending)
            logger.info("%s: answered %d questions", conv.sample_id, len(pending))
        finally:
            for p in perspectives:
                p.close()

    usage = llm.summary()
    logger.info(
        "Answer LLM usage: %d calls, in=%d out=%d cached=%d",
        usage["calls"],
        usage["input_tokens"],
        usage["output_tokens"],
        usage["cached_tokens"],
    )
    update_config(
        run_dir,
        "answer",
        {
            "answer_model": answer_model,
            "answer_style": answer_style,
            "no_memory": no_memory,
            "usage": usage,
            "recency_half_life_days": MemoryRetriever._RECENCY_HALF_LIFE_DAYS,
            # 적용 시점의 프롬프트 전문 (재현용)
            "answer_system_prompt": (
                DEFAULT_SYSTEM_PROMPT if answer_style == "production" else get_answer_system(dataset)
            ),
            "answered_this_run": total,
            "completed_at": datetime.now(UTC).strftime(TIMESTAMP_FORMAT),
        },
    )
    logger.info("Answer phase done: %d questions answered this run", total)


def _answer_one(
    conv: Conversation,
    qa: QAItem,
    perspectives: list[_Perspective],
    embedder: IEmbedder,
    llm: ILLM,
    build_messages: AnswerBuilder,
    answer_style: str,
    fallback_now: datetime,
    writer: JsonlWriter,
) -> None:
    if qa.question_date:
        now_dt = datetime.strptime(qa.question_date, TIMESTAMP_FORMAT).replace(tzinfo=UTC)
    else:
        now_dt = fallback_now

    retrieved: dict[str, list[dict[str, object]]] = {}
    results: dict[str, tuple[MemoryReadResult, list[Profile]]] = {}
    for p in perspectives:
        retriever = MemoryRetriever(p.storage, p.vector_index, embedder, now_fn=lambda now_dt=now_dt: now_dt)
        result = retriever.retrieve(qa.question, set())
        results[p.perspective] = (result, p.profiles)
        retrieved[p.perspective] = [
            {
                "episode_id": ep.id,
                "session_id": ep.session_id,
                "timestamp": ep.timestamp,
                "score": score,
                "text": ep.text,
            }
            for ep, score in zip(result.episodes, result.scores, strict=True)
        ]

    citations: list[int] = []
    if answer_style == "production":
        result, profiles = results.get(conv.perspectives[0], (None, []))
        messages = build_production_messages(qa.question, result, profiles)
    else:
        blocks = {
            name: (format_memories(result.episodes), format_profile(profiles))
            for name, (result, profiles) in results.items()
        }
        if not blocks:  # no-memory ablation
            blocks = {p: ("(no relevant memories found)", "(no profile facts)") for p in conv.perspectives}
        messages = build_messages(conv, qa.question, qa.question_date, blocks)

    stream = llm.generate(messages, tools=[])
    for _ in stream:
        pass
    if answer_style == "production":
        answer_text, citations = parse_production_answer(stream.text)
    else:
        answer_text = stream.text.strip()

    record = {
        "sample_id": conv.sample_id,
        "qa_index": qa.qa_index,
        "question": qa.question,
        "gold_answer": qa.answer,
        "adversarial": qa.adversarial,
        "adversarial_answer": qa.adversarial_answer,
        "category": qa.category,
        "evidence": qa.evidence,
        "evidence_sessions": [session_db_id(conv.sample_id, i) for i in qa.evidence_sessions],
        "perspectives": conv.perspectives,
        "answer": answer_text,
        "retrieved": retrieved,
    }
    if answer_style == "production":
        record["citations"] = citations
    writer.append(record)
