"""Score phase: LLM judge + token-F1 + failure attribution.

- 헤드라인 지표: adversarial(카테고리 5) 제외 judge 정확도 (Mem0 평가 관례).
- adversarial은 별도 리포트 — abstention("Not mentioned...")이 정답.
- 오답은 3단계로 귀속: extraction(에피소드가 아예 추출 안 됨) →
  retrieval(추출됐지만 검색 top-N이 evidence 세션을 못 덮음) → generation.
"""

from __future__ import annotations

import json
import logging
import re
import string
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluation.memory_bench.common import (
    DEFAULT_JUDGE_MODEL,
    SCORES_FILENAME,
    JsonlWriter,
    db_path,
    read_answers,
    update_config,
)
from evaluation.memory_bench.dataset import CATEGORY_NAMES
from evaluation.memory_bench.prompts import JUDGE_SCHEMA, build_judge_messages, gold_display
from voice_pipeline.llm.llm import OpenAILLM
from voice_pipeline.memory.storage import SQLiteMemoryStorage

logger = logging.getLogger("eval.memory_bench")

JUDGEMENTS_FILENAME = "judgements.jsonl"

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


# ---------------------------------------------------------------------------
# Token F1 (SQuAD-style)
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation/articles, collapse whitespace."""
    text = text.lower().translate(_PUNCT_TABLE)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def token_f1(prediction: str, gold: str) -> float:
    """SQuAD-style token overlap F1 between prediction and gold answer."""
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Failure attribution
# ---------------------------------------------------------------------------


def attribute_failure(
    evidence_sessions: list[str],
    sessions_with_episodes: set[str],
    retrieved_sessions: set[str],
) -> str:
    """Attribute a wrong answer to the first failing pipeline stage.

    Args:
        evidence_sessions: 정답 근거가 있는 세션 ID들.
        sessions_with_episodes: 두 관점 DB를 합쳐 에피소드가 1건 이상 추출된 세션 ID들.
        retrieved_sessions: 이 질문에서 검색된 에피소드들의 세션 ID들 (두 관점 합집합).

    Returns:
        ``"extraction"`` | ``"retrieval"`` | ``"generation"``.
        evidence 정보가 없으면 판단 불가라 ``"generation"``으로 둔다.
    """
    evidence = set(evidence_sessions)
    if evidence and not (evidence & sessions_with_episodes):
        return "extraction"
    if evidence and not (evidence & retrieved_sessions):
        return "retrieval"
    return "generation"


def _sessions_with_episodes(run_dir: Path, sample_id: str, speakers: list[str], session_ids: set[str]) -> set[str]:
    """Sessions (of one conversation) that produced at least one episode in either perspective DB."""
    non_empty: set[str] = set()
    remaining = list(session_ids)
    for speaker in speakers:
        path = db_path(run_dir, sample_id, speaker)
        if not path.exists():
            logger.warning("DB missing for %s [user=%s]: %s", sample_id, speaker, path)
            continue
        storage = SQLiteMemoryStorage(str(path))
        try:
            by_session = storage.get_episodes_by_session_ids(remaining)
            non_empty.update(sid for sid, episodes in by_session.items() if episodes)
        finally:
            storage.close()
    return non_empty


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


def _judge_one(llm: OpenAILLM, record: dict[str, Any]) -> str:
    """Return ``"CORRECT"``/``"WRONG"``, or ``"ERROR"`` on judge failure."""
    gold = gold_display(record["gold_answer"], record["category"] == 5, record["adversarial_answer"])
    messages = build_judge_messages(record["question"], gold, record["answer"])
    try:
        stream = llm.generate(messages, tools=[], response_format=JUDGE_SCHEMA)
        for _ in stream:
            pass
        label = json.loads(stream.text).get("label", "")
        return label if label in ("CORRECT", "WRONG") else "ERROR"
    except Exception:
        logger.warning("Judge call failed for %s#%d", record["sample_id"], record["qa_index"], exc_info=True)
        return "ERROR"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_run(run_dir: Path, workers: int = 8, judge_model: str = DEFAULT_JUDGE_MODEL) -> dict[str, Any]:
    """Judge all answers, attribute failures, and write ``scores.json``.

    기존 ``judgements.jsonl``에 있는 항목은 재사용한다 (resume / 재집계).
    """
    records = read_answers(run_dir)
    if not records:
        raise ValueError(f"No answers found in {run_dir} — run the answer phase first")

    judgements_path = run_dir / JUDGEMENTS_FILENAME
    existing: dict[tuple[str, int], str] = {}
    if judgements_path.exists():
        with judgements_path.open() as f:
            for line in f:
                if line.strip():
                    j = json.loads(line)
                    if j["label"] != "ERROR":
                        existing[(j["sample_id"], j["qa_index"])] = j["label"]

    pending = [r for r in records if (r["sample_id"], r["qa_index"]) not in existing]
    if pending:
        logger.info("Judging %d answers (%d cached) with %s", len(pending), len(existing), judge_model)
        llm = OpenAILLM(model=judge_model, temperature=0.0, reasoning_effort=None, max_tokens=128, tools=[])
        writer = JsonlWriter(judgements_path)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_judge_one, llm, r): r for r in pending}
            for future, record in futures.items():
                label = future.result()
                writer.append({"sample_id": record["sample_id"], "qa_index": record["qa_index"], "label": label})
                if label != "ERROR":
                    existing[(record["sample_id"], record["qa_index"])] = label

    # 대화별로 evidence 세션의 에피소드 존재 여부를 한 번에 조회.
    evidence_by_sample: dict[str, set[str]] = defaultdict(set)
    speakers_by_sample: dict[str, list[str]] = {}
    for r in records:
        evidence_by_sample[r["sample_id"]].update(r["evidence_sessions"])
        speakers_by_sample[r["sample_id"]] = r["speakers"]
    extracted_by_sample = {
        sample_id: _sessions_with_episodes(run_dir, sample_id, speakers_by_sample[sample_id], sids)
        for sample_id, sids in evidence_by_sample.items()
    }

    scores = _aggregate(records, existing, extracted_by_sample)
    (run_dir / SCORES_FILENAME).write_text(json.dumps(scores, indent=2, ensure_ascii=False) + "\n")
    update_config(
        run_dir,
        "score",
        {"judge_model": judge_model, "completed_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")},
    )
    return scores


def _aggregate(
    records: list[dict[str, Any]],
    labels: dict[tuple[str, int], str],
    extracted_by_sample: dict[str, set[str]],
) -> dict[str, Any]:
    per_category: dict[int, dict[str, float]] = defaultdict(lambda: {"n": 0, "correct": 0, "f1_sum": 0.0})
    attribution: dict[str, Counter] = defaultdict(Counter)
    per_conversation: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    judged = 0

    for r in records:
        label = labels.get((r["sample_id"], r["qa_index"]))
        if label is None:
            continue
        judged += 1
        category = r["category"]
        correct = label == "CORRECT"
        stats = per_category[category]
        stats["n"] += 1
        stats["correct"] += int(correct)
        if category != 5:
            stats["f1_sum"] += token_f1(r["answer"], r["gold_answer"])
        conv_stats = per_conversation[r["sample_id"]]
        conv_stats["n"] += 1
        conv_stats["correct"] += int(correct)

        if not correct and category != 5:
            retrieved_sessions = {str(ep["session_id"]) for episodes in r["retrieved"].values() for ep in episodes}
            stage = attribute_failure(
                r["evidence_sessions"], extracted_by_sample.get(r["sample_id"], set()), retrieved_sessions
            )
            attribution[CATEGORY_NAMES.get(category, str(category))][stage] += 1

    headline_n = sum(s["n"] for c, s in per_category.items() if c != 5)
    headline_correct = sum(s["correct"] for c, s in per_category.items() if c != 5)

    return {
        "judged": judged,
        "total_answers": len(records),
        "headline_accuracy": round(headline_correct / headline_n, 4) if headline_n else None,
        "headline_n": headline_n,
        "per_category": {
            CATEGORY_NAMES.get(c, str(c)): {
                "n": s["n"],
                "judge_accuracy": round(s["correct"] / s["n"], 4) if s["n"] else None,
                "mean_f1": round(s["f1_sum"] / s["n"], 4) if (s["n"] and c != 5) else None,
            }
            for c, s in sorted(per_category.items())
        },
        "failure_attribution": {cat: dict(counter) for cat, counter in sorted(attribution.items())},
        "per_conversation": {
            sample_id: {
                "n": s["n"],
                "judge_accuracy": round(s["correct"] / s["n"], 4) if s["n"] else None,
            }
            for sample_id, s in sorted(per_conversation.items())
        },
    }


def format_summary(scores: dict[str, Any]) -> str:
    """Human-readable summary of a scores dict."""
    lines = [
        f"Judged {scores['judged']}/{scores['total_answers']} answers",
        f"Headline accuracy (excl. adversarial): {scores['headline_accuracy']} (n={scores['headline_n']})",
        "",
        f"{'category':<14} {'n':>5} {'judge_acc':>10} {'mean_f1':>8}",
    ]
    for name, s in scores["per_category"].items():
        f1 = f"{s['mean_f1']:.4f}" if s["mean_f1"] is not None else "-"
        lines.append(f"{name:<14} {s['n']:>5} {s['judge_accuracy']:>10.4f} {f1:>8}")
    if scores["failure_attribution"]:
        lines.append("")
        lines.append("Failure attribution (wrong answers, excl. adversarial):")
        for cat, counter in scores["failure_attribution"].items():
            parts = ", ".join(f"{stage}={n}" for stage, n in sorted(counter.items()))
            lines.append(f"  {cat}: {parts}")
    return "\n".join(lines)
