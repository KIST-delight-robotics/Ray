"""Unit tests for the LongMemEval dataset loader and official judge helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.memory_bench.datasets.longmemeval import (
    build_official_judge_messages,
    load_longmemeval,
    parse_official_judge,
    sample_per_type,
)


def _item(question_id: str, question_type: str, dates: list[str], **overrides: object) -> dict:
    sessions = [[{"role": "user", "content": f"session {i} text", "has_answer": i == 0}] for i in range(len(dates))]
    item = {
        "question_id": question_id,
        "question_type": question_type,
        "question": "What car did I buy?",
        "answer": "A red hatchback",
        "question_date": "2023/04/10 (Mon) 23:07",
        "haystack_session_ids": [f"{question_id}_{i}" for i in range(len(dates))],
        "haystack_dates": dates,
        "haystack_sessions": sessions,
        "answer_session_ids": [f"{question_id}_0"],
    }
    item.update(overrides)
    return item


@pytest.fixture()
def lme_path(tmp_path: Path) -> Path:
    data = [
        # 날짜가 뒤섞인 문항 — 정렬 검증용 (원본 파일의 실제 특성)
        _item("q1", "temporal-reasoning", ["2023/04/10 (Mon) 17:50", "2023/04/09 (Sun) 14:47"]),
        _item("q2_abs", "knowledge-update", ["2023/03/01 (Wed) 10:00"], answer="No such purchase was discussed"),
        _item("q3", "temporal-reasoning", ["2023/02/01 (Wed) 10:00"]),
    ]
    path = tmp_path / "lme.json"
    path.write_text(json.dumps(data))
    return path


def test_single_user_perspective(lme_path: Path) -> None:
    conv = load_longmemeval(lme_path)[0]
    assert conv.perspectives == ["user"]
    assert conv.participants_desc == "a user and an AI assistant"
    assert len(conv.qa) == 1


def test_sessions_sorted_chronologically(lme_path: Path) -> None:
    conv = load_longmemeval(lme_path)[0]
    assert [s.index for s in conv.sessions] == [1, 2]
    assert conv.sessions[0].timestamp == "2023-04-09 14:47:00"  # 원본 2번째가 시간순 1번째로
    assert conv.sessions[1].timestamp == "2023-04-10 17:50:00"


def test_evidence_mapped_to_sorted_indices(lme_path: Path) -> None:
    qa = load_longmemeval(lme_path)[0].qa[0]
    # evidence는 원본 첫 세션(q1_0, 04/10) — 정렬 후 인덱스 2
    assert qa.evidence_sessions == [2]
    assert qa.question_date == "2023-04-10 23:07:00"


def test_abs_suffix_marks_abstention(lme_path: Path) -> None:
    conv = load_longmemeval(lme_path, sample_ids=["q2_abs"])[0]
    assert conv.qa[0].adversarial
    assert conv.qa[0].category == "knowledge-update"


def test_sample_per_type_is_deterministic(lme_path: Path) -> None:
    conversations = load_longmemeval(lme_path)
    sampled = sample_per_type(conversations, per_type=1)
    # 유형별 question_id 정렬 첫 항목: knowledge-update → q2_abs, temporal → q1
    assert [c.sample_id for c in sampled] == ["q2_abs", "q1"]


def test_official_judge_prompt_selection() -> None:
    normal = build_official_judge_messages("temporal-reasoning", "Q", "gold", "resp", abstention=False)
    assert "off-by-one errors" in normal[0]["content"]

    abstain = build_official_judge_messages("temporal-reasoning", "Q", "gold", "resp", abstention=True)
    assert "unanswerable" in abstain[0]["content"]

    with pytest.raises(KeyError):
        build_official_judge_messages("multi-hop", "Q", "gold", "resp", abstention=False)  # LoCoMo 유형


def test_official_judge_parse() -> None:
    assert parse_official_judge("Yes.") == "CORRECT"
    assert parse_official_judge("yes") == "CORRECT"
    assert parse_official_judge("No, the answer is wrong.") == "WRONG"
