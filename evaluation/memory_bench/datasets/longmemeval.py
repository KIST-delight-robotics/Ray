"""LongMemEval dataset: loader + 답변 프롬프트 + 공식 judge.

LoCoMo와의 구조 차이:

- 문항마다 전용 히스토리(user–assistant 세션들)가 붙는다 → 문항 1개 = Conversation 1개.
- 화자 매핑이 필요 없다 (턴 role이 이미 user/assistant) → perspectives = ["user"].
- 세션 날짜(``haystack_dates``)가 시간순 정렬돼 있지 않다 → 정렬 후 인덱스 부여.
- ``question_id``의 ``_abs`` 접미사 = abstention(무응답이 정답) 문항.
- ``question_date``가 있어 "현재"를 질문 시점으로 고정한다.

답변 프롬프트는 단일 사용자 전용. 규칙 문구(개인 사실 한정 등)를 하네스 차원에서
교정할 때는 locomo 쪽 프롬프트와 의미를 맞출 것.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.memory_bench.types import TIMESTAMP_FORMAT, Conversation, QAItem, Session, Turn

logger = logging.getLogger("eval.memory_bench")

QUESTION_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "temporal-reasoning",
    "knowledge-update",
    "multi-session",
)

# "2023/04/10 (Mon) 17:50"
_DATE_FORMAT = "%Y/%m/%d (%a) %H:%M"

USER_PERSPECTIVE = "user"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def parse_lme_datetime(date_time: str) -> datetime:
    """Parse a LongMemEval date string (e.g. ``"2023/04/10 (Mon) 17:50"``)."""
    return datetime.strptime(" ".join(date_time.strip().split()), _DATE_FORMAT)


def load_longmemeval(path: str | Path, sample_ids: list[str] | None = None) -> list[Conversation]:
    """Load and normalize a LongMemEval JSON file.

    Args:
        path: Path to ``longmemeval_s_cleaned.json`` / ``longmemeval_oracle.json`` 등.
        sample_ids: 지정 시 해당 ``question_id``의 문항만 반환.

    Returns:
        문항당 Conversation 1개. 세션은 날짜 오름차순으로 정렬해 1부터 인덱스 부여.
    """
    data = json.loads(Path(path).read_text())
    conversations: list[Conversation] = []
    for item in data:
        question_id = str(item["question_id"])
        if sample_ids is not None and question_id not in sample_ids:
            continue
        conversations.append(_parse_item(item, question_id))
    return conversations


def sample_per_type(conversations: list[Conversation], per_type: int) -> list[Conversation]:
    """유형별 앞에서 ``per_type``개씩 결정론적으로 샘플 (question_id 정렬 기준).

    abstention(_abs) 문항도 유형 안에 섞여 있으므로 별도 층은 두지 않는다.
    """
    by_type: dict[str, list[Conversation]] = defaultdict(list)
    for conv in conversations:
        by_type[conv.qa[0].category].append(conv)
    sampled: list[Conversation] = []
    for qtype in sorted(by_type):
        sampled.extend(sorted(by_type[qtype], key=lambda c: c.sample_id)[:per_type])
    return sampled


def _parse_item(item: dict, question_id: str) -> Conversation:
    conv = Conversation(
        sample_id=question_id,
        speaker_a="user",
        speaker_b="assistant",
        participants_desc="a user and an AI assistant",
        perspectives=[USER_PERSPECTIVE],
    )

    raw_sessions = item["haystack_sessions"]
    raw_dates = item["haystack_dates"]
    raw_ids = item["haystack_session_ids"]
    if not (len(raw_sessions) == len(raw_dates) == len(raw_ids)):
        raise ValueError(f"{question_id}: haystack sessions/dates/ids length mismatch")

    # 날짜 오름차순 정렬 후 1-based 인덱스 부여. 원본 session_id → 인덱스 매핑 유지.
    order = sorted(range(len(raw_sessions)), key=lambda i: parse_lme_datetime(raw_dates[i]))
    id_to_index: dict[str, int] = {}
    for new_index, orig_pos in enumerate(order, start=1):
        turns = [
            Turn(
                dia_id=f"{raw_ids[orig_pos]}:{turn_pos}",
                speaker=str(turn.get("role", "")),
                text=str(turn.get("content", "") or "").strip(),
            )
            for turn_pos, turn in enumerate(raw_sessions[orig_pos], start=1)
        ]
        conv.sessions.append(
            Session(
                index=new_index,
                date_time=str(raw_dates[orig_pos]),
                dt=parse_lme_datetime(raw_dates[orig_pos]),
                turns=[t for t in turns if t.text],
            )
        )
        id_to_index[str(raw_ids[orig_pos])] = new_index

    evidence_ids = [str(sid) for sid in item.get("answer_session_ids", [])]
    evidence_sessions = [id_to_index[sid] for sid in evidence_ids if sid in id_to_index]
    missing = [sid for sid in evidence_ids if sid not in id_to_index]
    if missing:
        logger.warning("%s: %d evidence session(s) not in haystack: %s", question_id, len(missing), missing[:3])

    question_date = str(item.get("question_date", ""))
    conv.qa.append(
        QAItem(
            qa_index=0,
            question=str(item["question"]).strip(),
            answer=str(item.get("answer") or "").strip(),
            category=str(item["question_type"]),
            evidence=evidence_ids,
            evidence_sessions=evidence_sessions,
            adversarial=question_id.endswith("_abs"),
            question_date=parse_lme_datetime(question_date).strftime(TIMESTAMP_FORMAT) if question_date else "",
        )
    )
    return conv


# ---------------------------------------------------------------------------
# Answer prompt (단일 사용자 전용)
# ---------------------------------------------------------------------------

ANSWER_SYSTEM = """\
You answer questions about past chat sessions between a user and an AI assistant, \
using memory notes extracted from those sessions. Each note is prefixed with the date \
it was recorded as [YYYY-MM-DD]. You may also see profile facts about the user.

## Rules
- Ground everything you say about the user in the provided memories and profile — do not \
invent personal facts. General knowledge and reasoning are fine.
- Answer the question directly and concisely.
- If the memories don't contain what's needed to answer, say the conversation doesn't \
mention it."""

_ANSWER_USER_TEMPLATE = """\
## Memories about the user
{memories}

## Profile of the user
{profile}

## Question (asked on {question_date})
{question}"""


def build_answer_messages(
    conv: Conversation,
    question: str,
    question_date: str,
    blocks: dict[str, tuple[str, str]],
) -> list[dict[str, Any]]:
    """Build the LongMemEval QA messages from the single user-perspective block.

    Args:
        conv: 대상 문항의 Conversation (미사용 — 시그니처 통일용).
        question: The benchmark question.
        question_date: 질문 시점 (컨텍스트에 표기 — temporal 질문의 기준).
        blocks: ``{"user": (memories, profile)}``.
    """
    memories, profile = blocks[USER_PERSPECTIVE]
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {
            "role": "user",
            "content": _ANSWER_USER_TEMPLATE.format(
                memories=memories,
                profile=profile,
                question=question,
                question_date=question_date[:10] or "an unknown date",
            ),
        },
    ]


# ---------------------------------------------------------------------------
# 공식 judge (출처: github.com/xiaowu0162/LongMemEval,
# src/evaluation/evaluate_qa.py — get_anscheck_prompt 원문 그대로)
#
# 원 하네스: gpt-4o(2024-08-06), temperature 0, max_tokens 10, 단일 user 메시지,
# 응답에 "yes" 포함 여부로 판정. abstention 문항(question_id 끝 "_abs")은 전용 프롬프트.
# ---------------------------------------------------------------------------

OFFICIAL_JUDGE_MODEL = "gpt-4o"

_ANSCHECK_DEFAULT = (
    "I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
    "response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct "
    "answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If "
    "the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: "
    "{question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\nIs the model response correct? "
    "Answer yes or no only."
)

_ANSCHECK_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
    "response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct "
    "answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If "
    "the response only contains a subset of the information required by the answer, answer no. In addition, "
    "do not penalize off-by-one errors for the number of days. If the question asks for the number of "
    "days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer "
    "is 18), the model's response is still correct. \n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\n"
    "Model Response: {response}\n\nIs the model response correct? Answer yes or no only."
)

_ANSCHECK_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
    "response contains the correct answer. Otherwise, answer no. If the response contains some previous "
    "information along with an updated answer, the response should be considered as correct as long as the "
    "updated answer is the required answer.\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel "
    "Response: {response}\n\nIs the model response correct? Answer yes or no only."
)

_ANSCHECK_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and a response from a model. "
    "Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does "
    "not need to reflect all the points in the rubric. The response is correct as long as it recalls and "
    "utilizes the user's personal information correctly.\n\nQuestion: {question}\n\nRubric: {answer}\n\n"
    "Model Response: {response}\n\nIs the model response correct? Answer yes or no only."
)

_ANSCHECK_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response from a model. Please answer "
    "yes if the model correctly identifies the question as unanswerable. The model could say that the "
    "information is incomplete, or some other information is given but the asked information is not.\n\n"
    "Question: {question}\n\nExplanation: {answer}\n\nModel Response: {response}\n\nDoes the model correctly "
    "identify the question as unanswerable? Answer yes or no only."
)

_ANSCHECK_BY_TYPE = {
    "single-session-user": _ANSCHECK_DEFAULT,
    "single-session-assistant": _ANSCHECK_DEFAULT,
    "multi-session": _ANSCHECK_DEFAULT,
    "temporal-reasoning": _ANSCHECK_TEMPORAL,
    "knowledge-update": _ANSCHECK_KNOWLEDGE_UPDATE,
    "single-session-preference": _ANSCHECK_PREFERENCE,
}


def build_official_judge_messages(
    qa_type: str,
    question: str,
    gold: str,
    answer: str,
    abstention: bool,
) -> list[dict[str, Any]]:
    """LongMemEval 공식 judge 메시지 (원 스크립트와 동일: 단일 user 메시지).

    Raises:
        KeyError: LongMemEval에 없는 유형 (LoCoMo 카테고리 등)에 쓰려 한 경우.
    """
    template = _ANSCHECK_ABSTENTION if abstention else _ANSCHECK_BY_TYPE[qa_type]
    return [{"role": "user", "content": template.format(question=question, answer=gold, response=answer)}]


def parse_official_judge(response_text: str) -> str:
    """원 스크립트의 판정 규칙: 응답에 "yes"가 포함되면 CORRECT."""
    return "CORRECT" if "yes" in response_text.lower() else "WRONG"
