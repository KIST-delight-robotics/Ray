"""Harness-shared prompts: 메모리/프로필 렌더링, 기본(ray) judge.

데이터셋별 답변 프롬프트와 데이터셋 전용 judge는 각 ``datasets/<name>.py``에
있다. 여기에는 어떤 데이터셋에도 의존하지 않는 공용 조각만 둔다.
프로덕션 프롬프트(voice_pipeline/memory/prompts.py)와도 분리 — 측정·채점용
프롬프트는 evaluation 패키지에 둔다 (wiring.py 독트린).
"""

from __future__ import annotations

from typing import Any

from voice_pipeline.memory.types import Episode, Profile


def format_memories(episodes: list[Episode]) -> str:
    """Render episodes chronologically as dated bullet notes.

    에피소드 본문에는 날짜가 없으므로(추출 프롬프트가 금지) timestamp를
    ``[YYYY-MM-DD]``로 앞에 붙인다 — temporal 질문은 이 날짜가 유일한 단서다.
    """
    if not episodes:
        return "(no relevant memories found)"
    ordered = sorted(episodes, key=lambda ep: ep.timestamp)
    return "\n".join(f"- [{ep.timestamp[:10]}] {ep.text}" for ep in ordered)


def format_profile(profiles: list[Profile]) -> str:
    """Render profile slots as ``topic/sub_topic: content`` lines."""
    if not profiles:
        return "(no profile facts)"
    return "\n".join(f"- {p.topic}/{p.sub_topic}: {p.content}" for p in profiles)


# ---------------------------------------------------------------------------
# 기본(ray) judge — 데이터셋 무관 gold 대조 채점
# ---------------------------------------------------------------------------

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "name": "answer_judgement",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": ["CORRECT", "WRONG"],
            },
        },
        "required": ["label"],
        "additionalProperties": False,
    },
}

_JUDGE_SYSTEM = """\
You grade answers to questions about a long multi-session conversation.

Given a question, the gold answer, and a model answer, decide whether the model answer is correct.

## Rules
- CORRECT if the model answer contains the gold answer's information. Extra wording or detail \
is fine as long as it does not contradict the gold answer.
- Dates and numbers must match the gold answer; formatting may differ ("8 May 2023" == "May 8, 2023").
- If the gold answer says the question is unanswerable, the model answer is CORRECT only if it \
states the information is not mentioned or unknown. Any substantive answer is WRONG."""

_JUDGE_USER_TEMPLATE = """\
## Question
{question}

## Gold answer
{gold}

## Model answer
{answer}"""

_UNANSWERABLE_GOLD_TEMPLATE = """\
The question is unanswerable from the conversation — the correct behavior is to say the \
information is not mentioned. (A tempting but WRONG answer would be: "{trap}")"""


def build_judge_messages(question: str, gold: str, answer: str) -> list[dict[str, Any]]:
    """Build messages for the judge LLM call."""
    return [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": _JUDGE_USER_TEMPLATE.format(question=question, gold=gold, answer=answer)},
    ]


def gold_display(answer: str, adversarial: bool, adversarial_answer: str) -> str:
    """Gold answer as shown to the judge (adversarial 항목은 무응답이 정답)."""
    if adversarial:
        return _UNANSWERABLE_GOLD_TEMPLATE.format(trap=adversarial_answer or "any specific answer")
    return answer
