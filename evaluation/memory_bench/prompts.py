"""Benchmark-only prompts: QA answering over retrieved memories, LLM judge.

프로덕션 프롬프트(voice_pipeline/memory/prompts.py)와 분리 — 측정·채점용
프롬프트는 evaluation 패키지에 둔다 (wiring.py 독트린).
"""

from __future__ import annotations

from typing import Any

from voice_pipeline.memory.types import Episode, Profile

ABSTAIN_ANSWER = "Not mentioned in the conversation"

_ANSWER_SYSTEM = """\
You answer questions about past conversations between two people, {speaker_a} and {speaker_b}.

You are given memory notes extracted from those conversations. Notes are grouped per person; \
within each group, "the user" refers to that person. Each note is prefixed with the date it was \
recorded as [YYYY-MM-DD]. You may also see profile facts about each person.

## Rules
- Every fact about {speaker_a} or {speaker_b} — their experiences, events, plans, and \
preferences — must come from the provided memories and profiles. Never invent or guess \
personal facts about them.
- You may use general world knowledge to interpret the question and to reason about \
real-world entities and concepts, but not as a source of personal facts about the speakers.
- Be concise: give the shortest answer that fully answers the question (a few words, \
not a full sentence).
- For questions asking for a date, answer in the form "8 May 2023". For questions about \
durations or ordering, reason from the [YYYY-MM-DD] dates on the notes.
- If the personal facts needed to answer are not in the provided memories, answer exactly: \
"{abstain}"."""

_ANSWER_USER_TEMPLATE = """\
## Memories about {speaker_a} (notes where "the user" = {speaker_a})
{memories_a}

## Profile of {speaker_a}
{profile_a}

## Memories about {speaker_b} (notes where "the user" = {speaker_b})
{memories_b}

## Profile of {speaker_b}
{profile_b}

## Question
{question}"""


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


def build_answer_messages(
    question: str,
    speaker_a: str,
    speaker_b: str,
    memories_a: str,
    profile_a: str,
    memories_b: str,
    profile_b: str,
) -> list[dict[str, Any]]:
    """Build messages for the QA answering LLM call."""
    return [
        {
            "role": "system",
            "content": _ANSWER_SYSTEM.format(speaker_a=speaker_a, speaker_b=speaker_b, abstain=ABSTAIN_ANSWER),
        },
        {
            "role": "user",
            "content": _ANSWER_USER_TEMPLATE.format(
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                memories_a=memories_a,
                profile_a=profile_a,
                memories_b=memories_b,
                profile_b=profile_b,
                question=question,
            ),
        },
    ]


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
