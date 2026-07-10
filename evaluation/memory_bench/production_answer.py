"""Production-style answerer — 벤치 답변자를 실제 레이 프롬프트·컨텍스트로 구성.

프로덕션 조립을 프로덕션 코드로 재현한다: ``DEFAULT_SYSTEM_PROMPT``(Block 1) +
``format_profile_block``(Block 2) + 질문(user) + ``format_memory_block``(Block 4,
[M1] 번호·인용 태그 계약 포함). 히스토리/이전 세션 요약(Block 3)은 벤치에 없으므로
생략 — 나머지는 ContextBuilder.build()와 동일한 순서·role이다.

용도: "같은 기억으로 실제 레이라면 몇 점인가"(제품축). 진단축(슬림 벤치 프롬프트)과
``answer --answer-style production``으로 병행 비교한다. 질문이 1인칭 user 시점인
LongMemEval 전용 — LoCoMo는 레이가 대화 참여자가 아니라 성립하지 않는다.
"""

from __future__ import annotations

from typing import Any

from voice_pipeline.context.formatters import format_memory_block, format_profile_block, parse_citation_tag
from voice_pipeline.llm.prompts import DEFAULT_SYSTEM_PROMPT
from voice_pipeline.memory.types import MemoryReadResult, Profile

PRODUCTION_MAX_TOKENS = 256  # 프로덕션 OpenAILLM 설정과 동일 (wiring.py)


def build_production_messages(
    question: str,
    memory_result: MemoryReadResult | None,
    profiles: list[Profile],
) -> list[dict[str, Any]]:
    """Assemble messages the way production ContextBuilder does (Blocks 1/2/4)."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
    profile_text = format_profile_block(profiles)
    if profile_text:
        messages.append({"role": "developer", "content": profile_text})
    messages.append({"role": "user", "content": question})
    if memory_result is not None:
        memory_text = format_memory_block(memory_result)
        if memory_text:
            messages.append({"role": "developer", "content": memory_text})
    return messages


def parse_production_answer(text: str) -> tuple[str, list[int]]:
    """Strip the ``[MEMORIES: ...]`` citation tag; return (answer, cited indices)."""
    return parse_citation_tag(text.strip())
