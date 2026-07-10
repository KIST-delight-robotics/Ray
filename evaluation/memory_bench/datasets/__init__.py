"""Dataset loaders — 각 벤치마크를 중립 포맷(:mod:`..types`)으로 변환한다."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from evaluation.memory_bench.datasets import locomo, longmemeval
from evaluation.memory_bench.types import Conversation

DATASET_NAMES = ("locomo", "longmemeval")

# 답변 프롬프트 빌더: (conv, question, question_date, blocks) -> messages
AnswerBuilder = Callable[[Conversation, str, str, dict[str, tuple[str, str]]], list[dict[str, Any]]]


def load_dataset(dataset: str, path: str | Path, sample_ids: list[str] | None = None) -> list[Conversation]:
    """Load a benchmark dataset by name into the neutral format."""
    if dataset == "locomo":
        return locomo.load_locomo(path, sample_ids)
    if dataset == "longmemeval":
        return longmemeval.load_longmemeval(path, sample_ids)
    raise ValueError(f"Unknown dataset: {dataset!r} (expected one of {DATASET_NAMES})")


def get_answer_builder(dataset: str) -> AnswerBuilder:
    """데이터셋 전용 답변 프롬프트 빌더를 반환한다."""
    builders: dict[str, AnswerBuilder] = {
        "locomo": locomo.build_answer_messages,
        "longmemeval": longmemeval.build_answer_messages,
    }
    if dataset not in builders:
        raise ValueError(f"Unknown dataset: {dataset!r} (expected one of {DATASET_NAMES})")
    return builders[dataset]


def get_answer_system(dataset: str) -> str:
    """데이터셋 전용 답변 시스템 프롬프트 템플릿 (config 기록용)."""
    systems = {"locomo": locomo.ANSWER_SYSTEM, "longmemeval": longmemeval.ANSWER_SYSTEM}
    if dataset not in systems:
        raise ValueError(f"Unknown dataset: {dataset!r} (expected one of {DATASET_NAMES})")
    return systems[dataset]
