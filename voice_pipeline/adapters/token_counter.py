"""Tiktoken-based token counter factory for LLM models."""

from __future__ import annotations

import logging

import tiktoken

from voice_pipeline.types import TokenCounter

logger = logging.getLogger("voice_pipeline.llm")

_FALLBACK_ENCODING = "o200k_base"  # 알 수 없는 모델의 fallback 인코딩 (GPT-4o 계열 기본)


def create_token_counter(model: str) -> TokenCounter:
    """Create a token counter for the given model.

    Uses ``tiktoken.encoding_for_model`` to resolve the correct encoding.
    Falls back to ``o200k_base`` (GPT-4o default) for unknown models.

    Args:
        model: OpenAI model name (e.g. ``"gpt-4o"``).

    Returns:
        A callable that counts tokens in a string.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning(
            "Unknown model %r for tiktoken, falling back to %s",
            model,
            _FALLBACK_ENCODING,
        )
        encoding = tiktoken.get_encoding(_FALLBACK_ENCODING)

    return lambda text: len(encoding.encode(text))
