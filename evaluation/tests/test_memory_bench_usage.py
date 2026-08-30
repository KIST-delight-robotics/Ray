"""Unit tests for UsageTrackingLLM."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from evaluation.memory_bench.common import UsageTrackingLLM
from voice_pipeline.types import ILLM, LLMMetrics, LLMResult, LLMStream, Usage

_DEFAULT_USAGE = Usage(100, 20, 5)


class _FakeLLM(ILLM):
    """Returns a fixed text with fixed usage; counts calls."""

    def __init__(self, text: str = "hello world", usage: Usage | None = _DEFAULT_USAGE) -> None:
        self._text = text
        self._usage = usage

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMStream:
        metrics = LLMMetrics(usage=self._usage, model="fake", latency_ms=1, ttft_ms=1) if self._usage else None

        def result_fn(text: str) -> LLMResult:
            return LLMResult(text=text, metrics=metrics)

        def gen() -> Generator[str, None, None]:
            yield self._text

        return LLMStream(gen(), result_fn=result_fn)


def _consume(stream: LLMStream) -> str:
    for _ in stream:
        pass
    return stream.text


def test_usage_accumulates_across_calls() -> None:
    llm = UsageTrackingLLM(_FakeLLM())
    assert _consume(llm.generate([{"role": "user", "content": "q1"}])) == "hello world"
    _consume(llm.generate([{"role": "user", "content": "q2"}]))

    usage = llm.summary()
    assert usage == {
        "calls": 2,
        "input_tokens": 200,
        "output_tokens": 40,
        "cached_tokens": 10,
        "missing_usage": 0,
    }


def test_result_passthrough() -> None:
    llm = UsageTrackingLLM(_FakeLLM())
    stream = llm.generate([{"role": "user", "content": "q"}])
    _consume(stream)
    assert stream.result.text == "hello world"
    assert stream.result.metrics is not None
    assert stream.result.metrics.usage.input_tokens == 100


def test_missing_usage_counted_separately() -> None:
    llm = UsageTrackingLLM(_FakeLLM(usage=None))
    _consume(llm.generate([{"role": "user", "content": "q"}]))

    usage = llm.summary()
    assert usage["calls"] == 1
    assert usage["missing_usage"] == 1
    assert usage["input_tokens"] == 0
