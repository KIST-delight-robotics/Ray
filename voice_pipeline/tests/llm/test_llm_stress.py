"""Stress tests for OpenAILLM with real OpenAI API.

Requires OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import pytest

from voice_pipeline.llm.llm import OpenAILLM

pytestmark = pytest.mark.requires_api


@pytest.fixture
def llm(openai_api_key: str) -> OpenAILLM:  # noqa: ARG001
    """Create an OpenAILLM with default config."""
    return OpenAILLM()


class TestRapidSequentialCalls:
    def test_five_back_to_back_calls(self, llm: OpenAILLM) -> None:
        for i in range(5):
            chunks = list(llm.generate([{"role": "user", "content": f"Count to {i + 1}."}]))
            text = "".join(chunks)
            assert len(text) > 0, f"Call {i + 1} produced empty response"


class TestPartialConsumption:
    def test_barge_in_pattern(self, llm: OpenAILLM) -> None:
        """Consume only the first chunk and close — no exception, no leak."""
        gen = llm.generate([{"role": "user", "content": "Write a long essay about the ocean."}])
        first = next(gen)
        gen.close()

        assert len(first) > 0

    def test_repeated_partial_consumption(self, llm: OpenAILLM) -> None:
        """Multiple partial-consume cycles in a row."""
        for _ in range(3):
            gen = llm.generate([{"role": "user", "content": "Tell me a long story."}])
            next(gen)
            gen.close()


class TestMaxTokensRespected:
    def test_short_max_tokens(self, openai_api_key: str) -> None:  # noqa: ARG002
        llm = OpenAILLM(max_tokens=20)
        chunks = list(
            llm.generate(
                [
                    {"role": "user", "content": "Write a very long detailed essay about space."},
                ]
            )
        )
        text = "".join(chunks)

        assert len(text) > 0
        assert len(text) < 200
