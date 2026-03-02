"""Integration tests for OpenAILLM with real OpenAI API.

Requires OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import pytest

from voice_pipeline.core.config import LLMConfig
from voice_pipeline.llm.exceptions import LLMError
from voice_pipeline.llm.llm import OpenAILLM

pytestmark = pytest.mark.requires_api


@pytest.fixture
def llm(openai_api_key: str) -> OpenAILLM:  # noqa: ARG001
    """Create an OpenAILLM with default config."""
    return OpenAILLM(LLMConfig())


class TestStreamingResponse:
    def test_basic_streaming(self, llm: OpenAILLM) -> None:
        messages = [
            {"role": "system", "content": "Reply in one sentence."},
            {"role": "user", "content": "What color is the sky?"},
        ]
        chunks = list(llm.generate(messages))
        text = "".join(chunks)

        assert len(text) > 0

    def test_conversation_context(self, llm: OpenAILLM) -> None:
        messages = [
            {"role": "system", "content": "Reply in one word."},
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "Add 3 to that."},
        ]
        chunks = list(llm.generate(messages))
        text = "".join(chunks)

        assert len(text) > 0


class TestErrorRecovery:
    def test_invalid_model_propagates_error(self, openai_api_key: str) -> None:  # noqa: ARG002
        llm = OpenAILLM(LLMConfig(model="not-a-real-model-xyz"))
        with pytest.raises(LLMError):
            list(llm.generate([{"role": "user", "content": "hi"}]))

    def test_recovery_after_error(self, openai_api_key: str) -> None:  # noqa: ARG002
        bad_llm = OpenAILLM(LLMConfig(model="not-a-real-model-xyz"))
        with pytest.raises(LLMError):
            list(bad_llm.generate([{"role": "user", "content": "hi"}]))

        good_llm = OpenAILLM(LLMConfig())
        chunks = list(good_llm.generate([{"role": "user", "content": "Say hello."}]))
        assert len("".join(chunks)) > 0
