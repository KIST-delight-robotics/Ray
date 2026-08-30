"""Integration tests for OpenAILLM with real OpenAI API.

Requires OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import pytest

from voice_pipeline.adapters.llm_openai import OpenAILLM

pytestmark = pytest.mark.requires_api


@pytest.fixture
def llm(openai_api_key: str) -> OpenAILLM:  # noqa: ARG001
    """Create an OpenAILLM with default config."""
    return OpenAILLM()


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

    def test_result_has_metrics(self, llm: OpenAILLM) -> None:
        """After consuming the stream, .result contains usage metrics."""
        messages = [
            {"role": "system", "content": "Reply in one word."},
            {"role": "user", "content": "Say hello."},
        ]
        stream = llm.generate(messages)
        for _ in stream:
            pass  # consume

        result = stream.result
        assert result.text
        assert result.metrics is not None
        assert result.metrics.usage.input_tokens > 0
        assert result.metrics.usage.output_tokens > 0
        assert result.metrics.model
        assert result.metrics.latency_ms > 0
        assert result.metrics.ttft_ms > 0

    def test_tools_none_uses_config_default(self, llm: OpenAILLM) -> None:
        """tools=None should use config defaults (web_search)."""
        messages = [{"role": "user", "content": "Say hi."}]
        stream = llm.generate(messages, tools=None)
        for _ in stream:
            pass
        assert stream.result.text

    def test_tools_empty_disables(self, llm: OpenAILLM) -> None:
        """tools=[] should explicitly disable tools."""
        messages = [{"role": "user", "content": "Say hi."}]
        stream = llm.generate(messages, tools=[])
        for _ in stream:
            pass
        assert stream.result.text


class TestErrorRecovery:
    def test_invalid_model_propagates_error(self, openai_api_key: str) -> None:  # noqa: ARG002
        llm = OpenAILLM(model="not-a-real-model-xyz")
        with pytest.raises(RuntimeError):
            list(llm.generate([{"role": "user", "content": "hi"}]))

    def test_recovery_after_error(self, openai_api_key: str) -> None:  # noqa: ARG002
        bad_llm = OpenAILLM(model="not-a-real-model-xyz")
        with pytest.raises(RuntimeError):
            list(bad_llm.generate([{"role": "user", "content": "hi"}]))

        good_llm = OpenAILLM()
        chunks = list(good_llm.generate([{"role": "user", "content": "Say hello."}]))
        assert len("".join(chunks)) > 0
