"""Unit tests for OpenAILLM (mocked OpenAI client)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import openai
import pytest

from voice_pipeline.core.config import LLMConfig
from voice_pipeline.llm.exceptions import LLMError
from voice_pipeline.llm.llm import OpenAILLM
from voice_pipeline.tests.llm.conftest import (
    FakeStreamEvent,
    create_mock_client,
    make_stream_events,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_llm(mock_client: MagicMock, config: LLMConfig | None = None) -> OpenAILLM:
    """Build an OpenAILLM with a mock client injected."""
    cfg = config or LLMConfig()
    with patch("voice_pipeline.llm.llm.openai.OpenAI", return_value=mock_client):
        return OpenAILLM(cfg)


def _system_msg(text: str) -> dict[str, str]:
    return {"role": "system", "content": text}


def _user_msg(text: str) -> dict[str, str]:
    return {"role": "user", "content": text}


def _assistant_msg(text: str) -> dict[str, str]:
    return {"role": "assistant", "content": text}


# ---------------------------------------------------------------------------
# TestGenerate
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_streaming_yields_text_chunks(self, llm_config: LLMConfig) -> None:
        chunks = ["Hello", ", ", "world!"]
        client = create_mock_client(make_stream_events(chunks))
        llm = _build_llm(client, llm_config)

        result = list(llm.generate([_user_msg("hi")]))

        assert result == chunks

    def test_empty_response(self, llm_config: LLMConfig) -> None:
        client = create_mock_client([])
        llm = _build_llm(client, llm_config)

        result = list(llm.generate([_user_msg("hi")]))

        assert result == []

    def test_non_text_events_ignored(self, llm_config: LLMConfig) -> None:
        events = [
            FakeStreamEvent("response.created"),
            FakeStreamEvent("response.output_text.delta", "hi"),
            FakeStreamEvent("response.completed"),
        ]
        client = create_mock_client(events)
        llm = _build_llm(client, llm_config)

        result = list(llm.generate([_user_msg("hi")]))

        assert result == ["hi"]


# ---------------------------------------------------------------------------
# TestSystemMessageExtraction
# ---------------------------------------------------------------------------


class TestSystemMessageExtraction:
    def test_system_message_routed_to_instructions(self, llm_config: LLMConfig) -> None:
        client = create_mock_client(make_stream_events(["ok"]))
        llm = _build_llm(client, llm_config)

        list(llm.generate([_system_msg("Be nice"), _user_msg("hi")]))

        call_kwargs = client.responses.create.call_args[1]
        assert call_kwargs["instructions"] == "Be nice"
        assert call_kwargs["input"] == [_user_msg("hi")]

    def test_no_system_message_omits_instructions(self, llm_config: LLMConfig) -> None:
        client = create_mock_client(make_stream_events(["ok"]))
        llm = _build_llm(client, llm_config)

        list(llm.generate([_user_msg("hi")]))

        call_kwargs = client.responses.create.call_args[1]
        assert "instructions" not in call_kwargs
        assert call_kwargs["input"] == [_user_msg("hi")]

    def test_multi_turn_with_system(self, llm_config: LLMConfig) -> None:
        client = create_mock_client(make_stream_events(["ok"]))
        llm = _build_llm(client, llm_config)

        messages = [_system_msg("Sys"), _user_msg("A"), _assistant_msg("B"), _user_msg("C")]
        list(llm.generate(messages))

        call_kwargs = client.responses.create.call_args[1]
        assert call_kwargs["instructions"] == "Sys"
        assert call_kwargs["input"] == [_user_msg("A"), _assistant_msg("B"), _user_msg("C")]


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_api_error_wrapped_in_llm_error(self, llm_config: LLMConfig) -> None:
        client = create_mock_client(
            side_effect=openai.APIConnectionError(request=MagicMock()),
        )
        llm = _build_llm(client, llm_config)

        with pytest.raises(LLMError):
            list(llm.generate([_user_msg("hi")]))

    def test_auth_error_wrapped(self, llm_config: LLMConfig) -> None:
        client = create_mock_client(
            side_effect=openai.AuthenticationError(
                message="bad key",
                response=MagicMock(status_code=401, headers={}),
                body=None,
            ),
        )
        llm = _build_llm(client, llm_config)

        with pytest.raises(LLMError):
            list(llm.generate([_user_msg("hi")]))

    def test_rate_limit_error_wrapped(self, llm_config: LLMConfig) -> None:
        client = create_mock_client(
            side_effect=openai.RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429, headers={}),
                body=None,
            ),
        )
        llm = _build_llm(client, llm_config)

        with pytest.raises(LLMError):
            list(llm.generate([_user_msg("hi")]))

    def test_streaming_error_wrapped(self, llm_config: LLMConfig) -> None:
        """Error during stream iteration is wrapped in LLMError."""
        client = create_mock_client()
        mock_stream = client.responses.create.return_value
        mock_stream.__iter__ = MagicMock(
            side_effect=openai.APIConnectionError(request=MagicMock()),
        )
        llm = _build_llm(client, llm_config)

        with pytest.raises(LLMError):
            list(llm.generate([_user_msg("hi")]))

    def test_unexpected_streaming_exception_wrapped(self, llm_config: LLMConfig) -> None:
        """Non-OpenAI exception during streaming is wrapped in LLMError."""
        client = create_mock_client()
        mock_stream = client.responses.create.return_value
        mock_stream.__iter__ = MagicMock(side_effect=RuntimeError("boom"))
        llm = _build_llm(client, llm_config)

        with pytest.raises(LLMError, match="boom"):
            list(llm.generate([_user_msg("hi")]))


# ---------------------------------------------------------------------------
# TestTimeoutWrapping
# ---------------------------------------------------------------------------


class TestTimeoutWrapping:
    def test_timeout_error_wrapped(self, llm_config: LLMConfig) -> None:
        client = create_mock_client(
            side_effect=openai.APITimeoutError(request=MagicMock()),
        )
        llm = _build_llm(client, llm_config)

        with pytest.raises(LLMError):
            list(llm.generate([_user_msg("hi")]))


# ---------------------------------------------------------------------------
# TestStreamCleanup
# ---------------------------------------------------------------------------


class TestStreamCleanup:
    def test_stream_closed_after_full_iteration(self, llm_config: LLMConfig) -> None:
        client = create_mock_client(make_stream_events(["a", "b"]))
        llm = _build_llm(client, llm_config)

        list(llm.generate([_user_msg("hi")]))

        client.responses.create.return_value.close.assert_called()

    def test_stream_closed_on_partial_iteration(self, llm_config: LLMConfig) -> None:
        """Barge-in: caller consumes one chunk then closes the iterator."""
        client = create_mock_client(make_stream_events(["a", "b", "c"]))
        llm = _build_llm(client, llm_config)

        gen = llm.generate([_user_msg("hi")])
        next(gen)  # consume first chunk
        gen.close()  # barge-in

        client.responses.create.return_value.close.assert_called()

    def test_stream_closed_on_error(self, llm_config: LLMConfig) -> None:
        client = create_mock_client()
        mock_stream = client.responses.create.return_value
        mock_stream.__iter__ = MagicMock(side_effect=RuntimeError("boom"))
        llm = _build_llm(client, llm_config)

        with pytest.raises(LLMError):
            list(llm.generate([_user_msg("hi")]))

        mock_stream.close.assert_called()

    def test_stream_closed_when_closed_before_first_next(self, llm_config: LLMConfig) -> None:
        """Close iterator before consuming any chunk — stream must still be released."""
        client = create_mock_client(make_stream_events(["a", "b"]))
        llm = _build_llm(client, llm_config)

        gen = llm.generate([_user_msg("hi")])
        gen.close()  # close before first next()

        client.responses.create.return_value.close.assert_called()


# ---------------------------------------------------------------------------
# TestClientConfig
# ---------------------------------------------------------------------------


class TestClientConfig:
    def test_max_retries_passed_to_client(self) -> None:
        config = LLMConfig(max_retries=5)
        with patch("voice_pipeline.llm.llm.openai.OpenAI") as mock_cls:
            OpenAILLM(config)
            mock_cls.assert_called_once_with(max_retries=5, timeout=30.0)

    def test_timeout_passed_to_client(self) -> None:
        config = LLMConfig(timeout_sec=60.0)
        with patch("voice_pipeline.llm.llm.openai.OpenAI") as mock_cls:
            OpenAILLM(config)
            mock_cls.assert_called_once_with(max_retries=2, timeout=60.0)

    def test_model_and_temperature_passed_to_create(self, llm_config: LLMConfig) -> None:
        client = create_mock_client(make_stream_events(["ok"]))
        llm = _build_llm(client, llm_config)

        list(llm.generate([_user_msg("hi")]))

        call_kwargs = client.responses.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_output_tokens"] == 256

    def test_custom_config_values_propagated(self) -> None:
        config = LLMConfig(model="gpt-4o-mini", temperature=0.3, max_tokens=100)
        client = create_mock_client(make_stream_events(["ok"]))
        llm = _build_llm(client, config)

        list(llm.generate([_user_msg("hi")]))

        call_kwargs = client.responses.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_output_tokens"] == 100

    def test_reasoning_effort_omitted_by_default(self, llm_config: LLMConfig) -> None:
        """Default config (gpt-4o) should not send reasoning param."""
        client = create_mock_client(make_stream_events(["ok"]))
        llm = _build_llm(client, llm_config)

        list(llm.generate([_user_msg("hi")]))

        call_kwargs = client.responses.create.call_args[1]
        assert "reasoning" not in call_kwargs

    def test_reasoning_effort_passed_when_set(self) -> None:
        config = LLMConfig(model="gpt-5.4", reasoning_effort="none")
        client = create_mock_client(make_stream_events(["ok"]))
        llm = _build_llm(client, config)

        list(llm.generate([_user_msg("hi")]))

        call_kwargs = client.responses.create.call_args[1]
        assert call_kwargs["reasoning"] == {"effort": "none"}

    def test_tools_omitted_when_empty(self) -> None:
        """Empty tools list should not send tools param."""
        config = LLMConfig(tools=[])
        client = create_mock_client(make_stream_events(["ok"]))
        llm = _build_llm(client, config)

        list(llm.generate([_user_msg("hi")]))

        call_kwargs = client.responses.create.call_args[1]
        assert "tools" not in call_kwargs

    def test_tools_resolved_from_names(self) -> None:
        config = LLMConfig(tools=["web_search"])
        client = create_mock_client(make_stream_events(["ok"]))
        llm = _build_llm(client, config)

        list(llm.generate([_user_msg("hi")]))

        call_kwargs = client.responses.create.call_args[1]
        assert call_kwargs["tools"] == [{"type": "web_search"}]

    def test_unknown_tool_name_skipped(self) -> None:
        config = LLMConfig(tools=["web_search", "nonexistent"])
        client = create_mock_client(make_stream_events(["ok"]))
        llm = _build_llm(client, config)

        list(llm.generate([_user_msg("hi")]))

        call_kwargs = client.responses.create.call_args[1]
        assert call_kwargs["tools"] == [{"type": "web_search"}]
