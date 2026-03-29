"""Shared fixtures for LLM tests."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

import pytest

from voice_pipeline.core.config import LLMConfig

# ---------------------------------------------------------------------------
# Unit test helpers
# ---------------------------------------------------------------------------


class FakeStreamEvent:
    """Minimal event object mimicking ResponseStreamEvent."""

    def __init__(
        self,
        event_type: str,
        delta: str = "",
        response: Any = None,
    ) -> None:
        self.type = event_type
        self.delta = delta
        self.response = response


class FakeUsageDetails:
    """Minimal usage details for testing."""

    def __init__(self, cached_tokens: int = 0, reasoning_tokens: int = 0) -> None:
        self.cached_tokens = cached_tokens
        self.reasoning_tokens = reasoning_tokens


class FakeUsage:
    """Minimal usage object for testing."""

    def __init__(
        self,
        input_tokens: int = 50,
        output_tokens: int = 10,
        input_tokens_details: FakeUsageDetails | None = None,
        output_tokens_details: FakeUsageDetails | None = None,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.input_tokens_details = input_tokens_details or FakeUsageDetails()
        self.output_tokens_details = output_tokens_details or FakeUsageDetails()


class FakeCompletedResponse:
    """Minimal completed response for testing."""

    def __init__(
        self,
        model: str = "gpt-4o",
        usage: FakeUsage | None = None,
        output: list[Any] | None = None,
    ) -> None:
        self.model = model
        self.usage = usage or FakeUsage()
        self.output = output or []


def make_stream_events(
    chunks: list[str],
    *,
    include_completed: bool = True,
    completed_response: FakeCompletedResponse | None = None,
) -> list[FakeStreamEvent]:
    """Build a sequence of text delta events with optional completed event."""
    events = [FakeStreamEvent("response.output_text.delta", c) for c in chunks]
    if include_completed:
        resp = completed_response or FakeCompletedResponse()
        events.append(FakeStreamEvent("response.completed", response=resp))
    return events


def create_mock_client(
    stream_events: list[FakeStreamEvent] | None = None,
    side_effect: Exception | None = None,
) -> MagicMock:
    """Create a mock ``openai.OpenAI`` client."""
    client = MagicMock()

    if side_effect is not None:
        client.responses.create.side_effect = side_effect
        return client

    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(return_value=iter(stream_events or []))
    mock_stream.close = MagicMock()
    client.responses.create.return_value = mock_stream
    return client


@pytest.fixture
def llm_config() -> LLMConfig:
    """Default LLMConfig for tests."""
    return LLMConfig()


# ---------------------------------------------------------------------------
# Integration test helpers
# ---------------------------------------------------------------------------

_SKIP_MSG = "OPENAI_API_KEY not set"


@pytest.fixture(scope="session")
def openai_api_key() -> str:
    """Read OPENAI_API_KEY from env, skip if absent."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip(_SKIP_MSG)
    return key
