"""Shared fixtures for LLM tests."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from voice_pipeline.core.config import LLMConfig

# ---------------------------------------------------------------------------
# Unit test helpers
# ---------------------------------------------------------------------------


class FakeStreamEvent:
    """Minimal event object mimicking ResponseStreamEvent."""

    def __init__(self, event_type: str, delta: str = "") -> None:
        self.type = event_type
        self.delta = delta


def make_stream_events(chunks: list[str]) -> list[FakeStreamEvent]:
    """Build a sequence of text delta events from string chunks."""
    return [FakeStreamEvent("response.output_text.delta", c) for c in chunks]


def create_mock_client(
    stream_events: list[FakeStreamEvent] | None = None,
    side_effect: Exception | None = None,
) -> MagicMock:
    """Create a mock ``openai.OpenAI`` client.

    Args:
        stream_events: Events the mock stream will yield.
        side_effect: Exception to raise from ``responses.create()``.

    Returns:
        A configured ``MagicMock`` that mimics the OpenAI client.
    """
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
