"""Shared fixtures for TTS tests."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Unit test helpers
# ---------------------------------------------------------------------------


def create_mock_client(
    chunks: list[bytes] | None = None,
    *,
    side_effect: Exception | None = None,
    streaming_error: Exception | None = None,
) -> MagicMock:
    """Create a mock ``openai.OpenAI`` client for TTS.

    The mock wires up ``audio.speech.with_streaming_response.create()``
    to return a context manager whose ``__enter__`` returns a response
    with ``iter_bytes()`` yielding *chunks*.

    Args:
        chunks: Audio byte chunks the mock stream will yield.
        side_effect: Exception to raise from ``with_streaming_response.create()``.
        streaming_error: Exception to raise during ``iter_bytes()`` iteration.

    Returns:
        A configured ``MagicMock`` that mimics the OpenAI client.
    """
    client = MagicMock()

    if side_effect is not None:
        client.audio.speech.with_streaming_response.create.side_effect = side_effect
        return client

    mock_response = MagicMock()

    if streaming_error is not None:

        def _bad_iter(**kwargs):  # noqa: ARG001
            raise streaming_error

        mock_response.iter_bytes = _bad_iter
    else:
        mock_response.iter_bytes = MagicMock(return_value=iter(chunks or []))

    # Context manager protocol: __enter__ returns mock_response
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_response)
    mock_cm.__exit__ = MagicMock(return_value=False)

    client.audio.speech.with_streaming_response.create.return_value = mock_cm

    return client


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
