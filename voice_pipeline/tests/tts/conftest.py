"""Shared fixtures for TTS tests."""

from __future__ import annotations

import base64
import os
from typing import Any
from unittest.mock import MagicMock

import pytest
from elevenlabs.types import StreamingAudioChunkWithTimestampsResponse

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


def make_elevenlabs_chunk(
    audio: bytes = b"",
    *,
    characters: list[str] | None = None,
    starts: list[float] | None = None,
    ends: list[float] | None = None,
) -> StreamingAudioChunkWithTimestampsResponse:
    """Build a real SDK chunk from wire-format data.

    Using the real pydantic type (validated from the JSON wire format) means
    unit tests catch attribute-name drift across SDK upgrades.

    Args:
        audio: Raw PCM bytes (base64-encoded into the chunk).
        characters: Alignment characters; None omits alignment entirely.
        starts: Per-character start times; defaults to 0.1s per character.
        ends: Per-character end times; defaults to 0.1s per character.

    Returns:
        A validated ``StreamingAudioChunkWithTimestampsResponse``.
    """
    payload: dict[str, Any] = {"audio_base64": base64.b64encode(audio).decode("ascii")}
    if characters is not None:
        if starts is None:
            starts = [i * 0.1 for i in range(len(characters))]
        if ends is None:
            ends = [(i + 1) * 0.1 for i in range(len(characters))]
        payload["alignment"] = {
            "characters": characters,
            "character_start_times_seconds": starts,
            "character_end_times_seconds": ends,
        }
    return StreamingAudioChunkWithTimestampsResponse.model_validate(payload)


def create_mock_elevenlabs_client(
    chunks: list[StreamingAudioChunkWithTimestampsResponse] | None = None,
    *,
    call_error: Exception | None = None,
    streaming_error: Exception | None = None,
) -> MagicMock:
    """Create a mock ``elevenlabs.ElevenLabs`` client for TTS.

    Mirrors the real SDK's laziness: ``stream_with_timestamps()`` returns a
    generator, so *call_error* raises at the first ``next()`` (like an HTTP
    error) and *streaming_error* raises after all *chunks* are yielded.

    The mock records stream lifecycle in ``client._stream_state``
    (``{"started": bool, "closed": bool}``) — ``closed`` becomes True when
    the generator's ``finally`` runs (exhaustion, error, or ``close()``).

    Args:
        chunks: SDK chunk objects the stream will yield (see
            :func:`make_elevenlabs_chunk`).
        call_error: Exception raised before the first chunk.
        streaming_error: Exception raised after the last chunk.

    Returns:
        A configured ``MagicMock`` that mimics the ElevenLabs client.
    """
    client = MagicMock()
    state = {"started": False, "closed": False}

    def _gen() -> Any:
        state["started"] = True
        try:
            if call_error is not None:
                raise call_error
            yield from chunks or []
            if streaming_error is not None:
                raise streaming_error
        finally:
            state["closed"] = True

    client.text_to_speech.stream_with_timestamps.side_effect = lambda *args, **kwargs: _gen()
    client._stream_state = state
    return client


# ---------------------------------------------------------------------------
# Integration test helpers
# ---------------------------------------------------------------------------

_SKIP_MSG = "OPENAI_API_KEY not set"
_ELEVENLABS_SKIP_MSG = "ELEVENLABS_API_KEY not set"


@pytest.fixture(scope="session")
def openai_api_key() -> str:
    """Read OPENAI_API_KEY from env, skip if absent."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip(_SKIP_MSG)
    return key


@pytest.fixture(scope="session")
def elevenlabs_api_key() -> str:
    """Read ELEVENLABS_API_KEY from env, skip if absent."""
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        pytest.skip(_ELEVENLABS_SKIP_MSG)
    return key
