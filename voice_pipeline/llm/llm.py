"""OpenAI Responses API streaming LLM implementation."""

from __future__ import annotations

import logging
from collections.abc import Generator, Iterator
from typing import Any

import openai

from voice_pipeline.core.config import LLMConfig
from voice_pipeline.core.interfaces import ILLM
from voice_pipeline.llm.exceptions import LLMError
from voice_pipeline.llm.tools import resolve_tools

logger = logging.getLogger("voice_pipeline.llm")


class OpenAILLM(ILLM):
    """LLM implementation using the OpenAI Responses API.

    Reads ``OPENAI_API_KEY`` from the environment. Streams text chunks
    via ``client.responses.create(stream=True)``.

    The iterator returned by :meth:`generate` must be fully consumed or
    explicitly closed (via :meth:`~Iterator.close` or exhaustion) to
    release the underlying HTTP connection.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._tools = resolve_tools(config.tools) if config.tools else []
        self._client = openai.OpenAI(
            max_retries=config.max_retries,
            timeout=config.timeout_sec,
        )

    def generate(self, messages: list[dict[str, Any]]) -> Iterator[str]:
        """Generate a streaming response from the given message history.

        System messages (``role == "system"``) are extracted and passed
        via the Responses API ``instructions`` parameter.  Remaining
        messages are passed as ``input``.

        Args:
            messages: List of message dicts (``role`` / ``content`` keys).

        Returns:
            Iterator yielding text chunks as they become available.

        Raises:
            LLMError: On any API or streaming error.
        """
        instructions, input_messages = _split_system_message(messages)

        try:
            kwargs: dict[str, Any] = {
                "model": self._config.model,
                "input": input_messages,
                "temperature": self._config.temperature,
                "max_output_tokens": self._config.max_tokens,
                "stream": True,
            }
            if instructions is not None:
                kwargs["instructions"] = instructions
            if self._config.reasoning_effort is not None:
                kwargs["reasoning"] = {"effort": self._config.reasoning_effort}
            if self._tools:
                kwargs["tools"] = self._tools

            stream = self._client.responses.create(**kwargs)
        except openai.OpenAIError as exc:
            logger.warning("OpenAI API error: %s", exc)
            raise LLMError(str(exc)) from exc

        return _SafeStreamIterator(stream, _iter_stream(stream))


class _SafeStreamIterator(Iterator[str]):
    """Wrapper that ensures the HTTP stream is closed even if iteration never starts.

    A bare generator's ``finally`` block only runs once the generator body has
    been entered (i.e., after the first ``next()``).  If the caller calls
    ``.close()`` before ever calling ``next()``, the underlying stream would
    leak.  This wrapper intercepts ``close()`` and closes the stream directly.
    """

    __slots__ = ("_stream", "_gen")

    def __init__(self, stream: Any, gen: Generator[str, None, None]) -> None:
        self._stream = stream
        self._gen = gen

    def __next__(self) -> str:
        return next(self._gen)

    def __iter__(self) -> Iterator[str]:
        return self

    def close(self) -> None:
        """Close the generator and ensure the HTTP stream is released."""
        try:
            self._gen.close()
        finally:
            _close_stream(self._stream)


def _split_system_message(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract the system message from the front of the message list.

    Returns:
        A tuple of (instructions, remaining_messages).  ``instructions``
        is ``None`` if the first message is not a system message.
    """
    if messages and messages[0].get("role") == "system":
        return messages[0]["content"], messages[1:]
    return None, messages


def _iter_stream(stream: Any) -> Iterator[str]:
    """Iterate over a Responses API stream, yielding text deltas.

    Ensures ``stream.close()`` is called even when the caller
    abandons iteration early (barge-in) or never starts iteration.
    """
    try:
        for event in stream:
            if event.type == "response.output_text.delta":
                yield event.delta
    except GeneratorExit:
        # Caller closed the iterator (barge-in) — fall through to finally.
        return
    except openai.OpenAIError as exc:
        logger.warning("OpenAI streaming error: %s", exc)
        raise LLMError(str(exc)) from exc
    except Exception as exc:
        logger.warning("Unexpected streaming error: %s", exc)
        raise LLMError(str(exc)) from exc
    finally:
        _close_stream(stream)


def _close_stream(stream: Any) -> None:
    """Close the stream, suppressing errors to avoid masking the original exception."""
    try:
        stream.close()
    except Exception:
        logger.debug("Error closing stream (suppressed)", exc_info=True)
