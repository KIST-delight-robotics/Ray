"""OpenAI Responses API 스트리밍 LLM.

도구(web_search 등) 정의와 토큰 비용도 여기(``_TOOL_REGISTRY``)에서 관리한다. 새 도구 추가 시
항목을 등록하고, ``token_cost`` 는 도구 유무에 따른 API ``input_tokens`` 차이로 측정한다.

환경변수 ``OPENAI_API_KEY`` 필요. API 제약: docs/modules/openai_responses_api_constraints.md
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import openai

from voice_pipeline.types import ILLM, LLMMetrics, LLMResult, LLMStream, ToolCall, Usage

logger = logging.getLogger("voice_pipeline.llm")

ToolDef = dict[str, Any]


@dataclass(frozen=True)
class _ToolEntry:
    """Tool definition + measured token cost."""

    definition: ToolDef
    token_cost: int  # measured via API input_tokens comparison


_TOOL_REGISTRY: dict[str, _ToolEntry] = {
    "web_search": _ToolEntry(
        definition={"type": "web_search"},
        token_cost=294,
    ),
}


def resolve_tools(names: list[str]) -> list[ToolDef]:
    """Resolve tool names to API-ready tool definitions.

    Unknown names are logged as warnings and skipped.
    """
    resolved = []
    for name in names:
        entry = _TOOL_REGISTRY.get(name)
        if entry is not None:
            resolved.append(entry.definition)
        else:
            logger.warning("Unknown tool name '%s', skipping", name)
    return resolved


def get_tools_token_cost(names: list[str]) -> int:
    """Return the total token cost of the given tools.

    Uses empirically measured values. Unknown tools are skipped.
    """
    total = 0
    for name in names:
        entry = _TOOL_REGISTRY.get(name)
        if entry is not None:
            total += entry.token_cost
    return total


class OpenAILLM(ILLM):
    """LLM implementation using the OpenAI Responses API.

    Reads ``OPENAI_API_KEY`` from the environment. Streams text chunks
    via ``client.responses.create(stream=True)``.

    Returns LLMStream: iterate for text deltas, access ``.result``
    after consumption for LLMResult (text, tool_calls, metrics).

    Args:
        model: OpenAI 모델 이름 (예: "gpt-4o", "gpt-4o-mini", "gpt-5.4").
        temperature: 샘플링 temperature (0.0~2.0).
        reasoning_effort: reasoning 모델용 effort 레벨 (gpt-5 계열). None=미적용
        max_tokens: 응답 최대 토큰 수.
        tools: 활성화할 도구 이름 목록. None이면 기본 도구, ``[]``이면 명시적 비활성화.
    """

    _MAX_RETRIES = 2  # 응답 실패 시 자동 재시도 횟수
    _TIMEOUT_SEC = 30.0  # 응답 대기 최대 시간 (초)
    _DEFAULT_TOOLS: tuple[str, ...] = ("web_search",)  # tools=None일 때 기본 도구

    def __init__(
        self,
        model: str = "gpt-5.4",
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        max_tokens: int = 256,
        tools: list[str] | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.tools: list[str] = list(tools) if tools is not None else list(self._DEFAULT_TOOLS)
        self._resolved_tools = resolve_tools(self.tools) if self.tools else []
        self._client = openai.OpenAI(
            max_retries=self._MAX_RETRIES,
            timeout=self._TIMEOUT_SEC,
        )

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMStream:
        """Generate a streaming response from the given message history.

        Args:
            messages: List of message dicts.
            tools: Tool definitions. None uses config defaults.
                Empty list explicitly disables tools.
            response_format: Structured output format specification.
                None = free-form text. Passed to the API as
                ``text={"format": response_format}`` when set.

        Returns:
            LLMStream yielding text chunks. After full iteration,
            .result provides LLMResult with text, tool_calls, metrics.
        """
        instructions, input_messages = _split_system_message(messages)

        # Resolve tools: None → config default, [] → no tools
        resolved_tools = self._resolved_tools if tools is None else tools

        start_time = time.monotonic()

        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "input": input_messages,
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "stream": True,
            }
            if instructions is not None:
                kwargs["instructions"] = instructions
            if self.reasoning_effort is not None:
                kwargs["reasoning"] = {"effort": self.reasoning_effort}
            if resolved_tools:
                kwargs["tools"] = resolved_tools
            if response_format is not None:
                kwargs["text"] = {"format": response_format}

            stream = self._client.responses.create(**kwargs)
        except openai.OpenAIError as exc:
            logger.warning("OpenAI API error: %s", exc)
            raise RuntimeError(str(exc)) from exc

        # Shared mutable state between generator and result_fn
        state = _StreamState(model=self.model, start_time=start_time)

        gen = _iter_stream(stream, state)

        def result_fn(full_text: str) -> LLMResult:
            return state.build_result(full_text)

        return LLMStream(
            gen,
            close_fn=lambda: _close_stream(stream),
            result_fn=result_fn,
        )


class _StreamState:
    """Mutable state shared between the stream generator and result builder."""

    __slots__ = (
        "model",
        "start_time",
        "first_token_time",
        "tool_calls",
        "completed_response",
    )

    def __init__(self, model: str, start_time: float) -> None:
        self.model = model
        self.start_time = start_time
        self.first_token_time: float | None = None
        self.tool_calls: list[ToolCall] = []
        self.completed_response: Any = None

    def build_result(self, full_text: str) -> LLMResult:
        """Build LLMResult from accumulated state."""
        end_time = time.monotonic()
        latency_ms = int((end_time - self.start_time) * 1000)
        ttft_ms = (
            int((self.first_token_time - self.start_time) * 1000) if self.first_token_time is not None else latency_ms
        )

        metrics = self._extract_metrics(latency_ms, ttft_ms)
        return LLMResult(
            text=full_text,
            tool_calls=tuple(self.tool_calls),
            metrics=metrics,
        )

    def _extract_metrics(self, latency_ms: int, ttft_ms: int) -> LLMMetrics | None:
        """Extract metrics from the completed response.

        ``response.completed`` 이벤트의 ``response``는 pydantic 객체일 수도,
        서버 페이로드에 따라 plain dict일 수도 있어 ``_field``로 양쪽을 허용한다.
        """
        resp = self.completed_response
        if resp is None:
            return None

        usage_data = _field(resp, "usage")
        input_tokens = _field(usage_data, "input_tokens")
        output_tokens = _field(usage_data, "output_tokens")
        if input_tokens is None or output_tokens is None:
            logger.debug("Usage missing from completed response (type=%s)", type(resp).__name__)
            return None

        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=_field(_field(usage_data, "input_tokens_details"), "cached_tokens", 0) or 0,
            reasoning_tokens=_field(_field(usage_data, "output_tokens_details"), "reasoning_tokens", 0) or 0,
        )

        model = _field(resp, "model", self.model) or self.model
        return LLMMetrics(
            usage=usage,
            model=model,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
        )


def _split_system_message(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract the system message from the front of the message list."""
    if messages and messages[0].get("role") == "system":
        return messages[0]["content"], messages[1:]
    return None, messages


def _field(obj: Any, key: str, default: Any = None) -> Any:
    """Read *key* from a pydantic object or a plain dict.

    Responses API 스트림의 completed payload가 SDK 파싱 결과에 따라
    객체/dict 어느 쪽으로도 도착할 수 있어 양쪽을 허용한다.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _iter_stream(stream: Any, state: _StreamState) -> Generator[str, None, None]:
    """Iterate over a Responses API stream, yielding text deltas.

    Captures tool_calls and completed response in the shared state.
    """
    try:
        for event in stream:
            if event.type == "response.output_text.delta":
                if state.first_token_time is None:
                    state.first_token_time = time.monotonic()
                yield event.delta

            elif event.type == "response.completed":
                state.completed_response = event.response
                # Extract tool calls from response output
                for output_item in _field(event.response, "output") or []:
                    if _field(output_item, "type") == "function_call":
                        state.tool_calls.append(
                            ToolCall(
                                call_id=_field(output_item, "call_id"),
                                name=_field(output_item, "name"),
                                arguments=_field(output_item, "arguments"),
                            )
                        )

    except GeneratorExit:
        return
    except openai.OpenAIError as exc:
        logger.warning("OpenAI streaming error: %s", exc)
        raise RuntimeError(str(exc)) from exc
    except Exception as exc:
        logger.warning("Unexpected streaming error: %s", exc)
        raise RuntimeError(str(exc)) from exc
    finally:
        _close_stream(stream)


def _close_stream(stream: Any) -> None:
    """Close the stream, suppressing errors."""
    try:
        stream.close()
    except Exception:
        logger.debug("Error closing stream (suppressed)", exc_info=True)
