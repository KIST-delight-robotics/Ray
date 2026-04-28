# LLM Module

Streaming text generation using the OpenAI Responses API.

API constraints (rate limits, token limits, etc.) are documented in [`openai_responses_api_constraints.md`](openai_responses_api_constraints.md).


## Setup

1. Get an API key from [OpenAI](https://platform.openai.com/api-keys).
2. Set the environment variable:

```bash
export OPENAI_API_KEY=sk-...
```


## `OpenAILLM.__init__` 인자

| 인자 | Default | 의미 |
|------|---------|------|
| `model` | `"gpt-4o"` | OpenAI 모델 이름 |
| `temperature` | `0.7` | 샘플링 temperature (0.0~2.0) |
| `max_tokens` | `256` | 응답 최대 토큰 수 (API `max_output_tokens`) |
| `tools` | `None` | 도구 이름 목록. None이면 기본 도구, `[]`이면 비활성화 |

## 클래스 변수

`OpenAILLM` 클래스 내부 상수.

| 변수 | 값 | 의미 |
|------|------|------|
| `_MAX_RETRIES` | `2` | 응답 실패 시 자동 재시도 횟수 |
| `_TIMEOUT_SEC` | `30.0` | 응답 대기 최대 시간 (초) |
| `_DEFAULT_TOOLS` | `("web_search",)` | `tools=None`일 때 기본 도구 |
| `_REASONING_EFFORT` | `None` | reasoning 모델용 effort 레벨 (gpt-5 계열). None=미적용 |


## Usage

### Basic streaming

```python
from voice_pipeline.llm import OpenAILLM

llm = OpenAILLM()
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]

stream = llm.generate(messages)
for chunk in stream:
    print(chunk, end="", flush=True)

# After consumption: access metrics
result = stream.result
print(f"\nTokens: in={result.metrics.usage.input_tokens}, out={result.metrics.usage.output_tokens}")
print(f"Latency: {result.metrics.latency_ms}ms, TTFT: {result.metrics.ttft_ms}ms")
```

### Barge-in (early termination)

The caller **must** either exhaust the iterator or call `.close()` to release the HTTP connection:

```python
stream = llm.generate(messages)
first_chunk = next(stream)
stream.close()  # releases the stream — .result not available after close
```

### Tool control

```python
stream = llm.generate(messages, tools=None)   # use config defaults (e.g. web_search)
stream = llm.generate(messages, tools=[])     # explicitly disable tools
```

### Token counter

```python
from voice_pipeline.llm import create_token_counter

counter = create_token_counter("gpt-4o")
print(counter("Hello, world!"))  # e.g. 4
```

### System message handling

If the first message has `role == "system"`, it is extracted and passed via the Responses API `instructions` parameter. Remaining messages are passed as `input`.

### Error handling

All errors (API errors, network issues, timeouts) are wrapped in `LLMError`. The orchestrator never sees raw SDK exceptions.


## Testing

### Unit tests (mocked)

```bash
uv run pytest voice_pipeline/tests/llm/test_llm.py voice_pipeline/tests/llm/test_token_counter.py -v
```

### Integration & stress tests (real API)

```bash
OPENAI_API_KEY=sk-... uv run pytest -m requires_api voice_pipeline/tests/llm/ -v
```

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
