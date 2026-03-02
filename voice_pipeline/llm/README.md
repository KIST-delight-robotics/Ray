# LLM Module

Streaming text generation using the OpenAI Responses API.

API constraints (rate limits, token limits, etc.) are documented in [`openai_responses_api_constraints.md`](openai_responses_api_constraints.md).


## Setup

1. Get an API key from [OpenAI](https://platform.openai.com/api-keys).
2. Set the environment variable:

```bash
export OPENAI_API_KEY=sk-...
```


## Config

### `LLMConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `"gpt-4o"` | OpenAI model name |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `max_tokens` | `int` | `256` | Maximum output tokens (maps to API `max_output_tokens`) |
| `max_retries` | `int` | `2` | SDK retry count for transient errors (429, 500, 503) |
| `timeout_sec` | `float` | `30.0` | Request timeout in seconds |


## Usage

### Basic streaming

```python
from voice_pipeline.core.config import LLMConfig
from voice_pipeline.llm import OpenAILLM

llm = OpenAILLM(LLMConfig())
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]

for chunk in llm.generate(messages):
    print(chunk, end="", flush=True)
```

### Barge-in (early termination)

The caller **must** either exhaust the iterator or call `.close()` to release the HTTP connection:

```python
gen = llm.generate(messages)
first_chunk = next(gen)
gen.close()  # releases the stream
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
