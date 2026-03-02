# OpenAI Responses API Constraints

Vendor-specific limits and behaviors relevant to the LLM module.


## Rate Limits

Rate limits vary by model and tier. Check your account's [rate limits page](https://platform.openai.com/settings/organization/limits).

| Resource | Typical (Tier 1) |
|----------|------------------|
| RPM (requests per minute) | 500 |
| TPM (tokens per minute) | 30,000 |

The SDK handles 429 responses automatically via `max_retries` with exponential backoff.


## Token Limits

| Model | Max Context | Max Output |
|-------|------------|------------|
| gpt-4o | 128,000 | 16,384 |
| gpt-4o-mini | 128,000 | 16,384 |

`LLMConfig.max_tokens` maps to the API `max_output_tokens` parameter.


## Streaming

- `client.responses.create(stream=True)` returns a `Stream[ResponseStreamEvent]`.
- Text chunks arrive as `response.output_text.delta` events with a `delta` field.
- The stream must be closed (exhausted or `stream.close()`) to release the HTTP connection.
- Streaming timeout applies to the initial connection; individual chunk delivery is not independently timed.


## Responses API vs Chat Completions

This module uses the **Responses API** (`client.responses.create`), not the legacy Chat Completions API (`client.chat.completions.create`).

Key differences:
- System message is passed via `instructions`, not in the `messages` array.
- Input messages go in `input`, not `messages`.
- Output token limit is `max_output_tokens`, not `max_tokens`.
- No `previous_response_id` used — we manage context ourselves via `ContextBuilder`.


## Error Types

All inherit from `openai.OpenAIError`:

| Exception | Cause |
|-----------|-------|
| `AuthenticationError` | Invalid or missing API key |
| `RateLimitError` | 429 — too many requests |
| `APIConnectionError` | Network connectivity issue |
| `APITimeoutError` | Request exceeded timeout |
| `BadRequestError` | Invalid model name, malformed request |
| `InternalServerError` | 500 from OpenAI |

All are caught and wrapped in `LLMError` by the module.


## Retry Behavior

The OpenAI SDK retries these automatically (configured via `max_retries`):
- 429 (rate limit)
- 500, 503 (server errors)
- Connection errors

Retries use exponential backoff. Non-retryable errors (401, 400, 404) fail immediately.
