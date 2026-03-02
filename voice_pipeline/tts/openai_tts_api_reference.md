# OpenAI Text-to-Speech (TTS) API Reference (Early 2025-2026)

Last verified: **2026-03-02**

This document summarizes the current OpenAI TTS API surface for Python (`openai` package) and HTTP.

## 1) Endpoint and Python SDK method

### REST endpoint
- **Method**: `POST`
- **Path**: `https://api.openai.com/v1/audio/speech`

### Python SDK methods (`openai` package)
- Non-streaming: `client.audio.speech.create(...)`
- Streaming: `client.audio.speech.with_streaming_response.create(...)`

### Non-streaming example
```python
from openai import OpenAI

client = OpenAI()

audio = client.audio.speech.create(
    model="gpt-4o-mini-tts",
    voice="alloy",
    input="Hello from OpenAI text-to-speech.",
    response_format="mp3",
)

audio.write_to_file("speech.mp3")
```

### Streaming example (binary audio chunks)
```python
from openai import OpenAI

client = OpenAI()

with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="alloy",
    input="Streaming audio chunk by chunk.",
    response_format="pcm",
    stream_format="audio",  # "audio" or "sse"
) as response:
    with open("speech.pcm", "wb") as f:
        for chunk in response.iter_bytes(chunk_size=4096):
            if chunk:
                f.write(chunk)
```

### Streaming example (SSE mode)
```python
from openai import OpenAI

client = OpenAI()

with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="alloy",
    input="Stream as SSE events.",
    stream_format="sse",
) as response:
    for line in response.iter_lines():
        # Parse SSE lines/events as needed
        print(line)
```

## 2) Request parameters

## Required body fields
- `input` (string): text to synthesize. Max length documented as **4096 characters**.
- `model` (string): TTS model name.
- `voice` (string): voice preset or supported voice identifier.

## Optional body fields
- `instructions` (string): additional style/voice directions.
  - Documented as supported for `gpt-4o-mini-tts`.
  - Documented as **not working** with `tts-1` and `tts-1-hd`.
- `response_format` (string): `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`.
- `speed` (number): range **0.25 to 4.0**, default `1.0`.
- `stream_format` (string): `sse` or `audio`.
  - API reference indicates default `sse`.
  - `sse` is documented as unsupported for `tts-1` / `tts-1-hd`.

## SDK transport options (Python method signature)
- `extra_headers`
- `extra_query`
- `extra_body`
- `timeout`

## Model options (as documented)
- `tts-1`
- `tts-1-hd`
- `gpt-4o-mini-tts`
- Model snapshot identifiers may also appear in model pages (example: `gpt-4o-mini-tts-2024-07-18`).

## Voice options
From OpenAI guide/API/docs, currently documented voices include:
- `alloy`
- `ash`
- `ballad`
- `coral`
- `echo`
- `fable`
- `onyx`
- `nova`
- `sage`
- `shimmer`
- `verse`

Additional voice mentions seen in current API/SDK references:
- `marin`
- `cedar`
- custom voice IDs (Audio object style)

Note: voice lists in different OpenAI pages/SDK type hints are not perfectly synchronized; validate against the live endpoint in production.

## 3) Response format details

## Binary response
- The `/audio/speech` endpoint returns audio bytes (`application/octet-stream`).

## Output format notes
- `mp3`: default in many examples.
- `opus`: low-latency streaming-oriented codec.
- `aac`: common for digital audio distribution.
- `flac`: lossless compression.
- `wav`: uncompressed container.
- `pcm`: raw PCM frames.

## PCM specifics
OpenAI’s TTS guide describes PCM output as:
- **24 kHz** sample rate
- **16-bit signed**
- **little-endian**
- **mono**
- **headerless raw PCM** (unlike WAV)

## Streaming transport
- OpenAI documents real-time TTS streaming via **HTTP chunked transfer encoding**.
- With `stream_format="audio"`, the body is streamed as audio bytes.
- With `stream_format="sse"`, the stream uses Server-Sent Events.

## 4) Streaming API details

### `with_streaming_response.create(...)`
- Returns a streamed response context manager.
- Use `with ... as response:` so the HTTP connection closes cleanly after streaming.

### Iterating over audio chunks
```python
with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="alloy",
    input="Chunked playback",
    response_format="mp3",
    stream_format="audio",
) as response:
    for chunk in response.iter_bytes(chunk_size=8192):
        # send chunk to player/socket/file
        pass
```

### `iter_bytes()` chunk size behavior
From current Python SDK + `httpx` behavior:
- Signature: `iter_bytes(chunk_size: int | None = None)`.
- If `chunk_size` is an integer, output is chunked to approximately that size (last chunk can be smaller).
- If `chunk_size=None`, chunk boundaries follow underlying decoder/network buffering and are **not fixed-size**.

Implementation detail: this is SDK/client behavior (not a strict server contract), so exact boundaries can vary.

## 5) Error handling and rate limits

## Common API status errors
OpenAI’s error guide highlights these common statuses:
- `401` authentication/key/org issues
- `403` permission or region restrictions
- `429` rate-limit or quota exhaustion
- `500` server error
- `503` overload / slow-down

## Common Python exceptions (`openai`)
- `BadRequestError`
- `AuthenticationError`
- `PermissionDeniedError`
- `NotFoundError`
- `RateLimitError`
- `InternalServerError`
- `APIConnectionError`
- `APITimeoutError`

## Rate limits
- Rate limits depend on usage tier/project and can change.
- Model pages show these published RPM examples:
  - `tts-1`: Tier 1 `500 RPM`, Tier 2 `5000 RPM`, Tier 3 `5000 RPM`, Tier 4 `10000 RPM`, Tier 5 `10000 RPM`.
  - `tts-1-hd`: same published RPM values.
- For `gpt-4o-mini-tts`, verify live limits in your account’s **Limits** page (tier-dependent).

## 6) Word-level timestamps

OpenAI TTS (`/audio/speech`) does **not** currently provide word-level timestamps in its response.

Why: the speech endpoint returns audio bytes and exposes no timestamp granularity parameter for synthesis output.

If you need aligned words, typical workaround is a second ASR step (transcribe generated audio with word timestamps).

## 7) Pricing (per 1M characters)

Important: OpenAI currently publishes TTS pricing primarily **per 1M text tokens**.

Published token pricing:
- `tts-1`: **$15.00 / 1M text tokens**
- `tts-1-hd`: **$30.00 / 1M text tokens**
- `gpt-4o-mini-tts`: **$0.60 / 1M text tokens**

Approximate conversion to **1M characters** (English heuristic: ~4 chars/token => 1M chars ~= 250k tokens):
- `tts-1`: ~$3.75 per 1M chars
- `tts-1-hd`: ~$7.50 per 1M chars
- `gpt-4o-mini-tts`: ~$0.15 per 1M chars

These character-based numbers are estimates only; actual billing is token-based and language/text-dependent.

## Sources
- OpenAI API reference, Create speech: https://platform.openai.com/docs/api-reference/audio/createSpeech
- OpenAI TTS guide: https://platform.openai.com/docs/guides/text-to-speech
- OpenAI model page (`tts-1`): https://platform.openai.com/docs/models/tts-1
- OpenAI model page (`tts-1-hd`): https://platform.openai.com/docs/models/tts-1-hd
- OpenAI model page (`gpt-4o-mini-tts`): https://platform.openai.com/docs/models/gpt-4o-mini-tts
- OpenAI pricing: https://platform.openai.com/pricing and https://openai.com/api/pricing
- OpenAI error codes: https://platform.openai.com/docs/guides/error-codes
- OpenAI rate limits: https://platform.openai.com/docs/guides/rate-limits
- Local SDK verification (`openai==1.109.1`):
  - `.../site-packages/openai/resources/audio/speech.py`
  - `.../site-packages/openai/types/audio/speech_create_params.py`
  - `.../site-packages/openai/_response.py`
  - `.../site-packages/openai/_legacy_response.py`
