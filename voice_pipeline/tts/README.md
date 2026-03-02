# TTS Module

Streaming text-to-speech using the OpenAI Audio API.

API constraints (models, rate limits, PCM format, etc.) are documented in [`openai_tts_api_reference.md`](openai_tts_api_reference.md).


## Setup

Set the OpenAI API key environment variable:

```bash
export OPENAI_API_KEY=sk-...
```


## Config

### `TTSConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `vendor` | `str` | `"openai"` | TTS vendor (currently only OpenAI) |
| `voice` | `str` | `"alloy"` | Voice preset |
| `model` | `str` | `"tts-1"` | Model: `tts-1`, `tts-1-hd`, or `gpt-4o-mini-tts` |
| `output_sample_rate` | `int` | `24000` | PCM output sample rate (fixed by OpenAI) |
| `speed` | `float` | `1.0` | Playback speed (0.25–4.0) |
| `timeout_sec` | `float` | `30.0` | Request timeout in seconds |
| `max_retries` | `int` | `2` | SDK retry count for transient errors |
| `instructions` | `str` | `""` | Voice/style instructions (`gpt-4o-mini-tts` only) |


## Usage

### Streaming synthesis

`synthesize()` returns a `TTSStream` that yields PCM audio chunks (24 kHz, 16-bit signed LE, mono).

```python
from voice_pipeline.core.config import TTSConfig
from voice_pipeline.tts import OpenAITTS

tts = OpenAITTS(TTSConfig())
stream = tts.synthesize("Hello world")

for chunk in stream:
    # Send chunk to audio output / C++ bridge
    pass

# After full iteration, audio and timestamps are available
audio = stream.audio          # bytes: full PCM audio
timestamps = stream.timestamps  # tuple: () for OpenAI (no word timestamps)
result = stream.result          # TTSResult(audio, timestamps)
```

### Partial consumption (barge-in)

If the user interrupts, close the stream to release the HTTP connection:

```python
stream = tts.synthesize("Long response text...")
first_chunk = next(stream)
# User interrupted — close immediately
stream.close()
```

### Save to file (testing utility)

Non-streaming convenience method (not on the ITTS interface):

```python
tts.save_to_file("Hello world", "output.wav")
```

### Model-specific instructions

The `instructions` parameter is only supported by `gpt-4o-mini-tts`. For other models it is ignored with a warning log.

```python
tts = OpenAITTS(TTSConfig(
    model="gpt-4o-mini-tts",
    instructions="Speak in a cheerful tone.",
))
```


## PCM output format

OpenAI TTS with `response_format="pcm"` produces:

| Property | Value |
|----------|-------|
| Sample rate | 24 kHz |
| Bit depth | 16-bit signed |
| Byte order | Little-endian |
| Channels | Mono |
| Header | None (raw PCM) |


## Word timestamps

OpenAI TTS does **not** support word-level timestamps. `stream.timestamps` returns `()`. For barge-in text truncation, use `DurationRatioTruncator` which estimates from audio duration.


## Testing

### Unit tests (mocked)

```bash
uv run pytest voice_pipeline/tests/tts/test_tts.py -v
```

35 tests with mocked OpenAI client — no API credentials needed.

### Integration & stress tests (real API)

```bash
OPENAI_API_KEY=sk-... uv run pytest -m requires_api voice_pipeline/tests/tts/ -v
```

### Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
