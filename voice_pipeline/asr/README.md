# ASR Module

Streaming speech recognition using Google Cloud Speech-to-Text V1.

API constraints (sample rate, stream duration, quotas, etc.) are documented in [`google_stt_v1_constraints.md`](google_stt_v1_constraints.md).


## Setup

1. Enable the **Cloud Speech-to-Text API** in your GCP project.
2. Create a service account with the `Cloud Speech Client` role.
3. Download the JSON key and set the environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```


## Config

### `ASRConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `language_code` | `str` | `"en-US"` | BCP-47 language code |
| `model` | `str` | `"latest_long"` | Recognition model |
| `interim_results` | `bool` | `True` | Return interim transcripts during streaming |

### `AudioConfig` (shared across all audio-consuming modules)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sample_rate` | `int` | `16000` | Sample rate in Hz (API: 8000–48000) |
| `channels` | `int` | `1` | Channel count (pipeline uses mono) |
| `sample_width` | `int` | `2` | Bytes per sample (pipeline uses 16-bit PCM / LINEAR16) |
| `frame_duration_ms` | `int` | `30` | Frame size for `feed_audio()` |


## Usage

### Methods

| Method | Description |
|--------|-------------|
| `start()` | Creates a gRPC client and streaming session. No-op if already running. |
| `feed_audio(frame)` | Enqueues one audio frame. Drops the frame with a warning if the queue is full. |
| `get_text()` | Returns the current recognized text. Non-blocking. |
| `reset()` | Ends the current stream, clears the transcript, and starts a new stream. |
| `stop()` | Tears down the stream and gRPC client. Safe to call twice. |

### Transcript accumulation

`get_text()` accumulates finalized results (`is_final`) and appends the current interim result after them.

```
Response sequence                  get_text() return value
─────────────────                  ──────────────────────
interim "hello"                    → "hello"
interim "hello how"                → "hello how"
final   "hello, how "             → "hello, how "
interim "are"                      → "hello, how are"
final   "are you doing?"           → "hello, how are you doing?"
```

Calling `reset()` clears all accumulated text.

### Internal queue

`feed_audio()` places frames into an internal queue (max 300 frames, ~9s at 30ms). When the queue is full, frames are dropped with a warning log. This is not a problem as long as the caller feeds frames at approximately real-time rate.

### gRPC stream lifetime

Google STT imposes a ~5 minute limit per streaming session. This module does not auto-restart; the caller must call `reset()` between turns to create a new session.


## Testing

### Unit tests (mocked)

```bash
uv run pytest voice_pipeline/tests/asr/test_asr.py -v
```

26 tests with mocked `SpeechClient` — no API credentials needed.

### Integration & stress tests (real API)

```bash
GOOGLE_APPLICATION_CREDENTIALS=creds.json \
ASR_TEST_WAV=/path/to/speech.wav \
ASR_TEST_LANG=ko-KR \
uv run pytest -m requires_api voice_pipeline/tests/asr/ -v
```

### Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_APPLICATION_CREDENTIALS` | Yes | — | Path to GCP service account JSON |
| `ASR_TEST_WAV` | Yes | — | Path to a speech WAV file |
| `ASR_TEST_LANG` | No | `en-US` | Language of the speech in the WAV file |

WAV files that are not mono/16-bit or outside 8000–48000 Hz are automatically converted via `ffmpeg`.
