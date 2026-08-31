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


## `GoogleCloudASR.__init__` 인자

| 인자 | Default | 의미 |
|------|---------|------|
| `language_code` | `"en-US"` | BCP-47 언어 코드 |

## 클래스 변수

`GoogleCloudASR` 클래스 내부 상수.

| 변수 | 값 | 의미 |
|------|------|------|
| `_MODEL` | `"latest_long"` | Google STT 모델 (장시간 음성 인식) |
| `_QUEUE_MAXSIZE` | `300` | 오디오 큐 최대 프레임 수 (~9초 @ 30ms 프레임) |
| `_QUEUE_GET_TIMEOUT_SEC` | `1.0` | 오디오 큐 poll 간격 (초) |
| `_THREAD_JOIN_TIMEOUT_SEC` | `5.0` | reader 스레드 종료 대기 (초) |
| `_SENTINEL` | `b""` | 스트림 종료 신호 |
| `_ENCODING_MAP` | dict | sample_width → AudioEncoding 매핑 |

샘플레이트·채널 수·sample_width는 `voice_pipeline/settings.py`에서 직접 참조한다.


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
uv run pytest voice_pipeline/tests/adapters/test_asr.py -v
```

26 tests with mocked `SpeechClient` — no API credentials needed.

### Integration & stress tests (real API)

```bash
GOOGLE_APPLICATION_CREDENTIALS=creds.json \
ASR_TEST_WAV=/path/to/speech.wav \
ASR_TEST_LANG=ko-KR \
uv run pytest -m requires_api voice_pipeline/tests/adapters/test_asr_integration.py voice_pipeline/tests/adapters/test_asr_stress.py -v
```

### Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_APPLICATION_CREDENTIALS` | Yes | — | Path to GCP service account JSON |
| `ASR_TEST_WAV` | Yes | — | Path to a speech WAV file |
| `ASR_TEST_LANG` | No | `en-US` | Language of the speech in the WAV file |

WAV files that are not mono/16-bit or outside 8000–48000 Hz are automatically converted via `ffmpeg`.
