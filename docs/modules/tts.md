# TTS Module

Streaming text-to-speech. `ITTS` 인터페이스 뒤의 구현은 **`OpenAITTS`** (OpenAI Audio API) 하나다.
대화 음성은 GPT-Live 모델이 직접 내므로, TTS 는 인사 WAV 사전 생성(`greeting_audio.py`)과 스크립트에서만 쓴다.


## Setup

```bash
export OPENAI_API_KEY=sk-...
```


## 클래스 변수

### `OpenAITTS`

| 변수 | 값 | 의미 |
|------|------|------|
| `OUTPUT_SAMPLE_RATE` | `24000` | OpenAI TTS API 고정 출력 샘플레이트 (Hz). 외부 참조용 공개 |
| `_VOICE` | `"ash"` | OpenAI 음성 프리셋 |
| `_MODEL` | `"tts-1"` | OpenAI TTS 모델 (`tts-1`, `tts-1-hd`, `gpt-4o-mini-tts`) |
| `_SPEED` | `1.0` | 재생 속도 (0.25~4.0) |
| `_INSTRUCTIONS` | `None` | 음성 스타일 지시문 (`gpt-4o-mini-tts` 전용) |
| `_SUPPORTS_INSTRUCTIONS` | `{"gpt-4o-mini-tts"}` | `instructions` 인자 지원 모델 |
| `_MAX_RETRIES` | `2` | 합성 실패 시 자동 재시도 횟수 |
| `_TIMEOUT_SEC` | `30.0` | 합성 응답 대기 최대 시간 (초) |
| `_CHUNK_SIZE` | `4096` | 스트리밍 오디오 버퍼 크기 (바이트) |

`voice_id`는 음성 식별자 (vendor + 설정), 캐시 무효화 등에 사용.


## Usage

### Streaming synthesis

`synthesize()` returns a `TTSStream` that yields PCM audio chunks (24 kHz, 16-bit signed LE, mono).

```python
stream = tts.synthesize("Hello world")

for chunk in stream:
    # Send chunk to audio output / C++ bridge
    pass

# After full iteration, audio is available
audio = stream.audio            # bytes: full PCM audio
```

### Save to file

Use `synthesize_to_wav()` to collect PCM from `synthesize()` and write a WAV file:

```python
from voice_pipeline.greeting_audio import synthesize_to_wav

synthesize_to_wav(tts, "Hello world", Path("output.wav"))
```

### Model-specific instructions

The `instructions` parameter is only supported by `gpt-4o-mini-tts`. For other models it is ignored with a warning log.

```python
OpenAITTS._MODEL = "gpt-4o-mini-tts"
OpenAITTS._INSTRUCTIONS = "Speak in a cheerful tone."
tts = OpenAITTS()
```


## PCM output format

OpenAI `response_format="pcm"`:

| Property | Value |
|----------|-------|
| Sample rate | 24 kHz |
| Bit depth | 16-bit signed |
| Byte order | Little-endian |
| Channels | Mono |
| Header | None (raw PCM) |


## Greeting Audio

`greeting_audio.py` 모듈은 인사 오디오를 TTS로 사전 합성한다. 목소리 설정(`voice_id`)이나 문구가 바뀌면 해시가
달라져 자동 재합성. 합성 실패 시 fallback 파일로 복구.

### `ensure_greeting_audio` 인자

| 인자 | Default | 의미 |
|------|---------|------|
| `tts` | — | 합성용 `ITTS` 인스턴스 |
| `text` | — | 인사 문구 |

### 모듈 상수

| 상수 | 값 | 의미 |
|------|------|------|
| `_AUDIO_DIR` | `"var/audio"` | TTS 로 생성한 인사 WAV 캐시 (C++ 작업 경로 기준) |
| `_FALLBACK_GREETING_PATH` | `"assets/audio/greeting.wav"` | TTS 실패 시 greeting fallback 파일 |


## Testing

### Unit tests (mocked)

```bash
uv run pytest voice_pipeline/tests/adapters/test_tts_openai.py -v
```

Mocked client — no API credentials needed.

### Integration & stress tests (real API)

```bash
OPENAI_API_KEY=sk-... uv run pytest -m requires_api voice_pipeline/tests/adapters/ -k tts -v
```

키가 없으면 자동 skip.

### Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | 사용/테스트 시 | — | OpenAI API key |
