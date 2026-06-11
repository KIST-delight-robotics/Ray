# TTS Module

Streaming text-to-speech. `ITTS` 인터페이스 뒤에 두 vendor 구현:

- **`ElevenLabsTTS`** (기본) — ElevenLabs `stream/with-timestamps` API. word timestamps 지원.
- **`OpenAITTS`** — OpenAI Audio API. timestamps 미지원.

vendor 선택은 `create_tts()` 팩토리 — 기본 vendor는 `factory.py`의 `_DEFAULT_VENDOR`.

OpenAI API constraints (models, rate limits, PCM format, etc.) are documented in [`openai_tts_api_reference.md`](openai_tts_api_reference.md).


## Setup

사용하는 vendor의 API 키 환경변수 설정:

```bash
export ELEVENLABS_API_KEY=...   # ElevenLabsTTS (기본 vendor)
export OPENAI_API_KEY=sk-...    # OpenAITTS
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

### `ElevenLabsTTS`

| 변수 | 값 | 의미 |
|------|------|------|
| `OUTPUT_SAMPLE_RATE` | `24000` | `pcm_24000` 출력 샘플레이트 (Hz). OpenAITTS와 동일 |
| `_VOICE_ID` | `"EXAVITQu4vr4xnSDxMaL"` | ElevenLabs voice ID (Sarah — 임시 영어 default voice) |
| `_MODEL` | `"eleven_flash_v2_5"` | 최저 지연 모델 (`eleven_turbo_v2_5`, `eleven_multilingual_v2` 등) |
| `_OUTPUT_FORMAT` | `"pcm_24000"` | 출력 포맷 (tier 제한 없음; `pcm_44100`은 Pro 전용) |
| `_VOICE_SETTINGS` | `None` | voice 세부 설정 dict (`stability`, `similarity_boost`, `style`, `speed`) |
| `_MAX_RETRIES` | `2` | 자동 재시도 횟수 (스트리밍 시작 전 한정) |
| `_TIMEOUT_SEC` | `10.0` | httpx timeout (SDK 기본 240s override) |
| `_MAX_TEXT_LEN` | `4096` | 입력 길이 상한 |

`voice_id`는 음성 식별자 (vendor + 설정), 캐시 무효화 등에 사용.


## Usage

### Vendor selection

```python
from voice_pipeline.tts import create_tts

tts = create_tts()          # 기본 vendor (elevenlabs)
tts = create_tts("openai")  # 명시적 선택
```

### Streaming synthesis

`synthesize()` returns a `TTSStream` that yields PCM audio chunks (24 kHz, 16-bit signed LE, mono).

```python
stream = tts.synthesize("Hello world")

for chunk in stream:
    # Send chunk to audio output / C++ bridge
    pass

# After full iteration, audio and timestamps are available
audio = stream.audio            # bytes: full PCM audio
timestamps = stream.timestamps  # ElevenLabs: word timestamps / OpenAI: ()
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

### Save to file

Use `synthesize_to_wav()` to collect PCM from `synthesize()` and write a WAV file:

```python
from voice_pipeline.tts.greeting_audio import synthesize_to_wav

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

두 vendor 모두 동일 (OpenAI `response_format="pcm"`, ElevenLabs `output_format="pcm_24000"`):

| Property | Value |
|----------|-------|
| Sample rate | 24 kHz |
| Bit depth | 16-bit signed |
| Byte order | Little-endian |
| Channels | Mono |
| Header | None (raw PCM) |


## Word timestamps

| Vendor | 지원 | barge-in 절단 경로 |
|--------|------|--------------------|
| `ElevenLabsTTS` | O | `truncate_by_timestamps` (정밀) |
| `OpenAITTS` | X (`()` 반환) | `truncate_by_ratio` (duration 비율 추정 fallback) |

ElevenLabs는 character alignment를 공백 기준 word로 집계해 제공. timestamps는 스트림 완전 소비 후에만 접근 가능 (`TTSStream` 계약). 집계는 best-effort — alignment 이상(길이 불일치, 시간 역전)은 절단/clamp 처리하며 예외를 던지지 않음.


## Greeting/Farewell Audio

`greeting_audio.py` 모듈은 greeting·farewell 오디오를 TTS로 사전 합성한다. 현재 TTS 설정(`voice_id` 기반 해시)이 바뀌면 자동 재합성. 합성 실패 시 fallback 파일로 복구.

### `ensure_greeting_audio` 인자

| 인자 | Default | 의미 |
|------|---------|------|
| `tts` | — | 합성용 `ITTS` 인스턴스 (필수) |

### 모듈 상수

| 상수 | 값 | 의미 |
|------|------|------|
| `_AUDIO_DIR` | `"assets/audio"` | 생성 오디오 파일 저장 디렉토리 (C++ 작업 경로 기준) |
| `_GREETING_TEXT` | `"Yes, how can I help you?"` | greeting 합성 텍스트 |
| `_FAREWELL_TEXT` | `"Talk to you next time!"` | farewell 합성 텍스트 |
| `_FALLBACK_GREETING_PATH` | `"assets/audio/greeting.wav"` | TTS 실패 시 greeting fallback 파일 |
| `_FALLBACK_FAREWELL_PATH` | `"assets/audio/farewell.wav"` | TTS 실패 시 farewell fallback 파일 |


## Testing

### Unit tests (mocked)

```bash
uv run pytest voice_pipeline/tests/tts -v
```

Mocked vendor clients — no API credentials needed.

### Integration & stress tests (real API)

```bash
OPENAI_API_KEY=sk-... ELEVENLABS_API_KEY=... uv run pytest -m requires_api voice_pipeline/tests/tts/ -v
```

키가 없는 vendor의 테스트는 자동 skip.

### Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | openai 사용/테스트 시 | — | OpenAI API key |
| `ELEVENLABS_API_KEY` | elevenlabs 사용/테스트 시 | — | ElevenLabs API key |
