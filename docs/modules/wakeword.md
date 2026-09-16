# Wakeword Module

Wakeword detection using Silero VAD for speech segmentation and Google Cloud STT for keyword matching.

## Architecture

```
feed_audio(frame) → bool
  │
  ├── 1. Rechunk: 480-sample frames → 512-sample VAD chunks
  ├── 2. Silero VAD: speech probability per chunk
  ├── 3. State machine: IDLE → SPEECH → TRAILING → recognition
  └── 4. Google STT recognize() (ko-KR + en-US) + keyword match (영문 단어 경계 / 한글 부분 문자열)
```

## Usage

```python
from voice_pipeline.adapters.wakeword import WakewordDetector

detector = WakewordDetector()

# In your audio loop:
for frame in audio_frames:
    if detector.feed_audio(frame):
        print("Wakeword detected!")
```

## `WakewordDetector.__init__` 인자

| 인자 | Default | 의미 |
|------|---------|------|
| `language_code` | `"ko-KR"` | Google STT 주 언어 (BCP-47). ko-KR 주 + en-US 대안이 한·영 모두 가장 잘 잡았다 |

## 클래스 변수

`WakewordDetector` 클래스 내부 상수.

| 변수 | 값 | 의미 |
|------|------|------|
| `_KEYWORDS` | `("ray", "레이")` | 감지할 트리거 단어 목록. 영문은 단어 경계, 한글은 부분 문자열 매칭(레이야) |
| `_ALTERNATIVE_LANGUAGE_CODES` | `("ko-KR", "en-US")` | 주 언어와 함께 인식할 언어. 주 언어는 제외해 보낸다 (최대 3) |
| `_VAD_CHUNK_SAMPLES` | `512` | VAD 입력 청크 샘플 수 |
| `_VAD_CHUNK_BYTES` | `1024` | 파생: 청크 바이트 수 (16-bit mono) |
| `_VAD_CHUNK_DURATION_MS` | `32` | 파생: 청크 길이 (512 @ 16kHz) |
| `_VAD_THRESHOLD` | `0.5` | VAD 음성 확률 임계값 |
| `_MAX_SPEECH_DURATION_SEC` | `3.0` | 이 시간 초과 시 강제 STT 인식 |
| `_PRE_BUFFER_MS` | `300` | 음성 시작 onset 캡처용 ring buffer 길이 (ms) |
| `_SPEECH_PAD_MS` | `300` | 음성 종료 검출용 후행 침묵 길이 (ms) |
| `_MIN_SPEECH_DURATION_MS` | `100` | 이 시간 미만 음성은 STT 스킵 (ms) |
| `_STT_TIMEOUT_SEC` | `5.0` | Google STT recognize() 응답 대기 시간 (초) |
| `_MAX_ALTERNATIVES` | `5` | STT 응답에 요청할 대안 수 |

샘플레이트·채널 수·sample_width는 `voice_pipeline/settings.py`에서 직접 참조한다.

## Dependencies

- **silero-vad**: Silero VAD model (~2MB JIT). Requires PyTorch.
- **google-cloud-speech**: Google Cloud STT for keyword recognition.

## Error Handling

- **Initialization failures** (model load, client creation): raise `RuntimeError`.
- **Runtime STT errors** (network, timeout): log warning, return `False` (fail closed).

## Testing

```bash
# Unit tests (mocked VAD + STT)
uv run pytest voice_pipeline/tests/adapters/test_wakeword.py -v

# Integration tests (real models + API)
WAKEWORD_TEST_WAV=path/to/wakeword.wav \
  uv run pytest voice_pipeline/tests/adapters/test_wakeword_integration.py -v -m "requires_api and requires_model"
```
