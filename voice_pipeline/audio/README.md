# Audio Module

Audio capture and wakeword detection for the voice pipeline.

## Audio Capture

`AudioInput` captures PCM audio from a microphone via PyAudio on a daemon
thread, pushing frames to an injected queue.

### Usage

```python
from voice_pipeline.audio import AudioInput  # (or audio.audio_input)

audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=300)
ai = AudioInput(audio_queue)
ai.start()
# ... consume frames from audio_queue ...
ai.stop()
```

### `AudioInput.__init__` 인자

| 인자 | Default | 의미 |
|------|---------|------|
| `audio_queue` | (필수) | 캡처된 PCM 프레임을 push할 공유 큐 (`queue.Queue[AudioFrame]`). |

### 클래스 변수

| 변수 | 값 | 의미 |
|------|------|------|
| `_THREAD_JOIN_TIMEOUT_SEC` | `2.0` | 캡처 스레드 종료 대기 시간 (초) |
| `_DEVICE_INDEX` | `None` | PyAudio 입력 디바이스 인덱스. `None`은 시스템 기본 장치 |
| `_CAPTURE_CHANNELS` | `None` | 캡처 채널 수. `None`은 mono (`audio.constants.CHANNELS`). ReSpeaker 6ch 펌웨어는 `6` |
| `_EXTRACT_CHANNEL` | `0` | 다중 채널 캡처 시 mono 추출에 사용할 채널 인덱스 |

샘플레이트·sample width·frame size는 `voice_pipeline/audio/constants.py`에서 직접 참조한다.

## Wakeword Detection

`WakewordDetector` detects trigger words in audio using Silero VAD for speech
segmentation and Google Cloud STT for keyword matching.

### Architecture

```
feed_audio(frame) → bool
  │
  ├── 1. Rechunk: 480-sample frames → 512-sample VAD chunks
  ├── 2. Silero VAD: speech probability per chunk
  ├── 3. State machine: IDLE → SPEECH → TRAILING → recognition
  └── 4. Google STT recognize() + word-boundary keyword match
```

### Usage

```python
from voice_pipeline.audio import WakewordDetector

detector = WakewordDetector(language_code="en-US")

# In your audio loop:
for frame in audio_frames:
    if detector.feed_audio(frame):
        print("Wakeword detected!")
```

### `WakewordDetector.__init__` 인자

| 인자 | Default | 의미 |
|------|---------|------|
| `language_code` | `"en-US"` | Google STT BCP-47 언어 코드 |

### 클래스 변수

`WakewordDetector` 클래스 내부 상수.

| 변수 | 값 | 의미 |
|------|------|------|
| `_KEYWORDS` | `("ray",)` | 감지할 트리거 단어 목록 |
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

샘플레이트·채널 수·sample_width는 `voice_pipeline/audio/constants.py`에서 직접 참조한다.

### Dependencies

- **silero-vad**: Silero VAD model (~2MB JIT). Requires PyTorch.
- **google-cloud-speech**: Google Cloud STT for keyword recognition.

### Error Handling

- **Initialization failures** (model load, client creation): raise `WakewordError`.
- **Runtime STT errors** (network, timeout): log warning, return `False` (fail closed).

### Testing

```bash
# Unit tests (mocked VAD + STT)
uv run pytest voice_pipeline/tests/audio/test_wakeword.py -v

# Integration tests (real models + API)
WAKEWORD_TEST_WAV=path/to/wakeword.wav \
  uv run pytest voice_pipeline/tests/audio/test_wakeword_integration.py -v -m "requires_api and requires_model"
```
