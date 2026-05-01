# Audio Module

Audio capture for the voice pipeline.

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
