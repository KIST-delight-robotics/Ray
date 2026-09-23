# CppBridge Module

WebSocket bridge between the Python voice pipeline and the C++ audio playback process.
C++ runs a WebSocket server; Python connects as a client.

## Usage

```python
from voice_pipeline.adapters.cpp_bridge import CppBridge

bridge = CppBridge()

bridge.connect()

# Streaming TTS audio
bridge.send_stream_start()
bridge.send_audio(pcm_bytes)
bridge.send_audio_end()

# File playback (greeting/farewell)
bridge.send_play_file("assets/audio/awake.wav")

# Stored song with motion CSVs (assets/audio/music/<name>.wav + assets/{head,mouth,led}Motion/)
# 열려 있는 스트림은 send_audio_end() 로 먼저 닫아야 한다 — C++ 는 재생 하나를 끝까지 처리한다
bridge.send_play_audio_csv("IAM")

# Interrupt playback (barge-in)
bridge.send_stop()

# Poll for events (non-blocking)
event = bridge.poll_event()
if event is not None:
    print(event.event_type)

bridge.disconnect()
```

## Message Protocol

All messages are JSON text frames over WebSocket.

### Python → C++

| Message | Format |
|---------|--------|
| Stream start | `{"type": "stream_start"}` — 선택 필드 `"live": true` 는 GPT-Live 스트림 표시: C++ 는 두 덩이(720 ms)가 찰 때까지 시계를 시작하지 않고(실시간 유입 여유), 오디오 기반 헤드모션 생성 대신 대기 모션을 유지한다. 필드가 없으면 기존 TTS 스트림 동작. |
| Audio | `{"type": "audio", "data": "<base64-pcm>"}` |
| Audio end | `{"type": "audio_end"}` |
| Stop | `{"type": "stop"}` |
| Play file | `{"type": "play_file", "file_path": "path/to/file.wav"}` |
| Play song (CSV) | `{"type": "play_audio_csv", "audio_name": "IAM"}` — `assets/audio/music/IAM.wav` 와 `assets/headMotion/IAM.csv`, `assets/mouthMotion/IAM-delta-big.csv`, `assets/ledMotion/IAM-led.csv` 를 묶어 재생 (`cpp/main.cpp` `csv_control_motor`). 끝나거나 `stop` 으로 끊기면 `playback_complete`. |

### C++ → Python

| Message | Format |
|---------|--------|
| Playback started | `{"type": "playback_started"}` |
| Playback complete | `{"type": "playback_complete"}` |

`playback_complete` is sent for both normal completion and after a `stop` interrupt.
Python distinguishes the two by tracking whether it sent `stop` (STOP_PENDING state).

C++ 메인 루프는 메시지 하나(스트림·파일·노래)를 끝까지 재생한 뒤 다음 메시지를 읽는다. 그래서 GPT-Live
세션(스트림 하나로 시작)이 노래를 틀 때는 `audio_end` → `playback_complete` → `play_audio_csv` →
`playback_complete` → `stream_start` 순으로 교대한다 (`engines/gpt_live/loop.py` "노래 재생").

## 클래스 변수

`CppBridge` 클래스 내부 상수.

| 변수 | 값 | 의미 |
|------|------|------|
| `_HOST` | `"localhost"` | C++ 프로세스 호스트 주소 |
| `_PORT` | `9200` | C++ 프로세스 WebSocket 포트 |
| `_RECONNECT_ATTEMPTS` | `3` | 연결 실패 시 재시도 횟수 |
| `_RECV_TIMEOUT_SEC` | `1.0` | 메시지 수신 polling 간격 (초) |
| `_CONNECT_TIMEOUT_SEC` | `5.0` | 연결 수립 최대 대기 시간 (초) |
| `_CLOSE_TIMEOUT_SEC` | `5.0` | 연결 종료 최대 대기 시간 (초) |
| `_RECONNECT_DELAY_SEC` | `1.0` | 연결 재시도 사이 대기 시간 (초) |
| `_THREAD_JOIN_TIMEOUT_SEC` | `5.0` | 수신 스레드 종료 대기 시간 (초) |

## Remote Deployment

C++ (RPi)와 Python (PC)을 분리 실행할 수 있다. C++ 서버가 `0.0.0.0`으로 바인드하므로 `_HOST`만 변경하면 된다.

```python
CppBridge._HOST = "192.168.x.x"  # RPi IP
bridge = CppBridge()
```

연결 확인: `scripts/test_ws_connection.py --host 192.168.x.x`

## Threading Model

- `connect()`, `disconnect()`, `send_*()`, `poll_event()` are called from the orchestrator thread only.
- A daemon receiver thread reads WebSocket messages and enqueues parsed `CppEvent` objects.
- Error propagation: connection loss is stored and raised on the next orchestrator call.

## Error Handling

`RuntimeError` is raised for:
- Calling `send_*()` before `connect()`
- Connection failure after all retries exhausted
- Connection lost during send or receive
