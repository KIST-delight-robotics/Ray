# CppBridge Module

WebSocket bridge between the Python voice pipeline and the C++ audio playback process.
C++ runs a WebSocket server; Python connects as a client.

## Usage

```python
from voice_pipeline.bridge import CppBridge

bridge = CppBridge()

bridge.connect()

# Streaming TTS audio
bridge.send_stream_start()
bridge.send_audio(pcm_bytes)
bridge.send_audio_end()

# File playback (greeting/farewell)
bridge.send_play_file("assets/audio/awake.wav")

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
| Stream start | `{"type": "stream_start"}` |
| Audio | `{"type": "audio", "data": "<base64-pcm>"}` |
| Audio end | `{"type": "audio_end"}` |
| Stop | `{"type": "stop"}` |
| Play file | `{"type": "play_file", "file_path": "path/to/file.wav"}` |

### C++ → Python

| Message | Format |
|---------|--------|
| Playback started | `{"type": "playback_started"}` |
| Playback complete | `{"type": "playback_complete"}` |

`playback_complete` is sent for both normal completion and after a `stop` interrupt.
Python distinguishes the two by tracking whether it sent `stop` (STOP_PENDING state).

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

`BridgeError` is raised for:
- Calling `send_*()` before `connect()`
- Connection failure after all retries exhausted
- Connection lost during send or receive
