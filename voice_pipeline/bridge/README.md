# CppBridge Module

WebSocket bridge between the Python voice pipeline and the C++ audio playback process.
C++ runs a WebSocket server; Python connects as a client.

## Usage

```python
from voice_pipeline.bridge import CppBridge
from voice_pipeline.core.config import CppBridgeConfig

config = CppBridgeConfig(host="localhost", port=8765)
bridge = CppBridge(config)

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

## Configuration

| Field | Default | Description |
|-------|---------|-------------|
| `host` | `localhost` | WebSocket server host |
| `port` | `8765` | WebSocket server port |
| `reconnect_attempts` | `3` | Max connection retries |
| `recv_timeout_sec` | `1.0` | Receiver loop poll interval |
| `connect_timeout_sec` | `5.0` | WebSocket handshake timeout |
| `close_timeout_sec` | `5.0` | WebSocket close handshake timeout |

## Remote Deployment

C++ (RPi)와 Python (PC)을 분리 실행할 수 있다. C++ 서버가 `0.0.0.0`으로 바인드하므로 `host`만 변경하면 된다.

```python
config = CppBridgeConfig(host="192.168.x.x", port=8765)  # RPi IP
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
