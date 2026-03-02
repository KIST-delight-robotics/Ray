# CppBridge Module

WebSocket bridge between the Python voice pipeline and the C++ audio playback process.

## Usage

```python
from voice_pipeline.bridge import CppBridge
from voice_pipeline.core.config import CppBridgeConfig

config = CppBridgeConfig(host="localhost", port=8765)
bridge = CppBridge(config)

bridge.connect()

# Send audio for playback
bridge.send_audio(pcm_bytes)

# Send control commands
bridge.send_stop()
bridge.send_greeting()
bridge.send_farewell()

# Poll for events (non-blocking)
event = bridge.poll_event()
if event is not None:
    print(event.event_type, event.position_sec)

bridge.disconnect()
```

## Message Protocol

All messages are JSON text frames over WebSocket.

### Python to C++

| Message | Format |
|---------|--------|
| Audio | `{"type": "audio", "data": "<base64-pcm>"}` |
| Stop | `{"type": "stop"}` |
| Greeting | `{"type": "greeting"}` |
| Farewell | `{"type": "farewell"}` |

### C++ to Python

| Message | Format |
|---------|--------|
| Playback started | `{"type": "playback_started"}` |
| Playback position | `{"type": "playback_position", "position_sec": 1.23}` |
| Playback complete | `{"type": "playback_complete"}` |
| Playback stopped | `{"type": "playback_stopped", "position_sec": 4.56}` |

## Configuration

| Field | Default | Description |
|-------|---------|-------------|
| `host` | `localhost` | WebSocket server host |
| `port` | `8765` | WebSocket server port |
| `reconnect_attempts` | `3` | Max connection retries |
| `recv_timeout_sec` | `1.0` | Receiver loop poll interval |
| `connect_timeout_sec` | `5.0` | WebSocket handshake timeout |
| `close_timeout_sec` | `5.0` | WebSocket close handshake timeout |

## Threading Model

- `connect()`, `disconnect()`, `send_*()`, `poll_event()` are called from the orchestrator thread only.
- A daemon receiver thread reads WebSocket messages and enqueues parsed `CppEvent` objects.
- Error propagation: connection loss is stored and raised on the next orchestrator call.

## Error Handling

`BridgeError` is raised for:
- Calling `send_*()` before `connect()`
- Connection failure after all retries exhausted
- Connection lost during send or receive
