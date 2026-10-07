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
| Stream start | `{"type": "stream_start"}` — 선택 필드 `"live": true` 는 GPT-Live 스트림 표시: C++ 는 프리버퍼와 추가 시작 여유가 찰 때까지 시계를 시작하지 않는다(값은 `cpp/main.cpp`의 live 상수 참조). 필드가 없으면 기존 TTS 스트림 동작. |
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

## Live 실행과 싱크 진단

로봇에서는 두 터미널에서 각각 `build/Ray`, `uv run ray`를 실행한다. 기기별 모터 값은
`config/robot.toml`을 사용한다. “레이”/“Ray”로 깨우면 인사 WAV와 live 연결을 시작한다.
GNOME 소리 설정 창의 마이크 점유로 입력 장치를 열지 못한 사례가 있으므로 진단 중에는 창을 닫는다.

- `var/log/pipeline/<시각>.log`: `GPT-Live connected in`, `LiveSessionLoop started`, 전사,
  `Audio chunk gap`(300 ms 이상, 발화·무음과 단계 포함), `Ending session`, 종료 시 `Audio summary`.
  `Audio lead`는 `voice_pipeline.live_session` 로거의 DEBUG 수준에서 15초마다 기록된다.
- `var/log/motion/<시각>/console.log`: C++ 콘솔 복제. `[split] padded/trimmed`와 `[Sound] underrun`을
  파이프라인의 조각 간격 로그와 비교한다.
- 같은 디렉터리의 `Standard_Log.csv`: 모터 틱(정상 40 ms, 대기 모드는 약 1 Hz 기록).
  `audio_sync_RESPONSES.csv`: `kEnableAudioSyncLog`가 켜져 있을 때 기록하는 싱크 진단.

```bash
uv run python scripts/hardware/audio_sync_report.py var/log/motion/<시각>/audio_sync_RESPONSES.csv
```

싱크 방어 확인 시 재생기 자체 무음(`[Sound] underrun`)이 없는지, 큰 정지의 split 채움 뒤
잉여 무음에서 trim이 일어나는지, CSV의 lag와 모터 틱 간격이 유지되는지 확인한다.
유실로 잉여가 남지 않은 경우에는 채운 양 전부를 되돌릴 수 없다.
실험별 수치와 해석은 [GPT-Live 실측 기록](../../scripts/gpt_live/FINDINGS.md#2-오디오-출력-특성)을 참고한다.
물리 싱크 측정은 `scripts/bench/live_sync_probe.py`, 네트워크 정지 구간 비교는
`scripts/hardware/net_probe.py`를 사용한다(각 도구의 `--help` 참고).
