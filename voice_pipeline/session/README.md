# Session Module

최상위 상태 머신 `SessionManager`.

SLEEP → GREETING → ACTIVE → FAREWELL → SLEEP 전이를 관리하며,
세션 진입마다 `session_factory`로 per-session 컴포넌트(Orchestrator,
ConversationHistory, session_id)를 새로 생성한다.

## Usage

```python
from voice_pipeline.session.session_manager import SessionComponents, SessionManager


def session_factory() -> SessionComponents:
    # Orchestrator + ConversationHistory + session_id 생성
    ...
    return SessionComponents(orchestrator=orch, history=hist, session_id=sid)


sm = SessionManager(
    audio_input=audio_input,
    wakeword=wakeword,
    session_factory=session_factory,
    cpp_bridge=bridge,
    led=led,
    greeting_audio_path="assets/audio/greeting.wav",
    farewell_audio_path="assets/audio/farewell.wav",
    audio_queue=audio_queue,
    on_session_end=on_session_end_cb,
)
sm.run()  # blocks until shutdown()
```

## `SessionManager.__init__` 인자

| 인자 | Default | 의미 |
|------|---------|------|
| `audio_input` | (필수) | 마이크 캡처 스레드 (`IAudioInput`). |
| `wakeword` | (필수) | 웨이크워드 감지기 (`IWakewordDetector`). |
| `session_factory` | (필수) | 세션 진입마다 `SessionComponents`를 생성하는 팩토리. |
| `cpp_bridge` | (필수) | C++ 오디오 재생 브릿지 (`ICppBridge`). |
| `led` | (필수) | LED 컨트롤러 (`ILEDController`). |
| `greeting_audio_path` | (필수) | greeting 오디오 파일 경로. |
| `farewell_audio_path` | (필수) | farewell 오디오 파일 경로. |
| `audio_queue` | `None` | AudioInput과 공유하는 프레임 큐. `None`이면 `AUDIO_QUEUE_SIZE` 크기로 내부 생성. |
| `on_session_end` | `None` | 세션 종료 콜백 `(session_id, started_at) -> None`. |

## 클래스 변수

| 변수 | 값 | 의미 |
|------|------|------|
| `AUDIO_QUEUE_SIZE` | `300` | 오디오 프레임 공유 큐 최대 크기 (frame 단위) |
| `_GREETING_TIMEOUT_SEC` | `10.0` | greeting 재생 완료 최대 대기 시간 (초) |
| `_FAREWELL_TIMEOUT_SEC` | `10.0` | farewell 재생 완료 최대 대기 시간 (초) |
| `_FRAME_TIMEOUT_SEC` | `0.1` | SLEEP 모드 프레임 대기 timeout (초) |
| `_POLL_INTERVAL_SEC` | `0.05` | greeting/farewell 재생 대기 polling 주기 (초) |

## Testing

```bash
uv run pytest voice_pipeline/tests/session/
```

세션 전이는 인라인 `SessionManager._run_*()` 메서드로 구현되어 있어
단위 테스트에서 `monkeypatch.setattr(SessionManager, "_GREETING_TIMEOUT_SEC", 0.01)`
식으로 타임아웃을 단축해 빠르게 검증한다.
