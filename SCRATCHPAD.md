# Scratchpad

Claude's working memory. Free-form notes, observations, and context carried across sessions.

## Current Status

Phase 6 complete. C++ code imported from RAG_test branch.

Next: C++ ↔ Python WebSocket 프로토콜 정렬 (아래 참조), 그리고 Phase 7 — Integration tests.

## C++ ↔ Python 브릿지 프로토콜 정렬 작업

### 배경
- C++ 코드(`cpp/main.cpp`)를 `RAG_test` 브랜치에서 가져옴 (커밋 `32e6096`)
- 기존 C++ 프로토콜과 새 Python 파이프라인(`voice_pipeline/bridge/cpp_bridge.py`) 프로토콜이 다름
- C++ 수정을 최소화하면서 양쪽 맞추기

### 합의된 프로토콜

**Python → C++:**

| 메시지 | 필드 | 설명 | C++ 기존 대응 |
|--------|------|------|---------------|
| `stream_start` | - | 스트리밍 준비 | `responses_only` |
| `audio` | `data` (base64 PCM) | 오디오 청크 | `responses_audio_chunk` |
| `audio_end` | - | 스트림 종료 | `responses_stream_end` |
| `stop` | - | 재생 중단 (barge-in) | `user_interruption` |
| `greeting` | - | 인사 재생 | `play_audio` (파일경로 방식) |
| `farewell` | - | 작별 재생 | `play_audio` (파일경로 방식) |

**C++ → Python:**

| 메시지 | 필드 | 설명 | C++ 기존 대응 |
|--------|------|------|---------------|
| `playback_started` | - | 재생 시작 (VAP 타이밍용) | (신규) |
| `playback_complete` | - | 정상 재생 완료 | `speaking_finished` |
| `playback_stopped` | - | 중단 완료 (stop 응답) | (신규) |

### Python 인터페이스 수정 필요 사항

`ICppBridge`에 추가할 메서드:
- `send_stream_start()` — Orchestrator가 오디오 드레인 시작 전 호출
- `send_audio_end()` — Orchestrator가 오디오 드레인 완료 후 호출

`CppBridge` 구현에 해당 JSON 전송 추가.

Orchestrator `_drain_audio_to_bridge()` 흐름:
```
send_stream_start() → send_audio(chunk) × N → send_audio_end()
```

### C++ 수정 사항

1. **WebSocket: 클라이언트 → 서버** (`ix::WebSocket` → `ix::WebSocketServer`, 포트 8765)
2. **onMessageCallback**: type 문자열 매핑 교체 (위 표 참조)
3. **robot_main_loop**: 응답 메시지 이름 변경 + `playback_started`/`playback_stopped` 전송 추가
4. **stop 처리**: `stop` 수신 → 재생 중단 → 큐 비우기 → `playback_stopped` 전송
5. **greeting/farewell**: 기존 `play_audio` 로직 재활용, 고정 파일경로 (`assets/audio/awake.wav`, `sleep.wav`)

### 삭제 가능 항목
- `turn_id` 관련 로직 (WebSocket 순서보장 + Python이 `playback_stopped` 대기 후 다음 턴 전송)
- `stt_done` 핸들링
- `responses_stream_start` (별도 메시지 불필요, `stream_start`에 통합)
- `play_music` / `play_audio_csv` (당장 불필요하면)

### 안 건드리는 것
- 모션 생성 (`generate_motion`, `control_motor`)
- 오디오 재생 (`CustomSoundStream`)
- 모터 제어 (`DynamixelDriver`)
- `read_and_split` (greeting/farewell 파일 재생에 재활용)

### VAP 로봇 오디오 입력 (미구현)
- VAP는 stereo 입력 (ch0=user, ch1=robot). 현재 Orchestrator는 user 오디오만 VAP에 넣고 있고, robot 오디오(TTS 출력)를 VAP에 피드하는 로직은 미구현.
- `playback_started` 이벤트를 받은 시점부터 TTS 청크를 VAP ch1에 실시간으로 넣으면 됨. C++의 실제 재생과 동기화되므로 별도 위치 추적 불필요.

### playback_position 관련
- 현재는 생략. 향후 VAP 로봇 오디오 피드 정밀 제어가 필요하면 추가.
- `playback_started`는 구현 (VAP 타이밍 시작점으로 사용).

### stale 청크 오염 방지
- `turn_id` 대신 WebSocket TCP 순서보장에 의존.
- Python은 `playback_stopped` 수신 후에만 다음 턴 오디오 전송 → 순서적으로 꼬이지 않음.
- C++은 `stop` 수신 시 버퍼/큐 전부 비움.

## Phase 5 Notes

### Orchestrator frame loop ordering
- Decision before drain (step 5 before 7) — interrupt processed before sending more audio.
- Deferred truncation check (step 9) runs every frame to catch generator completion.

### Barge-in Case C (deferred truncation)
- Approximate truncation saved immediately (DurationRatioTruncator from sent buffer length).
- `_pending_truncation` holds msg_id + stop_position. Each frame checks generator.stream_done.
- On stream completion: re-truncate with full ResponseData, update_message to correct.
- Cleanup in 5 places: stream_done, FAILED, new streaming, new prepare, session end.

### IConversationHistory breaking change
- add_user_message/add_assistant_message now return int IDs.
- Updated StubHistory in context tests to match.
- get_messages() returns new dicts (no _id), not deepcopy.

## Phase 4 Notes

### SpeechGenerator threading model
- `max_workers=2` so cancelled runs don't block new runs during API I/O.
- Python generators can't be interrupted cross-thread during `next()`. Only cooperative cancel via `threading.Event` + run-ID guard.
- Per-run `queue.Queue` isolation prevents stale audio contamination.
