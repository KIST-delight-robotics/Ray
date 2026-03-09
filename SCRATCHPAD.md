# Scratchpad

Claude's working memory. Free-form notes, observations, and context carried across sessions.

## Current Status

C++ ↔ Python WebSocket 프로토콜 정렬 완료.

- Python 측: types, interfaces, bridge, orchestrator, session, tests 모두 업데이트 (490 tests pass)
- C++ 측: WebSocket 서버 전환, 프로토콜 매핑, playback_started/playback_complete 전송, turn_id/STT_DONE_TIME 제거
- C++ 빌드 확인 (MOTOR_ENABLED=OFF)

Next: Phase 7 — Integration tests (Python ↔ C++ 실제 연결 테스트)

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
| `play_file` | `file_path` | 파일 재생 (인사/작별 등) | `play_audio` |

**C++ → Python:**

| 메시지 | 필드 | 설명 | C++ 기존 대응 |
|--------|------|------|---------------|
| `playback_started` | - | 재생 시작 (VAP 타이밍용) | (신규) |
| `playback_complete` | - | 재생 완료 (정상/중단 모두) | `speaking_finished` |

- `playback_stopped`는 별도로 두지 않음. Python이 `stop`을 보낸 상태인지 자체적으로 알고 있으므로, `playback_complete`를 받았을 때 문맥으로 구분.

### Python 수정 사항

**`ICppBridge` 인터페이스에 추가할 메서드:**
- `send_stream_start()`
- `send_audio_end()`
- `send_play_file(file_path: str)` — 기존 `send_greeting()`/`send_farewell()` 대체

**`CppBridge` 구현:** 해당 JSON 전송 추가.

**`CppEventType`:** `PLAYBACK_STOPPED` 제거. `PLAYBACK_COMPLETE` 하나로 통합.

**Orchestrator:**
- 오디오 드레인 흐름: `send_stream_start()` → `send_audio(chunk)` × N → `send_audio_end()`
- `stop` 전송 후 `playback_complete` 대기 (기존 STOP_PENDING 로직 유지, 이벤트 타입만 변경)

**SessionManager:**
- `send_greeting()` → `send_play_file("assets/audio/awake.wav")`
- `send_farewell()` → `send_play_file("assets/audio/sleep.wav")`

### C++ 수정 사항

1. **WebSocket: 클라이언트 → 서버** (`ix::WebSocket` → `ix::WebSocketServer`, 포트 8765)
2. **onMessageCallback**: type 문자열 매핑 교체 (위 표 참조)
3. **robot_main_loop**: `speaking_finished` → `playback_complete` rename + `playback_started` 전송 추가
4. **stop 처리**: 기존 `user_interruption` 로직 그대로, type명만 `stop`으로 변경. 스레드 종료 후 기존과 동일하게 `playback_complete` 전송.
5. **`play_file`**: 기존 `play_audio` 로직 재활용, type명만 변경

### 실제 삭제된 항목
- `turn_id` (전역 `current_turn_id`, 모든 메시지에서 제거)
- `STT_DONE_TIME` (전역 변수 및 관련 로직)
- `stt_done` 핸들러
- `responses_stream_start` (`stream_start`에 통합)

### 유지된 항목
- `play_music` / `play_audio_csv` (기존 로직 그대로)

### 안 건드리는 것
- 모션 생성 (`generate_motion`, `control_motor`)
- 오디오 재생 (`CustomSoundStream`)
- 모터 제어 (`DynamixelDriver`)
- `read_and_split` (파일 재생에 재활용)

### VAP 로봇 오디오 입력 (미구현)
- VAP는 stereo 입력 (ch0=user, ch1=robot). 현재 Orchestrator는 user 오디오만 VAP에 넣고 있고, robot 오디오(TTS 출력)를 VAP에 피드하는 로직은 미구현.
- `playback_started` 이벤트를 받은 시점부터 TTS 청크를 VAP ch1에 실시간으로 넣으면 됨. C++의 실제 재생과 동기화되므로 별도 위치 추적 불필요.

### playback_position 관련
- 현재는 생략. 향후 VAP 로봇 오디오 피드 정밀 제어가 필요하면 추가.
- `playback_started`는 구현 (VAP 타이밍 시작점으로 사용).

### stale 청크 오염 방지
- `turn_id` 대신 WebSocket TCP 순서보장에 의존.
- Python은 `playback_complete` 수신 후에만 다음 턴 오디오 전송 → 순서적으로 꼬이지 않음.
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
