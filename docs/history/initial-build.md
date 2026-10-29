# 초기 파이프라인 구축

> 2026-02 ~ 2026-07 초기 파이프라인 구축 작업의 기록. 완료 후 고정 — 이후 변경은 반영하지 않는다.

## 개요

cascade 엔진(ASR → 턴 감지 → LLM → TTS)으로 첫 파이프라인을 만들며 정한 결정 중, 대화 엔진이 GPT-Live 로 바뀐
뒤에도 쓰이는 코드에 대한 것 — C++ ↔ Python 프로토콜, 세션 수명, Silero VAD, 대화 기록 저장, 임베더, OpenAI·Google
어댑터. 글 속의 `Phase`, `SessionLoop`, `TurnDetector`, `ContextBuilder`, VAP 등은 당시 cascade 엔진의 구성 요소다.

cascade 전용 결정은 [archive/cascade/decisions.md](../archive/cascade/decisions.md), 장기기억은
[memory-v1.md](memory-v1.md) 에 있다.


## C++ ↔ Python 프로토콜

### WebSocket topology

- **C++ server, Python client**: C++가 `ix::WebSocketServer` port 9200. Python이 `websockets`로 접속. 단일 연결 —
  `g_client_ws`에 저장.

### 프로토콜 단순화

- **`turn_id` 제거**: WebSocket TCP 순서 보장 + Python 상태 머신(playback_complete 대기 후 다음 turn)으로 stale
  chunk 오염 방지. C++는 `stream_start` 시 버퍼 클리어.
- **`playback_stopped` 제거 → `playback_complete`로 통합**: Python이 자체 `Phase.STOPPING` 상태(+`_stop_pending_time`)로
  정상 완료/barge-in 구분. C++는 항상 `playback_complete`만 전송.
- **`playback_position` 스트림 제거**: `stop_pos = stop_pending_time - playback_start_time`으로 시간 기반 추정.
  Localhost에서 ±100ms 정확도 허용.
- **메시지 통합**: `stream_start` ← `responses_only` + `responses_stream_start`. `audio_end` ← `responses_stream_end`.
  `play_file` ← `play_audio` + `send_greeting` + `send_farewell`. `stop` ← `user_interruption`.
- **`playback_started` (신규)**: `control_motor` cycle 0에서 `soundStream.play()` 후 전송. Barge-in 위치 추정 및 VAP
  robot audio 타이밍 기준점.
- **Interrupt 시 `playback_complete` 전송**: C++가 interrupt cleanup 후에도 `playback_complete` 전송. Python의
  STOP_PENDING 상태가 정상 해소됨.


## 세션 수명

- **Session factory 도입**: 이전 설계는 단일 SessionLoop/ConversationHistory를 재사용하며 `reset()`으로 상태 정리.
  그러나 TurnDetector의 `_dialog_parts`가 `reset()`으로 안 지워져서 세션 간 상태 누수 발생. 수정: `session_factory`가
  매 세션마다 SessionLoop, TurnDetector, SpeechGenerator, ConversationHistory를 새로 생성 (ContextBuilder는
  SpeechGenerator 내부에서 생성되어 함께 재생성됨). (→ 이후 `wiring.py` 의 `ProcessComponents.create_session()` 으로
  옮겨짐 — [archive/eval/decisions.md](../archive/eval/decisions.md) 의 "배선 공유")
- **Three-tier lifecycle**:
  1. Process-level — 모델, API 클라이언트, 하드웨어, executor (비싼 초기화, 1회).
  2. Session-level — 상태 있는 orchestration 객체 (factory 재생성).
  3. Turn-level — 경량 `reset()` (ASR 버퍼, TurnDetector 프레임 카운터).


## Silero VAD

Silero 모델은 호출어 감지와 cascade 의 턴 감지가 같은 인스턴스를 썼다. 턴 감지 쪽의 commit 시점 리셋은
[archive/cascade/decisions.md](../archive/cascade/decisions.md#silero-vad--세션-내부-오염) 에 있다.

- **LSTM 상태는 세션 간 오염된다 — 세션 시작마다 `reset_states()` 필수**: LSTM 상태가 세션 간 유지되어, 직전 세션
  하나(23s)만 선행해도 조용한 음성(peak 0.107)의 감지율이 38.9%→0%로 붕괴 — 같은 녹음을 단독으로 돌리면 max 1.000.
  - 손상된 상태는 이후 오디오(무음 포함)로 자연 회복되지 않으며, 음량이 낮을수록 취약. ONNX/PyTorch 무관.
  - 리셋 비용 ~11µs(청크 추론의 1/60)로 무시 가능.
  - 워밍업 공백(~250ms)은 발화 *도중* 리셋에만 발생 — 리셋 후 발화가 *시작*되는 경우는 즉시 감지.
- **SLEEP 진입 리셋은 wakeword 모듈 책임** (2026-06): wakeword는 인식 사이클 종료마다 자체 리셋해 SLEEP *중*
  오염은 자가 회복되지만, ACTIVE→SLEEP 잔류 오염은 "발화가 threshold를 넘어야 리셋 발동" 구조라 조용한 호출에 회복
  기회가 없음(미감지→리셋 불발 순환). `WakewordDetector.reset()`을 `__main__`의 SLEEP 전환 지점에서 호출 —
  wakeword의 `_pre_buffer`/`_vad_buffer` 잔류 오디오(STT에 묵은 청크 전송 문제)까지 모듈 내부에서 함께 정리.


## 외부 API 어댑터

- **Google Cloud STT V1**: V2는 batch/adaptation 등 불필요한 기능만 추가. 실시간 스트리밍 용도에는 V1으로 충분.
- **OpenAI Responses API**: System message는 `instructions` param으로 전달, `input`에 넣지 않음.
  `previous_response_id`는 사용하지 않음 — ContextBuilder에서 토큰 예산 기반으로 직접 컨텍스트를 관리.
- **OpenAI TTS WAV header quirk (역사적)**: OpenAI TTS가 WAV 포맷 요청 시 `n_frames = INT_MAX`인 malformed 헤더를
  반환하는 경우 있음. 기록 당시엔 `response_format="pcm"` raw 스트리밍이라 해당 없음 — WAV가 필요한 소비자에 넘길
  때만 주의.


## 대화 기록 (ConversationHistory)

- **Write-through SQLite**: 매 메시지 즉시 INSERT. 세션 중 crash 시 최대 진행 중 turn 1개만 유실 (batch-at-end였으면
  전체 세션 유실). `save()`는 `ended_at` 설정 + WAL checkpoint만 수행.
- **Graduated DB corruption recovery**: 정상 open → WAL 파일 삭제 재시도 → corrupt 파일 백업 + 신규 DB 생성. RPi 전원
  차단 시 대부분 WAL만 손상되므로 WAL 삭제만으로 복구되는 경우가 많음.
- **Responses API format 직접 저장**: 중간 canonical format 없이 vendor-specific dict 그대로 `item_json`에 저장.
  Vendor 교체 시 migration script 필요 (의도적 트레이드오프).


## 임베더

- **기본 임베더 = ONNX qint8 ARM64**: `create_embedder` 기본값이 `onnx/model_qint8_arm64.onnx`. qint8 배치 임베딩
  기준으로 TurnDetector `_SIMILARITY_THRESHOLD`는 0.85로 상향.
- **sentence-transformers 3.x 핀**: 5.x는 `transformers>=5.0` 필요, `optimum`의 ONNX runtime과 충돌. 추론 전용이므로
  기능 차이 없음.
