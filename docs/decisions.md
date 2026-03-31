# Decision Log

비자명한 설계 결정과 시행착오 기록. 코드에서 바로 읽을 수 있는 구현 상세는 제외.


## ASR

- **Google Cloud STT V1**: V2는 batch/adaptation 등 불필요한 기능만 추가. 실시간 스트리밍 용도에는 V1으로 충분.


## LLM

- **OpenAI Responses API**: System message는 `instructions` param으로 전달, `input`에 넣지 않음. `previous_response_id`는 사용하지 않음 — ContextBuilder에서 토큰 예산 기반으로 직접 컨텍스트를 관리.


## TTS

- **WAV header quirk**: OpenAI TTS가 `n_frames = INT_MAX`인 malformed WAV를 반환하는 경우 있음. ASR 등 헤더 검증하는 소비자에 넘길 때는 ffmpeg re-encode 필요.


## TurnGPT

- **KV cache 활용**: ASR이 incremental update(같은 prefix, 늘어나는 suffix)를 보내므로 매번 전체 dialog를 재처리하면 낭비. 토큰 prefix 비교 후 새 토큰만 forward. 동일 입력이면 캐시된 확률 반환.
- **Context window eviction**: max 1024 tokens(GPT-2 한계). 초과 시 oldest turn을 `<ts>` 기준으로 삭제, 80% 이하까지. Eviction은 KV cache 전체 무효화 (full rebuild). 80% headroom은 limit 근처에서 thrashing 방지.
- **ONNX backend**: RPi 추론 성능용. PyTorch는 여전히 필요 (토큰화에 torch tensor 사용) — 목표는 속도 개선이지 PyTorch 제거가 아님. `onnx_model_path` 설정 여부로 backend 선택. KV cache 지원은 ONNX input name에 `past_key_0` 존재 여부로 감지.
- **ONNX threads = 2**: RPi 5 (4코어) 벤치마크 1–4스레드. 2스레드가 fp32/int8 모두 최적 (int8 42ms, fp32 111ms). 4스레드는 contention으로 오히려 느림.
- **int8 양자화**: 4x 작고 2.6x 빠르며 TRP 차이 무시 가능 (~0.04). 배포 권장.


## MaAI VAP

- **INT8 양자화 부적합**: CPC 인코더(Conv1D+LSTM, 2.5M params)에 `quantize_dynamic` 적용 시 1.9x 느려지고 정확도 대폭 하락. TurnGPT(MatMul 위주, 163M params)와 달리 작은 Conv 커널에서는 양자화 오버헤드가 연산 절감을 초과하고, 모델이 이미 L2 캐시에 들어가 메모리 대역폭 이점 없음.
- **배치 처리 무의미**: No-cache 배치(N frames)는 KV cache + 1 frame보다 느림. KV cache가 이미 반복 연산을 제거하므로 텐서 크기를 키워도 추가 이점 없음.
- **ONNX transformer export**: Transformer를 ONNX export하여 전체 파이프라인(encoder+transformer)을 ORT로 실행. PyTorch dispatch overhead 완전 제거. Mean 24ms (vs PyTorch 106ms, 3.9x). `torch.compile` warmup 100프레임 문제도 해소. 변환 시 dict KV cache → 12개 flat stacked tensor. Cross-attention source 순서 주의 (원본 입력을 src로 전달, 업데이트된 값 아님). 수치 차이 max 6.8e-6.
- **ORT 싱글스레드 최적**: Transformer ONNX 추가 후에도 `ort_threads=1`이 최적. `ort_threads=4`는 동기화 비용으로 2x 느려짐 (48ms vs 24ms).
- **PyTorch `p_now` 리스트 반환 버그**: `VapGPT.forward()`가 `p_now`을 `[speaker1, speaker2]` 리스트로 반환. `float(out["p_now"])`은 항상 실패하지만 integration test 없어 미발견. `p_now[0]` 추출로 수정.
- **torch.compile 한계**: 작은 텐서([1,1,256]) 연산에서 커널 launch + 메모리 할당 오버헤드가 남아 이론치(0.4ms) 대비 32ms. 10ms 이하를 원하면 ONNX export 필요. 현재 ONNX가 기본.


## TurnDetector

- **Paper-based two-path algorithm** (Skantze & Irfan, 2025): Path 1 (VAP sustained robot-favor) OR Path 2 (TurnGPT graduated silence timeout). 단일 모델 의존 회피 — VAP가 빠른 응답(~500ms), TurnGPT가 VAP 실패 시 eventual turn-taking 보장.
- **Backchannel vs interrupt 구분**: ROBOT_TURN에서 robot_audio 있을 때 `p_now > threshold`만으로는 interrupt 판단 불충분 — `p_fut`도 user를 favor해야 함. 백채널("응", "네")은 p_now만 spike하고 p_fut는 robot-favoring 유지.
- **`robot_audio=None` 시 unconditional interrupt 제거**: ROBOT_TURN에서 robot audio 없을 때(generation gap, PLAYBACK_STARTED 전) `user_is_speaking=True`만으로 interrupt 하던 것을 제거. 원인: 1–3s generation gap 동안 주변 소음이 false interrupt 유발 (실측: 7회 LLM 호출, 4.7s 지연). VAP는 robot 채널 없이 interrupt/backchannel 구분 불가. 수정: `TurnDecision.none()` 반환. 사용자 추가 발화는 orchestrator의 ASR text change cancel (0.5s grace) 로 처리.
- **`user_is_speaking` 전제조건**: 논문 pseudocode (Appendix A lines 56-61) 요구. 없으면 turn-shift 직후 VAP context가 아직 user-biased일 때 transient spike로 false interrupt 발생.
- **VAP error default `VAPResult(0, 0, False)`**: "robot favored"처럼 보이지만, transient error는 500ms 지속 안 되므로 Path 1 false trigger 없음. Persistent failure 시 Path 2(TurnGPT + silence)가 독립 작동.


## Async Thread Separation (VAP, TurnGPT)

- **별도 스레드 필요 이유**: RPi 5 worst case VAP 24ms + TurnGPT 30ms = 54ms, 30ms frame budget 초과. ONNX Runtime이 GIL 해제하므로 별도 스레드에서 진정한 병렬성 확보.
- **1-frame TurnGPT delay 허용**: `process_frame()`에서 poll 후 submit. Frame N의 submit 결과는 Frame N+1에서 poll. 30ms 지연은 500ms+ 단위의 turn-taking 판단에 무시 가능.


## SpeechGenerator

- **Streaming API**: `poll_audio() → bytes | None` + `stream_done`으로 TTS 청크를 즉시 CppBridge에 전달. 전체 합성 완료 대기 불필요.
- **`max_workers=2`**: 1이면 새 `prepare()`가 cancel된 run이 blocking API call(1–3s)에서 빠져나올 때까지 대기. 2면 즉시 시작. Turn detector 시그널 빈도상 3+ 동시 취소는 비현실적.


## C++ ↔ Python Protocol

### WebSocket topology
- **C++ server, Python client**: C++가 `ix::WebSocketServer` port 8765. Python이 `websockets`로 접속. 단일 연결 — `g_client_ws`에 저장.

### Protocol 단순화
- **`turn_id` 제거**: WebSocket TCP 순서 보장 + Python 상태 머신(playback_complete 대기 후 다음 turn)으로 stale chunk 오염 방지. C++는 `stream_start` 시 버퍼 클리어.
- **`playback_stopped` 제거 → `playback_complete`로 통합**: Python이 자체 `STOP_PENDING` 상태로 정상 완료/barge-in 구분. C++는 항상 `playback_complete`만 전송.
- **`playback_position` 스트림 제거**: `stop_pos = stop_pending_time - playback_start_time`으로 시간 기반 추정. Localhost에서 ±100ms 정확도 허용.
- **메시지 통합**: `stream_start` ← `responses_only` + `responses_stream_start`. `audio_end` ← `responses_stream_end`. `play_file` ← `play_audio` + `send_greeting` + `send_farewell`. `stop` ← `user_interruption`.
- **`playback_started` (신규)**: `control_motor` cycle 0에서 `soundStream.play()` 후 전송. Barge-in 위치 추정 및 VAP robot audio 타이밍 기준점.
- **Interrupt 시 `playback_complete` 전송**: C++가 interrupt cleanup 후에도 `playback_complete` 전송. Python의 STOP_PENDING 상태가 정상 해소됨.


## Session Lifecycle

- **Session factory 도입**: 이전 설계는 단일 Orchestrator/ConversationHistory를 재사용하며 `reset()`으로 상태 정리. 그러나 TurnDetector의 `_dialog_parts`가 `reset()`으로 안 지워져서 세션 간 상태 누수 발생. 수정: `session_factory`가 매 세션마다 Orchestrator, TurnDetector, SpeechGenerator, ContextBuilder, ConversationHistory를 새로 생성.
- **Three-tier lifecycle**: (1) Process-level — 모델, API 클라이언트, 하드웨어, executor (비싼 초기화, 1회). (2) Session-level — 상태 있는 orchestration 객체 (factory 재생성). (3) Turn-level — 경량 `reset()` (ASR 버퍼, TurnDetector 프레임 카운터).


## Similarity

- **Sentence embedding > SequenceMatcher**: SequenceMatcher는 문자 겹침 측정 — `"what is your"` vs `"what is your name"`이 0.87 (0.8 threshold에 차단됨). Sentence embedding (all-MiniLM-L6-v2)은 0.66 (정상 통과). 논문이 semantic similarity 사용.
- **sentence-transformers 3.x 핀**: 5.x는 `transformers>=5.0` 필요, `optimum`의 ONNX runtime과 충돌. 추론 전용이므로 기능 차이 없음.
