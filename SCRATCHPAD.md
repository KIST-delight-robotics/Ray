# Scratchpad

Claude's working memory. Free-form notes, observations, and context carried across sessions.

## Current Status

`input_text` 리팩토링 + 방어 가드 + 테스트 완료. 543 tests pass, ruff 0 errors.

### 완료된 작업 (2026-03-14, input_text 방어 가드 + 테스트)

**Codex + 직접 분석 발견 사항 해결**
1. ~~**`_begin_streaming()`에 빈 `input_text` 가드 없음**~~ → ✅ 방어 가드 추가 (`if not user_text: return`)
2. ~~**`_handle_prepare`에 `_awaiting_response` 가드 없음**~~ → ✅ 가드 추가 (`if self._awaiting_response: return`)
3. ~~**`input_text` lifecycle 단위 테스트 부재**~~ → ✅ 5개 테스트 추가 (prepare→set, cancel→clear, reset→clear, 연속 덮어쓰기, get_response_data→clear)
4. **turn_shift 시 generator FAILED → 턴 소실** → 변경 불필요 (의도된 에러 정책)
5. ~~**`get_response_data()` 후 `_input_text` 잔류**~~ → ✅ `get_response_data()`에서 `_input_text = ""` 클리어

### 완료된 작업 (2026-03-14, input_text 리팩토링)

**`_prepared_text` → `SpeechGenerator.input_text` 이동**
- `ISpeechGenerator`에 `input_text` abstract property 추가
- `SpeechGenerator`: `_input_text` 필드, `prepare()`에서 저장, `cancel()`/`reset()`에서 클리어
- Orchestrator: `_prepared_text`, `_saved_user_text` 완전 제거, `_begin_streaming()` 파라미터 제거 → `generator.input_text` 읽기
- 수동 리셋 7군데 → 0 (generator lifecycle이 자동 관리)
- 테스트 7개 업데이트

### 완료된 작업 (2026-03-13, history 일관성 + lint 정리)

**1. history 텍스트 일관성 수정**
- **문제**: 유사도 게이트가 re-prepare 차단 시, history에는 최종 ASR 텍스트가 기록되지만 응답은 prepare 시점 텍스트 기반 → 불일치
- **해결**: `_prepared_text` 필드 추가. `generator.prepare()` 호출 시점의 텍스트를 추적, `_begin_streaming`에서 history 기록에 사용
- 리셋 지점: `_begin_streaming`, `_handle_interrupt` awaiting, `_check_generator_completion` FAILED, `_start_session`

**2. `_handle_prepare` 데드 코드 제거**
- `_awaiting_response` 분기 (saved_user_text 결합 로직) 제거 — turn_shift 후 TurnDetector가 ROBOT_TURN으로 전이해서 prepare 시그널 도달 불가
- 관련 테스트 (`test_prepare_combines_text_during_awaiting`) 삭제

**3. 프로젝트 전체 ruff lint 정리**
- 27 파일, 56 errors 수정 (E501, F841, F821, B007, B905, SIM102, SIM105, SIM108, SIM115)

### 완료된 작업 (2026-03-13, VAP transformer ONNX)

**1. Transformer ONNX export (`onnx_export.py`)**
- `TransformerONNXWrapper`: dict KV cache → 12개 flat stacked tensor I/O
- ALiBi 마스크 pre-compute + 슬라이싱 (상태 변이 제거)
- einops → `torch.permute` 교체
- Cross-attention 순서 보존: `z1_in/z2_in` 저장 후 원본을 cross-attn source로 전달
- `export_transformer_onnx()`: opset 17, dynamic axes on T_cached/T_total

**2. 수치 동일성 검증**
- Transformer 단독 (500 synthetic frames): max diff 1.5e-6, PASSED
- Full pipeline enc+tfm (300 synthetic frames): max diff 2.0e-6, PASSED
- Full pipeline CANDOR 실제 음성 120초 (1,200 frames): max diff 6.8e-6, drift 없음, PASSED

**3. 성능 (CANDOR 120초, RPi 5, ort=1)**

| 단계 | Mean | 비중 |
|------|------|------|
| ONNX Encoder (2ch) | 16.2ms | 67.4% |
| ONNX Transformer | 7.8ms | 32.4% |
| Cache+trim | 0.1ms | 0.3% |
| **Total** | **24.0ms** | 100% |

- RTF: 4.16x (budget 100ms의 76% 여유)
- Budget 초과: 0% (1,200프레임 중 0건)
- vs PyTorch: 106.2ms → 24.0ms (3.9x speedup)
- 스레드 스윕: ORT=1이 최적, ORT=4는 2x 느려짐

**4. 프로덕션 통합 (`maai_vap.py`)**
- `MaAIVAPConfig.use_onnx_transformer` (기본값 True)
- `_process_transformer_onnx()`: numpy KV cache, ORT 추론
- `_process_transformer_pytorch()`: 기존 경로 (fallback)
- PyTorch `p_now` 리스트 반환 형식 버그 수정 (`float([a,b])` → `float(a)`)

**5. Integration test (`test_maai_vap_integration.py`)**
- 15 tests: BasicOperation(3), Stereo(2), Reset(2), TurnCycle(2), OnnxPytorchEquivalence(4), PyTorchMode(2)
- ONNX vs PyTorch 수치 일치 검증 (< 1e-4)

### 완료된 작업 (2026-03-13, robot_audio 비대칭 + 마이크 에러)

**1. robot_audio 배치 비대칭 수정 (#2)**
- **문제**: 배치 드레인 시 user_audio=N*30ms이지만 robot_audio=30ms 1개 → VAP에 (N-1)*30ms 무음
- **해결**: `_get_robot_audio_combined(frame_count)` 추가 — 재생 버퍼에서 N개 연속 프레임 추출
- 기존 `get_robot_audio_chunk()` 샘플 정렬 버그 수정: `int(elapsed*rate*width)` → `int(elapsed*rate)*width`
- IVAP docstring을 멀티프레임 사용에 맞게 보정
- 6개 테스트 추가 (TestRobotAudioCombined)

**2. 마이크 캡처 스레드 에러 전파**
- **문제**: 캡처 스레드 사망 시 `_error`에 저장만 하고 SessionManager SLEEP 루프 무한 대기
- **해결**: `IAudioInput.error` 프로퍼티 추가, `_run_sleep`에서 queue 타임아웃마다 체크 → 에러 raise

**3. config 기본값 설정** (이전 세션 미커밋분)
- VAP/TurnGPT 모델 경로 기본값 설정, PyTorch 테스트에 `onnx_model_path=""` 명시

**4. mock C++ WebSocket 서버** (이전 세션 미커밋분)
- `mock_cpp_server.py` 커밋

### 이전 Codex 리뷰 발견 사항 (2026-03-13) — 해결 상태

1. ~~**High: awaiting 경로에서 `_prepared_text` 빈 경우 history에 `""` 기록 가능**~~ → ✅ `input_text` 리팩토링으로 구조 개선 (방어 가드는 아직 미추가, 위 미완료 #1 참고)
2. **Medium: `_handle_prepare`에 `_awaiting_response` 가드 없음** → 미해결 (위 미완료 #2)
3. **Low: turn_shift 시 generator FAILED → 턴 소실** → 미해결 (위 미완료 #4)

~~**해결 방안: `_prepared_text`를 SpeechGenerator `input_text`로 이동**~~ → ✅ 완료 (2026-03-14)

**#5 배치 내 상태 전이 누락** (실측 후 판단)
- 배치 N프레임에서 VAP 결과 1개로 타이머를 N*30ms 증분 → 배치 중간 발화 상태 변화 반영 안 됨
- VAP 추론 주기(100ms ≈ 3프레임)이고 배치가 주로 2-3프레임이므로 영향 제한적
- **실제 파이프라인 실행해서 조기 turn_shift 발생 여부 확인 필요**

**VAP/TurnGPT 예산 초과 시 처리 미구현**
- VAP budget 초과 20-25% (P95: 150ms, budget 100ms). 프레임 드롭 or 5Hz 폴백 설계 필요
- 현재는 배치 드레인만 있고 명시적 budget 초과 처리 없음

**prepare → turn_shift 텍스트 불일치 문제** — ✅ 해결 (2026-03-13, `_prepared_text` 도입)

### 완료된 작업 (2026-03-12, 배치 드레인 + mock 서버 + config)

**1. 추론 지연 시 프레임 배치 드레인 (Orchestrator + TurnDetector)**
- **문제**: VAP/TurnGPT 추론이 30ms 프레임 예산 초과 시 오디오 큐에 프레임이 밀림
- **해결**: `_run_frame()`에서 첫 프레임 후 밀린 프레임을 non-blocking drain → ASR 개별 feed → 유저 오디오 concat → TurnDetector 1회 호출 + `frame_count` 전달
- **변경 파일**:
  - `core/interfaces.py`: `ITurnDetector.process_frame`에 `frame_count: int = 1` 추가
  - `turn_taking/turn_detector.py`: 타이머 증분을 `frame_duration_sec * frame_count`로 보정 (silence, vap_favor_robot, last_asr_change)
  - `orchestrator/orchestrator.py`: `_drain_available_frames()` 추가, `_MAX_BATCH_FRAMES = 10` 상한, `_process_turn_detector`에 frame_count 전달
  - `turn_taking/vap.py`: step counter 잔여 샘플 보존 (`= 0` → `%= self._step_samples`)

**2. mock C++ WebSocket 서버 (`mock_cpp_server.py`)**
- C++ 프로세스 없이 전체 파이프라인 테스트용 PyAudio 재생 서버
- 프로토콜: stream_start/audio/audio_end/stop/play_file ↔ playback_started/playback_complete
- `AudioPlayer`: `stop_event` + `end_of_stream` 이벤트 기반 즉시 중단 (큐 sentinel 방식 X)
- `MockCppServer`: `ws.recv(timeout=0.05)` 폴링 + `response_queue` + `_monitor_completion` 스레드로 메시지 수신/재생 완료 분리 → stop 메시지를 재생 중에도 수신 가능
- 검증 완료: normal flow, barge-in (stop after audio_end), stop before audio_end — 모두 playback_complete 1회, duplicate 없음

**3. config 기본값에 모델 경로 설정**
- `VAPConfig.model_path`: `"external/VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt"`
- `TurnGPTConfig.onnx_model_path`: `"models/turngpt/turngpt_v2_kvcache_int8.onnx"`
- `TurnGPTConfig.tokenizer_path`: `"models/turngpt/tokenizer"`
- PyTorch 백엔드 테스트에 `onnx_model_path=""` 명시 추가 (ONNX 기본값이 비어있지 않으므로)
- config 기본값 테스트는 경로 하드코딩 대신 `assert cfg.model_path` (truthy 체크)

### 미완료 — 다음 세션에서 진행

**#2+#5 배치 드레인 시 robot_audio 비대칭 + 배치 내 상태 전이 누락**
- 현재: 5프레임 배치에서 유저 오디오 150ms / 로봇 오디오 30ms 1개 → VAP 버퍼에 120ms 공백
- 현재: 배치 중간에 유저가 말을 멈춰도 마지막 VAP 결과만 보고 전체 배치를 침묵으로 처리 → 턴 시프트 조기 발생 가능
- **계획된 해결**: 배치 내 프레임별로 VAP `feed_audio`만 호출 (추론은 step 간격 유지), TurnGPT는 1회만. 이러면 robot_audio도 프레임별 전달, 상태 전이도 정확 추적. VAP feed_audio 자체는 버퍼 append만이라 성능 영향 없음.

### 완료된 작업 (2026-03-12, session factory + entry point)
- **3단계 생명주기 모델 도입**: process-level (모델/API/executor) → session-level (factory 재생성) → turn-level (reset)
- **SpeechGenerator**: 외부 `ThreadPoolExecutor` 주입 지원 (`_owns_executor` 플래그)
- **Orchestrator**: `_save_history()` 제거 — SessionManager에 save 책임 일원화
- **SessionManager**: `session_factory: Callable[[], SessionComponents]` 패턴, `_session_lock` 추가, orchestrator.run() 예외 시에도 FAREWELL 전이 보장
- **`voice_pipeline/__main__.py`**: process-level 싱글턴 + factory 클로저 + Windows signal 핸들링
- **`pyproject.toml`**: `ray` 엔트리 포인트 추가 (`uv run ray`)

### 완료된 작업 (2026-03-12, TurnGPT)
- **TurnGPT eviction 버그 수정** (2건):
  1. `_clear_cache()` 제거 — eviction 시 매 호출 full recompute (~1,500ms) → cache 재활용으로 incremental 유지
  2. O(N) eviction 루프 → `keep_turns` 슬라이싱 (tokenize N회 → 1회)
- `TurnGPTConfig.keep_turns` 추가 (기본값 2), `_EVICTION_HEADROOM` 제거
- TurnGPT 스트레스 테스트: 150초 연속, eviction 경계 포함
- VAP 스파이크 분석: per-stage profiling (encoder/transformer/cache trim)
- `vap_console_view.py`: ONNX 파이프라인 + 오디오 동기화 재생 뷰어

### 완료된 작업 (2026-03-11)
- MaAI VAP: ONNX encoder + PyTorch transformer 커스텀 파이프라인
- 최적 설정: pt=1, ort=1 (싱글스레드), mean 66ms (10Hz budget 100ms)
- torch.compile: transformer 22% 추가 가속 (mean 56ms)
- 실제 음성(CANDOR) 수치 동일성 검증 PASSED (max diff 1.4e-5)
- `MaAIVAPWrapper(IVAP)` 구현 (`turn_taking/maai_vap.py`)
- `MaAIVAPConfig` 추가 (`core/config.py`)
- TurnGPT KV-cache 버그 수정: 입력이 캐시보다 짧을 때 0-token forward 에러
- 부하 테스트: VAP + TurnGPT 동시 60초, CPU 45% (여유 있음)

### TurnGPT 스트레스 테스트 결과 (eviction 수정 후, keep_turns=2)

| | FP32 Mean | FP32 P95 | INT8 Mean | INT8 P95 |
|---|---|---|---|---|
| 수정 전 (ratio=0.8 + bug) | 366ms | 1,541ms | — | — |
| **수정 후 (keep_turns=2)** | **70ms** | **104ms** | **30ms** | **43ms** |

- keep_turns 값 차이(0 vs 2)는 CPU governor 노이즈에 묻힘 → 기본값 2 유지
- INT8: >100ms = 1.1% (5건/450), 실사용 문제 없음

### VAP 스파이크 분석 결과 (10Hz, CANDOR 실제 음성)
- 스파이크 원인: **100% Transformer** (encoder/cache trim 무관)
- CPU governor (ondemand) 주파수 변동이 원인. GC 아님 (disable해도 동일)
- 분포: 60-80ms에 79% 밀집, 이후 연속적 long tail (bimodal 아님)
- budget(100ms) 초과: **20-25%** (3회 반복 측정 확인)
- P95: 150±4ms (안정적), P99: 192-230ms (변동)
- 메모리: drift 없음 (RSS ±1MB)

### 부하 테스트 결과 (VAP + TurnGPT 동시, CANDOR 실제 음성)
| | Mean | Median | P95 | Max |
|---|------|--------|-----|-----|
| VAP (10Hz) | 83.5ms | 68.8ms | 139.4ms | 178.3ms |
| TurnGPT (~3Hz) | 74.6ms | 70.5ms | 104.9ms | 119.2ms |
| CPU 전체 | 45.4% | | | 58.0% |

### Next
- **`uv run ray` 실행 테스트** — 하드웨어 + 외부 서비스 연결 상태에서 전체 파이프라인 검증
- Phase 7 — Integration tests (Python ↔ C++ 실제 연결 테스트)
- MaAI VAP vs VAP-Realtime 최종 선택 후 파이프라인 통합
- VAP budget 초과 20-25% 허용 설계 (프레임 드롭 or 5Hz 폴백)
- TurnGPT: dialog 누적 시 비동기 cache prefill (선택적 최적화)

---

## VAP (MaAI) ONNX 경량화 작업 (2026-03-11)

### 목표
RPi 5에서 VAP 10Hz 실시간 동작 (프레임당 100ms 예산)

### 배경 — 왜 VAP가 느린가
- MaAI VAP = CPC Encoder (Conv5 + LSTM) + Transformer (GPT cross-attention) + Classifier
- 전체 ~9M params. Encoder ~2M, Transformer ~7M
- 기존 MaAI 10Hz/5s: **~110ms/frame** (예산 100ms 초과, RTF 0.91x)
- 5Hz/3s: ~115ms (예산 200ms, RTF 1.73x — 가능)
- 병목: encoder ~40ms (양 채널), transformer ~70ms

### 시도 1: Encoder만 ONNX 변환 (기존 스크립트)
- `scripts/convert_maai_encoder_onnx.py` (이전 버전)
- **문제 1**: einops Rearrange가 ONNX에서 불필요한 Transpose 7개 생성
- **문제 2**: ORT 세션 설정 누락 (스레드, 그래프 최적화 미적용)
- **문제 3**: downsample 가중치가 랜덤 초기값 (VAP state dict 미적용)
- 결과: FP32 ONNX가 PyTorch보다 같거나 느림

### 시도 2: 전체 파이프라인 ONNX 통합
- 분석 결과 **불가능**: LSTM 내부 상태, KV cache Python dict, 동적 캐시 트리밍 등
- `encoder_components.py:148-153`, `vap.py:137-144`, `model.py:294-311`이 주요 차단점

### 시도 3: 수정된 ONNX 변환 + 커스텀 파이프라인
**변환 스크립트 수정 (`convert_maai_encoder_onnx.py` v2):**
- einops → `torch.permute()` 교체 (Transpose 7→4)
- MaAI 인스턴스에서 encoder 추출 (학습된 downsample 가중치 보장)
- ORT 세션: `ORT_ENABLE_ALL` + `intra_op_num_threads` 설정

**커스텀 파이프라인 (`VapOnnxPipeline` in `benchmark_maai_custom_pipeline.py`):**
- MaAI.process() 우회: numpy → ONNX encoder → torch.from_numpy → vap.forward()
- monkey-patch 방식 대비 torch↔numpy 변환 1회 절감
- 수치 동일성 검증 PASSED (max diff < 1e-5, 50프레임)

### 스레드 스윕 벤치마크 결과 (10Hz/5s, 200프레임)

| Pipeline | Threads | Mean | Median | RTF | Status |
|----------|---------|------|--------|-----|--------|
| MaAI | pt=1 | 96.0ms | 87.7ms | 1.04x | OK |
| MaAI | pt=2 | 118.6ms | 106.0ms | 0.84x | SLOW |
| MaAI | pt=4 | 111.6ms | 101.5ms | 0.90x | SLOW |
| **Custom** | **pt=1,ort=1** | **63.7ms** | **59.2ms** | **1.57x** | **OK** |
| Custom | pt=1,ort=2 | 82.5ms | 73.7ms | 1.21x | OK |
| Custom | pt=2,ort=1 | 96.4ms | 88.3ms | 1.04x | OK |
| Custom | pt=4,ort=1 | 66.2ms | 58.4ms | 1.51x | OK |
| Custom | pt=4,ort=4 | 88.9ms | 80.9ms | 1.12x | OK |

**핵심 발견:**
1. **싱글스레드가 멀티스레드보다 빠름** — RPi 4코어에서 이 모델 크기는 스레드 동기화 비용 > 병렬 이득
2. **Custom pt=1,ort=1 = 63.7ms** — 10Hz 예산(100ms)의 57% 여유. 이전에 불가능했던 10Hz 실시간 달성
3. MaAI도 pt=1이 pt=4보다 빠름 (96ms vs 112ms)

### 실제 음성 수치 동일성 검증 (CANDOR 데이터셋, 1200프레임/120초)
- p_now max diff: 1.4e-5, p_future: 1.3e-5, vad: 6.7e-6
- 후반부 drift ratio 2.0x이나 절대값 극소 → LSTM 누적 오차 없음
- **결론: ONNX encoder와 원본 PyTorch encoder는 수치적으로 동일**

### 프로파일링 (실제 음성, 10Hz/5s, pt=1,ort=1)

| 단계 | Mean | 비중 |
|------|------|------|
| ONNX encoder (2ch) | 17.5ms | 23.8% |
| Transformer fwd | 55.6ms | 75.6% |
| 기타 (변환/캐시) | 0.4ms | 0.5% |

### torch.compile A/B 테스트 (동일 프로세스 내 비교)

| Variant | Mean | Median | P95 |
|---------|------|--------|-----|
| no_grad (baseline) | 72.1ms | 64.0ms | 121.6ms |
| torch.compile (warmed) | 56.1ms | 46.7ms | 105.5ms |

- **torch.compile: mean 22% 감소, median 27% 감소**
- 컴파일 캐시: `/tmp/torchinductor_*` (135MB). `TORCHINDUCTOR_CACHE_DIR`로 영구 경로 지정 가능
- 첫 실행 시 warmup ~100프레임(10초) 필요, 캐시 있으면 재시작 시 빠르게 로드

### P95/Max 스파이크 원인 분석
- **Python GC: 아님** (0회 발생)
- **CPU governor (`ondemand`)가 원인** — 1.5~2.4GHz 주파수 스케일링으로 지터 발생
- `performance` governor 테스트 → thermal throttling으로 오히려 악화 (mean 99ms, max 550ms)
- **결론: 하드웨어 한계. ondemand 유지가 최적. 간헐적 budget 초과는 허용 설계 필요**

### 남아있는 작업
- `VapOnnxPipeline` + `torch.compile`을 `voice_pipeline/turn_taking/vap.py`에 통합 (IVAP 구현)
- 통합 시 결정 필요: 기존 VAPWrapper (VAP-Realtime) vs MaAI VAP 중 어떤 모델 사용할지
- INT8 양자화 추가 테스트 (추가 속도 개선 가능)
- 스레드 설정을 config에 반영 (`ort_threads`, `pt_threads`)
- TurnGPT 스파이크 테스트 (2코어 최적이나 지터 미확인)
- `TORCHINDUCTOR_CACHE_DIR` 영구 경로 설정

### 관련 파일
- `scripts/convert_maai_encoder_onnx.py` — ONNX 변환 (v2, MaAI에서 가중치 추출)
- `scripts/benchmark_maai_custom_pipeline.py` — 커스텀 파이프라인 벤치마크 + VapOnnxPipeline 클래스
- `scripts/benchmark_maai_onnx_encoder.py` — encoder 단독 벤치마크
- `scripts/verify_onnx_equivalence.py` — 실제 음성 수치 동일성 검증 + 단독 벤치마크
- `scripts/_test_torch_optimizations.py` — no_grad/inference_mode/torch.compile A/B 테스트
- `scripts/_profile_pipeline_split.py` — encoder/transformer 시간 분리 프로파일링
- `docs/vap_onnx_thread_sweep_results.txt` — 스레드 스윕 결과 원본
- `docs/vap_rpi_benchmark.md` — 이전 VAP 벤치마크
- `external/MaAI/` — MaAI 원본 레포 (참조용)
- `external/VAP-Realtime/` — VAP-Realtime 레포 (MaAI에 통합됨)
- `models/maai_encoder_5hz.onnx`, `models/maai_encoder_10hz.onnx` — 변환된 ONNX (context=20 가중치)

