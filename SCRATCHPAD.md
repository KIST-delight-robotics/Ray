# Scratchpad

Claude's working memory. Free-form notes, observations, and context carried across sessions.

## Current Status

VAP + TurnGPT 스트레스 테스트 완료. TurnGPT eviction 버그 수정. 517 tests pass.

### 완료된 작업 (2026-03-12)
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
- Phase 7 — Integration tests (Python ↔ C++ 실제 연결 테스트)
- 파이프라인 러너 스크립트 작성 (전체 end-to-end 실행)
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

