# VAP Raspberry Pi Benchmark Results

- **Date**: 2026-03-10
- **Hardware**: Raspberry Pi (Linux 6.8.0-1048-raspi, 4 cores ARM)
- **PyTorch**: 2.10.0+cpu (4 threads)
- **Model**: VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt (5,785,117 params)
- **Model load time**: 1.55s


## Part 1: Raw model.probs() latency (step=0.1s)

| Context | Mean | Median | Min | Max | P95 | P99 | RT Factor |
|---------|------|--------|-----|-----|-----|-----|-----------|
| 5s | 1,198 ms | 1,208 ms | 1,068 ms | 1,293 ms | 1,280 ms | 1,290 ms | 0.08x |
| 10s | 2,452 ms | 2,509 ms | 2,107 ms | 2,835 ms | 2,665 ms | 2,801 ms | 0.04x |
| 20s | 5,368 ms | 5,334 ms | 5,013 ms | 5,858 ms | 5,683 ms | 5,823 ms | 0.02x |


## Part 2: Full feed_audio() pipeline (step=0.1s)

| Context | Mean | Median | Min | Max | P95 | P99 | RT Factor |
|---------|------|--------|-----|-----|-----|-----|-----------|
| 5s | 1,366 ms | 1,263 ms | 1,119 ms | 2,620 ms | 1,860 ms | 2,468 ms | 0.07x |
| 20s | 4,704 ms | 4,631 ms | 4,370 ms | 5,445 ms | 5,007 ms | 5,358 ms | 0.02x |


## Part 3: Sustained streaming (30s audio, context=5s)

- Simulated: 30.0s of audio
- Wall clock: 300.68s
- Overall RTF: 0.10x
- Inferences: 327
- Per-inference mean: 918.5 ms, median: 1,146.7 ms, P95: 1,426.0 ms


## Part 4: Feasible step_sec (context=5s)

Mean inference latency: 1,538 ms

| Step (s) | Budget (ms) | RTF | Feasible |
|----------|-------------|------|----------|
| 0.05 | 50 | 0.03x | NO |
| 0.10 | 100 | 0.07x | NO |
| 0.20 | 200 | 0.13x | NO |
| 0.30 | 300 | 0.20x | NO |
| 0.50 | 500 | 0.33x | NO |
| 1.00 | 1000 | 0.65x | NO |


## Per-stage profiling (single inference)

| Stage | 5s context | 20s context | Share (5s) |
|-------|-----------|-------------|------------|
| CPC Encoder (x2 channels) | 947 ms | 4,112 ms | 81% |
| AR channel (x2) | 27 ms | 235 ms | 2% |
| AR cross-channel (GPTStereo) | 197 ms | 1,578 ms | 17% |
| Heads | 1 ms | 5 ms | <1% |
| **Total** | **1,172 ms** | **5,930 ms** | |


## Bottleneck

CPC Encoder processes the **entire raw waveform** (80k~320k samples) through CNN layers every inference call.
No incremental/cached encoding — full recomputation each time.


## Conclusion (Original VAP)

PyTorch VAP is **not real-time feasible** on Raspberry Pi under any configuration.

---

# MaAI (Real-time VAP) Raspberry Pi Benchmark Results

- **Date**: 2026-03-10
- **Hardware**: Raspberry Pi (Linux 6.8.0-1048-raspi, 4 cores ARM)
- **PyTorch**: 2.10.0+cpu (4 threads)
- **Package**: maai 0.1.16 (pip)
- **Language**: English (en)


## Key optimizations over original VAP

1. **Incremental encoder**: Only new audio frame (~1920 samples) goes through CPC, not the entire buffer
2. **KV Cache**: GPT transformer reuses past key/value, only computes new token


## All English model configs (KV Cache ON, 30s streaming)

| frame_rate | context | Mean | Median | P95 | Budget | RTF | Feasible |
|-----------|---------|------|--------|-----|--------|-----|----------|
| 5Hz | 3s | 115 ms | 106 ms | 187 ms | 200 ms | 1.73x | YES |
| 5Hz | 5s | 122 ms | 113 ms | 188 ms | 200 ms | 1.64x | YES |
| 5Hz | 20s | 135 ms | 129 ms | 201 ms | 200 ms | 1.48x | YES |
| 10Hz | 3s | 107 ms | 99 ms | 171 ms | 100 ms | 0.94x | NO |
| 10Hz | 5s | 110 ms | 102 ms | 183 ms | 100 ms | 0.91x | NO |
| 10Hz | 20s | 119 ms | 110 ms | 191 ms | 100 ms | 0.84x | NO |
| 20Hz | 2.5s | 104 ms | 97 ms | 162 ms | 50 ms | 0.48x | NO |
| 20Hz | 20s | 116 ms | 105 ms | 197 ms | 50 ms | 0.43x | NO |


## KV Cache effect (10Hz, context=5s)

| KV Cache | Mean | Median | P95 | RTF |
|----------|------|--------|-----|-----|
| ON | 110 ms | 102 ms | 183 ms | 0.91x |
| OFF | 127 ms | 118 ms | 197 ms | 0.79x |


## Observations

- CPC encoder is a fixed cost (~100ms per frame) regardless of frame_rate or context
- Higher frame_rate does not reduce per-frame latency — it only tightens the budget
- KV cache gives ~15% speedup by avoiding GPT recomputation
- **5Hz models are the only real-time feasible option on Raspberry Pi**
- 5Hz + 3s context is the fastest (RTF 1.73x, 73% headroom)


## Conclusion (MaAI)

MaAI at **5Hz** is real-time feasible on Raspberry Pi with comfortable margin.
10Hz is borderline (~6-16% over budget). 20Hz is not feasible.

---

# torch.compile 효과 (MaAI 10Hz)

- **Date**: 2026-03-12
- **PyTorch**: 2.10.0+cpu, `torch.compile(mode="default")`

## Transformer 병목 분석

MaAI 파이프라인 시간 분해 (10Hz, context=5s, KV cache ON):

| 구간 | 시간 | 비중 |
|------|------|------|
| ONNX 인코더 (×2 ch) | ~16ms | 26% |
| PyTorch 트랜스포머 | ~47ms | 74% |

트랜스포머의 이론 연산량은 7.2M FLOPs (<0.4ms @18GFLOPS)이나 실측 47ms.
258개 PyTorch 모듈 호출의 dispatch overhead + 텐서 할당이 99%를 차지.

## torch.compile 적용 결과 (트랜스포머 단독, 200회)

| 방식 | Mean | P50 | P5 | P95 |
|------|------|-----|-----|-----|
| Eager | 66.8ms | 56.5ms | 41.6ms | 134.1ms |
| **torch.compile** | **39.7ms** | **31.9ms** | **20.4ms** | **95.2ms** |

## 전체 파이프라인 (인코더 + 트랜스포머, 10Hz)

| 방식 | Mean | P50 | P95 | Budget | RTF |
|------|------|-----|-----|--------|-----|
| Eager | 85ms | 75ms | 152ms | 100ms | 1.17x |
| **torch.compile** | **47ms** | **42ms** | **109ms** | **100ms** | **2.12x** |

torch.compile로 **10Hz가 안정적으로 실시간 가능** (53% 여유).

---

# VAP + TurnGPT 동시 실행 부하 테스트

- **Date**: 2026-03-12
- **Duration**: 30초 스트리밍 시뮬레이션

## 설정

| 모델 | 설정 |
|------|------|
| VAP (MaAI) | 10Hz, context=5s, torch.compile=ON, ort_threads=1, pt_threads=1 |
| TurnGPT | ONNX KV-cache int8, ~3Hz, ort_threads=2 |

## 결과

| 모델 | Mean | P50 | P95 | Max | Budget | >Budget |
|------|------|-----|-----|-----|--------|---------|
| VAP 10Hz | 52.6ms | 47.0ms | 80.9ms | 106.7ms | 100ms | 1.0% |
| TurnGPT 3Hz | 28.7ms | 27.9ms | 42.6ms | 53.3ms | 330ms | 0% |

## CPU 사용률

| | Mean | Max |
|---|------|-----|
| Overall | 34.3% | 44.3% |
| Core 0 | 33.5% | 50.5% |
| Core 1 | 34.6% | 58.6% |
| Core 2 | 33.7% | 55.0% |
| Core 3 | 30.7% | 51.0% |

ASR, LLM, TTS, AudioInput 등 나머지 파이프라인에 ~60% 여유.

---

# INT8 양자화 시도 (결론: 부적합)

- **Date**: 2026-03-12
- **방법**: `onnxruntime.quantization.quantize_dynamic` (QInt8)

## 모델 크기

| Model | FP32 | INT8 | 비율 |
|-------|------|------|------|
| 5Hz encoder | 12.1 MB | 3.1 MB | 25% |
| 10Hz encoder | 9.6 MB | 2.4 MB | 26% |

## 속도 (역효과)

| Config | FP32 | INT8 | 변화 |
|--------|------|------|------|
| 5Hz 인코더 | 13.7ms | 26.8ms | **1.9x 느려짐** |
| 10Hz 인코더 | 8.0ms | 15.3ms | **1.9x 느려짐** |

## 정확도 (대폭 하락)

Embedding max diff ~1.3 (TurnGPT int8의 MAE 0.06 대비 매우 큼).

## 원인

CPC 인코더는 Conv1D+LSTM 구조(2.5M params, 10MB)로:
1. ARM의 `ConvInteger` 커널이 fp32 Conv 대비 비최적화
2. 모델이 L2 캐시에 들어가므로 메모리 대역폭 절감 이점 없음
3. `DynamicQuantizeLinear` 오버헤드가 연산 절감보다 큼
