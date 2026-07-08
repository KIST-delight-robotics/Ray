# TurnGPT 종합 벤치마크 (speaker_ids 반영)

> 측정일: 2026-03-09
> 환경: Raspberry Pi 5 (ARM Cortex-A76 4-core, 8GB RAM, CPU only)
> 모델: TurnGPT (GPT-2 124M 기반, SODA fine-tuned)
> 체크포인트: epoch=47, val_loss=0.6526
> ONNX Runtime: intra_op_num_threads=4, ORT_ENABLE_ALL
> **참고**: 이후 스레드 수 벤치마크 결과, 기본값을 2로 변경함 (4보다 빠르고 CPU 여유 확보). 아래 수치는 4 스레드 기준.
> 스크립트: `scripts/benchmark_turngpt_all.py`


## 테스트 대상 (7가지 추론 방식)

| # | 방식 | 설명 |
|---|---|---|
| 1 | PyTorch | 원본 TurnGPT 체크포인트, `model.forward()` |
| 2 | ONNX NC FP32 | no-cache, float32 |
| 3 | ONNX NC INT8 | no-cache, int8 dynamic quantization |
| 4 | ONNX KV FP32 (prefill) | KV-cache, 전체 입력 한번에 처리 |
| 5 | ONNX KV INT8 (prefill) | KV-cache prefill, int8 |
| 6 | ONNX KV FP32 (incr) | KV-cache, 이전 턴 prefill + 마지막 턴 incremental |
| 7 | ONNX KV INT8 (incr) | KV-cache incremental, int8 |

모든 방식에서 **speaker_ids 사용** (기존 벤치마크의 누락 문제 해결).


## 모델 크기

| 모델 | 크기 | 원본 대비 |
|---|---|---|
| PyTorch checkpoint | 1,426 MB | — |
| ONNX FP32 (NC / KV 동일) | 622 MB | 44% |
| ONNX INT8 (NC / KV 동일) | 156 MB | 11% |

INT8 양자화: `onnxruntime.quantization.quantize_dynamic` (QInt8). ONNX FP32 모델에 적용.
weight만 INT8로 양자화하고 activation은 런타임에 동적 계산하는 방식.


## A. 정확도 검증 — 수작업 대화

8건의 대화에 대해 각 방식의 TRP(Turn-shift Relevance Probability) 비교.

| 대화 유형 | 입력 대화 |
|---|---|
| Complete answer | hello how are you\<ts\>i am doing well thank you |
| Mid-sentence | what do you think about\<ts\>i think that |
| Location answer | where are you from\<ts\>i am from new york |
| Question ending | i had a great day today\<ts\>oh really what did you do |
| Incomplete thought | tell me about your day\<ts\>well first i went to the |
| Acknowledgment | i just got a new job\<ts\>oh congratulations |
| Single turn | hi there how are you doing today |
| Goodbye | it was nice talking to you\<ts\>yeah you too goodbye |

| 대화 유형 | PyTorch | NC FP32 | NC INT8 | KV FP32 pf | KV INT8 pf | KV FP32 inc | KV INT8 inc |
|---|---|---|---|---|---|---|---|
| Complete answer | 0.0283 | 0.0283 | 0.0270 | 0.0283 | 0.0270 | 0.0283 | 0.0267 |
| Mid-sentence | 0.0005 | 0.0005 | 0.0007 | 0.0005 | 0.0007 | 0.0005 | 0.0009 |
| Location answer | 0.4501 | 0.4501 | 0.1577 | 0.4501 | 0.1577 | 0.4501 | 0.2030 |
| Question ending | 0.9991 | 0.9991 | 0.9964 | 0.9991 | 0.9964 | 0.9991 | 0.9985 |
| Incomplete thought | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Acknowledgment | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 |
| Single turn | 0.9991 | 0.9991 | 0.9988 | 0.9991 | 0.9988 | 0.9991 | 0.9988 |
| Goodbye | 0.9326 | 0.9326 | 0.9164 | 0.9326 | 0.9164 | 0.9326 | 0.9286 |

### PyTorch 대비 최대 절대 오차

| 방식 | Max Diff |
|---|---|
| ONNX NC FP32 | 7.24e-06 |
| ONNX KV FP32 (pf) | 7.24e-06 |
| ONNX KV FP32 (incr) | 7.24e-06 |
| ONNX NC INT8 | **2.92e-01** |
| ONNX KV INT8 (pf) | **2.92e-01** |
| ONNX KV INT8 (incr) | **2.47e-01** |

**FP32 ONNX**: PyTorch와 사실상 동일 (오차 < 0.00001).

**INT8**: "Location answer" (TRP≈0.45, 모델 불확실 구간)에서 0.45→0.16으로 큰 편차 발생.
턴 전환(TRP>0.9) / 유지(TRP<0.01) 판정 자체는 뒤집히지 않지만, 중간값 구간에서 정확도 저하가 있음.


## B. 속도 — 토큰 수별 스케일링 (full forward)

입력 토큰 수에 따른 latency. Warmup 3회, 측정 15회, mean 기준.

> **참고**: 이 테스트는 **캐시 미적용 상태에서 전체 토큰을 한번에 넣는 single full forward** 측정.
> NC와 KV prefill이 비슷하거나 KV가 약간 느린 이유는, 둘 다 동일한 연산을 수행하되
> KV 모델은 캐싱을 위한 오버헤드가 추가되기 때문.
> KV cache의 실제 이점은 Section C의 incremental 모드(prefix 캐싱 후 새 토큰만 처리)에서 나타남.

| 토큰 | PyTorch | NC FP32 | NC INT8 | KV FP32 pf | KV INT8 pf |
|---|---|---|---|---|---|
| 4 | 362 ms | 90 ms | 30 ms | 89 ms | 28 ms |
| 11 | 403 ms | 112 ms | 38 ms | 119 ms | 40 ms |
| 19 | 444 ms | 149 ms | 53 ms | 155 ms | 57 ms |
| 36 | 531 ms | 243 ms | 103 ms | 252 ms | 114 ms |
| 63 | 680 ms | 367 ms | 176 ms | 389 ms | 198 ms |

### PyTorch 대비 Speedup

| 토큰 | NC FP32 | NC INT8 | KV FP32 pf | KV INT8 pf |
|---|---|---|---|---|
| 4 | 4.0x | **12.2x** | 4.1x | **13.1x** |
| 11 | 3.6x | **10.6x** | 3.4x | **10.0x** |
| 19 | 3.0x | **8.4x** | 2.9x | **7.9x** |
| 36 | 2.2x | **5.2x** | 2.1x | **4.6x** |
| 63 | 1.9x | **3.9x** | 1.7x | **3.4x** |

INT8이 전 구간에서 가장 빠름. 짧은 시퀀스(4t)에서 13x, 긴 시퀀스(63t)에서도 3.4x 이상.


## C. KV Cache Incremental 분석

prefix를 캐싱한 후, 새 토큰만 추가 처리하는 incremental 모드 성능.
Warmup 5회, 측정 20회, mean 기준.

### prefix 크기별 latency (ms)

| Prefix | NC FP32 | NC INT8 | KV pf FP32 | KV pf INT8 | KV 1t FP32 | KV 1t INT8 | KV 10t FP32 | KV 10t INT8 |
|---|---|---|---|---|---|---|---|---|
| 19 | 147 | 57 | 157 | 60 | 84 | 29 | 119 | 43 |
| 64 | 373 | 184 | 368 | 184 | 83 | 32 | 113 | 48 |
| 128 | 742 | 373 | 733 | 407 | 87 | 37 | 122 | 56 |

핵심:
- **KV 1-token FP32**: 83–87ms — prefix 크기에 무관하게 거의 일정
- **KV 1-token INT8**: 29–37ms — prefix 크기에 무관하게 거의 일정
- **KV 10-token INT8**: 43–56ms
- full forward는 prefix에 비례해 증가 (NC FP32: 147→742ms)

### Incremental 정확도 검증

| 방식 | full prefill TRP | incr TRP | diff |
|---|---|---|---|
| KV FP32 | 0.028257 | 0.028257 | 0.00e+00 |
| KV INT8 | 0.027012 | 0.033061 | 6.05e-03 |

FP32: 완전 일치. INT8: 약간의 오차 존재 (양자화 영향).


## D. SODA 테스트셋 정확도 검증

SODA test split 147,198건 중 5,000건 균등 샘플링. 각 방식 1회 추론.
스크립트: `scripts/benchmark_turngpt_accuracy.py`

- 토큰 분포: min=24, median=129, max=605, mean=141
- raw 데이터: `docs/turngpt_accuracy_soda_5k.csv`

### D1. PyTorch 대비 전체 통계

| 방식 | Mean Diff | Std Diff | Max Diff | MAE | Correlation |
|---|---|---|---|---|---|
| NC FP32 | -0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| KV FP32 (pf) | -0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| KV FP32 (incr) | -0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| NC INT8 | -0.0321 | 0.0903 | **0.6199** | 0.0647 | 0.9562 |
| KV INT8 (pf) | -0.0321 | 0.0903 | **0.6199** | 0.0647 | 0.9562 |
| KV INT8 (incr) | -0.0310 | 0.0844 | **0.5548** | 0.0608 | 0.9618 |

**FP32**: 모든 방식에서 PyTorch와 사실상 동일 (MAE=0, 상관계수=1. 부동소수점 오차 ~1e-6 수준).

**INT8**: 평균 오차(MAE) 0.06, 최대 0.62. 전체적으로 TRP를 낮게 추정하는 경향 (Mean Diff=-0.03).

### D2. PyTorch TRP 구간별 MAE

| TRP 구간 | N | NC FP32 | NC INT8 | KV FP32 pf | KV INT8 pf | KV FP32 inc | KV INT8 inc |
|---|---|---|---|---|---|---|---|
| Low (< 0.05) | 156 | 0.0000 | 0.0168 | 0.0000 | 0.0168 | 0.0000 | 0.0151 |
| Mid-low (0.05–0.3) | 737 | 0.0000 | 0.0663 | 0.0000 | 0.0663 | 0.0000 | 0.0603 |
| **Mid (0.3–0.7)** | **1,348** | **0.0000** | **0.0979** | **0.0000** | **0.0979** | **0.0000** | **0.0946** |
| Mid-high (0.7–0.95) | 1,693 | 0.0000 | 0.0717 | 0.0000 | 0.0717 | 0.0000 | 0.0667 |
| High (> 0.95) | 1,066 | 0.0000 | 0.0175 | 0.0000 | 0.0175 | 0.0000 | 0.0160 |

INT8 오차는 중간값(0.3–0.7) 구간에서 가장 크고 (MAE 0.098), 극단값에서 작음 (< 0.02).

### D3. 턴 판정 일치율 (threshold=0.5)

| 방식 | 일치 | 불일치 | 일치율 |
|---|---|---|---|
| NC FP32 | 5,000/5,000 | 0 | **100.0%** |
| KV FP32 (pf) | 5,000/5,000 | 0 | **100.0%** |
| KV FP32 (incr) | 5,000/5,000 | 0 | **100.0%** |
| NC INT8 | 4,628/5,000 | 372 | 92.6% |
| KV INT8 (pf) | 4,628/5,000 | 372 | 92.6% |
| KV INT8 (incr) | 4,627/5,000 | 373 | 92.5% |

INT8은 5,000건 중 **372건(7.4%)에서 턴 판정이 뒤집힘** (threshold 0.5 기준).


## 종합 결론

### 정확도 (SODA 5,000건 기준)

| 방식 | MAE | Max Diff | 턴 판정 일치율 |
|---|---|---|---|
| ONNX FP32 (NC / KV) | 0.0000 | 0.0000 | 100.0% |
| ONNX INT8 | 0.0647 | **0.6199** | **92.6%** |

### 속도 (실사용 시나리오별)

| 시나리오 | 권장 방식 | 예상 latency |
|---|---|---|
| 1회성 full forward (짧은 대화) | ONNX NC INT8 | 30–57ms (4–19t) |
| 1회성 full forward (긴 대화) | ONNX NC INT8 | 103–176ms (36–63t) |
| 스트리밍 ASR (토큰 추가) | ONNX KV FP32 1-token | ~85ms (prefix 무관) |
| 스트리밍 ASR (토큰 추가) | ONNX KV INT8 1-token | ~30ms (prefix 무관) |
| 문장 단위 업데이트 | ONNX KV FP32 10-token | ~118ms |
| 문장 단위 업데이트 | ONNX KV INT8 10-token | ~49ms |

### 트레이드오프 요약

| | FP32 | INT8 |
|---|---|---|
| 모델 크기 | 622 MB | 156 MB |
| 정확도 (MAE) | PyTorch 동일 | 0.065 |
| 턴 판정 일치율 | 100% | 92.6% |
| KV 1-token | ~85 ms | ~30 ms |

INT8은 속도/크기에서 유리하지만, **7.4%의 턴 판정 불일치**가 발생.
실시간 대화에서 이 정도의 불일치가 허용 가능한지는 실사용 테스트로 판단 필요.
