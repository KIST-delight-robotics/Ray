# Turn-Taking Module

Wraps external turn-taking models (VAP, TurnGPT) and fuses their outputs via `TurnDetector`.

## External Dependencies

VAP는 MaAI(아래 "MaAI VAP")만 사용. 오리지널 VoiceActivityProjection은
RPi 실시간 불가(torch ~1.2s/추론, `docs/benchmarks/vap_rpi_benchmark.md`)로 제거됨.

### TurnGPT

Repository: <https://github.com/ErikEkstedt/TurnGPT>

#### Setup

```bash
git clone https://github.com/ErikEkstedt/TurnGPT.git external/TurnGPT
uv pip install -e external/TurnGPT
```

#### Backends

Select via `TurnGPTWrapper._BACKEND = "onnx" | "pytorch"` (class var, 생성 전 변경).

**ONNX** (default, recommended for RPi): Uses ONNX Runtime. Reads `_ONNX_MODEL_PATH`, `_TOKENIZER_PATH`, `_ONNX_THREADS`. Requires `onnxruntime` and `transformers` packages. PyTorch is still required (tokenization uses torch tensors).

**PyTorch**: Uses `load_from_checkpoint` (PyTorch Lightning). Reads `_CHECKPOINT_PATH`, `_DEVICE`.

ONNX models are exported via scripts in `turngpt_training/scripts/`. Place them in `models/turngpt/`. Tokenizer is included at `models/turngpt/tokenizer/`. Four variants available:

| Variant | Size | Notes |
|---------|------|-------|
| `turngpt_v2.onnx` | 623MB | fp32, no KV cache |
| `turngpt_v2_kvcache.onnx` | 623MB | fp32, KV cache |
| `turngpt_v2_int8.onnx` | 157MB | int8 quantized, no KV cache |
| `turngpt_v2_kvcache_int8.onnx` | 157MB | int8 quantized, KV cache |

RPi 5 benchmarks (int8 kvcache, 2 threads): ~42ms/inference, CPU ~189%.

For integration and stress tests, export the env var:

```bash
export TURNGPT_CHECKPOINT_PATH=/path/to/turngpt.ckpt
```

#### `TurnGPTWrapper` 클래스 변수

| 변수 | 값 | 의미 |
|------|------|------|
| `_BACKEND` | `"onnx"` | 추론 백엔드 (`"onnx"` / `"pytorch"`) |
| `_ONNX_MODEL_PATH` | `"models/turngpt/turngpt_v2_kvcache_int8.onnx"` | ONNX 모델 파일 경로 |
| `_TOKENIZER_PATH` | `"models/turngpt/tokenizer"` | ONNX 토크나이저 디렉토리 |
| `_ONNX_THREADS` | `2` | ONNX Runtime intra-op 스레드 수 (RPi 5 4-코어 최적 2) |
| `_CHECKPOINT_PATH` | `"models/turngpt/turngpt.ckpt"` | PyTorch 체크포인트 경로 (PyTorch 모드) |
| `_NUM_LAYERS` | `12` | GPT-2 레이어 수 |
| `_NUM_HEADS` | `12` | GPT-2 attention head 수 |
| `_HEAD_DIM` | `64` | GPT-2 head 차원 |
| `_FALLBACK_PROBABILITY` | `0.0` | 추론 실패/빈 입력 시 반환 확률 |
| `_DEVICE` | `"cpu"` | PyTorch 디바이스 (`"cpu"` / `"cuda"`) |
| `_MAX_CONTEXT_TOKENS` | `1024` | 모델 입력 최대 토큰 수. `0`이면 무제한 |
| `_KEEP_TURNS` | `2` | 토큰 초과 시 유지할 최근 완료 턴 수 |
| `_ONNX_PROVIDERS` | `("CPUExecutionProvider",)` | ONNX Runtime 실행 프로바이더 |

### TurnDetector

Combines VAP and TurnGPT outputs with timing heuristics. No external dependencies beyond the two wrappers above.

#### `TurnDetector.__init__` 인자

| 인자 | Default | 의미 |
|------|---------|------|
| `vap` | — | VAP 모델 (`IVAP`). 세션마다 신규 `AsyncVAP` 주입 |
| `turngpt` | — | TurnGPT 어댑터 (`ThreadedTurnGPT` / `SyncTurnGPTAdapter`) |
| `embedder` | — | 임베딩 공급자 (`IEmbedder`). prepare 유사도 게이트용 |

#### `TurnDetector` 클래스 변수

| 변수 | 값 | 의미 |
|------|------|------|
| `_VAP_USER_THRESHOLD` | `0.5` | p_now/p_fut가 이 값 미만이면 "robot 선호" |
| `_MIN_GAP_TIME_SEC` | `0.5` | turn-shift 판단에 필요한 VAP robot-선호 지속 시간 (초) |
| `_TURNGPT_THRESHOLDS` | `((0.3, 0.5), (0.2, 1.0), (0.1, 2.0), (0.0, 3.0))` | 단계별 `(prob 하한, timeout 초)` |
| `_INTERRUPT_USER_THRESHOLD` | `0.5` | p_now/p_fut가 이 값 초과면 "user 선호" |
| `_PREPARE_TURNGPT_THRESHOLD` | `0.2` | prepare 트리거 TurnGPT 확률 |
| `_PREPARE_TIMEOUT_SEC` | `0.2` | 마지막 ASR 변화 후 prepare 트리거까지 시간 (초) |
| `_SIMILARITY_THRESHOLD` | `0.8` | 직전 prepare 텍스트와의 유사도 이 값 이상이면 skip |

### MaAI VAP

Repository: <https://github.com/MaAI-Kyoto/MaAI>

#### Setup

```bash
git clone https://github.com/MaAI-Kyoto/MaAI.git external/MaAI
uv pip install -e external/MaAI
```

#### ONNX Export

ONNX models must be pre-exported before use. Export script:

```bash
uv run python -m scripts.export_maai_onnx [--lang en] [--frame-rate 10]
```

Output: `models/maai/encoder_{frame_rate}hz.onnx`, `models/maai/transformer_{lang}.onnx`

#### Backends

**Full ONNX** (default, recommended): Both CPC encoder and GPT transformer run via ONNX Runtime. No PyTorch dependency at inference time. Mean latency ~24ms on RPi 5.

**Hybrid** (`_USE_ONNX_TRANSFORMER = False`): ONNX encoder + PyTorch transformer. Optional `torch.compile` via `_USE_TORCH_COMPILE` for ~22% speedup. Requires `maai` package and `torch`. 생성 전 class var 변경.

#### `MaAIVAPModel.__init__` 인자

| 인자 | Default | 의미 |
|------|---------|------|
| `tts_sample_rate` | — | Robot(TTS) 출력 샘플레이트 (필수) |

#### `MaAIVAPModel` 클래스 변수

| 변수 | 값 | 의미 |
|------|------|------|
| `ENCODER_ONNX_PATH` | `"models/maai/encoder_10hz_5s.onnx"` | encoder ONNX 기본 경로 (외부 참조 가능) |
| `TRANSFORMER_ONNX_PATH` | `"models/maai/transformer_en_5s.onnx"` | transformer ONNX 기본 경로 (외부 참조 가능) |
| `_USE_ONNX_TRANSFORMER` | `True` | transformer 백엔드 선택 (True=ONNX / False=PyTorch fallback) |
| `_USE_TORCH_COMPILE` | `True` | torch.compile 활성화 (PyTorch fallback 전용) |
| `_FRAME_RATE` | `10` | VAP 추론 프레임 레이트 (Hz) |
| `_CONTEXT_LEN_SEC` | `5.0` | KV 캐시 컨텍스트 길이 (초) |
| `_ORT_THREADS` | `1` | ONNX Runtime intra-op 스레드 수 (RPi 5 최적 1) |
| `_PT_THREADS` | `1` | PyTorch 스레드 수 (PyTorch fallback 전용) |
| `_DEFAULT_RESULT` | `VAPResult(0.0, 0.0, False)` | 초기/실패 시 반환값 |
| `_LANG` | `"en"` | MaAI 언어 코드 (PyTorch fallback 전용) |
| `_VAD_THRESHOLD` | `0.5` | `user_is_speaking` 임계값 |
| `_TORCH_DEVICE` | `"cpu"` | MaAI PyTorch 디바이스 |
| `_TORCH_COMPILE_MODE` | `"reduce-overhead"` | torch.compile 모드 |
| `_MAAI_DIM` | `256` | MaAI hidden dimension |
| `_MAAI_NUM_HEADS` | `4` | MaAI attention head 수 |
| `_MAAI_HEAD_DIM` | `64` | MaAI head 차원 (`_MAAI_DIM / _MAAI_NUM_HEADS`) |
| `_MAAI_CH_LAYERS` | `1` | MaAI channel-attention 레이어 수 |
| `_MAAI_CROSS_LAYERS` | `3` | MaAI cross-attention 레이어 수 |
| `_FRAME_CTX_PADDING` | `320` | encoder 입력 padding (MaAI 아키텍처 고정) |

#### RPi 5 Performance (full ONNX, ort_threads=1)

| Stage | Mean | % of Total |
|-------|------|------------|
| Encoder (2ch) | 16.2ms | 67% |
| Transformer | 7.8ms | 32% |
| Total | **24.0ms** | 100% |

RTF: 4.16x (100ms budget). Budget exceeded: 0%.

## Module Structure

```
voice_pipeline/turn_taking/
├── __init__.py
├── exceptions.py       # TurnTakingError, VAPError, TurnGPTError, TurnDetectorError
├── maai_vap.py         # MaAIVAPModel — MaAI ONNX inference (synchronous)
├── threaded_vap.py     # ThreadedVAP(IVAP) — runs a VAP model on a bg thread (10Hz)
├── turngpt.py          # TurnGPTWrapper(ITurnGPT)
├── threaded_turngpt.py # ThreadedTurnGPT + SyncTurnGPTAdapter — bg thread wrapper
├── turn_detector.py    # TurnDetector(ITurnDetector)
└── README.md

scripts/
├── export_maai_onnx.py     # MaAI ONNX export (wrappers + CLI)
├── generate_test_wav.py    # Test audio file generator
├── bench/
│   ├── bench_turn_model.py
│   ├── bench_turn_concurrent.py
│   └── bench_memory_retrieve.py
└── hardware/               # Live hardware test scripts
```
