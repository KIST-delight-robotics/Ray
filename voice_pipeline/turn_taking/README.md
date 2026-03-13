# Turn-Taking Module

Wraps external turn-taking models (VAP, TurnGPT) and fuses their outputs via `TurnDetector`.

## External Dependencies

### VAP (Voice Activity Projection)

Repository: <https://github.com/ErikEkstedt/VoiceActivityProjection>

#### Setup

```bash
git clone https://github.com/ErikEkstedt/VoiceActivityProjection.git external/VoiceActivityProjection
uv pip install -e external/VoiceActivityProjection
uv pip install einops omegaconf
```

#### Model File

A pre-trained state_dict is included in the repo at `example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt`. Set `VAPConfig.model_path` to the file path.

#### CPC Checkpoint Caveat

The VAP encoder uses a CPC (Contrastive Predictive Coding) component that may **auto-download** a checkpoint on first use if the expected file is missing. See `vap/encoder_components.py` lines 371–377. Ensure network access is available on first run, or pre-download the checkpoint.

#### Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | `""` | Path to VAP state_dict `.pt` file |
| `context_sec` | `20.0` | Rolling buffer duration (seconds) |
| `step_sec` | `0.1` | Inference interval (seconds) |
| `tt_time` | `0.5` | Turn-taking lookahead for averaging (seconds) |
| `device` | `"cpu"` | Torch device (`"cpu"` or `"cuda"`) |
| `vad_threshold` | `0.5` | Threshold for `user_is_speaking` |

### TurnGPT

Repository: <https://github.com/ErikEkstedt/TurnGPT>

#### Setup

```bash
git clone https://github.com/ErikEkstedt/TurnGPT.git external/TurnGPT
uv pip install -e external/TurnGPT
```

#### Backends

**PyTorch** (default): Uses `load_from_checkpoint` (PyTorch Lightning). Set `TurnGPTConfig.checkpoint_path`.

**ONNX** (recommended for RPi): Uses ONNX Runtime. Set `TurnGPTConfig.onnx_model_path` and `tokenizer_path`. Requires `onnxruntime` and `transformers` packages. PyTorch is still required (tokenization uses torch tensors).

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

#### Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `checkpoint_path` | `""` | Path to TurnGPT checkpoint file (PyTorch mode) |
| `onnx_model_path` | `""` | Path to ONNX model file. When set, uses ONNX Runtime |
| `tokenizer_path` | `""` | Path to saved tokenizer directory (required for ONNX mode) |
| `device` | `"cpu"` | Torch device (PyTorch mode only) |
| `max_context_tokens` | `1024` | Max tokens before old turns are evicted (GPT-2 limit). `0` = no limit |
| `onnx_threads` | `2` | ONNX Runtime intra-op threads. 2 is optimal on RPi 5 (4-core) |

### TurnDetector

Combines VAP and TurnGPT outputs with timing heuristics. No external dependencies beyond the two wrappers above.

#### Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vap_user_threshold` | `0.5` | p_now/p_fut below this = "favors robot" |
| `min_gap_time_sec` | `0.5` | Sustained VAP robot-favor duration for turn-shift |
| `turngpt_thresholds` | `((0.3, 0.5), (0.2, 1.0), (0.1, 2.0), (0.0, 3.0))` | Graduated (prob, timeout_sec) pairs |
| `interrupt_user_threshold` | `0.5` | p_now/p_fut above this = "favors user" |
| `prepare_turngpt_threshold` | `0.2` | TurnGPT prob above this triggers prepare |
| `prepare_timeout_sec` | `0.2` | Time since last ASR change to trigger prepare |
| `prepare_similarity_threshold` | `0.8` | Skip prepare if text similarity ≥ this |

### MaAI VAP

Repository: <https://github.com/Seongbuming/MaAI>

#### Setup

```bash
git clone https://github.com/Seongbuming/MaAI.git external/MaAI
uv pip install -e external/MaAI
```

#### Backends

**Full ONNX** (default, recommended): Both CPC encoder and GPT transformer run via ONNX Runtime. No PyTorch inference at runtime. Mean latency ~24ms on RPi 5.

**Hybrid** (`use_onnx_transformer=False`): ONNX encoder + PyTorch transformer. Optional `torch.compile` for ~22% speedup. Useful if ONNX export fails for a new model variant.

Both ONNX models are auto-exported from the loaded MaAI weights at initialization. No manual export step needed.

#### Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lang` | `"en"` | Language code for MaAI model |
| `frame_rate` | `10` | VAP inference frame rate (Hz) |
| `context_len_sec` | `5.0` | KV cache context length (seconds) |
| `vad_threshold` | `0.5` | Threshold for `user_is_speaking` |
| `ort_threads` | `1` | ONNX Runtime intra-op threads. 1 is optimal on RPi 5 |
| `pt_threads` | `1` | PyTorch threads (minimal impact in full ONNX mode) |
| `use_onnx_transformer` | `True` | Use ONNX transformer (True) or PyTorch (False) |
| `use_torch_compile` | `True` | Enable torch.compile (PyTorch mode only) |

#### RPi 5 Performance (full ONNX, ort_threads=1)

| Stage | Mean | % of Total |
|-------|------|------------|
| Encoder (2ch) | 16.2ms | 67% |
| Transformer | 7.8ms | 32% |
| Total | **24.0ms** | 100% |

RTF: 4.16x (100ms budget). Budget exceeded: 0%.

## Module Structure

```
turn_taking/
├── __init__.py
├── exceptions.py       # TurnTakingError, VAPError, TurnGPTError, TurnDetectorError
├── vap.py              # VAPWrapper(IVAP) — VAP-Realtime
├── maai_vap.py         # MaAIVAPWrapper(IVAP) — MaAI VAP (ONNX)
├── onnx_export.py      # ONNX export wrappers (encoder + transformer)
├── turngpt.py          # TurnGPTWrapper(ITurnGPT)
├── turn_detector.py    # TurnDetector(ITurnDetector)
└── README.md
```
