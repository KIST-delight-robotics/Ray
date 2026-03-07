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

#### Checkpoint

TurnGPT uses `load_from_checkpoint` (PyTorch Lightning). Set `TurnGPTConfig.checkpoint_path` to the checkpoint file.

For integration and stress tests, export the env var:

```bash
export TURNGPT_CHECKPOINT_PATH=/path/to/turngpt.ckpt
```

#### Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `checkpoint_path` | `""` | Path to TurnGPT checkpoint file |
| `device` | `"cpu"` | Torch device (`"cpu"` or `"cuda"`) |
| `max_context_tokens` | `1024` | Max tokens before old turns are evicted (GPT-2 limit). `0` = no limit |

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

## Module Structure

```
turn_taking/
├── __init__.py
├── exceptions.py       # TurnTakingError, VAPError, TurnGPTError, TurnDetectorError
├── vap.py              # VAPWrapper(IVAP)
├── turngpt.py          # TurnGPTWrapper(ITurnGPT)
├── turn_detector.py    # TurnDetector(ITurnDetector)
└── README.md
```
