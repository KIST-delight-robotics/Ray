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

Setup and usage documented when the TurnGPT wrapper is implemented.

## Module Structure

```
turn_taking/
├── __init__.py
├── exceptions.py       # TurnTakingError, VAPError, TurnGPTError, TurnDetectorError
├── vap.py              # VAPWrapper(IVAP)
├── turngpt.py          # TurnGPTWrapper(ITurnGPT)        [future]
├── turn_detector.py    # TurnDetector(ITurnDetector)      [future]
└── README.md
```
