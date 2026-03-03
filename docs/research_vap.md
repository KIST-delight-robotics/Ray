# VoiceActivityProjection Source Audit (for real-time spoken dialogue use)

Repository cloned:
- URL: `https://github.com/ErikEkstedt/VoiceActivityProjection`
- Commit checked: `f39a78b` (`git log -1 --oneline`)

## 1) Installation instructions

### What the repo README says

Quoted from `README.md`:

```text
README.md:51-61
## Installation

* Create conda env: `conda create -n voice_activity_projection python=3`
  - source env: `conda source voice_activity_projection`
  - Working with `python 3.9` but I don't think it matters too much...
* PyTorch: `conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch`
    - Have not tested all versions but should work from `torch 1.12.1` as of time of writing...
* Install **`VoiceActivityProjection`** (this repo):
  * cd to root directory and run:
    * `pip install -r requirements.txt`
    * `pip install -e .`
```

### Packaging metadata in this repo

Quoted from `setup.py`:

```python
setup.py:5-13
setup(
    name="vap",
    version="0.0.0",
    description="Voice Activity Projection: Self-Supervised Learning of Turn-taking Events",
    author="erikekst",
    author_email="erikekst@kth.se",
    url="https://github.com/ErikEkstedt/VoiceActivityProjection",
    packages=["vap"],
)
```

Quoted from `requirements.txt`:

```text
requirements.txt:1-8
einops
hydra-core
numpy
omegaconf
pytorch-lightning
wandb
pytest
praat-parselmouth
```

## 2) How to load the model

### Loading from a state dict (the path implemented in `run.py`)

```python
run.py:197-202
if args.checkpoint is None:
    print("From state-dict: ", args.state_dict)
    model = VapGPT(conf)
    sd = torch.load(args.state_dict)
    model.load_state_dict(sd)
```

### Checkpoint loading path is not implemented

```python
run.py:203-207
else:
    from vap.train import VAPModel

    print("From Lightning checkpoint: ", args.checkpoint)
    raise NotImplementedError("Not implemeted from checkpoint...")
```

### Device + eval mode

```python
run.py:208-213
device = "cpu"
if torch.cuda.is_available():
    model = model.to("cuda")
    device = "cuda"
model = model.eval()
```

### Core model class and defaults

```python
vap/model.py:125-133
class VapGPT(nn.Module):
    def __init__(self, conf: Optional[VapConfig] = None):
        super().__init__()
        if conf is None:
            conf = VapConfig()
        self.conf = conf
        self.sample_rate = conf.sample_rate
        self.frame_hz = conf.frame_hz
```

```python
vap/model.py:42-47
@dataclass
class VapConfig:
    sample_rate: int = 16_000
    frame_hz: int = 50
    bin_times: List[float] = field(default_factory=lambda: BIN_TIMES)
```

## 3) Inference API

### 3.1 Input format

### Input tensor shape expected by the model

```python
vap/model.py:169-173
def encode_audio(self, audio: torch.Tensor) -> Tuple[Tensor, Tensor]:
    assert (
        audio.shape[1] == 2
    ), f"audio VAP ENCODER: {audio.shape} != (B, 2, n_samples)"
```

So the model expects **stereo waveform tensor** shaped `(B, 2, n_samples)`.

### How `run.py` constructs that input

```python
run.py:217-221
waveform, _ = load_waveform(args.audio, sample_rate=model.sample_rate)
...
if waveform.shape[0] == 1:
    waveform = torch.cat((waveform, torch.zeros_like(waveform)))
waveform = waveform.unsqueeze(0)
```

This means:
- `load_waveform(...)` gives `(channels, n_samples)`.
- If mono (1 channel), the script appends a silent second channel.
- Then it adds batch dim to make `(1, 2, n_samples)`.

### Audio loading behavior (resample support)

```python
vap/audio.py:39-45
def load_waveform(
    path: str,
    sample_rate: Optional[int] = 16000,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    mono: bool = False,
) -> Tuple[torch.Tensor, int]:
```

```python
vap/audio.py:65-69
if sample_rate is not None:
    if sr != sample_rate:
        x = AF.resample(x, orig_freq=sr, new_freq=sample_rate)
        sr = sample_rate
return x, sr
```

### 3.2 Output format

### `model.probs(...)` return dictionary

```python
vap/model.py:212-218
ret = {
    "probs": probs,
    "vad": vad,
    "p_now": p_now,
    "p_future": p_future,
    "H": H,
}
```

Possible extra key:

```python
vap/model.py:220-224
if vad is not None:
    labels = self.objective.get_labels(vad)
    ret["loss"] = self.objective.loss_vap(
        out["logits"], labels, reduction="none"
    )
```

### Raw forward output (before `probs()` post-processing)

```python
vap/model.py:257-264
# Outputs
v1 = self.va_classifier(out["x1"])
v2 = self.va_classifier(out["x2"])
vad = torch.cat((v1, v2), dim=-1)
logits = self.vap_head(out["x"])

ret = {"logits": logits, "vad": vad}
```

### 3.3 What each returned value means

### `probs`

```python
vap/model.py:188-190
out = self(waveform)
probs = out["logits"].softmax(dim=-1)
vad = out["vad"].sigmoid()
```

`probs` is softmax over VAP discrete classes for each frame.

### `vad`

From the same snippet above, this is sigmoid of the VAD logits (`out["vad"]`), giving per-speaker voice activity probabilities.

### `H` (entropy)

```python
vap/model.py:192-203
# Calculate entropy over each projection-window prediction ...
h = -probs * probs.log2()  # Entropy
H = h.sum(dim=-1)  # average entropy per frame
```

`H` is per-frame entropy over the VAP class distribution.

### `p_now` and `p_future`

```python
vap/model.py:205-210
p_now = self.objective.probs_next_speaker_aggregate(
    probs, from_bin=now_lims[0], to_bin=now_lims[-1]
)
p_future = self.objective.probs_next_speaker_aggregate(
    probs, from_bin=future_lims[0], to_bin=future_lims[1]
)
```

Default limits are defined here:

```python
vap/model.py:185-187
now_lims: List[int] = [0, 1],
future_lims: List[int] = [2, 3],
```

Bin definition:

```python
vap/model.py:19
BIN_TIMES: list = [0.2, 0.4, 0.6, 0.8]
```

```python
vap/objective.py:10-11
def bin_times_to_frames(bin_times: List[float], frame_hz: int) -> List[int]:
    return (torch.tensor(bin_times) * frame_hz).long().tolist()
```

At default `frame_hz=50`, bins become `[10, 20, 30, 40]` frames, i.e. 2.0 s total horizon (`0.2+0.4+0.6+0.8`).

Aggregation logic:

```python
vap/objective.py:199-204
abp = states[:, :, from_bin : to_bin + 1].sum(-1)  # sum speaker activity bins
# Dot product over all states
p_all = torch.einsum("bid,dc->bic", probs, abp)
# normalize
p_all /= p_all.sum(-1, keepdim=True) + 1e-5
return p_all
```

So `p_now` and `p_future` are per-frame, per-speaker normalized probabilities derived by aggregating selected future bins.

## 4) Real-time / streaming usage patterns

## Pattern A: Chunked long-audio inference in `run.py`

`step_extraction(...)` folds waveform into overlapping windows and appends only new step frames:

```python
run.py:42-47
chunk_time = context_time + step_time
...
step_samples = int(step_time * model.sample_rate)
chunk_samples = int(chunk_time * model.sample_rate)
```

```python
run.py:55-57
folds = waveform.unfold(
    dimension=-1, size=chunk_samples, step=step_samples
).permute(2, 0, 1, 3)
```

```python
run.py:85-92
o = model.probs(w.to(device))
out["vad"] = torch.cat([out["vad"], o["vad"][:, -step_frames:]], dim=1)
out["p_now"] = torch.cat([out["p_now"], o["p_now"][:, -step_frames:]], dim=1)
out["p_future"] = torch.cat(
    [out["p_future"], o["p_future"][:, -step_frames:]], dim=1
)
out["probs"] = torch.cat([out["probs"], o["probs"][:, -step_frames:]], dim=1)
out["H"] = torch.cat([out["H"], o["H"][:, -step_frames:]], dim=1)
```

For trailing remainder not covered by `unfold`, it runs one more chunk and appends omitted frames:

```python
run.py:100-103
if expected_frames != processed_frames:
    omitted_frames = expected_frames - processed_frames
    omitted_samples = model.sample_rate * omitted_frames / model.frame_hz
```

```python
run.py:111-119
w = waveform[..., -chunk_samples:]
o = model.probs(w.to(device))
out["vad"] = torch.cat([out["vad"], o["vad"][:, -omitted_frames:]], dim=1)
...
out["H"] = torch.cat([out["H"], o["H"][:, -omitted_frames:]], dim=1)
```

## Pattern B: Live microphone + rolling context in `sds/run_sds.py`

The SDS script keeps a fixed-size rolling audio buffer and repeatedly calls `model.probs(...)`:

```python
sds/run_sds.py:173-175
n_samples = round(conf.context * conf.sample_rate)
self.x = torch.zeros((1, 2, n_samples))
self.device = "cpu"
```

```python
sds/run_sds.py:218-220
self.x = self.x.roll(-chunk_size, -1)
self.x[0, 0, -chunk_size:] = a.to(self.device)
self.x[0, 1, -chunk_size:] = b.to(self.device)
```

```python
sds/run_sds.py:241-243
out = self.model.probs(self.x)
p = out["p_now"][0, -self.tt_frames :, 0].mean().item()
# p = out["p_future"][0, -self.tt_frames :, 0].mean().item()
```

It then publishes probability over ZMQ:

```python
sds/run_sds.py:249-251
self.socket.send_string(self.conf.topic, zmq.SNDMORE)
self.socket.send(json.dumps(p).encode())  # send a single float
```

### Where `tt_frames` comes from

```python
sds/run_sds.py:182-184
# The number of frames to average the turn-shift probabiltites in
self.tt_frames = round(conf.tt_time * self.model.frame_hz)
```

## 5) Dependencies and whether extra repos are required

## Core dependencies explicitly listed

```text
requirements.txt:1-8
einops
hydra-core
numpy
omegaconf
pytorch-lightning
wandb
pytest
praat-parselmouth
```

## Extra repos referenced in requirements comments

```text
requirements.txt:10-13
# These are dependencies for the othe packages used in this repo
# https://github.com/ErikEkstedt/vap_turn_taking
# https://github.com/ErikEkstedt/datasets_turntaking
datasets
```

```text
requirements.txt:16-21
# must download and install via
# 'pip install -r requirements.txt'
# 'pip install -e .'
# in the respective repos
# git+https://github.com/ErikEkstedt/vap_turn_taking.git
# git+https://github.com/ErikEkstedt/datasets_turntaking.git
```

## Training-specific extra repo (`vap_dataset`)

```text
README.md:35
Training the model requires the private [vap_dataset](https://github.com/ErikEkstedt/vap_dataset) repo (for now)
```

```python
vap/train.py:21
from vap_dataset.datamodule import VapDataModule
```

```python
vap/train_mono.py:19
from vap_dataset.datamodule import VapDataModule
```

## Inference code path imports (no `vap_dataset` import in `run.py`)

```python
run.py:8-16
from vap.model import VapGPT, VapConfig, load_older_state_dict
from vap.audio import load_waveform
from vap.utils import (
    batch_to_device,
    everything_deterministic,
    tensor_dict_to_json,
    write_json,
)
from vap.plot_utils import plot_stereo
```

## 6) Caveats / gotchas (source-backed)

## A) `--checkpoint` path in `run.py` is currently unusable

```python
run.py:206
raise NotImplementedError("Not implemeted from checkpoint...")
```

## B) Chunk CLI args are defined but not passed into `step_extraction(...)`

CLI args:

```python
run.py:169-179
parser.add_argument(
    "--chunk_time",
    type=float,
    default=30,
    help="Duration of each chunk processed by model",
)
parser.add_argument(
    "--step_time",
    type=float,
    default=5,
    help="Increment to process in a step",
)
```

Call site:

```python
run.py:234-237
if args.chunk:
    # raise NotImplementedError("step extraction not implemented")
    out = step_extraction(waveform, model, device)
```

`args.chunk_time` and `args.step_time` are not forwarded; `step_extraction` therefore uses its defaults (`context_time=20`, `step_time=5`) from:

```python
run.py:27-28
def step_extraction(
    ...,
    context_time=20,
    step_time=5,
```

## C) Long-input memory warning hard-coded in `run.py`

```python
run.py:223-229
# Maximum known duration with a 24Gb 'NVIDIA GeForce RTX 3090' is 164s
if duration > 160:
    print(
        f"WARNING: Can't fit {duration} > 160s on 24Gb 'NVIDIA GeForce RTX 3090' GPU"
    )
    print("WARNING: Change code if this is not what you want.")
    args.chunk = True
```

## D) `VapGPT.probs` shadows input `vad` argument

Signature has optional `vad` input:

```python
vap/model.py:181-185
def probs(
    self,
    waveform: Tensor,
    vad: Optional[Tensor] = None,
```

Then local variable is reassigned:

```python
vap/model.py:190
vad = out["vad"].sigmoid()
```

Then condition uses reassigned variable:

```python
vap/model.py:220-224
if vad is not None:
    labels = self.objective.get_labels(vad)
    ret["loss"] = self.objective.loss_vap(
        out["logits"], labels, reduction="none"
    )
```

So the passed-in `vad` argument is not used as-is in this method; the branch condition is applied to model-produced `vad` after reassignment.

## E) CPC encoder checkpoint can trigger download if missing

```python
vap/encoder_components.py:371-377
if exists(CHECKPOINTS["cpc"]):
    checkpoint = torch.load(CHECKPOINTS["cpc"], map_location="cpu")
else:
    checkpoint_url = "https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints/60k_epoch4-d0f474de.pt"
    checkpoint = torch.hub.load_state_dict_from_url(
        checkpoint_url, progress=False, map_location="cpu"
    )
```

## F) Some scripts import packages not listed in `requirements.txt`

`run.py` imports matplotlib and uses tqdm in chunk mode:

```python
run.py:3
import matplotlib.pyplot as plt
```

```python
run.py:77
from tqdm import tqdm
```

SDS script imports pyaudio and zmq:

```python
sds/run_sds.py:2
import pyaudio
```

```python
sds/run_sds.py:6
import zmq
```

`vap/extraction.py` imports tqdm and pandas:

```python
vap/extraction.py:2
from tqdm import tqdm
```

```python
vap/extraction.py:4
import pandas as pd
```

But those are not present in `requirements.txt:1-8`.

## G) `vap/extraction.py` has a likely signature mismatch in non-step path

`extract()` calls:

```python
vap/extraction.py:268
out = self.model(waveform, vad=vad)
```

Current `VapGPT.forward` signature is:

```python
vap/model.py:249
def forward(self, waveform: Tensor, attention: bool = False) -> Dict[str, Tensor]:
```

So `vad=...` is not a declared parameter in this `forward`.

## 7) Minimal practical recipe for real-time SDS integration (from source)

1. Load model from `.pt` state dict (`run.py:197-202` or `sds/run_sds.py:51-56`).
2. Keep a rolling tensor `x` of shape `(1, 2, context_seconds*16000)` (`sds/run_sds.py:173-175`).
3. For each new stereo chunk: deinterleave, roll buffer, write newest samples (`sds/run_sds.py:210-220`).
4. Run `out = model.probs(x)` (`sds/run_sds.py:241`).
5. Use a windowed mean over recent frames, e.g. `out["p_now"][0, -tt_frames:, 0].mean()` (`sds/run_sds.py:242`).
6. Publish/store this scalar as your turn-taking control signal (`sds/run_sds.py:249-255`).

All steps above are directly implemented in `sds/run_sds.py`.
