# TurnGPT research notes (source-audited)

Repository: `https://github.com/ErikEkstedt/TurnGPT` (local clone in `TurnGPT/`)

## 1) Installation

### What the repo itself says to install

The README gives these exact steps:

> "Create conda env: `conda create -n turngpt python=3`"  
> "PyTorch: `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`"  
> "Dependencies: `pip install -r requirements.txt`"  
> "Install [Datasets turn-taking](https://github.com/ErikEkstedt/datasets_turntaking)"  
> "cd into this repo and install turngpt: `pip install -e .`"  
> Source: `README.md:27-35`

`requirements.txt` contains:

```txt
transformers
tokenizers
pytorch-lightning
wandb
matplotlib
einops
pytest
```

Source: `requirements.txt:1-7`

`setup.py` installs package `turngpt`:

> `setup(name="turngpt", ... packages=["turngpt"])`  
> Source: `setup.py:4-12`

### Is `datasets_turntaking` required?

For **training with `turngpt/train.py`**, yes:

> `from datasets_turntaking import DialogTextDM`  
> Source: `turngpt/train.py:9`

For **core model inference API** (`turngpt/model.py`, `turngpt/tokenizer.py`, `turngpt/generation.py`), imports are from `transformers` / `torch` / local `turngpt.*`, e.g.:

> `from transformers import GPT2LMHeadModel, GPT2Config`  
> Source: `turngpt/model.py:6`

> `from transformers import AutoTokenizer`  
> Source: `turngpt/tokenizer.py:10`

> `import torch`  
> Source: `turngpt/generation.py:1`

No `datasets_turntaking` import appears in those inference modules.

## 2) How to load the model

### A) Load from checkpoint (trained TurnGPT)

The README usage shows:

> `model = TurnGPT.load_from_checkpoint("PATH/TO/checkpoint.ckpt")`  
> Source: `README.md:156-159`

`generation.py` and `model.py` also load checkpoints this way:

> `model = TurnGPT.load_from_checkpoint(args.checkpoint)`  
> Source: `turngpt/generation.py:418`

> `model = TurnGPT.load_from_checkpoint(chpt).to("cuda")`  
> Source: `turngpt/model.py:713`

Checkpoint loading restores tokenizer (if saved in checkpoint) and resizes embeddings:

> `checkpoint["tokenizer"] = self.tokenizer`  
> Source: `turngpt/model.py:546-549`

> `self.tokenizer = checkpoint["tokenizer"]`  
> `self.transformer.resize_token_embeddings(new_num_tokens=len(self.tokenizer))`  
> Source: `turngpt/model.py:550-561`

### B) Fresh initialization (base GPT2/DialoGPT + TurnGPT wrapper)

Model constructor loads GPT-2/DialoGPT family via Hugging Face:

> `self.transformer = load_transformer(pretrained_model_name_or_path, pretrained=pretrained, **model_kwargs)`  
> Source: `turngpt/model.py:275-278`

Supported names are explicitly checked:

> `if not ("gpt2" in ... or "dialogpt" in ...)` -> `raise NotImplementedError(...)`  
> Source: `turngpt/model.py:38-44`

For fresh models, tokenizer + resized embeddings must be initialized:

> `model.init_tokenizer()`  
> `model.initialize_special_embeddings()`  
> Source: `README.md:141-143`

And implementation:

> `self.tokenizer = SpokenDialogTokenizer(self.name_or_path)`  
> `self.transformer.resize_token_embeddings(new_num_tokens=len(self.tokenizer))`  
> Source: `turngpt/model.py:301-307`

## 3) Inference API

## 3.1 Primary high-level API

The high-level helper is:

> `def string_list_to_trp(self, string_or_list, add_post_eos_token=False, **model_kwargs):`  
> Source: `turngpt/model.py:131-133`

It tokenizes input, runs model forward, and returns derived probabilities:

> `out = self(t["input_ids"], speaker_ids=t["speaker_ids"], **model_kwargs)`  
> `out["probs"] = out["logits"].softmax(dim=-1)`  
> `out["trp_probs"] = self.get_trp(out["probs"])`  
> `out["tokens"] = self.get_tokens(t["input_ids"])`  
> `if "mc_logits" in out: out["trp_proj"] = out["mc_logits"].sigmoid()`  
> Source: `turngpt/model.py:137-143`

TRP extraction is specifically EOS (`<ts>`) probability:

> `def get_trp(self, x): return x[..., self.tokenizer.eos_token_id]`  
> Source: `turngpt/model.py:75-76`

## 3.2 Input format

Tokenizer accepts exactly these forms:

> "`text` is String"  
> "`text` is List[str]"  
> "`text` is List[List[str]]"  
> Source: `turngpt/tokenizer.py:205-211`

For List[str], tokenizer builds one dialog string and inserts `<ts>` delimiters, and by default appends final `<ts>`:

> `include_end_ts: bool = True`  
> `if include_end_ts: dialog_string += self.eos_token`  
> Source: `turngpt/tokenizer.py:199, 241-243`

`string_list_to_trp` uses `tokenize_strings`, which calls tokenizer with `return_tensors="pt"`:

> `t = self.tokenizer(string_or_list, return_tensors="pt")`  
> Source: `turngpt/model.py:83`

Special tokens used by tokenizer:

> `"eos_token": "<ts>"`  
> `"additional_special_tokens": ["<speaker1>", "<speaker2>"]`  
> Source: `turngpt/tokenizer.py:20-24`

Speaker IDs are computed automatically from `<ts>` positions:

> `batch, eos_idx = torch.where(input_ids == self.eos_token_id)`  
> `speaker_ids = torch.ones_like(input_ids) * self.sp1_token_id`  
> (then alternates speaker spans around EOS indices)  
> Source: `turngpt/tokenizer.py:265-279`

## 3.3 Output format

### From `string_list_to_trp`

Output includes model output plus extra keys:

- `logits`: raw LM logits (from `self.transformer.lm_head(hidden_states)`)  
  Source: `turngpt/model.py:507, 137`
- `past_key_values`: transformer cache from forward pass  
  Source: `turngpt/model.py:534`
- `probs`: `softmax(logits)` over vocab  
  Source: `turngpt/model.py:138`
- `trp_probs`: EOS token probability per position  
  Source: `turngpt/model.py:75-76, 139`
- `tokens`: decoded tokens of the provided `input_ids`  
  Source: `turngpt/model.py:140, 98-128`
- Optional `trp_proj`: sigmoid of projection head logits when `trp_projection_steps > 0`  
  Source: `turngpt/model.py:142, 515-519`

README summarizes expected keys as:

> `dict_keys(['logits', 'past_key_values', 'probs', 'trp_probs', 'tokens'])`  
> Source: `README.md:187`

### From low-level `forward`

`forward(...)` returns `GPT2DoubleHeadsModelOutput` with:

> `loss`, `mc_loss`, `logits`, `mc_logits`, `past_key_values`, `hidden_states`, `attentions`  
> Source: `turngpt/model.py:529-537`

`speaker_ids` is passed into GPT-2 as `token_type_ids`:

> `token_type_ids=speaker_ids`  
> Source: `turngpt/model.py:489`

### Meaning of TRP probability over time

Training loss shifts logits/labels so position `t` predicts token `t+1`:

> `shift_logits = logits[..., :-1, :]`  
> `shift_labels = labels[..., 1:]`  
> Source: `turngpt/model.py:398-399`

So `trp_probs` is the model's probability that the **next token** is `<ts>` at each position.

## 4) Real-time usage pattern (ASR transcript -> turn-shift probability)

Below is the source-grounded flow to run online.

1. Load checkpointed model and set eval mode.

> `model = TurnGPT.load_from_checkpoint(args.checkpoint)`  
> `model = model.eval()`  
> Source: `turngpt/generation.py:418-419`

2. Maintain running dialog text with `<ts>` separators for completed turns. Use tokenizer/model path that computes speaker IDs and TRP.

> `t = self.tokenizer(string_or_list, return_tensors="pt")`  
> `out = self(t["input_ids"], speaker_ids=t["speaker_ids"], **model_kwargs)`  
> `out["trp_probs"] = self.get_trp(out["probs"])`  
> Source: `turngpt/model.py:83, 137, 139`

3. At each ASR update, run inference and read the latest TRP probability from the time axis.

Mechanics used in code:

> `out["probs"] = out["logits"].softmax(dim=-1)`  
> `get_trp(x) -> x[..., eos_token_id]`  
> Source: `turngpt/model.py:138, 75-76`

4. For lower-latency token-by-token inference, reuse KV cache as shown in generation code.

> `out = model(**batch, use_cache=True)`  
> `batch["past_key_values"] = out["past_key_values"]`  
> `batch["input_ids"] = next_token.unsqueeze(-1)`  
> Source: `turngpt/generation.py:179, 197-199`

This is the implemented incremental pattern: first call with full context, then call with only newest token + `past_key_values`.

### Concrete online loop (library calls only)

```python
from turngpt import TurnGPT

model = TurnGPT.load_from_checkpoint("PATH/TO/checkpoint.ckpt").eval()

# Keep finalized turns separated by <ts>.
dialog_text = "hello how are you<ts> i am good thanks<ts> currently speaking partial"

out = model.string_list_to_trp(dialog_text, add_post_eos_token=False)
p_turn_shift_next = out["trp_probs"][0, -1].item()
```

Why this matches implementation:

> `string_list_to_trp(... add_post_eos_token=False ...)` tokenizes then runs forward and computes `trp_probs`  
> Source: `turngpt/model.py:131-139`

> `if isinstance(string_or_list, str) and add_post_eos_token: ... string_or_list += self.tokenizer.eos_token`  
> Source: `turngpt/model.py:79-82`

> `get_trp(x) -> x[..., eos_token_id]` (`<ts>` probability axis)  
> Source: `turngpt/model.py:75-76`

## 5) Caveats / gotchas (from source)

1. README examples use outdated import path `convlm`, but package/module is `turngpt`.

> `from convlm.tokenizer import SpokenDialogTokenizer`  
> `from convlm.turngpt import TurnGPT`  
> Source: `README.md:63, 113`

> `from .model import TurnGPT...` (actual package export)  
> Source: `turngpt/__init__.py:1`

2. If you pass `List[str]` dialogs, tokenizer appends trailing `<ts>` by default.

> `include_end_ts: bool = True`  
> `if include_end_ts: dialog_string += self.eos_token`  
> Source: `turngpt/tokenizer.py:199, 241-243`

3. Tokenizer warning: avoid a space before `<ts>` when composing raw dialog strings.

> "Do not have spaces prior to `eos_token`/<ts> ... this is bad!"  
> Source: `turngpt/tokenizer.py:81-89`

4. Text is normalized by default (lowercased, punctuation removed).

> `normalization=True`  
> `Replace(Regex(r'[\.\,\!\?\:\;\)\(\[\]"\-]'), "")`  
> `Lowercase()`  
> Source: `turngpt/tokenizer.py:145, 60-66`

5. Fresh model setup requires tokenizer initialization + embedding resize; checkpoint load handles this via `on_load_checkpoint` only if tokenizer was saved.

> `self.tokenizer = SpokenDialogTokenizer(self.name_or_path)`  
> `self.transformer.resize_token_embeddings(...)`  
> Source: `turngpt/model.py:301-307`

> `if "tokenizer" in checkpoint: ... self.tokenizer = checkpoint["tokenizer"] ... resize_token_embeddings(...)`  
> Source: `turngpt/model.py:552-561`

6. Supported base model families are restricted to GPT2/DialoGPT patterns.

> `if not ("gpt2" ... or "dialogpt" ...) : raise NotImplementedError(...)`  
> Source: `turngpt/model.py:38-44`
