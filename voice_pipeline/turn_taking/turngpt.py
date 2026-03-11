"""TurnGPT model wrapper for text-based turn-shift prediction.

Wraps the external TurnGPT model behind the ITurnGPT interface.
Predicts the probability that the current speaker's turn is ending,
based on ``<ts>``-delimited dialog text.

Supports two backends:
- **PyTorch**: loads the full TurnGPT checkpoint (requires ``turngpt`` package).
- **ONNX**: loads an exported ONNX model via ONNX Runtime (requires
  ``onnxruntime`` and a separately saved tokenizer).

Both backends use KV cache to avoid reprocessing stable prefix tokens.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import Tensor

from voice_pipeline.core.config import TurnGPTConfig
from voice_pipeline.core.interfaces import ITurnGPT
from voice_pipeline.turn_taking.exceptions import TurnGPTError

logger = logging.getLogger("voice_pipeline.turn_taking.turngpt")

_DEFAULT_PROBABILITY = 0.0
_EVICTION_HEADROOM = 0.8

# GPT-2 architecture constants (TurnGPT is always GPT-2 based)
_NUM_LAYERS = 12
_NUM_HEADS = 12
_HEAD_DIM = 64


class TurnGPTWrapper(ITurnGPT):
    """ITurnGPT implementation wrapping the external TurnGPT model.

    Maintains a KV cache across ``predict()`` calls so that only new tokens
    are forwarded through the model when the dialog prefix is unchanged.

    Backend is selected by config: if ``onnx_model_path`` is set, uses ONNX
    Runtime; otherwise loads the PyTorch checkpoint.
    """

    def __init__(self, config: TurnGPTConfig) -> None:
        self._max_context_tokens = config.max_context_tokens
        self._backend: str  # "pytorch" or "onnx"

        if config.onnx_model_path:
            self._backend = "onnx"
            self._init_onnx(config)
        else:
            self._backend = "pytorch"
            self._init_pytorch(config)

        self._cached_input_ids: Tensor | None = None
        self._past_key_values: Any = None  # tuple (pytorch) or dict (onnx)
        self._cached_trp_prob: float = _DEFAULT_PROBABILITY

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_pytorch(self, config: TurnGPTConfig) -> None:
        try:
            from turngpt import TurnGPT

            model = TurnGPT.load_from_checkpoint(config.checkpoint_path)
            self._model = model.to(config.device).eval()
            self._tokenizer = model.tokenizer
        except Exception as exc:
            raise TurnGPTError(f"Failed to load TurnGPT model: {exc}") from exc

    def _init_onnx(self, config: TurnGPTConfig) -> None:
        if not config.tokenizer_path:
            raise TurnGPTError(
                "tokenizer_path is required when onnx_model_path is set"
            )
        try:
            import onnxruntime as ort
            from transformers import GPT2TokenizerFast

            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.intra_op_num_threads = config.onnx_threads
            self._ort_session = ort.InferenceSession(
                config.onnx_model_path, so, providers=["CPUExecutionProvider"],
            )
            self._hf_tokenizer = GPT2TokenizerFast.from_pretrained(
                config.tokenizer_path,
            )
            self._eos_token_id = self._hf_tokenizer.eos_token_id
            self._sp1_id = self._hf_tokenizer.convert_tokens_to_ids("<speaker1>")
            self._sp2_id = self._hf_tokenizer.convert_tokens_to_ids("<speaker2>")

            # Detect model type: KV-cache models have past_key_* inputs
            input_names = {inp.name for inp in self._ort_session.get_inputs()}
            self._onnx_has_kv = "past_key_0" in input_names
        except TurnGPTError:
            raise
        except Exception as exc:
            raise TurnGPTError(f"Failed to load ONNX model: {exc}") from exc

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, dialog_text: str) -> float:
        """Predict turn-shift probability for the given dialog.

        Args:
            dialog_text: Full conversation text with ``<ts>`` separators
                between completed turns. No trailing ``<ts>`` for the
                current in-progress turn.

        Returns:
            Turn-shift probability in [0, 1].
        """
        if not dialog_text or not dialog_text.strip():
            return _DEFAULT_PROBABILITY

        try:
            input_ids, speaker_ids = self._tokenize_with_window(dialog_text)
            if self._backend == "onnx":
                return self._onnx_forward_with_cache(input_ids, speaker_ids)
            return self._pytorch_forward_with_cache(input_ids, speaker_ids)
        except Exception:
            logger.warning("TurnGPT inference error, returning default", exc_info=True)
            self._clear_cache()
            return _DEFAULT_PROBABILITY

    def reset(self) -> None:
        """Reset internal state for a new conversation."""
        self._clear_cache()

    # ------------------------------------------------------------------
    # Tokenization (shared)
    # ------------------------------------------------------------------

    def _tokenize_with_window(self, dialog_text: str) -> tuple[Tensor, Tensor]:
        """Tokenize dialog, evicting old turns if over max_context_tokens."""
        input_ids, speaker_ids = self._tokenize(dialog_text)

        max_tokens = self._max_context_tokens
        if max_tokens <= 0 or input_ids.shape[-1] <= max_tokens:
            return input_ids, speaker_ids

        target = int(max_tokens * _EVICTION_HEADROOM)
        parts = dialog_text.split("<ts>")

        while len(parts) > 1:
            parts.pop(0)
            trimmed = "<ts>".join(parts)
            input_ids, speaker_ids = self._tokenize(trimmed)
            if input_ids.shape[-1] <= target:
                break

        if input_ids.shape[-1] > max_tokens:
            input_ids = input_ids[:, -max_tokens:]
            speaker_ids = speaker_ids[:, -max_tokens:]

        self._clear_cache()
        return input_ids, speaker_ids

    def _tokenize(self, text: str) -> tuple[Tensor, Tensor]:
        """Tokenize text and return (input_ids, speaker_ids) as tensors."""
        if self._backend == "pytorch":
            encoded = self._tokenizer(text, return_tensors="pt")
            return encoded["input_ids"], encoded["speaker_ids"]

        encoded = self._hf_tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"]
        speaker_ids = _build_speaker_ids(
            input_ids, self._eos_token_id, self._sp1_id, self._sp2_id,
        )
        return input_ids, speaker_ids

    # ------------------------------------------------------------------
    # PyTorch backend
    # ------------------------------------------------------------------

    def _pytorch_forward_with_cache(
        self, input_ids: Tensor, speaker_ids: Tensor,
    ) -> float:
        cached = self._cached_input_ids
        past = self._past_key_values

        if cached is not None and past is not None:
            prefix_len = _common_prefix_length(cached, input_ids)
            if prefix_len > 0:
                if prefix_len == input_ids.shape[-1] == cached.shape[-1]:
                    return self._cached_trp_prob

                sliced_past = _slice_past_pytorch(past, prefix_len)
                new_ids = input_ids[:, prefix_len:]
                new_speaker = speaker_ids[:, prefix_len:]

                out = self._model(
                    new_ids,
                    speaker_ids=new_speaker,
                    past_key_values=sliced_past,
                    use_cache=True,
                )
                return self._pytorch_update_cache(input_ids, out)

        out = self._model(input_ids, speaker_ids=speaker_ids, use_cache=True)
        return self._pytorch_update_cache(input_ids, out)

    def _pytorch_update_cache(self, input_ids: Tensor, out: dict) -> float:
        self._cached_input_ids = input_ids
        self._past_key_values = out.get("past_key_values")
        probs = out["logits"].softmax(dim=-1)
        trp = self._model.get_trp(probs)[0, -1].item()
        self._cached_trp_prob = trp
        return trp

    # ------------------------------------------------------------------
    # ONNX backend
    # ------------------------------------------------------------------

    def _onnx_forward_with_cache(
        self, input_ids: Tensor, speaker_ids: Tensor,
    ) -> float:
        cached = self._cached_input_ids
        past = self._past_key_values

        if self._onnx_has_kv and cached is not None and past is not None:
            prefix_len = _common_prefix_length(cached, input_ids)
            if prefix_len > 0:
                if prefix_len == input_ids.shape[-1] == cached.shape[-1]:
                    return self._cached_trp_prob

                sliced_past = _slice_past_onnx(past, prefix_len)
                new_ids = input_ids[:, prefix_len:]
                new_speaker = speaker_ids[:, prefix_len:]

                trp, presents = self._onnx_run(
                    new_ids, new_speaker, prefix_len, sliced_past,
                )
                self._cached_input_ids = input_ids
                self._past_key_values = presents
                self._cached_trp_prob = trp
                return trp

        if self._onnx_has_kv:
            trp, presents = self._onnx_run(
                input_ids, speaker_ids, 0, _empty_past(),
            )
            self._cached_input_ids = input_ids
            self._past_key_values = presents
            self._cached_trp_prob = trp
            return trp

        # No-cache ONNX model
        trp = self._onnx_run_no_cache(input_ids, speaker_ids)
        self._cached_input_ids = input_ids
        self._cached_trp_prob = trp
        return trp

    def _onnx_run(
        self,
        input_ids: Tensor,
        speaker_ids: Tensor,
        position_offset: int,
        past: dict[str, np.ndarray],
    ) -> tuple[float, dict[str, np.ndarray]]:
        """Run ONNX KV-cache model. Returns (trp, presents)."""
        ids_np = input_ids.numpy()
        sp_np = speaker_ids.numpy()
        seq_len = ids_np.shape[1]
        pos_np = np.arange(
            position_offset, position_offset + seq_len, dtype=np.int64,
        ).reshape(1, -1)

        feeds = {
            "input_ids": ids_np,
            "speaker_ids": sp_np,
            "position_ids": pos_np,
            **past,
        }
        outputs = self._ort_session.run(None, feeds)

        presents = {}
        for i in range(_NUM_LAYERS):
            presents[f"past_key_{i}"] = outputs[1 + i * 2]
            presents[f"past_value_{i}"] = outputs[2 + i * 2]

        trp = _extract_trp_numpy(outputs[0], self._eos_token_id)
        return trp, presents

    def _onnx_run_no_cache(
        self, input_ids: Tensor, speaker_ids: Tensor,
    ) -> float:
        """Run ONNX no-cache model. Returns trp."""
        feeds = {
            "input_ids": input_ids.numpy(),
            "speaker_ids": speaker_ids.numpy(),
        }
        outputs = self._ort_session.run(None, feeds)
        return _extract_trp_numpy(outputs[0], self._eos_token_id)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _clear_cache(self) -> None:
        self._cached_input_ids = None
        self._past_key_values = None
        self._cached_trp_prob = _DEFAULT_PROBABILITY


# ======================================================================
# Module-level helpers
# ======================================================================


def _common_prefix_length(a: Tensor, b: Tensor) -> int:
    """Return the length of the common token prefix between two tensors."""
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    min_len = min(a_flat.shape[0], b_flat.shape[0])
    if min_len == 0:
        return 0
    matches = a_flat[:min_len] == b_flat[:min_len]
    if matches.all():
        return min_len
    return int((~matches).nonzero(as_tuple=False)[0].item())


def _build_speaker_ids(
    input_ids: Tensor, eos_id: int, sp1_id: int, sp2_id: int,
) -> Tensor:
    """Build speaker_ids tensor from input_ids, alternating at <ts> tokens."""
    speaker_ids = torch.full_like(input_ids, sp1_id)
    batch, eos_idx = torch.where(input_ids == eos_id)
    for b in batch.unique():
        tmp_eos = eos_idx[batch == b]
        if len(tmp_eos) == 1:
            speaker_ids[b, tmp_eos[0] + 1 :] = sp2_id
        elif len(tmp_eos) > 1:
            start = tmp_eos[0]
            for i, eos in enumerate(tmp_eos[1:]):
                if i % 2 == 0:
                    speaker_ids[b, start + 1 : eos + 1] = sp2_id
                start = eos
            if i % 2 == 1:
                speaker_ids[b, start + 1 :] = sp2_id
    return speaker_ids


def _extract_trp_numpy(logits: np.ndarray, eos_token_id: int) -> float:
    """Extract turn-shift probability from logits via softmax."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    return float(probs[0, -1, eos_token_id])


def _empty_past() -> dict[str, np.ndarray]:
    """Create empty past KV tensors for ONNX KV-cache model."""
    d: dict[str, np.ndarray] = {}
    for i in range(_NUM_LAYERS):
        d[f"past_key_{i}"] = np.zeros(
            (1, _NUM_HEADS, 0, _HEAD_DIM), dtype=np.float32,
        )
        d[f"past_value_{i}"] = np.zeros(
            (1, _NUM_HEADS, 0, _HEAD_DIM), dtype=np.float32,
        )
    return d


def _slice_past_pytorch(past_key_values: tuple, prefix_len: int) -> tuple:
    """Slice PyTorch KV cache to keep only the first ``prefix_len`` positions."""
    return tuple(
        (k[:, :, :prefix_len, :], v[:, :, :prefix_len, :])
        for k, v in past_key_values
    )


def _slice_past_onnx(
    past: dict[str, np.ndarray], prefix_len: int,
) -> dict[str, np.ndarray]:
    """Slice ONNX KV cache to keep only the first ``prefix_len`` positions."""
    return {k: v[:, :, :prefix_len, :] for k, v in past.items()}
