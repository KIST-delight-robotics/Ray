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
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from voice_pipeline.core.interfaces import ITurnGPT
from voice_pipeline.turn_taking.exceptions import TurnGPTError

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger("voice_pipeline.turn_taking.turngpt")


class TurnGPTWrapper(ITurnGPT):
    """ITurnGPT implementation wrapping the external TurnGPT model.

    Maintains a KV cache across ``predict()`` calls so that only new tokens
    are forwarded through the model when the dialog prefix is unchanged.
    """

    # 백엔드 / 모델 경로 (배포 튜닝값)
    _BACKEND: Literal["onnx", "pytorch"] = "onnx"  # 추론 경로: ONNX Runtime vs PyTorch Lightning
    _ONNX_MODEL_PATH: str = (
        "models/turngpt/turngpt_v2_kvcache_int8.onnx"  # ONNX 모델 파일 경로 (기본값은 RPi용 int8+KV캐시)
    )
    _TOKENIZER_PATH: str = "models/turngpt/tokenizer"  # ONNX 모드 토크나이저 디렉토리
    _ONNX_THREADS: int = 2  # ONNX Runtime intra-op 스레드 수 (RPi 5 4-코어 기준 2가 최적)
    _CHECKPOINT_PATH: str = "models/turngpt/turngpt.ckpt"  # PyTorch 체크포인트 경로 (PyTorch 모드)

    # GPT-2 아키텍처 (TurnGPT는 GPT-2 기반 고정)
    _NUM_LAYERS = 12
    _NUM_HEADS = 12
    _HEAD_DIM = 64

    _FALLBACK_PROBABILITY = 0.0  # 추론 실패/빈 입력 시 반환할 turn-shift 확률
    _DEVICE = "cpu"  # PyTorch 디바이스 ("cpu" / "cuda")
    _MAX_CONTEXT_TOKENS = 1024  # 모델 입력 최대 토큰 수 (GPT-2 상한). 초과 시 오래된 턴 제거. 0이면 무제한
    _KEEP_TURNS = 2  # 토큰 초과 시 유지할 최근 완료 턴 수 (진행 중 턴은 항상 유지)
    _ONNX_PROVIDERS = ("CPUExecutionProvider",)  # ONNX Runtime 실행 프로바이더

    def __init__(self) -> None:
        if self._BACKEND == "onnx":
            self._init_onnx()
        else:
            self._init_pytorch()

        self._cached_input_ids: Tensor | np.ndarray | None = None
        self._past_key_values: Any = None  # tuple (pytorch) or dict (onnx)
        self._cached_trp_prob: float = self._FALLBACK_PROBABILITY

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_pytorch(self) -> None:
        try:
            import torch
            from turngpt import TurnGPT

            self._torch = torch
            model = TurnGPT.load_from_checkpoint(self._CHECKPOINT_PATH)
            self._model = model.to(self._DEVICE).eval()
            self._tokenizer = model.tokenizer
        except Exception as exc:
            raise TurnGPTError(f"Failed to load TurnGPT model: {exc}") from exc

    def _init_onnx(self) -> None:
        if not self._TOKENIZER_PATH:
            raise TurnGPTError("TurnGPT tokenizer path must not be empty")
        try:
            import onnxruntime as ort
            from transformers import GPT2TokenizerFast

            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.intra_op_num_threads = self._ONNX_THREADS
            self._ort_session = ort.InferenceSession(
                self._ONNX_MODEL_PATH,
                so,
                providers=list(self._ONNX_PROVIDERS),
            )
            self._hf_tokenizer = GPT2TokenizerFast.from_pretrained(
                self._TOKENIZER_PATH,
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
            return self._FALLBACK_PROBABILITY

        try:
            input_ids, speaker_ids = self._tokenize_with_window(dialog_text)
            if self._BACKEND == "onnx":
                return self._onnx_forward_with_cache(input_ids, speaker_ids)
            return self._pytorch_forward_with_cache(input_ids, speaker_ids)
        except Exception:
            logger.warning("TurnGPT inference error, returning default", exc_info=True)
            self._clear_cache()
            return self._FALLBACK_PROBABILITY

    def reset(self) -> None:
        """Reset internal state for a new conversation."""
        self._clear_cache()

    # ------------------------------------------------------------------
    # Tokenization (shared)
    # ------------------------------------------------------------------

    def _tokenize_with_window(self, dialog_text: str) -> tuple[Tensor | np.ndarray, Tensor | np.ndarray]:
        """Tokenize dialog, evicting old turns if over max_context_tokens.

        Keeps the last ``_KEEP_TURNS`` completed turns plus the current
        incomplete turn.  ``_MAX_CONTEXT_TOKENS`` acts as a hard truncation
        safety net.

        The KV cache is NOT cleared here — the forward methods use prefix
        matching against ``_cached_input_ids`` to decide whether to reuse
        the cache or recompute.
        """
        input_ids, speaker_ids = self._tokenize(dialog_text)

        max_tokens = self._MAX_CONTEXT_TOKENS
        if max_tokens <= 0 or input_ids.shape[-1] <= max_tokens:
            return input_ids, speaker_ids

        parts = dialog_text.split("<ts>")
        # parts[-1] is the current incomplete turn, parts[:-1] are completed.
        n_keep = self._KEEP_TURNS + 1  # +1 for current incomplete turn
        if len(parts) > n_keep:
            trimmed = "<ts>".join(parts[-n_keep:])
            input_ids, speaker_ids = self._tokenize(trimmed)

        if input_ids.shape[-1] > max_tokens:
            input_ids = input_ids[:, -max_tokens:]
            speaker_ids = speaker_ids[:, -max_tokens:]

        return input_ids, speaker_ids

    def _tokenize(self, text: str) -> tuple[Tensor | np.ndarray, Tensor | np.ndarray]:
        """Tokenize text and return (input_ids, speaker_ids).

        Returns torch Tensors for PyTorch backend, numpy arrays for ONNX.
        """
        if self._BACKEND == "pytorch":
            encoded = self._tokenizer(text, return_tensors="pt")
            return encoded["input_ids"], encoded["speaker_ids"]

        encoded = self._hf_tokenizer(text, return_tensors="np")
        input_ids = encoded["input_ids"]
        speaker_ids = _build_speaker_ids(
            input_ids,
            self._eos_token_id,
            self._sp1_id,
            self._sp2_id,
        )
        return input_ids, speaker_ids

    # ------------------------------------------------------------------
    # PyTorch backend
    # ------------------------------------------------------------------

    def _pytorch_forward_with_cache(
        self,
        input_ids: Tensor,
        speaker_ids: Tensor,
    ) -> float:
        with self._torch.no_grad():
            return self._pytorch_forward_with_cache_inner(input_ids, speaker_ids)

    def _pytorch_forward_with_cache_inner(
        self,
        input_ids: Tensor,
        speaker_ids: Tensor,
    ) -> float:
        cached = self._cached_input_ids
        past = self._past_key_values

        if cached is not None and past is not None:
            prefix_len = _common_prefix_length(cached, input_ids)
            if prefix_len > 0:
                if prefix_len == input_ids.shape[-1] == cached.shape[-1]:
                    return self._cached_trp_prob

                if prefix_len < input_ids.shape[-1]:
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
                # Input is a strict prefix of cached — fall through to
                # full recompute.

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
        self,
        input_ids: np.ndarray,
        speaker_ids: np.ndarray,
    ) -> float:
        cached = self._cached_input_ids
        past = self._past_key_values

        if self._onnx_has_kv and cached is not None and past is not None:
            prefix_len = _common_prefix_length(cached, input_ids)
            if prefix_len > 0:
                if prefix_len == input_ids.shape[-1] == cached.shape[-1]:
                    return self._cached_trp_prob

                if prefix_len < input_ids.shape[-1]:
                    sliced_past = _slice_past_onnx(past, prefix_len)
                    new_ids = input_ids[:, prefix_len:]
                    new_speaker = speaker_ids[:, prefix_len:]

                    trp, presents = self._onnx_run(
                        new_ids,
                        new_speaker,
                        prefix_len,
                        sliced_past,
                    )
                    self._cached_input_ids = input_ids
                    self._past_key_values = presents
                    self._cached_trp_prob = trp
                    return trp
                # prefix_len == input_ids length but cached is longer:
                # input is a strict prefix of cached — fall through to
                # full recompute since we cannot reuse the longer cache.

        if self._onnx_has_kv:
            trp, presents = self._onnx_run(
                input_ids,
                speaker_ids,
                0,
                self._empty_past(),
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
        input_ids: np.ndarray,
        speaker_ids: np.ndarray,
        position_offset: int,
        past: dict[str, np.ndarray],
    ) -> tuple[float, dict[str, np.ndarray]]:
        """Run ONNX KV-cache model. Returns (trp, presents)."""
        seq_len = input_ids.shape[1]
        pos_np = np.arange(
            position_offset,
            position_offset + seq_len,
            dtype=np.int64,
        ).reshape(1, -1)

        feeds = {
            "input_ids": input_ids,
            "speaker_ids": speaker_ids,
            "position_ids": pos_np,
            **past,
        }
        outputs = self._ort_session.run(None, feeds)

        presents = {}
        for i in range(self._NUM_LAYERS):
            presents[f"past_key_{i}"] = outputs[1 + i * 2]
            presents[f"past_value_{i}"] = outputs[2 + i * 2]

        trp = _extract_trp_numpy(outputs[0], self._eos_token_id)
        return trp, presents

    def _onnx_run_no_cache(
        self,
        input_ids: np.ndarray,
        speaker_ids: np.ndarray,
    ) -> float:
        """Run ONNX no-cache model. Returns trp."""
        feeds = {
            "input_ids": input_ids,
            "speaker_ids": speaker_ids,
        }
        outputs = self._ort_session.run(None, feeds)
        return _extract_trp_numpy(outputs[0], self._eos_token_id)

    def _empty_past(self) -> dict[str, np.ndarray]:
        """Create empty past KV tensors for ONNX KV-cache model."""
        d: dict[str, np.ndarray] = {}
        shape = (1, self._NUM_HEADS, 0, self._HEAD_DIM)
        for i in range(self._NUM_LAYERS):
            d[f"past_key_{i}"] = np.zeros(shape, dtype=np.float32)
            d[f"past_value_{i}"] = np.zeros(shape, dtype=np.float32)
        return d

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _clear_cache(self) -> None:
        self._cached_input_ids = None
        self._past_key_values = None
        self._cached_trp_prob = self._FALLBACK_PROBABILITY


# ======================================================================
# Module-level helpers
# ======================================================================


def _common_prefix_length(a: Tensor | np.ndarray, b: Tensor | np.ndarray) -> int:
    """Return the length of the common token prefix between two arrays.

    Works with both numpy arrays and torch tensors via duck typing:
    both support .reshape(), .shape, ==, .all(), and .argmin().
    """
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    min_len = min(a_flat.shape[0], b_flat.shape[0])
    if min_len == 0:
        return 0
    matches = a_flat[:min_len] == b_flat[:min_len]
    if matches.all():
        return min_len
    # First mismatch index. Multiply by 1 to convert bool→int
    # (torch.argmax does not accept bool, numpy does; * 1 works for both).
    return int((~matches * 1).argmax())


def _build_speaker_ids(
    input_ids: np.ndarray,
    eos_id: int,
    sp1_id: int,
    sp2_id: int,
) -> np.ndarray:
    """Build speaker_ids array from input_ids, alternating at <ts> tokens."""
    speaker_ids = np.full_like(input_ids, sp1_id)
    batch, eos_idx = np.where(input_ids == eos_id)
    for b in np.unique(batch):
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


def _slice_past_pytorch(past_key_values: tuple, prefix_len: int) -> tuple:
    """Slice PyTorch KV cache to keep only the first ``prefix_len`` positions."""
    return tuple((k[:, :, :prefix_len, :], v[:, :, :prefix_len, :]) for k, v in past_key_values)


def _slice_past_onnx(
    past: dict[str, np.ndarray],
    prefix_len: int,
) -> dict[str, np.ndarray]:
    """Slice ONNX KV cache to keep only the first ``prefix_len`` positions."""
    return {k: v[:, :, :prefix_len, :] for k, v in past.items()}
