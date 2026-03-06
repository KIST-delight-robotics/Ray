"""TurnGPT model wrapper for text-based turn-shift prediction.

Wraps the external TurnGPT model behind the ITurnGPT interface.
Predicts the probability that the current speaker's turn is ending,
based on ``<ts>``-delimited dialog text.

Uses KV cache to avoid reprocessing stable prefix tokens on each call,
and a configurable context window to stay within GPT-2 position limits.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from voice_pipeline.core.config import TurnGPTConfig
from voice_pipeline.core.interfaces import ITurnGPT
from voice_pipeline.turn_taking.exceptions import TurnGPTError

logger = logging.getLogger("voice_pipeline.turn_taking.turngpt")

_DEFAULT_PROBABILITY = 0.0
_EVICTION_HEADROOM = 0.8


class TurnGPTWrapper(ITurnGPT):
    """ITurnGPT implementation wrapping the external TurnGPT model.

    Maintains a KV cache across ``predict()`` calls so that only new tokens
    are forwarded through the model when the dialog prefix is unchanged.
    """

    def __init__(self, config: TurnGPTConfig) -> None:
        self._max_context_tokens = config.max_context_tokens
        try:
            from turngpt import TurnGPT

            model = TurnGPT.load_from_checkpoint(config.checkpoint_path)
            self._model = model.to(config.device).eval()
        except Exception as exc:
            raise TurnGPTError(f"Failed to load TurnGPT model: {exc}") from exc

        self._cached_input_ids: Tensor | None = None
        self._past_key_values: tuple | None = None
        self._cached_trp_prob: float = _DEFAULT_PROBABILITY

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
            return self._forward_with_cache(input_ids, speaker_ids)
        except Exception:
            logger.warning("TurnGPT inference error, returning default", exc_info=True)
            self._clear_cache()
            return _DEFAULT_PROBABILITY

    def reset(self) -> None:
        """Reset internal state for a new conversation."""
        self._clear_cache()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize_with_window(self, dialog_text: str) -> tuple[Tensor, Tensor]:
        """Tokenize dialog, evicting old turns if over max_context_tokens."""
        encoded = self._model.tokenizer(dialog_text, return_tensors="pt")
        input_ids = encoded["input_ids"]
        speaker_ids = encoded["speaker_ids"]

        max_tokens = self._max_context_tokens
        if max_tokens <= 0 or input_ids.shape[-1] <= max_tokens:
            return input_ids, speaker_ids

        target = int(max_tokens * _EVICTION_HEADROOM)
        parts = dialog_text.split("<ts>")

        while len(parts) > 1:
            parts.pop(0)
            trimmed = "<ts>".join(parts)
            encoded = self._model.tokenizer(trimmed, return_tensors="pt")
            input_ids = encoded["input_ids"]
            speaker_ids = encoded["speaker_ids"]
            if input_ids.shape[-1] <= target:
                break

        # Fallback: single turn still too long — left-truncate at token level
        if input_ids.shape[-1] > max_tokens:
            input_ids = input_ids[:, -max_tokens:]
            speaker_ids = speaker_ids[:, -max_tokens:]

        self._clear_cache()
        return input_ids, speaker_ids

    def _forward_with_cache(self, input_ids: Tensor, speaker_ids: Tensor) -> float:
        """Run model forward, reusing KV cache when possible."""
        cached = self._cached_input_ids
        past = self._past_key_values

        if cached is not None and past is not None:
            prefix_len = self._common_prefix_length(cached, input_ids)
            if prefix_len > 0:
                if prefix_len == input_ids.shape[-1] == cached.shape[-1]:
                    return self._cached_trp_prob

                sliced_past = _slice_past(past, prefix_len)
                new_ids = input_ids[:, prefix_len:]
                new_speaker = speaker_ids[:, prefix_len:]

                out = self._model(
                    new_ids,
                    speaker_ids=new_speaker,
                    past_key_values=sliced_past,
                    use_cache=True,
                )
                return self._update_cache(input_ids, out)

        out = self._model(input_ids, speaker_ids=speaker_ids, use_cache=True)
        return self._update_cache(input_ids, out)

    def _update_cache(self, input_ids: Tensor, out: dict) -> float:
        """Store cache and extract TRP probability from model output."""
        self._cached_input_ids = input_ids
        self._past_key_values = out.get("past_key_values")

        probs = out["logits"].softmax(dim=-1)
        trp = self._model.get_trp(probs)[0, -1].item()
        self._cached_trp_prob = trp
        return trp

    def _clear_cache(self) -> None:
        self._cached_input_ids = None
        self._past_key_values = None
        self._cached_trp_prob = _DEFAULT_PROBABILITY

    @staticmethod
    def _common_prefix_length(a: Tensor, b: Tensor) -> int:
        """Return the length of the common token prefix between two 1D/2D tensors."""
        a_flat = a.reshape(-1)
        b_flat = b.reshape(-1)
        min_len = min(a_flat.shape[0], b_flat.shape[0])
        if min_len == 0:
            return 0
        matches = a_flat[:min_len] == b_flat[:min_len]
        if matches.all():
            return min_len
        return int((~matches).nonzero(as_tuple=False)[0].item())


def _slice_past(past_key_values: tuple, prefix_len: int) -> tuple:
    """Slice KV cache to keep only the first ``prefix_len`` positions."""
    return tuple((k[:, :, :prefix_len, :], v[:, :, :prefix_len, :]) for k, v in past_key_values)
