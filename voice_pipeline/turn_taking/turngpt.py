"""TurnGPT model wrapper for text-based turn-shift prediction.

Wraps the external TurnGPT model behind the ITurnGPT interface.
Predicts the probability that the current speaker's turn is ending,
based on ``<ts>``-delimited dialog text.
"""

from __future__ import annotations

import logging

import torch

from voice_pipeline.core.config import TurnGPTConfig
from voice_pipeline.core.interfaces import ITurnGPT
from voice_pipeline.turn_taking.exceptions import TurnGPTError

logger = logging.getLogger("voice_pipeline.turn_taking.turngpt")

_DEFAULT_PROBABILITY = 0.0


class TurnGPTWrapper(ITurnGPT):
    """ITurnGPT implementation wrapping the external TurnGPT model.

    Stateless between calls — no internal buffer or cache. Each ``predict``
    call is independent.
    """

    def __init__(self, config: TurnGPTConfig) -> None:
        try:
            from turngpt import TurnGPT

            model = TurnGPT.load_from_checkpoint(config.checkpoint_path)
            self._model = model.to(config.device).eval()
        except Exception as exc:
            raise TurnGPTError(f"Failed to load TurnGPT model: {exc}") from exc

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
            out = self._model.string_list_to_trp(dialog_text, add_post_eos_token=False)
            return out["trp_probs"][0, -1].item()
        except Exception:
            logger.warning("TurnGPT inference error, returning default", exc_info=True)
            return _DEFAULT_PROBABILITY

    def reset(self) -> None:
        """Reset internal state for a new conversation.

        No-op for this stateless wrapper; satisfies the interface contract.
        """
