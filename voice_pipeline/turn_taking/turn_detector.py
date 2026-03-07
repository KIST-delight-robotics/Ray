"""Combined turn-taking detector.

Fuses VAP (audio-based), TurnGPT (text-based), and timing heuristics
into a single per-frame ``TurnDecision``.
"""

from __future__ import annotations

import logging
import time
from difflib import SequenceMatcher
from typing import Literal

from voice_pipeline.core.config import AudioConfig, TurnDetectorConfig
from voice_pipeline.core.interfaces import IVAP, ITurnDetector, ITurnGPT
from voice_pipeline.core.types import AudioFrame, TurnDecision, VAPResult

logger = logging.getLogger("voice_pipeline.turn_taking.turn_detector")

_DEFAULT_VAP_RESULT = VAPResult(0.0, 0.0, False)


class TurnDetector(ITurnDetector):
    """ITurnDetector implementation fusing VAP + TurnGPT + timing heuristics.

    Single-threaded: only called from the Orchestrator's sync frame loop.
    No internal locking needed.
    """

    def __init__(
        self,
        vap: IVAP,
        turngpt: ITurnGPT,
        config: TurnDetectorConfig,
        audio_config: AudioConfig,
    ) -> None:
        self._vap = vap
        self._turngpt = turngpt
        self._config = config
        self._audio_config = audio_config

        # Per-frame state
        self._prev_asr_text: str = ""
        self._text_stable_since: float = 0.0
        self._prepare_fired: bool = False
        self._silence_frame_count: int = 0

        # TurnGPT context
        self._dialog_turns: list[str] = []
        self._current_partial: str = ""

    def process_frame(
        self,
        user_audio: AudioFrame,
        asr_text: str,
        robot_audio: AudioFrame | None = None,
    ) -> TurnDecision:
        """Process one pipeline frame and return a turn decision."""
        # 1. Feed audio to VAP (with error resilience)
        try:
            vap_result = self._vap.feed_audio(user_audio, robot_audio)
        except Exception:
            logger.warning("VAP error, using default result", exc_info=True)
            vap_result = _DEFAULT_VAP_RESULT

        # 2. Interrupt check: robot speaking + user speaking
        if robot_audio is not None and vap_result.user_is_speaking:
            return TurnDecision(interrupt=True)

        # 3. Text change detection
        normalized = asr_text.lower().strip()
        if normalized and self._prev_asr_text:
            ratio = SequenceMatcher(None, self._prev_asr_text, normalized).ratio()
            text_changed = ratio < self._config.text_similarity_threshold
        elif normalized != self._prev_asr_text:
            # One is empty, the other is not
            text_changed = True
        else:
            text_changed = False

        if text_changed:
            self._text_stable_since = time.monotonic()
            self._prepare_fired = False
            self._silence_frame_count = 0
            self._prev_asr_text = normalized
            self._current_partial = asr_text.strip()

        # 4. Silence tracking
        if vap_result.user_is_speaking:
            self._silence_frame_count = 0
        else:
            self._silence_frame_count += 1

        # 5. Prepare check
        if self._prev_asr_text and not self._prepare_fired and self._text_stable_since > 0.0:
            elapsed_ms = (time.monotonic() - self._text_stable_since) * 1000
            if elapsed_ms >= self._config.prepare_stable_ms:
                try:
                    dialog = self._build_turngpt_dialog()
                    prob = self._turngpt.predict(dialog)
                except Exception:
                    logger.warning("TurnGPT error during prepare check", exc_info=True)
                    prob = 0.0

                if prob > self._config.turngpt_threshold:
                    self._prepare_fired = True
                    return TurnDecision(prepare=True)

        # 6. Turn-shift check (requires prior speech)
        if self._prev_asr_text:
            if self._silence_frame_count >= self._config.turn_shift_silence_frames:
                return TurnDecision(turn_shift=True)

            if self._text_stable_since > 0.0:
                elapsed_ms = (time.monotonic() - self._text_stable_since) * 1000
                if elapsed_ms >= self._config.hard_silence_timeout_ms:
                    return TurnDecision(turn_shift=True)

        # 7. No action
        return TurnDecision.none()

    def notify_turn_complete(self, role: Literal["user", "robot"], text: str) -> None:
        """Inform the detector that a turn was completed."""
        if not text or not text.strip():
            return
        self._dialog_turns.append(text.strip())
        self._current_partial = ""

    def reset(self) -> None:
        """Reset per-frame tracking state for a new turn.

        Preserves dialog context (``_dialog_turns``) for TurnGPT.
        """
        self._prev_asr_text = ""
        self._text_stable_since = 0.0
        self._prepare_fired = False
        self._silence_frame_count = 0
        self._current_partial = ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_turngpt_dialog(self) -> str:
        """Build ``<ts>``-delimited dialog string for TurnGPT."""
        parts = list(self._dialog_turns)
        if self._current_partial:
            parts.append(self._current_partial)
            return "<ts>".join(parts)
        # No current partial — all completed turns, join with <ts>
        if parts:
            return "<ts>".join(parts)
        return ""
