"""Combined turn-taking detector.

Fuses VAP (audio-based), TurnGPT (text-based), and timing heuristics
into a single per-frame TurnDecision.

Design reference: Skantze & Irfan (2025) — "Applying General Turn-taking
Models to Conversational Human-Robot Interaction".
"""

from __future__ import annotations

import enum
import logging
from difflib import SequenceMatcher
from typing import Literal

from voice_pipeline.core.config import AudioConfig, TurnDetectorConfig
from voice_pipeline.core.interfaces import IVAP, ITurnDetector
from voice_pipeline.core.types import AudioFrame, TurnDecision, VAPResult
from voice_pipeline.turn_taking.async_turngpt import AsyncTurnGPT, SyncTurnGPTAdapter

logger = logging.getLogger("voice_pipeline.turn_taking")


class _TurnState(enum.Enum):
    USER_TURN = "user_turn"
    ROBOT_TURN = "robot_turn"


class TurnDetector(ITurnDetector):
    """Combined turn-taking detector using VAP + TurnGPT + timing heuristics.

    Two independent OR paths for turn-shift detection:
    - Path 1 (VAP): Sustained robot-favoring probabilities.
    - Path 2 (TurnGPT): Graduated silence timeout based on completion probability.

    Interrupt detection during ROBOT_TURN uses VAP to distinguish
    genuine interrupts from backchannels.
    """

    def __init__(
        self,
        vap: IVAP,
        turngpt: AsyncTurnGPT | SyncTurnGPTAdapter,
        config: TurnDetectorConfig,
        audio_config: AudioConfig,
    ) -> None:
        self._vap = vap
        self._turngpt = turngpt
        self._config = config

        self._frame_duration_sec = audio_config.frame_duration_ms / 1000.0

        # Internal state
        self._turn_state = _TurnState.USER_TURN
        self._dialog_parts: list[str] = []

        # Per-frame tracking (reset between turns)
        self._prev_asr_text: str = ""
        self._vap_favor_robot_elapsed_sec: float = 0.0
        self._silence_elapsed_sec: float = 0.0
        self._last_asr_change_elapsed_sec: float = 0.0
        self._asr_has_changed: bool = False
        self._last_prepare_text: str = ""
        self._turngpt_prob: float = 0.0

    # ------------------------------------------------------------------
    # ITurnDetector interface
    # ------------------------------------------------------------------

    def process_frame(
        self,
        user_audio: AudioFrame,
        asr_text: str,
        robot_audio: AudioFrame | None = None,
        frame_count: int = 1,
    ) -> TurnDecision:
        """Process one pipeline frame and return a turn decision."""
        vap_result = self._vap.feed_audio(user_audio, robot_audio)

        if self._turn_state is _TurnState.ROBOT_TURN:
            return self._process_robot_turn(vap_result, robot_audio)

        # --- USER_TURN ---
        elapsed = self._frame_duration_sec * frame_count

        # Poll latest TurnGPT result (non-blocking)
        latest_prob = self._turngpt.poll_result()
        if latest_prob is not None:
            self._turngpt_prob = latest_prob

        # Track ASR text changes
        text_changed = asr_text != self._prev_asr_text
        if text_changed and asr_text:
            self._turngpt.submit(self._build_dialog(asr_text))
            self._last_asr_change_elapsed_sec = 0.0
            self._asr_has_changed = True
        self._prev_asr_text = asr_text

        # Update timers (scaled by frame_count)
        if not vap_result.user_is_speaking:
            self._silence_elapsed_sec += elapsed
        else:
            self._silence_elapsed_sec = 0.0
            self._vap_favor_robot_elapsed_sec = 0.0
        self._last_asr_change_elapsed_sec += elapsed

        # --- Turn-shift check (only when user NOT speaking and text exists) ---
        if (
            not vap_result.user_is_speaking
            and asr_text
            and self._check_turn_shift(vap_result, elapsed)
        ):
            self._turn_state = _TurnState.ROBOT_TURN
            self._reset_per_frame_state()
            return TurnDecision(turn_shift=True)

        # --- Prepare check ---
        if self._check_prepare(asr_text):
            return TurnDecision(prepare=True)

        return TurnDecision.none()

    def notify_turn_complete(self, role: Literal["user", "robot"], text: str) -> None:
        """Append completed turn text to dialog context for TurnGPT."""
        if not text:
            return
        self._dialog_parts.append(text)

    def reset(self) -> None:
        """Reset per-frame tracking state for a new turn.

        Does NOT clear dialog_parts (TurnGPT context persists across turns).
        """
        self._turn_state = _TurnState.USER_TURN
        self._reset_per_frame_state()

    def _reset_per_frame_state(self) -> None:
        """Clear all per-frame tracking variables."""
        self._turngpt.clear_pending()
        self._prev_asr_text = ""
        self._vap_favor_robot_elapsed_sec = 0.0
        self._silence_elapsed_sec = 0.0
        self._last_asr_change_elapsed_sec = 0.0
        self._asr_has_changed = False
        self._last_prepare_text = ""
        self._turngpt_prob = 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_turn_shift(self, vap_result: VAPResult, elapsed: float) -> bool:
        """Check two OR paths for turn-shift.

        Path 1: VAP sustained robot-favor.
        Path 2: TurnGPT graduated silence timeout.
        """
        cfg = self._config

        # Path 1 — VAP: both p_now and p_fut favor robot (below user threshold)
        if vap_result.p_now < cfg.vap_user_threshold and vap_result.p_fut < cfg.vap_user_threshold:
            self._vap_favor_robot_elapsed_sec += elapsed
            if self._vap_favor_robot_elapsed_sec >= cfg.min_gap_time_sec:
                return True
        else:
            self._vap_favor_robot_elapsed_sec = 0.0

        # Path 2 — TurnGPT graduated timeout
        timeout = self._get_turngpt_timeout()
        return self._silence_elapsed_sec >= timeout

    def _get_turngpt_timeout(self) -> float:
        """Look up the silence timeout from graduated TurnGPT thresholds."""
        for prob_threshold, timeout in self._config.turngpt_thresholds:
            if self._turngpt_prob >= prob_threshold:
                return timeout
        # Fallback: last entry should always match (prob >= 0.0)
        return self._config.turngpt_thresholds[-1][1]

    def _process_robot_turn(
        self, vap_result: VAPResult, robot_audio: AudioFrame | None
    ) -> TurnDecision:
        """Interrupt detection during ROBOT_TURN."""
        cfg = self._config

        if robot_audio is not None:
            # VAP has both channels — distinguish interrupt vs backchannel
            if (
                vap_result.p_now > cfg.interrupt_user_threshold
                and vap_result.p_fut > cfg.interrupt_user_threshold
            ):
                return TurnDecision(interrupt=True)
            # p_now > threshold but p_fut <= threshold -> backchannel, no action
        else:
            # No robot audio (gap before playback starts)
            if vap_result.user_is_speaking:
                return TurnDecision(interrupt=True)

        return TurnDecision.none()

    def _check_prepare(self, asr_text: str) -> bool:
        """Check if speculative generation should be triggered."""
        if not self._asr_has_changed or not asr_text:
            return False

        cfg = self._config
        condition = (
            self._turngpt_prob > cfg.prepare_turngpt_threshold
            or self._last_asr_change_elapsed_sec >= cfg.prepare_timeout_sec
        )
        if not condition:
            return False

        # Similarity gate: skip if text is too similar to last prepare
        if self._last_prepare_text:
            similarity = SequenceMatcher(None, self._last_prepare_text, asr_text).ratio()
            if similarity >= cfg.prepare_similarity_threshold:
                return False

        self._last_prepare_text = asr_text
        self._asr_has_changed = False
        return True

    def _build_dialog(self, current_text: str) -> str:
        """Format dialog for TurnGPT: completed turns joined with <ts>."""
        if self._dialog_parts:
            return "<ts>".join(self._dialog_parts) + "<ts>" + current_text
        return current_text
