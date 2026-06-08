"""Combined turn-taking detector.

Fuses VAP (audio-based), TurnGPT (text-based), and timing heuristics
into a single per-frame TurnDecision.

Design reference: Skantze & Irfan (2025) — "Applying General Turn-taking
Models to Conversational Human-Robot Interaction".
"""

from __future__ import annotations

import enum
import logging
import time
from collections.abc import Callable
from typing import Literal

import numpy as np

from voice_pipeline.audio.constants import FRAME_DURATION_MS
from voice_pipeline.core.interfaces import IVAP, IEmbedder, ITurnDetector
from voice_pipeline.core.types import AudioFrame, TurnDecision, VAPResult
from voice_pipeline.turn_taking.async_turngpt import AsyncTurnGPT, SyncTurnGPTAdapter

logger = logging.getLogger("voice_pipeline.turn_taking")


class _TurnState(enum.Enum):
    USER_TURN = "user_turn"
    PENDING = "pending"  # turn_shift fired, awaiting commit (begin_streaming) or rewind (cancel)
    ROBOT_TURN = "robot_turn"


class TurnDetector(ITurnDetector):
    """Combined turn-taking detector using VAP + TurnGPT + timing heuristics.

    Two independent OR paths for turn-shift detection:
    - Path 1 (VAP): Sustained robot-favoring probabilities.
    - Path 2 (TurnGPT): Graduated silence timeout based on completion probability.

    Interrupt detection during ROBOT_TURN uses VAP to distinguish
    genuine interrupts from backchannels.

    After turn_shift the detector is PENDING (tentative). If the user
    resumes — VAP favoring the user, or dissimilar new ASR text — it emits
    ``cancel`` and rewinds to USER_TURN with state preserved. Otherwise the
    SessionLoop calls ``commit()`` (at begin_streaming) to enter ROBOT_TURN.

    Args:
        vap: 오디오 기반 voice activity projection 모델 (``IVAP``).
        turngpt: 텍스트 기반 turn-shift 예측 어댑터.
        embedder: prepare 유사도 게이트용 임베딩 공급자 (``IEmbedder``).
    """

    # VAP turn-shift 판정 (Path 1)
    _VAP_USER_THRESHOLD = 0.5  # p_now/p_fut가 이 값 미만이면 robot 선호
    _MIN_GAP_TIME_SEC = 0.5  # turn-shift 판정에 필요한 robot-선호 지속 시간 (초)

    # TurnGPT 단계별 timeout (Path 2) — (prob 하한, 무음 timeout 초)
    _TURNGPT_THRESHOLDS = (
        (0.3, 0.5),
        (0.2, 1.0),
        (0.1, 2.0),
        (0.0, 3.0),
    )

    # ROBOT_TURN 중 interrupt 판정
    _INTERRUPT_USER_THRESHOLD = 0.5  # p_now/p_fut가 이 값 초과면 user 선호

    # 외부 VAD 임계값 (vad_fn 사용 시)
    _EXT_VAD_THRESHOLD = 0.5

    # speculative 생성 트리거 (prepare)
    _PREPARE_TURNGPT_THRESHOLD = 0.2  # TurnGPT 확률이 이 값 초과면 prepare
    _PREPARE_TIMEOUT_SEC = 0.2  # 마지막 ASR 변화 후 이 시간 경과면 prepare
    _SIMILARITY_THRESHOLD = 0.8  # 직전 prepare 텍스트와의 유사도 이 값 이상이면 skip

    def __init__(
        self,
        vap: IVAP,
        turngpt: AsyncTurnGPT | SyncTurnGPTAdapter,
        embedder: IEmbedder,
        vad_fn: Callable[[AudioFrame], float] | None = None,
    ) -> None:
        self._vap = vap
        self._turngpt = turngpt
        self._embedder = embedder
        self._vad_fn = vad_fn

        self._frame_duration_sec = FRAME_DURATION_MS / 1000.0

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
        self._debug_vad_counter: int = 0

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

        if self._vad_fn is not None:
            vad_score = self._vad_fn(user_audio)
            user_is_speaking = vad_score > self._EXT_VAD_THRESHOLD
            if self._debug_vad_counter % 33 == 0:
                logger.debug("VAD score=%.3f speaking=%s silence=%.2fs", vad_score, user_is_speaking, self._silence_elapsed_sec)
            self._debug_vad_counter += 1
        else:
            user_is_speaking = vap_result.user_is_speaking

        if self._turn_state is _TurnState.ROBOT_TURN:
            return self._process_robot_turn(vap_result, user_is_speaking, robot_audio)

        # --- USER_TURN / PENDING: shared per-frame update ---
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
        if not user_is_speaking:
            self._silence_elapsed_sec += elapsed
        else:
            if self._silence_elapsed_sec > 0.5:
                logger.debug("Silence reset at %.2fs (was speaking)", self._silence_elapsed_sec)
            self._silence_elapsed_sec = 0.0
            self._vap_favor_robot_elapsed_sec = 0.0
        self._last_asr_change_elapsed_sec += elapsed

        # PENDING: turn-shift already fired but uncommitted. Same per-frame
        # tracking (so a rewind continues seamlessly), but only watch for the
        # user resuming → cancel. No turn_shift/prepare re-fire.
        if self._turn_state is _TurnState.PENDING:
            return self._process_pending(vap_result, user_is_speaking, asr_text, text_changed)

        # --- USER_TURN ---
        # Evaluate turn-shift first so its VAP-sustain timer keeps advancing even
        # on frames where prepare preempts the shift below (see _check_turn_shift's
        # side effect). The decision itself is acted on after the prepare check.
        turn_shift_ready = not user_is_speaking and asr_text and self._check_turn_shift(vap_result, elapsed)

        # Prepare preempts turn-shift on a fresh dissimilar change (→ regenerate).
        if self._check_prepare(asr_text):
            prob = self._turngpt_prob
            thresh = self._PREPARE_TURNGPT_THRESHOLD
            reason = (
                f"turngpt={prob:.2f}>{thresh:.2f}"
                if prob > thresh
                else f"timeout={self._last_asr_change_elapsed_sec:.2f}s"
            )
            logger.debug("PREPARE (%s): text=%r", reason, asr_text[:60])
            return TurnDecision(prepare=True)

        # --- Turn-shift (text has settled / still matches the last prepare) ---
        if turn_shift_ready:
            logger.info("TURN_SHIFT: %r", asr_text[:60])
            logger.debug(
                "TURN_SHIFT detail: p_now=%.2f p_fut=%.2f turngpt=%.2f silence=%.2fs",
                vap_result.p_now,
                vap_result.p_fut,
                self._turngpt_prob,
                self._silence_elapsed_sec,
            )
            # Tentative: enter PENDING (commit/rewind decides). Do NOT wipe
            # per-frame state — a cancel must be able to resume this turn.
            self._turn_state = _TurnState.PENDING
            return TurnDecision(turn_shift=True)

        return TurnDecision.none()

    def notify_turn_complete(self, role: Literal["user", "robot"], text: str) -> None:
        """Append completed turn text to dialog context for TurnGPT."""
        if not text:
            return
        self._dialog_parts.append(text)

    def commit(self, text: str) -> None:
        """Commit the pending turn-shift (PENDING → ROBOT_TURN).

        Appends *text* as the completed user turn to TurnGPT dialog context,
        then wipes per-frame tracking. After commit, cancel is no longer
        possible. Called by SessionLoop at begin_streaming.
        """
        if text:
            self._dialog_parts.append(text)
        self._turn_state = _TurnState.ROBOT_TURN
        self._reset_per_frame_state()

    def reset(self) -> None:
        """Reset to a fresh USER_TURN for a new turn.

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
        # Path 1 — VAP: both p_now and p_fut favor robot (below user threshold)
        if vap_result.p_now < self._VAP_USER_THRESHOLD and vap_result.p_fut < self._VAP_USER_THRESHOLD:
            self._vap_favor_robot_elapsed_sec += elapsed
            if self._vap_favor_robot_elapsed_sec >= self._MIN_GAP_TIME_SEC:
                return True
        else:
            self._vap_favor_robot_elapsed_sec = 0.0

        # Path 2 — TurnGPT graduated timeout
        timeout = self._get_turngpt_timeout()
        return self._silence_elapsed_sec >= timeout

    def _get_turngpt_timeout(self) -> float:
        """Look up the silence timeout from graduated TurnGPT thresholds."""
        for prob_threshold, timeout in self._TURNGPT_THRESHOLDS:
            if self._turngpt_prob >= prob_threshold:
                return timeout
        # Fallback: last entry should always match (prob >= 0.0)
        return self._TURNGPT_THRESHOLDS[-1][1]

    def _process_robot_turn(
        self, vap_result: VAPResult, user_is_speaking: bool, robot_audio: AudioFrame | None
    ) -> TurnDecision:
        """Interrupt detection during ROBOT_TURN.

        Follows Skantze & Irfan (2025) pseudocode: user_is_speaking is a
        prerequisite for interrupt checking. When user IS speaking and
        robot_audio is available, use p_now/p_fut to distinguish interrupt
        vs backchannel. Without robot_audio, VAP lacks the robot channel
        to make this distinction, so no interrupt decision is made.
        """
        if not user_is_speaking:
            return TurnDecision.none()

        # VAP needs both channels to distinguish interrupt vs backchannel
        if (
            robot_audio is not None
            and vap_result.p_now > self._INTERRUPT_USER_THRESHOLD
            and vap_result.p_fut > self._INTERRUPT_USER_THRESHOLD
        ):
            logger.info("INTERRUPT (vap)")
            logger.debug(
                "INTERRUPT detail: p_now=%.2f p_fut=%.2f",
                vap_result.p_now,
                vap_result.p_fut,
            )
            return TurnDecision(interrupt=True)

        return TurnDecision.none()

    def _process_pending(
        self, vap_result: VAPResult, user_is_speaking: bool, asr_text: str, text_changed: bool
    ) -> TurnDecision:
        """Cancel detection during the tentative PENDING window.

        The user reclaiming the floor means the turn_shift was premature →
        cancel. ``user_is_speaking`` is a prerequisite (mirrors interrupt) —
        the user must actually be speaking — then one of two signals confirms
        the reclaim:
        - VAP: p_now/p_fut both favor the user (immediate; each VAP result
          already integrates ~100ms, so no sustain is needed).
        - ASR: new text dissimilar to the last prepared text — the basis of
          the (speculative) response (a finalization stays similar, ignored).
        On cancel, rewind to USER_TURN with per-frame state preserved.
        """
        if not user_is_speaking:
            return TurnDecision.none()

        if vap_result.p_now > self._INTERRUPT_USER_THRESHOLD and vap_result.p_fut > self._INTERRUPT_USER_THRESHOLD:
            logger.info("CANCEL (vap): p_now=%.2f p_fut=%.2f", vap_result.p_now, vap_result.p_fut)
            self._turn_state = _TurnState.USER_TURN
            return TurnDecision(cancel=True)

        if text_changed and asr_text and self._last_prepare_text:
            similarity = self._text_similarity(self._last_prepare_text, asr_text)
            if similarity < self._SIMILARITY_THRESHOLD:
                logger.info(
                    "CANCEL (asr): similarity=%.2f %r → %r",
                    similarity,
                    self._last_prepare_text[:40],
                    asr_text[:40],
                )
                self._turn_state = _TurnState.USER_TURN
                return TurnDecision(cancel=True)

        return TurnDecision.none()

    _SIMILARITY_SLOW_MS = 100

    def _text_similarity(self, a: str, b: str) -> float:
        """Cosine similarity between two texts via the embedder."""
        t0 = time.monotonic()
        vecs = self._embedder.embed_batch([a, b])
        elapsed_ms = (time.monotonic() - t0) * 1000
        if elapsed_ms > self._SIMILARITY_SLOW_MS:
            logger.warning("Similarity slow: %.0fms (budget %dms)", elapsed_ms, self._SIMILARITY_SLOW_MS)
        return float(np.dot(vecs[0], vecs[1]) / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]) + 1e-9))

    def _check_prepare(self, asr_text: str) -> bool:
        """Check if speculative generation should be triggered."""
        if not self._asr_has_changed or not asr_text:
            return False

        condition = (
            self._turngpt_prob > self._PREPARE_TURNGPT_THRESHOLD
            or self._last_asr_change_elapsed_sec >= self._PREPARE_TIMEOUT_SEC
        )
        if not condition:
            return False

        # Similarity gate: skip if text is too similar to last prepare
        if self._last_prepare_text:
            similarity = self._text_similarity(self._last_prepare_text, asr_text)
            if similarity >= self._SIMILARITY_THRESHOLD:
                logger.debug(
                    "PREPARE skipped (similarity=%.2f): %r → %r",
                    similarity,
                    self._last_prepare_text[:40],
                    asr_text[:40],
                )
                self._asr_has_changed = False
                return False

        self._last_prepare_text = asr_text
        self._asr_has_changed = False
        return True

    def _build_dialog(self, current_text: str) -> str:
        """Format dialog for TurnGPT: completed turns joined with <ts>."""
        if self._dialog_parts:
            return "<ts>".join(self._dialog_parts) + "<ts>" + current_text
        return current_text
