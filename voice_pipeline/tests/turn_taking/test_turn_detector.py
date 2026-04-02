"""Unit tests for TurnDetector.

All external dependencies (IVAP, ITurnGPT) are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from voice_pipeline.core.config import AudioConfig, TurnDetectorConfig
from voice_pipeline.core.interfaces import IVAP, IEmbedder, ITurnGPT
from voice_pipeline.core.types import TurnDecision, VAPResult
from voice_pipeline.turn_taking.async_turngpt import SyncTurnGPTAdapter
from voice_pipeline.turn_taking.turn_detector import TurnDetector, _TurnState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 30ms frames at 16kHz -> each frame = 0.03s
AUDIO_CFG = AudioConfig(sample_rate=16000, channels=1, frame_duration_ms=30)
FRAME = b"\x00" * (16000 * 30 // 1000 * 2)  # 30ms of silence at 16kHz, 16-bit
ROBOT_FRAME = b"\x00" * 100  # dummy robot audio


def _make_embedder_mock(similarity: float = 0.0) -> MagicMock:
    """Create a mock IEmbedder that returns vectors with the given cosine similarity."""
    mock = MagicMock(spec=IEmbedder)
    if similarity >= 1.0:
        # Identical vectors
        vec = np.array([1.0, 0.0], dtype=np.float32)
        mock.embed_batch.return_value = np.array([vec, vec])
    elif similarity <= 0.0:
        # Orthogonal vectors
        mock.embed_batch.return_value = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    else:
        # Vectors at desired angle: cos(theta) = similarity
        theta = np.arccos(similarity)
        vec_a = np.array([1.0, 0.0], dtype=np.float32)
        vec_b = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        mock.embed_batch.return_value = np.array([vec_a, vec_b])
    return mock


def _make_detector(
    config: TurnDetectorConfig | None = None,
    vap_results: list[VAPResult] | None = None,
    turngpt_prob: float = 0.0,
    embedder: MagicMock | None = None,
) -> tuple[TurnDetector, MagicMock, MagicMock]:
    """Create a TurnDetector with mocked VAP and TurnGPT.

    Returns (detector, mock_vap, mock_turngpt).
    """
    mock_vap = MagicMock(spec=IVAP)
    mock_turngpt = MagicMock(spec=ITurnGPT)

    if vap_results:
        mock_vap.feed_audio.side_effect = vap_results
    else:
        # Default: no speech, neutral probabilities
        mock_vap.feed_audio.return_value = VAPResult(0.5, 0.5, False)

    mock_turngpt.predict.return_value = turngpt_prob

    cfg = config or TurnDetectorConfig()
    if embedder is None:
        embedder = _make_embedder_mock(similarity=0.0)
    adapter = SyncTurnGPTAdapter(mock_turngpt)
    detector = TurnDetector(mock_vap, adapter, embedder, cfg, AUDIO_CFG)
    return detector, mock_vap, mock_turngpt


def _silent_robot_favoring(n: int) -> list[VAPResult]:
    """N frames where VAP favors robot and user is not speaking."""
    return [VAPResult(p_now=0.2, p_fut=0.2, user_is_speaking=False)] * n


def _process_n_frames(
    detector: TurnDetector,
    n: int,
    asr_text: str = "",
    robot_audio: bytes | None = None,
) -> list[TurnDecision]:
    """Process n frames and return all decisions."""
    return [detector.process_frame(FRAME, asr_text, robot_audio) for _ in range(n)]


# ---------------------------------------------------------------------------
# Test 1: No-op frame
# ---------------------------------------------------------------------------


class TestNoOpFrame:
    def test_no_speech_no_text_returns_none(self):
        detector, _, _ = _make_detector()
        decision = detector.process_frame(FRAME, "")
        assert decision == TurnDecision.none()

    def test_all_fields_false(self):
        detector, _, _ = _make_detector()
        decision = detector.process_frame(FRAME, "")
        assert not decision.turn_shift
        assert not decision.interrupt
        assert not decision.prepare


# ---------------------------------------------------------------------------
# Test 2: VAP turn-shift path (Path 1)
# ---------------------------------------------------------------------------


class TestVAPTurnShift:
    def test_sustained_robot_favor_triggers_turn_shift(self):
        """500ms of sustained robot-favoring VAP -> turn_shift."""
        # 500ms / 30ms = ~17 frames needed
        n_frames = 18
        vap_results = _silent_robot_favoring(n_frames)
        detector, _, _ = _make_detector(vap_results=vap_results)

        # Feed one frame with text to seed ASR (required for turn_shift)
        detector.process_frame(FRAME, "hello")

        # Remaining frames: user not speaking, text present
        decisions = []
        for _i in range(1, n_frames):
            decisions.append(detector.process_frame(FRAME, "hello"))

        assert any(d.turn_shift for d in decisions)

    def test_vap_timer_resets_when_favor_stops(self):
        """VAP favors robot then stops before 500ms -> no turn_shift."""
        # 10 frames (~300ms) of robot favor, then neutral
        vap_results = (
            _silent_robot_favoring(11) + [VAPResult(0.8, 0.8, False)] * 10  # neutral/user-favoring
        )
        detector, _, _ = _make_detector(
            vap_results=vap_results,
            turngpt_prob=0.0,  # low prob -> 3s timeout (won't fire)
        )

        # Seed ASR text
        detector.process_frame(FRAME, "hello")

        decisions = [detector.process_frame(FRAME, "hello") for _ in range(20)]
        assert not any(d.turn_shift for d in decisions)

    def test_vap_timer_resets_when_user_speaks(self):
        """VAP favors robot 300ms -> user speaks briefly -> resumes favor.

        Timer should restart from zero after speech, not accumulate.
        """
        # 10 frames robot favor (~300ms) + 3 frames user speaking + 10 frames robot favor (~300ms)
        vap_results = (
            _silent_robot_favoring(11)  # 1 seed + 10 favor
            + [VAPResult(0.2, 0.2, True)] * 3  # user speaking (resets timer)
            + _silent_robot_favoring(10)  # resume favor, but only 300ms
        )
        detector, _, _ = _make_detector(
            vap_results=vap_results,
            turngpt_prob=0.0,  # low prob -> 3s timeout (won't fire)
        )

        detector.process_frame(FRAME, "hello")
        decisions = [detector.process_frame(FRAME, "hello") for _ in range(23)]
        # Neither 300ms segment is enough for 500ms threshold
        assert not any(d.turn_shift for d in decisions)


# ---------------------------------------------------------------------------
# Test 4: TurnGPT graduated timeout
# ---------------------------------------------------------------------------


class TestTurnGPTTimeout:
    def test_high_prob_short_timeout(self):
        """prob=0.3 -> 500ms timeout -> turn_shift after ~17 frames of silence."""
        n_frames = 20
        vap_results = [VAPResult(0.8, 0.8, False)] * n_frames  # VAP won't trigger Path 1
        detector, _, turngpt = _make_detector(vap_results=vap_results, turngpt_prob=0.3)

        # First frame: set ASR text (triggers TurnGPT call)
        detector.process_frame(FRAME, "hello")

        # Remaining: silence with text present
        decisions = [detector.process_frame(FRAME, "hello") for _ in range(n_frames - 1)]
        assert any(d.turn_shift for d in decisions)

    def test_low_prob_long_timeout(self):
        """prob=0.05 -> needs 3000ms (100 frames) of silence."""
        n_frames = 105
        vap_results = [VAPResult(0.8, 0.8, False)] * n_frames
        detector, _, _ = _make_detector(vap_results=vap_results, turngpt_prob=0.05)

        detector.process_frame(FRAME, "hello")

        decisions = [detector.process_frame(FRAME, "hello") for _ in range(n_frames - 1)]
        # Should fire around frame 100 (3000ms / 30ms)
        shift_indices = [i for i, d in enumerate(decisions) if d.turn_shift]
        assert len(shift_indices) > 0
        # First turn_shift should be around 3s mark
        assert shift_indices[0] >= 95  # ~2850ms minimum


# ---------------------------------------------------------------------------
# Test 6: Turn-shift transitions to ROBOT_TURN
# ---------------------------------------------------------------------------


class TestRobotTurnTransition:
    def test_after_turn_shift_no_interrupt_without_robot_audio(self):
        """After turn_shift, user speech without robot_audio produces no interrupt.

        Without robot audio, VAP cannot distinguish interrupt from backchannel.
        The orchestrator handles this case via awaiting cancel on ASR text change.
        """
        n_shift = 20
        vap_shift = _silent_robot_favoring(n_shift)
        detector, mock_vap, _ = _make_detector(vap_results=vap_shift)

        # Drive to turn_shift
        detector.process_frame(FRAME, "hello")
        for _ in range(n_shift - 1):
            d = detector.process_frame(FRAME, "hello")
            if d.turn_shift:
                break

        assert detector._turn_state is _TurnState.ROBOT_TURN

        # Switch mock to user-speaking for the interrupt frame
        mock_vap.feed_audio.side_effect = None
        mock_vap.feed_audio.return_value = VAPResult(0.8, 0.8, True)

        # In ROBOT_TURN without robot_audio -> no interrupt (deferred to orchestrator)
        decision = detector.process_frame(FRAME, "", None)
        assert decision == TurnDecision.none()


# ---------------------------------------------------------------------------
# Test 7: Interrupt with robot_audio
# ---------------------------------------------------------------------------


class TestInterruptWithRobotAudio:
    def test_both_favor_user_triggers_interrupt(self):
        """Both p_now and p_fut favor user + speaking -> interrupt."""
        detector, mock_vap, _ = _make_detector()
        detector._turn_state = _TurnState.ROBOT_TURN
        mock_vap.feed_audio.return_value = VAPResult(0.8, 0.8, True)

        decision = detector.process_frame(FRAME, "", ROBOT_FRAME)
        assert decision.interrupt

    def test_not_speaking_no_interrupt(self):
        """user_is_speaking=False -> no interrupt even if p_now/p_fut favor user."""
        detector, mock_vap, _ = _make_detector()
        detector._turn_state = _TurnState.ROBOT_TURN
        mock_vap.feed_audio.return_value = VAPResult(0.8, 0.8, False)

        decision = detector.process_frame(FRAME, "", ROBOT_FRAME)
        assert decision == TurnDecision.none()

    def test_backchannel_no_interrupt(self):
        """p_now favors user, p_fut favors robot -> backchannel, no interrupt."""
        detector, mock_vap, _ = _make_detector()
        detector._turn_state = _TurnState.ROBOT_TURN
        mock_vap.feed_audio.return_value = VAPResult(0.8, 0.2, True)

        decision = detector.process_frame(FRAME, "", ROBOT_FRAME)
        assert decision == TurnDecision.none()


# ---------------------------------------------------------------------------
# Test 9: Interrupt without robot_audio
# ---------------------------------------------------------------------------


class TestInterruptWithoutRobotAudio:
    def test_user_speaking_no_interrupt_without_robot_audio(self):
        """In ROBOT_TURN without robot_audio, user_is_speaking -> no interrupt.

        Without robot audio, VAP cannot distinguish interrupt from backchannel.
        No interrupt decision is made; orchestrator handles via awaiting cancel.
        """
        detector, mock_vap, _ = _make_detector()
        detector._turn_state = _TurnState.ROBOT_TURN
        mock_vap.feed_audio.return_value = VAPResult(0.3, 0.3, True)

        decision = detector.process_frame(FRAME, "", None)
        assert decision == TurnDecision.none()

    def test_no_user_speech_no_interrupt(self):
        """In ROBOT_TURN without robot_audio, no speech -> no interrupt."""
        detector, mock_vap, _ = _make_detector()
        detector._turn_state = _TurnState.ROBOT_TURN
        mock_vap.feed_audio.return_value = VAPResult(0.3, 0.3, False)

        decision = detector.process_frame(FRAME, "", None)
        assert decision == TurnDecision.none()


# ---------------------------------------------------------------------------
# Test 10: Prepare fires on TurnGPT threshold
# ---------------------------------------------------------------------------


class TestPrepare:
    def test_prepare_on_turngpt_threshold(self):
        """Text changed + TurnGPT prob > 0.2 -> prepare.

        With submit/poll, the TurnGPT result is available one frame after
        the text change that triggered the submit.
        """
        n_frames = 3
        vap_results = [VAPResult(0.5, 0.5, True)] * n_frames
        detector, _, _ = _make_detector(vap_results=vap_results, turngpt_prob=0.3)

        # Frame 1: text changes, submit fires (result not yet available)
        detector.process_frame(FRAME, "hello")

        # Frame 2: poll picks up prob=0.3, prepare fires
        decision = detector.process_frame(FRAME, "hello")
        assert decision.prepare

    def test_prepare_on_timeout(self):
        """Text changed + 200ms elapsed since change -> prepare."""
        n_frames = 10  # 300ms
        vap_results = [VAPResult(0.5, 0.5, True)] * n_frames
        detector, _, _ = _make_detector(
            vap_results=vap_results,
            turngpt_prob=0.1,  # below 0.2 threshold
        )

        # First frame: text changes
        d = detector.process_frame(FRAME, "hello")
        assert not d.prepare  # turngpt_prob=0.1 < 0.2, not enough time elapsed

        # After ~7 frames (210ms), prepare should fire
        decisions = [detector.process_frame(FRAME, "hello") for _ in range(n_frames - 1)]
        assert any(d.prepare for d in decisions)


# ---------------------------------------------------------------------------
# Test 12: Prepare similarity gate
# ---------------------------------------------------------------------------


class TestPrepareSimilarityGate:
    def test_similar_text_skips_prepare(self):
        """Similar text to last prepare -> skipped."""
        n_frames = 5
        vap_results = [VAPResult(0.5, 0.5, True)] * n_frames
        mock_emb = _make_embedder_mock(similarity=0.9)
        detector, _, _ = _make_detector(
            vap_results=vap_results,
            turngpt_prob=0.5,
            embedder=mock_emb,
        )

        # Frame 1: text changes, submit fires
        detector.process_frame(FRAME, "hello world")
        # Frame 2: first prepare fires (no last_prepare_text yet)
        d1 = detector.process_frame(FRAME, "hello world")
        assert d1.prepare

        # Frame 3: similar text change, submit fires
        detector.process_frame(FRAME, "hello worlds")
        # Frame 4: similarity gate blocks (0.9 >= 0.8 threshold)
        d2 = detector.process_frame(FRAME, "hello worlds")
        assert not d2.prepare

    def test_different_text_fires_prepare(self):
        """Sufficiently different text -> prepare fires again."""
        n_frames = 4
        vap_results = [VAPResult(0.5, 0.5, True)] * n_frames
        mock_emb = _make_embedder_mock(similarity=0.3)
        detector, _, _ = _make_detector(
            vap_results=vap_results,
            turngpt_prob=0.5,
            embedder=mock_emb,
        )

        # Frame 1-2: first prepare fires
        detector.process_frame(FRAME, "hello world")
        d1 = detector.process_frame(FRAME, "hello world")
        assert d1.prepare

        # Frame 3: different text, prepare fires (similarity low)
        d2 = detector.process_frame(FRAME, "completely different sentence here")
        assert d2.prepare


# ---------------------------------------------------------------------------
# Test 13: reset() returns to USER_TURN
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_returns_to_user_turn(self):
        detector, _, _ = _make_detector()
        detector._turn_state = _TurnState.ROBOT_TURN
        detector._silence_elapsed_sec = 5.0
        detector._turngpt_prob = 0.9

        detector.reset()

        assert detector._turn_state is _TurnState.USER_TURN
        assert detector._silence_elapsed_sec == 0.0
        assert detector._turngpt_prob == 0.0

    def test_reset_preserves_dialog_parts(self):
        detector, _, _ = _make_detector()
        detector.notify_turn_complete("user", "hello")
        detector.notify_turn_complete("robot", "hi there")

        detector.reset()

        assert len(detector._dialog_parts) == 2


# ---------------------------------------------------------------------------
# Test 14: notify_turn_complete builds dialog
# ---------------------------------------------------------------------------


class TestNotifyTurnComplete:
    def test_builds_dialog_format(self):
        detector, _, _ = _make_detector()
        detector.notify_turn_complete("user", "hello")
        detector.notify_turn_complete("robot", "hi there")

        dialog = detector._build_dialog("how are you")
        assert dialog == "hello<ts>hi there<ts>how are you"

    def test_empty_text_ignored(self):
        detector, _, _ = _make_detector()
        detector.notify_turn_complete("user", "")
        detector.notify_turn_complete("robot", "hi")

        assert detector._dialog_parts == ["hi"]

    def test_single_turn_no_prefix(self):
        detector, _, _ = _make_detector()
        dialog = detector._build_dialog("hello")
        assert dialog == "hello"


# ---------------------------------------------------------------------------
# Test 15: Turn-shift requires non-empty ASR text
# ---------------------------------------------------------------------------


class TestPrepareEmptyText:
    def test_prepare_not_fired_on_empty_text(self):
        """If ASR text goes from non-empty to empty, prepare should not fire."""
        n_frames = 4
        vap_results = [VAPResult(0.5, 0.5, True)] * n_frames
        detector, _, _ = _make_detector(vap_results=vap_results, turngpt_prob=0.5)

        # Frame 1: text changes, submit fires
        detector.process_frame(FRAME, "hello")
        # Frame 2: poll picks up prob, prepare fires
        d1 = detector.process_frame(FRAME, "hello")
        assert d1.prepare

        # Frame 3: ASR clears to empty — should NOT fire prepare
        d2 = detector.process_frame(FRAME, "")
        assert not d2.prepare


# ---------------------------------------------------------------------------
# Test 15: Turn-shift requires non-empty ASR text
# ---------------------------------------------------------------------------


class TestTurnShiftRequiresText:
    def test_silence_without_text_no_turn_shift(self):
        """Even with sustained robot-favoring VAP, empty ASR -> no turn_shift."""
        n_frames = 25
        vap_results = _silent_robot_favoring(n_frames)
        detector, _, _ = _make_detector(vap_results=vap_results)

        decisions = _process_n_frames(detector, n_frames, asr_text="")
        assert not any(d.turn_shift for d in decisions)


# ---------------------------------------------------------------------------
# Test 16: Turn-shift only when user not speaking
# ---------------------------------------------------------------------------


class TestTurnShiftOnlyWhenNotSpeaking:
    def test_user_speaking_blocks_turn_shift(self):
        """Even with robot-favoring p_now/p_fut, user_is_speaking blocks shift."""
        n_frames = 25
        # VAP favors robot but user IS speaking
        vap_results = [VAPResult(0.2, 0.2, True)] * n_frames
        detector, _, _ = _make_detector(vap_results=vap_results, turngpt_prob=0.5)

        detector.process_frame(FRAME, "hello")
        decisions = [detector.process_frame(FRAME, "hello") for _ in range(n_frames - 1)]
        assert not any(d.turn_shift for d in decisions)
