"""Unit tests for the combined TurnDetector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from voice_pipeline.core.config import AudioConfig, TurnDetectorConfig
from voice_pipeline.core.types import TurnDecision, VAPResult
from voice_pipeline.turn_taking.turn_detector import TurnDetector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPEAKING = VAPResult(p_now=0.8, p_fut=0.7, user_is_speaking=True)
_SILENT = VAPResult(p_now=0.1, p_fut=0.1, user_is_speaking=False)
_FRAME = b"\x00" * 960  # 30ms at 16kHz, 16-bit mono
_ROBOT_FRAME = b"\x00" * 1440  # 30ms at 24kHz, 16-bit mono


def _make_detector(
    *,
    config: TurnDetectorConfig | None = None,
    vap_result: VAPResult = _SILENT,
    turngpt_prob: float = 0.0,
) -> tuple[TurnDetector, MagicMock, MagicMock]:
    """Create a TurnDetector with mocked VAP and TurnGPT."""
    mock_vap = MagicMock()
    mock_vap.feed_audio.return_value = vap_result

    mock_turngpt = MagicMock()
    mock_turngpt.predict.return_value = turngpt_prob

    cfg = config or TurnDetectorConfig()
    audio_cfg = AudioConfig()
    detector = TurnDetector(mock_vap, mock_turngpt, cfg, audio_cfg)
    return detector, mock_vap, mock_turngpt


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    def test_constructor_stores_config(self):
        cfg = TurnDetectorConfig(turn_shift_silence_frames=50)
        detector, _, _ = _make_detector(config=cfg)
        assert detector._config.turn_shift_silence_frames == 50

    def test_initial_state(self):
        detector, _, _ = _make_detector()
        assert detector._prev_asr_text == ""
        assert detector._prepare_fired is False
        assert detector._silence_frame_count == 0
        assert detector._dialog_turns == []
        assert detector._current_partial == ""


# ---------------------------------------------------------------------------
# TestInterrupt
# ---------------------------------------------------------------------------


class TestInterrupt:
    def test_robot_audio_and_user_speaking_triggers_interrupt(self):
        detector, _, _ = _make_detector(vap_result=_SPEAKING)
        result = detector.process_frame(_FRAME, "", robot_audio=_ROBOT_FRAME)
        assert result.interrupt is True

    def test_no_robot_audio_no_interrupt(self):
        """No interrupt when robot_audio is None (robot not speaking)."""
        detector, _, _ = _make_detector(vap_result=_SPEAKING)
        result = detector.process_frame(_FRAME, "", robot_audio=None)
        assert result.interrupt is False

    def test_robot_audio_user_silent_no_interrupt(self):
        detector, _, _ = _make_detector(vap_result=_SILENT)
        result = detector.process_frame(_FRAME, "", robot_audio=_ROBOT_FRAME)
        assert result.interrupt is False


# ---------------------------------------------------------------------------
# TestTextChange
# ---------------------------------------------------------------------------


class TestTextChange:
    @patch("voice_pipeline.turn_taking.turn_detector.time")
    def test_text_change_resets_stability(self, mock_time):
        mock_time.monotonic.return_value = 10.0
        detector, _, _ = _make_detector()

        detector.process_frame(_FRAME, "hello world")
        assert detector._prev_asr_text == "hello world"
        assert detector._text_stable_since == 10.0

        mock_time.monotonic.return_value = 11.0
        detector.process_frame(_FRAME, "hello world changed")
        assert detector._text_stable_since == 11.0
        assert detector._prepare_fired is False

    def test_normalization_no_false_change(self):
        """'Hello' vs 'hello' should not trigger a text change."""
        detector, _, _ = _make_detector()
        detector.process_frame(_FRAME, "Hello")
        detector.process_frame(_FRAME, "hello")
        # Both normalize to "hello" — no reset expected
        # The first frame sets it; the second shouldn't change stability
        assert detector._prev_asr_text == "hello"

    def test_whitespace_normalization(self):
        """Leading/trailing whitespace should not trigger change."""
        detector, _, _ = _make_detector()
        detector.process_frame(_FRAME, "test")
        detector.process_frame(_FRAME, "  test  ")
        assert detector._prev_asr_text == "test"

    @patch("voice_pipeline.turn_taking.turn_detector.time")
    def test_asr_correction_detected(self, mock_time):
        """Significant text correction should be detected as change."""
        mock_time.monotonic.return_value = 1.0
        detector, _, _ = _make_detector()
        detector.process_frame(_FRAME, "I want to go")

        mock_time.monotonic.return_value = 2.0
        detector.process_frame(_FRAME, "I want to know")
        assert detector._text_stable_since == 2.0

    @patch("voice_pipeline.turn_taking.turn_detector.time")
    def test_text_change_resets_prepare_flag(self, mock_time):
        mock_time.monotonic.return_value = 1.0
        detector, _, _ = _make_detector(turngpt_prob=0.5)
        detector.process_frame(_FRAME, "hello")

        # Advance past stability window
        mock_time.monotonic.return_value = 2.0
        detector.process_frame(_FRAME, "hello")
        assert detector._prepare_fired is True

        # Text changes — prepare flag resets
        mock_time.monotonic.return_value = 3.0
        detector.process_frame(_FRAME, "completely different text")
        assert detector._prepare_fired is False


# ---------------------------------------------------------------------------
# TestPrepare
# ---------------------------------------------------------------------------


class TestPrepare:
    @patch("voice_pipeline.turn_taking.turn_detector.time")
    def test_prepare_fires_when_stable_and_turngpt_above_threshold(self, mock_time):
        mock_time.monotonic.return_value = 1.0
        detector, _, _ = _make_detector(turngpt_prob=0.5)

        detector.process_frame(_FRAME, "hello there")
        # Advance past prepare_stable_ms (default 800ms)
        mock_time.monotonic.return_value = 2.0
        result = detector.process_frame(_FRAME, "hello there")
        assert result.prepare is True

    @patch("voice_pipeline.turn_taking.turn_detector.time")
    def test_prepare_one_shot(self, mock_time):
        """Prepare fires only once per text revision."""
        mock_time.monotonic.return_value = 1.0
        detector, _, _ = _make_detector(turngpt_prob=0.5)

        detector.process_frame(_FRAME, "hello")
        mock_time.monotonic.return_value = 2.0
        result1 = detector.process_frame(_FRAME, "hello")
        assert result1.prepare is True

        mock_time.monotonic.return_value = 3.0
        result2 = detector.process_frame(_FRAME, "hello")
        assert result2.prepare is False

    @patch("voice_pipeline.turn_taking.turn_detector.time")
    def test_text_change_allows_re_fire(self, mock_time):
        mock_time.monotonic.return_value = 1.0
        detector, _, _ = _make_detector(turngpt_prob=0.5)

        detector.process_frame(_FRAME, "hello")
        mock_time.monotonic.return_value = 2.0
        detector.process_frame(_FRAME, "hello")
        assert detector._prepare_fired is True

        # Text changes
        mock_time.monotonic.return_value = 3.0
        detector.process_frame(_FRAME, "goodbye world")
        assert detector._prepare_fired is False

        # New prepare fires
        mock_time.monotonic.return_value = 4.0
        result = detector.process_frame(_FRAME, "goodbye world")
        assert result.prepare is True

    @patch("voice_pipeline.turn_taking.turn_detector.time")
    def test_turngpt_below_threshold_no_prepare(self, mock_time):
        mock_time.monotonic.return_value = 1.0
        detector, _, _ = _make_detector(turngpt_prob=0.1)

        detector.process_frame(_FRAME, "hello")
        mock_time.monotonic.return_value = 2.0
        result = detector.process_frame(_FRAME, "hello")
        assert result.prepare is False
        assert detector._prepare_fired is False

    @patch("voice_pipeline.turn_taking.turn_detector.time")
    def test_empty_text_no_prepare(self, mock_time):
        mock_time.monotonic.return_value = 1.0
        detector, _, _ = _make_detector(turngpt_prob=0.9)

        mock_time.monotonic.return_value = 2.0
        result = detector.process_frame(_FRAME, "")
        assert result.prepare is False


# ---------------------------------------------------------------------------
# TestTurnShift
# ---------------------------------------------------------------------------


class TestTurnShift:
    def test_silence_frames_trigger_turn_shift(self):
        cfg = TurnDetectorConfig(turn_shift_silence_frames=5)
        detector, _, _ = _make_detector(config=cfg)

        # First frame sets prev_asr_text (silence_count goes 0→1)
        detector.process_frame(_FRAME, "hello")
        # Frames 2-4: count goes 2,3,4 — still below 5
        for _ in range(3):
            result = detector.process_frame(_FRAME, "hello")
            assert result.turn_shift is False

        # Frame 5: count=5 >= 5 → turn_shift
        result = detector.process_frame(_FRAME, "hello")
        assert result.turn_shift is True

    @patch("voice_pipeline.turn_taking.turn_detector.time")
    def test_hard_silence_timeout(self, mock_time):
        cfg = TurnDetectorConfig(
            turn_shift_silence_frames=1000,  # Won't fire from frame count
            hard_silence_timeout_ms=500,
        )
        mock_time.monotonic.return_value = 1.0
        detector, _, _ = _make_detector(config=cfg)

        detector.process_frame(_FRAME, "hello")
        # Text hasn't changed, but hard timeout passes
        mock_time.monotonic.return_value = 1.6
        result = detector.process_frame(_FRAME, "hello")
        assert result.turn_shift is True

    def test_no_turn_shift_without_prior_speech(self):
        """No turn_shift when _prev_asr_text is empty (no speech yet)."""
        cfg = TurnDetectorConfig(turn_shift_silence_frames=1)
        detector, _, _ = _make_detector(config=cfg)

        # Multiple silent frames with no text
        for _ in range(10):
            result = detector.process_frame(_FRAME, "")
        assert result.turn_shift is False

    def test_user_speaking_resets_silence_counter(self):
        cfg = TurnDetectorConfig(turn_shift_silence_frames=3)
        detector, mock_vap, _ = _make_detector(config=cfg)

        detector.process_frame(_FRAME, "hello")
        # Two silent frames
        detector.process_frame(_FRAME, "hello")
        detector.process_frame(_FRAME, "hello")

        # User speaks — resets counter
        mock_vap.feed_audio.return_value = _SPEAKING
        detector.process_frame(_FRAME, "hello")
        assert detector._silence_frame_count == 0

        # Need 3 more silent frames now
        mock_vap.feed_audio.return_value = _SILENT
        result = detector.process_frame(_FRAME, "hello")
        assert result.turn_shift is False


# ---------------------------------------------------------------------------
# TestNotifyTurnComplete
# ---------------------------------------------------------------------------


class TestNotifyTurnComplete:
    def test_appends_to_dialog_and_clears_partial(self):
        detector, _, _ = _make_detector()
        detector._current_partial = "work in progress"
        detector.notify_turn_complete("user", "Hello there")
        assert detector._dialog_turns == ["Hello there"]
        assert detector._current_partial == ""

    def test_empty_text_is_noop(self):
        detector, _, _ = _make_detector()
        detector.notify_turn_complete("user", "")
        detector.notify_turn_complete("robot", "   ")
        assert detector._dialog_turns == []

    def test_both_roles_accepted(self):
        detector, _, _ = _make_detector()
        detector.notify_turn_complete("user", "hi")
        detector.notify_turn_complete("robot", "hello")
        assert detector._dialog_turns == ["hi", "hello"]

    def test_text_is_stripped(self):
        detector, _, _ = _make_detector()
        detector.notify_turn_complete("user", "  hello  ")
        assert detector._dialog_turns == ["hello"]


# ---------------------------------------------------------------------------
# TestReset
# ---------------------------------------------------------------------------


class TestReset:
    def test_clears_per_frame_state(self):
        detector, _, _ = _make_detector()
        detector._prev_asr_text = "something"
        detector._text_stable_since = 42.0
        detector._prepare_fired = True
        detector._silence_frame_count = 10
        detector._current_partial = "partial"

        detector.reset()

        assert detector._prev_asr_text == ""
        assert detector._text_stable_since == 0.0
        assert detector._prepare_fired is False
        assert detector._silence_frame_count == 0
        assert detector._current_partial == ""

    def test_preserves_dialog_context(self):
        detector, _, _ = _make_detector()
        detector._dialog_turns = ["turn1", "turn2"]
        detector.reset()
        assert detector._dialog_turns == ["turn1", "turn2"]

    @patch("voice_pipeline.turn_taking.turn_detector.time")
    def test_process_frame_works_after_reset(self, mock_time):
        mock_time.monotonic.return_value = 1.0
        detector, _, _ = _make_detector(turngpt_prob=0.5)

        detector.process_frame(_FRAME, "first turn")
        mock_time.monotonic.return_value = 2.0
        detector.process_frame(_FRAME, "first turn")
        assert detector._prepare_fired is True

        detector.reset()

        mock_time.monotonic.return_value = 3.0
        detector.process_frame(_FRAME, "second turn")
        mock_time.monotonic.return_value = 4.0
        result = detector.process_frame(_FRAME, "second turn")
        assert result.prepare is True


# ---------------------------------------------------------------------------
# TestPriorityOrder
# ---------------------------------------------------------------------------


class TestPriorityOrder:
    @patch("voice_pipeline.turn_taking.turn_detector.time")
    def test_interrupt_overrides_turn_shift(self, mock_time):
        """When both interrupt and turn_shift conditions hold, interrupt wins."""
        cfg = TurnDetectorConfig(turn_shift_silence_frames=1)
        mock_time.monotonic.return_value = 1.0
        detector, mock_vap, _ = _make_detector(config=cfg, vap_result=_SPEAKING)

        # Set up prior speech so turn_shift would be valid
        detector._prev_asr_text = "hello"
        detector._silence_frame_count = 10

        result = detector.process_frame(_FRAME, "hello", robot_audio=_ROBOT_FRAME)
        assert result.interrupt is True
        assert result.turn_shift is False

    @patch("voice_pipeline.turn_taking.turn_detector.time")
    def test_interrupt_overrides_prepare(self, mock_time):
        mock_time.monotonic.return_value = 1.0
        detector, _, _ = _make_detector(vap_result=_SPEAKING, turngpt_prob=0.9)

        detector.process_frame(_FRAME, "hello")
        mock_time.monotonic.return_value = 2.0
        result = detector.process_frame(_FRAME, "hello", robot_audio=_ROBOT_FRAME)
        assert result.interrupt is True
        assert result.prepare is False

    @patch("voice_pipeline.turn_taking.turn_detector.time")
    def test_prepare_overrides_turn_shift(self, mock_time):
        """When both prepare and turn_shift conditions hold, prepare wins."""
        cfg = TurnDetectorConfig(turn_shift_silence_frames=2)
        mock_time.monotonic.return_value = 1.0
        detector, _, _ = _make_detector(config=cfg, turngpt_prob=0.5)

        # Frame 1: text changes, count 0→1
        detector.process_frame(_FRAME, "hello")
        # Frame 2: count 1→2 (meets turn_shift threshold)
        # Also enough time for prepare to fire
        mock_time.monotonic.return_value = 2.0
        result = detector.process_frame(_FRAME, "hello")
        assert result.prepare is True
        assert result.turn_shift is False


# ---------------------------------------------------------------------------
# TestDialogBuilding
# ---------------------------------------------------------------------------


class TestDialogBuilding:
    def test_completed_turns_joined_with_ts(self):
        detector, _, _ = _make_detector()
        detector._dialog_turns = ["hello", "hi there"]
        assert detector._build_turngpt_dialog() == "hello<ts>hi there"

    def test_partial_appended_without_trailing_ts(self):
        detector, _, _ = _make_detector()
        detector._dialog_turns = ["hello"]
        detector._current_partial = "I want to"
        assert detector._build_turngpt_dialog() == "hello<ts>I want to"

    def test_empty_dialog_with_partial(self):
        detector, _, _ = _make_detector()
        detector._current_partial = "just started"
        assert detector._build_turngpt_dialog() == "just started"

    def test_empty_dialog_empty_partial(self):
        detector, _, _ = _make_detector()
        assert detector._build_turngpt_dialog() == ""

    def test_no_trailing_ts_on_completed_only(self):
        detector, _, _ = _make_detector()
        detector._dialog_turns = ["a", "b", "c"]
        result = detector._build_turngpt_dialog()
        assert result == "a<ts>b<ts>c"
        assert not result.endswith("<ts>")


# ---------------------------------------------------------------------------
# TestErrorResilience
# ---------------------------------------------------------------------------


class TestErrorResilience:
    def test_vap_error_uses_default_result(self):
        detector, mock_vap, _ = _make_detector()
        mock_vap.feed_audio.side_effect = RuntimeError("VAP crashed")

        # Should not raise, should return no-action
        result = detector.process_frame(_FRAME, "hello")
        assert result == TurnDecision.none()

    @patch("voice_pipeline.turn_taking.turn_detector.time")
    def test_turngpt_error_skips_prepare(self, mock_time):
        """TurnGPT error during prepare check → skip prepare, continue."""
        mock_time.monotonic.return_value = 1.0
        detector, _, mock_turngpt = _make_detector(turngpt_prob=0.5)
        mock_turngpt.predict.side_effect = RuntimeError("TurnGPT crashed")

        detector.process_frame(_FRAME, "hello")
        mock_time.monotonic.return_value = 2.0
        result = detector.process_frame(_FRAME, "hello")
        # Prepare should not fire due to error
        assert result.prepare is False
        # Should still be able to reach turn_shift later

    @patch("voice_pipeline.turn_taking.turn_detector.time")
    def test_turngpt_error_still_allows_turn_shift(self, mock_time):
        """After TurnGPT error, turn_shift still works via silence frames."""
        cfg = TurnDetectorConfig(turn_shift_silence_frames=3)
        mock_time.monotonic.return_value = 1.0
        detector, _, mock_turngpt = _make_detector(config=cfg, turngpt_prob=0.5)
        mock_turngpt.predict.side_effect = RuntimeError("TurnGPT crashed")

        # Frame 1: text changes, count 0→1
        detector.process_frame(_FRAME, "hello")
        mock_time.monotonic.return_value = 2.0
        # Frame 2: count 1→2
        detector.process_frame(_FRAME, "hello")
        # Frame 3: count 2→3 >= 3 → turn_shift
        result = detector.process_frame(_FRAME, "hello")
        assert result.turn_shift is True
