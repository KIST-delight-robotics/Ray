"""Unit tests for TurnDetector.

All external dependencies (IVAP, ITurnGPT) are mocked.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np

from voice_pipeline.core.interfaces import IVAP, IEmbedder, ITurnGPT
from voice_pipeline.core.types import TurnDecision, VAPResult
from voice_pipeline.trace.trace_store import InMemoryCallStore
from voice_pipeline.turn_taking.async_turngpt import SyncTurnGPTAdapter
from voice_pipeline.turn_taking.turn_detector import TurnDetector, _TurnState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 30ms frames at 16kHz -> each frame = 0.03s
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
    vap_results: list[VAPResult] | None = None,
    turngpt_prob: float = 0.0,
    embedder: MagicMock | None = None,
) -> tuple[TurnDetector, MagicMock, MagicMock]:
    """Create a TurnDetector with mocked VAP and TurnGPT.

    Returns (detector, mock_vap, mock_turngpt). Tests that need non-default
    thresholds should ``monkeypatch.setattr(TurnDetector, "_...", value)``
    before calling.
    """
    mock_vap = MagicMock(spec=IVAP)
    mock_turngpt = MagicMock(spec=ITurnGPT)

    if vap_results:
        # TurnDetector reads latest_result after each feed_audio; advance the
        # sequence on feed_audio so frame i sees vap_results[i] (last value
        # held once exhausted).
        _seq = iter(vap_results)

        def _advance(*_args, **_kwargs):
            nxt = next(_seq, None)
            if nxt is not None:
                mock_vap.latest_result = nxt

        mock_vap.feed_audio.side_effect = _advance
    else:
        # Default: no speech, neutral probabilities
        mock_vap.latest_result = VAPResult(0.5, 0.5, False)

    mock_turngpt.predict.return_value = turngpt_prob

    if embedder is None:
        embedder = _make_embedder_mock(similarity=0.0)
    adapter = SyncTurnGPTAdapter(mock_turngpt)
    detector = TurnDetector(mock_vap, adapter, embedder)
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
        assert not decision.cancel


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

        shifted = [d for d in decisions if d.turn_shift]
        assert shifted
        assert shifted[0].turn_shift_reason == "vap"

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
        shifted = [d for d in decisions if d.turn_shift]
        assert shifted
        assert shifted[0].turn_shift_reason == "turngpt_0.5"

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
        assert decisions[shift_indices[0]].turn_shift_reason == "turngpt_3.0"


# ---------------------------------------------------------------------------
# Test 6: turn_shift enters PENDING (tentative), not ROBOT_TURN
# ---------------------------------------------------------------------------


class TestTurnShiftEntersPending:
    def test_turn_shift_enters_pending_preserving_state(self):
        """After turn_shift the detector is PENDING and per-frame state is
        preserved (so a cancel can resume the same turn)."""
        n_shift = 20
        detector, _, _ = _make_detector(vap_results=_silent_robot_favoring(n_shift))
        detector.process_frame(FRAME, "hello")
        for _ in range(n_shift - 1):
            if detector.process_frame(FRAME, "hello").turn_shift:
                break
        assert detector._turn_state is _TurnState.PENDING
        assert detector._last_prepare_text == "hello"  # prepare baseline preserved for cancel
        assert detector._prev_asr_text == "hello"  # NOT wiped at turn_shift


# ---------------------------------------------------------------------------
# Test 6a2: turn_shift defers to prepare on a pending dissimilar change
# ---------------------------------------------------------------------------


class TestTurnShiftPrepareDefer:
    def test_defers_to_prepare_on_pending_change(self):
        """A meaningful unprepared ASR change at turn_shift -> prepare, not shift."""
        detector, mock_vap, _ = _make_detector(embedder=_make_embedder_mock(similarity=0.3))
        mock_vap.latest_result = VAPResult(0.2, 0.2, False)  # robot-favor, silent
        detector._silence_elapsed_sec = 3.0  # turn_shift_ready via low-prob timeout
        detector._prev_asr_text = "completely different text"  # avoid resetting timers this frame
        detector._last_prepare_text = "old text"
        detector._asr_has_changed = True
        detector._last_asr_change_elapsed_sec = 0.5  # prepare condition met

        decision = detector.process_frame(FRAME, "completely different text")
        assert decision.prepare
        assert detector._turn_state is _TurnState.USER_TURN  # did NOT shift

    def test_shifts_when_no_pending_change(self):
        """Text settled (no pending change) at turn_shift -> turn_shift fires."""
        detector, mock_vap, _ = _make_detector()
        mock_vap.latest_result = VAPResult(0.2, 0.2, False)
        detector._silence_elapsed_sec = 3.0
        detector._prev_asr_text = "hello"
        detector._last_prepare_text = "hello"
        detector._asr_has_changed = False  # nothing pending → prepare won't fire

        decision = detector.process_frame(FRAME, "hello")
        assert decision.turn_shift
        assert detector._turn_state is _TurnState.PENDING

    def test_fresh_dissimilar_change_preempts_shift(self):
        """발화 조건(turngpt/0.2s 스로틀) 미충족인 신규 변화도 turn_shift 직전에는 prepare가 선점."""
        detector, mock_vap, _ = _make_detector(embedder=_make_embedder_mock(similarity=0.3))
        mock_vap.latest_result = VAPResult(0.2, 0.2, False)
        detector._silence_elapsed_sec = 3.0  # turn_shift 조건 충족 (turngpt_3.0)
        detector._prev_asr_text = "is there"
        detector._last_prepare_text = "is"
        detector._asr_has_changed = True
        detector._last_asr_change_elapsed_sec = 0.03  # 스로틀(0.2s) 미충족 — 우회 대상

        decision = detector.process_frame(FRAME, "is there")
        assert decision.prepare
        assert not decision.turn_shift
        assert detector._turn_state is _TurnState.USER_TURN

    def test_fresh_similar_change_records_skip_and_shifts(self):
        """유사한 신규 변화는 skip을 기록하고 같은 프레임에서 shift가 진행된다."""
        store = InMemoryCallStore()
        mock_vap = MagicMock(spec=IVAP)
        mock_vap.latest_result = VAPResult(0.2, 0.2, False)
        mock_turngpt = MagicMock(spec=ITurnGPT)
        mock_turngpt.predict.return_value = 0.0
        detector = TurnDetector(
            mock_vap,
            SyncTurnGPTAdapter(mock_turngpt),
            _make_embedder_mock(similarity=0.95),
            call_store=store,
            session_id="s",
        )
        detector._silence_elapsed_sec = 3.0
        detector._prev_asr_text = "is there."
        detector._last_prepare_text = "is there"
        detector._asr_has_changed = True
        detector._last_asr_change_elapsed_sec = 0.03

        decision = detector.process_frame(FRAME, "is there.")
        assert decision.turn_shift
        assert detector._turn_state is _TurnState.PENDING

        recs = [r for r in store.records if r.operation == "prepare_gate"]
        assert len(recs) == 1
        assert json.loads(recs[0].metadata)["decision"] == "skip"


# ---------------------------------------------------------------------------
# Test 6b: PENDING cancel detection
# ---------------------------------------------------------------------------


class TestPendingCancel:
    def test_vap_user_favor_cancels_and_rewinds(self):
        """In PENDING, p_now/p_fut favoring the user -> cancel + rewind."""
        detector, mock_vap, _ = _make_detector()
        detector._turn_state = _TurnState.PENDING
        detector._last_prepare_text = "hello"
        mock_vap.latest_result = VAPResult(0.8, 0.8, True)

        decision = detector.process_frame(FRAME, "hello")
        assert decision.cancel
        assert detector._turn_state is _TurnState.USER_TURN

    def test_dissimilar_asr_cancels(self):
        """In PENDING, dissimilar new ASR text -> cancel (user continued)."""
        detector, mock_vap, _ = _make_detector(embedder=_make_embedder_mock(similarity=0.3))
        detector._turn_state = _TurnState.PENDING
        detector._last_prepare_text = "hello"
        # speaking, but VAP probs don't cross the user-favor threshold → ASR path
        mock_vap.latest_result = VAPResult(0.2, 0.2, True)

        decision = detector.process_frame(FRAME, "hello tell me more about it")
        assert decision.cancel
        assert detector._turn_state is _TurnState.USER_TURN

    def test_similar_asr_no_cancel(self):
        """In PENDING, a finalization (high similarity) does NOT cancel."""
        detector, mock_vap, _ = _make_detector(embedder=_make_embedder_mock(similarity=0.95))
        detector._turn_state = _TurnState.PENDING
        detector._last_prepare_text = "hello"
        mock_vap.latest_result = VAPResult(0.2, 0.2, True)

        decision = detector.process_frame(FRAME, "hello.")
        assert not decision.cancel
        assert detector._turn_state is _TurnState.PENDING

    def test_no_signal_stays_pending(self):
        """In PENDING with robot-favor VAP and unchanged text -> no cancel."""
        detector, mock_vap, _ = _make_detector()
        detector._turn_state = _TurnState.PENDING
        detector._last_prepare_text = "hello"
        detector._prev_asr_text = "hello"
        mock_vap.latest_result = VAPResult(0.2, 0.2, False)

        decision = detector.process_frame(FRAME, "hello")
        assert decision == TurnDecision.none()
        assert detector._turn_state is _TurnState.PENDING

    def test_not_speaking_no_cancel_even_user_favor(self):
        """Gate: user-favor VAP but user NOT speaking -> no cancel (thrash guard)."""
        detector, mock_vap, _ = _make_detector()
        detector._turn_state = _TurnState.PENDING
        detector._last_prepare_text = "hello"
        detector._prev_asr_text = "hello"
        mock_vap.latest_result = VAPResult(0.8, 0.8, False)

        decision = detector.process_frame(FRAME, "hello")
        assert not decision.cancel
        assert detector._turn_state is _TurnState.PENDING


# ---------------------------------------------------------------------------
# Test 6c: commit() enters ROBOT_TURN
# ---------------------------------------------------------------------------


class TestCommit:
    def test_commit_enters_robot_turn_and_records_dialog(self):
        detector, _, _ = _make_detector()
        detector._turn_state = _TurnState.PENDING
        detector._prev_asr_text = "hello"
        detector._last_prepare_text = "hello"

        detector.commit("hello")

        assert detector._turn_state is _TurnState.ROBOT_TURN
        assert detector._dialog_parts == ["hello"]
        assert detector._prev_asr_text == ""  # wiped
        assert detector._last_prepare_text == ""  # wiped


class TestExchangeIndex:
    """The shared call-store turn counter must track the exchange index so that
    user-turn (vap/turngpt) and robot-turn (llm/tts) call records of the same
    exchange share one index, never colliding with the next exchange."""

    def _detector(self, store: InMemoryCallStore) -> TurnDetector:
        mock_vap = MagicMock(spec=IVAP)
        adapter = SyncTurnGPTAdapter(MagicMock(spec=ITurnGPT))
        return TurnDetector(mock_vap, adapter, _make_embedder_mock(), call_store=store)

    def test_counter_tracks_exchange_across_turns(self):
        store = InMemoryCallStore()
        detector = self._detector(store)

        # Exchange 0: user turn and its generation share index 0.
        assert store.current_turn_index == 0  # seeded at construction
        detector.commit("first")  # user→robot; generation of exchange 0
        assert store.current_turn_index == 0

        detector.reset()  # entering user turn of exchange 1
        assert store.current_turn_index == 1
        detector.commit("second")  # generation of exchange 1
        assert store.current_turn_index == 1

        detector.reset()  # exchange 2
        assert store.current_turn_index == 2


# ---------------------------------------------------------------------------
# Test 7: Interrupt with robot_audio
# ---------------------------------------------------------------------------


class TestInterruptWithRobotAudio:
    def test_both_favor_user_triggers_interrupt(self):
        """Both p_now and p_fut favor user + speaking -> interrupt."""
        detector, mock_vap, _ = _make_detector()
        detector._turn_state = _TurnState.ROBOT_TURN
        mock_vap.latest_result = VAPResult(0.8, 0.8, True)

        decision = detector.process_frame(FRAME, "", ROBOT_FRAME)
        assert decision.interrupt

    def test_not_speaking_no_interrupt(self):
        """user_is_speaking=False -> no interrupt even if p_now/p_fut favor user."""
        detector, mock_vap, _ = _make_detector()
        detector._turn_state = _TurnState.ROBOT_TURN
        mock_vap.latest_result = VAPResult(0.8, 0.8, False)

        decision = detector.process_frame(FRAME, "", ROBOT_FRAME)
        assert decision == TurnDecision.none()

    def test_backchannel_no_interrupt(self):
        """p_now favors user, p_fut favors robot -> backchannel, no interrupt."""
        detector, mock_vap, _ = _make_detector()
        detector._turn_state = _TurnState.ROBOT_TURN
        mock_vap.latest_result = VAPResult(0.8, 0.2, True)

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
        mock_vap.latest_result = VAPResult(0.3, 0.3, True)

        decision = detector.process_frame(FRAME, "", None)
        assert decision == TurnDecision.none()

    def test_no_user_speech_no_interrupt(self):
        """In ROBOT_TURN without robot_audio, no speech -> no interrupt."""
        detector, mock_vap, _ = _make_detector()
        detector._turn_state = _TurnState.ROBOT_TURN
        mock_vap.latest_result = VAPResult(0.3, 0.3, False)

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
# Test 12a: VAD reset on commit
# ---------------------------------------------------------------------------


class TestVADResetOnCommit:
    def _make_detector_with_reset(self, reset_fn) -> TurnDetector:
        mock_vap = MagicMock(spec=IVAP)
        mock_vap.latest_result = VAPResult(0.5, 0.5, False)
        mock_turngpt = MagicMock(spec=ITurnGPT)
        mock_turngpt.predict.return_value = 0.0
        return TurnDetector(
            mock_vap,
            SyncTurnGPTAdapter(mock_turngpt),
            _make_embedder_mock(),
            vad_reset_fn=reset_fn,
        )

    def test_commit_calls_vad_reset(self):
        reset_fn = MagicMock()
        detector = self._make_detector_with_reset(reset_fn)
        detector._turn_state = _TurnState.PENDING

        detector.commit("hello")
        reset_fn.assert_called_once()

    def test_reset_does_not_call_vad_reset(self):
        """reset() (new turn, playback just ended) must NOT reset VAD —
        the user may already be speaking at that point."""
        reset_fn = MagicMock()
        detector = self._make_detector_with_reset(reset_fn)

        detector.reset()
        reset_fn.assert_not_called()

    def test_vad_reset_failure_does_not_break_commit(self):
        reset_fn = MagicMock(side_effect=RuntimeError("boom"))
        detector = self._make_detector_with_reset(reset_fn)
        detector._turn_state = _TurnState.PENDING

        detector.commit("hello")
        assert detector._turn_state is _TurnState.ROBOT_TURN

    def test_no_vad_reset_fn_commit_ok(self):
        detector, _, _ = _make_detector()
        detector._turn_state = _TurnState.PENDING
        detector.commit("hello")
        assert detector._turn_state is _TurnState.ROBOT_TURN


# ---------------------------------------------------------------------------
# Test 12b: similarity gate call recording
# ---------------------------------------------------------------------------


class TestSimilarityGateRecording:
    def _make_recording_detector(
        self,
        similarity: float,
        vap_result: VAPResult,
        turngpt_prob: float = 0.5,
    ) -> tuple[TurnDetector, InMemoryCallStore]:
        store = InMemoryCallStore()
        mock_vap = MagicMock(spec=IVAP)
        mock_vap.latest_result = vap_result
        mock_turngpt = MagicMock(spec=ITurnGPT)
        mock_turngpt.predict.return_value = turngpt_prob
        detector = TurnDetector(
            mock_vap,
            SyncTurnGPTAdapter(mock_turngpt),
            _make_embedder_mock(similarity=similarity),
            call_store=store,
            session_id="sess-1",
        )
        return detector, store

    def test_prepare_skip_recorded(self):
        """Skipped prepare -> prepare_gate record with decision=skip and metadata."""
        detector, store = self._make_recording_detector(0.9, VAPResult(0.5, 0.5, True))

        detector.process_frame(FRAME, "hello world")
        d1 = detector.process_frame(FRAME, "hello world")
        assert d1.prepare
        # First prepare computes no similarity — nothing recorded yet
        assert store.records == []

        detector.process_frame(FRAME, "hello worlds")
        d2 = detector.process_frame(FRAME, "hello worlds")
        assert not d2.prepare

        assert len(store.records) == 1
        rec = store.records[0]
        assert rec.module == "similarity_gate"
        assert rec.operation == "prepare_gate"
        assert rec.session_id == "sess-1"
        meta = json.loads(rec.metadata)
        assert meta["decision"] == "skip"
        assert abs(meta["similarity"] - 0.9) < 0.01
        assert meta["threshold"] == TurnDetector._SIMILARITY_THRESHOLD
        assert rec.turn_index == 0  # turn_index lives on the column now, not metadata
        assert meta["prev_text"] == "hello world"
        assert meta["new_text"] == "hello worlds"

    def test_prepare_regenerate_recorded(self):
        """Dissimilar prepare -> prepare_gate record with decision=regenerate."""
        detector, store = self._make_recording_detector(0.3, VAPResult(0.5, 0.5, True))

        detector.process_frame(FRAME, "hello world")
        d1 = detector.process_frame(FRAME, "hello world")
        assert d1.prepare
        d2 = detector.process_frame(FRAME, "completely different sentence")
        assert d2.prepare

        recs = [r for r in store.records if r.operation == "prepare_gate"]
        assert len(recs) == 1
        meta = json.loads(recs[0].metadata)
        assert meta["decision"] == "regenerate"

    def test_pending_cancel_recorded(self):
        """PENDING dissimilar text -> cancel_gate record with decision=cancel."""
        detector, store = self._make_recording_detector(0.3, VAPResult(0.2, 0.2, True))
        detector._turn_state = _TurnState.PENDING
        detector._last_prepare_text = "hello"

        decision = detector.process_frame(FRAME, "hello tell me more")
        assert decision.cancel

        recs = [r for r in store.records if r.operation == "cancel_gate"]
        assert len(recs) == 1
        meta = json.loads(recs[0].metadata)
        assert meta["decision"] == "cancel"
        assert meta["prev_text"] == "hello"

    def test_pending_keep_recorded(self):
        """PENDING similar finalization -> cancel_gate record with decision=keep."""
        detector, store = self._make_recording_detector(0.95, VAPResult(0.2, 0.2, True))
        detector._turn_state = _TurnState.PENDING
        detector._last_prepare_text = "hello"

        decision = detector.process_frame(FRAME, "hello.")
        assert not decision.cancel

        recs = [r for r in store.records if r.operation == "cancel_gate"]
        assert len(recs) == 1
        meta = json.loads(recs[0].metadata)
        assert meta["decision"] == "keep"

    def test_turn_index_increments_on_commit(self):
        """Records after commit() carry the next turn_index."""
        detector, store = self._make_recording_detector(0.95, VAPResult(0.2, 0.2, True))
        detector._turn_state = _TurnState.PENDING
        detector._last_prepare_text = "hello"
        detector.process_frame(FRAME, "hello.")

        detector.commit("hello.")
        detector.reset()

        detector._turn_state = _TurnState.PENDING
        detector._last_prepare_text = "next turn"
        detector.process_frame(FRAME, "next turn text")

        recs = [r for r in store.records if r.operation == "cancel_gate"]
        assert [r.turn_index for r in recs] == [0, 1]

    def test_no_call_store_no_records(self):
        """Without a call_store the gate works and records nothing."""
        detector, _, _ = _make_detector(
            vap_results=[VAPResult(0.5, 0.5, True)] * 4,
            turngpt_prob=0.5,
            embedder=_make_embedder_mock(similarity=0.9),
        )
        detector.process_frame(FRAME, "hello world")
        assert detector.process_frame(FRAME, "hello world").prepare
        detector.process_frame(FRAME, "hello worlds")
        assert not detector.process_frame(FRAME, "hello worlds").prepare


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
