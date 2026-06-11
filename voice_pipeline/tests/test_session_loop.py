"""Tests for voice_pipeline.session_loop."""

from __future__ import annotations

import queue
import time
from unittest.mock import MagicMock, patch

import pytest

from voice_pipeline.core.types import (
    AudioFrame,
    CppEvent,
    CppEventType,
    GeneratorState,
    LEDState,
    Phase,
    PipelineTrace,
    ResponseData,
    TurnDecision,
    WordTimestamp,
)
from voice_pipeline.session_loop import SessionLoop, _PendingTruncation
from voice_pipeline.trace.trace_store import InMemoryTraceStore
from voice_pipeline.tts.openai_tts import OpenAITTS

# ---------------------------------------------------------------------------
# Fixture helper
# ---------------------------------------------------------------------------


def _make_session_loop(
    monkeypatch: pytest.MonkeyPatch,
    *,
    exit_keywords: tuple[str, ...] = ("bye", "goodbye"),
    session_timeout_sec: float = 30.0,
    frame_timeout_sec: float = 0.01,
    stop_pending_timeout_sec: float = 5.0,
    output_sample_rate: int = OpenAITTS.OUTPUT_SAMPLE_RATE,
    audio_queue: queue.Queue[AudioFrame] | None = None,
) -> tuple[SessionLoop, dict[str, MagicMock]]:
    """Create an SessionLoop with all dependencies mocked.

    Class var overrides (timeouts, exit keywords) are applied via
    ``monkeypatch`` so they auto-revert at test teardown.
    ``audio_queue`` defaults to an empty queue.
    """
    monkeypatch.setattr(SessionLoop, "_EXIT_KEYWORDS", exit_keywords)
    monkeypatch.setattr(SessionLoop, "_SESSION_TIMEOUT_SEC", session_timeout_sec)
    monkeypatch.setattr(SessionLoop, "_FRAME_TIMEOUT_SEC", frame_timeout_sec)
    monkeypatch.setattr(SessionLoop, "_STOP_PENDING_TIMEOUT_SEC", stop_pending_timeout_sec)

    mocks = {
        "asr": MagicMock(),
        "turn_detector": MagicMock(),
        "generator": MagicMock(),
        "bridge": MagicMock(),
        "history": MagicMock(),
        "led": MagicMock(),
    }

    # Defaults
    mocks["asr"].get_text.return_value = ""
    mocks["turn_detector"].process_frame.return_value = TurnDecision.none()
    mocks["generator"].state = GeneratorState.IDLE
    mocks["generator"].stream_done = False
    mocks["generator"].poll_audio.return_value = None
    mocks["bridge"].poll_event.return_value = None
    mocks["history"].add_user_message.return_value = 0
    mocks["history"].add_assistant_message.return_value = 1

    orch = SessionLoop(
        asr=mocks["asr"],
        turn_detector=mocks["turn_detector"],
        speech_generator=mocks["generator"],
        cpp_bridge=mocks["bridge"],
        history=mocks["history"],
        led=mocks["led"],
        audio_queue=audio_queue if audio_queue is not None else queue.Queue(),
        tts_sample_rate=output_sample_rate,
    )
    return orch, mocks


def _frame(size: int = 960) -> AudioFrame:
    """Create a dummy audio frame."""
    return b"\x00" * size


def _audio_queue_with(*frames: AudioFrame) -> queue.Queue[AudioFrame]:
    q: queue.Queue[AudioFrame] = queue.Queue()
    for f in frames:
        q.put(f)
    return q


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_start_stop_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """start_session calls asr.start and LED IDLE; end calls reset/OFF."""
        orch, mocks = _make_session_loop(monkeypatch, session_timeout_sec=0.0)
        # With 0 timeout, session ends immediately
        orch.run()
        mocks["asr"].start.assert_called_once()
        mocks["asr"].stop.assert_called_once()
        mocks["generator"].reset.assert_called_once()

        # LED sequence: IDLE (start) → OFF (end)
        led_calls = [c.args[0] for c in mocks["led"].set_state.call_args_list]
        assert led_calls[0] == LEDState.IDLE
        assert led_calls[-1] == LEDState.OFF

    def test_start_session_resets_internal_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """run() resets all internal state from a previous session."""
        orch, mocks = _make_session_loop(monkeypatch, session_timeout_sec=0.0)
        # Simulate dirty state from a previous session
        orch._phase = Phase.PLAYING
        orch._sent_audio_buffer = bytearray(b"\xff" * 100)

        orch.run()

        # After run(), state should have been reset at start
        # (end_session also resets some, but start must handle it)
        assert orch._phase is Phase.LISTENING


# ---------------------------------------------------------------------------
# Turn shift tests
# ---------------------------------------------------------------------------


class TestTurnShift:
    def test_turn_shift_streaming_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When generator is STREAMING on turn_shift, history records input_text."""
        orch, mocks = _make_session_loop(monkeypatch)
        mocks["asr"].get_text.return_value = "hello world"
        mocks["turn_detector"].process_frame.return_value = TurnDecision(turn_shift=True)
        mocks["generator"].state = GeneratorState.STREAMING
        mocks["generator"].poll_audio.return_value = None
        mocks["generator"].stream_done = False
        # Simulate earlier prepare with partial text
        mocks["generator"].input_text = "hello"

        q = _audio_queue_with(_frame())
        orch._start_session()
        orch._audio_queue = q
        orch._run_frame()

        # History records generator.input_text, not current ASR text
        mocks["history"].add_user_message.assert_called_once_with("hello")
        mocks["asr"].reset.assert_called_once()
        assert orch._phase is Phase.STREAMING
        mocks["bridge"].send_stream_start.assert_called_once()

    def test_turn_shift_not_ready_sets_awaiting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When generator not ready on turn_shift, set awaiting."""
        orch, mocks = _make_session_loop(monkeypatch)
        mocks["asr"].get_text.return_value = "hello"
        mocks["turn_detector"].process_frame.return_value = TurnDecision(turn_shift=True)
        mocks["generator"].state = GeneratorState.IDLE

        q = _audio_queue_with(_frame())
        orch._start_session()
        orch._audio_queue = q
        orch._run_frame()

        assert orch._phase is Phase.AWAITING
        mocks["generator"].prepare.assert_called_once_with("hello")

    def test_turn_shift_preparing_sets_awaiting_no_double_prepare(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When generator is PREPARING, set awaiting without calling prepare again."""
        orch, mocks = _make_session_loop(monkeypatch)
        mocks["asr"].get_text.return_value = "hello"
        mocks["turn_detector"].process_frame.return_value = TurnDecision(turn_shift=True)
        mocks["generator"].state = GeneratorState.PREPARING

        q = _audio_queue_with(_frame())
        orch._start_session()
        orch._audio_queue = q
        orch._run_frame()

        assert orch._phase is Phase.AWAITING
        mocks["generator"].prepare.assert_not_called()


# ---------------------------------------------------------------------------
# Awaiting response tests
# ---------------------------------------------------------------------------


class TestAwaitingResponse:
    def test_awaiting_completion_begins_streaming(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When awaiting and generator becomes STREAMING, history records input_text."""
        orch, mocks = _make_session_loop(monkeypatch)
        orch._start_session()
        orch._phase = Phase.AWAITING
        mocks["generator"].state = GeneratorState.STREAMING
        mocks["generator"].input_text = "hello"
        mocks["generator"].poll_audio.return_value = None
        mocks["generator"].stream_done = False

        q = _audio_queue_with()
        orch._audio_queue = q
        orch._run_frame()

        assert orch._phase is Phase.STREAMING
        # Uses generator.input_text — matches what LLM actually saw
        mocks["history"].add_user_message.assert_called_once_with("hello")

    def test_begin_streaming_uses_generator_input_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """History records generator.input_text (what LLM actually saw)."""
        orch, mocks = _make_session_loop(monkeypatch)
        orch._start_session()
        mocks["generator"].input_text = "earlier partial"
        mocks["generator"].state = GeneratorState.STREAMING
        mocks["generator"].poll_audio.return_value = None
        mocks["generator"].stream_done = False

        orch._begin_streaming()

        mocks["history"].add_user_message.assert_called_once_with("earlier partial")

    def test_cancel_during_awaiting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cancel decision during AWAITING cancels generation → LISTENING."""
        orch, mocks = _make_session_loop(monkeypatch)
        orch._start_session()
        orch._phase = Phase.AWAITING
        mocks["asr"].get_text.return_value = ""
        mocks["turn_detector"].process_frame.return_value = TurnDecision(cancel=True)

        q = _audio_queue_with(_frame())
        orch._audio_queue = q
        orch._run_frame()

        mocks["generator"].cancel.assert_called_once()
        # Detector self-rewinds on cancel; SessionLoop must NOT reset it.
        mocks["turn_detector"].reset.assert_not_called()
        assert orch._phase is Phase.LISTENING

    def test_cancel_invokes_on_cancel_callback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """on_cancel callback fires when a tentative turn_shift is retracted."""
        cancels = []
        orch, mocks = _make_session_loop(monkeypatch)
        orch._on_cancel_cb = lambda: cancels.append(1)
        orch._start_session()
        orch._phase = Phase.AWAITING
        mocks["turn_detector"].process_frame.return_value = TurnDecision(cancel=True)

        orch._audio_queue = _audio_queue_with(_frame())
        orch._run_frame()

        assert cancels == [1]

    def test_on_cancel_callback_error_suppressed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """on_cancel callback exceptions must not break cancel handling."""
        orch, mocks = _make_session_loop(monkeypatch)
        orch._on_cancel_cb = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        orch._start_session()
        orch._phase = Phase.AWAITING
        mocks["turn_detector"].process_frame.return_value = TurnDecision(cancel=True)

        orch._audio_queue = _audio_queue_with(_frame())
        orch._run_frame()

        assert orch._phase is Phase.LISTENING

    def test_awaiting_generator_failed_skips_turn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Generator FAILED during awaiting skips turn and resets turn_detector."""
        orch, mocks = _make_session_loop(monkeypatch)
        orch._start_session()
        orch._phase = Phase.AWAITING
        mocks["generator"].state = GeneratorState.FAILED

        q = _audio_queue_with()
        orch._audio_queue = q
        orch._run_frame()

        assert orch._phase is Phase.LISTENING
        mocks["turn_detector"].reset.assert_called_once()


# ---------------------------------------------------------------------------
# Prepare tests
# ---------------------------------------------------------------------------


class TestPrepare:
    def test_prepare_cancels_and_restarts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Prepare signal triggers generator.prepare with current text."""
        orch, mocks = _make_session_loop(monkeypatch)
        mocks["asr"].get_text.return_value = "how are you"
        mocks["turn_detector"].process_frame.return_value = TurnDecision(prepare=True)

        q = _audio_queue_with(_frame())
        orch._start_session()
        orch._audio_queue = q
        orch._run_frame()

        mocks["generator"].prepare.assert_called_once_with("how are you")


# ---------------------------------------------------------------------------
# Interrupt tests
# ---------------------------------------------------------------------------


class TestInterrupt:
    def test_interrupt_during_playing_sends_stop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Interrupt during PLAYING sends stop and enters STOP_PENDING."""
        orch, mocks = _make_session_loop(monkeypatch)
        orch._start_session()
        orch._phase = Phase.PLAYING
        mocks["asr"].get_text.return_value = "wait"
        mocks["turn_detector"].process_frame.return_value = TurnDecision(interrupt=True)

        q = _audio_queue_with(_frame())
        orch._audio_queue = q
        orch._run_frame()

        mocks["bridge"].send_stop.assert_called_once()
        assert orch._phase == Phase.STOPPING


# ---------------------------------------------------------------------------
# Barge-in truncation tests
# ---------------------------------------------------------------------------


class TestBargeIn:
    def test_case_a_timestamps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Case A: ResponseData with timestamps → truncate_by_timestamps."""
        orch, mocks = _make_session_loop(monkeypatch)
        orch._start_session()
        orch._phase = Phase.STOPPING
        orch._playback_start_time = time.monotonic() - 1.0
        orch._stop_pending_time = time.monotonic() - 0.65

        timestamps = [
            WordTimestamp("hello", 0.0, 0.3),
            WordTimestamp("world", 0.4, 0.7),
        ]
        orch._current_response = ResponseData(text="hello world", audio=b"\x00" * 100, timestamps=timestamps)

        with patch("voice_pipeline.session_loop.truncate_by_timestamps", return_value="hello") as mock_trunc:
            orch._on_playback_interrupted()

            mock_trunc.assert_called_once()
            call_args = mock_trunc.call_args[0]
            assert call_args[0] == "hello world"
            assert 0.2 < call_args[1] < 0.5

        mocks["history"].add_assistant_message.assert_called_once()
        assert mocks["history"].add_assistant_message.call_args[0][0] == "hello"

    def test_case_b_no_timestamps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Case B: ResponseData without timestamps → truncate_by_ratio."""
        orch, mocks = _make_session_loop(monkeypatch, output_sample_rate=24000)
        orch._start_session()
        orch._phase = Phase.STOPPING
        orch._playback_start_time = time.monotonic() - 1.0
        orch._stop_pending_time = time.monotonic() - 0.75

        audio = b"\x00" * 48000
        orch._current_response = ResponseData(text="hello world foo bar", audio=audio, timestamps=[])

        with patch("voice_pipeline.session_loop.truncate_by_ratio", return_value="hello") as mock_trunc:
            orch._on_playback_interrupted()
            mock_trunc.assert_called_once()

        mocks["history"].add_assistant_message.assert_called_once()
        assert mocks["history"].add_assistant_message.call_args[0][0] == "hello"

    def test_case_c_no_response_data_deferred(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Case C: no ResponseData → approximate truncation + deferred."""
        orch, mocks = _make_session_loop(monkeypatch, output_sample_rate=24000)
        orch._start_session()
        orch._phase = Phase.STOPPING
        orch._current_response = None
        orch._playback_start_time = time.monotonic() - 1.0
        orch._stop_pending_time = time.monotonic() - 0.5
        orch._sent_audio_buffer = bytearray(b"\x00" * 48000)

        mocks["generator"].get_text.return_value = "hello world"
        mocks["generator"].stream_done = False
        mocks["history"].add_assistant_message.return_value = 42

        with patch("voice_pipeline.session_loop.truncate_by_ratio", return_value="hello"):
            orch._on_playback_interrupted()

        assert orch._pending_truncation is not None
        assert orch._pending_truncation.msg_id == 42

    def test_no_playback_start_uses_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When playback_started was never received, stop_pos defaults to 0."""
        orch, mocks = _make_session_loop(monkeypatch)
        orch._start_session()
        orch._phase = Phase.STOPPING
        orch._playback_start_time = 0.0

        timestamps = [WordTimestamp("a", 0.0, 0.5)]
        orch._current_response = ResponseData(text="a", audio=b"\x00", timestamps=timestamps)

        with patch("voice_pipeline.session_loop.truncate_by_timestamps", return_value="a") as mock_trunc:
            orch._on_playback_interrupted()
            mock_trunc.assert_called_once_with("a", 0.0, timestamps)


# ---------------------------------------------------------------------------
# Deferred truncation tests
# ---------------------------------------------------------------------------


class TestDeferredTruncation:
    def test_stream_done_with_timestamps_updates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When stream finishes with timestamps, update_message with precise text."""
        orch, mocks = _make_session_loop(monkeypatch)
        orch._start_session()

        timestamps = [
            WordTimestamp("hi", 0.0, 0.3),
            WordTimestamp("there", 0.4, 0.7),
        ]
        response_data = ResponseData(text="hi there", audio=b"\x00" * 100, timestamps=timestamps)
        mocks["generator"].stream_done = True
        mocks["generator"].get_response_data.return_value = response_data
        orch._pending_truncation = _PendingTruncation(msg_id=5, stop_position_sec=0.35)

        with patch("voice_pipeline.session_loop.truncate_by_timestamps", return_value="hi") as mock_trunc:
            orch._check_deferred_truncation()
            mock_trunc.assert_called_once_with("hi there", 0.35, timestamps)

        mocks["history"].update_message.assert_called_once_with(5, "hi")
        assert orch._pending_truncation is None

    def test_stream_done_no_timestamps_uses_ratio(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When stream finishes without timestamps, use truncate_by_ratio."""
        orch, mocks = _make_session_loop(monkeypatch, output_sample_rate=24000)
        orch._start_session()

        audio = b"\x00" * 48000  # 1.0 sec
        response_data = ResponseData(text="hello world", audio=audio, timestamps=[])
        mocks["generator"].stream_done = True
        mocks["generator"].get_response_data.return_value = response_data

        orch._pending_truncation = _PendingTruncation(msg_id=5, stop_position_sec=0.5)

        with patch("voice_pipeline.session_loop.truncate_by_ratio", return_value="hello"):
            orch._check_deferred_truncation()

        mocks["history"].update_message.assert_called_once_with(5, "hello")
        assert orch._pending_truncation is None

    def test_generator_failed_clears_pending(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Generator FAILED clears pending truncation without update."""
        orch, mocks = _make_session_loop(monkeypatch)
        orch._start_session()
        mocks["generator"].state = GeneratorState.FAILED

        orch._pending_truncation = _PendingTruncation(msg_id=5, stop_position_sec=0.5)
        orch._check_deferred_truncation()

        mocks["history"].update_message.assert_not_called()
        assert orch._pending_truncation is None


# ---------------------------------------------------------------------------
# Robot audio chunk tests
# ---------------------------------------------------------------------------


class TestRobotAudioChunk:
    def test_correct_chunk_from_buffer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_robot_audio_chunk extracts 30ms at playback position."""
        orch, _ = _make_session_loop(monkeypatch, output_sample_rate=24000)
        orch._phase = Phase.PLAYING
        orch._playback_start_time = time.monotonic()

        # 30ms @ 24kHz 16-bit = 24000 * 0.03 * 2 = 1440 bytes
        orch._sent_audio_buffer = bytearray(b"\x01" * 2880)

        chunk = orch.get_robot_audio_chunk()
        assert chunk is not None
        assert len(chunk) == 1440

    def test_not_playing_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns None when not PLAYING."""
        orch, _ = _make_session_loop(monkeypatch)
        orch._phase = Phase.LISTENING
        assert orch.get_robot_audio_chunk() is None

    def test_no_playback_start_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns None if playback_started event was never received."""
        orch, _ = _make_session_loop(monkeypatch, output_sample_rate=24000)
        orch._phase = Phase.PLAYING
        orch._playback_start_time = 0.0
        orch._sent_audio_buffer = bytearray(b"\x00" * 2880)

        assert orch.get_robot_audio_chunk() is None

    def test_insufficient_buffer_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns None if buffer doesn't have enough data."""
        orch, _ = _make_session_loop(monkeypatch, output_sample_rate=24000)
        orch._phase = Phase.PLAYING
        orch._playback_start_time = time.monotonic()
        orch._sent_audio_buffer = bytearray(b"\x00" * 10)

        assert orch.get_robot_audio_chunk() is None


# ---------------------------------------------------------------------------
# Robot audio combined tests
# ---------------------------------------------------------------------------


class TestRobotAudioCombined:
    """Tests for _get_robot_audio_combined (batch robot audio extraction)."""

    def test_combined_extracts_n_frames(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Combined extraction returns frame_count * frame_bytes of audio."""
        orch, _ = _make_session_loop(monkeypatch, output_sample_rate=24000)
        orch._phase = Phase.PLAYING
        # 30ms @ 24kHz 16-bit = 1440 bytes per frame
        frame_bytes = 1440
        frame_count = 3
        total_bytes = frame_bytes * (frame_count + 2)  # extra buffer
        orch._sent_audio_buffer = bytearray(b"\xab" * total_bytes)
        # Set playback start so elapsed covers frame_count frames
        # elapsed ≈ frame_count * 30ms → current_start at frame_count * frame_bytes
        elapsed_sec = frame_count * 0.030
        orch._playback_start_time = time.monotonic() - elapsed_sec

        result = orch._get_robot_audio_combined(frame_count)
        assert result is not None
        assert len(result) == frame_bytes * frame_count

    def test_not_playing_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns None when not in PLAYING state."""
        orch, _ = _make_session_loop(monkeypatch)
        orch._phase = Phase.LISTENING
        assert orch._get_robot_audio_combined(3) is None

    def test_no_playback_start_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns None if playback_started event was never received."""
        orch, _ = _make_session_loop(monkeypatch, output_sample_rate=24000)
        orch._phase = Phase.PLAYING
        orch._playback_start_time = 0.0
        orch._sent_audio_buffer = bytearray(b"\x00" * 10000)
        assert orch._get_robot_audio_combined(3) is None

    def test_batch_start_negative_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns None when playback just started and batch can't cover full range."""
        orch, _ = _make_session_loop(monkeypatch, output_sample_rate=24000)
        orch._phase = Phase.PLAYING
        frame_bytes = 1440
        # elapsed ≈ 10ms → only ~0.33 frames elapsed, batch of 3 needs 3 frames back
        orch._playback_start_time = time.monotonic() - 0.010
        orch._sent_audio_buffer = bytearray(b"\x00" * frame_bytes * 10)
        assert orch._get_robot_audio_combined(3) is None

    def test_insufficient_buffer_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns None when buffer doesn't have enough data for batch_end."""
        orch, _ = _make_session_loop(monkeypatch, output_sample_rate=24000)
        orch._phase = Phase.PLAYING
        # elapsed far exceeds buffer
        orch._playback_start_time = time.monotonic() - 5.0
        orch._sent_audio_buffer = bytearray(b"\x00" * 100)
        assert orch._get_robot_audio_combined(2) is None

    def test_frame_count_one_matches_single_chunk_length(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """frame_count=1 returns same length as get_robot_audio_chunk."""
        orch, _ = _make_session_loop(monkeypatch, output_sample_rate=24000)
        orch._phase = Phase.PLAYING
        frame_bytes = 1440
        orch._sent_audio_buffer = bytearray(b"\x01" * frame_bytes * 5)
        orch._playback_start_time = time.monotonic() - 0.060  # ~2 frames in

        single = orch.get_robot_audio_chunk()
        combined = orch._get_robot_audio_combined(1)

        assert single is not None
        assert combined is not None
        assert len(single) == len(combined) == frame_bytes


# ---------------------------------------------------------------------------
# Exit keyword tests
# ---------------------------------------------------------------------------


class TestExitKeyword:
    def test_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        orch, _ = _make_session_loop(monkeypatch, exit_keywords=("bye",))
        assert orch._check_exit_keyword("Bye") is True
        assert orch._check_exit_keyword("BYE") is True

    def test_word_boundary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        orch, _ = _make_session_loop(monkeypatch, exit_keywords=("bye",))
        assert orch._check_exit_keyword("bye friend") is True
        assert orch._check_exit_keyword("goodbye") is False  # "bye" not a separate word

    def test_punctuation_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        orch, _ = _make_session_loop(monkeypatch, exit_keywords=("bye",))
        assert orch._check_exit_keyword("bye!") is True
        assert orch._check_exit_keyword("bye.") is True

    def test_exit_keyword_in_turn_shift(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Turn shift with exit keyword returns True (end session)."""
        orch, mocks = _make_session_loop(monkeypatch, exit_keywords=("bye",))
        mocks["asr"].get_text.return_value = "bye"
        mocks["turn_detector"].process_frame.return_value = TurnDecision(turn_shift=True)

        q = _audio_queue_with(_frame())
        orch._start_session()
        orch._audio_queue = q
        result = orch._run_frame()

        assert result is True

    def test_empty_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        orch, _ = _make_session_loop(monkeypatch, exit_keywords=("bye",))
        assert orch._check_exit_keyword("") is False


# ---------------------------------------------------------------------------
# Session timeout tests
# ---------------------------------------------------------------------------


class TestSessionTimeout:
    def test_timeout_triggers_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Session exits after timeout with no text change."""
        orch, mocks = _make_session_loop(monkeypatch, session_timeout_sec=0.0)
        orch._start_session()

        q = _audio_queue_with()
        orch._audio_queue = q
        result = orch._run_frame()
        assert result is True

    def test_paused_during_playing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Timeout is paused during PLAYING."""
        orch, mocks = _make_session_loop(monkeypatch, session_timeout_sec=0.0)
        orch._start_session()
        orch._phase = Phase.PLAYING

        q = _audio_queue_with()
        orch._audio_queue = q
        result = orch._run_frame()
        assert result is False

    def test_paused_during_awaiting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Timeout is paused during awaiting_response."""
        orch, mocks = _make_session_loop(monkeypatch, session_timeout_sec=0.0)
        orch._start_session()
        orch._phase = Phase.AWAITING

        q = _audio_queue_with()
        orch._audio_queue = q
        result = orch._run_frame()
        assert result is False

    def test_text_change_resets_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Text change resets the timeout timer."""
        orch, mocks = _make_session_loop(monkeypatch, session_timeout_sec=0.05)
        orch._start_session()

        # First frame: no text → timer started
        mocks["asr"].get_text.return_value = ""
        q = _audio_queue_with()
        orch._audio_queue = q
        orch._run_frame()

        # Second frame: text changes → timer resets
        mocks["asr"].get_text.return_value = "hello"
        q = _audio_queue_with(_frame())
        orch._audio_queue = q
        orch._run_frame()

        # Should not timeout immediately
        q = _audio_queue_with()
        orch._audio_queue = q
        result = orch._run_frame()
        assert result is False


# ---------------------------------------------------------------------------
# STOP_PENDING watchdog tests
# ---------------------------------------------------------------------------


class TestStopPendingWatchdog:
    def test_watchdog_forces_idle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """STOP_PENDING watchdog timeout forces IDLE."""
        orch, mocks = _make_session_loop(monkeypatch, stop_pending_timeout_sec=0.0)
        orch._start_session()
        orch._phase = Phase.STOPPING
        orch._stop_pending_time = time.monotonic() - 1.0

        q = _audio_queue_with()
        orch._audio_queue = q
        orch._run_frame()

        assert orch._phase == Phase.LISTENING

    def test_stale_complete_ignored_after_watchdog(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After watchdog forces IDLE, stale PLAYBACK_COMPLETE is ignored."""
        orch, mocks = _make_session_loop(monkeypatch, stop_pending_timeout_sec=0.0)
        orch._start_session()
        orch._phase = Phase.LISTENING  # After watchdog

        # Stale event arrives
        event = CppEvent(CppEventType.PLAYBACK_COMPLETE)
        mocks["bridge"].poll_event.side_effect = [event, None]

        q = _audio_queue_with()
        orch._audio_queue = q
        orch._run_frame()

        # Should remain IDLE, no history save for this
        assert orch._phase == Phase.LISTENING
        mocks["history"].add_assistant_message.assert_not_called()


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestRequestStop:
    def test_request_stop_exits_frame_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """request_stop() causes _run_frame() to return True immediately."""
        orch, mocks = _make_session_loop(monkeypatch, session_timeout_sec=100.0)
        orch._start_session()
        orch.request_stop()

        q = _audio_queue_with(_frame())
        orch._audio_queue = q
        result = orch._run_frame()

        assert result is True

    def test_run_clears_stale_stop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """run() clears a stale stop event from a previous session."""
        orch, mocks = _make_session_loop(monkeypatch, session_timeout_sec=0.0)
        # Set stop before run — should be cleared
        orch.request_stop()

        # run() should clear the event and proceed normally (exit via timeout)
        orch.run()

        # If stale stop wasn't cleared, _end_session wouldn't be called properly
        mocks["asr"].start.assert_called_once()
        mocks["asr"].stop.assert_called_once()


class TestErrorHandling:
    def test_asr_error_continues(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ASR errors don't terminate the session."""
        orch, mocks = _make_session_loop(monkeypatch, session_timeout_sec=100.0)
        orch._start_session()
        mocks["asr"].feed_audio.side_effect = RuntimeError("ASR fail")
        mocks["asr"].get_text.side_effect = RuntimeError("ASR fail")

        q = _audio_queue_with(_frame())
        orch._audio_queue = q
        result = orch._run_frame()

        assert result is False

    def test_bridge_error_terminates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CppBridge error terminates the session."""
        orch, mocks = _make_session_loop(monkeypatch)
        orch._start_session()
        mocks["bridge"].poll_event.side_effect = RuntimeError("Bridge fail")

        q = _audio_queue_with()
        orch._audio_queue = q
        result = orch._run_frame()

        assert result is True


# ---------------------------------------------------------------------------
# Cpp event tests
# ---------------------------------------------------------------------------


class TestCppEvents:
    def test_playback_started_records_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PLAYBACK_STARTED event records start time for position estimation."""
        orch, mocks = _make_session_loop(monkeypatch)
        orch._start_session()
        orch._phase = Phase.STREAMING

        event = CppEvent(CppEventType.PLAYBACK_STARTED)
        mocks["bridge"].poll_event.side_effect = [event, None]

        before = time.monotonic()
        q = _audio_queue_with()
        orch._audio_queue = q
        orch._run_frame()
        after = time.monotonic()

        assert before <= orch._playback_start_time <= after

    def test_playback_complete_saves_and_resets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PLAYBACK_COMPLETE saves full text and resets to IDLE."""
        orch, mocks = _make_session_loop(monkeypatch)
        orch._start_session()
        orch._phase = Phase.PLAYING
        orch._current_response = ResponseData(text="hi there", audio=b"\x00", timestamps=[])

        event = CppEvent(CppEventType.PLAYBACK_COMPLETE)
        mocks["bridge"].poll_event.side_effect = [event, None]

        q = _audio_queue_with()
        orch._audio_queue = q
        orch._run_frame()

        mocks["history"].add_assistant_message.assert_called_once()
        assert mocks["history"].add_assistant_message.call_args[0][0] == "hi there"
        mocks["turn_detector"].notify_turn_complete.assert_called_once_with("robot", "hi there")
        assert orch._phase == Phase.LISTENING

    def test_playback_complete_in_stop_pending_triggers_interrupted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PLAYBACK_COMPLETE during STOP_PENDING triggers barge-in handling."""
        orch, mocks = _make_session_loop(monkeypatch)
        orch._start_session()
        orch._phase = Phase.STOPPING
        orch._playback_start_time = time.monotonic() - 1.0
        orch._stop_pending_time = time.monotonic() - 0.5
        orch._current_response = ResponseData(text="hello world", audio=b"\x00" * 100, timestamps=[])

        event = CppEvent(CppEventType.PLAYBACK_COMPLETE)
        mocks["bridge"].poll_event.side_effect = [event, None]

        with patch("voice_pipeline.session_loop.truncate_by_ratio", return_value="hello"):
            q = _audio_queue_with()
            orch._audio_queue = q
            orch._run_frame()

        assert orch._phase == Phase.LISTENING
        mocks["history"].add_assistant_message.assert_called_once()
        assert mocks["history"].add_assistant_message.call_args[0][0] == "hello"

    def test_playback_complete_ignored_when_idle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PLAYBACK_COMPLETE is ignored when in IDLE state."""
        orch, mocks = _make_session_loop(monkeypatch)
        orch._start_session()
        orch._phase = Phase.LISTENING

        event = CppEvent(CppEventType.PLAYBACK_COMPLETE)
        mocks["bridge"].poll_event.side_effect = [event, None]

        q = _audio_queue_with()
        orch._audio_queue = q
        orch._run_frame()

        assert orch._phase == Phase.LISTENING
        mocks["history"].add_assistant_message.assert_not_called()


# ---------------------------------------------------------------------------
# Drain audio tests
# ---------------------------------------------------------------------------


class TestDrainAudio:
    def test_drain_sends_all_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Drain sends all available chunks to bridge."""
        orch, mocks = _make_session_loop(monkeypatch)
        orch._start_session()
        orch._phase = Phase.PLAYING

        chunks = [b"\x01" * 100, b"\x02" * 100]
        mocks["generator"].poll_audio.side_effect = chunks + [None]
        mocks["generator"].stream_done = False

        orch._drain_audio_to_bridge()

        assert mocks["bridge"].send_audio.call_count == 2
        assert len(orch._sent_audio_buffer) == 200

    def test_drain_gets_response_data_on_stream_done(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When stream_done after drain, get_response_data is called and audio_end sent."""
        orch, mocks = _make_session_loop(monkeypatch)
        orch._start_session()
        orch._phase = Phase.PLAYING

        response = ResponseData(text="hi", audio=b"\x00", timestamps=[])
        mocks["generator"].poll_audio.return_value = None
        mocks["generator"].stream_done = True
        mocks["generator"].get_response_data.return_value = response

        orch._drain_audio_to_bridge()

        assert orch._current_response is response
        mocks["bridge"].send_audio_end.assert_called_once()

    def test_drain_sends_audio_end_only_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """audio_end is sent only once even if drain is called multiple times."""
        orch, mocks = _make_session_loop(monkeypatch)
        orch._start_session()
        orch._phase = Phase.PLAYING

        mocks["generator"].poll_audio.return_value = None
        mocks["generator"].stream_done = True
        mocks["generator"].get_response_data.return_value = ResponseData(text="hi", audio=b"\x00", timestamps=[])

        orch._drain_audio_to_bridge()
        orch._drain_audio_to_bridge()

        mocks["bridge"].send_audio_end.assert_called_once()


# ===================================================================
# Audio starvation
# ===================================================================


class TestAudioStarvation:
    def test_starvation_terminates_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Session terminates when no audio frames arrive for starvation timeout."""
        orch, mocks = _make_session_loop(monkeypatch)
        orch._start_session()

        # Push _last_frame_time back beyond the starvation threshold
        orch._last_frame_time = time.monotonic() - (SessionLoop._AUDIO_STARVATION_TIMEOUT_SEC + 0.1)

        assert orch._run_frame() is True

    def test_starvation_resets_on_frame(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Receiving a frame resets the starvation timer."""
        audio_queue: queue.Queue[AudioFrame] = queue.Queue()
        orch, mocks = _make_session_loop(monkeypatch, audio_queue=audio_queue)
        orch._start_session()

        # Expire starvation timer
        orch._last_frame_time = time.monotonic() - 100.0

        # Push a frame — should reset timer and NOT terminate
        audio_queue.put(b"\x00" * 960)

        assert orch._run_frame() is False
        # Timer was refreshed — verify by checking it's recent
        assert time.monotonic() - orch._last_frame_time < 1.0

    def test_starvation_not_paused_during_playback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Audio starvation fires even during PLAYING state."""
        orch, mocks = _make_session_loop(monkeypatch)
        orch._start_session()

        orch._phase = Phase.PLAYING
        orch._last_frame_time = time.monotonic() - (SessionLoop._AUDIO_STARVATION_TIMEOUT_SEC + 0.1)

        assert orch._run_frame() is True


# ---------------------------------------------------------------------------
# Cancel (turn_shift was premature — user continued)
# ---------------------------------------------------------------------------


class TestCancel:
    def test_cancel_decision_cancels_generation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cancel decision during AWAITING discards generation → LISTENING."""
        orch, mocks = _make_session_loop(monkeypatch)
        orch._start_session()
        orch._phase = Phase.AWAITING
        mocks["turn_detector"].process_frame.return_value = TurnDecision(cancel=True)

        q = _audio_queue_with(_frame())
        orch._audio_queue = q
        orch._run_frame()

        mocks["generator"].cancel.assert_called_once()
        assert orch._phase is Phase.LISTENING

    def test_cancel_does_not_reset_detector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SessionLoop does not reset the detector on cancel (it self-rewinds)."""
        orch, mocks = _make_session_loop(monkeypatch)
        orch._start_session()
        orch._phase = Phase.AWAITING
        mocks["turn_detector"].process_frame.return_value = TurnDecision(cancel=True)

        q = _audio_queue_with(_frame())
        orch._audio_queue = q
        orch._run_frame()

        mocks["turn_detector"].reset.assert_not_called()

    def test_cancel_preserves_asr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cancel keeps ASR (the same user turn continues — no asr.reset)."""
        orch, mocks = _make_session_loop(monkeypatch)
        orch._start_session()
        orch._phase = Phase.AWAITING
        mocks["asr"].get_text.return_value = "hello"
        mocks["turn_detector"].process_frame.return_value = TurnDecision(cancel=True)

        q = _audio_queue_with(_frame())
        orch._audio_queue = q
        orch._run_frame()

        mocks["asr"].reset.assert_not_called()


# ---------------------------------------------------------------------------
# Utterance storage (memory integration)
# ---------------------------------------------------------------------------


def _make_session_loop_with_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SessionLoop, dict[str, MagicMock]]:
    """Create an SessionLoop with memory_storage, session_id, and token_counter."""
    mocks = {
        "asr": MagicMock(),
        "turn_detector": MagicMock(),
        "generator": MagicMock(),
        "bridge": MagicMock(),
        "history": MagicMock(),
        "led": MagicMock(),
        "memory_storage": MagicMock(),
        "token_counter": MagicMock(return_value=5),
    }

    mocks["asr"].get_text.return_value = ""
    mocks["turn_detector"].process_frame.return_value = TurnDecision.none()
    mocks["generator"].state = GeneratorState.IDLE
    mocks["generator"].stream_done = False
    mocks["generator"].poll_audio.return_value = None
    mocks["bridge"].poll_event.return_value = None
    mocks["history"].add_user_message.return_value = 0
    mocks["history"].add_assistant_message.return_value = 1

    orch = SessionLoop(
        asr=mocks["asr"],
        turn_detector=mocks["turn_detector"],
        speech_generator=mocks["generator"],
        cpp_bridge=mocks["bridge"],
        history=mocks["history"],
        led=mocks["led"],
        audio_queue=queue.Queue(),
        tts_sample_rate=OpenAITTS.OUTPUT_SAMPLE_RATE,
        memory_storage=mocks["memory_storage"],
        session_id="test-session-id",
        token_counter=mocks["token_counter"],
    )
    return orch, mocks


class TestUtteranceStorage:
    def test_user_utterance_saved_on_begin_streaming(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """User utterance is saved when streaming begins."""
        orch, mocks = _make_session_loop_with_memory(monkeypatch)
        orch._start_session()
        mocks["generator"].state = GeneratorState.STREAMING
        mocks["generator"].input_text = "hello there"
        mocks["generator"].poll_audio.return_value = None
        mocks["generator"].stream_done = False

        orch._begin_streaming()

        mocks["memory_storage"].add_utterance.assert_called_once()
        call_args = mocks["memory_storage"].add_utterance.call_args
        assert call_args[0][0] == "test-session-id"
        assert call_args[0][1] == "user"
        assert call_args[0][2] == "hello there"

    def test_assistant_utterance_saved_on_playback_complete(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Assistant utterance is saved on normal playback completion."""
        orch, mocks = _make_session_loop_with_memory(monkeypatch)
        orch._start_session()
        orch._phase = Phase.PLAYING
        orch._current_response = ResponseData(text="I'm fine", audio=b"\x00" * 100)

        orch._on_playback_complete()

        calls = mocks["memory_storage"].add_utterance.call_args_list
        assert len(calls) == 1
        assert calls[0][0][1] == "assistant"
        assert calls[0][0][2] == "I'm fine"

    def test_assistant_utterance_saved_on_interrupt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Truncated assistant utterance is saved on barge-in."""
        orch, mocks = _make_session_loop_with_memory(monkeypatch)
        orch._start_session()
        orch._phase = Phase.STOPPING
        orch._playback_start_time = time.monotonic() - 1.0
        orch._stop_pending_time = time.monotonic() - 0.5
        orch._current_response = ResponseData(
            text="I am doing well today",
            audio=b"\x00" * 48000,  # 1 second at 24kHz 16-bit
        )
        with patch("voice_pipeline.session_loop.truncate_by_timestamps", return_value="I am doing"):
            orch._on_playback_interrupted()

        calls = mocks["memory_storage"].add_utterance.call_args_list
        assert len(calls) == 1
        assert calls[0][0][1] == "assistant"
        assert calls[0][0][2] == "I am doing"

    def test_no_utterance_without_memory_storage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without memory_storage, no utterance saving occurs."""
        orch, mocks = _make_session_loop(monkeypatch)
        orch._start_session()
        orch._phase = Phase.PLAYING
        orch._current_response = ResponseData(text="hello", audio=b"\x00" * 100)

        orch._on_playback_complete()

        # No memory_storage → no add_utterance call (and no error)
        mocks["history"].add_assistant_message.assert_called_once()

    def test_utterance_save_error_doesnt_crash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """add_utterance error is logged but doesn't crash the session loop."""
        orch, mocks = _make_session_loop_with_memory(monkeypatch)
        orch._start_session()
        mocks["memory_storage"].add_utterance.side_effect = RuntimeError("DB error")
        orch._phase = Phase.PLAYING
        orch._current_response = ResponseData(text="hello", audio=b"\x00" * 100)

        orch._on_playback_complete()  # Should not raise

        mocks["history"].add_assistant_message.assert_called_once()


# ---------------------------------------------------------------------------
# Pipeline trace tests
# ---------------------------------------------------------------------------


def _make_session_loop_with_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SessionLoop, dict[str, MagicMock], InMemoryTraceStore]:
    """Create an SessionLoop with InMemoryTraceStore."""
    orch, mocks = _make_session_loop(monkeypatch)
    store = InMemoryTraceStore()
    orch._trace_store = store
    orch._session_id = "test-session"
    mocks["generator"].trace = PipelineTrace(run_id=1, pipeline_mode="full", prepare_ts=time.monotonic())
    return orch, mocks, store


class TestPipelineTraceCompleted:
    def test_completed_on_playback_complete(self, monkeypatch: pytest.MonkeyPatch) -> None:
        orch, mocks, store = _make_session_loop_with_trace(monkeypatch)
        orch._start_session()
        orch._phase = Phase.PLAYING
        orch._current_response = ResponseData(text="hello", audio=b"\x00" * 100)

        orch._on_playback_complete()

        assert len(store.traces) == 1
        assert store.traces[0].outcome == "completed"

    def test_completed_resets_attempts_counter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        orch, mocks, store = _make_session_loop_with_trace(monkeypatch)
        orch._start_session()
        orch._speculative_attempts = 3
        orch._phase = Phase.PLAYING
        orch._current_response = ResponseData(text="hi", audio=b"\x00" * 100)

        orch._on_playback_complete()

        assert store.traces[0].speculative_attempts == 3
        assert orch._speculative_attempts == 0


class TestPipelineTraceTruncated:
    def test_truncated_on_barge_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        orch, mocks, store = _make_session_loop_with_trace(monkeypatch)
        orch._start_session()
        orch._phase = Phase.STOPPING
        orch._playback_start_time = time.monotonic() - 1.0
        orch._stop_pending_time = time.monotonic() - 0.5
        orch._current_response = ResponseData(text="hello world", audio=b"\x00" * 4800)

        with patch("voice_pipeline.session_loop.truncate_by_ratio", return_value="hello"):
            orch._on_playback_interrupted()

        assert len(store.traces) == 1
        assert store.traces[0].outcome == "truncated"

    def test_interrupt_latency_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        orch, mocks, store = _make_session_loop_with_trace(monkeypatch)
        orch._start_session()
        orch._phase = Phase.PLAYING
        orch._playback_start_time = time.monotonic() - 1.0
        orch._current_response = ResponseData(text="hello world", audio=b"\x00" * 4800)

        orch._handle_interrupt()
        assert orch._phase is Phase.STOPPING

        trace = mocks["generator"].trace
        assert trace.interrupt_ts > 0

        with patch("voice_pipeline.session_loop.truncate_by_ratio", return_value="hello"):
            orch._on_playback_interrupted()

        assert trace.interrupt_ack_ts >= trace.interrupt_ts
        record = trace.to_record()
        assert record["interrupt_latency_ms"] > 0


class TestPipelineTraceCancelled:
    def test_cancelled_on_cancel_decision(self, monkeypatch: pytest.MonkeyPatch) -> None:
        orch, mocks, store = _make_session_loop_with_trace(monkeypatch)
        orch._start_session()
        orch._phase = Phase.AWAITING

        orch._handle_cancel()

        assert len(store.traces) == 1
        assert store.traces[0].outcome == "cancelled"

    def test_cancelled_on_user_continued_speaking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        orch, mocks, store = _make_session_loop_with_trace(monkeypatch)
        orch._start_session()
        orch._phase = Phase.AWAITING

        mocks["turn_detector"].process_frame.return_value = TurnDecision(cancel=True)

        q = _audio_queue_with(_frame())
        orch._audio_queue = q
        orch._run_frame()

        assert len(store.traces) == 1
        assert store.traces[0].outcome == "cancelled"

    def test_cancelled_on_generator_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        orch, mocks, store = _make_session_loop_with_trace(monkeypatch)
        orch._start_session()
        orch._phase = Phase.AWAITING
        mocks["generator"].state = GeneratorState.FAILED

        orch._check_generator_completion()

        assert len(store.traces) == 1
        assert store.traces[0].outcome == "cancelled"


class TestPipelineTraceSpeculative:
    def test_speculative_attempts_counted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        orch, mocks, store = _make_session_loop_with_trace(monkeypatch)
        orch._start_session()

        # Two speculative prepares
        orch._handle_prepare("text A")
        orch._handle_prepare("text B")

        # Then turn_shift → streaming → playback
        mocks["generator"].state = GeneratorState.STREAMING
        mocks["generator"].input_text = "text B"
        orch._handle_turn_shift("text B")
        orch._phase = Phase.PLAYING
        orch._current_response = ResponseData(text="response", audio=b"\x00" * 100)

        orch._on_playback_complete()

        assert len(store.traces) == 1
        assert store.traces[0].speculative_attempts == 2

    def test_speculative_replace_does_not_save(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Replacing a speculative prepare should NOT save the old trace."""
        orch, mocks, store = _make_session_loop_with_trace(monkeypatch)
        orch._start_session()

        orch._handle_prepare("text A")
        orch._handle_prepare("text B")

        # Only speculative replacements — no turn completion yet
        assert len(store.traces) == 0


class TestPipelineTraceDisabled:
    def test_no_error_when_store_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        orch, mocks = _make_session_loop(monkeypatch)
        orch._start_session()
        mocks["generator"].trace = PipelineTrace(run_id=1)
        orch._phase = Phase.PLAYING
        orch._current_response = ResponseData(text="hello", audio=b"\x00" * 100)

        orch._on_playback_complete()  # should not raise

    def test_no_error_when_trace_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        orch, mocks, store = _make_session_loop_with_trace(monkeypatch)
        orch._start_session()
        mocks["generator"].trace = None
        orch._phase = Phase.PLAYING
        orch._current_response = ResponseData(text="hello", audio=b"\x00" * 100)

        orch._on_playback_complete()  # should not raise
        assert len(store.traces) == 0


class TestPipelineTraceTimestamps:
    def test_begin_streaming_sets_trace_timestamps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        orch, mocks, store = _make_session_loop_with_trace(monkeypatch)
        orch._start_session()
        orch._turn_shift_time = time.monotonic()
        mocks["generator"].state = GeneratorState.STREAMING
        mocks["generator"].input_text = "hello"

        orch._begin_streaming()

        trace = mocks["generator"].trace
        assert trace.turn_shift_ts > 0
        assert trace.begin_streaming_ts > 0
        assert trace.session_id != ""

    def test_playback_started_sets_trace_timestamp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        orch, mocks, store = _make_session_loop_with_trace(monkeypatch)
        orch._start_session()
        orch._phase = Phase.STREAMING
        mocks["bridge"].poll_event.side_effect = [
            CppEvent(CppEventType.PLAYBACK_STARTED),
            None,
        ]

        orch._poll_cpp_events()

        trace = mocks["generator"].trace
        assert trace.playback_started_ts > 0
