"""Tests for voice_pipeline.orchestrator.orchestrator."""

from __future__ import annotations

import queue
import time
from unittest.mock import MagicMock, patch

from voice_pipeline.core.config import AudioConfig, OrchestratorConfig, TTSConfig
from voice_pipeline.core.types import (
    AudioFrame,
    CppEvent,
    CppEventType,
    GeneratorState,
    LEDState,
    PlaybackState,
    ResponseData,
    TurnDecision,
    WordTimestamp,
)
from voice_pipeline.orchestrator.orchestrator import Orchestrator, _PendingTruncation

# ---------------------------------------------------------------------------
# Fixture helper
# ---------------------------------------------------------------------------


def _make_orchestrator(
    *,
    exit_keywords: tuple[str, ...] = ("bye", "goodbye"),
    session_timeout_sec: float = 30.0,
    frame_timeout_sec: float = 0.01,
    stop_pending_timeout_sec: float = 5.0,
    output_sample_rate: int = 24000,
) -> tuple[Orchestrator, dict[str, MagicMock]]:
    """Create an Orchestrator with all dependencies mocked."""
    mocks = {
        "asr": MagicMock(),
        "turn_detector": MagicMock(),
        "generator": MagicMock(),
        "bridge": MagicMock(),
        "history": MagicMock(),
        "truncator": MagicMock(),
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

    config = OrchestratorConfig(
        exit_keywords=exit_keywords,
        session_timeout_sec=session_timeout_sec,
        frame_timeout_sec=frame_timeout_sec,
        stop_pending_timeout_sec=stop_pending_timeout_sec,
    )
    tts_config = TTSConfig(output_sample_rate=output_sample_rate)
    audio_config = AudioConfig()

    orch = Orchestrator(
        asr=mocks["asr"],
        turn_detector=mocks["turn_detector"],
        speech_generator=mocks["generator"],
        cpp_bridge=mocks["bridge"],
        history=mocks["history"],
        truncator=mocks["truncator"],
        led=mocks["led"],
        config=config,
        tts_config=tts_config,
        audio_config=audio_config,
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
    def test_start_stop_session(self) -> None:
        """start_session calls asr.start and LED IDLE; end calls reset/OFF."""
        orch, mocks = _make_orchestrator(session_timeout_sec=0.0)
        # With 0 timeout, session ends immediately
        orch.run(_audio_queue_with())
        mocks["asr"].start.assert_called_once()
        mocks["asr"].stop.assert_called_once()
        mocks["generator"].reset.assert_called_once()

        # LED sequence: IDLE (start) → OFF (end)
        led_calls = [c.args[0] for c in mocks["led"].set_state.call_args_list]
        assert led_calls[0] == LEDState.IDLE
        assert led_calls[-1] == LEDState.OFF

    def test_start_session_resets_internal_state(self) -> None:
        """run() resets all internal state from a previous session."""
        orch, mocks = _make_orchestrator(session_timeout_sec=0.0)
        # Simulate dirty state from a previous session
        orch._playback_state = PlaybackState.PLAYING
        orch._awaiting_response = True
        orch._sent_audio_buffer = bytearray(b"\xff" * 100)

        orch.run(_audio_queue_with())

        # After run(), state should have been reset at start
        # (end_session also resets some, but start must handle it)
        assert orch._awaiting_response is False


# ---------------------------------------------------------------------------
# Turn shift tests
# ---------------------------------------------------------------------------


class TestTurnShift:
    def test_turn_shift_streaming_ready(self) -> None:
        """When generator is STREAMING on turn_shift, history records input_text."""
        orch, mocks = _make_orchestrator()
        mocks["asr"].get_text.return_value = "hello world"
        mocks["turn_detector"].process_frame.return_value = TurnDecision(turn_shift=True)
        mocks["generator"].state = GeneratorState.STREAMING
        mocks["generator"].poll_audio.return_value = None
        mocks["generator"].stream_done = False
        # Simulate earlier prepare with partial text
        mocks["generator"].input_text = "hello"

        q = _audio_queue_with(_frame())
        orch._start_session()
        orch._run_frame(q)

        # History records generator.input_text, not current ASR text
        mocks["history"].add_user_message.assert_called_once_with("hello")
        mocks["asr"].reset.assert_called_once()
        assert orch._playback_state == PlaybackState.PLAYING
        mocks["bridge"].send_stream_start.assert_called_once()

    def test_turn_shift_not_ready_sets_awaiting(self) -> None:
        """When generator not ready on turn_shift, set awaiting."""
        orch, mocks = _make_orchestrator()
        mocks["asr"].get_text.return_value = "hello"
        mocks["turn_detector"].process_frame.return_value = TurnDecision(turn_shift=True)
        mocks["generator"].state = GeneratorState.IDLE

        q = _audio_queue_with(_frame())
        orch._start_session()
        orch._run_frame(q)

        assert orch._awaiting_response is True
        mocks["generator"].prepare.assert_called_once_with("hello")

    def test_turn_shift_preparing_sets_awaiting_no_double_prepare(self) -> None:
        """When generator is PREPARING, set awaiting without calling prepare again."""
        orch, mocks = _make_orchestrator()
        mocks["asr"].get_text.return_value = "hello"
        mocks["turn_detector"].process_frame.return_value = TurnDecision(turn_shift=True)
        mocks["generator"].state = GeneratorState.PREPARING

        q = _audio_queue_with(_frame())
        orch._start_session()
        orch._run_frame(q)

        assert orch._awaiting_response is True
        mocks["generator"].prepare.assert_not_called()


# ---------------------------------------------------------------------------
# Awaiting response tests
# ---------------------------------------------------------------------------


class TestAwaitingResponse:
    def test_awaiting_completion_begins_streaming(self) -> None:
        """When awaiting and generator becomes STREAMING, history records input_text."""
        orch, mocks = _make_orchestrator()
        orch._start_session()
        orch._awaiting_response = True
        mocks["generator"].state = GeneratorState.STREAMING
        mocks["generator"].input_text = "hello"
        mocks["generator"].poll_audio.return_value = None
        mocks["generator"].stream_done = False

        q = _audio_queue_with()
        orch._run_frame(q)

        assert orch._awaiting_response is False
        assert orch._playback_state == PlaybackState.PLAYING
        # Uses generator.input_text — matches what LLM actually saw
        mocks["history"].add_user_message.assert_called_once_with("hello")

    def test_begin_streaming_uses_generator_input_text(self) -> None:
        """History records generator.input_text (what LLM actually saw)."""
        orch, mocks = _make_orchestrator()
        orch._start_session()
        mocks["generator"].input_text = "earlier partial"
        mocks["generator"].state = GeneratorState.STREAMING
        mocks["generator"].poll_audio.return_value = None
        mocks["generator"].stream_done = False

        orch._begin_streaming()

        mocks["history"].add_user_message.assert_called_once_with("earlier partial")

    def test_awaiting_interrupt_cancels(self) -> None:
        """Interrupt during awaiting cancels generator and resets."""
        orch, mocks = _make_orchestrator()
        orch._start_session()
        orch._awaiting_response = True
        mocks["asr"].get_text.return_value = ""
        mocks["turn_detector"].process_frame.return_value = TurnDecision(interrupt=True)

        q = _audio_queue_with(_frame())
        orch._run_frame(q)

        mocks["generator"].cancel.assert_called_once()
        mocks["turn_detector"].reset.assert_called_once()
        assert orch._awaiting_response is False

    def test_awaiting_generator_failed_skips_turn(self) -> None:
        """Generator FAILED during awaiting skips turn and resets turn_detector."""
        orch, mocks = _make_orchestrator()
        orch._start_session()
        orch._awaiting_response = True
        mocks["generator"].state = GeneratorState.FAILED

        q = _audio_queue_with()
        orch._run_frame(q)

        assert orch._awaiting_response is False
        mocks["turn_detector"].reset.assert_called_once()


# ---------------------------------------------------------------------------
# Prepare tests
# ---------------------------------------------------------------------------


class TestPrepare:
    def test_prepare_cancels_and_restarts(self) -> None:
        """Prepare signal triggers generator.prepare with current text."""
        orch, mocks = _make_orchestrator()
        mocks["asr"].get_text.return_value = "how are you"
        mocks["turn_detector"].process_frame.return_value = TurnDecision(prepare=True)

        q = _audio_queue_with(_frame())
        orch._start_session()
        orch._run_frame(q)

        mocks["generator"].prepare.assert_called_once_with("how are you")


# ---------------------------------------------------------------------------
# Interrupt tests
# ---------------------------------------------------------------------------


class TestInterrupt:
    def test_interrupt_during_playing_sends_stop(self) -> None:
        """Interrupt during PLAYING sends stop and enters STOP_PENDING."""
        orch, mocks = _make_orchestrator()
        orch._start_session()
        orch._playback_state = PlaybackState.PLAYING
        mocks["asr"].get_text.return_value = "wait"
        mocks["turn_detector"].process_frame.return_value = TurnDecision(interrupt=True)

        q = _audio_queue_with(_frame())
        orch._run_frame(q)

        mocks["bridge"].send_stop.assert_called_once()
        assert orch._playback_state == PlaybackState.STOP_PENDING


# ---------------------------------------------------------------------------
# Barge-in truncation tests
# ---------------------------------------------------------------------------


class TestBargeIn:
    def test_case_a_timestamps(self) -> None:
        """Case A: ResponseData with timestamps → TimestampTruncator."""
        orch, mocks = _make_orchestrator()
        orch._start_session()
        orch._playback_state = PlaybackState.STOP_PENDING
        orch._playback_start_time = time.monotonic() - 1.0
        orch._stop_pending_time = time.monotonic() - 0.65

        timestamps = [
            WordTimestamp("hello", 0.0, 0.3),
            WordTimestamp("world", 0.4, 0.7),
        ]
        orch._current_response = ResponseData(
            text="hello world", audio=b"\x00" * 100, timestamps=timestamps
        )
        mocks["truncator"].truncate.return_value = "hello"

        orch._on_playback_interrupted()

        mocks["truncator"].truncate.assert_called_once()
        call_args = mocks["truncator"].truncate.call_args[0]
        assert call_args[0] == "hello world"
        # stop_pos ≈ stop_pending_time - playback_start_time ≈ 0.35
        assert 0.2 < call_args[1] < 0.5
        mocks["history"].add_assistant_message.assert_called_once_with("hello")

    def test_case_b_no_timestamps(self) -> None:
        """Case B: ResponseData without timestamps → DurationRatioTruncator."""
        orch, mocks = _make_orchestrator(output_sample_rate=24000)
        orch._start_session()
        orch._playback_state = PlaybackState.STOP_PENDING
        orch._playback_start_time = time.monotonic() - 1.0
        orch._stop_pending_time = time.monotonic() - 0.75

        # 48000 bytes @ 24kHz 16-bit = 1.0 sec
        audio = b"\x00" * 48000
        orch._current_response = ResponseData(
            text="hello world foo bar", audio=audio, timestamps=[]
        )

        with patch("voice_pipeline.orchestrator.orchestrator.DurationRatioTruncator") as MockTrunc:
            mock_instance = MockTrunc.return_value
            mock_instance.truncate.return_value = "hello"
            orch._on_playback_interrupted()

            MockTrunc.assert_called_once_with(1.0)
            mock_instance.truncate.assert_called_once()

        mocks["history"].add_assistant_message.assert_called_once_with("hello")

    def test_case_c_no_response_data_deferred(self) -> None:
        """Case C: no ResponseData → approximate truncation + deferred."""
        orch, mocks = _make_orchestrator(output_sample_rate=24000)
        orch._start_session()
        orch._playback_state = PlaybackState.STOP_PENDING
        orch._current_response = None
        orch._playback_start_time = time.monotonic() - 1.0
        orch._stop_pending_time = time.monotonic() - 0.5
        # 48000 bytes = 1.0 sec at 24kHz 16-bit
        orch._sent_audio_buffer = bytearray(b"\x00" * 48000)

        mocks["generator"].get_text.return_value = "hello world"
        mocks["generator"].stream_done = False

        with patch("voice_pipeline.orchestrator.orchestrator.DurationRatioTruncator") as MockTrunc:
            mock_instance = MockTrunc.return_value
            mock_instance.truncate.return_value = "hello"
            mocks["history"].add_assistant_message.return_value = 42

            orch._on_playback_interrupted()

            MockTrunc.assert_called_once_with(1.0)

        assert orch._pending_truncation is not None
        assert orch._pending_truncation.msg_id == 42

    def test_no_playback_start_uses_zero(self) -> None:
        """When playback_started was never received, stop_pos defaults to 0."""
        orch, mocks = _make_orchestrator()
        orch._start_session()
        orch._playback_state = PlaybackState.STOP_PENDING
        orch._playback_start_time = 0.0

        timestamps = [WordTimestamp("a", 0.0, 0.5)]
        orch._current_response = ResponseData(text="a", audio=b"\x00", timestamps=timestamps)
        mocks["truncator"].truncate.return_value = "a"

        orch._on_playback_interrupted()

        mocks["truncator"].truncate.assert_called_once_with("a", 0.0, timestamps)


# ---------------------------------------------------------------------------
# Deferred truncation tests
# ---------------------------------------------------------------------------


class TestDeferredTruncation:
    def test_stream_done_with_timestamps_updates(self) -> None:
        """When stream finishes with timestamps, update_message with precise text."""
        orch, mocks = _make_orchestrator()
        orch._start_session()

        timestamps = [
            WordTimestamp("hi", 0.0, 0.3),
            WordTimestamp("there", 0.4, 0.7),
        ]
        response_data = ResponseData(text="hi there", audio=b"\x00" * 100, timestamps=timestamps)
        mocks["generator"].stream_done = True
        mocks["generator"].get_response_data.return_value = response_data
        mocks["truncator"].truncate.return_value = "hi"

        orch._pending_truncation = _PendingTruncation(msg_id=5, stop_position_sec=0.35)
        orch._check_deferred_truncation()

        mocks["truncator"].truncate.assert_called_once_with("hi there", 0.35, timestamps)
        mocks["history"].update_message.assert_called_once_with(5, "hi")
        assert orch._pending_truncation is None

    def test_stream_done_no_timestamps_uses_ratio(self) -> None:
        """When stream finishes without timestamps, use DurationRatioTruncator."""
        orch, mocks = _make_orchestrator(output_sample_rate=24000)
        orch._start_session()

        audio = b"\x00" * 48000  # 1.0 sec
        response_data = ResponseData(text="hello world", audio=audio, timestamps=[])
        mocks["generator"].stream_done = True
        mocks["generator"].get_response_data.return_value = response_data

        orch._pending_truncation = _PendingTruncation(msg_id=5, stop_position_sec=0.5)

        with patch("voice_pipeline.orchestrator.orchestrator.DurationRatioTruncator") as MockTrunc:
            mock_instance = MockTrunc.return_value
            mock_instance.truncate.return_value = "hello"
            orch._check_deferred_truncation()

            MockTrunc.assert_called_once_with(1.0)

        mocks["history"].update_message.assert_called_once_with(5, "hello")
        assert orch._pending_truncation is None

    def test_generator_failed_clears_pending(self) -> None:
        """Generator FAILED clears pending truncation without update."""
        orch, mocks = _make_orchestrator()
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
    def test_correct_chunk_from_buffer(self) -> None:
        """get_robot_audio_chunk extracts 30ms at playback position."""
        orch, _ = _make_orchestrator(output_sample_rate=24000)
        orch._playback_state = PlaybackState.PLAYING
        orch._playback_start_time = time.monotonic()

        # 30ms @ 24kHz 16-bit = 24000 * 0.03 * 2 = 1440 bytes
        orch._sent_audio_buffer = bytearray(b"\x01" * 2880)

        chunk = orch.get_robot_audio_chunk()
        assert chunk is not None
        assert len(chunk) == 1440

    def test_not_playing_returns_none(self) -> None:
        """Returns None when not PLAYING."""
        orch, _ = _make_orchestrator()
        orch._playback_state = PlaybackState.IDLE
        assert orch.get_robot_audio_chunk() is None

    def test_no_playback_start_returns_none(self) -> None:
        """Returns None if playback_started event was never received."""
        orch, _ = _make_orchestrator(output_sample_rate=24000)
        orch._playback_state = PlaybackState.PLAYING
        orch._playback_start_time = 0.0
        orch._sent_audio_buffer = bytearray(b"\x00" * 2880)

        assert orch.get_robot_audio_chunk() is None

    def test_insufficient_buffer_returns_none(self) -> None:
        """Returns None if buffer doesn't have enough data."""
        orch, _ = _make_orchestrator(output_sample_rate=24000)
        orch._playback_state = PlaybackState.PLAYING
        orch._playback_start_time = time.monotonic()
        orch._sent_audio_buffer = bytearray(b"\x00" * 10)

        assert orch.get_robot_audio_chunk() is None


# ---------------------------------------------------------------------------
# Robot audio combined tests
# ---------------------------------------------------------------------------


class TestRobotAudioCombined:
    """Tests for _get_robot_audio_combined (batch robot audio extraction)."""

    def test_combined_extracts_n_frames(self) -> None:
        """Combined extraction returns frame_count * frame_bytes of audio."""
        orch, _ = _make_orchestrator(output_sample_rate=24000)
        orch._playback_state = PlaybackState.PLAYING
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

    def test_not_playing_returns_none(self) -> None:
        """Returns None when not in PLAYING state."""
        orch, _ = _make_orchestrator()
        orch._playback_state = PlaybackState.IDLE
        assert orch._get_robot_audio_combined(3) is None

    def test_no_playback_start_returns_none(self) -> None:
        """Returns None if playback_started event was never received."""
        orch, _ = _make_orchestrator(output_sample_rate=24000)
        orch._playback_state = PlaybackState.PLAYING
        orch._playback_start_time = 0.0
        orch._sent_audio_buffer = bytearray(b"\x00" * 10000)
        assert orch._get_robot_audio_combined(3) is None

    def test_batch_start_negative_returns_none(self) -> None:
        """Returns None when playback just started and batch can't cover full range."""
        orch, _ = _make_orchestrator(output_sample_rate=24000)
        orch._playback_state = PlaybackState.PLAYING
        frame_bytes = 1440
        # elapsed ≈ 10ms → only ~0.33 frames elapsed, batch of 3 needs 3 frames back
        orch._playback_start_time = time.monotonic() - 0.010
        orch._sent_audio_buffer = bytearray(b"\x00" * frame_bytes * 10)
        assert orch._get_robot_audio_combined(3) is None

    def test_insufficient_buffer_returns_none(self) -> None:
        """Returns None when buffer doesn't have enough data for batch_end."""
        orch, _ = _make_orchestrator(output_sample_rate=24000)
        orch._playback_state = PlaybackState.PLAYING
        # elapsed far exceeds buffer
        orch._playback_start_time = time.monotonic() - 5.0
        orch._sent_audio_buffer = bytearray(b"\x00" * 100)
        assert orch._get_robot_audio_combined(2) is None

    def test_frame_count_one_matches_single_chunk_length(self) -> None:
        """frame_count=1 returns same length as get_robot_audio_chunk."""
        orch, _ = _make_orchestrator(output_sample_rate=24000)
        orch._playback_state = PlaybackState.PLAYING
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
    def test_case_insensitive(self) -> None:
        orch, _ = _make_orchestrator(exit_keywords=("bye",))
        assert orch._check_exit_keyword("Bye") is True
        assert orch._check_exit_keyword("BYE") is True

    def test_word_boundary(self) -> None:
        orch, _ = _make_orchestrator(exit_keywords=("bye",))
        assert orch._check_exit_keyword("bye friend") is True
        assert orch._check_exit_keyword("goodbye") is False  # "bye" not a separate word

    def test_punctuation_stripped(self) -> None:
        orch, _ = _make_orchestrator(exit_keywords=("bye",))
        assert orch._check_exit_keyword("bye!") is True
        assert orch._check_exit_keyword("bye.") is True

    def test_exit_keyword_in_turn_shift(self) -> None:
        """Turn shift with exit keyword returns True (end session)."""
        orch, mocks = _make_orchestrator(exit_keywords=("bye",))
        mocks["asr"].get_text.return_value = "bye"
        mocks["turn_detector"].process_frame.return_value = TurnDecision(turn_shift=True)

        q = _audio_queue_with(_frame())
        orch._start_session()
        result = orch._run_frame(q)

        assert result is True

    def test_empty_text(self) -> None:
        orch, _ = _make_orchestrator(exit_keywords=("bye",))
        assert orch._check_exit_keyword("") is False


# ---------------------------------------------------------------------------
# Session timeout tests
# ---------------------------------------------------------------------------


class TestSessionTimeout:
    def test_timeout_triggers_exit(self) -> None:
        """Session exits after timeout with no text change."""
        orch, mocks = _make_orchestrator(session_timeout_sec=0.0)
        orch._start_session()

        q = _audio_queue_with()
        result = orch._run_frame(q)
        assert result is True

    def test_paused_during_playing(self) -> None:
        """Timeout is paused during PLAYING."""
        orch, mocks = _make_orchestrator(session_timeout_sec=0.0)
        orch._start_session()
        orch._playback_state = PlaybackState.PLAYING

        q = _audio_queue_with()
        result = orch._run_frame(q)
        assert result is False

    def test_paused_during_awaiting(self) -> None:
        """Timeout is paused during awaiting_response."""
        orch, mocks = _make_orchestrator(session_timeout_sec=0.0)
        orch._start_session()
        orch._awaiting_response = True

        q = _audio_queue_with()
        result = orch._run_frame(q)
        assert result is False

    def test_text_change_resets_timeout(self) -> None:
        """Text change resets the timeout timer."""
        orch, mocks = _make_orchestrator(session_timeout_sec=0.05)
        orch._start_session()

        # First frame: no text → timer started
        mocks["asr"].get_text.return_value = ""
        q = _audio_queue_with()
        orch._run_frame(q)

        # Second frame: text changes → timer resets
        mocks["asr"].get_text.return_value = "hello"
        q = _audio_queue_with(_frame())
        orch._run_frame(q)

        # Should not timeout immediately
        q = _audio_queue_with()
        result = orch._run_frame(q)
        assert result is False


# ---------------------------------------------------------------------------
# STOP_PENDING watchdog tests
# ---------------------------------------------------------------------------


class TestStopPendingWatchdog:
    def test_watchdog_forces_idle(self) -> None:
        """STOP_PENDING watchdog timeout forces IDLE."""
        orch, mocks = _make_orchestrator(stop_pending_timeout_sec=0.0)
        orch._start_session()
        orch._playback_state = PlaybackState.STOP_PENDING
        orch._stop_pending_time = time.monotonic() - 1.0

        q = _audio_queue_with()
        orch._run_frame(q)

        assert orch._playback_state == PlaybackState.IDLE

    def test_stale_complete_ignored_after_watchdog(self) -> None:
        """After watchdog forces IDLE, stale PLAYBACK_COMPLETE is ignored."""
        orch, mocks = _make_orchestrator(stop_pending_timeout_sec=0.0)
        orch._start_session()
        orch._playback_state = PlaybackState.IDLE  # After watchdog

        # Stale event arrives
        event = CppEvent(CppEventType.PLAYBACK_COMPLETE)
        mocks["bridge"].poll_event.side_effect = [event, None]

        q = _audio_queue_with()
        orch._run_frame(q)

        # Should remain IDLE, no history save for this
        assert orch._playback_state == PlaybackState.IDLE
        mocks["history"].add_assistant_message.assert_not_called()


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestRequestStop:
    def test_request_stop_exits_frame_loop(self) -> None:
        """request_stop() causes _run_frame() to return True immediately."""
        orch, mocks = _make_orchestrator(session_timeout_sec=100.0)
        orch._start_session()
        orch.request_stop()

        q = _audio_queue_with(_frame())
        result = orch._run_frame(q)

        assert result is True

    def test_run_clears_stale_stop(self) -> None:
        """run() clears a stale stop event from a previous session."""
        orch, mocks = _make_orchestrator(session_timeout_sec=0.0)
        # Set stop before run — should be cleared
        orch.request_stop()

        # run() should clear the event and proceed normally (exit via timeout)
        orch.run(_audio_queue_with())

        # If stale stop wasn't cleared, _end_session wouldn't be called properly
        mocks["asr"].start.assert_called_once()
        mocks["asr"].stop.assert_called_once()


class TestErrorHandling:
    def test_asr_error_continues(self) -> None:
        """ASR errors don't terminate the session."""
        orch, mocks = _make_orchestrator(session_timeout_sec=100.0)
        orch._start_session()
        mocks["asr"].feed_audio.side_effect = RuntimeError("ASR fail")
        mocks["asr"].get_text.side_effect = RuntimeError("ASR fail")

        q = _audio_queue_with(_frame())
        result = orch._run_frame(q)

        assert result is False

    def test_bridge_error_terminates(self) -> None:
        """CppBridge error terminates the session."""
        orch, mocks = _make_orchestrator()
        orch._start_session()
        mocks["bridge"].poll_event.side_effect = RuntimeError("Bridge fail")

        q = _audio_queue_with()
        result = orch._run_frame(q)

        assert result is True


# ---------------------------------------------------------------------------
# Cpp event tests
# ---------------------------------------------------------------------------


class TestCppEvents:
    def test_playback_started_records_time(self) -> None:
        """PLAYBACK_STARTED event records start time for position estimation."""
        orch, mocks = _make_orchestrator()
        orch._start_session()
        orch._playback_state = PlaybackState.PLAYING

        event = CppEvent(CppEventType.PLAYBACK_STARTED)
        mocks["bridge"].poll_event.side_effect = [event, None]

        before = time.monotonic()
        q = _audio_queue_with()
        orch._run_frame(q)
        after = time.monotonic()

        assert before <= orch._playback_start_time <= after

    def test_playback_complete_saves_and_resets(self) -> None:
        """PLAYBACK_COMPLETE saves full text and resets to IDLE."""
        orch, mocks = _make_orchestrator()
        orch._start_session()
        orch._playback_state = PlaybackState.PLAYING
        orch._current_response = ResponseData(text="hi there", audio=b"\x00", timestamps=[])

        event = CppEvent(CppEventType.PLAYBACK_COMPLETE)
        mocks["bridge"].poll_event.side_effect = [event, None]

        q = _audio_queue_with()
        orch._run_frame(q)

        mocks["history"].add_assistant_message.assert_called_once_with("hi there")
        mocks["turn_detector"].notify_turn_complete.assert_called_once_with("robot", "hi there")
        assert orch._playback_state == PlaybackState.IDLE

    def test_playback_complete_in_stop_pending_triggers_interrupted(self) -> None:
        """PLAYBACK_COMPLETE during STOP_PENDING triggers barge-in handling."""
        orch, mocks = _make_orchestrator()
        orch._start_session()
        orch._playback_state = PlaybackState.STOP_PENDING
        orch._playback_start_time = time.monotonic() - 1.0
        orch._stop_pending_time = time.monotonic() - 0.5
        orch._current_response = ResponseData(
            text="hello world", audio=b"\x00" * 100, timestamps=[]
        )

        event = CppEvent(CppEventType.PLAYBACK_COMPLETE)
        mocks["bridge"].poll_event.side_effect = [event, None]

        with patch("voice_pipeline.orchestrator.orchestrator.DurationRatioTruncator") as MockTrunc:
            mock_instance = MockTrunc.return_value
            mock_instance.truncate.return_value = "hello"

            q = _audio_queue_with()
            orch._run_frame(q)

        assert orch._playback_state == PlaybackState.IDLE
        mocks["history"].add_assistant_message.assert_called_once_with("hello")

    def test_playback_complete_ignored_when_idle(self) -> None:
        """PLAYBACK_COMPLETE is ignored when in IDLE state."""
        orch, mocks = _make_orchestrator()
        orch._start_session()
        orch._playback_state = PlaybackState.IDLE

        event = CppEvent(CppEventType.PLAYBACK_COMPLETE)
        mocks["bridge"].poll_event.side_effect = [event, None]

        q = _audio_queue_with()
        orch._run_frame(q)

        assert orch._playback_state == PlaybackState.IDLE
        mocks["history"].add_assistant_message.assert_not_called()


# ---------------------------------------------------------------------------
# Drain audio tests
# ---------------------------------------------------------------------------


class TestDrainAudio:
    def test_drain_sends_all_chunks(self) -> None:
        """Drain sends all available chunks to bridge."""
        orch, mocks = _make_orchestrator()
        orch._start_session()
        orch._playback_state = PlaybackState.PLAYING

        chunks = [b"\x01" * 100, b"\x02" * 100]
        mocks["generator"].poll_audio.side_effect = chunks + [None]
        mocks["generator"].stream_done = False

        orch._drain_audio_to_bridge()

        assert mocks["bridge"].send_audio.call_count == 2
        assert len(orch._sent_audio_buffer) == 200

    def test_drain_gets_response_data_on_stream_done(self) -> None:
        """When stream_done after drain, get_response_data is called and audio_end sent."""
        orch, mocks = _make_orchestrator()
        orch._start_session()
        orch._playback_state = PlaybackState.PLAYING

        response = ResponseData(text="hi", audio=b"\x00", timestamps=[])
        mocks["generator"].poll_audio.return_value = None
        mocks["generator"].stream_done = True
        mocks["generator"].get_response_data.return_value = response

        orch._drain_audio_to_bridge()

        assert orch._current_response is response
        mocks["bridge"].send_audio_end.assert_called_once()

    def test_drain_sends_audio_end_only_once(self) -> None:
        """audio_end is sent only once even if drain is called multiple times."""
        orch, mocks = _make_orchestrator()
        orch._start_session()
        orch._playback_state = PlaybackState.PLAYING

        mocks["generator"].poll_audio.return_value = None
        mocks["generator"].stream_done = True
        mocks["generator"].get_response_data.return_value = ResponseData(
            text="hi", audio=b"\x00", timestamps=[]
        )

        orch._drain_audio_to_bridge()
        orch._drain_audio_to_bridge()

        mocks["bridge"].send_audio_end.assert_called_once()


# ===================================================================
# Audio starvation
# ===================================================================


class TestAudioStarvation:
    def test_starvation_terminates_session(self) -> None:
        """Session terminates when no audio frames arrive for starvation timeout."""
        orch, mocks = _make_orchestrator()
        orch._start_session()

        # Push _last_frame_time back beyond the starvation threshold
        orch._last_frame_time = time.monotonic() - (
            orch._config.audio_starvation_timeout_sec + 0.1
        )

        audio_queue: queue.Queue[AudioFrame] = queue.Queue()
        assert orch._run_frame(audio_queue) is True

    def test_starvation_resets_on_frame(self) -> None:
        """Receiving a frame resets the starvation timer."""
        orch, mocks = _make_orchestrator()
        orch._start_session()

        # Expire starvation timer
        orch._last_frame_time = time.monotonic() - 100.0

        # Push a frame — should reset timer and NOT terminate
        audio_queue: queue.Queue[AudioFrame] = queue.Queue()
        audio_queue.put(b"\x00" * 960)

        assert orch._run_frame(audio_queue) is False
        # Timer was refreshed — verify by checking it's recent
        assert time.monotonic() - orch._last_frame_time < 1.0

    def test_starvation_not_paused_during_playback(self) -> None:
        """Audio starvation fires even during PLAYING state."""
        orch, mocks = _make_orchestrator()
        orch._start_session()

        orch._playback_state = PlaybackState.PLAYING
        orch._last_frame_time = time.monotonic() - (
            orch._config.audio_starvation_timeout_sec + 0.1
        )

        audio_queue: queue.Queue[AudioFrame] = queue.Queue()
        assert orch._run_frame(audio_queue) is True


# ---------------------------------------------------------------------------
# Awaiting cancel on ASR text change
# ---------------------------------------------------------------------------


class TestAwaitingCancel:
    def test_cancel_on_text_change_after_grace(self) -> None:
        """ASR text change after grace period cancels generation."""
        orch, mocks = _make_orchestrator()
        orch._start_session()
        orch._awaiting_response = True
        # Simulate turn_shift happened well before grace period
        orch._turn_shift_time = time.monotonic() - 1.0
        orch._last_asr_text = "hello"

        mocks["asr"].get_text.return_value = "hello world"

        q = _audio_queue_with(_frame())
        orch._run_frame(q)

        mocks["generator"].cancel.assert_called_once()
        mocks["turn_detector"].reset.assert_called_once()
        assert orch._awaiting_response is False

    def test_no_cancel_within_grace_period(self) -> None:
        """ASR text change within grace period does NOT cancel."""
        orch, mocks = _make_orchestrator()
        orch._start_session()
        orch._awaiting_response = True
        # turn_shift just happened (within grace period)
        orch._turn_shift_time = time.monotonic()
        orch._last_asr_text = "hello"

        mocks["asr"].get_text.return_value = "hello world"

        q = _audio_queue_with(_frame())
        orch._run_frame(q)

        mocks["generator"].cancel.assert_not_called()
        assert orch._awaiting_response is True

    def test_cancel_resets_state(self) -> None:
        """After awaiting cancel, state returns to normal for new turn."""
        orch, mocks = _make_orchestrator()
        orch._start_session()
        orch._awaiting_response = True
        orch._audio_end_sent = True
        orch._turn_shift_time = time.monotonic() - 1.0
        orch._last_asr_text = "hello"

        mocks["asr"].get_text.return_value = "hello world"

        q = _audio_queue_with(_frame())
        orch._run_frame(q)

        assert orch._awaiting_response is False
        assert orch._audio_end_sent is False
        mocks["turn_detector"].reset.assert_called_once()

    def test_no_cancel_when_not_awaiting(self) -> None:
        """Text change without awaiting_response does not trigger cancel."""
        orch, mocks = _make_orchestrator()
        orch._start_session()
        orch._awaiting_response = False
        orch._turn_shift_time = time.monotonic() - 1.0
        orch._last_asr_text = "hello"

        mocks["asr"].get_text.return_value = "hello world"

        q = _audio_queue_with(_frame())
        orch._run_frame(q)

        mocks["generator"].cancel.assert_not_called()
