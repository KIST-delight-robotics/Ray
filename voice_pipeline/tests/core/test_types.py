"""Tests for voice_pipeline.core.types."""

import pytest

from voice_pipeline.core.types import (
    CppEvent,
    CppEventType,
    GeneratorState,
    LEDState,
    PipelineTrace,
    PlaybackState,
    ResponseData,
    SystemMode,
    TTSResult,
    TTSStream,
    TurnDecision,
    VAPResult,
    WordTimestamp,
)


class TestTurnDecision:
    def test_none_factory(self) -> None:
        decision = TurnDecision.none()
        assert decision.turn_shift is False
        assert decision.interrupt is False
        assert decision.prepare is False

    def test_single_signal_turn_shift(self) -> None:
        decision = TurnDecision(turn_shift=True)
        assert decision.turn_shift is True
        assert decision.interrupt is False

    def test_single_signal_interrupt(self) -> None:
        decision = TurnDecision(interrupt=True)
        assert decision.interrupt is True

    def test_single_signal_prepare(self) -> None:
        decision = TurnDecision(prepare=True)
        assert decision.prepare is True

    def test_multiple_signals_raises(self) -> None:
        with pytest.raises(ValueError, match="at most one signal"):
            TurnDecision(turn_shift=True, interrupt=True)

    def test_all_signals_raises(self) -> None:
        with pytest.raises(ValueError, match="at most one signal"):
            TurnDecision(turn_shift=True, interrupt=True, prepare=True)

    def test_frozen(self) -> None:
        decision = TurnDecision.none()
        with pytest.raises(AttributeError):
            decision.turn_shift = True  # type: ignore[misc]


class TestVAPResult:
    def test_construction(self) -> None:
        result = VAPResult(p_now=0.8, p_fut=0.3, user_is_speaking=True)
        assert result.p_now == 0.8
        assert result.p_fut == 0.3
        assert result.user_is_speaking is True

    def test_frozen(self) -> None:
        result = VAPResult(p_now=0.5, p_fut=0.5, user_is_speaking=False)
        with pytest.raises(AttributeError):
            result.p_now = 0.9  # type: ignore[misc]


class TestWordTimestamp:
    def test_construction(self) -> None:
        ts = WordTimestamp(word="hello", start_sec=0.0, end_sec=0.5)
        assert ts.word == "hello"
        assert ts.start_sec == 0.0
        assert ts.end_sec == 0.5


class TestTTSResult:
    def test_construction_no_timestamps(self) -> None:
        result = TTSResult(audio=b"\x00\x01")
        assert result.audio == b"\x00\x01"
        assert result.timestamps == ()

    def test_construction_with_timestamps(self) -> None:
        ts = (WordTimestamp(word="hi", start_sec=0.0, end_sec=0.2),)
        result = TTSResult(audio=b"\x00", timestamps=ts)
        assert len(result.timestamps) == 1
        assert result.timestamps[0].word == "hi"


class TestTTSStream:
    @staticmethod
    def _make_gen(chunks: list[bytes]):
        """Helper: create a generator yielding byte chunks."""
        yield from chunks

    def test_iteration_yields_chunks(self) -> None:
        chunks = [b"\x01\x02", b"\x03\x04"]
        stream = TTSStream(self._make_gen(chunks))
        collected = list(stream)
        assert collected == chunks

    def test_audio_after_iteration(self) -> None:
        chunks = [b"\x01\x02", b"\x03\x04"]
        stream = TTSStream(self._make_gen(chunks))
        list(stream)
        assert stream.audio == b"\x01\x02\x03\x04"

    def test_audio_before_iteration_raises(self) -> None:
        stream = TTSStream(self._make_gen([b"\x01"]))
        with pytest.raises(RuntimeError, match="not available"):
            _ = stream.audio

    def test_timestamps_default_empty(self) -> None:
        stream = TTSStream(self._make_gen([b"\x01"]))
        list(stream)
        assert stream.timestamps == ()

    def test_timestamps_with_fn(self) -> None:
        ts = (WordTimestamp(word="hi", start_sec=0.0, end_sec=0.2),)
        stream = TTSStream(self._make_gen([b"\x01"]), timestamps_fn=lambda: ts)
        list(stream)
        assert stream.timestamps == ts

    def test_timestamps_before_iteration_raises(self) -> None:
        stream = TTSStream(self._make_gen([b"\x01"]))
        with pytest.raises(RuntimeError, match="not available"):
            _ = stream.timestamps

    def test_timestamps_fn_cached(self) -> None:
        call_count = 0

        def counting_fn() -> tuple[WordTimestamp, ...]:
            nonlocal call_count
            call_count += 1
            return ()

        stream = TTSStream(self._make_gen([b"\x01"]), timestamps_fn=counting_fn)
        list(stream)
        _ = stream.timestamps
        _ = stream.timestamps
        assert call_count == 1

    def test_result_property(self) -> None:
        chunks = [b"\x01\x02", b"\x03\x04"]
        stream = TTSStream(self._make_gen(chunks))
        list(stream)
        result = stream.result
        assert isinstance(result, TTSResult)
        assert result.audio == b"\x01\x02\x03\x04"
        assert result.timestamps == ()

    def test_close_before_iteration(self) -> None:
        close_called = False

        def on_close() -> None:
            nonlocal close_called
            close_called = True

        stream = TTSStream(self._make_gen([b"\x01"]), close_fn=on_close)
        stream.close()
        assert close_called

    def test_close_idempotent(self) -> None:
        call_count = 0

        def on_close() -> None:
            nonlocal call_count
            call_count += 1

        stream = TTSStream(self._make_gen([b"\x01"]), close_fn=on_close)
        stream.close()
        stream.close()
        assert call_count == 1

    def test_close_stops_iteration(self) -> None:
        stream = TTSStream(self._make_gen([b"\x01", b"\x02", b"\x03"]))
        next(stream)
        stream.close()
        # After close, next() should raise StopIteration
        with pytest.raises(StopIteration):
            next(stream)

    def test_done_not_set_on_close(self) -> None:
        """close() should not set _done; audio should remain unavailable."""
        stream = TTSStream(self._make_gen([b"\x01", b"\x02"]))
        next(stream)
        stream.close()
        with pytest.raises(RuntimeError, match="not available"):
            _ = stream.audio

    def test_close_fn_error_suppressed(self) -> None:
        def bad_close() -> None:
            raise RuntimeError("close error")

        stream = TTSStream(self._make_gen([b"\x01"]), close_fn=bad_close)
        # Should not raise
        stream.close()

    def test_empty_stream(self) -> None:
        stream = TTSStream(self._make_gen([]))
        collected = list(stream)
        assert collected == []
        assert stream.audio == b""


class TestResponseData:
    def test_has_timestamps_false(self) -> None:
        data = ResponseData(text="hello", audio=b"\x00")
        assert data.has_timestamps is False

    def test_has_timestamps_true(self) -> None:
        ts = [WordTimestamp(word="hello", start_sec=0.0, end_sec=0.5)]
        data = ResponseData(text="hello", audio=b"\x00", timestamps=ts)
        assert data.has_timestamps is True

    def test_mutable(self) -> None:
        data = ResponseData(text="hello", audio=b"\x00")
        data.text = "world"
        assert data.text == "world"


class TestCppEvent:
    def test_playback_started(self) -> None:
        event = CppEvent(event_type=CppEventType.PLAYBACK_STARTED)
        assert event.event_type == CppEventType.PLAYBACK_STARTED

    def test_playback_complete(self) -> None:
        event = CppEvent(event_type=CppEventType.PLAYBACK_COMPLETE)
        assert event.event_type == CppEventType.PLAYBACK_COMPLETE

    def test_frozen(self) -> None:
        event = CppEvent(event_type=CppEventType.PLAYBACK_STARTED)
        import pytest

        with pytest.raises(AttributeError):
            event.event_type = CppEventType.PLAYBACK_COMPLETE  # type: ignore[misc]


class TestLEDState:
    def test_values(self) -> None:
        assert LEDState.OFF.value == "off"
        assert LEDState.SLEEPING.value == "sleeping"
        assert LEDState.IDLE.value == "idle"

    def test_member_count(self) -> None:
        assert len(LEDState) == 3


class TestEnums:
    def test_system_mode_values(self) -> None:
        assert SystemMode.SLEEP.value == "sleep"
        assert SystemMode.GREETING.value == "greeting"
        assert SystemMode.ACTIVE.value == "active"
        assert SystemMode.FAREWELL.value == "farewell"

    def test_playback_state_values(self) -> None:
        assert PlaybackState.IDLE.value == "idle"
        assert PlaybackState.PLAYING.value == "playing"
        assert PlaybackState.STOP_PENDING.value == "stop_pending"

    def test_generator_state_values(self) -> None:
        assert GeneratorState.IDLE.value == "idle"
        assert GeneratorState.PREPARING.value == "preparing"
        assert GeneratorState.STREAMING.value == "streaming"
        assert GeneratorState.FAILED.value == "failed"
        assert len(GeneratorState) == 4

    def test_cpp_event_type_values(self) -> None:
        assert len(CppEventType) == 2


class TestPipelineTrace:
    """Tests for PipelineTrace dataclass."""

    def test_defaults(self) -> None:
        trace = PipelineTrace()
        assert trace.session_id == ""
        assert trace.outcome == ""
        assert trace.speculative_attempts == 1
        assert trace.prepare_ts == 0.0
        assert trace.llm_ttft_ms == 0.0

    def test_mutable(self) -> None:
        trace = PipelineTrace()
        trace.prepare_ts = 100.0
        trace.outcome = "completed"
        assert trace.prepare_ts == 100.0
        assert trace.outcome == "completed"

    def test_to_record_full_pipeline(self) -> None:
        trace = PipelineTrace(
            session_id="s1",
            run_id=3,
            pipeline_mode="full",
            created_at="2026-04-07 12:00:00",
            outcome="completed",
            speculative_attempts=2,
            prepare_ts=1000.0,
            turn_shift_ts=1000.8,
            begin_streaming_ts=1001.0,
            playback_started_ts=1001.05,
            pipeline_start_ts=1000.01,
            memory_done_ts=1000.06,
            context_done_ts=1000.07,
            llm_start_ts=1000.07,
            llm_first_token_ts=1000.25,
            llm_done_ts=1000.7,
            tts_start_ts=1000.7,
            tts_first_chunk_ts=1000.85,
            tts_done_ts=1001.0,
            llm_ttft_ms=180.0,
        )
        rec = trace.to_record()
        assert rec["session_id"] == "s1"
        assert rec["run_id"] == 3
        assert rec["outcome"] == "completed"
        assert rec["speculative_attempts"] == 2
        assert rec["memory_ms"] == pytest.approx(50.0, abs=1)
        assert rec["context_ms"] == pytest.approx(10.0, abs=1)
        assert rec["llm_ms"] == pytest.approx(630.0, abs=1)
        assert rec["llm_ttft_ms"] == 180.0
        assert rec["tts_ms"] == pytest.approx(300.0, abs=1)
        assert rec["tts_ttfc_ms"] == pytest.approx(150.0, abs=1)
        assert rec["prepare_to_streaming_ms"] == pytest.approx(850.0, abs=1)
        assert rec["turn_shift_to_playback_ms"] == pytest.approx(250.0, abs=1)
        assert rec["speculative_ms"] == pytest.approx(800.0, abs=1)
        assert rec["bridge_ms"] == pytest.approx(50.0, abs=1)

    def test_to_record_zero_timestamps(self) -> None:
        """Unreached stages produce 0.0 durations."""
        trace = PipelineTrace(
            session_id="s1",
            outcome="cancelled",
            prepare_ts=100.0,
            pipeline_start_ts=100.01,
            memory_done_ts=100.05,
        )
        rec = trace.to_record()
        assert rec["context_ms"] == 0.0
        assert rec["llm_ms"] == 0.0
        assert rec["tts_ms"] == 0.0
        assert rec["turn_shift_to_playback_ms"] == 0.0
        assert rec["speculative_ms"] == 0.0
        assert rec["memory_ms"] == pytest.approx(40.0, abs=1)

    def test_summary_completed(self) -> None:
        trace = PipelineTrace(
            outcome="completed",
            turn_shift_ts=100.0,
            playback_started_ts=100.12,
            prepare_ts=99.0,
            pipeline_start_ts=99.01,
            memory_done_ts=99.05,
            context_done_ts=99.06,
            llm_start_ts=99.06,
            llm_done_ts=99.7,
            tts_start_ts=99.7,
            tts_done_ts=100.0,
            begin_streaming_ts=100.0,
            tts_first_chunk_ts=99.85,
            llm_ttft_ms=150.0,
            speculative_attempts=2,
        )
        s = trace.summary()
        assert "outcome=completed" in s
        assert "ts→pb=" in s
        assert "spec=" in s
        assert "attempts=2" in s

    def test_summary_cancelled_minimal(self) -> None:
        trace = PipelineTrace(outcome="cancelled")
        s = trace.summary()
        assert "outcome=cancelled" in s
        assert "ts→pb=" not in s
