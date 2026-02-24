"""Tests for voice_pipeline.core.types."""

import pytest

from voice_pipeline.core.types import (
    CppEvent,
    CppEventType,
    PlaybackState,
    ResponseData,
    SystemMode,
    TTSResult,
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
    def test_position_none_by_default(self) -> None:
        event = CppEvent(event_type=CppEventType.PLAYBACK_STARTED)
        assert event.position_sec is None

    def test_position_with_value(self) -> None:
        event = CppEvent(
            event_type=CppEventType.PLAYBACK_STOPPED,
            position_sec=2.5,
        )
        assert event.position_sec == 2.5

    def test_playback_complete_no_position(self) -> None:
        event = CppEvent(event_type=CppEventType.PLAYBACK_COMPLETE)
        assert event.position_sec is None


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

    def test_cpp_event_type_values(self) -> None:
        assert len(CppEventType) == 4
