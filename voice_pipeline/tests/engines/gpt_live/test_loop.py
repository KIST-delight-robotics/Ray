"""Tests for voice_pipeline.engines.gpt_live.loop (GPT-Live 엔진 세션 루프)."""

from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from voice_pipeline.adapters.audio_input import AudioInput
from voice_pipeline.adapters.cpp_bridge import CppBridge, CppEvent, CppEventType
from voice_pipeline.adapters.gpt_live import (
    GPTLiveSession,
    LiveAudio,
    LiveClosed,
    LiveFunctionCall,
    LiveResponseDone,
    LiveStarted,
    LiveTranscript,
)
from voice_pipeline.adapters.led import LEDState
from voice_pipeline.engines.gpt_live.dj import Segment, SegmentPicker
from voice_pipeline.engines.gpt_live.instructions import (
    DEFAULT_BACKEND_INSTRUCTIONS,
    DJ_BEGIN_INSTRUCTIONS,
    DJ_ENDED_INSTRUCTIONS,
    DJ_RETURN_INSTRUCTIONS,
    DJ_SONG_ENDED_INSTRUCTIONS,
    FAREWELL_COMMENTARY,
    SONG_BACKEND_INSTRUCTIONS,
    SONG_ENDED_INSTRUCTIONS,
    SONG_PLAYING_INSTRUCTIONS,
    build_segment_thinking,
)
from voice_pipeline.engines.gpt_live.loop import LiveSessionLoop, Phase, SongState
from voice_pipeline.engines.gpt_live.songs import Song
from voice_pipeline.engines.gpt_live.tools import (
    END_CONVERSATION_TOOL,
    PLAY_DJ_SONG_TOOL,
    PLAY_SONG_TOOL,
    SEARCH_MEMORY_TOOL,
    START_DJ_TOOL,
    STOP_DJ_TOOL,
    STOP_SONG_TOOL,
)
from voice_pipeline.settings import BRIDGE_SAMPLE_RATE, SAMPLE_RATE

SILENCE_100MS = bytes(BRIDGE_SAMPLE_RATE * 2 // 10)
VOICE_100MS = b"\x10\x27" * (BRIDGE_SAMPLE_RATE // 10)  # 10000 값 반복


def _make_loop(
    monkeypatch: pytest.MonkeyPatch,
    *,
    live_events: list[object] | None = None,
    frames: int = 0,
    end_min_wait: float = 0.0,
    end_silence: float = 0.05,
    end_max_wait: float = 1.0,
    end_nudge_after: float = 30.0,
    session_timeout: float = 30.0,
    tool_handlers: dict | None = None,
    wait_for_playback: bool = False,
    executor: ThreadPoolExecutor | None = None,
    song_catalog: dict[str, Song] | None = None,
    dj_picker: SegmentPicker | None = None,
    dj_quiet: float = 30.0,
) -> tuple[LiveSessionLoop, dict[str, MagicMock], queue.Queue]:
    """Live 세션·브리지·히스토리를 모킹한 루프. ``live_events`` 는 poll_event 가 순서대로 돌려준다."""
    monkeypatch.setattr(LiveSessionLoop, "_FRAME_TIMEOUT_SEC", 0.005)
    monkeypatch.setattr(LiveSessionLoop, "_SONG_SILENCE_SEC", 0.03)
    monkeypatch.setattr(LiveSessionLoop, "_SONG_MAX_WAIT_SEC", 1.0)
    monkeypatch.setattr(LiveSessionLoop, "_DJ_SILENCE_SEC", 0.03)
    monkeypatch.setattr(LiveSessionLoop, "_DJ_MAX_WAIT_SEC", 1.0)
    monkeypatch.setattr(LiveSessionLoop, "_DJ_QUIET_SEC", dj_quiet)
    monkeypatch.setattr(LiveSessionLoop, "_END_MIN_WAIT_SEC", end_min_wait)
    monkeypatch.setattr(LiveSessionLoop, "_END_SILENCE_SEC", end_silence)
    monkeypatch.setattr(LiveSessionLoop, "_END_MAX_WAIT_SEC", end_max_wait)
    monkeypatch.setattr(LiveSessionLoop, "_END_NUDGE_AFTER_SEC", end_nudge_after)
    monkeypatch.setattr(LiveSessionLoop, "_SESSION_TIMEOUT_SEC", session_timeout)
    monkeypatch.setattr(LiveSessionLoop, "_AUDIO_STARVATION_TIMEOUT_SEC", 30.0)

    live = MagicMock(spec=GPTLiveSession)
    live.start.return_value = LiveStarted(session_id="live_x", expires_at=time.time() + 7200)
    pending = list(live_events or [])
    live.poll_event.side_effect = lambda: pending.pop(0) if pending else None

    bridge = MagicMock(spec=CppBridge)
    bridge.poll_event.return_value = None
    history = MagicMock()
    led = MagicMock()
    memory = MagicMock()
    audio_input = MagicMock(spec=AudioInput)

    audio_queue: queue.Queue = queue.Queue()
    for _ in range(frames):
        audio_queue.put(bytes(SAMPLE_RATE * 2 * 30 // 1000))  # 30 ms 16k 프레임

    loop = LiveSessionLoop(
        live=live,
        cpp_bridge=bridge,
        history=history,
        led=led,
        audio_queue=audio_queue,
        memory_storage=memory,
        session_id="sess-1",
        token_counter=len,
        tool_handlers=tool_handlers,
        executor=executor,
        wait_for_playback_complete=wait_for_playback,
        song_catalog=song_catalog,
        dj_picker=dj_picker,
        audio_input=audio_input,
    )
    mocks = {"live": live, "bridge": bridge, "history": history, "led": led, "memory": memory}
    mocks["audio_input"] = audio_input
    return loop, mocks, audio_queue


def _seg(speaker: str, text: str, *, closed: bool, start_ms: int = 0, end_ms: int = 1000) -> LiveTranscript:
    return LiveTranscript(f"seg-{start_ms}", speaker, text, start_ms, end_ms, closed=closed)


# ---------------------------------------------------------------------------
# Start / stream / audio
# ---------------------------------------------------------------------------


class TestStartAndAudio:
    def test_start_opens_single_live_stream(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(monkeypatch, live_events=[LiveClosed("close_requested")])
        loop.run()
        m["live"].start.assert_called_once()
        m["bridge"].send_stream_start.assert_called_once_with(live=True)
        m["led"].set_state.assert_called_with(LEDState.IDLE)

    def test_mic_frames_are_resampled_and_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(monkeypatch, live_events=[LiveClosed("close_requested")], frames=2)
        loop.run()
        sent = b"".join(c.args[0] for c in m["live"].send_audio.call_args_list)
        # 16k 30ms ×2 = 960 샘플 → 24k 는 1440 샘플 = 2880 바이트
        assert len(sent) == 960 * BRIDGE_SAMPLE_RATE // SAMPLE_RATE * 2

    def test_output_audio_including_silence_goes_to_bridge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(
            monkeypatch,
            live_events=[LiveAudio(VOICE_100MS), LiveAudio(SILENCE_100MS), LiveClosed("close_requested")],
        )
        loop.run()
        assert [c.args[0] for c in m["bridge"].send_audio.call_args_list] == [VOICE_100MS, SILENCE_100MS]

    def test_close_sends_audio_end_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(monkeypatch, live_events=[LiveClosed("expired")])
        loop.run()
        m["bridge"].send_audio_end.assert_called_once()
        m["live"].close.assert_called_once_with(graceful=True)
        assert loop.exit_reason == "live_closed:expired"


# ---------------------------------------------------------------------------
# Transcripts → history / utterances
# ---------------------------------------------------------------------------


class TestTranscripts:
    def test_closed_segments_are_stored_open_ones_are_not(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [
            _seg("user", "오늘 날씨", closed=False),
            _seg("user", "오늘 날씨 어때", closed=True),
            _seg("assistant", "맑아.", closed=True, start_ms=2000),
            LiveClosed("close_requested"),
        ]
        loop, m, _ = _make_loop(monkeypatch, live_events=events)
        loop.run()
        m["history"].add_user_message.assert_called_once_with("오늘 날씨 어때")
        m["history"].add_assistant_message.assert_called_once_with("맑아.")
        roles = [c.args[1] for c in m["memory"].add_utterance.call_args_list]
        assert roles == ["user", "assistant"]

    def test_empty_closed_segment_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(monkeypatch, live_events=[_seg("user", "  ", closed=True), LiveClosed("x")])
        loop.run()
        m["history"].add_user_message.assert_not_called()


# ---------------------------------------------------------------------------
# Ending
# ---------------------------------------------------------------------------


class TestEnding:
    def test_exit_keyword_mutes_then_closes_after_silence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(monkeypatch, live_events=[_seg("user", "그래, 잘 가!", closed=False)])
        loop.run()
        m["live"].mute_input.assert_called_once()
        m["live"].close.assert_called_once_with(graceful=True)
        m["bridge"].send_audio_end.assert_called_once()
        assert loop.exit_reason == "exit_keyword"

    def test_greeting_does_not_trigger_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(monkeypatch, live_events=[_seg("user", "안녕 레이!", closed=True), LiveClosed("x")])
        loop.run()
        m["live"].mute_input.assert_not_called()

    def test_ending_waits_while_model_still_speaking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 종료 결정 뒤 음성 조각이 계속 오면 닫지 않고, 무음이 이어지면 닫는다
        loop, m, _ = _make_loop(
            monkeypatch,
            live_events=[_seg("user", "이제 갈게", closed=False)] + [LiveAudio(VOICE_100MS)] * 3,
            end_silence=0.1,
            end_max_wait=5.0,
        )
        t0 = time.monotonic()
        loop.run()
        assert time.monotonic() - t0 >= 0.1
        assert loop.exit_reason == "exit_keyword"

    def test_ending_max_wait_forces_close(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pending = [_seg("user", "goodbye", closed=False)]
        loop, m, _ = _make_loop(monkeypatch, live_events=pending, end_silence=100.0, end_max_wait=0.1)
        # 무음 판정이 절대 안 나도록 매 폴마다 음성 조각을 준다
        m["live"].poll_event.side_effect = lambda: pending.pop(0) if pending else LiveAudio(VOICE_100MS)
        t0 = time.monotonic()
        loop.run()
        assert 0.1 <= time.monotonic() - t0 < 2.0
        m["live"].close.assert_called_once()

    def test_end_conversation_tool_submits_output_without_continue(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [
            LiveFunctionCall(call_id="call_1", name=END_CONVERSATION_TOOL, arguments="{}", delegation_id="d1"),
            LiveResponseDone(delegation_id="d1"),
        ]
        loop, m, _ = _make_loop(monkeypatch, live_events=events)
        loop.run()
        m["live"].submit_function_output.assert_called_once()
        assert m["live"].submit_function_output.call_args.args[0] == "call_1"
        m["live"].continue_response.assert_not_called()
        m["live"].mute_input.assert_called_once()
        assert loop.exit_reason == "end_conversation_tool"

    def test_silent_ending_nudges_a_goodbye_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [
            LiveFunctionCall(call_id="call_1", name=END_CONVERSATION_TOOL, arguments="{}", delegation_id="d1"),
            LiveResponseDone(delegation_id="d1"),
        ]
        loop, m, _ = _make_loop(
            monkeypatch, live_events=events, end_min_wait=0.3, end_silence=0.05, end_nudge_after=0.05
        )
        loop.run()
        m["live"].append_commentary.assert_called_once_with(FAREWELL_COMMENTARY)
        assert loop.exit_reason == "end_conversation_tool"

    def test_ending_with_goodbye_in_progress_does_not_nudge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(monkeypatch, end_min_wait=0.3, end_silence=0.05, end_nudge_after=0.05)
        ending = lambda: loop._phase is Phase.ENDING  # noqa: E731
        _staged_live_events(
            m,
            [
                (None, LiveFunctionCall("call_1", END_CONVERSATION_TOOL, "{}", "d1")),
                (None, LiveResponseDone("d1")),
                (ending, LiveAudio(VOICE_100MS)),  # 위임과 동시에 인사 중
            ],
        )
        loop.run()
        m["live"].append_commentary.assert_not_called()

    def test_other_tool_runs_handler_and_continues(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [
            LiveFunctionCall(call_id="c9", name="get_battery_level", arguments="{}", delegation_id="d1"),
            LiveResponseDone(delegation_id="d1"),
            LiveClosed("close_requested"),
        ]
        handlers = {"get_battery_level": lambda _a: json.dumps({"percent": 73})}
        loop, m, _ = _make_loop(monkeypatch, live_events=events, tool_handlers=handlers)
        loop.run()
        m["live"].submit_function_output.assert_called_once_with("c9", json.dumps({"percent": 73}))
        m["live"].continue_response.assert_called_once()
        m["live"].mute_input.assert_not_called()

    def test_unknown_tool_returns_error_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [
            LiveFunctionCall(call_id="c1", name="nope", arguments="{}", delegation_id=None),
            LiveResponseDone(delegation_id=None),
            LiveClosed("x"),
        ]
        loop, m, _ = _make_loop(monkeypatch, live_events=events)
        loop.run()
        output = json.loads(m["live"].submit_function_output.call_args.args[1])
        assert "error" in output

    def test_idle_timeout_ends_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(monkeypatch, session_timeout=0.05)
        loop.run()
        assert loop.exit_reason == "idle_timeout"
        m["live"].mute_input.assert_called_once()

    def test_request_stop_ends_gracefully(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(monkeypatch)
        loop.request_stop()
        loop.run()
        assert loop.exit_reason == "stop_requested"
        m["live"].close.assert_called_once_with(graceful=True)

    def test_bridge_playback_complete_before_audio_end_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(monkeypatch)
        m["bridge"].poll_event.side_effect = [CppEvent(CppEventType.PLAYBACK_COMPLETE), None]
        loop.run()
        assert loop.exit_reason == "bridge_stream_ended"
        m["bridge"].send_audio_end.assert_not_called()  # 스트림은 이미 C++ 가 끝냈다

    def test_bridge_error_closes_live_and_reraises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(monkeypatch, live_events=[LiveAudio(VOICE_100MS)])
        m["bridge"].send_audio.side_effect = RuntimeError("Connection lost")
        with pytest.raises(RuntimeError):
            loop.run()
        m["live"].close.assert_called_once_with(graceful=False)
        assert loop.exit_reason == "error"


class TestHelpers:
    def test_phase_starts_active(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, _, _ = _make_loop(monkeypatch)
        assert loop._phase == Phase.ACTIVE

    def test_resample_ratio(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, _, _ = _make_loop(monkeypatch)
        out = loop._resample(bytes(480 * 2))
        assert len(out) == 720 * 2


class TestSilencePadding:
    """서버 출력이 실시간보다 짧으면 무음 조각으로 채우고, 앞서면 무음 조각을 버린다. 말소리는 건드리지 않는다."""

    def _prime(self, loop: LiveSessionLoop, *, behind_sec: float) -> None:
        # 첫 조각이 behind_sec 전에 왔고 그동안 보낸 오디오는 0 → lead = -behind_sec
        loop._stream_open = True  # run() 없이 부르므로 스트림이 열린 상태를 명시 (닫혀 있으면 조각을 버린다)
        loop._first_audio_time = time.monotonic() - behind_sec
        loop._audio_sent_sec = 0.0

    def test_pads_after_silent_chunk_when_behind(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(monkeypatch)
        self._prime(loop, behind_sec=1.0)
        loop._on_audio(SILENCE_100MS)
        sent = [c.args[0] for c in m["bridge"].send_audio.call_args_list]
        assert sent[0] == SILENCE_100MS
        assert all(chunk == SILENCE_100MS for chunk in sent)
        # 0.1 s 조각을 보낸 뒤 lead -0.9 → 0 까지 0.1 s 씩 채움 = 9개 추가
        assert 8 <= len(sent) - 1 <= 10
        assert loop._padded_sec > 0.5

    def test_does_not_pad_after_voice_chunk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(monkeypatch)
        self._prime(loop, behind_sec=1.0)
        loop._on_audio(VOICE_100MS)
        assert [c.args[0] for c in m["bridge"].send_audio.call_args_list] == [VOICE_100MS]
        assert loop._padded_sec == 0.0

    def test_no_pad_within_band(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(monkeypatch)
        self._prime(loop, behind_sec=0.05)  # 조각 보낸 뒤 lead +0.05 → 채우지 않음
        loop._on_audio(SILENCE_100MS)
        assert m["bridge"].send_audio.call_count == 1

    def test_pads_one_chunk_at_quiet_gap_when_urgent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """말 사이 쉼(rms<30, 정확히 0 아님)에서는 lead 가 -0.3 아래일 때 한 조각만 채운다."""
        quiet = b"\x02\x00" * (BRIDGE_SAMPLE_RATE // 10)  # 값 2 → rms 2
        loop, m, _ = _make_loop(monkeypatch)
        self._prime(loop, behind_sec=1.0)
        loop._on_audio(quiet)
        sent = [c.args[0] for c in m["bridge"].send_audio.call_args_list]
        assert sent == [quiet, SILENCE_100MS]

    def test_quiet_gap_not_padded_when_lead_mildly_behind(self, monkeypatch: pytest.MonkeyPatch) -> None:
        quiet = b"\x02\x00" * (BRIDGE_SAMPLE_RATE // 10)
        loop, m, _ = _make_loop(monkeypatch)
        self._prime(loop, behind_sec=0.25)  # 조각 보낸 뒤 lead -0.15: 완전 무음이면 채우지만 쉼에서는 안 채움
        loop._on_audio(quiet)
        assert m["bridge"].send_audio.call_count == 1

    def test_quiet_gap_never_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        quiet = b"\x02\x00" * (BRIDGE_SAMPLE_RATE // 10)
        loop, m, _ = _make_loop(monkeypatch)
        loop._stream_open = True
        loop._first_audio_time = time.monotonic()
        loop._audio_sent_sec = 1.0
        loop._on_audio(quiet)
        assert m["bridge"].send_audio.call_count == 1

    def test_drops_silent_chunk_when_ahead_but_keeps_voice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(monkeypatch)
        loop._stream_open = True
        loop._first_audio_time = time.monotonic()
        loop._audio_sent_sec = 1.0  # 1 s 앞서 있음
        loop._on_audio(SILENCE_100MS)
        loop._on_audio(VOICE_100MS)
        assert [c.args[0] for c in m["bridge"].send_audio.call_args_list] == [VOICE_100MS]
        assert loop._dropped_sec == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# 인사 WAV 와 연결 겹치기 / LED
# ---------------------------------------------------------------------------


class TestGreetingOverlap:
    def test_stream_start_waits_for_greeting_playback_complete(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 프레임 1: 출력 조각이 오지만 WAV 재생 중 → 버림. 그 프레임의 브리지 이벤트로 playback_complete → stream_start.
        # 프레임 3: 이후 조각은 스트림으로 전달.
        loop, m, _ = _make_loop(monkeypatch, wait_for_playback=True)
        frames = [[LiveAudio(SILENCE_100MS)], [], [LiveAudio(VOICE_100MS), LiveClosed("close_requested")]]

        def poll_live():
            if not frames:
                return None
            if frames[0]:
                return frames[0].pop(0)
            frames.pop(0)
            return None

        m["live"].poll_event.side_effect = poll_live
        bridge_pending = [CppEvent(CppEventType.PLAYBACK_COMPLETE)]
        m["bridge"].poll_event.side_effect = lambda: bridge_pending.pop(0) if bridge_pending else None

        loop.run()

        m["bridge"].send_stream_start.assert_called_once_with(live=True)
        assert [c.args[0] for c in m["bridge"].send_audio.call_args_list] == [VOICE_100MS]
        assert loop._discarded_before_stream_sec == pytest.approx(0.1)
        names = [c[0] for c in m["bridge"].mock_calls]
        assert names.index("send_stream_start") < names.index("send_audio")
        m["bridge"].send_audio_end.assert_called_once()
        assert loop.exit_reason == "live_closed:close_requested"

    def test_stream_start_falls_back_after_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(LiveSessionLoop, "_STREAM_START_MAX_WAIT_SEC", 0.0)
        loop, m, _ = _make_loop(monkeypatch, wait_for_playback=True)
        frames = [[], [LiveClosed("close_requested")]]

        def poll_live():
            if not frames:
                return None
            if frames[0]:
                return frames[0].pop(0)
            frames.pop(0)
            return None

        m["live"].poll_event.side_effect = poll_live
        loop.run()
        m["bridge"].send_stream_start.assert_called_once_with(live=True)
        m["bridge"].send_audio_end.assert_called_once()

    def test_led_idle_on_connect_and_sleeping_when_ending(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(monkeypatch, live_events=[_seg("user", "그래, 잘 가!", closed=False)])
        loop.run()
        states = [c.args[0] for c in m["led"].set_state.call_args_list]
        assert states == [LEDState.IDLE, LEDState.SLEEPING]


# ---------------------------------------------------------------------------
# Tool handlers on an executor
# ---------------------------------------------------------------------------


class TestAsyncTools:
    def test_handler_runs_on_executor_and_result_is_submitted_later(self, monkeypatch: pytest.MonkeyPatch) -> None:
        release = threading.Event()
        loop_thread: list[str] = []

        def slow_handler(_args: str) -> str:
            loop_thread.append(threading.current_thread().name)
            release.wait(2.0)
            return json.dumps({"ok": True})

        events = [
            LiveFunctionCall(call_id="c1", name="slow", arguments="{}", delegation_id="d1"),
            LiveResponseDone(delegation_id="d1"),
        ]
        with ThreadPoolExecutor(max_workers=1) as executor:
            loop, m, _ = _make_loop(monkeypatch, live_events=events, tool_handlers={"slow": slow_handler})
            loop._executor = executor
            loop._start_session()
            for _ in range(5):  # 핸들러가 막혀 있는 동안 프레임 루프는 계속 돈다
                assert loop._run_frame() is False
            m["live"].submit_function_output.assert_not_called()
            release.set()
            deadline = time.monotonic() + 2.0
            while not m["live"].submit_function_output.called and time.monotonic() < deadline:
                loop._run_frame()
            m["live"].submit_function_output.assert_called_once_with("c1", json.dumps({"ok": True}))
            m["live"].continue_response.assert_called_once()
            assert loop_thread and loop_thread[0] != threading.current_thread().name
            loop._finish(graceful=True)

    def test_result_arriving_during_ending_is_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        release = threading.Event()
        events = [
            LiveFunctionCall(call_id="c1", name="slow", arguments="{}", delegation_id="d1"),
            LiveResponseDone(delegation_id="d1"),
        ]
        with ThreadPoolExecutor(max_workers=1) as executor:
            loop, m, _ = _make_loop(monkeypatch, live_events=events, tool_handlers={"slow": lambda _a: _wait(release)})
            loop._executor = executor
            loop._start_session()
            loop._run_frame()
            loop.request_stop()
            loop._run_frame()
            assert loop._phase is Phase.ENDING
            release.set()
            deadline = time.monotonic() + 2.0
            while loop._inflight and time.monotonic() < deadline:
                loop._run_frame()
            m["live"].submit_function_output.assert_not_called()
            m["live"].continue_response.assert_not_called()
            loop._finish(graceful=True)


def _wait(release: threading.Event) -> str:
    release.wait(2.0)
    return "{}"


# ---------------------------------------------------------------------------
# Memory: instructions builder + search handler
# ---------------------------------------------------------------------------


class TestToolFailure:
    def test_search_tool_failure_becomes_error_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(_a: str) -> str:
            raise RuntimeError("index down")

        events = [
            LiveFunctionCall(call_id="c1", name=SEARCH_MEMORY_TOOL, arguments='{"query": "x"}', delegation_id="d1"),
            LiveResponseDone(delegation_id="d1"),
            LiveClosed("x"),
        ]
        loop, m, _ = _make_loop(monkeypatch, live_events=events, tool_handlers={SEARCH_MEMORY_TOOL: boom})
        loop.run()
        output = json.loads(m["live"].submit_function_output.call_args.args[1])
        assert output == {"error": "index down"}
        m["live"].continue_response.assert_called_once()


# ---------------------------------------------------------------------------
# 노래 재생 (play_song / stop_song)
# ---------------------------------------------------------------------------

SONGS = {"IAM": Song(key="IAM", title="I AM", artist="IVE", aliases=("아이엠",))}
PLAYBACK_COMPLETE = CppEvent(event_type=CppEventType.PLAYBACK_COMPLETE)


def _call(name: str, args: dict | None = None, call_id: str = "c1") -> LiveFunctionCall:
    return LiveFunctionCall(call_id, name, json.dumps(args or {}), None)


def _wire_bridge_completion(bridge: MagicMock) -> queue.Queue:
    """C++ 흉내: audio_end 로 스트림이 닫히거나 stop 으로 노래가 끊기면 playback_complete 를 돌려준다.

    돌려주는 큐에 PLAYBACK_COMPLETE 를 넣으면 노래의 자연 종료를 흉내 낼 수 있다.
    """
    events: queue.Queue = queue.Queue()
    bridge.poll_event.side_effect = lambda: events.get_nowait() if not events.empty() else None
    bridge.send_audio_end.side_effect = lambda: events.put(PLAYBACK_COMPLETE)
    bridge.send_stop.side_effect = lambda: events.put(PLAYBACK_COMPLETE)
    return events


def _staged_live_events(m: dict[str, MagicMock], stages: list) -> None:
    """``stages`` = [(조건 함수 | None, 이벤트)] — 조건이 참일 때 순서대로 하나씩 내보낸다."""
    pending = list(stages)

    def poll():
        if not pending:
            return None
        cond, event = pending[0]
        if cond is None or cond():
            pending.pop(0)
            return event
        return None

    m["live"].poll_event.side_effect = poll


class TestSongPlayback:
    def test_full_sequence_play_then_stop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(monkeypatch, song_catalog=SONGS)
        _wire_bridge_completion(m["bridge"])
        playing = lambda: loop._song_state is SongState.PLAYING  # noqa: E731
        # 정지 요청은 play_song 결과가 제출된 뒤(실제로는 노래 도중 수십 초 뒤)에 온다
        submitted = lambda: playing() and m["live"].submit_function_output.called  # noqa: E731
        # 정지 결과 제출까지 끝난 뒤에 세션이 닫힌다 (실제로는 한참 뒤)
        reopened = lambda: loop._song_state is SongState.NONE and m["live"].continue_response.call_count == 2  # noqa: E731
        _staged_live_events(
            m,
            [
                (None, _call(PLAY_SONG_TOOL, {"song": "IAM"})),
                (None, LiveResponseDone(None)),
                (None, LiveAudio(VOICE_100MS)),  # 위임 시 한마디 — 끝나길 기다린 뒤 스트림을 닫는다
                (playing, LiveAudio(VOICE_100MS)),  # 재생 중 모델 음성 — 나갈 스트림이 없어 버린다
                (playing, _seg("user", "옆사람 잡담", closed=True, start_ms=5000)),
                (submitted, _call(STOP_SONG_TOOL, call_id="c2")),
                (submitted, LiveResponseDone(None)),
                (reopened, LiveClosed("close_requested")),
            ],
        )
        order = MagicMock()  # 브리지·세션 호출 순서를 한 곳에 기록
        order.attach_mock(m["bridge"], "bridge")
        order.attach_mock(m["live"], "live")
        loop.run()

        live, bridge, ai = m["live"], m["bridge"], m["audio_input"]
        names = [c[0] for c in order.mock_calls]
        # 1) 결과는 노래가 시작된 뒤에 제출된다 (play_audio_csv 가 submit 보다 먼저)
        outputs = {c.args[0]: json.loads(c.args[1]) for c in live.submit_function_output.call_args_list}
        assert outputs["c1"] == {"status": "playing", "song": "IAM", "title": "I AM", "artist": "IVE"}
        assert outputs["c2"] == {"status": "stopped"}
        assert live.continue_response.call_count == 2  # 정지 결과 뒤에도 이어간다 — 백엔드가 "멈췼다" 를 답한다
        assert names.index("bridge.send_play_audio_csv") < names.index("live.submit_function_output")
        # 정지도 같다: stop 은 즉시, 결과 제출은 스트림을 다시 연 뒤 (위임 시 "끌게" 는 닫힌 스트림에서 버려진다)
        submits = [i for i, n in enumerate(names) if n == "live.submit_function_output"]
        assert names.index("bridge.send_stop") < submits[1]
        assert names.index("bridge.send_stream_start", names.index("bridge.send_stop")) < submits[1]
        # 2) 위임 시 한마디는 스트림으로 나갔고, 그 뒤 스트림을 닫았다 (audio_end 는 send_audio 뒤)
        assert [c.args[0] for c in bridge.send_audio.call_args_list] == [VOICE_100MS]
        assert names.index("bridge.send_audio") < names.index("bridge.send_audio_end")
        # 3) 배수 완료 → 원본 채널·재생 지시 → play_audio_csv. 백엔드는 요청 수락 시 교체, 끝나면 복원
        bridge.send_play_audio_csv.assert_called_once_with("IAM")
        live.update_backend.assert_called_once()
        assert live.update_backend.call_args.args[0] == SONG_BACKEND_INSTRUCTIONS
        assert [t["name"] for t in live.update_backend.call_args.args[1]] == ["stop_song"]
        live.restore_backend.assert_called_once()
        assert names.index("live.update_backend") < names.index("bridge.send_audio_end")
        continues = [i for i, n in enumerate(names) if n == "live.continue_response"]
        assert continues[1] < names.index("live.restore_backend") < names.index("live.close")  # 정지 후속 응답 뒤 복원
        assert [c.args[0] for c in ai.set_raw_capture.call_args_list][:2] == [True, False]
        appended = [
            c.args[0].split("\nCurrent transfer:")[0]
            for c in live.append_instructions.call_args_list
            if "is cancelled." not in c.args[0]
        ]
        assert appended == [SONG_PLAYING_INSTRUCTIONS, SONG_ENDED_INSTRUCTIONS]
        live.append_thinking.assert_not_called()
        # 4) 재생 중 전사는 기록하지 않는다
        m["history"].add_user_message.assert_not_called()
        m["memory"].add_utterance.assert_not_called()
        # 5) stop → playback_complete → 스트림 재개 (시작 1회 + 재개 1회)
        bridge.send_stop.assert_called_once()
        assert bridge.send_stream_start.call_count == 2
        assert loop.exit_reason == "live_closed:close_requested"

    def test_lead_accounting_restarts_when_stream_reopens_after_song(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 노래 전에 보낸 출력 누적이 남아 있으면 재개 뒤 lead 가 수 분으로 계산돼 무음 조각을 전부 버렸다 (2026-09-28)
        loop, m, _ = _make_loop(monkeypatch, song_catalog=SONGS)
        _wire_bridge_completion(m["bridge"])
        playing = lambda: loop._song_state is SongState.PLAYING  # noqa: E731
        submitted = lambda: playing() and m["live"].submit_function_output.called  # noqa: E731
        reopened = lambda: loop._song_state is SongState.NONE and m["bridge"].send_stream_start.call_count == 2  # noqa: E731
        after_silence = lambda: reopened() and m["bridge"].send_audio.call_count >= 7  # noqa: E731
        _staged_live_events(
            m,
            [
                *[(None, LiveAudio(VOICE_100MS)) for _ in range(6)],  # 노래 전 대화 출력 0.6 s (버림 문턱 0.3 s 초과)
                (None, _call(PLAY_SONG_TOOL, {"song": "IAM"})),
                (None, LiveResponseDone(None)),
                (submitted, _call(STOP_SONG_TOOL, call_id="c2")),
                (submitted, LiveResponseDone(None)),
                (reopened, LiveAudio(SILENCE_100MS)),  # 재개 직후 무음 조각 — 버리지 않고 보내야 한다
                (after_silence, LiveClosed("x")),
            ],
        )
        loop.run()
        sent = [c.args[0] for c in m["bridge"].send_audio.call_args_list]
        assert sent[:6] == [VOICE_100MS] * 6
        assert SILENCE_100MS in sent[6:]
        assert loop._dropped_sec == 0.0

    def test_unknown_song_is_reported_without_side_effects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [_call(PLAY_SONG_TOOL, {"song": "Nope"}), LiveResponseDone(None), LiveClosed("x")]
        loop, m, _ = _make_loop(monkeypatch, live_events=events, song_catalog=SONGS)
        loop.run()
        out = json.loads(m["live"].submit_function_output.call_args.args[1])
        assert "unknown song" in out["error"]
        m["live"].continue_response.assert_called_once()
        m["bridge"].send_play_audio_csv.assert_not_called()
        assert m["bridge"].send_audio_end.call_count == 1  # 종료 시 한 번만

    def test_stop_when_nothing_plays_is_a_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [_call(STOP_SONG_TOOL), LiveResponseDone(None), LiveClosed("x")]
        loop, m, _ = _make_loop(monkeypatch, live_events=events, song_catalog=SONGS)
        loop.run()
        assert json.loads(m["live"].submit_function_output.call_args.args[1]) == {"status": "not_playing"}
        m["bridge"].send_stop.assert_not_called()

    def test_idle_timeout_is_suspended_while_playing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(monkeypatch, song_catalog=SONGS, session_timeout=0.15)
        _wire_bridge_completion(m["bridge"])
        started = {"t": 0.0}

        def playing_long_enough() -> bool:
            if loop._song_state is not SongState.PLAYING:
                return False
            started["t"] = started["t"] or time.monotonic()
            return time.monotonic() - started["t"] > 0.4  # 타임아웃(0.15 s)보다 오래 재생 중

        _staged_live_events(
            m,
            [
                (None, _call(PLAY_SONG_TOOL, {"song": "IAM"})),
                (None, LiveResponseDone(None)),
                (playing_long_enough, LiveClosed("close_requested")),
            ],
        )
        loop.run()
        assert loop.exit_reason == "live_closed:close_requested"  # idle_timeout 이 아니다
        m["live"].mute_input.assert_not_called()

    def test_exit_keyword_is_ignored_while_playing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(monkeypatch, song_catalog=SONGS)
        _wire_bridge_completion(m["bridge"])
        playing = lambda: loop._song_state is SongState.PLAYING  # noqa: E731
        _staged_live_events(
            m,
            [
                (None, _call(PLAY_SONG_TOOL, {"song": "IAM"})),
                (None, LiveResponseDone(None)),
                (playing, _seg("user", "레이 잘 가", closed=False, start_ms=7000)),
                (playing, LiveClosed("close_requested")),
            ],
        )
        loop.run()
        m["live"].mute_input.assert_not_called()  # 종료 시퀀스에 들어가지 않았다
        assert loop.exit_reason == "live_closed:close_requested"

    def test_stop_waits_for_acknowledgement_to_end_before_reopening(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 정지 위임 시 모델이 "끌게" 를 말한다 — 그 음성은 닫힌 스트림에서 버려지고, 끝난 뒤에야 스트림을 다시 연다
        loop, m, _ = _make_loop(monkeypatch, song_catalog=SONGS)
        monkeypatch.setattr(LiveSessionLoop, "_SONG_SILENCE_SEC", 0.15)
        _wire_bridge_completion(m["bridge"])
        playing = lambda: loop._song_state is SongState.PLAYING  # noqa: E731
        submitted = lambda: playing() and m["live"].submit_function_output.called  # noqa: E731
        stopping = lambda: loop._song_state is SongState.STOPPING  # noqa: E731
        done = lambda: loop._song_state is SongState.NONE and m["live"].continue_response.call_count == 2  # noqa: E731
        _staged_live_events(
            m,
            [
                (None, _call(PLAY_SONG_TOOL, {"song": "IAM"})),
                (None, LiveResponseDone(None)),
                (submitted, _call(STOP_SONG_TOOL, call_id="c2")),
                (submitted, LiveResponseDone(None)),
                (stopping, LiveAudio(VOICE_100MS)),  # "응, 끌게" — 스트림 닫힘, 버려져야 한다
                (stopping, LiveAudio(VOICE_100MS)),
                (done, LiveClosed("close_requested")),
            ],
        )
        order = MagicMock()
        order.attach_mock(m["bridge"], "bridge")
        order.attach_mock(m["live"], "live")
        t0 = time.monotonic()
        loop.run()
        names = [c[0] for c in order.mock_calls]
        m["bridge"].send_audio.assert_not_called()  # 인사말은 소리로 나가지 않았다
        stop_i = names.index("bridge.send_stop")
        reopen_i = names.index("bridge.send_stream_start", stop_i)
        assert reopen_i > stop_i
        assert time.monotonic() - t0 >= 0.15  # 무음 판정을 기다렸다
        assert loop.exit_reason == "live_closed:close_requested"

    def test_end_conversation_while_playing_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(monkeypatch, song_catalog=SONGS)
        _wire_bridge_completion(m["bridge"])
        playing = lambda: loop._song_state is SongState.PLAYING  # noqa: E731
        _staged_live_events(
            m,
            [
                (None, _call(PLAY_SONG_TOOL, {"song": "IAM"})),
                (None, LiveResponseDone(None)),
                (playing, _call(END_CONVERSATION_TOOL, call_id="c2")),
                (playing, LiveResponseDone(None)),
                (playing, LiveClosed("close_requested")),
            ],
        )
        loop.run()
        outputs = {c.args[0]: json.loads(c.args[1]) for c in m["live"].submit_function_output.call_args_list}
        assert "only stop_song" in outputs["c2"]["error"]
        m["live"].mute_input.assert_not_called()
        assert loop.exit_reason == "live_closed:close_requested"

    def test_request_stop_while_playing_reopens_stream_and_ends_normally(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(monkeypatch, song_catalog=SONGS)
        _wire_bridge_completion(m["bridge"])

        def stop_when_playing() -> bool:
            if loop._song_state is SongState.PLAYING:
                loop.request_stop()
            return False

        _staged_live_events(
            m,
            [
                (None, _call(PLAY_SONG_TOOL, {"song": "IAM"})),
                (None, LiveResponseDone(None)),
                (stop_when_playing, LiveClosed("never")),
            ],
        )
        loop.run()
        m["bridge"].send_stop.assert_called_once()
        assert m["bridge"].send_stream_start.call_count == 2  # 끊긴 노래의 playback_complete 로 스트림 재개
        assert m["bridge"].send_audio_end.call_count == 2  # 노래 전 1회 + 정상 종료 1회 (FAREWELL 이 기다리는 것)
        assert loop.exit_reason == "stop_requested"

    def test_play_song_while_playing_is_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(monkeypatch, song_catalog=SONGS)
        _wire_bridge_completion(m["bridge"])
        playing = lambda: loop._song_state is SongState.PLAYING  # noqa: E731
        _staged_live_events(
            m,
            [
                (None, _call(PLAY_SONG_TOOL, {"song": "IAM"})),
                (None, LiveResponseDone(None)),
                (playing, _call(PLAY_SONG_TOOL, {"song": "IAM"}, call_id="c2")),
                (playing, LiveResponseDone(None)),
                (playing, LiveClosed("close_requested")),
            ],
        )
        loop.run()
        outputs = {c.args[0]: json.loads(c.args[1]) for c in m["live"].submit_function_output.call_args_list}
        assert "only stop_song" in outputs["c2"]["error"]
        m["bridge"].send_play_audio_csv.assert_called_once()

    def test_ending_before_song_starts_resolves_the_pending_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 결과를 보류한 채 종료가 시작되면 Future 를 끝내 줘야 한다 (제출은 종료 중이라 버려진다)
        loop, m, _ = _make_loop(monkeypatch, song_catalog=SONGS)
        monkeypatch.setattr(LiveSessionLoop, "_SONG_SILENCE_SEC", 10.0)  # 무음 판정이 안 나게
        monkeypatch.setattr(LiveSessionLoop, "_SONG_MAX_WAIT_SEC", 10.0)
        _wire_bridge_completion(m["bridge"])

        def stop_when_pending() -> bool:
            if loop._song_state is SongState.PENDING:
                loop.request_stop()
            return False

        _staged_live_events(
            m,
            [
                (None, _call(PLAY_SONG_TOOL, {"song": "IAM"})),
                (None, LiveResponseDone(None)),
                (stop_when_pending, LiveClosed("never")),
            ],
        )
        loop.run()
        assert loop.exit_reason == "stop_requested"
        assert not loop._inflight
        m["bridge"].send_play_audio_csv.assert_not_called()
        m["live"].submit_function_output.assert_not_called()


# ---------------------------------------------------------------------------
# DJ 모드 (start_dj / play_dj_song / stop_dj)
# ---------------------------------------------------------------------------

SEG1 = Segment("drawer", "서랍 속에 남아 있는 것들", SONGS["IAM"], "- 서랍은 느리게 바뀐다")
SEG2 = Segment("rain", "비 오는 날의 소리", SONGS["IAM"], "- 빗소리")


def _picker(*segments: Segment | None) -> MagicMock:
    picker = MagicMock(spec=SegmentPicker)
    picker.pick.side_effect = list(segments) if segments else [None]
    return picker


def _outputs(live: MagicMock) -> dict[str, dict]:
    return {c.args[0]: json.loads(c.args[1]) for c in live.submit_function_output.call_args_list}


def _appended(live: MagicMock) -> list[str]:
    return [c.args[0] for c in live.append_instructions.call_args_list]


class TestDJ:
    def test_start_holds_result_pushes_segment_then_signal_then_submits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        picker = _picker(SEG1)
        loop, m, _ = _make_loop(monkeypatch, song_catalog=SONGS, dj_picker=picker)
        state: list[tuple[bool, Segment | None]] = []

        def submitted() -> bool:
            if m["live"].submit_function_output.called:
                state.append((loop._dj_active, loop._dj_segment))  # 세션 종료가 상태를 지우기 전에 잡아 둔다
                return True
            return False

        _staged_live_events(
            m,
            [
                (None, _call(START_DJ_TOOL)),
                (None, LiveResponseDone(None)),
                (None, LiveAudio(VOICE_100MS)),  # 위임 시 한마디 — 끝나길 기다린 뒤 소재를 넣는다
                (submitted, LiveClosed("x")),
            ],
        )
        order = MagicMock()
        order.attach_mock(m["live"], "live")
        loop.run()

        live = m["live"]
        assert _outputs(live)["c1"] == {"status": "started", "segment": "drawer", "song": "IAM"}
        live.continue_response.assert_called_once()
        picker.pick.assert_called_once()
        # 소재는 대화 모델(thinking)과 백엔드(Current segment 절, 툴은 그대로) 양쪽에
        live.append_thinking.assert_called_once_with(build_segment_thinking(SEG1))
        assert _appended(live) == [DJ_BEGIN_INSTRUCTIONS]
        live.update_backend.assert_called_once()
        backend = live.update_backend.call_args.args[0]
        assert backend.startswith(DEFAULT_BACKEND_INSTRUCTIONS) and "## Current segment" in backend
        assert "서랍 속에 남아 있는 것들" in backend and "Ray's voice" not in backend
        assert len(live.update_backend.call_args.args) == 1  # 툴 집합은 세션 시작 그대로
        # 순서: 소재 → 시작 신호 → 결과 제출 → 이어가기 (제출이 먼저면 모델이 소재 없이 말을 시작한다)
        names = [c[0] for c in order.mock_calls]
        assert (
            names.index("live.append_thinking")
            < names.index("live.append_instructions")
            < names.index("live.submit_function_output")
            < names.index("live.continue_response")
        )
        assert state[0] == (True, SEG1)

    def test_start_without_picker_or_segments_or_twice_is_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, _, _ = _make_loop(monkeypatch, song_catalog=SONGS)
        assert "unavailable" in json.loads(loop._submit_tool(_call(START_DJ_TOOL)).result())["error"]
        loop, _, _ = _make_loop(monkeypatch, song_catalog=SONGS, dj_picker=_picker(None))
        assert "no DJ segments" in json.loads(loop._submit_tool(_call(START_DJ_TOOL)).result())["error"]
        loop, _, _ = _make_loop(monkeypatch, song_catalog=SONGS, dj_picker=_picker(SEG1, SEG2))
        first = loop._submit_tool(_call(START_DJ_TOOL))
        assert not first.done()
        assert "already active" in json.loads(loop._submit_tool(_call(START_DJ_TOOL, call_id="c2")).result())["error"]

    def test_play_dj_song_then_natural_end_pushes_next_segment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        picker = _picker(SEG1, SEG2)
        loop, m, _ = _make_loop(monkeypatch, song_catalog=SONGS, dj_picker=picker)
        bridge_events = _wire_bridge_completion(m["bridge"])
        started = lambda: "c1" in _outputs(m["live"])  # noqa: E731
        playing = lambda: loop._song_state is SongState.PLAYING and "c2" in _outputs(m["live"])  # noqa: E731

        def song_finishes() -> bool:
            if playing():
                bridge_events.put(PLAYBACK_COMPLETE)  # 자연 종료
                return True
            return False

        state: list[tuple[bool, Segment | None]] = []

        def reopened() -> bool:
            if loop._song_state is SongState.NONE and picker.pick.call_count == 2:
                state.append((loop._dj_active, loop._dj_segment))
                return True
            return False

        _staged_live_events(
            m,
            [
                (None, _call(START_DJ_TOOL)),
                (None, LiveResponseDone(None)),
                (started, _call(PLAY_DJ_SONG_TOOL, call_id="c2")),  # 곡 소개 뒤 위임
                (started, LiveResponseDone(None)),
                (started, LiveAudio(VOICE_100MS)),
                (song_finishes, LiveAudio(SILENCE_100MS)),
                (reopened, LiveClosed("x")),
            ],
        )
        loop.run()

        live = m["live"]
        outputs = _outputs(live)
        assert outputs["c2"] == {"status": "playing", "song": "IAM", "title": "I AM", "artist": "IVE"}
        m["bridge"].send_play_audio_csv.assert_called_once_with("IAM")  # 인자 없이 현재 소재의 곡
        # 자연 종료 → 다음 소재가 양쪽에 들어가고 노래 끝 신호로 깨운다. DJ 는 계속
        assert picker.pick.call_count == 2
        assert [c.args[0] for c in live.append_thinking.call_args_list] == [
            build_segment_thinking(SEG1),
            build_segment_thinking(SEG2),
        ]
        assert _appended(live) == [DJ_BEGIN_INSTRUCTIONS, SONG_PLAYING_INSTRUCTIONS, DJ_SONG_ENDED_INSTRUCTIONS]
        assert state[0] == (True, SEG2)  # DJ 는 계속, 소재는 둘째로
        # 백엔드: 소재 1 절 → 노래용 → 소재 2 절 (기본으로의 복원은 없다)
        updates = [c.args[0] for c in live.update_backend.call_args_list]
        assert "서랍" in updates[0] and updates[1] == SONG_BACKEND_INSTRUCTIONS and "비 오는 날" in updates[2]
        live.restore_backend.assert_not_called()

    def test_user_stopping_the_song_leaves_dj_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        picker = _picker(SEG1, SEG2)
        loop, m, _ = _make_loop(monkeypatch, song_catalog=SONGS, dj_picker=picker)
        _wire_bridge_completion(m["bridge"])
        started = lambda: "c1" in _outputs(m["live"])  # noqa: E731
        playing = lambda: loop._song_state is SongState.PLAYING and "c2" in _outputs(m["live"])  # noqa: E731
        reopened = lambda: loop._song_state is SongState.NONE and "c3" in _outputs(m["live"])  # noqa: E731
        _staged_live_events(
            m,
            [
                (None, _call(START_DJ_TOOL)),
                (None, LiveResponseDone(None)),
                (started, _call(PLAY_DJ_SONG_TOOL, call_id="c2")),
                (started, LiveResponseDone(None)),
                (playing, _call(STOP_SONG_TOOL, call_id="c3")),
                (playing, LiveResponseDone(None)),
                (reopened, LiveClosed("x")),
            ],
        )
        loop.run()

        live = m["live"]
        assert _outputs(live)["c3"] == {"status": "stopped"}
        assert not loop._dj_active and loop._dj_segment is None
        picker.pick.assert_called_once()  # 다음 소재로 넘어가지 않는다
        assert _appended(live) == [DJ_BEGIN_INSTRUCTIONS, SONG_PLAYING_INSTRUCTIONS, DJ_ENDED_INSTRUCTIONS]
        live.restore_backend.assert_called_once()  # 정지 결과의 후속 응답 뒤 기본으로

    def test_stop_dj_restores_backend_and_announces_end(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(monkeypatch, song_catalog=SONGS, dj_picker=_picker(SEG1))
        started = lambda: "c1" in _outputs(m["live"])  # noqa: E731
        stopped = lambda: "c2" in _outputs(m["live"])  # noqa: E731
        _staged_live_events(
            m,
            [
                (None, _call(START_DJ_TOOL)),
                (None, LiveResponseDone(None)),
                (started, _call(STOP_DJ_TOOL, call_id="c2")),
                (started, LiveResponseDone(None)),
                (stopped, LiveClosed("x")),
            ],
        )
        loop.run()

        live = m["live"]
        assert _outputs(live)["c2"] == {"status": "stopped"}
        assert not loop._dj_active
        live.restore_backend.assert_called_once()
        assert _appended(live) == [DJ_BEGIN_INSTRUCTIONS, DJ_ENDED_INSTRUCTIONS]
        assert live.continue_response.call_count == 2  # 종료 결과 뒤에도 이어간다 — 모델이 한마디 한다

    def test_dj_tools_outside_dj_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, _, _ = _make_loop(monkeypatch, song_catalog=SONGS, dj_picker=_picker(SEG1))
        assert json.loads(loop._submit_tool(_call(STOP_DJ_TOOL)).result()) == {"status": "not_active"}
        assert "not in DJ mode" in json.loads(loop._submit_tool(_call(PLAY_DJ_SONG_TOOL)).result())["error"]

    def test_quiet_during_dj_sends_return_signal_once_per_silence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(monkeypatch, song_catalog=SONGS, dj_picker=_picker(SEG1), dj_quiet=0.1)
        nudged_at: list[float] = []

        def long_after_nudge() -> bool:
            if DJ_RETURN_INSTRUCTIONS in _appended(m["live"]):
                nudged_at.append(nudged_at[-1] if nudged_at else time.monotonic())
                return time.monotonic() - nudged_at[0] > 0.35  # 재촉 뒤 침묵이 이어져도 다시 보내지 않는다
            return False

        _staged_live_events(
            m,
            [
                (None, _call(START_DJ_TOOL)),
                (None, LiveResponseDone(None)),
                (long_after_nudge, LiveClosed("x")),
            ],
        )
        loop.run()
        assert _appended(m["live"]) == [DJ_BEGIN_INSTRUCTIONS, DJ_RETURN_INSTRUCTIONS]

    def test_no_return_signal_while_song_or_delegation_is_pending(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(monkeypatch, song_catalog=SONGS, dj_picker=_picker(SEG1), dj_quiet=0.05)
        loop._dj_active, loop._dj_segment = True, SEG1
        loop._dj_signal_time = time.monotonic() - 1.0
        loop._song_state = SongState.PLAYING
        loop._maybe_nudge_dj(time.monotonic())
        loop._song_state = SongState.NONE
        loop._inflight.append((_call(SEARCH_MEMORY_TOOL), Future()))
        loop._maybe_nudge_dj(time.monotonic())
        m["live"].append_instructions.assert_not_called()
        loop._inflight.clear()
        loop._maybe_nudge_dj(time.monotonic())
        m["live"].append_instructions.assert_called_once_with(DJ_RETURN_INSTRUCTIONS)

    def test_ending_during_dj_resets_state_without_touching_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, m, _ = _make_loop(monkeypatch, song_catalog=SONGS, dj_picker=_picker(SEG1))
        started = lambda: "c1" in _outputs(m["live"])  # noqa: E731
        _staged_live_events(
            m,
            [
                (None, _call(START_DJ_TOOL)),
                (None, LiveResponseDone(None)),
                (started, _seg("user", "오늘은 여기까지", closed=True)),
            ],
        )
        loop.run()
        assert loop.exit_reason == "exit_keyword" and not loop._dj_active
        m["live"].restore_backend.assert_not_called()
        assert _appended(m["live"]) == [DJ_BEGIN_INSTRUCTIONS]
