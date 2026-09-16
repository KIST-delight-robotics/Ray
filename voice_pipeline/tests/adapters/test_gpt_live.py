"""Tests for voice_pipeline.adapters.gpt_live — SDK 이벤트를 프로젝트 이벤트로 바꾸는 부분.

실제 서버 없이 ``connection_factory`` 로 가짜 연결을 주입한다. 가짜 연결은 SDK 타입의 서버 이벤트를
순서대로 돌려주고, 다 소진되면 연결 종료 예외를 낸다.
"""

from __future__ import annotations

import base64
import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from openai.types.live import (
    InputTranscriptDeltaEvent,
    OutputAudioDeltaEvent,
    OutputTranscriptDeltaEvent,
    SessionClosedEvent,
)

from voice_pipeline.adapters.gpt_live import (
    GPTLiveSession,
    LiveAudio,
    LiveClosed,
    LiveFunctionCall,
    LiveResponseDone,
    LiveSessionConfig,
    LiveTranscript,
    LiveUsage,
)


class _Closed(Exception):
    pass


class FakeConnection:
    """SDK LiveConnection 표면의 최소 흉내."""

    def __init__(self, events: list[Any]) -> None:
        self._events = list(events)
        self.session = MagicMock()
        self.response = MagicMock()
        self.closed = False

    def recv(self) -> Any:
        if self._events:
            ev = self._events.pop(0)
            if isinstance(ev, Exception):
                raise ev
            return ev
        # 이벤트가 끝나면 살짝 기다렸다 연결 종료로 취급 — 수신 스레드가 빠져나오게
        time.sleep(0.01)
        raise _Closed("closed")

    def close(self, **_: Any) -> None:
        self.closed = True


def _started() -> Any:
    ev = MagicMock()
    ev.type = "session.started"
    ev.session.id = "live_abc"
    ev.session.expires_at = time.time() + 7200
    return ev


def _audio(pcm: bytes) -> OutputAudioDeltaEvent:
    return OutputAudioDeltaEvent(type="session.output_audio.delta", delta=base64.b64encode(pcm).decode())


def _user(text: str, start: int, end: int) -> InputTranscriptDeltaEvent:
    return InputTranscriptDeltaEvent(
        type="session.input_transcript.delta", event_id=f"u{start}", delta=text, start_ms=start, end_ms=end
    )


def _ray(text: str, start: int, end: int) -> OutputTranscriptDeltaEvent:
    return OutputTranscriptDeltaEvent(
        type="session.output_transcript.delta", event_id=f"a{start}", delta=text, start_ms=start, end_ms=end
    )


def _backend(inner: dict[str, Any], delegation_id: str = "d1") -> Any:
    ev = MagicMock()
    ev.type = "response.event"
    ev.event = inner
    ev.delegation_id = delegation_id
    return ev


def _closed(reason: str = "close_requested") -> Any:
    ev = MagicMock(spec=SessionClosedEvent)
    ev.type = "session.closed"
    ev.reason = reason
    return ev


def _usage(seconds: float, ratio: float | None) -> Any:
    ev = MagicMock()  # pydantic 필드는 spec 으로 잡히지 않아 plain mock
    ev.type = "session.usage.updated"
    ev.usage.seconds = seconds
    ev.context_window = None if ratio is None else MagicMock(usage_ratio=ratio)
    return ev


def _drain(session: GPTLiveSession, *, want: int, timeout: float = 2.0) -> list[Any]:
    out: list[Any] = []
    deadline = time.monotonic() + timeout
    while len(out) < want and time.monotonic() < deadline:
        ev = session.poll_event()
        if ev is None:
            time.sleep(0.005)
            continue
        out.append(ev)
    return out


def _session(events: list[Any]) -> tuple[GPTLiveSession, FakeConnection]:
    conn = FakeConnection([_started(), *events])
    session = GPTLiveSession(LiveSessionConfig(instructions="hi"), connection_factory=lambda: conn)
    return session, conn


# ---------------------------------------------------------------------------


class TestConfig:
    def test_client_delegation_when_no_backend(self) -> None:
        cfg = LiveSessionConfig(instructions="x").to_session_dict()
        assert "delegation" not in cfg
        assert cfg["audio"]["format"] == {"type": "audio/pcm", "rate": 24000}

    def test_responses_delegation_includes_web_search_and_tools(self) -> None:
        tool = {"type": "function", "name": "f", "parameters": {"type": "object", "properties": {}}}
        cfg = LiveSessionConfig(instructions="x", backend_model="m", tools=(tool,)).to_session_dict()
        tools = cfg["delegation"]["responses"]["tools"]
        assert tools[0] == {"type": "web_search"} and tools[1] == tool


class TestStart:
    def test_start_sends_config_and_returns_started(self) -> None:
        session, conn = _session([])
        started = session.start()
        assert started.session_id == "live_abc"
        conn.session.start.assert_called_once()
        assert conn.session.start.call_args.kwargs["session"]["model"] == "gpt-live-1"
        session.close()
        assert conn.closed

    def test_start_error_event_raises(self) -> None:
        err = MagicMock()
        err.type = "error"
        err.error.message = "no access"
        conn = FakeConnection([err])
        session = GPTLiveSession(LiveSessionConfig(instructions="x"), connection_factory=lambda: conn)
        with pytest.raises(RuntimeError, match="rejected"):
            session.start()
        assert conn.closed

    def test_connect_failure_raises_runtime_error(self) -> None:
        def boom() -> Any:
            raise OSError("network down")

        session = GPTLiveSession(LiveSessionConfig(instructions="x"), connection_factory=boom)
        with pytest.raises(RuntimeError, match="connect failed"):
            session.start()


class TestEvents:
    def test_audio_delta_is_decoded(self) -> None:
        pcm = b"\x01\x02" * 100
        session, _ = _session([_audio(pcm)])
        session.start()
        events = _drain(session, want=1)
        session.close()
        assert events == [LiveAudio(pcm)]

    def test_transcript_fragments_are_grouped_per_speaker(self) -> None:
        session, _ = _session([_user("오늘", 0, 200), _user(" 날씨", 200, 400), _ray("맑아", 1000, 1200), _closed()])
        session.start()
        events = _drain(session, want=6)
        session.close()
        transcripts = [e for e in events if isinstance(e, LiveTranscript)]
        closed = [t for t in transcripts if t.closed]
        assert [(t.speaker, t.text.strip()) for t in closed] == [("user", "오늘 날씨"), ("assistant", "맑아")]
        assert any(isinstance(e, LiveClosed) and e.reason == "close_requested" for e in events)

    def test_backend_function_call_and_done(self) -> None:
        call = {
            "type": "response.output_item.done",
            "item": {"type": "function_call", "call_id": "c1", "name": "end_conversation", "arguments": "{}"},
        }
        session, _ = _session([_backend(call), _backend({"type": "response.completed"})])
        session.start()
        events = _drain(session, want=2)
        session.close()
        assert events[0] == LiveFunctionCall("c1", "end_conversation", "{}", "d1")
        assert events[1] == LiveResponseDone("d1")

    def test_backend_message_items_are_ignored(self) -> None:
        msg = {"type": "response.output_item.done", "item": {"type": "message"}}
        session, _ = _session([_backend(msg), _backend({"type": "response.completed"})])
        session.start()
        events = _drain(session, want=1)
        session.close()
        assert events == [LiveResponseDone("d1")]

    def test_usage_event(self) -> None:
        session, _ = _session([_usage(14.0, 0.0131)])
        session.start()
        events = _drain(session, want=1)
        session.close()
        assert events == [LiveUsage(14.0, 0.0131)]

    def test_connection_drop_yields_connection_lost(self) -> None:
        session, _ = _session([])  # started 뒤 바로 recv 예외
        session.start()
        events = _drain(session, want=1)
        session.close()
        assert events == [LiveClosed("connection_lost")]


class TestSend:
    def test_send_audio_base64(self) -> None:
        session, conn = _session([])
        session.start()
        session.send_audio(b"\x00\x01")
        conn.session.input_audio.append.assert_called_once_with(audio=base64.b64encode(b"\x00\x01").decode())
        session.close()

    def test_tool_output_and_continue(self) -> None:
        session, conn = _session([])
        session.start()
        session.submit_function_output("c1", '{"ok":true}')
        session.continue_response()
        item = conn.response.item.create.call_args.kwargs["item"]
        assert item == {"type": "function_call_output", "call_id": "c1", "output": '{"ok":true}'}
        conn.response.create.assert_called_once()
        session.close()

    def test_send_before_start_raises(self) -> None:
        session = GPTLiveSession(LiveSessionConfig(instructions="x"), connection_factory=lambda: None)
        with pytest.raises(RuntimeError, match="not started"):
            session.send_audio(b"")

    def test_close_sends_session_close_and_closes_socket(self) -> None:
        session, conn = _session([])
        session.start()
        session.close()
        conn.session.close.assert_called_once()
        assert conn.closed
