"""Unit tests for ProcessComponents.create_session assembly logic.

build_components()는 실제 모델·API 클라이언트를 로드하므로 유닛 테스트 대상이
아니다. 여기서는 세션 조립(create_session)의 배선 계약만 검증한다 — 세션 컴포넌트
클래스들은 wiring 모듈 네임스페이스에서 mock으로 치환한다.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import voice_pipeline.wiring as wiring
from voice_pipeline.engines.gpt_live.radio import Radio
from voice_pipeline.engines.gpt_live.tools import SEARCH_MEMORY_TOOL
from voice_pipeline.memory.types import UserNote
from voice_pipeline.trace import record_call
from voice_pipeline.wiring import ProcessComponents


def _make_components() -> ProcessComponents:
    return ProcessComponents(
        silero_vad_model=MagicMock(),
        bridge=MagicMock(),
        led=MagicMock(),
        storage=MagicMock(),
        executor=MagicMock(),
        token_counter=MagicMock(return_value=1),
        embedder=MagicMock(),
        memory_storage=MagicMock(),
        retry_handler=MagicMock(),
        vector_index=MagicMock(),
        audio_queue=MagicMock(),
        audio_input=MagicMock(),
        shutdown_event=threading.Event(),
    )


class TestCreateSession:
    """세션마다 GPT-Live 세션과 세션 루프를 새로 조립한다."""

    def test_session_id_set_on_trace_context(self, monkeypatch, call_log):
        monkeypatch.setattr(wiring, "GPTLiveSession", MagicMock())
        monkeypatch.setattr(wiring, "LiveSessionLoop", MagicMock())
        comps = _make_components()

        result = comps.create_session(memory_enabled=False)
        record_call("tts", "synthesize", "m", 1.0)

        assert call_log.records[0].session_id == result.session_id

    def test_returns_live_session_loop(self, monkeypatch):
        live_cls = MagicMock(name="GPTLiveSession")
        loop_cls = MagicMock(name="LiveSessionLoop")
        monkeypatch.setattr(wiring, "GPTLiveSession", live_cls)
        monkeypatch.setattr(wiring, "LiveSessionLoop", loop_cls)
        monkeypatch.setattr(wiring, "ConversationHistory", MagicMock(name="ConversationHistory"))
        comps = _make_components()
        comps.memory_storage.get_user_notes.return_value = [
            UserNote(1, "profile", "The user's name is 재헌.", "s-old", "2026-09-01 00:00:00"),
            UserNote(2, "preference", "Speak formally.", "s-old", "2026-09-01 00:00:00"),
        ]
        comps.memory_storage.get_recent_sessions.return_value = [("s-old", "2026-09-10 10:00:00")]
        comps.memory_storage.get_utterances.return_value = [
            ("user", "듄 2 봤어", "2026-09-10 10:00:01", 4),
            ("assistant", "어땠어요?", "2026-09-10 10:00:03", 3),
        ]

        result = comps.create_session(tool_handlers={"x": lambda a: a})

        assert result.session_loop is loop_cls.return_value
        live_cls.assert_called_once()
        config = live_cls.call_args.args[0]
        assert config.backend_model == wiring._LIVE_BACKEND_MODEL
        assert config.sample_rate == 24000
        assert {t["name"] for t in config.tools} == {"end_conversation", SEARCH_MEMORY_TOOL}
        # 사용자 노트·최근 세션은 시작 instructions 에 들어간다
        assert "About the user\n- The user's name is 재헌." in config.instructions
        assert "How the user wants Ray to talk\n- Speak formally." in config.instructions
        assert "Recent conversations:" in config.instructions
        assert "User: 듄 2 봤어\nRay: 어땠어요?" in config.instructions
        # 말투 선호는 백엔드에 따로 넣지 않는다 — 세션 지시문이 백엔드에도 공유된다
        assert "Ray's voice" not in config.backend_instructions
        kwargs = loop_cls.call_args.kwargs
        assert kwargs["live"] is live_cls.return_value
        assert kwargs["session_id"] == result.session_id
        assert set(kwargs["tool_handlers"]) == {"x", SEARCH_MEMORY_TOOL}
        assert kwargs["executor"] is comps.executor

    def test_memory_disabled_drops_search_tool(self, monkeypatch):
        live_cls = MagicMock(name="GPTLiveSession")
        monkeypatch.setattr(wiring, "GPTLiveSession", live_cls)
        loop_cls = MagicMock(name="LiveSessionLoop")
        monkeypatch.setattr(wiring, "LiveSessionLoop", loop_cls)
        monkeypatch.setattr(wiring, "ConversationHistory", MagicMock())
        comps = _make_components()

        comps.create_session(memory_enabled=False)

        assert loop_cls.call_args.kwargs["memory_storage"] is None
        assert SEARCH_MEMORY_TOOL not in loop_cls.call_args.kwargs["tool_handlers"]
        config = live_cls.call_args.args[0]
        assert [t["name"] for t in config.tools] == ["end_conversation"]
        assert "About the user" not in config.instructions
        comps.memory_storage.get_user_notes.assert_not_called()

    def test_device_settings_handlers_and_tools_are_wired(self, monkeypatch):
        live_cls = MagicMock(name="GPTLiveSession")
        monkeypatch.setattr(wiring, "GPTLiveSession", live_cls)
        loop_cls = MagicMock(name="LiveSessionLoop")
        monkeypatch.setattr(wiring, "LiveSessionLoop", loop_cls)
        monkeypatch.setattr(wiring, "ConversationHistory", MagicMock())
        comps = _make_components()
        comps.device_settings = MagicMock()
        comps.device_settings.tool_handlers.return_value = {
            "adjust_volume": lambda a: a,
            "set_brightness": lambda a: a,
            "get_device_settings": lambda a: a,
        }

        comps.create_session(memory_enabled=False)

        # 핸들러가 등록된 툴만 백엔드에 노출된다 (end_conversation 은 루프 내장)
        config = live_cls.call_args.args[0]
        device_tools = {"adjust_volume", "set_brightness", "get_device_settings"}
        assert {t["name"] for t in config.tools} == {"end_conversation", *device_tools}
        assert set(loop_cls.call_args.kwargs["tool_handlers"]) == device_tools

    def test_radio_tools_are_wired_when_picker_present(self, monkeypatch):
        live_cls = MagicMock(name="GPTLiveSession")
        monkeypatch.setattr(wiring, "GPTLiveSession", live_cls)
        loop_cls = MagicMock(name="LiveSessionLoop")
        monkeypatch.setattr(wiring, "LiveSessionLoop", loop_cls)
        monkeypatch.setattr(wiring, "ConversationHistory", MagicMock())
        comps = _make_components()
        comps.radio_picker = MagicMock()
        comps.radio_judge = MagicMock()

        comps.create_session(memory_enabled=False)

        config = live_cls.call_args.args[0]
        assert [t["name"] for t in config.tools] == ["end_conversation", "start_radio", "resume_radio"]
        assert config.backend_instructions.rstrip().endswith("Radio talk status: off")
        comps.radio_picker.new_session.assert_called_once()  # 세션마다 세션 내 사용 기록을 비운다
        radio = loop_cls.call_args.kwargs["radio"]
        assert isinstance(radio, Radio) and radio.status == "off"

    def test_no_radio_without_scripts(self, monkeypatch):
        live_cls = MagicMock(name="GPTLiveSession")
        monkeypatch.setattr(wiring, "GPTLiveSession", live_cls)
        loop_cls = MagicMock(name="LiveSessionLoop")
        monkeypatch.setattr(wiring, "LiveSessionLoop", loop_cls)
        monkeypatch.setattr(wiring, "ConversationHistory", MagicMock())
        comps = _make_components()

        comps.create_session(memory_enabled=False)

        assert loop_cls.call_args.kwargs["radio"] is None
        assert "## Radio talk" not in live_cls.call_args.args[0].backend_instructions

    def test_only_end_conversation_without_handlers(self, monkeypatch):
        live_cls = MagicMock(name="GPTLiveSession")
        monkeypatch.setattr(wiring, "GPTLiveSession", live_cls)
        monkeypatch.setattr(wiring, "LiveSessionLoop", MagicMock())
        monkeypatch.setattr(wiring, "ConversationHistory", MagicMock())
        comps = _make_components()

        comps.create_session(memory_enabled=False)

        assert [t["name"] for t in live_cls.call_args.args[0].tools] == ["end_conversation"]
