"""Unit tests for ProcessComponents.create_session assembly logic.

build_components()는 실제 모델·API 클라이언트를 로드하므로 유닛 테스트 대상이
아니다. 여기서는 세션 조립(create_session)의 배선 계약만 검증한다 — 세션 컴포넌트
클래스들은 wiring 모듈 네임스페이스에서 mock으로 치환한다.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

import voice_pipeline.wiring as wiring
from voice_pipeline.session_loop import SessionComponents
from voice_pipeline.wiring import ProcessComponents

_SESSION_CLASSES = [
    "ThreadedTurnGPT",
    "ConversationHistory",
    "MemoryRetriever",
    "TurnDetector",
    "SpeechGenerator",
    "SessionLoop",
]


def _make_components() -> ProcessComponents:
    return ProcessComponents(
        language_code="en-US",
        asr=MagicMock(),
        llm=MagicMock(),
        raw_tts=MagicMock(),
        tts=MagicMock(output_sample_rate=24000),
        vap=MagicMock(),
        turngpt=MagicMock(),
        silero_vad_model=MagicMock(),
        vad_fn=lambda frame: 0.0,
        reset_vad=MagicMock(),
        bridge=MagicMock(),
        led=MagicMock(),
        storage=MagicMock(),
        executor=MagicMock(),
        token_counter=MagicMock(return_value=1),
        tools_token_cost=294,
        embedder=MagicMock(),
        memory_storage=MagicMock(),
        trace_store=MagicMock(),
        call_store=MagicMock(),
        retry_handler=MagicMock(),
        vector_index=MagicMock(),
        audio_queue=MagicMock(),
        audio_input=MagicMock(),
        shutdown_event=threading.Event(),
    )


@pytest.fixture
def session_mocks(monkeypatch) -> dict[str, MagicMock]:
    mocks = {}
    for name in _SESSION_CLASSES:
        mock = MagicMock(name=name)
        monkeypatch.setattr(wiring, name, mock)
        mocks[name] = mock
    return mocks


class TestCreateSession:
    def test_returns_session_components_with_fresh_id(self, session_mocks):
        comps = _make_components()

        result = comps.create_session()

        assert isinstance(result, SessionComponents)
        assert result.session_id
        assert result.session_loop is session_mocks["SessionLoop"].return_value
        assert result.history is session_mocks["ConversationHistory"].return_value

    def test_session_id_stamped_on_tracked_singletons(self, session_mocks):
        comps = _make_components()

        result = comps.create_session()

        assert comps.tts.session_id == result.session_id
        assert comps.embedder.session_id == result.session_id
        assert comps.retry_handler.session_id == result.session_id
        assert comps.vap.session_id == result.session_id

    def test_models_reset_before_assembly(self, session_mocks):
        comps = _make_components()

        comps.create_session()

        comps.vap.reset.assert_called_once()
        comps.turngpt.reset.assert_called_once()
        comps.reset_vad.assert_called_once()

    def test_vap_reused_not_recreated(self, session_mocks):
        comps = _make_components()

        comps.create_session()

        kwargs = session_mocks["TurnDetector"].call_args.args
        assert kwargs[0] is comps.vap

    def test_session_loop_kwargs_pass_through(self, session_mocks):
        comps = _make_components()
        on_turn_shift = MagicMock()

        comps.create_session(
            disable_exit_keywords=True,
            skip_generation=True,
            record_path="/tmp/rec.wav",
            on_turn_shift=on_turn_shift,
        )

        kwargs = session_mocks["SessionLoop"].call_args.kwargs
        assert kwargs["disable_exit_keywords"] is True
        assert kwargs["skip_generation"] is True
        assert kwargs["record_path"] == "/tmp/rec.wav"
        assert kwargs["on_turn_shift"] is on_turn_shift

    def test_memory_enabled_by_default(self, session_mocks):
        comps = _make_components()

        comps.create_session()

        session_mocks["MemoryRetriever"].assert_called_once()
        assert session_mocks["SessionLoop"].call_args.kwargs["memory_storage"] is comps.memory_storage
        assert session_mocks["SpeechGenerator"].call_args.kwargs["memory_storage"] is comps.memory_storage

    def test_memory_disabled(self, session_mocks):
        comps = _make_components()

        comps.create_session(memory_enabled=False)

        session_mocks["MemoryRetriever"].assert_not_called()
        assert session_mocks["SessionLoop"].call_args.kwargs["memory_storage"] is None
        assert session_mocks["SpeechGenerator"].call_args.kwargs["retriever"] is None

    def test_previous_threaded_turngpt_stopped_on_next_session(self, session_mocks):
        session_mocks["ThreadedTurnGPT"].side_effect = lambda *a, **k: MagicMock()
        comps = _make_components()

        comps.create_session()
        first_turngpt = comps._prev_threaded[0]
        comps.create_session()

        first_turngpt.stop.assert_called_once()
        assert len(comps._prev_threaded) == 1
