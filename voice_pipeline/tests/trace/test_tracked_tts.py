"""Tests for TrackedTTS wrapper."""

from __future__ import annotations

import json
from collections.abc import Generator
from unittest.mock import MagicMock

import pytest

from voice_pipeline.core.interfaces import ITTS
from voice_pipeline.core.types import TTSStream
from voice_pipeline.tts.exceptions import TTSError
from voice_pipeline.trace.trace_store import InMemoryCallStore
from voice_pipeline.trace.tracked_tts import TrackedTTS


class _FakeTTS(ITTS):
    output_sample_rate: int = 24000
    voice_id: str = "fake|test"
    model_name: str = "fake-tts"

    def synthesize(self, text: str) -> TTSStream:
        def gen() -> Generator[bytes, None, None]:
            yield b"\x00" * 100

        return TTSStream(gen())


class _FailingTTS(ITTS):
    output_sample_rate: int = 24000
    voice_id: str = "fake|fail"
    model_name: str = "fake-tts"

    def __init__(self, error: Exception) -> None:
        self._error = error

    def synthesize(self, text: str) -> TTSStream:
        raise self._error


class TestTrackedTTS:
    @pytest.fixture()
    def setup(self) -> tuple[TrackedTTS, InMemoryCallStore]:
        inner = _FakeTTS()
        store = InMemoryCallStore()
        wrapper = TrackedTTS(inner, store)
        wrapper.session_id = "sess-1"
        return wrapper, store

    def test_synthesize_passthrough(self, setup: tuple[TrackedTTS, InMemoryCallStore]) -> None:
        wrapper, _ = setup
        stream = wrapper.synthesize("hello world")
        chunks = list(stream)
        assert len(chunks) == 1
        assert chunks[0] == b"\x00" * 100

    def test_synthesize_records_ok(self, setup: tuple[TrackedTTS, InMemoryCallStore]) -> None:
        wrapper, store = setup
        wrapper.synthesize("hello")
        assert len(store.records) == 1
        rec = store.records[0]
        assert rec.session_id == "sess-1"
        assert rec.module == "tts"
        assert rec.operation == "synthesize"
        assert rec.model == "fake-tts"
        assert rec.status == "ok"
        assert rec.elapsed_ms >= 0
        meta = json.loads(rec.metadata)
        assert meta["text_len"] == 5

    def test_synthesize_records_error(self) -> None:
        inner = _FailingTTS(TTSError("API error"))
        store = InMemoryCallStore()
        wrapper = TrackedTTS(inner, store)
        wrapper.session_id = "sess-1"

        with pytest.raises(TTSError):
            wrapper.synthesize("test")

        assert len(store.records) == 1
        rec = store.records[0]
        assert rec.status == "error"
        meta = json.loads(rec.metadata)
        assert "API error" in meta["error"]

    def test_synthesize_records_timeout(self) -> None:
        inner = _FailingTTS(TTSError("TTS timeout (5.0s): connection timed out"))
        store = InMemoryCallStore()
        wrapper = TrackedTTS(inner, store)

        with pytest.raises(TTSError):
            wrapper.synthesize("test")

        assert store.records[0].status == "timeout"

    def test_output_sample_rate_delegates(self, setup: tuple[TrackedTTS, InMemoryCallStore]) -> None:
        wrapper, _ = setup
        assert wrapper.output_sample_rate == 24000

    def test_voice_id_delegates(self, setup: tuple[TrackedTTS, InMemoryCallStore]) -> None:
        wrapper, _ = setup
        assert wrapper.voice_id == "fake|test"

    def test_model_name_delegates(self, setup: tuple[TrackedTTS, InMemoryCallStore]) -> None:
        wrapper, _ = setup
        assert wrapper.model_name == "fake-tts"

    def test_store_error_swallowed(self) -> None:
        inner = _FakeTTS()
        store = MagicMock()
        store.record.side_effect = RuntimeError("db error")
        wrapper = TrackedTTS(inner, store)
        stream = wrapper.synthesize("test")
        assert list(stream)

    def test_session_id_mutable(self) -> None:
        inner = _FakeTTS()
        store = InMemoryCallStore()
        wrapper = TrackedTTS(inner, store)
        wrapper.session_id = "s1"
        wrapper.synthesize("a")
        wrapper.session_id = "s2"
        wrapper.synthesize("b")
        assert store.records[0].session_id == "s1"
        assert store.records[1].session_id == "s2"
