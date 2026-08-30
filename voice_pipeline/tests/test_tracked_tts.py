"""Tests for TrackedTTS wrapper."""

from __future__ import annotations

import json
import time
from collections.abc import Generator
from unittest.mock import MagicMock

import pytest

from voice_pipeline.tests.fakes import RecordingCallStore
from voice_pipeline.trace import TrackedTTS
from voice_pipeline.types import ITTS, TTSStream


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
    def setup(self) -> tuple[TrackedTTS, RecordingCallStore]:
        inner = _FakeTTS()
        store = RecordingCallStore()
        wrapper = TrackedTTS(inner, store)
        wrapper.session_id = "sess-1"
        return wrapper, store

    def test_synthesize_passthrough(self, setup: tuple[TrackedTTS, RecordingCallStore]) -> None:
        wrapper, _ = setup
        stream = wrapper.synthesize("hello world")
        chunks = list(stream)
        assert len(chunks) == 1
        assert chunks[0] == b"\x00" * 100

    def test_synthesize_records_ok(self, setup: tuple[TrackedTTS, RecordingCallStore]) -> None:
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
        inner = _FailingTTS(RuntimeError("API error"))
        store = RecordingCallStore()
        wrapper = TrackedTTS(inner, store)
        wrapper.session_id = "sess-1"

        with pytest.raises(RuntimeError):
            wrapper.synthesize("test")

        assert len(store.records) == 1
        rec = store.records[0]
        assert rec.status == "error"
        meta = json.loads(rec.metadata)
        assert "API error" in meta["error"]

    def test_synthesize_records_timeout(self) -> None:
        inner = _FailingTTS(RuntimeError("TTS timeout (5.0s): connection timed out"))
        store = RecordingCallStore()
        wrapper = TrackedTTS(inner, store)

        with pytest.raises(RuntimeError):
            wrapper.synthesize("test")

        assert store.records[0].status == "timeout"

    def test_output_sample_rate_delegates(self, setup: tuple[TrackedTTS, RecordingCallStore]) -> None:
        wrapper, _ = setup
        assert wrapper.output_sample_rate == 24000

    def test_voice_id_delegates(self, setup: tuple[TrackedTTS, RecordingCallStore]) -> None:
        wrapper, _ = setup
        assert wrapper.voice_id == "fake|test"

    def test_model_name_delegates(self, setup: tuple[TrackedTTS, RecordingCallStore]) -> None:
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
        store = RecordingCallStore()
        wrapper = TrackedTTS(inner, store)
        wrapper.session_id = "s1"
        wrapper.synthesize("a")
        wrapper.session_id = "s2"
        wrapper.synthesize("b")
        assert store.records[0].session_id == "s1"
        assert store.records[1].session_id == "s2"


class _ScriptedStreamTTS(ITTS):
    """TTS whose stream yields scripted chunks with per-chunk delays."""

    output_sample_rate: int = 24000
    voice_id: str = "fake|scripted"
    model_name: str = "fake-tts"

    def __init__(self, chunks: list[bytes], delays: list[float]) -> None:
        self._chunks = chunks
        self._delays = delays

    def synthesize(self, text: str) -> TTSStream:
        def gen() -> Generator[bytes, None, None]:
            for chunk, delay in zip(self._chunks, self._delays, strict=True):
                if delay:
                    time.sleep(delay)
                yield chunk

        return TTSStream(gen())


def _one_sec_chunk() -> bytes:
    return b"\x00" * (24000 * 2)  # 1초 분량 오디오


class TestStreamMonitoring:
    def _make(self, inner: ITTS) -> tuple[TrackedTTS, RecordingCallStore]:
        store = RecordingCallStore()
        wrapper = TrackedTTS(inner, store)
        wrapper.session_id = "sess-1"
        return wrapper, store

    def _stream_record(self, store: RecordingCallStore):
        recs = [r for r in store.records if r.operation == "stream"]
        assert len(recs) == 1
        return recs[0]

    def test_fast_stream_records_ok(self) -> None:
        inner = _ScriptedStreamTTS([_one_sec_chunk(), _one_sec_chunk()], [0, 0])
        wrapper, store = self._make(inner)

        chunks = list(wrapper.synthesize("hello"))
        assert len(chunks) == 2

        rec = self._stream_record(store)
        assert rec.status == "ok"
        meta = json.loads(rec.metadata)
        assert meta["completed"] is True
        assert meta["audio_sec"] == pytest.approx(2.0)
        assert meta["min_headroom_sec"] > 0

    def test_stalled_stream_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """청크 간 공백이 임계 이상이면 stalled."""
        monkeypatch.setattr(TrackedTTS, "_STALL_GAP_MS", 30.0)
        inner = _ScriptedStreamTTS([_one_sec_chunk(), _one_sec_chunk()], [0, 0.06])
        wrapper, store = self._make(inner)

        list(wrapper.synthesize("hello"))

        rec = self._stream_record(store)
        assert rec.status == "stalled"
        meta = json.loads(rec.metadata)
        assert meta["max_gap_ms"] >= 30.0

    def test_slow_stream_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """수신 오디오가 실시간(경과 시간)보다 뒤처지면 slow."""
        monkeypatch.setattr(TrackedTTS, "_STALL_GAP_MS", 10_000.0)
        tiny = b"\x00" * 480  # 10ms 분량
        inner = _ScriptedStreamTTS([tiny, tiny, tiny], [0, 0.03, 0.03])
        wrapper, store = self._make(inner)

        list(wrapper.synthesize("hello"))

        rec = self._stream_record(store)
        assert rec.status == "slow"
        meta = json.loads(rec.metadata)
        assert meta["min_headroom_sec"] < 0

    def test_early_close_records_partial(self) -> None:
        inner = _ScriptedStreamTTS([_one_sec_chunk(), _one_sec_chunk()], [0, 0])
        wrapper, store = self._make(inner)

        stream = wrapper.synthesize("hello")
        next(iter(stream))
        stream.close()

        rec = self._stream_record(store)
        meta = json.loads(rec.metadata)
        assert meta["completed"] is False
        assert meta["audio_sec"] == pytest.approx(1.0)

    def test_stream_interface_preserved(self) -> None:
        """래핑 후에도 audio/timestamps 속성이 동작한다."""
        inner = _ScriptedStreamTTS([_one_sec_chunk()], [0])
        wrapper, _ = self._make(inner)

        stream = wrapper.synthesize("hello")
        list(stream)
        assert stream.audio == _one_sec_chunk()
        assert stream.timestamps == ()
