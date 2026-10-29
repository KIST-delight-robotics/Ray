"""trace 모듈 API — 싱크 설치, 컨텍스트 스탬핑, no-op 동작."""

from __future__ import annotations

from voice_pipeline import trace
from voice_pipeline.tests.fakes import RecordingCallStore


class TestNoSink:
    def test_record_call_is_noop(self) -> None:
        trace.record_call("tts", "synthesize", "m", 1.0)  # 예외 없이 무시


class TestContextStamping:
    def test_record_call_stamps_session(self, call_log: RecordingCallStore) -> None:
        trace.set_session("s1")
        trace.record_call("embedding", "embed", "minilm", 12.5, status="error")
        (rec,) = call_log.records
        assert (rec.session_id, rec.module, rec.status) == ("s1", "embedding", "error")
        assert rec.timestamp


class TestIsolation:
    def test_reset_clears_everything(self, call_log: RecordingCallStore) -> None:
        trace.set_session("s")
        trace.reset()
        trace.record_call("tts", "synthesize", "m", 1.0)
        assert call_log.records == []

    def test_close_closes_sinks_and_resets(self) -> None:
        class ClosableSink(RecordingCallStore):
            closed = False

            def close(self) -> None:
                self.closed = True

        sink = ClosableSink()
        trace.install(call_store=sink)
        trace.close()
        assert sink.closed
        trace.record_call("tts", "synthesize", "m", 1.0)
        assert sink.records == []
