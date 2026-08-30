"""trace 모듈 API — 싱크 설치, 컨텍스트 스탬핑, no-op 동작."""

from __future__ import annotations

from voice_pipeline import trace
from voice_pipeline.tests.fakes import RecordingCallStore, RecordingTraceStore
from voice_pipeline.trace import PipelineTrace


class TestNoSink:
    def test_record_call_is_noop(self) -> None:
        trace.record_call("tts", "synthesize", "m", 1.0)  # 예외 없이 무시

    def test_capture_call_returns_none(self) -> None:
        assert trace.capture_call("tts", "synthesize", "m", 1.0) is None

    def test_save_turn_is_noop(self) -> None:
        trace.save_turn(PipelineTrace(run_id=1))


class TestContextStamping:
    def test_record_call_stamps_session_and_turn(self, call_log: RecordingCallStore) -> None:
        trace.set_session("s1")
        trace.set_turn(3)
        trace.record_call("vap", "feed_audio", "maai-vap", 12.5, status="overrun")
        (rec,) = call_log.records
        assert (rec.session_id, rec.turn_index, rec.module, rec.status) == ("s1", 3, "vap", "overrun")
        assert rec.timestamp

    def test_turn_index_override(self, call_log: RecordingCallStore) -> None:
        trace.set_turn(3)
        trace.record_call("similarity_gate", "prepare_gate", "embedder", 1.0, turn_index=7)
        assert call_log.records[0].turn_index == 7

    def test_capture_then_write(self, call_log: RecordingCallStore) -> None:
        trace.set_turn(1)
        first = trace.capture_call("turngpt", "predict", "turngpt", 1.0)
        trace.set_turn(2)
        second = trace.capture_call("turngpt", "predict", "turngpt", 1.0)
        assert first is not None and second is not None
        trace.write_calls([first, second])
        assert [r.turn_index for r in call_log.records] == [1, 2]  # 캡처 시점의 턴이 찍힌다

    def test_save_turn_stamps_session_and_turn(self, turn_log: RecordingTraceStore) -> None:
        trace.set_session("s2")
        trace.set_turn(4)
        trace.save_turn(PipelineTrace(run_id=9, outcome="completed"))
        (t,) = turn_log.traces
        assert (t.session_id, t.turn_index, t.run_id) == ("s2", 4, 9)
        assert t.to_record()["turn_index"] == 4

    def test_current_turn(self) -> None:
        assert trace.current_turn() == 0
        trace.set_turn(5)
        assert trace.current_turn() == 5


class TestIsolation:
    def test_reset_clears_everything(self, call_log: RecordingCallStore) -> None:
        trace.set_session("s"), trace.set_turn(2)
        trace.reset()
        trace.record_call("tts", "synthesize", "m", 1.0)
        assert call_log.records == [] and trace.current_turn() == 0

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
