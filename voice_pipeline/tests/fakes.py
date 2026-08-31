"""테스트 전용 기록 스토어. 프로덕션 SQLite 스토어 대신 호출/트레이스를 리스트에 쌓아 검증한다."""

from __future__ import annotations

from voice_pipeline.trace import CallRecord, PipelineTrace


class RecordingTraceStore:
    def __init__(self) -> None:
        self.traces: list[PipelineTrace] = []

    def save(self, trace: PipelineTrace) -> None:
        self.traces.append(trace)

    def close(self) -> None:
        pass


class RecordingCallStore:
    def __init__(self) -> None:
        self.records: list[CallRecord] = []
        self._turn_index = 0

    def record(self, record: CallRecord) -> None:
        self.records.append(record)

    def set_turn_index(self, index: int) -> None:
        self._turn_index = index

    @property
    def current_turn_index(self) -> int:
        return self._turn_index

    def close(self) -> None:
        pass
