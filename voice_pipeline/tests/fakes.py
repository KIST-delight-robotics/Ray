"""테스트 전용 기록 스토어. 프로덕션 SQLite 스토어 대신 호출 기록을 리스트에 쌓아 검증한다."""

from __future__ import annotations

from voice_pipeline.trace import CallRecord


class RecordingCallStore:
    def __init__(self) -> None:
        self.records: list[CallRecord] = []

    def record(self, record: CallRecord) -> None:
        self.records.append(record)

    def close(self) -> None:
        pass
