"""전 테스트 공용 픽스처.

trace 모듈은 logging처럼 프로세스 전역 상태(싱크·세션·턴)를 가지므로 테스트 사이에 반드시 초기화한다.
"""

from __future__ import annotations

import pytest

from voice_pipeline import trace
from voice_pipeline.tests.fakes import RecordingCallStore, RecordingTraceStore


@pytest.fixture(autouse=True)
def _trace_isolation():
    trace.reset()
    yield
    trace.reset()


@pytest.fixture
def call_log() -> RecordingCallStore:
    """호출 기록 싱크. 테스트가 ``call_log.records`` 로 검증한다."""
    store = RecordingCallStore()
    trace.install(call_store=store)
    return store


@pytest.fixture
def turn_log() -> RecordingTraceStore:
    """턴 기록 싱크. 테스트가 ``turn_log.traces`` 로 검증한다."""
    store = RecordingTraceStore()
    trace.install(trace_store=store)
    return store
