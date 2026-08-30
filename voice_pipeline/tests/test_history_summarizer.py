"""Tests for voice_pipeline.prompt."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from voice_pipeline.history import HistoryTurn
from voice_pipeline.prompt import HistorySummarizer
from voice_pipeline.tests.fakes import RecordingCallStore
from voice_pipeline.trace import install, set_session
from voice_pipeline.types import LLMMetrics, LLMResult, LLMStream, Usage

_JOIN_TIMEOUT = 5.0


def _word_counter(text: str) -> int:
    return len(text.split()) if text.strip() else 0


def _make_stream(text: str, output_tokens: int) -> LLMStream:
    def gen():
        yield text

    metrics = LLMMetrics(
        usage=Usage(input_tokens=100, output_tokens=output_tokens),
        model="fake-mini",
        latency_ms=5,
        ttft_ms=1,
    )
    return LLMStream(gen(), result_fn=lambda full: LLMResult(text=full, metrics=metrics))


class FakeLLM:
    """Records generate() calls and returns a canned summary stream."""

    def __init__(self, response: str = "A concise merged summary.", output_tokens: int = 50) -> None:
        self.model = "fake-mini"
        self.response = response
        self.output_tokens = output_tokens
        self.error: Exception | None = None
        self.block_event: threading.Event | None = None
        self.calls: list[dict[str, Any]] = []

    def generate(self, messages, tools=None, response_format=None) -> LLMStream:
        self.calls.append({"messages": messages, "tools": tools})
        if self.block_event is not None:
            self.block_event.wait(_JOIN_TIMEOUT)
        if self.error is not None:
            raise self.error
        return _make_stream(self.response, self.output_tokens)


def _make_turns(n: int, tokens_each: int = 10) -> list[HistoryTurn]:
    """Alternating user/assistant single-item turns, user first (turn_id 0..n-1)."""
    turns = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        turns.append(
            HistoryTurn(
                items=({"role": role, "content": f"{role} message {i}"},),
                token_count=tokens_each,
                turn_id=i,
            )
        )
    return turns


class FakeSummaryBackend:
    """Records save_rolling_summary calls; optionally raises."""

    def __init__(self) -> None:
        self.saved: list[tuple[str, str, int]] = []
        self.error: Exception | None = None

    def save_rolling_summary(self, session_id: str, summary_text: str, through_turn_id: int) -> None:
        if self.error is not None:
            raise self.error
        self.saved.append((session_id, summary_text, through_turn_id))


def _make_summarizer(
    llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
    *,
    budget: int = 100,
    keep_recent: int = 4,
    summary_backend: FakeSummaryBackend | None = None,
) -> HistorySummarizer:
    monkeypatch.setattr(HistorySummarizer, "_KEEP_RECENT_TURNS", keep_recent)
    return HistorySummarizer(
        llm,
        _word_counter,
        budget,
        session_id="test-session",
        summary_backend=summary_backend,
    )


def _join(summarizer: HistorySummarizer) -> None:
    thread = summarizer._thread
    if thread is not None:
        thread.join(timeout=_JOIN_TIMEOUT)
        assert not thread.is_alive()


class TestTrigger:
    def test_below_threshold_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = FakeLLM()
        s = _make_summarizer(llm, monkeypatch, budget=10_000)
        s.maybe_schedule(_make_turns(10))
        assert llm.calls == []
        assert s.snapshot() is None

    def test_trigger_summarizes_and_swaps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 10 turns × (10 + 3 framing) = 130 ≥ 75 (trigger for budget 100)
        llm = FakeLLM(response="Summary of the early exchanges.")
        s = _make_summarizer(llm, monkeypatch, budget=100, keep_recent=4)
        s.maybe_schedule(_make_turns(10))
        _join(s)

        snap = s.snapshot()
        assert snap is not None
        # keep_recent=4 → candidates are turns 0..5; live[6] is a user turn (no shift)
        assert snap.through_turn_id == 5
        assert snap.block_text.startswith("[Earlier in this conversation]")
        assert "Summary of the early exchanges." in snap.block_text
        assert snap.token_count == _word_counter(snap.block_text)

    def test_transcript_covers_only_candidates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = FakeLLM()
        s = _make_summarizer(llm, monkeypatch, budget=100, keep_recent=4)
        s.maybe_schedule(_make_turns(10))
        _join(s)

        prompt = llm.calls[0]["messages"][1]["content"]
        assert "User: user message 0" in prompt
        assert "Ray: assistant message 5" in prompt
        assert "message 6" not in prompt  # kept turns stay out of the summary input

    def test_cutoff_aligned_to_exchange_boundary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 11 turns, keep_recent=4 → raw cutoff would keep turns 7.. (assistant
        # first) → shifted back so the kept window starts at user turn 6.
        llm = FakeLLM()
        s = _make_summarizer(llm, monkeypatch, budget=100, keep_recent=4)
        s.maybe_schedule(_make_turns(11))
        _join(s)

        snap = s.snapshot()
        assert snap is not None
        assert snap.through_turn_id == 5
        prompt = llm.calls[0]["messages"][1]["content"]
        assert "message 6" not in prompt

    def test_all_turns_kept_means_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Over threshold but fewer turns than keep_recent → nothing to summarize
        llm = FakeLLM()
        s = _make_summarizer(llm, monkeypatch, budget=10, keep_recent=4)
        s.maybe_schedule(_make_turns(3))
        assert llm.calls == []

    def test_tools_disabled_for_summary_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = FakeLLM()
        s = _make_summarizer(llm, monkeypatch)
        s.maybe_schedule(_make_turns(10))
        _join(s)
        assert llm.calls[0]["tools"] == []


class TestRollingMerge:
    def test_second_pass_includes_previous_summary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = FakeLLM(response="First rolling summary.")
        s = _make_summarizer(llm, monkeypatch, budget=100, keep_recent=4)
        s.maybe_schedule(_make_turns(10))
        _join(s)
        assert s.snapshot() is not None

        llm.response = "Second rolling summary."
        s.maybe_schedule(_make_turns(20))
        _join(s)

        assert len(llm.calls) == 2
        second_prompt = llm.calls[1]["messages"][1]["content"]
        assert "Previous summary:\nFirst rolling summary." in second_prompt
        # Already-covered turns are not re-sent
        assert "message 0" not in second_prompt

        snap = s.snapshot()
        assert "Second rolling summary." in snap.block_text
        assert snap.through_turn_id > 5


class TestConcurrency:
    def test_single_job_in_flight(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = FakeLLM()
        llm.block_event = threading.Event()
        s = _make_summarizer(llm, monkeypatch, budget=100, keep_recent=4)
        turns = _make_turns(10)

        s.maybe_schedule(turns)
        s.maybe_schedule(turns)  # in flight → no second job
        assert len(llm.calls) == 1

        llm.block_event.set()
        _join(s)
        assert s.snapshot() is not None

    def test_close_prevents_scheduling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = FakeLLM()
        s = _make_summarizer(llm, monkeypatch, budget=100, keep_recent=4)
        s.close()
        s.maybe_schedule(_make_turns(10))
        assert llm.calls == []

    def test_close_discards_in_flight_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = FakeLLM()
        llm.block_event = threading.Event()
        s = _make_summarizer(llm, monkeypatch, budget=100, keep_recent=4)
        s.maybe_schedule(_make_turns(10))

        s.close()
        llm.block_event.set()
        _join(s)
        assert s.snapshot() is None


class TestFailureHandling:
    def test_llm_error_keeps_state_and_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = FakeLLM()
        llm.error = RuntimeError("api down")
        s = _make_summarizer(llm, monkeypatch, budget=100, keep_recent=4)
        turns = _make_turns(10)

        s.maybe_schedule(turns)
        _join(s)
        assert s.snapshot() is None

        # in-flight released → next trigger retries
        llm.error = None
        s.maybe_schedule(turns)
        _join(s)
        assert len(llm.calls) == 2
        assert s.snapshot() is not None

    def test_hard_cap_hit_discards_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = RecordingCallStore()
        install(call_store=store)
        llm = FakeLLM(output_tokens=HistorySummarizer._HARD_CAP_TOKENS)
        s = _make_summarizer(llm, monkeypatch, budget=100, keep_recent=4)
        s.maybe_schedule(_make_turns(10))
        _join(s)

        assert s.snapshot() is None
        assert store.records[0].status == "truncated"

    def test_empty_summary_discarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = FakeLLM(response="   ")
        s = _make_summarizer(llm, monkeypatch, budget=100, keep_recent=4)
        s.maybe_schedule(_make_turns(10))
        _join(s)
        assert s.snapshot() is None


class TestPersistence:
    def test_summary_persisted_on_swap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = FakeSummaryBackend()
        llm = FakeLLM(response="Merged summary text.")
        s = _make_summarizer(llm, monkeypatch, budget=100, keep_recent=4, summary_backend=backend)
        s.maybe_schedule(_make_turns(10))
        _join(s)

        assert backend.saved == [("test-session", "Merged summary text.", 5)]

    def test_discarded_result_not_persisted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = FakeSummaryBackend()
        llm = FakeLLM(output_tokens=HistorySummarizer._HARD_CAP_TOKENS)  # truncated → discarded
        s = _make_summarizer(llm, monkeypatch, budget=100, keep_recent=4, summary_backend=backend)
        s.maybe_schedule(_make_turns(10))
        _join(s)

        assert backend.saved == []

    def test_persist_failure_keeps_snapshot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = FakeSummaryBackend()
        backend.error = RuntimeError("disk full")
        llm = FakeLLM()
        s = _make_summarizer(llm, monkeypatch, budget=100, keep_recent=4, summary_backend=backend)
        s.maybe_schedule(_make_turns(10))
        _join(s)

        # Persistence is best-effort — the in-memory swap must survive
        assert s.snapshot() is not None

    def test_closed_summarizer_does_not_persist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = FakeSummaryBackend()
        llm = FakeLLM()
        llm.block_event = threading.Event()
        s = _make_summarizer(llm, monkeypatch, budget=100, keep_recent=4, summary_backend=backend)
        s.maybe_schedule(_make_turns(10))

        s.close()
        llm.block_event.set()
        _join(s)
        assert backend.saved == []


class TestObservability:
    def test_call_recorded_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = RecordingCallStore()
        install(call_store=store)
        set_session("test-session")
        llm = FakeLLM()
        s = _make_summarizer(llm, monkeypatch, budget=100, keep_recent=4)
        s.maybe_schedule(_make_turns(10))
        _join(s)

        assert len(store.records) == 1
        record = store.records[0]
        assert record.module == "history_summarizer"
        assert record.operation == "summarize"
        assert record.model == "fake-mini"
        assert record.status == "ok"
        assert record.session_id == "test-session"
