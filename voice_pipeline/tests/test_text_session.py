"""TextSession unit tests — LLM/retriever mocked, real history/context chain."""

from __future__ import annotations

import pytest

from voice_pipeline.history import ConversationHistory, SQLiteStorageBackend
from voice_pipeline.text_session import TextSession
from voice_pipeline.types import LLMMetrics, LLMResult, Usage


def _token_counter(text: str) -> int:
    return len(text.split())


class FakeStream:
    """Duck-typed LLMStream: yields chunks, exposes .result after full iteration."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = iter(chunks)
        self._text = "".join(chunks)
        self.result: LLMResult | None = None
        self.closed = False

    def __iter__(self) -> FakeStream:
        return self

    def __next__(self) -> str:
        try:
            return next(self._chunks)
        except StopIteration:
            self.result = LLMResult(
                text=self._text,
                metrics=LLMMetrics(
                    usage=Usage(input_tokens=5, output_tokens=3),
                    model="fake",
                    latency_ms=20.0,
                    ttft_ms=10.0,
                ),
            )
            raise

    def close(self) -> None:
        self.closed = True


class FakeLLM:
    def __init__(self, chunks: list[str] | None = None) -> None:
        self.chunks = chunks if chunks is not None else ["Hello ", "there."]
        self.calls: list[list[dict]] = []

    def generate(self, messages, tools=None, response_format=None) -> FakeStream:
        self.calls.append(messages)
        return FakeStream(self.chunks)


def _make_session(llm: FakeLLM | None = None, **kwargs) -> tuple[TextSession, ConversationHistory]:
    history = ConversationHistory(SQLiteStorageBackend(":memory:"), _token_counter)
    session = TextSession(
        llm=llm or FakeLLM(),
        history=history,
        token_counter=_token_counter,
        system_prompt="You are a helpful robot.",
        **kwargs,
    )
    return session, history


class TestSend:
    def test_send_returns_text_and_records_history(self):
        session, history = _make_session()

        resp = session.send("What is the speed of light?")

        assert resp == "Hello there."
        contents = [str(item.get("content")) for t in history.get_turns() for item in t.items]
        assert any("speed of light" in c for c in contents)
        assert any("Hello there." in c for c in contents)
        assert session.last_metrics is not None
        assert len(session.traces) == 1
        assert session.traces[0].total_ms > 0

    def test_empty_response_raises(self):
        session, _ = _make_session(llm=FakeLLM(chunks=[""]))

        with pytest.raises(RuntimeError):
            session.send("hi")


class TestInject:
    def test_inject_records_without_llm_call(self):
        llm = FakeLLM()
        session, history = _make_session(llm=llm)

        session.inject("user", "seed question")
        session.inject("assistant", "seed answer")

        assert llm.calls == []
        contents = [str(item.get("content")) for t in history.get_turns() for item in t.items]
        assert any("seed question" in c for c in contents)
        assert any("seed answer" in c for c in contents)

    def test_inject_rejects_unknown_role(self):
        session, _ = _make_session()
        with pytest.raises(ValueError):
            session.inject("narrator", "x")


class TestMemory:
    def test_retriever_receives_context_query_and_own_session_excluded(self):
        class FakeRetriever:
            def __init__(self) -> None:
                self.calls: list[tuple[str, set[str]]] = []

            def retrieve(self, query, exclude_session_ids):
                self.calls.append((query, set(exclude_session_ids)))
                return None

            def update_citations(self, cited_indices) -> None:
                pass

        retriever = FakeRetriever()
        session, _ = _make_session(retriever=retriever)

        session.send("What movie did I watch?")

        (query, excludes) = retriever.calls[0]
        assert "What movie did I watch?" in query
        assert session.session_id in excludes
        assert session.memory_results == [None]
