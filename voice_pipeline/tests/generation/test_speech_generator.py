"""Unit tests for SpeechGenerator."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from voice_pipeline.core.interfaces import (
    ILLM,
    ITTS,
    IConversationHistory,
    IMemoryRetriever,
)
from voice_pipeline.core.types import (
    GeneratorState,
    HistoryTurn,
    LLMResult,
    LLMStream,
    ResponseData,
    TTSStream,
    WordTimestamp,
)
from voice_pipeline.generation.speech_generator import SpeechGenerator
from voice_pipeline.memory.types import Episode, MemoryReadResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tts_mock() -> MagicMock:
    """Create a Mock TTS with required ITTS attributes set.

    Abstract properties (output_sample_rate, voice_id) are recognized by
    spec=ITTS but not auto-populated, so we set them explicitly here.
    """
    tts = MagicMock(spec=ITTS)
    tts.output_sample_rate = 24000
    tts.voice_id = "test|mock"
    return tts


def _make_llm_stream(
    chunks: list[str],
    result: LLMResult | None = None,
) -> LLMStream:
    """Create an LLMStream from a list of text chunks."""

    def gen():
        yield from chunks

    return LLMStream(
        gen(),
        result_fn=lambda t: result or LLMResult(text=t),
    )


def _make_tts_stream(
    chunks: list[bytes],
    timestamps: tuple[WordTimestamp, ...] = (),
) -> TTSStream:
    """Create a TTSStream from a list of chunks."""

    def gen():
        yield from chunks

    return TTSStream(gen(), timestamps_fn=lambda: timestamps)


def _make_slow_tts_stream(
    chunks: list[bytes],
    delay: float = 0.05,
    timestamps: tuple[WordTimestamp, ...] = (),
) -> TTSStream:
    """Create a TTSStream that yields chunks with a delay."""

    def gen():
        for chunk in chunks:
            time.sleep(delay)
            yield chunk

    return TTSStream(gen(), timestamps_fn=lambda: timestamps)


def _make_history_mock() -> MagicMock:
    """Create a mock IConversationHistory with default behavior."""
    history = MagicMock(spec=IConversationHistory)
    history.get_turns.return_value = []
    return history


def _make_generator(
    *,
    llm_chunks: list[str] | None = None,
    tts_chunks: list[bytes] | None = None,
    tts_timestamps: tuple[WordTimestamp, ...] = (),
    retriever: MagicMock | None = None,
    session_id: str | None = None,
) -> tuple[SpeechGenerator, dict[str, MagicMock]]:
    """Create a SpeechGenerator with mock dependencies."""
    llm = MagicMock(spec=ILLM)
    if llm_chunks is not None:
        llm.generate.return_value = _make_llm_stream(llm_chunks)
    else:
        llm.generate.return_value = _make_llm_stream(["Hello", " there!"])

    tts = _make_tts_mock()
    if tts_chunks is not None:
        tts.synthesize.return_value = _make_tts_stream(tts_chunks, tts_timestamps)
    else:
        tts.synthesize.return_value = _make_tts_stream([b"\x00" * 100, b"\x01" * 100], tts_timestamps)

    history = _make_history_mock()

    gen = SpeechGenerator(
        llm=llm,
        tts=tts,
        history=history,
        token_counter=lambda t: max(1, len(t) // 4),
        system_prompt="test",
        retriever=retriever,
        session_id=session_id,
    )
    return gen, {"llm": llm, "tts": tts, "history": history}


def _wait_for_state(
    gen: SpeechGenerator,
    target: GeneratorState,
    timeout: float = 2.0,
) -> None:
    """Wait until generator reaches the target state."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if gen.state == target:
            return
        time.sleep(0.01)
    raise TimeoutError(f"Timed out waiting for state {target.value}, got {gen.state.value}")


def _wait_for_stream_done(gen: SpeechGenerator, timeout: float = 2.0) -> None:
    """Wait until stream_done becomes True."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if gen.stream_done:
            return
        time.sleep(0.01)
    raise TimeoutError("Timed out waiting for stream_done")


def _drain_audio(gen: SpeechGenerator, timeout: float = 2.0) -> list[bytes]:
    """Poll all audio chunks until stream_done."""
    chunks: list[bytes] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        chunk = gen.poll_audio()
        if chunk is not None:
            chunks.append(chunk)
        elif gen.stream_done:
            break
        else:
            time.sleep(0.01)
    return chunks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPrepareAndStream:
    """prepare → PREPARING → STREAMING → poll all → stream_done → get_response_data → IDLE."""

    def test_full_lifecycle(self):
        gen, mocks = _make_generator(
            tts_timestamps=(WordTimestamp("Hello", 0.0, 0.3),),
        )

        assert gen.state == GeneratorState.IDLE

        gen.prepare("hello")
        # Should transition through PREPARING → STREAMING
        _wait_for_state(gen, GeneratorState.STREAMING)

        chunks = _drain_audio(gen)
        assert len(chunks) == 2
        assert chunks[0] == b"\x00" * 100
        assert chunks[1] == b"\x01" * 100

        assert gen.stream_done
        assert gen.get_text() == "Hello there!"

        data = gen.get_response_data()
        assert isinstance(data, ResponseData)
        assert data.text == "Hello there!"
        assert data.audio == b"\x00" * 100 + b"\x01" * 100
        assert gen.state == GeneratorState.IDLE

        gen.shutdown()


class TestPrepareRestart:
    """prepare while PREPARING: old run discarded, new run completes."""

    def test_restart_during_preparing(self):
        call_count = 0
        proceed_event = threading.Event()

        def slow_generate(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call blocks until cancelled or timeout
                proceed_event.wait(timeout=2.0)
                return _make_llm_stream(["stale"])
            return _make_llm_stream(["fresh", " response"])

        llm = MagicMock(spec=ILLM)
        llm.generate.side_effect = slow_generate

        tts = _make_tts_mock()
        tts.synthesize.return_value = _make_tts_stream([b"\xaa" * 50])

        history = _make_history_mock()
        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
        )

        gen.prepare("first")
        time.sleep(0.05)  # Let first run start

        gen.prepare("second")
        proceed_event.set()  # Unblock first run

        _wait_for_state(gen, GeneratorState.STREAMING)
        _wait_for_stream_done(gen)

        assert gen.get_text() == "fresh response"
        gen.shutdown()


class TestPrepareDuringStreaming:
    """prepare while STREAMING: old stream abandoned, new run starts."""

    def test_restart_during_streaming(self):
        llm = MagicMock(spec=ILLM)
        llm.generate.return_value = _make_llm_stream(["response"])

        call_count = 0

        def make_tts_stream(text):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_slow_tts_stream([b"\x01"] * 20, delay=0.05)
            return _make_tts_stream([b"\x02" * 50])

        tts = _make_tts_mock()
        tts.synthesize.side_effect = make_tts_stream

        history = _make_history_mock()
        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
        )

        gen.prepare("first")
        _wait_for_state(gen, GeneratorState.STREAMING)

        # Restart while streaming
        llm.generate.return_value = _make_llm_stream(["new response"])
        gen.prepare("second")
        _wait_for_state(gen, GeneratorState.STREAMING)
        _wait_for_stream_done(gen)

        assert gen.get_text() == "new response"
        chunks = _drain_audio(gen)
        # All chunks must be from the fresh run only (no stale \x01 bytes)
        for c in chunks:
            assert c == b"\x02" * 50, f"Stale chunk detected: {c!r}"

        gen.shutdown()


class TestCancel:
    """cancel during PREPARING → IDLE; cancel during STREAMING → IDLE."""

    def test_cancel_during_preparing(self):
        block_event = threading.Event()

        def slow_generate(messages):
            block_event.wait(timeout=2.0)
            return _make_llm_stream(["text"])

        llm = MagicMock(spec=ILLM)
        llm.generate.side_effect = slow_generate

        tts = _make_tts_mock()

        history = _make_history_mock()
        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
        )

        gen.prepare("text")
        time.sleep(0.05)
        gen.cancel()
        assert gen.state == GeneratorState.IDLE

        block_event.set()
        gen.shutdown()

    def test_cancel_during_streaming(self):
        gen, mocks = _make_generator()
        mocks["tts"].synthesize.return_value = _make_slow_tts_stream([b"\x00"] * 20, delay=0.05)
        gen.prepare("text")
        _wait_for_state(gen, GeneratorState.STREAMING)

        gen.cancel()
        assert gen.state == GeneratorState.IDLE
        # Cancel should clear queued audio and state
        assert gen.poll_audio() is None
        assert not gen.stream_done

        gen.shutdown()


class TestPollAudioEmpty:
    """poll_audio returns None when queue empty, stream_done is False."""

    def test_empty_poll(self):
        gen, mocks = _make_generator()

        assert gen.poll_audio() is None
        assert not gen.stream_done

        gen.shutdown()


class TestStreamDone:
    """stream_done becomes True after TTS producer finishes all chunks."""

    def test_stream_done_flag(self):
        gen, mocks = _make_generator()

        gen.prepare("hello")
        _wait_for_stream_done(gen)

        assert gen.stream_done
        gen.shutdown()


class TestGetTextNotReady:
    """get_text before STREAMING → RuntimeError."""

    def test_get_text_idle(self):
        gen, mocks = _make_generator()

        with pytest.raises(RuntimeError, match="not available"):
            gen.get_text()

        gen.shutdown()


class TestGetResponseDataIdempotent:
    """get_response_data callable multiple times per run."""

    def test_idempotent_response_data(self):
        gen, mocks = _make_generator()

        gen.prepare("hello")
        _wait_for_stream_done(gen)
        _drain_audio(gen)

        # First call transitions to IDLE
        data1 = gen.get_response_data()
        assert data1.text == "Hello there!"

        # Second call returns same data (idempotent until next prepare)
        data2 = gen.get_response_data()
        assert data2.text == data1.text
        assert data2.audio == data1.audio

        # Text is still accessible after get_response_data
        assert gen.get_text() == "Hello there!"

        gen.shutdown()


class TestGetResponseDataNotDone:
    """get_response_data before stream_done → RuntimeError."""

    def test_not_done(self):
        gen, mocks = _make_generator()

        with pytest.raises(RuntimeError):
            gen.get_response_data()

        gen.shutdown()


class TestPipelineFailure:
    """LLM raises → FAILED; TTS raises → FAILED."""

    def test_llm_error(self):
        llm = MagicMock(spec=ILLM)
        llm.generate.side_effect = RuntimeError("LLM error")

        tts = _make_tts_mock()

        history = _make_history_mock()
        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
        )
        gen.prepare("hello")
        _wait_for_state(gen, GeneratorState.FAILED)

        assert gen.state == GeneratorState.FAILED
        tts.synthesize.assert_not_called()

        gen.shutdown()

    def test_tts_error(self):
        llm = MagicMock(spec=ILLM)
        llm.generate.return_value = _make_llm_stream(["some text"])

        tts = _make_tts_mock()
        tts.synthesize.side_effect = RuntimeError("TTS error")

        history = _make_history_mock()
        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
        )
        gen.prepare("hello")
        _wait_for_state(gen, GeneratorState.FAILED)

        assert gen.state == GeneratorState.FAILED
        gen.shutdown()


class TestEmptyLLMResponse:
    """LLM returns empty text → FAILED (no TTS call)."""

    def test_empty_text(self):
        llm = MagicMock(spec=ILLM)
        llm.generate.return_value = _make_llm_stream(["", "  ", ""])

        tts = _make_tts_mock()

        history = _make_history_mock()
        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
        )
        gen.prepare("hello")
        _wait_for_state(gen, GeneratorState.FAILED)

        tts.synthesize.assert_not_called()
        gen.shutdown()

    def test_whitespace_only(self):
        llm = MagicMock(spec=ILLM)
        llm.generate.return_value = _make_llm_stream(["   \n\t  "])

        tts = _make_tts_mock()

        history = _make_history_mock()
        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
        )
        gen.prepare("hello")
        _wait_for_state(gen, GeneratorState.FAILED)

        tts.synthesize.assert_not_called()
        gen.shutdown()


class TestZeroChunkTTS:
    """TTS stream yields nothing → FAILED."""

    def test_zero_chunks(self):
        gen, mocks = _make_generator(llm_chunks=["some text"], tts_chunks=[])
        gen.prepare("hello")
        _wait_for_state(gen, GeneratorState.FAILED)

        assert gen.state == GeneratorState.FAILED
        gen.shutdown()


class TestShutdown:
    """shutdown cancels in-flight and cleans up executor."""

    def test_shutdown_cancels(self):
        cancel_observed = threading.Event()

        def cancellable_generate(messages):
            # Wait on the cancel_event that shutdown() will set
            cancel_observed.wait(timeout=2.0)
            return _make_llm_stream(["text"])

        llm = MagicMock(spec=ILLM)
        llm.generate.side_effect = cancellable_generate

        tts = _make_tts_mock()

        history = _make_history_mock()
        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
        )
        # Capture the cancel event before submitting
        gen.prepare("hello")
        with gen._lock:
            cancel_event = gen._cancel_event
        # Wire: when cancel_event is set, unblock the LLM mock
        threading.Thread(
            target=lambda: (cancel_event.wait(timeout=2.0), cancel_observed.set()),
            daemon=True,
        ).start()
        time.sleep(0.05)

        gen.shutdown()  # Sets cancel_event → unblocks LLM → executor shuts down
        assert cancel_event.is_set()


class TestStaleRunDiscarded:
    """Old run completes after new prepare(), no state/queue contamination."""

    def test_stale_run_no_contamination(self):
        call_count = 0
        first_run_event = threading.Event()

        def sequenced_generate(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                first_run_event.wait(timeout=2.0)
                return _make_llm_stream(["stale text"])
            return _make_llm_stream(["fresh text"])

        llm = MagicMock(spec=ILLM)
        llm.generate.side_effect = sequenced_generate

        tts = _make_tts_mock()
        tts.synthesize.return_value = _make_tts_stream([b"\xff" * 50])

        history = _make_history_mock()
        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
        )

        gen.prepare("first")
        time.sleep(0.05)

        gen.prepare("second")
        first_run_event.set()

        _wait_for_stream_done(gen)
        _drain_audio(gen)

        assert gen.get_text() == "fresh text"
        # No stale chunks in queue
        assert gen.poll_audio() is None

        gen.shutdown()


class TestExternalExecutor:
    """External executor injection: works normally, shutdown doesn't close it."""

    def test_external_executor_works(self):
        executor = ThreadPoolExecutor(max_workers=2)
        llm = MagicMock(spec=ILLM)
        llm.generate.return_value = _make_llm_stream(["Hello", " there!"])
        tts = _make_tts_mock()
        tts.synthesize.return_value = _make_tts_stream([b"\x00" * 100, b"\x01" * 100])
        history = _make_history_mock()

        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
            executor=executor,
        )

        gen.prepare("hello")
        _wait_for_stream_done(gen)
        chunks = _drain_audio(gen)
        assert len(chunks) == 2
        assert gen.get_text() == "Hello there!"

        gen.shutdown()
        # External executor should still be alive
        assert not executor._shutdown
        executor.shutdown(wait=True)

    def test_shutdown_does_not_close_external_executor(self):
        executor = ThreadPoolExecutor(max_workers=2)
        llm = MagicMock(spec=ILLM)
        llm.generate.return_value = _make_llm_stream(["Hello", " there!"])
        tts = _make_tts_mock()
        tts.synthesize.return_value = _make_tts_stream([b"\x00" * 100, b"\x01" * 100])
        history = _make_history_mock()

        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
            executor=executor,
        )
        gen.shutdown()

        # Verify executor is still functional by submitting work
        future = executor.submit(lambda: 42)
        assert future.result(timeout=1.0) == 42

        executor.shutdown(wait=True)


class TestInputTextLifecycle:
    """input_text property: set on prepare, cleared on cancel/reset/get_response_data."""

    def test_prepare_sets_input_text(self):
        gen, mocks = _make_generator()

        assert gen.input_text == ""
        gen.prepare("hello")
        assert gen.input_text == "hello"

        _wait_for_stream_done(gen)
        gen.shutdown()

    def test_cancel_clears_input_text(self):
        gen, mocks = _make_generator()

        gen.prepare("hello")
        assert gen.input_text == "hello"

        gen.cancel()
        assert gen.input_text == ""

        gen.shutdown()

    def test_reset_clears_input_text(self):
        gen, mocks = _make_generator()

        gen.prepare("hello")
        assert gen.input_text == "hello"

        gen.reset()
        assert gen.input_text == ""

        gen.shutdown()

    def test_consecutive_prepare_overwrites(self):
        block_event = threading.Event()

        def slow_generate(messages):
            block_event.wait(timeout=2.0)
            return _make_llm_stream(["text"])

        llm = MagicMock(spec=ILLM)
        llm.generate.side_effect = slow_generate

        tts = _make_tts_mock()
        tts.synthesize.return_value = _make_tts_stream([b"\x00" * 50])

        history = _make_history_mock()
        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
        )

        gen.prepare("first")
        assert gen.input_text == "first"

        gen.prepare("second")
        assert gen.input_text == "second"

        block_event.set()
        gen.shutdown()

    def test_get_response_data_clears_input_text(self):
        gen, mocks = _make_generator()

        gen.prepare("hello")
        _wait_for_stream_done(gen)
        _drain_audio(gen)

        assert gen.input_text == "hello"
        gen.get_response_data()
        assert gen.input_text == ""

        gen.shutdown()


class TestCancelDoesNotSetFailed:
    """Cancelled run's late exception doesn't flip to FAILED."""

    def test_cancel_prevents_failed(self):
        proceed_event = threading.Event()

        def failing_generate(messages):
            proceed_event.wait(timeout=2.0)
            raise RuntimeError("Late failure")

        llm = MagicMock(spec=ILLM)
        llm.generate.side_effect = failing_generate

        tts = _make_tts_mock()

        history = _make_history_mock()
        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
        )

        gen.prepare("hello")
        time.sleep(0.05)
        gen.cancel()
        assert gen.state == GeneratorState.IDLE

        proceed_event.set()
        time.sleep(0.1)  # Give time for the exception handler

        # State should still be IDLE, not FAILED
        assert gen.state == GeneratorState.IDLE

        gen.shutdown()


# ---------------------------------------------------------------------------
# Memory integration tests (Phase 4)
# ---------------------------------------------------------------------------


def _make_episode(text: str, eid: int = 1) -> Episode:
    return Episode(
        id=eid,
        text=text,
        timestamp="2026-03-15 14:00:00",
        session_id="s1",
        importance=1.0,
        last_cited_at="2026-03-15 14:00:00",
    )


class TestMemoryIntegration:
    """Tests for retriever integration and citation parsing in the pipeline."""

    def test_retrieve_called_with_query(self) -> None:
        retriever = MagicMock(spec=IMemoryRetriever)
        retriever.retrieve.return_value = MemoryReadResult([], [], {})

        gen, mocks = _make_generator(
            llm_chunks=["Response text"],
            retriever=retriever,
            session_id="current-session",
        )
        gen.prepare("hello")
        _wait_for_stream_done(gen)
        _drain_audio(gen)

        retriever.retrieve.assert_called_once()
        call_args = retriever.retrieve.call_args
        assert "hello" in call_args[0][0]  # query contains current text
        gen.shutdown()

    def test_citation_parsed_and_stripped(self) -> None:
        """LLM output with citation tag → tag stripped, cited_memory_ids populated."""
        llm = MagicMock(spec=ILLM)
        llm.generate.return_value = _make_llm_stream(["Great movie!", "\n[MEMORIES: M1, M2]"])

        tts = _make_tts_mock()
        tts.synthesize.return_value = _make_tts_stream([b"\x00" * 100])

        ep1 = _make_episode("Ep one", eid=10)
        ep2 = _make_episode("Ep two", eid=20)
        mem_result = MemoryReadResult([ep1, ep2], [0.9, 0.8], {1: 10, 2: 20})

        retriever = MagicMock(spec=IMemoryRetriever)
        retriever.retrieve.return_value = mem_result

        history = _make_history_mock()
        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
            retriever=retriever,
        )
        gen.prepare("tell me about movies")
        _wait_for_stream_done(gen)
        _drain_audio(gen)

        data = gen.get_response_data()
        # Tag stripped from text
        assert "[MEMORIES" not in data.text
        assert data.text == "Great movie!"
        # TTS received clean text
        tts.synthesize.assert_called_once_with("Great movie!")
        # Cited IDs resolved
        assert data.cited_memory_ids == [10, 20]
        # Retriever updated
        retriever.update_citations.assert_called_once_with([1, 2])
        gen.shutdown()

    def test_no_citation_tag_no_update(self) -> None:
        """LLM output without citation tag → no update_citations call."""
        retriever = MagicMock(spec=IMemoryRetriever)
        retriever.retrieve.return_value = MemoryReadResult([], [], {})

        gen, mocks = _make_generator(llm_chunks=["Just a response"], retriever=retriever)
        gen.prepare("hi")
        _wait_for_stream_done(gen)
        _drain_audio(gen)

        data = gen.get_response_data()
        assert data.text == "Just a response"
        assert data.cited_memory_ids == []
        retriever.update_citations.assert_not_called()
        gen.shutdown()

    def test_retriever_error_graceful(self) -> None:
        """Retriever failure → pipeline continues without memory."""
        retriever = MagicMock(spec=IMemoryRetriever)
        retriever.retrieve.side_effect = RuntimeError("DB error")

        gen, mocks = _make_generator(llm_chunks=["Fallback response"], retriever=retriever)
        gen.prepare("hi")
        _wait_for_stream_done(gen)
        _drain_audio(gen)

        data = gen.get_response_data()
        assert data.text == "Fallback response"
        gen.shutdown()

    def test_query_includes_history_turns(self) -> None:
        """Retriever query includes recent history turns."""
        retriever = MagicMock(spec=IMemoryRetriever)
        retriever.retrieve.return_value = MemoryReadResult([], [], {})

        llm = MagicMock(spec=ILLM)
        llm.generate.return_value = _make_llm_stream(["Response"])
        tts = _make_tts_mock()
        tts.synthesize.return_value = _make_tts_stream([b"\x00" * 100, b"\x01" * 100])

        history = MagicMock(spec=IConversationHistory)
        history.get_turns.return_value = [
            HistoryTurn(items=({"role": "user", "content": "prev question"},), token_count=2),
            HistoryTurn(items=({"role": "assistant", "content": "prev answer"},), token_count=2),
        ]

        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
            retriever=retriever,
        )
        gen.prepare("current question")
        _wait_for_stream_done(gen)
        _drain_audio(gen)

        query = retriever.retrieve.call_args[0][0]
        assert "prev question" in query
        assert "prev answer" in query
        assert "current question" in query
        gen.shutdown()

    def test_no_retriever_backward_compatible(self) -> None:
        """Without retriever, pipeline works exactly as before."""
        gen, mocks = _make_generator()
        gen.prepare("hello")
        _wait_for_stream_done(gen)
        _drain_audio(gen)

        data = gen.get_response_data()
        assert data.text == "Hello there!"
        assert data.cited_memory_ids == []
        gen.shutdown()


# ---------------------------------------------------------------------------
# Sentence mode helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def sentence_mode(monkeypatch):
    """Activate sentence mode with min_flush_words=1 for a test.

    Non-autouse: full-mode regression tests in this file must not be
    polluted. Request explicitly via fixture argument.
    """
    monkeypatch.setattr(SpeechGenerator, "_PIPELINE_MODE", "sentence")
    monkeypatch.setattr(SpeechGenerator, "_MIN_FLUSH_WORDS", 1)


def _make_sequential_tts(
    responses: list[list[bytes]],
    timestamps_per_call: list[tuple[WordTimestamp, ...]] | None = None,
) -> MagicMock:
    """Create a TTS mock that returns different streams for each call."""
    tts = _make_tts_mock()
    call_idx = [0]

    def _side_effect(text: str) -> TTSStream:
        idx = call_idx[0]
        call_idx[0] += 1
        chunks = responses[idx] if idx < len(responses) else [b"\x00" * 50]
        ts = timestamps_per_call[idx] if timestamps_per_call and idx < len(timestamps_per_call) else ()
        return _make_tts_stream(chunks, ts)

    tts.synthesize.side_effect = _side_effect
    return tts


# ---------------------------------------------------------------------------
# Sentence mode tests
# ---------------------------------------------------------------------------


class TestSentenceModeSingleSentence:
    """Single sentence in sentence mode behaves like full mode."""

    def test_single_sentence(self, sentence_mode) -> None:
        llm = MagicMock(spec=ILLM)
        llm.generate.return_value = _make_llm_stream(["Hello there!"])
        tts = _make_sequential_tts([[b"\x00" * 100, b"\x01" * 100]])

        history = _make_history_mock()
        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
        )
        gen.prepare("hi")
        _wait_for_stream_done(gen)
        chunks = _drain_audio(gen)

        data = gen.get_response_data()
        assert data.text == "Hello there!"
        assert len(chunks) >= 1
        assert tts.synthesize.call_count == 1
        gen.shutdown()


class TestSentenceModeMultipleSentences:
    """Multiple sentences → multiple TTS calls, audio in order."""

    def test_two_sentences(self, sentence_mode) -> None:
        llm = MagicMock(spec=ILLM)
        llm.generate.return_value = _make_llm_stream(["First sentence. Second sentence."])
        tts = _make_sequential_tts([[b"\x01" * 100], [b"\x02" * 100]])

        history = _make_history_mock()
        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
        )
        gen.prepare("hi")
        _wait_for_stream_done(gen)
        chunks = _drain_audio(gen)

        data = gen.get_response_data()
        assert "First sentence" in data.text
        assert "Second sentence" in data.text
        assert tts.synthesize.call_count == 2
        # Audio from both sentences present
        all_audio = b"".join(chunks)
        assert b"\x01" * 100 in all_audio
        assert b"\x02" * 100 in all_audio
        # Order: first sentence audio before second
        idx1 = all_audio.index(b"\x01" * 100)
        idx2 = all_audio.index(b"\x02" * 100)
        assert idx1 < idx2
        gen.shutdown()

    def test_streaming_chunks_across_sentences(self, sentence_mode) -> None:
        """LLM yields text in small chunks that span sentence boundaries."""
        llm = MagicMock(spec=ILLM)
        llm.generate.return_value = _make_llm_stream(["Hello ", "there. ", "How are ", "you? "])
        tts = _make_sequential_tts([[b"\x01" * 50], [b"\x02" * 50]])

        history = _make_history_mock()
        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
        )
        gen.prepare("hi")
        _wait_for_stream_done(gen)
        _drain_audio(gen)

        data = gen.get_response_data()
        assert tts.synthesize.call_count == 2
        assert "Hello there" in data.text
        assert "How are you" in data.text
        gen.shutdown()

    def test_three_sentences(self, sentence_mode) -> None:
        llm = MagicMock(spec=ILLM)
        llm.generate.return_value = _make_llm_stream(["First. Second! Third? "])
        tts = _make_sequential_tts([[b"\x01" * 50], [b"\x02" * 50], [b"\x03" * 50]])

        history = _make_history_mock()
        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
        )
        gen.prepare("hi")
        _wait_for_stream_done(gen)
        _drain_audio(gen)

        gen.get_response_data()
        assert tts.synthesize.call_count == 3
        gen.shutdown()


class TestSentenceModeMinFlushWords:
    """Short sentences below threshold accumulate with next."""

    def test_short_accumulated_with_next(self, monkeypatch) -> None:
        monkeypatch.setattr(SpeechGenerator, "_PIPELINE_MODE", "sentence")
        monkeypatch.setattr(SpeechGenerator, "_MIN_FLUSH_WORDS", 4)
        llm = MagicMock(spec=ILLM)
        # "Sure!" (1 word) + "That sounds really great." (4 words) = 5 total
        llm.generate.return_value = _make_llm_stream(["Sure! That sounds really great."])
        tts = _make_sequential_tts([[b"\x00" * 100]])

        history = _make_history_mock()
        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
        )
        gen.prepare("hi")
        _wait_for_stream_done(gen)
        _drain_audio(gen)

        data = gen.get_response_data()
        # "Sure!" was accumulated with next — only 1 TTS call
        assert tts.synthesize.call_count == 1
        assert "Sure" in data.text
        assert "great" in data.text
        gen.shutdown()


class TestSentenceModeCitation:
    """Citation tag handling in sentence mode."""

    def test_citation_stripped(self, sentence_mode) -> None:
        llm = MagicMock(spec=ILLM)
        llm.generate.return_value = _make_llm_stream(["Great movie! [MEMORIES: M1, M2]"])
        tts = _make_sequential_tts([[b"\x00" * 100]])

        ep1 = _make_episode("Ep one", eid=10)
        ep2 = _make_episode("Ep two", eid=20)
        mem_result = MemoryReadResult([ep1, ep2], [0.9, 0.8], {1: 10, 2: 20})
        retriever = MagicMock(spec=IMemoryRetriever)
        retriever.retrieve.return_value = mem_result

        history = _make_history_mock()
        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
            retriever=retriever,
        )
        gen.prepare("movies")
        _wait_for_stream_done(gen)
        _drain_audio(gen)

        data = gen.get_response_data()
        assert "[MEMORIES" not in data.text
        assert "Great movie" in data.text
        assert data.cited_memory_ids == [10, 20]
        retriever.update_citations.assert_called_once_with([1, 2])
        gen.shutdown()

    def test_citation_after_multiple_sentences(self, sentence_mode) -> None:
        llm = MagicMock(spec=ILLM)
        llm.generate.return_value = _make_llm_stream(["First sentence. Second sentence. [MEMORIES: M1]"])
        tts = _make_sequential_tts([[b"\x01" * 50], [b"\x02" * 50]])

        ep = _make_episode("Ep", eid=5)
        mem_result = MemoryReadResult([ep], [0.9], {1: 5})
        retriever = MagicMock(spec=IMemoryRetriever)
        retriever.retrieve.return_value = mem_result

        history = _make_history_mock()
        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
            retriever=retriever,
        )
        gen.prepare("hi")
        _wait_for_stream_done(gen)
        _drain_audio(gen)

        data = gen.get_response_data()
        assert "[MEMORIES" not in data.text
        assert tts.synthesize.call_count == 2
        assert data.cited_memory_ids == [5]
        gen.shutdown()


class TestSentenceModeCancel:
    """Cancel during sentence pipeline."""

    def test_cancel_returns_idle(self, sentence_mode) -> None:
        llm = MagicMock(spec=ILLM)
        # Slow LLM: give time to cancel
        llm.generate.return_value = _make_llm_stream(["Hello. World. "])
        tts = _make_sequential_tts([[b"\x00" * 100], [b"\x00" * 100]])

        history = _make_history_mock()
        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
        )
        gen.prepare("hi")
        gen.cancel()

        time.sleep(0.2)  # Let pipeline wind down
        assert gen.state == GeneratorState.IDLE
        gen.shutdown()


class TestSentenceModeEmptyLLM:
    """Empty LLM output → FAILED in sentence mode."""

    def test_empty_text(self, sentence_mode) -> None:
        llm = MagicMock(spec=ILLM)
        llm.generate.return_value = _make_llm_stream([""])
        tts = _make_tts_mock()

        history = _make_history_mock()
        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
        )
        gen.prepare("hi")
        _wait_for_state(gen, GeneratorState.FAILED)

        assert gen.state == GeneratorState.FAILED
        gen.shutdown()

    def test_whitespace_only(self, sentence_mode) -> None:
        llm = MagicMock(spec=ILLM)
        llm.generate.return_value = _make_llm_stream(["   \n  "])
        tts = _make_tts_mock()

        history = _make_history_mock()
        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
        )
        gen.prepare("hi")
        _wait_for_state(gen, GeneratorState.FAILED)

        assert gen.state == GeneratorState.FAILED
        gen.shutdown()


class TestSentenceModeStateTransitions:
    """PREPARING → STREAMING → stream_done lifecycle."""

    def test_state_progression(self, sentence_mode) -> None:
        llm = MagicMock(spec=ILLM)
        llm.generate.return_value = _make_llm_stream(["Hello world. "])
        tts = _make_sequential_tts([[b"\x00" * 100]])

        history = _make_history_mock()
        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
        )
        gen.prepare("hi")

        assert gen.state == GeneratorState.PREPARING
        _wait_for_state(gen, GeneratorState.STREAMING)
        assert gen.state == GeneratorState.STREAMING

        _wait_for_stream_done(gen)
        assert gen.stream_done

        _drain_audio(gen)
        data = gen.get_response_data()
        assert gen.state == GeneratorState.IDLE
        assert data.text == "Hello world."
        gen.shutdown()

    def test_get_text_during_streaming(self, sentence_mode) -> None:
        """get_text() returns accumulated text while streaming."""
        llm = MagicMock(spec=ILLM)
        llm.generate.return_value = _make_llm_stream(["Hello world. "])
        tts = _make_sequential_tts([[b"\x00" * 100]])

        history = _make_history_mock()
        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
        )
        gen.prepare("hi")
        _wait_for_state(gen, GeneratorState.STREAMING)

        text = gen.get_text()
        assert "Hello world" in text
        gen.shutdown()


class TestSentenceModeTimestamps:
    """Timestamp offset correction across sentences."""

    def test_timestamps_offset_adjusted(self, sentence_mode) -> None:
        llm = MagicMock(spec=ILLM)
        llm.generate.return_value = _make_llm_stream(["First. Second. "])

        # Sentence 1: 48000 bytes = 1.0s at 24kHz 16-bit
        # Sentence 2: timestamps should be offset by 1.0s
        ts1 = (WordTimestamp(word="First", start_sec=0.0, end_sec=0.5),)
        ts2 = (WordTimestamp(word="Second", start_sec=0.0, end_sec=0.6),)
        tts = _make_sequential_tts(
            [[b"\x00" * 48000], [b"\x00" * 48000]],
            timestamps_per_call=[ts1, ts2],
        )

        history = _make_history_mock()
        gen = SpeechGenerator(
            llm=llm,
            tts=tts,
            history=history,
            token_counter=lambda t: max(1, len(t) // 4),
            system_prompt="test",
        )
        gen.prepare("hi")
        _wait_for_stream_done(gen)
        _drain_audio(gen)

        data = gen.get_response_data()
        assert len(data.timestamps) == 2
        # First sentence timestamps unchanged
        assert data.timestamps[0].word == "First"
        assert abs(data.timestamps[0].start_sec - 0.0) < 0.01
        # Second sentence timestamps offset by 1.0s
        assert data.timestamps[1].word == "Second"
        assert abs(data.timestamps[1].start_sec - 1.0) < 0.01
        assert abs(data.timestamps[1].end_sec - 1.6) < 0.01
        gen.shutdown()


class TestFullModeRegression:
    """Explicit pipeline_mode='full' still works after dispatch refactor."""

    def test_full_mode_explicit(self) -> None:
        gen, mocks = _make_generator()
        gen.prepare("hello")
        _wait_for_stream_done(gen)
        _drain_audio(gen)

        data = gen.get_response_data()
        assert data.text == "Hello there!"
        gen.shutdown()
