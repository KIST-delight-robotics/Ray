"""Tests for OpenAIRetryHandler."""

from __future__ import annotations

import json
import logging

from voice_pipeline.tests.fakes import RecordingCallStore
from voice_pipeline.trace import OpenAIRetryHandler


def _emit(handler: OpenAIRetryHandler, message: str) -> None:
    """Simulate a log record emission."""
    record = logging.LogRecord(
        name="openai._base_client",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=message,
        args=(),
        exc_info=None,
    )
    handler.emit(record)


class TestOpenAIRetryHandler:
    def test_parses_tts_retry(self) -> None:
        store = RecordingCallStore()
        handler = OpenAIRetryHandler(store)
        handler.session_id = "sess-1"

        _emit(handler, "Retrying request to /audio/speech in 0.387022 seconds")

        assert len(store.records) == 1
        rec = store.records[0]
        assert rec.session_id == "sess-1"
        assert rec.module == "tts"
        assert rec.operation == "synthesize"
        assert rec.status == "retry"
        meta = json.loads(rec.metadata)
        assert meta["endpoint"] == "/audio/speech"
        assert meta["retry_delay_sec"] == 0.387022

    def test_parses_llm_retry(self) -> None:
        store = RecordingCallStore()
        handler = OpenAIRetryHandler(store)

        _emit(handler, "Retrying request to /responses in 0.412858 seconds")

        assert len(store.records) == 1
        rec = store.records[0]
        assert rec.module == "llm"
        assert rec.operation == "generate"

    def test_parses_embeddings_retry(self) -> None:
        store = RecordingCallStore()
        handler = OpenAIRetryHandler(store)

        _emit(handler, "Retrying request to /embeddings in 1.5 seconds")

        assert len(store.records) == 1
        rec = store.records[0]
        assert rec.module == "embedder"
        assert rec.operation == "embed"

    def test_unknown_endpoint(self) -> None:
        store = RecordingCallStore()
        handler = OpenAIRetryHandler(store)

        _emit(handler, "Retrying request to /v1/something in 0.5 seconds")

        assert len(store.records) == 1
        rec = store.records[0]
        assert rec.module == "unknown"
        assert rec.operation == "/v1/something"

    def test_ignores_non_retry_message(self) -> None:
        store = RecordingCallStore()
        handler = OpenAIRetryHandler(store)

        _emit(handler, "HTTP Request: POST https://api.openai.com/v1/audio/speech")

        assert len(store.records) == 0

    def test_session_id_propagated(self) -> None:
        store = RecordingCallStore()
        handler = OpenAIRetryHandler(store)
        handler.session_id = "s1"

        _emit(handler, "Retrying request to /audio/speech in 0.5 seconds")
        handler.session_id = "s2"
        _emit(handler, "Retrying request to /audio/speech in 0.5 seconds")

        assert store.records[0].session_id == "s1"
        assert store.records[1].session_id == "s2"

    def test_store_error_swallowed(self) -> None:
        """Handler should not raise even if store fails."""
        from unittest.mock import MagicMock

        store = MagicMock()
        store.record.side_effect = RuntimeError("db error")
        handler = OpenAIRetryHandler(store)

        _emit(handler, "Retrying request to /audio/speech in 0.5 seconds")
        # No exception raised

    def test_integrates_with_logger(self) -> None:
        """Handler works when attached to a real logger."""
        store = RecordingCallStore()
        handler = OpenAIRetryHandler(store)
        handler.session_id = "sess-1"

        logger = logging.getLogger("openai._base_client.test")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            logger.info("Retrying request to /audio/speech in 0.962817 seconds")
        finally:
            logger.removeHandler(handler)

        assert len(store.records) == 1
        assert store.records[0].module == "tts"
