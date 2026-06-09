"""Logging handler that captures OpenAI SDK retry events as CallRecords."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

from voice_pipeline.core.interfaces import ICallStore
from voice_pipeline.core.types import CallRecord

_PATTERN = re.compile(r"Retrying request to (/\S+) in ([\d.]+) seconds")

_ENDPOINT_MAP: dict[str, tuple[str, str]] = {
    "/audio/speech": ("tts", "synthesize"),
    "/responses": ("llm", "generate"),
    "/embeddings": ("embedder", "embed"),
}


class OpenAIRetryHandler(logging.Handler):
    """Intercepts OpenAI SDK retry log messages and records them.

    Attach to ``logging.getLogger("openai._base_client")`` to capture
    retry events across all OpenAI API calls (TTS, LLM, embeddings).

    Args:
        call_store: Store to persist call records.
    """

    def __init__(self, call_store: ICallStore) -> None:
        super().__init__()
        self._store = call_store
        self.session_id: str = ""

    def emit(self, record: logging.LogRecord) -> None:
        m = _PATTERN.search(record.getMessage())
        if not m:
            return

        endpoint = m.group(1)
        delay_sec = m.group(2)
        module, operation = _ENDPOINT_MAP.get(endpoint, ("unknown", endpoint))

        call_record = CallRecord(
            session_id=self.session_id,
            timestamp=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
            module=module,
            operation=operation,
            model="",
            elapsed_ms=0.0,
            status="retry",
            metadata=f'{{"endpoint": "{endpoint}", "retry_delay_sec": {delay_sec}}}',
        )
        try:
            self._store.record(call_record)
        except Exception:
            pass
