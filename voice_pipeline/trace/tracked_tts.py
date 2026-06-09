"""Call-tracking decorator for ITTS."""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime

from voice_pipeline.core.interfaces import ICallStore, ITTS
from voice_pipeline.core.types import CallRecord, TTSStream

logger = logging.getLogger("voice_pipeline.trace")


class TrackedTTS(ITTS):
    """ITTS wrapper that records per-call execution data to an ICallStore.

    Transparent drop-in: all ITTS methods delegate to the inner TTS,
    with timing and status recorded around ``synthesize``.

    Args:
        inner: The underlying TTS implementation.
        call_store: Store to persist call records.
    """

    def __init__(self, inner: ITTS, call_store: ICallStore) -> None:
        self._inner = inner
        self._store = call_store
        self.session_id: str = ""

    @property
    def output_sample_rate(self) -> int:
        return self._inner.output_sample_rate

    @property
    def voice_id(self) -> str:
        return self._inner.voice_id

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    def synthesize(self, text: str) -> TTSStream:
        t0 = time.monotonic()
        try:
            stream = self._inner.synthesize(text)
            elapsed_ms = (time.monotonic() - t0) * 1000
            self._record(
                "synthesize",
                elapsed_ms,
                status="ok",
                metadata=json.dumps({"text_len": len(text)}),
            )
            return stream
        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            status = "timeout" if "timeout" in str(exc).lower() else "error"
            self._record(
                "synthesize",
                elapsed_ms,
                status=status,
                metadata=json.dumps({"text_len": len(text), "error": str(exc)[:200]}),
            )
            raise

    def _record(
        self,
        operation: str,
        elapsed_ms: float,
        *,
        status: str = "ok",
        metadata: str | None = None,
    ) -> None:
        record = CallRecord(
            session_id=self.session_id,
            timestamp=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
            module="tts",
            operation=operation,
            model=self._inner.model_name,
            elapsed_ms=elapsed_ms,
            status=status,
            metadata=metadata,
        )
        try:
            self._store.record(record)
        except Exception:
            logger.warning("Failed to record TTS call", exc_info=True)
