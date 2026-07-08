"""Call-tracking decorator for IEmbedder."""

from __future__ import annotations

import json
import logging
import time

import numpy as np

from voice_pipeline.core.interfaces import ICallStore, IEmbedder
from voice_pipeline.core.types import CallRecord, utc_now_str

logger = logging.getLogger("voice_pipeline.trace")


class TrackedEmbedder(IEmbedder):
    """IEmbedder wrapper that records per-call execution data to an ICallStore.

    Transparent drop-in: all IEmbedder methods delegate to the inner
    embedder, with timing recorded around ``embed`` and ``embed_batch``.

    Args:
        inner: The underlying embedder implementation.
        call_store: Store to persist call records.
    """

    def __init__(self, inner: IEmbedder, call_store: ICallStore) -> None:
        self._inner = inner
        self._store = call_store
        self.session_id: str = ""

    def embed(self, text: str) -> np.ndarray:
        t0 = time.monotonic()
        result = self._inner.embed(text)
        self._record("embed", (time.monotonic() - t0) * 1000)
        return result

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        t0 = time.monotonic()
        result = self._inner.embed_batch(texts)
        self._record(
            "embed_batch",
            (time.monotonic() - t0) * 1000,
            metadata=json.dumps({"count": len(texts)}),
        )
        return result

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    @property
    def model_name(self) -> str:
        return self._inner.model_name

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
            timestamp=utc_now_str(),
            module="embedder",
            operation=operation,
            model=self._inner.model_name,
            elapsed_ms=elapsed_ms,
            status=status,
            metadata=metadata,
            turn_index=self._store.current_turn_index,
        )
        try:
            self._store.record(record)
        except Exception:
            logger.warning("Failed to record embedder call", exc_info=True)
