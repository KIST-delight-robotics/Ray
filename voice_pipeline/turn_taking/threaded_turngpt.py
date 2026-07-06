"""Threaded TurnGPT — runs TurnGPT inference on a dedicated background thread.

Provides a submit/poll interface instead of the synchronous ITurnGPT.predict().
A SyncTurnGPTAdapter is also provided for unit tests that use mock ITurnGPT.
"""

from __future__ import annotations

import logging
import threading
import time

from voice_pipeline.core.interfaces import ICallStore, ITurnGPT
from voice_pipeline.core.types import CallRecord, utc_now_str

logger = logging.getLogger("voice_pipeline.turn_taking")


class ThreadedTurnGPT:
    """Threaded TurnGPT wrapper with submit/poll pattern.

    ``submit()`` enqueues text for background inference (fire-and-forget).
    ``poll_result()`` returns the latest result or None.

    If ``clear_pending()`` is called before inference completes, the stale
    result is automatically discarded.

    Args:
        turngpt: The underlying (synchronous) ITurnGPT implementation.
        call_store: Optional call store for latency recording. Records are
            buffered in memory and flushed on ``stop()``.
        session_id: Session identifier for call records.
    """

    def __init__(
        self,
        turngpt: ITurnGPT,
        call_store: ICallStore | None = None,
        session_id: str = "",
    ) -> None:
        self._turngpt = turngpt
        self._pending_text: str | None = None
        self._latest_prob: float | None = None
        self._lock = threading.Lock()
        self._work_event = threading.Event()
        self._stop_event = threading.Event()
        self._pending_reset = False
        self._call_store = call_store
        self._session_id = session_id
        self._call_records: list[CallRecord] = []
        self._thread = threading.Thread(target=self._run, daemon=True, name="threaded-turngpt")
        self._thread.start()

    def submit(self, dialog_text: str) -> None:
        """Submit text for background TurnGPT inference."""
        with self._lock:
            self._pending_text = dialog_text
        self._work_event.set()

    def poll_result(self) -> float | None:
        """Return the latest inference result, or None if not available.

        Consumes the result — subsequent calls return None until a new
        result is produced.
        """
        with self._lock:
            prob = self._latest_prob
            self._latest_prob = None
            return prob

    def clear_pending(self) -> None:
        """Discard any pending submission and buffered result."""
        with self._lock:
            self._pending_text = None
            self._latest_prob = None

    def reset(self) -> None:
        """Reset TurnGPT state (delegates to background thread for KV cache safety)."""
        with self._lock:
            self._pending_text = None
            self._latest_prob = None
            self._pending_reset = True
        self._work_event.set()

    def stop(self) -> None:
        """Signal the background thread to exit, wait, and flush call records."""
        self._stop_event.set()
        self._work_event.set()
        self._thread.join(timeout=2.0)
        self._flush_call_records()

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._work_event.wait()
            self._work_event.clear()

            if self._stop_event.is_set():
                break

            # Handle reset on the inference thread (KV cache safety)
            with self._lock:
                if self._pending_reset:
                    self._turngpt.reset()
                    self._pending_reset = False
                text = self._pending_text

            if text is None:
                continue

            t0 = time.monotonic()
            prob = self._turngpt.predict(text)
            elapsed_ms = (time.monotonic() - t0) * 1000

            with self._lock:
                # Only store result if submission hasn't been cleared
                if self._pending_text is not None:
                    self._latest_prob = prob
                    if elapsed_ms > 100:
                        logger.warning(
                            "TurnGPT inference slow: %.0fms text=%r",
                            elapsed_ms,
                            text[:60],
                        )
                else:
                    logger.debug(
                        "TurnGPT result discarded (cleared): %.0fms",
                        elapsed_ms,
                    )

            if self._call_store is not None:
                status = "ok" if elapsed_ms <= 100 else "slow"
                self._call_records.append(
                    CallRecord(
                        session_id=self._session_id,
                        timestamp=utc_now_str(),
                        module="turngpt",
                        operation="predict",
                        model="turngpt",
                        elapsed_ms=elapsed_ms,
                        status=status,
                        turn_index=self._call_store.current_turn_index,
                    )
                )

    # ------------------------------------------------------------------
    # Call record flush
    # ------------------------------------------------------------------

    def _flush_call_records(self) -> None:
        if not self._call_store or not self._call_records:
            return
        try:
            for record in self._call_records:
                self._call_store.record(record)
        except Exception:
            logger.warning("Failed to flush TurnGPT call records", exc_info=True)
        self._call_records.clear()


class SyncTurnGPTAdapter:
    """Wraps a synchronous ITurnGPT into the submit/poll interface.

    Inference runs synchronously in ``submit()``, making this suitable
    for unit tests where mocked ITurnGPT returns immediately.
    """

    def __init__(self, turngpt: ITurnGPT) -> None:
        self._turngpt = turngpt
        self._result: float | None = None

    def submit(self, dialog_text: str) -> None:
        """Run predict synchronously and buffer the result."""
        self._result = self._turngpt.predict(dialog_text)

    def poll_result(self) -> float | None:
        """Return buffered result (consumed on read)."""
        r = self._result
        self._result = None
        return r

    def clear_pending(self) -> None:
        """Discard any buffered result."""
        self._result = None

    def reset(self) -> None:
        """Reset the underlying TurnGPT."""
        self._turngpt.reset()

    def stop(self) -> None:
        """No-op for synchronous adapter."""
