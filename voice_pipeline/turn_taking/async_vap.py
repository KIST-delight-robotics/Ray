"""Async VAP wrapper — runs VAP inference on a dedicated background thread.

Implements IVAP so it can be used as a drop-in replacement in TurnDetector.
The background thread runs at a fixed rate (default 10Hz), draining buffered
audio frames and running inference without blocking the main frame loop.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import UTC, datetime

from voice_pipeline.core.interfaces import ICallStore, IVAP
from voice_pipeline.core.types import AudioFrame, CallRecord, VAPResult

logger = logging.getLogger("voice_pipeline.turn_taking")


class AsyncVAP(IVAP):
    """IVAP wrapper that runs inference on a background thread.

    ``feed_audio()`` buffers the audio pair and returns the latest cached result
    immediately (non-blocking). A daemon thread drains the buffer at *frame_rate*
    Hz and updates the cached result after each inference call.

    Args:
        vap: The underlying (synchronous) IVAP implementation.
        frame_rate: Target inference rate in Hz. Defaults to 10.
        call_store: Optional call store for latency recording. Records are
            buffered in memory and flushed on ``stop()``.
        session_id: Session identifier for call records.
    """

    def __init__(
        self,
        vap: IVAP,
        frame_rate: int = 10,
        call_store: ICallStore | None = None,
        session_id: str = "",
    ) -> None:
        self._vap = vap
        self._interval = 1.0 / frame_rate
        self._buffer: list[tuple[AudioFrame, AudioFrame | None]] = []
        self._buffer_lock = threading.Lock()
        self._latest_result = VAPResult(0.0, 0.0, False)
        self._call_store = call_store
        self._session_id = session_id
        self._call_records: list[CallRecord] = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="async-vap")
        self._thread.start()

    def feed_audio(self, user_audio: AudioFrame, robot_audio: AudioFrame | None = None) -> VAPResult:
        """Buffer audio and return the latest cached VAP result (non-blocking)."""
        with self._buffer_lock:
            self._buffer.append((user_audio, robot_audio))
        return self._latest_result

    def reset(self) -> None:
        """Clear buffer and reset the underlying VAP.

        Called at session start when the inference thread is idle.
        """
        with self._buffer_lock:
            self._buffer.clear()
        self._vap.reset()
        self._latest_result = VAPResult(0.0, 0.0, False)

    def stop(self) -> None:
        """Signal the background thread to exit, wait, and flush call records."""
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._flush_call_records()

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _drain_buffer(self) -> list[tuple[AudioFrame, AudioFrame | None]]:
        with self._buffer_lock:
            items = self._buffer[:]
            self._buffer.clear()
        return items

    def _run(self) -> None:
        while not self._stop_event.is_set():
            start = time.monotonic()

            audio_pairs = self._drain_buffer()
            if audio_pairs:
                user_combined = b"".join(u for u, _ in audio_pairs)
                robot_parts = [r for _, r in audio_pairs if r is not None]
                robot_combined = b"".join(robot_parts) if robot_parts else None
                result = self._vap.feed_audio(user_combined, robot_combined)
                self._latest_result = result  # atomic ref write (CPython)

            elapsed = time.monotonic() - start
            remaining = self._interval - elapsed
            if audio_pairs and remaining <= 0:
                logger.warning(
                    "VAP cycle overrun: %.0fms (budget %.0fms, behind %.0fms)",
                    elapsed * 1000,
                    self._interval * 1000,
                    -remaining * 1000,
                )

            if audio_pairs and self._call_store is not None:
                status = "ok" if remaining > 0 else "overrun"
                self._call_records.append(CallRecord(
                    session_id=self._session_id,
                    timestamp=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
                    module="vap",
                    operation="feed_audio",
                    model="maai-vap",
                    elapsed_ms=elapsed * 1000,
                    status=status,
                ))

            if remaining > 0:
                self._stop_event.wait(remaining)

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
            logger.warning("Failed to flush VAP call records", exc_info=True)
        self._call_records.clear()
