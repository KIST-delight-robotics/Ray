"""Threaded VAP runtime — runs a VAP model on a dedicated background thread.

Implements the command/query ``IVAP`` the pipeline depends on: ``feed_audio``
buffers audio (non-blocking) and a daemon thread drains the buffer at
*frame_rate* Hz, runs the model, and updates ``latest_result``. The model
itself (e.g. ``MaAIVAPModel``) is a plain inference object held here — it does
not implement ``IVAP``; only this runtime does.

Process-lifetime object: created once, reused across sessions via ``reset()``
(clears the buffer + model state) and the mutable ``session_id`` (re-stamps
call records). ``stop()`` joins the thread at shutdown.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Protocol

from voice_pipeline.core.interfaces import IVAP, ICallStore
from voice_pipeline.core.types import AudioFrame, CallRecord, VAPResult, utc_now_str

logger = logging.getLogger("voice_pipeline.turn_taking")


class _VAPModel(Protocol):
    """Synchronous VAP inference model held by ``ThreadedVAP``."""

    def infer(self, user_audio: AudioFrame, robot_audio: AudioFrame | None = None) -> VAPResult: ...

    def reset(self) -> None: ...


class ThreadedVAP(IVAP):
    """IVAP runtime that runs a VAP model on a background thread.

    ``feed_audio()`` buffers the audio pair and returns immediately. A daemon
    thread drains the buffer at *frame_rate* Hz, runs ``model.infer`` on the
    concatenated audio, and updates the cached ``latest_result``.

    Args:
        model: The synchronous VAP inference model (``infer`` + ``reset``).
        frame_rate: Target inference rate in Hz. Defaults to 10.
        call_store: Optional call store for latency recording. Records are
            buffered in memory and flushed on ``reset()`` / ``stop()``.
        session_id: Session identifier stamped onto call records (mutable).
    """

    def __init__(
        self,
        model: _VAPModel,
        frame_rate: int = 10,
        call_store: ICallStore | None = None,
        session_id: str = "",
    ) -> None:
        self._model = model
        self._interval = 1.0 / frame_rate
        self._buffer: list[tuple[AudioFrame, AudioFrame | None]] = []
        self._buffer_lock = threading.Lock()
        # Serializes model access between the inference thread and a
        # main-thread reset(), since both touch the model's internal state.
        self._infer_lock = threading.Lock()
        self._latest_result = VAPResult(0.0, 0.0, False)
        self._call_store = call_store
        self.session_id = session_id
        self._call_records: list[CallRecord] = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="threaded-vap")
        self._thread.start()

    def feed_audio(self, user_audio: AudioFrame, robot_audio: AudioFrame | None = None) -> None:
        """Buffer one pipeline frame (non-blocking). The thread runs inference."""
        with self._buffer_lock:
            self._buffer.append((user_audio, robot_audio))

    @property
    def latest_result(self) -> VAPResult:
        """Most recent voice-activity estimate (non-consuming)."""
        return self._latest_result

    def reset(self) -> None:
        """Clear buffer, flush pending records, and reset the model.

        Called at session start. The model reset runs under ``_infer_lock`` so
        it cannot race an in-flight inference on the thread.
        """
        with self._buffer_lock:
            self._buffer.clear()
        self._flush_call_records()
        with self._infer_lock:
            self._model.reset()
        self._latest_result = VAPResult(0.0, 0.0, False)

    def stop(self) -> None:
        """Signal the inference thread to exit, wait, and flush call records."""
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._flush_call_records()

    # ------------------------------------------------------------------
    # Inference thread
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
                with self._infer_lock:
                    result = self._model.infer(user_combined, robot_combined)
                self._latest_result = result  # atomic ref write (CPython)

            elapsed = time.monotonic() - start
            remaining = self._interval - elapsed
            if audio_pairs and self._call_store is not None:
                status = "ok" if remaining > 0 else "overrun"
                record = CallRecord(
                    session_id=self.session_id,
                    timestamp=utc_now_str(),
                    module="vap",
                    operation="feed_audio",
                    model="maai-vap",
                    elapsed_ms=elapsed * 1000,
                    status=status,
                    turn_index=self._call_store.current_turn_index,
                )
                with self._buffer_lock:
                    self._call_records.append(record)

            if remaining > 0:
                self._stop_event.wait(remaining)

    def _flush_call_records(self) -> None:
        """Drain buffered call records to the store (safe across threads)."""
        if not self._call_store:
            return
        with self._buffer_lock:
            records = self._call_records[:]
            self._call_records.clear()
        for record in records:
            try:
                self._call_store.record(record)
            except Exception:
                logger.warning("Failed to flush VAP call records", exc_info=True)
                return
