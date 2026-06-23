"""Call-tracking decorator for ITTS."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Generator

from voice_pipeline.core.interfaces import ITTS, ICallStore
from voice_pipeline.core.types import CallRecord, TTSStream, utc_now_str

logger = logging.getLogger("voice_pipeline.trace")


class TrackedTTS(ITTS):
    """ITTS wrapper that records per-call execution data to an ICallStore.

    Transparent drop-in: all ITTS methods delegate to the inner TTS,
    with timing and status recorded around ``synthesize`` and across the
    returned stream's chunk delivery (``operation="stream"``) — slow or
    stalled network delivery is classified from chunk arrival timing.

    Args:
        inner: The underlying TTS implementation.
        call_store: Store to persist call records.
    """

    _STALL_GAP_MS = 500.0  # 청크 간 공백이 이 값 이상이면 stalled — 네트워크 스톨
    _SLOW_HEADROOM_SEC = 0.0  # (수신 오디오 − 경과 시간) 최소값이 이 값 미만이면 slow — 실시간 미달

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
            return TTSStream(
                self._monitor_stream(stream, len(text)),
                close_fn=stream.close,
                timestamps_fn=lambda: stream.timestamps,
            )
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

    def _monitor_stream(self, stream: TTSStream, text_len: int) -> Generator[bytes, None, None]:
        """Yield chunks from *stream*, recording delivery timing on exhaustion or close.

        chunk 간 간격에는 소비자(SpeechGenerator)의 청크 처리 시간도 포함되지만,
        소비 측은 큐 적재뿐이라 사실상 네트워크 수신 타이밍을 측정한다.
        """
        bytes_per_sec = self._inner.output_sample_rate * 2
        t_start = time.monotonic()
        t_first = 0.0
        t_prev = 0.0
        audio_bytes = 0
        max_gap_ms = 0.0
        min_headroom_sec = math.inf
        completed = False
        warned = False
        error: str | None = None
        try:
            for chunk in stream:
                now = time.monotonic()
                if audio_bytes == 0:
                    t_first = now
                else:
                    gap_ms = (now - t_prev) * 1000
                    if gap_ms > max_gap_ms:
                        max_gap_ms = gap_ms
                t_prev = now
                audio_bytes += len(chunk)
                headroom_sec = audio_bytes / bytes_per_sec - (now - t_first)
                if headroom_sec < min_headroom_sec:
                    min_headroom_sec = headroom_sec
                if not warned and (max_gap_ms >= self._STALL_GAP_MS or headroom_sec < self._SLOW_HEADROOM_SEC):
                    warned = True
                    logger.warning(
                        "TTS stream slow: headroom=%.2fs max_gap=%.0fms (audio %.1fs received)",
                        headroom_sec,
                        max_gap_ms,
                        audio_bytes / bytes_per_sec,
                    )
                yield chunk
            completed = True
        except GeneratorExit:
            raise  # cancelled by close() — finally records partial stats
        except Exception as exc:
            error = str(exc)[:200]
            raise
        finally:
            if audio_bytes > 0 or error is not None:
                if error is not None:
                    status = "error"
                elif max_gap_ms >= self._STALL_GAP_MS:
                    status = "stalled"
                elif min_headroom_sec < self._SLOW_HEADROOM_SEC:
                    status = "slow"
                else:
                    status = "ok"
                metadata = {
                    "text_len": text_len,
                    "audio_sec": round(audio_bytes / bytes_per_sec, 2),
                    "ttfc_ms": round((t_first - t_start) * 1000, 1) if audio_bytes else None,
                    "min_headroom_sec": (round(min_headroom_sec, 3) if math.isfinite(min_headroom_sec) else None),
                    "max_gap_ms": round(max_gap_ms, 1),
                    "completed": completed,
                }
                if error is not None:
                    metadata["error"] = error
                self._record(
                    "stream",
                    (time.monotonic() - t_start) * 1000,
                    status=status,
                    metadata=json.dumps(metadata),
                )

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
            module="tts",
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
            logger.warning("Failed to record TTS call", exc_info=True)
