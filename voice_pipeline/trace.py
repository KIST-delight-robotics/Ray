"""실행 기록 (관측용). 없어도 대화는 돈다.

- ``PipelineTrace`` → ``SQLiteTraceStore``: 턴 하나의 타임스탬프·결과.
- ``CallRecord`` → ``SQLiteCallStore``: 외부 호출 1건(모듈/작업/지연/상태).
- ``TrackedTTS`` / ``TrackedEmbedder``: ITTS / IEmbedder 를 감싸 호출을 기록하는 래퍼.
- ``OpenAIRetryHandler``: openai SDK의 재시도 로그를 CallRecord로 바꾸는 logging 핸들러.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voice_pipeline.types import ITTS, IEmbedder, TTSStream, utc_now_str

logger = logging.getLogger("voice_pipeline.trace")

# ---------------------------------------------------------------------------
# Pipeline latency tracing
# ---------------------------------------------------------------------------


@dataclass
class PipelineTrace:
    """Timing trace for one turn's response generation pipeline.

    Accumulates monotonic timestamps from SpeechGenerator (background
    thread) and Orchestrator (main thread).  Stored once per turn —
    speculative prepare() replacements within a turn are not stored
    individually; only the final pipeline run's timing is recorded.

    ``to_record()`` converts raw timestamps to millisecond durations
    for SQLite storage (monotonic values are meaningless across
    process restarts).

    Thread safety: the SpeechGenerator background thread writes
    pipeline-stage fields while the Orchestrator main thread writes
    orchestrator-level fields.  Individual float assignments are
    atomic under CPython's GIL, and the fields written by each thread
    do not overlap.
    """

    # -- Identity / metadata --
    session_id: str = ""
    run_id: int = 0
    pipeline_mode: str = "full"
    created_at: str = ""
    outcome: str = ""
    speculative_attempts: int = 1

    # -- Link to conversation history --
    user_msg_id: int = 0

    # -- Orchestrator-level monotonic timestamps --
    prepare_ts: float = 0.0
    turn_shift_ts: float = 0.0
    begin_streaming_ts: float = 0.0
    playback_started_ts: float = 0.0

    # -- Turn-shift metadata --
    turn_shift_reason: str = ""

    # -- Interrupt monotonic timestamps --
    interrupt_ts: float = 0.0
    interrupt_ack_ts: float = 0.0

    # -- SpeechGenerator pipeline-stage monotonic timestamps --
    pipeline_start_ts: float = 0.0
    memory_done_ts: float = 0.0
    context_done_ts: float = 0.0
    llm_start_ts: float = 0.0
    llm_first_token_ts: float = 0.0
    llm_done_ts: float = 0.0
    tts_start_ts: float = 0.0
    tts_first_chunk_ts: float = 0.0
    tts_done_ts: float = 0.0

    # -- From LLMMetrics (already in ms) --
    llm_ttft_ms: float = 0.0

    @staticmethod
    def _delta_ms(start: float, end: float) -> float:
        """Compute millisecond delta, returning 0.0 if either timestamp is missing."""
        if start <= 0 or end <= 0 or end < start:
            return 0.0
        return (end - start) * 1000

    def to_record(self) -> dict[str, object]:
        """Convert to a flat dict of computed durations for DB storage."""
        speculative_ms = (
            max(0.0, (self.turn_shift_ts - self.prepare_ts) * 1000)
            if self.turn_shift_ts > 0 and self.prepare_ts > 0
            else 0.0
        )
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "pipeline_mode": self.pipeline_mode,
            "created_at": self.created_at,
            "outcome": self.outcome,
            "speculative_attempts": self.speculative_attempts,
            "user_msg_id": self.user_msg_id,
            "memory_ms": self._delta_ms(self.pipeline_start_ts, self.memory_done_ts),
            "context_ms": self._delta_ms(self.memory_done_ts, self.context_done_ts),
            "llm_ms": self._delta_ms(self.llm_start_ts, self.llm_done_ts),
            "llm_ttft_ms": self.llm_ttft_ms,
            "tts_ms": self._delta_ms(self.tts_start_ts, self.tts_done_ts),
            "tts_ttfc_ms": self._delta_ms(self.tts_start_ts, self.tts_first_chunk_ts),
            "prepare_to_streaming_ms": self._delta_ms(self.prepare_ts, self.tts_first_chunk_ts),
            "turn_shift_to_playback_ms": self._delta_ms(self.turn_shift_ts, self.playback_started_ts),
            "speculative_ms": speculative_ms,
            "bridge_ms": self._delta_ms(self.begin_streaming_ts, self.playback_started_ts),
            "interrupt_latency_ms": self._delta_ms(self.interrupt_ts, self.interrupt_ack_ts),
            "turn_shift_reason": self.turn_shift_reason,
        }

    def summary(self) -> str:
        """One-line latency summary for logging."""
        r = self.to_record()
        parts = [f"outcome={self.outcome}"]
        ts_to_pb = r["turn_shift_to_playback_ms"]
        if ts_to_pb:
            parts.append(f"ts→pb={ts_to_pb:.0f}ms")
        spec = r["speculative_ms"]
        if spec:
            parts.append(f"spec={spec:.0f}ms")
        for key in ("memory_ms", "context_ms", "llm_ms", "tts_ms", "bridge_ms"):
            v = r[key]
            if v:
                label = key.removesuffix("_ms")
                parts.append(f"{label}={v:.0f}ms")
        ttft = r["llm_ttft_ms"]
        if ttft:
            parts.append(f"llm_ttft={ttft:.0f}ms")
        ttfc = r["tts_ttfc_ms"]
        if ttfc:
            parts.append(f"tts_ttfc={ttfc:.0f}ms")
        interrupt_ms = r["interrupt_latency_ms"]
        if interrupt_ms:
            parts.append(f"interrupt={interrupt_ms:.0f}ms")
        if self.speculative_attempts > 1:
            parts.append(f"attempts={self.speculative_attempts}")
        return " | ".join(parts)


@dataclass
class CallRecord:
    """Single module call record — latency, status, and optional metadata.

    ``turn_index`` is the conversation exchange this call belongs to (0-based),
    letting per-call data be attributed to a specific question/turn within a
    multi-turn session. Stamped at construction time from the shared turn
    counter (see SQLiteCallStore.current_turn_index).
    """

    session_id: str
    timestamp: str
    module: str
    operation: str
    model: str
    elapsed_ms: float
    status: str = "ok"
    metadata: str | None = None
    turn_index: int = 0


_COLUMNS = (
    "session_id",
    "run_id",
    "pipeline_mode",
    "created_at",
    "outcome",
    "speculative_attempts",
    "user_msg_id",
    "memory_ms",
    "context_ms",
    "llm_ms",
    "llm_ttft_ms",
    "tts_ms",
    "tts_ttfc_ms",
    "prepare_to_streaming_ms",
    "turn_shift_to_playback_ms",
    "speculative_ms",
    "bridge_ms",
    "interrupt_latency_ms",
    "turn_shift_reason",
)


class SQLiteTraceStore:
    """Persists PipelineTrace records to a SQLite database.

    Opens its own connection to the shared DB file (WAL mode).
    Thread-safe: a lock serializes all connection access.
    """

    def __init__(self, db_path: str) -> None:
        self._lock = threading.Lock()
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_traces (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id              TEXT    NOT NULL,
                run_id                  INTEGER NOT NULL,
                pipeline_mode           TEXT    NOT NULL,
                created_at              TEXT    NOT NULL,
                outcome                 TEXT    NOT NULL,
                speculative_attempts    INTEGER NOT NULL DEFAULT 1,
                user_msg_id             INTEGER NOT NULL DEFAULT 0,
                memory_ms               REAL    NOT NULL DEFAULT 0,
                context_ms              REAL    NOT NULL DEFAULT 0,
                llm_ms                  REAL    NOT NULL DEFAULT 0,
                llm_ttft_ms             REAL    NOT NULL DEFAULT 0,
                tts_ms                  REAL    NOT NULL DEFAULT 0,
                tts_ttfc_ms             REAL    NOT NULL DEFAULT 0,
                prepare_to_streaming_ms REAL    NOT NULL DEFAULT 0,
                turn_shift_to_playback_ms REAL  NOT NULL DEFAULT 0,
                speculative_ms          REAL    NOT NULL DEFAULT 0,
                bridge_ms               REAL    NOT NULL DEFAULT 0,
                interrupt_latency_ms    REAL    NOT NULL DEFAULT 0,
                turn_shift_reason       TEXT    NOT NULL DEFAULT ''
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_session ON pipeline_traces(session_id)")
        self._migrate(self._conn)
        self._conn.commit()

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(pipeline_traces)")}
        if "turn_shift_reason" not in existing:
            conn.execute("ALTER TABLE pipeline_traces ADD COLUMN turn_shift_reason TEXT NOT NULL DEFAULT ''")

    def save(self, trace: PipelineTrace) -> None:
        """Persist a trace record."""
        record = trace.to_record()
        values = tuple(record[col] for col in _COLUMNS)
        placeholders = ", ".join("?" for _ in _COLUMNS)
        col_names = ", ".join(_COLUMNS)
        with self._lock:
            self._conn.execute(
                f"INSERT INTO pipeline_traces ({col_names}) VALUES ({placeholders})",
                values,
            )
            self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()


_CALL_COLUMNS = (
    "session_id",
    "timestamp",
    "module",
    "operation",
    "model",
    "elapsed_ms",
    "status",
    "metadata",
    "turn_index",
)


class SQLiteCallStore:
    """Persists per-call execution records to a SQLite database.

    Opens its own connection to the shared DB file (WAL mode).
    Thread-safe: a lock serializes all connection access.
    """

    def __init__(self, db_path: str) -> None:
        self._lock = threading.Lock()
        self._turn_index = 0
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS call_records (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                module      TEXT NOT NULL,
                operation   TEXT NOT NULL,
                model       TEXT NOT NULL,
                elapsed_ms  REAL NOT NULL,
                status      TEXT NOT NULL DEFAULT 'ok',
                metadata    TEXT,
                turn_index  INTEGER NOT NULL DEFAULT 0
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_call_session ON call_records(session_id)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_call_module ON call_records(module)")
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(call_records)")}
        if "turn_index" not in existing:
            self._conn.execute("ALTER TABLE call_records ADD COLUMN turn_index INTEGER NOT NULL DEFAULT 0")
        self._conn.commit()

    def set_turn_index(self, index: int) -> None:
        """Set the current exchange index stamped onto new call records."""
        self._turn_index = index

    @property
    def current_turn_index(self) -> int:
        """Current exchange index (0-based) for stamping call records."""
        return self._turn_index

    def record(self, record: CallRecord) -> None:
        """Persist a single call record."""
        values = tuple(getattr(record, col) for col in _CALL_COLUMNS)
        placeholders = ", ".join("?" for _ in _CALL_COLUMNS)
        col_names = ", ".join(_CALL_COLUMNS)
        with self._lock:
            self._conn.execute(
                f"INSERT INTO call_records ({col_names}) VALUES ({placeholders})",
                values,
            )
            self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()


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

    def __init__(self, call_store: SQLiteCallStore) -> None:
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
            timestamp=utc_now_str(),
            module=module,
            operation=operation,
            model="",
            elapsed_ms=0.0,
            status="retry",
            metadata=f'{{"endpoint": "{endpoint}", "retry_delay_sec": {delay_sec}}}',
            turn_index=self._store.current_turn_index,
        )
        try:
            self._store.record(call_record)
        except Exception:
            pass


class TrackedTTS(ITTS):
    """ITTS wrapper that records per-call execution data to an SQLiteCallStore.

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

    def __init__(self, inner: ITTS, call_store: SQLiteCallStore) -> None:
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


class TrackedEmbedder(IEmbedder):
    """IEmbedder wrapper that records per-call execution data to an SQLiteCallStore.

    Transparent drop-in: all IEmbedder methods delegate to the inner
    embedder, with timing recorded around ``embed`` and ``embed_batch``.

    Args:
        inner: The underlying embedder implementation.
        call_store: Store to persist call records.
    """

    def __init__(self, inner: IEmbedder, call_store: SQLiteCallStore) -> None:
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
