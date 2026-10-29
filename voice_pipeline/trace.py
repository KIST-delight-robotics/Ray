"""실행 기록 (관측용). 없어도 대화는 돈다.

``CallRecord`` (외부 호출 1건 = 1행) → ``record_call()`` → ``SQLiteCallStore``.

컨텍스트(session_id)는 이 모듈이 들고 있다 — wiring이 ``set_session()``으로 갱신하고, 레코드를
만들 때 자동으로 찍힌다. ``TrackedEmbedder``는 IEmbedder 를 감싸 호출을 기록하는 래퍼,
``OpenAIRetryHandler``는 openai SDK 재시도 로그를 CallRecord로 바꾸는 logging 핸들러.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from voice_pipeline.types import IEmbedder, utc_now_str

logger = logging.getLogger("voice_pipeline.trace")


@dataclass
class CallRecord:
    """Single module call record — latency, status, and optional metadata."""

    session_id: str
    timestamp: str
    module: str
    operation: str
    model: str
    elapsed_ms: float
    status: str = "ok"
    metadata: str | None = None


_CALL_COLUMNS = (
    "session_id",
    "timestamp",
    "module",
    "operation",
    "model",
    "elapsed_ms",
    "status",
    "metadata",
)


class SQLiteCallStore:
    """Persists per-call execution records to a SQLite database.

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
            CREATE TABLE IF NOT EXISTS call_records (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                module      TEXT NOT NULL,
                operation   TEXT NOT NULL,
                model       TEXT NOT NULL,
                elapsed_ms  REAL NOT NULL,
                status      TEXT NOT NULL DEFAULT 'ok',
                metadata    TEXT
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_call_session ON call_records(session_id)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_call_module ON call_records(module)")
        self._conn.commit()

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


# ---------------------------------------------------------------------------
# 기록 API — logging처럼 모듈 전역. 프로세스당 세션 하나를 전제한다.
#
# wiring이 install()로 싱크를 꽂고 세션마다 set_session()을 부른다. emitter는 record_call()
# 한 줄이면 되고, 싱크가 없으면 no-op.
# ---------------------------------------------------------------------------


class _CallSink(Protocol):
    def record(self, record: CallRecord) -> None: ...


_call_sink: _CallSink | None = None
_session_id = ""


def install(call_store: _CallSink | None = None) -> None:
    """싱크 설치 (프로세스 시작 시 1회). 이전 싱크는 닫지 않는다."""
    global _call_sink
    _call_sink = call_store


def reset() -> None:
    """싱크와 컨텍스트를 초기 상태로 (테스트 격리용)."""
    global _call_sink, _session_id
    _call_sink = None
    _session_id = ""


def close() -> None:
    """싱크를 닫고 초기화 (프로세스 종료 시)."""
    close_fn = getattr(_call_sink, "close", None)
    if close_fn is not None:
        try:
            close_fn()
        except Exception:
            logger.warning("Failed to close trace sink", exc_info=True)
    reset()


def set_session(session_id: str) -> None:
    global _session_id
    _session_id = session_id


def record_call(
    module: str,
    operation: str,
    model: str,
    elapsed_ms: float,
    *,
    status: str = "ok",
    metadata: str | None = None,
) -> None:
    """외부 호출 1건을 현재 컨텍스트로 기록한다. 싱크가 없으면 no-op."""
    sink = _call_sink
    if sink is None:
        return
    record = CallRecord(
        session_id=_session_id,
        timestamp=utc_now_str(),
        module=module,
        operation=operation,
        model=model,
        elapsed_ms=elapsed_ms,
        status=status,
        metadata=metadata,
    )
    try:
        sink.record(record)
    except Exception:
        logger.warning("Failed to record call (%s.%s)", module, operation, exc_info=True)


# openai SDK 1.x: "Retrying request to /responses in 0.4 seconds"
# openai SDK 3.x: "Retrying request in 0.4 seconds"  (엔드포인트가 로그에서 빠짐 → 모듈 귀속 불가)
_PATTERN = re.compile(r"Retrying request(?: to (/\S+))? in ([\d.]+) seconds")

_ENDPOINT_MAP: dict[str, tuple[str, str]] = {
    "/audio/speech": ("tts", "synthesize"),
    "/responses": ("llm", "generate"),
    "/embeddings": ("embedder", "embed"),
}


class OpenAIRetryHandler(logging.Handler):
    """Intercepts OpenAI SDK retry log messages and records them.

    Attach to ``logging.getLogger("openai._base_client")`` to capture
    retry events across all OpenAI API calls (TTS, LLM, embeddings).
    """

    def emit(self, record: logging.LogRecord) -> None:
        m = _PATTERN.search(record.getMessage())
        if not m:
            return

        endpoint = m.group(1) or ""
        delay_sec = m.group(2)
        module, operation = _ENDPOINT_MAP.get(endpoint, ("unknown", endpoint or "retry"))

        record_call(
            module,
            operation,
            "",
            0.0,
            status="retry",
            metadata=f'{{"endpoint": "{endpoint}", "retry_delay_sec": {delay_sec}}}',
        )


class TrackedEmbedder(IEmbedder):
    """IEmbedder wrapper that records per-call execution data via ``record_call``.

    Transparent drop-in: all IEmbedder methods delegate to the inner
    embedder, with timing recorded around ``embed`` and ``embed_batch``.

    Args:
        inner: The underlying embedder implementation.
    """

    def __init__(self, inner: IEmbedder) -> None:
        self._inner = inner

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
        record_call("embedder", operation, self._inner.model_name, elapsed_ms, status=status, metadata=metadata)
