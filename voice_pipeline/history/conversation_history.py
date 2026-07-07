"""Session-scoped conversation history backed by a storage backend."""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from typing import Any

from voice_pipeline.core.interfaces import IConversationHistory, IStorageBackend
from voice_pipeline.core.types import HistoryTurn, LLMMetrics, TokenCounter
from voice_pipeline.history.exceptions import HistoryError
from voice_pipeline.history.storage_backend import TIMESTAMP_FORMAT

logger = logging.getLogger("voice_pipeline.history")


class ConversationHistory(IConversationHistory):
    """Entry-based conversation history backed by a storage backend.

    The storage backend is the single source of truth for all reads
    and writes. Every mutation is persisted immediately for crash safety.

    Thread-safe via threading.Lock: writes from Orchestrator (main thread),
    reads from SpeechGenerator background thread (via ContextBuilder).
    """

    def __init__(self, backend: IStorageBackend, token_counter: TokenCounter) -> None:
        self._backend = backend
        self._token_counter = token_counter
        self._lock = threading.Lock()
        self._session_id: str | None = None
        self._next_msg_id: int = 0
        self._next_turn_id: int = 0

    def _require_session(self) -> str:
        if self._session_id is None:
            raise HistoryError("No active session. Call new_session() first.")
        return self._session_id

    def _allocate_msg_id(self) -> int:
        msg_id = self._next_msg_id
        self._next_msg_id += 1
        return msg_id

    def _allocate_turn_id(self) -> int:
        turn_id = self._next_turn_id
        self._next_turn_id += 1
        return turn_id

    def _compute_token_count(self, text: str, metrics: LLMMetrics | None) -> int:
        """Use LLM output_tokens if available, fallback to token_counter."""
        if metrics is not None:
            return metrics.usage.output_tokens
        return self._safe_count(text)

    def _safe_count(self, text: str) -> int:
        """Count tokens with graceful fallback on error."""
        try:
            return self._token_counter(text)
        except Exception:
            logger.warning("Token counter failed, using 0", exc_info=True)
            return 0

    @staticmethod
    def _serialize_metrics(metrics: LLMMetrics | None) -> str | None:
        """Serialize LLMMetrics to JSON string for DB storage."""
        if metrics is None:
            return None
        return json.dumps(
            {
                "usage": {
                    "input_tokens": metrics.usage.input_tokens,
                    "output_tokens": metrics.usage.output_tokens,
                    "cached_tokens": metrics.usage.cached_tokens,
                    "reasoning_tokens": metrics.usage.reasoning_tokens,
                },
                "model": metrics.model,
                "latency_ms": metrics.latency_ms,
                "ttft_ms": metrics.ttft_ms,
            }
        )

    def _write_message(
        self,
        session_id: str,
        msg_id: int,
        turn_id: int,
        item: dict[str, Any],
        token_count: int,
        metrics: LLMMetrics | None = None,
    ) -> None:
        """Persist message to backend. Logs warning on failure."""
        try:
            self._backend.append_message(
                session_id,
                msg_id,
                turn_id,
                item,
                token_count,
                self._serialize_metrics(metrics),
            )
        except Exception:
            logger.warning(
                "Failed to persist message %d — message will be missing from history",
                msg_id,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def new_session(self, session_id: str) -> None:
        """Start a new session, clearing any previous state."""
        with self._lock:
            self._session_id = session_id
            self._next_msg_id = 0
            self._next_turn_id = 0
            started_at = datetime.now(UTC).strftime(TIMESTAMP_FORMAT)
            try:
                self._backend.create_session(session_id, started_at)
            except Exception:
                logger.warning("Failed to create session in backend", exc_info=True)
            logger.debug("Started new session: %s", session_id)

    def add_user_message(self, text: str) -> int:
        """Append a user message. Auto-assigns turn_id."""
        with self._lock:
            session_id = self._require_session()
            msg_id = self._allocate_msg_id()
            turn_id = self._allocate_turn_id()
            token_count = self._safe_count(text)
            item: dict[str, Any] = {"role": "user", "content": text}
            self._write_message(session_id, msg_id, turn_id, item, token_count)
            return msg_id

    def add_assistant_message(self, text: str, metrics: LLMMetrics | None = None) -> int:
        """Append an assistant text message. Auto-assigns turn_id."""
        with self._lock:
            session_id = self._require_session()
            msg_id = self._allocate_msg_id()
            turn_id = self._allocate_turn_id()
            token_count = self._compute_token_count(text, metrics)
            item: dict[str, Any] = {"role": "assistant", "content": text}
            self._write_message(session_id, msg_id, turn_id, item, token_count, metrics)
            return msg_id

    def add_message(
        self,
        item: dict[str, Any],
        turn_id: int | None = None,
        metrics: LLMMetrics | None = None,
    ) -> tuple[int, int]:
        """Append a message in Responses API input format."""
        with self._lock:
            session_id = self._require_session()
            msg_id = self._allocate_msg_id()
            if turn_id is None:
                turn_id = self._allocate_turn_id()

            if metrics is not None:
                token_count = metrics.usage.output_tokens
            else:
                text = item.get("content") or item.get("output") or item.get("arguments") or ""
                token_count = self._safe_count(text) if text else 0

            self._write_message(session_id, msg_id, turn_id, item, token_count, metrics)
            return msg_id, turn_id

    def begin_turn(self) -> int:
        """Allocate a new turn_id for grouping multiple messages."""
        with self._lock:
            self._require_session()
            return self._allocate_turn_id()

    def update_message(self, msg_id: int, text: str) -> None:
        """Update message text. Recomputes token_count internally."""
        with self._lock:
            session_id = self._require_session()
            result = self._backend.load_message(session_id, msg_id)
            if result is None:
                raise HistoryError(f"No message with ID {msg_id}")
            _, _, item, _ = result
            item["content"] = text
            token_count = self._safe_count(text)
            try:
                self._backend.update_message(session_id, msg_id, item, token_count)
            except Exception:
                logger.warning(
                    "Failed to update message %d in backend",
                    msg_id,
                    exc_info=True,
                )

    def get_messages(self) -> list[dict[str, Any]]:
        """Retrieve all messages as a flat list for LLM input."""
        with self._lock:
            session_id = self._require_session()
            rows = self._backend.load_session(session_id)
            return [row[2] for row in rows]

    def get_turns(self) -> list[HistoryTurn]:
        """Retrieve messages grouped by turn for context budgeting."""
        with self._lock:
            session_id = self._require_session()
            rows = self._backend.load_session(session_id)

            groups: dict[int, list[tuple[dict[str, Any], int]]] = {}
            first_msg_id: dict[int, int] = {}
            for msg_id, turn_id, item, token_count in rows:
                if turn_id not in groups:
                    groups[turn_id] = []
                    first_msg_id[turn_id] = msg_id
                groups[turn_id].append((item, token_count))

            sorted_turn_ids = sorted(groups.keys(), key=lambda tid: first_msg_id[tid])
            return [
                HistoryTurn(
                    items=tuple(item for item, _ in groups[tid]),
                    token_count=sum(tc for _, tc in groups[tid]),
                    turn_id=tid,
                )
                for tid in sorted_turn_ids
            ]

    def save(self) -> None:
        """Finalize the current session (sets ended_at)."""
        with self._lock:
            session_id = self._require_session()
            ended_at = datetime.now(UTC).strftime(TIMESTAMP_FORMAT)
            try:
                self._backend.end_session(session_id, ended_at)
            except Exception:
                logger.warning("Failed to end session in backend", exc_info=True)
