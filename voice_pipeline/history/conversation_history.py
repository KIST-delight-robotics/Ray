"""Session-scoped conversation history store."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from voice_pipeline.core.interfaces import IConversationHistory, IStorageBackend
from voice_pipeline.history.exceptions import HistoryError

logger = logging.getLogger("voice_pipeline.history")


class ConversationHistory(IConversationHistory):
    """Manages conversation messages for a single session.

    Pure data repository. Message dict schema follows the
    ``{"role": ..., "content": ...}`` convention; vendor-specific
    details are determined by the LLM implementation.
    """

    def __init__(self, backend: IStorageBackend) -> None:
        self._backend = backend
        self._session_id: str | None = None
        self._messages: list[dict[str, Any]] = []
        self._next_id: int = 0
        self._started_at: str = ""

    def _require_session(self) -> str:
        if self._session_id is None:
            raise HistoryError("No active session. Call new_session() first.")
        return self._session_id

    def new_session(self, session_id: str) -> None:
        """Start a new session, clearing any in-memory state."""
        self._session_id = session_id
        self._messages = []
        self._next_id = 0
        self._started_at = datetime.now(UTC).isoformat()
        logger.info("Started new session: %s", session_id)

    def _allocate_id(self) -> int:
        msg_id = self._next_id
        self._next_id += 1
        return msg_id

    def add_user_message(self, text: str) -> int:
        """Append a user message to the current session."""
        self._require_session()
        msg_id = self._allocate_id()
        self._messages.append({"role": "user", "content": text, "_id": msg_id})
        return msg_id

    def add_assistant_message(self, text: str) -> int:
        """Append an assistant message to the current session."""
        self._require_session()
        msg_id = self._allocate_id()
        self._messages.append({"role": "assistant", "content": text, "_id": msg_id})
        return msg_id

    def update_message(self, message_id: int, text: str) -> None:
        """Update the content of an existing message by ID."""
        self._require_session()
        for msg in self._messages:
            if msg.get("_id") == message_id:
                msg["content"] = text
                return
        raise HistoryError(f"No message with ID {message_id}")

    def get_messages(self) -> list[dict[str, Any]]:
        """Retrieve all conversation messages (internal IDs stripped)."""
        self._require_session()
        return [{k: v for k, v in msg.items() if k != "_id"} for msg in self._messages]

    def clear(self) -> None:
        """Remove all messages from the current session in memory."""
        self._require_session()
        self._messages.clear()

    def save(self) -> None:
        """Persist the current session to the storage backend."""
        session_id = self._require_session()
        clean = [{k: v for k, v in msg.items() if k != "_id"} for msg in self._messages]
        metadata = {"started_at": self._started_at}
        self._backend.save(session_id, clean, metadata)
