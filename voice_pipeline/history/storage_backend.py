"""Storage backend implementations for conversation history."""

from __future__ import annotations

import copy
import logging
from typing import Any

from voice_pipeline.core.interfaces import IStorageBackend

logger = logging.getLogger("voice_pipeline.history")


class MemoryStorageBackend(IStorageBackend):
    """In-memory storage backend. Data is lost when the process exits."""

    def __init__(self) -> None:
        self._store: dict[str, list[dict[str, Any]]] = {}

    def load(self, session_id: str) -> list[dict[str, Any]]:
        """Load messages for a session, returning empty list if not found."""
        data = self._store.get(session_id)
        if data is None:
            return []
        return copy.deepcopy(data)

    def save(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Persist messages for a session."""
        self._store[session_id] = copy.deepcopy(messages)
        logger.debug("Saved %d messages for session %s", len(messages), session_id)

    def delete(self, session_id: str) -> None:
        """Delete stored messages for a session. No-op if not found."""
        self._store.pop(session_id, None)
