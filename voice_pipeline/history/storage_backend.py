"""Storage backend implementations for conversation history."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

from voice_pipeline.core.config import ConversationHistoryConfig
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

    def save(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist messages for a session."""
        self._store[session_id] = copy.deepcopy(messages)
        logger.debug("Saved %d messages for session %s", len(messages), session_id)

    def delete(self, session_id: str) -> None:
        """Delete stored messages for a session. No-op if not found."""
        self._store.pop(session_id, None)


class FileStorageBackend(IStorageBackend):
    """JSON file storage backend. Each session is saved as a separate file."""

    def __init__(self, directory: str) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        logger.info("File storage: %s", self._dir)

    def _path(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.json"

    def load(self, session_id: str) -> list[dict[str, Any]]:
        """Load messages from JSON file."""
        path = self._path(session_id)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load session file: %s", path)
            return []
        # Handle wrapped format (dict with "messages" key) and legacy (bare list)
        if isinstance(data, dict):
            return data.get("messages", [])
        return data

    def save(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save messages with metadata to JSON file."""
        path = self._path(session_id)
        doc: dict[str, Any] = {"session_id": session_id}
        if metadata:
            doc.update(metadata)
        doc["messages"] = messages
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved %d messages → %s", len(messages), path)

    def delete(self, session_id: str) -> None:
        """Delete session file."""
        path = self._path(session_id)
        if path.exists():
            path.unlink()


def create_storage_backend(config: ConversationHistoryConfig) -> IStorageBackend:
    """Factory: create a storage backend from config."""
    if config.storage_backend == "memory":
        return MemoryStorageBackend()
    elif config.storage_backend == "file":
        if not config.storage_path:
            raise ValueError("storage_path is required for file backend")
        return FileStorageBackend(config.storage_path)
    else:
        raise ValueError(f"Unknown storage backend: {config.storage_backend!r}")
