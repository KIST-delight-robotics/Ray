"""Module interfaces for the voice pipeline.

Only interfaces needed by the next implementation phase are defined here.
New interfaces are added just before their consuming phase begins.

Current: Phase 2 interfaces (history, utterance_truncator, context_builder).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from voice_pipeline.core.types import WordTimestamp

# ---------------------------------------------------------------------------
# StorageBackend
# ---------------------------------------------------------------------------


class IStorageBackend(ABC):
    """Persistence backend for conversation history.

    Implementations: memory, file, database.
    """

    @abstractmethod
    def load(self, session_id: str) -> list[dict[str, Any]]:
        """Load messages for a session.

        Args:
            session_id: Unique session identifier.

        Returns:
            List of message dicts, or empty list if session not found.
        """

    @abstractmethod
    def save(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Persist messages for a session.

        Args:
            session_id: Unique session identifier.
            messages: List of message dicts to persist.
        """

    @abstractmethod
    def delete(self, session_id: str) -> None:
        """Delete stored messages for a session.

        Args:
            session_id: Unique session identifier.
        """


# ---------------------------------------------------------------------------
# ConversationHistory
# ---------------------------------------------------------------------------


class IConversationHistory(ABC):
    """Session-scoped conversation history store.

    Pure data repository. Message dict schema is vendor-specific
    and determined by LLM implementation.
    """

    @abstractmethod
    def new_session(self, session_id: str) -> None:
        """Start a new session, clearing any in-memory state.

        Args:
            session_id: Unique identifier for this conversation session.
        """

    @abstractmethod
    def add_user_message(self, text: str) -> None:
        """Append a user message to the current session.

        Args:
            text: The user's transcribed utterance (final ASR result).
        """

    @abstractmethod
    def add_assistant_message(self, text: str) -> None:
        """Append an assistant message to the current session.

        Called with full response text on normal playback completion,
        or with truncated text on barge-in interruption.

        Args:
            text: The robot's spoken text (full or truncated).
        """

    @abstractmethod
    def get_messages(self, max_turns: int | None = None) -> list[dict[str, Any]]:
        """Retrieve conversation messages.

        Args:
            max_turns: If given, return only the most recent N turns
                (one turn = one user + one assistant message).
                None returns all messages.

        Returns:
            List of message dicts in vendor-specific format.
        """

    @abstractmethod
    def clear(self) -> None:
        """Remove all messages from the current session in memory."""

    @abstractmethod
    def save(self) -> None:
        """Persist the current session to the storage backend."""


# ---------------------------------------------------------------------------
# UtteranceTruncator
# ---------------------------------------------------------------------------


class IUtteranceTruncator(ABC):
    """Truncates spoken text to match a barge-in stop position.

    Strategy interface with two implementations:
    - TimestampTruncator: uses word-level timestamps for precision.
    - DurationRatioTruncator: estimates from total audio duration ratio.
    """

    @abstractmethod
    def truncate(
        self,
        text: str,
        stop_position_sec: float,
        timestamps: list[WordTimestamp],
    ) -> str:
        """Return the portion of text spoken before the stop point.

        Args:
            text: Full response text that was being played.
            stop_position_sec: Playback position in seconds when the
                robot was interrupted.
            timestamps: Word-level timestamps from TTS. May be empty
                if the TTS implementation does not support them.

        Returns:
            Truncated text representing what was actually spoken.
        """


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------


class IContextBuilder(ABC):
    """Assembles LLM context from conversation history and current input.

    IConversationHistory is injected via the constructor — it does not
    change per call. The build() method only takes the current turn's text.
    """

    @abstractmethod
    def build(self, current_text: str) -> list[dict[str, Any]]:
        """Build the message list for an LLM call.

        Args:
            current_text: Current ASR transcription (the user's in-progress
                turn, not yet committed to history).

        Returns:
            List of message dicts suitable for passing to ILLM.generate().
        """
