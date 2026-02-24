"""Tests for voice_pipeline.core.interfaces."""

import pytest

from voice_pipeline.core.interfaces import (
    IContextBuilder,
    IConversationHistory,
    IStorageBackend,
    IUtteranceTruncator,
)


class TestInterfacesAreAbstract:
    """Verify that all interfaces cannot be instantiated directly."""

    def test_storage_backend_abstract(self) -> None:
        with pytest.raises(TypeError):
            IStorageBackend()  # type: ignore[abstract]

    def test_conversation_history_abstract(self) -> None:
        with pytest.raises(TypeError):
            IConversationHistory()  # type: ignore[abstract]

    def test_utterance_truncator_abstract(self) -> None:
        with pytest.raises(TypeError):
            IUtteranceTruncator()  # type: ignore[abstract]

    def test_context_builder_abstract(self) -> None:
        with pytest.raises(TypeError):
            IContextBuilder()  # type: ignore[abstract]
