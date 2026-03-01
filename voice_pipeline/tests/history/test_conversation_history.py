"""Tests for voice_pipeline.history.conversation_history."""

import pytest

from voice_pipeline.history.conversation_history import ConversationHistory
from voice_pipeline.history.exceptions import HistoryError
from voice_pipeline.history.storage_backend import MemoryStorageBackend


class TestConversationHistory:
    def _make_history(self) -> ConversationHistory:
        return ConversationHistory(backend=MemoryStorageBackend())

    def test_no_session_raises(self) -> None:
        h = self._make_history()
        with pytest.raises(HistoryError):
            h.get_messages()

    def test_add_user_without_session_raises(self) -> None:
        h = self._make_history()
        with pytest.raises(HistoryError):
            h.add_user_message("hello")

    def test_add_assistant_without_session_raises(self) -> None:
        h = self._make_history()
        with pytest.raises(HistoryError):
            h.add_assistant_message("hi")

    def test_clear_without_session_raises(self) -> None:
        h = self._make_history()
        with pytest.raises(HistoryError):
            h.clear()

    def test_save_without_session_raises(self) -> None:
        h = self._make_history()
        with pytest.raises(HistoryError):
            h.save()

    def test_new_session_and_get_messages(self) -> None:
        h = self._make_history()
        h.new_session("s1")
        assert h.get_messages() == []

    def test_add_user_message(self) -> None:
        h = self._make_history()
        h.new_session("s1")
        h.add_user_message("hello")
        assert h.get_messages() == [{"role": "user", "content": "hello"}]

    def test_add_assistant_message(self) -> None:
        h = self._make_history()
        h.new_session("s1")
        h.add_assistant_message("hi there")
        assert h.get_messages() == [{"role": "assistant", "content": "hi there"}]

    def test_conversation_flow(self) -> None:
        h = self._make_history()
        h.new_session("s1")
        h.add_user_message("hello")
        h.add_assistant_message("hi")
        h.add_user_message("how are you")
        h.add_assistant_message("good")
        messages = h.get_messages()
        assert len(messages) == 4
        assert messages[0] == {"role": "user", "content": "hello"}
        assert messages[3] == {"role": "assistant", "content": "good"}

    def test_get_messages_returns_copy_list(self) -> None:
        h = self._make_history()
        h.new_session("s1")
        h.add_user_message("hello")
        msgs = h.get_messages()
        msgs.append({"role": "user", "content": "injected"})
        assert len(h.get_messages()) == 1

    def test_get_messages_returns_deep_copy(self) -> None:
        """Mutating dict fields in the returned list must not affect internal state."""
        h = self._make_history()
        h.new_session("s1")
        h.add_user_message("hello")
        msgs = h.get_messages()
        msgs[0]["content"] = "tampered"
        assert h.get_messages()[0]["content"] == "hello"

    def test_clear(self) -> None:
        h = self._make_history()
        h.new_session("s1")
        h.add_user_message("hello")
        h.clear()
        assert h.get_messages() == []

    def test_save_and_restore(self) -> None:
        backend = MemoryStorageBackend()
        h = ConversationHistory(backend=backend)
        h.new_session("s1")
        h.add_user_message("hello")
        h.add_assistant_message("hi")
        h.save()
        loaded = backend.load("s1")
        assert loaded == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

    def test_new_session_clears_previous(self) -> None:
        h = self._make_history()
        h.new_session("s1")
        h.add_user_message("hello")
        h.new_session("s2")
        assert h.get_messages() == []

    def test_save_persists_before_new_session(self) -> None:
        """Saved data survives new_session() in the backend."""
        backend = MemoryStorageBackend()
        h = ConversationHistory(backend=backend)
        h.new_session("s1")
        h.add_user_message("hello")
        h.save()
        h.new_session("s2")
        assert backend.load("s1") == [{"role": "user", "content": "hello"}]
