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
        msg_id = h.add_user_message("hello")
        assert isinstance(msg_id, int)
        assert h.get_messages() == [{"role": "user", "content": "hello"}]

    def test_add_assistant_message(self) -> None:
        h = self._make_history()
        h.new_session("s1")
        msg_id = h.add_assistant_message("hi there")
        assert isinstance(msg_id, int)
        assert h.get_messages() == [{"role": "assistant", "content": "hi there"}]

    def test_add_returns_sequential_ids(self) -> None:
        h = self._make_history()
        h.new_session("s1")
        id0 = h.add_user_message("hello")
        id1 = h.add_assistant_message("hi")
        id2 = h.add_user_message("bye")
        assert id0 == 0
        assert id1 == 1
        assert id2 == 2

    def test_update_message(self) -> None:
        h = self._make_history()
        h.new_session("s1")
        msg_id = h.add_assistant_message("full text here")
        h.update_message(msg_id, "truncated")
        assert h.get_messages() == [{"role": "assistant", "content": "truncated"}]

    def test_update_message_invalid_id_raises(self) -> None:
        h = self._make_history()
        h.new_session("s1")
        h.add_user_message("hello")
        with pytest.raises(HistoryError):
            h.update_message(999, "nope")

    def test_get_messages_strips_id(self) -> None:
        h = self._make_history()
        h.new_session("s1")
        h.add_user_message("hello")
        msgs = h.get_messages()
        assert "_id" not in msgs[0]

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

    def test_get_messages_returns_independent_dicts(self) -> None:
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

    def test_save_strips_internal_ids(self) -> None:
        backend = MemoryStorageBackend()
        h = ConversationHistory(backend=backend)
        h.new_session("s1")
        h.add_user_message("hello")
        h.save()
        loaded = backend.load("s1")
        assert "_id" not in loaded[0]

    def test_new_session_resets_ids(self) -> None:
        h = self._make_history()
        h.new_session("s1")
        id0 = h.add_user_message("hello")
        h.new_session("s2")
        id1 = h.add_user_message("world")
        assert id0 == 0
        assert id1 == 0

    def test_save_persists_before_new_session(self) -> None:
        """Saved data survives new_session() in the backend."""
        backend = MemoryStorageBackend()
        h = ConversationHistory(backend=backend)
        h.new_session("s1")
        h.add_user_message("hello")
        h.save()
        h.new_session("s2")
        assert backend.load("s1") == [{"role": "user", "content": "hello"}]
