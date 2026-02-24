"""Tests for voice_pipeline.history.storage_backend."""

from voice_pipeline.history.storage_backend import MemoryStorageBackend


class TestMemoryStorageBackend:
    def test_load_empty(self) -> None:
        backend = MemoryStorageBackend()
        assert backend.load("nonexistent") == []

    def test_save_and_load(self) -> None:
        backend = MemoryStorageBackend()
        messages = [{"role": "user", "content": "hello"}]
        backend.save("s1", messages)
        loaded = backend.load("s1")
        assert loaded == messages

    def test_load_returns_deep_copy(self) -> None:
        backend = MemoryStorageBackend()
        backend.save("s1", [{"role": "user", "content": "hello"}])
        loaded = backend.load("s1")
        loaded[0]["content"] = "modified"
        assert backend.load("s1")[0]["content"] == "hello"

    def test_save_stores_deep_copy(self) -> None:
        backend = MemoryStorageBackend()
        messages = [{"role": "user", "content": "hello"}]
        backend.save("s1", messages)
        messages[0]["content"] = "modified"
        assert backend.load("s1")[0]["content"] == "hello"

    def test_delete_existing(self) -> None:
        backend = MemoryStorageBackend()
        backend.save("s1", [{"role": "user", "content": "hello"}])
        backend.delete("s1")
        assert backend.load("s1") == []

    def test_delete_nonexistent_is_noop(self) -> None:
        backend = MemoryStorageBackend()
        backend.delete("nonexistent")  # should not raise

    def test_multiple_sessions(self) -> None:
        backend = MemoryStorageBackend()
        backend.save("s1", [{"role": "user", "content": "a"}])
        backend.save("s2", [{"role": "user", "content": "b"}])
        assert backend.load("s1")[0]["content"] == "a"
        assert backend.load("s2")[0]["content"] == "b"

    def test_overwrite_session(self) -> None:
        backend = MemoryStorageBackend()
        backend.save("s1", [{"role": "user", "content": "old"}])
        backend.save("s1", [{"role": "user", "content": "new"}])
        assert backend.load("s1") == [{"role": "user", "content": "new"}]
