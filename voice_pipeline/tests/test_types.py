"""Tests for voice_pipeline.types."""

import pytest

from voice_pipeline.types import TTSStream


class TestTTSStream:
    @staticmethod
    def _make_gen(chunks: list[bytes]):
        """Helper: create a generator yielding byte chunks."""
        yield from chunks

    def test_iteration_yields_chunks(self) -> None:
        chunks = [b"\x01\x02", b"\x03\x04"]
        stream = TTSStream(self._make_gen(chunks))
        collected = list(stream)
        assert collected == chunks

    def test_audio_after_iteration(self) -> None:
        chunks = [b"\x01\x02", b"\x03\x04"]
        stream = TTSStream(self._make_gen(chunks))
        list(stream)
        assert stream.audio == b"\x01\x02\x03\x04"

    def test_audio_before_iteration_raises(self) -> None:
        stream = TTSStream(self._make_gen([b"\x01"]))
        with pytest.raises(RuntimeError, match="not available"):
            _ = stream.audio

    def test_close_before_iteration(self) -> None:
        close_called = False

        def on_close() -> None:
            nonlocal close_called
            close_called = True

        stream = TTSStream(self._make_gen([b"\x01"]), close_fn=on_close)
        stream.close()
        assert close_called

    def test_close_idempotent(self) -> None:
        call_count = 0

        def on_close() -> None:
            nonlocal call_count
            call_count += 1

        stream = TTSStream(self._make_gen([b"\x01"]), close_fn=on_close)
        stream.close()
        stream.close()
        assert call_count == 1

    def test_close_stops_iteration(self) -> None:
        stream = TTSStream(self._make_gen([b"\x01", b"\x02", b"\x03"]))
        next(stream)
        stream.close()
        # After close, next() should raise StopIteration
        with pytest.raises(StopIteration):
            next(stream)

    def test_done_not_set_on_close(self) -> None:
        """close() should not set _done; audio should remain unavailable."""
        stream = TTSStream(self._make_gen([b"\x01", b"\x02"]))
        next(stream)
        stream.close()
        with pytest.raises(RuntimeError, match="not available"):
            _ = stream.audio

    def test_close_fn_error_suppressed(self) -> None:
        def bad_close() -> None:
            raise RuntimeError("close error")

        stream = TTSStream(self._make_gen([b"\x01"]), close_fn=bad_close)
        # Should not raise
        stream.close()

    def test_empty_stream(self) -> None:
        stream = TTSStream(self._make_gen([]))
        collected = list(stream)
        assert collected == []
        assert stream.audio == b""
