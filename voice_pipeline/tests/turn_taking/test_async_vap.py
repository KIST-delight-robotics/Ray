"""Unit tests for AsyncVAP wrapper."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from voice_pipeline.core.interfaces import IVAP
from voice_pipeline.core.types import VAPResult
from voice_pipeline.turn_taking.async_vap import AsyncVAP

FRAME = b"\x00" * 960  # 30ms at 16kHz, 16-bit


def _make_async_vap(
    vap_result: VAPResult | None = None,
    frame_rate: int = 100,
) -> tuple[AsyncVAP, MagicMock]:
    """Create AsyncVAP with mocked underlying VAP.

    Uses high frame_rate (100Hz) for fast test cycles.
    """
    mock_vap = MagicMock(spec=IVAP)
    result = vap_result or VAPResult(0.6, 0.4, True)
    mock_vap.feed_audio.return_value = result
    async_vap = AsyncVAP(mock_vap, frame_rate=frame_rate)
    return async_vap, mock_vap


class TestFeedAudioNonBlocking:
    def test_returns_immediately(self):
        """feed_audio should return without waiting for inference."""
        async_vap, _ = _make_async_vap()
        try:
            start = time.monotonic()
            async_vap.feed_audio(FRAME)
            elapsed = time.monotonic() - start
            # Should be near-instant (< 1ms typically)
            assert elapsed < 0.01
        finally:
            async_vap.stop()

    def test_returns_cached_result(self):
        """feed_audio returns the initial default result before any inference."""
        async_vap, _ = _make_async_vap()
        try:
            # Before background thread runs, should return the default
            result = async_vap.feed_audio(FRAME)
            assert isinstance(result, VAPResult)
        finally:
            async_vap.stop()


class TestResultUpdate:
    def test_background_updates_result(self):
        """After feeding audio, background thread updates the cached result."""
        expected = VAPResult(0.7, 0.3, True)
        async_vap, _ = _make_async_vap(vap_result=expected)
        try:
            async_vap.feed_audio(FRAME)
            # Wait for background thread to process
            time.sleep(0.1)
            result = async_vap.feed_audio(FRAME)
            assert result == expected
        finally:
            async_vap.stop()


class TestBufferDrain:
    def test_multiple_frames_combined(self):
        """Multiple buffered frames are concatenated into one inference call."""
        async_vap, mock_vap = _make_async_vap()
        try:
            frame_a = b"\x01" * 960
            frame_b = b"\x02" * 960
            async_vap.feed_audio(frame_a)
            async_vap.feed_audio(frame_b)
            time.sleep(0.1)  # Let background thread process

            # Should have been called with concatenated audio
            assert mock_vap.feed_audio.called
            args = mock_vap.feed_audio.call_args
            user_audio = args[0][0]
            assert user_audio == frame_a + frame_b
        finally:
            async_vap.stop()

    def test_robot_audio_pairs(self):
        """Robot audio is concatenated alongside user audio."""
        async_vap, mock_vap = _make_async_vap()
        try:
            user_a = b"\x01" * 960
            robot_a = b"\x10" * 100
            user_b = b"\x02" * 960
            robot_b = b"\x20" * 100
            async_vap.feed_audio(user_a, robot_a)
            async_vap.feed_audio(user_b, robot_b)
            time.sleep(0.1)

            args = mock_vap.feed_audio.call_args
            assert args[0][0] == user_a + user_b
            assert args[0][1] == robot_a + robot_b
        finally:
            async_vap.stop()

    def test_mixed_none_robot_audio(self):
        """When some frames have robot_audio=None, only non-None are combined."""
        async_vap, mock_vap = _make_async_vap()
        try:
            user_a = b"\x01" * 960
            user_b = b"\x02" * 960
            robot_b = b"\x20" * 100
            async_vap.feed_audio(user_a, None)
            async_vap.feed_audio(user_b, robot_b)
            time.sleep(0.1)

            args = mock_vap.feed_audio.call_args
            assert args[0][0] == user_a + user_b
            assert args[0][1] == robot_b
        finally:
            async_vap.stop()

    def test_all_none_robot_audio(self):
        """When all frames have robot_audio=None, None is passed to VAP."""
        async_vap, mock_vap = _make_async_vap()
        try:
            async_vap.feed_audio(FRAME, None)
            async_vap.feed_audio(FRAME, None)
            time.sleep(0.1)

            args = mock_vap.feed_audio.call_args
            assert args[0][1] is None
        finally:
            async_vap.stop()


class TestReset:
    def test_clears_buffer_and_result(self):
        """reset() clears the buffer and resets the cached result."""
        expected = VAPResult(0.7, 0.3, True)
        async_vap, mock_vap = _make_async_vap(vap_result=expected)
        try:
            async_vap.feed_audio(FRAME)
            time.sleep(0.1)
            # Result should be updated
            assert async_vap.feed_audio(FRAME) == expected

            async_vap.reset()
            # After reset, result returns to default
            result = async_vap._latest_result
            assert result == VAPResult(0.0, 0.0, False)
            # Underlying VAP reset was called
            mock_vap.reset.assert_called_once()
        finally:
            async_vap.stop()


class TestStop:
    def test_thread_joins(self):
        """stop() causes the background thread to exit."""
        async_vap, _ = _make_async_vap()
        async_vap.stop()
        assert not async_vap._thread.is_alive()

    def test_idempotent_stop(self):
        """Calling stop() twice should not raise."""
        async_vap, _ = _make_async_vap()
        async_vap.stop()
        async_vap.stop()  # Should not raise
