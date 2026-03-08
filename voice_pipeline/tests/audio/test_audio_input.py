"""Tests for voice_pipeline.audio.audio_input."""

from __future__ import annotations

import queue
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

# Import directly to avoid __init__.py triggering torch import
from voice_pipeline.audio.audio_input import AudioInput
from voice_pipeline.audio.exceptions import AudioInputError
from voice_pipeline.core.config import AudioConfig, AudioInputConfig
from voice_pipeline.core.types import AudioFrame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audio_input(
    *,
    queue_size: int = 10,
    device_index: int | None = None,
):
    """Create AudioInput with mocked PyAudio."""
    audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=queue_size)
    audio_config = AudioConfig()
    config = AudioInputConfig(device_index=device_index)

    # Provide a mock pyaudio module
    mock_pyaudio = MagicMock()
    with patch.object(AudioInput, "__init__", lambda self, *a, **kw: None):
        ai = AudioInput.__new__(AudioInput)

    ai._queue = audio_queue
    ai._audio_config = audio_config
    ai._config = config
    ai._stop_event = threading.Event()
    ai._thread = None
    ai._error = None
    ai._pyaudio_module = mock_pyaudio

    return ai, audio_queue


class TestAudioInputImport:
    def test_import_error_raises(self) -> None:
        """Missing pyaudio raises AudioInputError at construction."""
        # Temporarily hide pyaudio
        original = sys.modules.get("pyaudio")
        sys.modules["pyaudio"] = None  # type: ignore[assignment]
        try:
            import importlib

            import voice_pipeline.audio.audio_input as mod

            importlib.reload(mod)

            with pytest.raises(AudioInputError, match="PyAudio is not installed"):
                mod.AudioInput(
                    queue.Queue(),
                    AudioConfig(),
                    AudioInputConfig(),
                )
        finally:
            if original is not None:
                sys.modules["pyaudio"] = original
            else:
                sys.modules.pop("pyaudio", None)
            # Reload to restore normal state
            import importlib

            import voice_pipeline.audio.audio_input as mod

            importlib.reload(mod)


class TestStartStop:
    def test_start_stop_lifecycle(self) -> None:
        """start() creates thread, stop() joins it."""
        ai, _ = _make_audio_input()

        mock_pa_instance = MagicMock()
        mock_stream = MagicMock()
        # Block reads until stop is set
        mock_stream.read.side_effect = _block_until_stop(ai)
        mock_pa_instance.open.return_value = mock_stream
        ai._pyaudio_module.PyAudio.return_value = mock_pa_instance

        ai.start()
        assert ai._thread is not None
        time.sleep(0.05)  # Let thread start
        assert ai._thread.is_alive()

        ai.stop()
        assert ai._thread is None

    def test_start_idempotent(self) -> None:
        """Calling start() twice doesn't create a second thread."""
        ai, _ = _make_audio_input()

        mock_pa_instance = MagicMock()
        mock_stream = MagicMock()
        mock_stream.read.side_effect = _block_until_stop(ai)
        mock_pa_instance.open.return_value = mock_stream
        ai._pyaudio_module.PyAudio.return_value = mock_pa_instance

        ai.start()
        time.sleep(0.05)
        first_thread = ai._thread
        ai.start()
        assert ai._thread is first_thread

        ai.stop()

    def test_stop_idempotent(self) -> None:
        """Calling stop() without start() doesn't raise."""
        ai, _ = _make_audio_input()
        ai.stop()  # Should not raise


class TestCapture:
    def test_frames_pushed_to_queue(self) -> None:
        """Captured frames are pushed to the queue."""
        ai, audio_queue = _make_audio_input()

        mock_pa_instance = MagicMock()
        mock_stream = MagicMock()
        frames = [b"\x01" * 960, b"\x02" * 960, b"\x03" * 960]
        mock_stream.read.side_effect = _frames_then_stop(ai, frames)
        mock_pa_instance.open.return_value = mock_stream
        ai._pyaudio_module.PyAudio.return_value = mock_pa_instance

        ai.start()
        ai._thread.join(timeout=2.0)

        collected = []
        while not audio_queue.empty():
            collected.append(audio_queue.get_nowait())

        assert collected == frames

    def test_queue_full_drops_frame(self) -> None:
        """When queue is full, frames are dropped (not blocked)."""
        ai, audio_queue = _make_audio_input(queue_size=1)

        mock_pa_instance = MagicMock()
        mock_stream = MagicMock()
        # Send 5 frames, queue can hold 1
        frames = [b"\x00" * 960] * 5
        mock_stream.read.side_effect = _frames_then_stop(ai, frames)
        mock_pa_instance.open.return_value = mock_stream
        ai._pyaudio_module.PyAudio.return_value = mock_pa_instance

        ai.start()
        ai._thread.join(timeout=2.0)

        # Thread should have finished without blocking
        assert not ai._thread.is_alive() or ai._stop_event.is_set()

    def test_device_open_failure(self) -> None:
        """Device open failure sets _error and thread exits."""
        ai, _ = _make_audio_input()

        mock_pa_instance = MagicMock()
        mock_pa_instance.open.side_effect = OSError("No such device")
        ai._pyaudio_module.PyAudio.return_value = mock_pa_instance

        ai.start()
        ai._thread.join(timeout=2.0)

        assert ai._error is not None
        assert "No such device" in str(ai._error)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _block_until_stop(ai):
    """Return a side_effect that blocks until stop is set."""

    def _read(*args, **kwargs):
        while not ai._stop_event.is_set():
            time.sleep(0.01)
        return b"\x00" * 960

    return _read


def _frames_then_stop(ai, frames: list[bytes]):
    """Return a side_effect that yields given frames then stops."""
    idx = 0

    def _read(*args, **kwargs):
        nonlocal idx
        if idx < len(frames):
            data = frames[idx]
            idx += 1
            if idx >= len(frames):
                ai._stop_event.set()
            return data
        ai._stop_event.set()
        return b""

    return _read
