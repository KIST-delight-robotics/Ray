"""Tests for voice_pipeline.audio.audio_input."""

from __future__ import annotations

import array
import queue
import struct
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

# Import directly to avoid __init__.py triggering torch import
from voice_pipeline.audio.audio_input import AudioInput
from voice_pipeline.audio.constants import FRAME_SIZE_SAMPLES
from voice_pipeline.audio.exceptions import AudioInputError
from voice_pipeline.core.types import AudioFrame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audio_input(
    monkeypatch,
    *,
    queue_size: int = 10,
    device_index: int | None = None,
    device_name: str | None = None,
    capture_channels: int | None = None,
    extract_channel: int = 0,
):
    """Create AudioInput with mocked PyAudio."""
    monkeypatch.setattr(AudioInput, "_DEVICE_INDEX", device_index)
    monkeypatch.setattr(AudioInput, "_DEVICE_NAME", device_name)
    monkeypatch.setattr(AudioInput, "_CAPTURE_CHANNELS", capture_channels)
    monkeypatch.setattr(AudioInput, "_EXTRACT_CHANNEL", extract_channel)
    audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=queue_size)

    # Provide a mock pyaudio module
    mock_pyaudio = MagicMock()
    with patch.object(AudioInput, "__init__", lambda self, *a, **kw: None):
        ai = AudioInput.__new__(AudioInput)

    ai._queue = audio_queue
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
                mod.AudioInput(queue.Queue())
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
    def test_start_stop_lifecycle(self, monkeypatch) -> None:
        """start() creates thread, stop() joins it."""
        ai, _ = _make_audio_input(monkeypatch)

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

    def test_start_idempotent(self, monkeypatch) -> None:
        """Calling start() twice doesn't create a second thread."""
        ai, _ = _make_audio_input(monkeypatch)

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

    def test_stop_idempotent(self, monkeypatch) -> None:
        """Calling stop() without start() doesn't raise."""
        ai, _ = _make_audio_input(monkeypatch)
        ai.stop()  # Should not raise


class TestCapture:
    def test_frames_pushed_to_queue(self, monkeypatch) -> None:
        """Captured frames are pushed to the queue."""
        ai, audio_queue = _make_audio_input(monkeypatch)

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

    def test_queue_full_drops_frame(self, monkeypatch) -> None:
        """When queue is full, frames are dropped (not blocked)."""
        ai, audio_queue = _make_audio_input(monkeypatch, queue_size=1)

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

    def test_device_open_failure(self, monkeypatch) -> None:
        """Device open failure sets _error and thread exits."""
        ai, _ = _make_audio_input(monkeypatch)

        mock_pa_instance = MagicMock()
        mock_pa_instance.open.side_effect = OSError("No such device")
        ai._pyaudio_module.PyAudio.return_value = mock_pa_instance

        ai.start()
        ai._thread.join(timeout=2.0)

        assert ai._error is not None
        assert "No such device" in str(ai._error)


class TestMultiChannelExtract:
    """Tests for multi-channel capture → mono extraction."""

    def test_extracts_first_channel_from_6ch(self, monkeypatch) -> None:
        """6ch frames should be reduced to mono using channel 0."""
        ai, audio_queue = _make_audio_input(monkeypatch, capture_channels=6, extract_channel=0)

        mock_pa_instance = MagicMock()
        mock_stream = MagicMock()

        # Build a 6ch frame: ch0=100, ch1=200, ..., ch5=600 (repeated per sample)
        samples_per_ch = FRAME_SIZE_SAMPLES
        raw = b""
        for _ in range(samples_per_ch):
            for ch in range(6):
                raw += struct.pack("<h", (ch + 1) * 100)

        mock_stream.read.side_effect = _frames_then_stop(ai, [raw])
        mock_pa_instance.open.return_value = mock_stream
        ai._pyaudio_module.PyAudio.return_value = mock_pa_instance

        ai.start()
        ai._thread.join(timeout=2.0)

        frame = audio_queue.get_nowait()
        mono = array.array("h", frame)
        assert len(mono) == samples_per_ch
        assert all(s == 100 for s in mono)

    def test_extracts_nonzero_channel(self, monkeypatch) -> None:
        """extract_channel=3 should pick the 4th channel."""
        ai, audio_queue = _make_audio_input(monkeypatch, capture_channels=6, extract_channel=3)

        mock_pa_instance = MagicMock()
        mock_stream = MagicMock()

        samples_per_ch = FRAME_SIZE_SAMPLES
        raw = b""
        for _ in range(samples_per_ch):
            for ch in range(6):
                raw += struct.pack("<h", (ch + 1) * 100)

        mock_stream.read.side_effect = _frames_then_stop(ai, [raw])
        mock_pa_instance.open.return_value = mock_stream
        ai._pyaudio_module.PyAudio.return_value = mock_pa_instance

        ai.start()
        ai._thread.join(timeout=2.0)

        frame = audio_queue.get_nowait()
        mono = array.array("h", frame)
        assert len(mono) == samples_per_ch
        assert all(s == 400 for s in mono)

    def test_mono_passthrough_no_extraction(self, monkeypatch) -> None:
        """capture_channels=None should pass frames through unchanged."""
        ai, audio_queue = _make_audio_input(monkeypatch, capture_channels=None)

        mock_pa_instance = MagicMock()
        mock_stream = MagicMock()
        mono_frame = b"\x01" * (FRAME_SIZE_SAMPLES * 2)
        mock_stream.read.side_effect = _frames_then_stop(ai, [mono_frame])
        mock_pa_instance.open.return_value = mock_stream
        ai._pyaudio_module.PyAudio.return_value = mock_pa_instance

        ai.start()
        ai._thread.join(timeout=2.0)

        frame = audio_queue.get_nowait()
        assert frame == mono_frame

    def test_extract_channel_out_of_range_raises(self, monkeypatch) -> None:
        """extract_channel >= capture_channels should raise AudioInputError."""
        ai, _ = _make_audio_input(monkeypatch, capture_channels=6, extract_channel=6)

        mock_pa_instance = MagicMock()
        mock_stream = MagicMock()
        mock_pa_instance.open.return_value = mock_stream
        ai._pyaudio_module.PyAudio.return_value = mock_pa_instance

        ai.start()
        ai._thread.join(timeout=2.0)

        assert isinstance(ai._error, AudioInputError)
        assert "extract_channel" in str(ai._error)


class TestDeviceResolution:
    """Tests for name-based input device resolution."""

    @staticmethod
    def _mock_pa(devices: list[dict]) -> MagicMock:
        pa = MagicMock()
        pa.get_device_count.return_value = len(devices)
        pa.get_device_info_by_index.side_effect = lambda i: devices[i]
        return pa

    def test_explicit_index_wins_over_name(self, monkeypatch) -> None:
        """_DEVICE_INDEX가 지정되면 이름 탐색 없이 그대로 사용."""
        ai, _ = _make_audio_input(monkeypatch, device_index=3, device_name="respeaker")
        pa = self._mock_pa([])

        assert ai._resolve_device_index(pa) == 3
        pa.get_device_count.assert_not_called()

    def test_name_match_case_insensitive(self, monkeypatch) -> None:
        """이름 부분 문자열이 대소문자 무시로 매칭된다."""
        ai, _ = _make_audio_input(monkeypatch, device_name="respeaker")
        pa = self._mock_pa(
            [
                {"name": "HDMI", "maxInputChannels": 0},
                {"name": "reSpeaker Flex XVF3800 C16K6Ch: USB Audio (hw:1,0)", "maxInputChannels": 6},
                {"name": "pipewire", "maxInputChannels": 64},
            ]
        )

        assert ai._resolve_device_index(pa) == 1

    def test_output_only_device_not_matched(self, monkeypatch) -> None:
        """이름이 맞아도 입력 채널이 없는 장치는 건너뛴다."""
        ai, _ = _make_audio_input(monkeypatch, device_name="respeaker")
        pa = self._mock_pa(
            [
                {"name": "reSpeaker playback", "maxInputChannels": 0},
                {"name": "pipewire", "maxInputChannels": 64},
            ]
        )

        with pytest.raises(AudioInputError, match="No input device matching"):
            ai._resolve_device_index(pa)

    def test_no_match_error_lists_candidates(self, monkeypatch) -> None:
        """매칭 실패 에러에 사용 가능한 입력 장치 목록이 포함된다."""
        ai, _ = _make_audio_input(monkeypatch, device_name="respeaker")
        pa = self._mock_pa([{"name": "pipewire", "maxInputChannels": 64}])

        with pytest.raises(AudioInputError, match="pipewire"):
            ai._resolve_device_index(pa)

    def test_name_none_uses_system_default(self, monkeypatch) -> None:
        """인덱스와 이름 모두 None이면 시스템 기본 장치(None)."""
        ai, _ = _make_audio_input(monkeypatch)
        pa = self._mock_pa([])

        assert ai._resolve_device_index(pa) is None

    def test_resolution_failure_sets_error(self, monkeypatch) -> None:
        """매칭 실패 시 캡처 스레드가 _error를 남기고 종료한다."""
        ai, _ = _make_audio_input(monkeypatch, device_name="respeaker")
        mock_pa_instance = self._mock_pa([{"name": "pipewire", "maxInputChannels": 64}])
        ai._pyaudio_module.PyAudio.return_value = mock_pa_instance

        ai.start()
        ai._thread.join(timeout=2.0)

        assert isinstance(ai._error, AudioInputError)


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
