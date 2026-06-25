"""Unit tests for MaAIVAPWrapper's async machinery (buffer/drain/thread).

The model-bearing parts (ONNX load, inference) are covered by
``test_maai_vap_integration.py`` (``requires_model``). Here we exercise only
the command/query threading layer: ``feed_audio`` buffers, a daemon thread
drains and combines frames into one ``_infer`` call, and ``latest_result``
reflects the result — all without loading a real model.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

from voice_pipeline.core.types import VAPResult
from voice_pipeline.turn_taking.maai_vap import MaAIVAPWrapper

FRAME = b"\x00" * 960  # 30ms at 16kHz, 16-bit


def _make_threaded_vap(
    infer_result: VAPResult | None = None,
    frame_rate: int = 100,
) -> tuple[MaAIVAPWrapper, MagicMock]:
    """Build a MaAIVAPWrapper with only its async machinery initialized.

    Bypasses ``__init__`` (which loads ONNX) via ``__new__`` and stubs
    ``_infer`` so the drain thread runs without a real model. Uses a high
    frame_rate (100Hz) for fast test cycles.
    """
    vap = MaAIVAPWrapper.__new__(MaAIVAPWrapper)
    vap._frame_rate = frame_rate
    vap._interval = 1.0 / frame_rate
    vap._buffer = []
    vap._buffer_lock = threading.Lock()
    vap._latest_result = VAPResult(0.0, 0.0, False)
    vap._call_store = None
    vap.session_id = ""
    vap._call_records = []
    vap._stop_event = threading.Event()

    result = infer_result or VAPResult(0.6, 0.4, True)

    def _fake_infer(user_audio, robot_audio=None):
        vap._latest_result = result
        return result

    infer_mock = MagicMock(side_effect=_fake_infer)
    vap._infer = infer_mock

    vap._thread = threading.Thread(target=vap._run, daemon=True, name="maai-vap-test")
    vap._thread.start()
    return vap, infer_mock


class TestFeedAudioCommand:
    def test_returns_none_and_non_blocking(self):
        """feed_audio is a command: returns None without waiting for inference."""
        vap, _ = _make_threaded_vap()
        try:
            start = time.monotonic()
            result = vap.feed_audio(FRAME)
            elapsed = time.monotonic() - start
            assert result is None
            assert elapsed < 0.01
        finally:
            vap.stop()


class TestLatestResult:
    def test_default_before_inference(self):
        """latest_result is the default until the thread runs inference."""
        vap, _ = _make_threaded_vap()
        try:
            assert vap.latest_result == VAPResult(0.0, 0.0, False)
        finally:
            vap.stop()

    def test_updated_by_background_thread(self):
        """After feeding audio, the background thread updates latest_result."""
        expected = VAPResult(0.7, 0.3, True)
        vap, _ = _make_threaded_vap(infer_result=expected)
        try:
            vap.feed_audio(FRAME)
            time.sleep(0.1)
            assert vap.latest_result == expected
        finally:
            vap.stop()


class TestBufferDrain:
    def test_multiple_frames_combined(self):
        """Multiple buffered frames are concatenated into one inference call."""
        vap, infer = _make_threaded_vap()
        try:
            frame_a = b"\x01" * 960
            frame_b = b"\x02" * 960
            vap.feed_audio(frame_a)
            vap.feed_audio(frame_b)
            time.sleep(0.1)

            assert infer.called
            user_audio = infer.call_args[0][0]
            assert user_audio == frame_a + frame_b
        finally:
            vap.stop()

    def test_robot_audio_pairs(self):
        """Robot audio is concatenated alongside user audio."""
        vap, infer = _make_threaded_vap()
        try:
            user_a, robot_a = b"\x01" * 960, b"\x10" * 100
            user_b, robot_b = b"\x02" * 960, b"\x20" * 100
            vap.feed_audio(user_a, robot_a)
            vap.feed_audio(user_b, robot_b)
            time.sleep(0.1)

            args = infer.call_args[0]
            assert args[0] == user_a + user_b
            assert args[1] == robot_a + robot_b
        finally:
            vap.stop()

    def test_mixed_none_robot_audio(self):
        """When some frames have robot_audio=None, only non-None are combined."""
        vap, infer = _make_threaded_vap()
        try:
            user_a = b"\x01" * 960
            user_b, robot_b = b"\x02" * 960, b"\x20" * 100
            vap.feed_audio(user_a, None)
            vap.feed_audio(user_b, robot_b)
            time.sleep(0.1)

            args = infer.call_args[0]
            assert args[0] == user_a + user_b
            assert args[1] == robot_b
        finally:
            vap.stop()

    def test_all_none_robot_audio(self):
        """When all frames have robot_audio=None, None is passed to inference."""
        vap, infer = _make_threaded_vap()
        try:
            vap.feed_audio(FRAME, None)
            vap.feed_audio(FRAME, None)
            time.sleep(0.1)

            assert infer.call_args[0][1] is None
        finally:
            vap.stop()


class TestStop:
    def test_thread_joins(self):
        """stop() causes the background thread to exit."""
        vap, _ = _make_threaded_vap()
        vap.stop()
        assert not vap._thread.is_alive()

    def test_idempotent_stop(self):
        """Calling stop() twice should not raise."""
        vap, _ = _make_threaded_vap()
        vap.stop()
        vap.stop()
