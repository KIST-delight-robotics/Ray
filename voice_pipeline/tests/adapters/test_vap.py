"""Unit tests for ThreadedVAP (buffer/drain/thread scheduling).

The VAP model is injected as a lightweight fake, so these exercise only the
threading/scheduling layer — no real ONNX model is loaded. Model inference
itself is covered by test_maai_vap_integration.py (requires_model).
"""

from __future__ import annotations

import threading
import time

from voice_pipeline.adapters.vap import ThreadedVAP, VAPResult
from voice_pipeline.types import AudioFrame

FRAME = b"\x00" * 960  # 30ms at 16kHz, 16-bit


class FakeVAPModel:
    """Records infer() calls and returns a fixed result."""

    def __init__(self, result: VAPResult) -> None:
        self._result = result
        self.calls: list[tuple[AudioFrame, AudioFrame | None]] = []
        self.reset_count = 0
        self._lock = threading.Lock()

    def infer(self, user_audio: AudioFrame, robot_audio: AudioFrame | None = None) -> VAPResult:
        with self._lock:
            self.calls.append((user_audio, robot_audio))
        return self._result

    def reset(self) -> None:
        self.reset_count += 1

    def last_call(self) -> tuple[AudioFrame, AudioFrame | None]:
        with self._lock:
            return self.calls[-1]


def _make_threaded_vap(
    result: VAPResult | None = None,
    frame_rate: int = 100,
) -> tuple[ThreadedVAP, FakeVAPModel]:
    """Create ThreadedVAP wrapping a fake model. 100Hz for fast test cycles."""
    model = FakeVAPModel(result or VAPResult(0.6, 0.4, True))
    vap = ThreadedVAP(model, frame_rate=frame_rate)
    return vap, model


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
        vap, _ = _make_threaded_vap(result=expected)
        try:
            vap.feed_audio(FRAME)
            time.sleep(0.1)
            assert vap.latest_result == expected
        finally:
            vap.stop()


class TestBufferDrain:
    def test_multiple_frames_combined(self):
        """Multiple buffered frames are concatenated into one inference call."""
        vap, model = _make_threaded_vap()
        try:
            frame_a = b"\x01" * 960
            frame_b = b"\x02" * 960
            vap.feed_audio(frame_a)
            vap.feed_audio(frame_b)
            time.sleep(0.1)

            assert model.calls
            user_audio, _ = model.last_call()
            assert user_audio == frame_a + frame_b
        finally:
            vap.stop()

    def test_robot_audio_pairs(self):
        """Robot audio is concatenated alongside user audio."""
        vap, model = _make_threaded_vap()
        try:
            user_a, robot_a = b"\x01" * 960, b"\x10" * 100
            user_b, robot_b = b"\x02" * 960, b"\x20" * 100
            vap.feed_audio(user_a, robot_a)
            vap.feed_audio(user_b, robot_b)
            time.sleep(0.1)

            user_audio, robot_audio = model.last_call()
            assert user_audio == user_a + user_b
            assert robot_audio == robot_a + robot_b
        finally:
            vap.stop()

    def test_mixed_none_robot_audio(self):
        """When some frames have robot_audio=None, only non-None are combined."""
        vap, model = _make_threaded_vap()
        try:
            user_a = b"\x01" * 960
            user_b, robot_b = b"\x02" * 960, b"\x20" * 100
            vap.feed_audio(user_a, None)
            vap.feed_audio(user_b, robot_b)
            time.sleep(0.1)

            user_audio, robot_audio = model.last_call()
            assert user_audio == user_a + user_b
            assert robot_audio == robot_b
        finally:
            vap.stop()

    def test_all_none_robot_audio(self):
        """When all frames have robot_audio=None, None is passed to inference."""
        vap, model = _make_threaded_vap()
        try:
            vap.feed_audio(FRAME, None)
            vap.feed_audio(FRAME, None)
            time.sleep(0.1)

            _, robot_audio = model.last_call()
            assert robot_audio is None
        finally:
            vap.stop()


class TestReset:
    def test_resets_model_and_result(self):
        """reset() clears the cached result and resets the model."""
        vap, model = _make_threaded_vap(result=VAPResult(0.7, 0.3, True))
        try:
            vap.feed_audio(FRAME)
            time.sleep(0.1)
            assert vap.latest_result == VAPResult(0.7, 0.3, True)

            vap.reset()
            assert vap.latest_result == VAPResult(0.0, 0.0, False)
            assert model.reset_count == 1
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
