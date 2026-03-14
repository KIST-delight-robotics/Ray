"""Integration tests for MaAIVAPWrapper with real MaAI model.

Tests both full ONNX and PyTorch transformer fallback modes through the
IVAP interface, verifying that outputs are numerically equivalent.

Requires:
  - ``maai`` package installed (editable from external/MaAI)

Run with:
  uv run pytest -m requires_model voice_pipeline/tests/turn_taking/test_maai_vap_integration.py
"""

from __future__ import annotations

import math
import struct
from typing import TYPE_CHECKING

import pytest

from voice_pipeline.core.config import AudioConfig, MaAIVAPConfig, TTSConfig
from voice_pipeline.core.types import VAPResult

if TYPE_CHECKING:
    from voice_pipeline.turn_taking.maai_vap import MaAIVAPWrapper

pytestmark = pytest.mark.requires_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 16000
_FRAME_SAMPLES = 480  # 30ms @ 16kHz


def _pcm_silence(n_samples: int = _FRAME_SAMPLES) -> bytes:
    return struct.pack(f"<{n_samples}h", *([0] * n_samples))


def _pcm_tone(n_samples: int = _FRAME_SAMPLES, amplitude: int = 10000) -> bytes:
    """Sine wave PCM (simulated speech-like energy)."""
    samples = [
        int(amplitude * math.sin(2 * math.pi * 440 * i / _SAMPLE_RATE)) for i in range(n_samples)
    ]
    return struct.pack(f"<{n_samples}h", *samples)


def _pcm_robot(n_samples: int = 720, amplitude: int = 5000) -> bytes:
    """Robot audio at 24kHz (720 samples = 30ms)."""
    return struct.pack(f"<{n_samples}h", *([amplitude] * n_samples))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_wrapper(transformer_onnx_path: str = "") -> MaAIVAPWrapper:
    from voice_pipeline.turn_taking.maai_vap import MaAIVAPWrapper

    cfg = MaAIVAPConfig(
        frame_rate=10,
        context_len_sec=5.0,
        ort_threads=1,
        pt_threads=1,
        transformer_onnx_path=transformer_onnx_path,
        use_torch_compile=False,
    )
    audio_cfg = AudioConfig(sample_rate=_SAMPLE_RATE, channels=1, frame_duration_ms=30)
    tts_cfg = TTSConfig(output_sample_rate=24000)
    return MaAIVAPWrapper(cfg, audio_cfg, tts_cfg)


@pytest.fixture(scope="module")
def onnx_wrapper():
    """MaAIVAPWrapper with ONNX encoder + ONNX transformer."""
    return _make_wrapper(transformer_onnx_path=MaAIVAPConfig.transformer_onnx_path)


@pytest.fixture(scope="module")
def pytorch_wrapper():
    """MaAIVAPWrapper with ONNX encoder + PyTorch transformer."""
    return _make_wrapper(transformer_onnx_path="")


@pytest.fixture(autouse=True)
def _reset(onnx_wrapper, pytorch_wrapper):
    onnx_wrapper.reset()
    pytorch_wrapper.reset()


# ---------------------------------------------------------------------------
# Tests: Basic operation
# ---------------------------------------------------------------------------


class TestBasicOperation:
    """Verify wrapper loads and produces valid VAPResult."""

    def test_silence_returns_result(self, onnx_wrapper):
        result = onnx_wrapper.feed_audio(_pcm_silence())
        assert isinstance(result, VAPResult)
        assert 0.0 <= result.p_now <= 1.0
        assert 0.0 <= result.p_fut <= 1.0
        assert isinstance(result.user_is_speaking, bool)

    def test_inference_triggers_after_step(self, onnx_wrapper):
        frame = _pcm_tone()
        results = [onnx_wrapper.feed_audio(frame) for _ in range(20)]
        has_inference = any(r.p_now != 0.0 or r.p_fut != 0.0 for r in results)
        assert has_inference, "Expected at least one non-zero inference result"

    def test_values_in_valid_range(self, onnx_wrapper):
        frame = _pcm_tone()
        for _ in range(20):
            result = onnx_wrapper.feed_audio(frame)
        assert 0.0 <= result.p_now <= 1.0
        assert 0.0 <= result.p_fut <= 1.0


# ---------------------------------------------------------------------------
# Tests: Stereo (robot audio)
# ---------------------------------------------------------------------------


class TestStereoInput:
    def test_robot_audio_accepted(self, onnx_wrapper):
        for _ in range(20):
            result = onnx_wrapper.feed_audio(_pcm_tone(), _pcm_robot())
        assert isinstance(result, VAPResult)
        assert 0.0 <= result.p_now <= 1.0

    def test_stereo_produces_valid_results(self, onnx_wrapper):
        for _ in range(20):
            result = onnx_wrapper.feed_audio(_pcm_tone(), _pcm_robot())
        assert 0.0 <= result.p_now <= 1.0
        assert 0.0 <= result.p_fut <= 1.0


# ---------------------------------------------------------------------------
# Tests: Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_returns_default(self, onnx_wrapper):
        for _ in range(20):
            onnx_wrapper.feed_audio(_pcm_tone())
        onnx_wrapper.reset()
        result = onnx_wrapper.feed_audio(_pcm_silence())
        assert result == VAPResult(0.0, 0.0, False)

    def test_reset_allows_fresh_inference(self, onnx_wrapper):
        for _ in range(20):
            onnx_wrapper.feed_audio(_pcm_tone())
        onnx_wrapper.reset()
        for _ in range(20):
            result = onnx_wrapper.feed_audio(_pcm_tone())
        assert result.p_now != 0.0 or result.p_fut != 0.0


# ---------------------------------------------------------------------------
# Tests: Turn cycle (orchestrator usage pattern)
# ---------------------------------------------------------------------------


class TestTurnCycle:
    def test_two_turns_no_state_bleed(self, onnx_wrapper):
        frame = _pcm_tone()

        for _ in range(20):
            onnx_wrapper.feed_audio(frame)
        turn1 = onnx_wrapper.feed_audio(frame)

        onnx_wrapper.reset()

        for _ in range(20):
            onnx_wrapper.feed_audio(frame)
        turn2 = onnx_wrapper.feed_audio(frame)

        assert turn1.p_now != 0.0 or turn1.p_fut != 0.0
        assert turn2.p_now != 0.0 or turn2.p_fut != 0.0

    def test_silence_then_speech(self, onnx_wrapper):
        for _ in range(20):
            onnx_wrapper.feed_audio(_pcm_silence())
        silence_result = onnx_wrapper.feed_audio(_pcm_silence())

        onnx_wrapper.reset()

        for _ in range(20):
            onnx_wrapper.feed_audio(_pcm_tone())
        speech_result = onnx_wrapper.feed_audio(_pcm_tone())

        assert isinstance(silence_result, VAPResult)
        assert isinstance(speech_result, VAPResult)


# ---------------------------------------------------------------------------
# Tests: ONNX vs PyTorch numerical equivalence
# ---------------------------------------------------------------------------


class TestOnnxPytorchEquivalence:
    """Both modes must produce identical outputs given the same input."""

    def test_silence_equivalence(self, onnx_wrapper, pytorch_wrapper):
        frame = _pcm_silence()
        r_onnx = onnx_wrapper.feed_audio(frame)
        r_pt = pytorch_wrapper.feed_audio(frame)
        assert r_onnx == r_pt

    def test_speech_equivalence(self, onnx_wrapper, pytorch_wrapper):
        frame = _pcm_tone()
        for _ in range(20):
            r_onnx = onnx_wrapper.feed_audio(frame)
            r_pt = pytorch_wrapper.feed_audio(frame)

        assert abs(r_onnx.p_now - r_pt.p_now) < 1e-4, (
            f"p_now mismatch: onnx={r_onnx.p_now}, pt={r_pt.p_now}"
        )
        assert abs(r_onnx.p_fut - r_pt.p_fut) < 1e-4, (
            f"p_fut mismatch: onnx={r_onnx.p_fut}, pt={r_pt.p_fut}"
        )
        assert r_onnx.user_is_speaking == r_pt.user_is_speaking

    def test_stereo_equivalence(self, onnx_wrapper, pytorch_wrapper):
        user = _pcm_tone()
        robot = _pcm_robot()
        for _ in range(20):
            r_onnx = onnx_wrapper.feed_audio(user, robot)
            r_pt = pytorch_wrapper.feed_audio(user, robot)

        assert abs(r_onnx.p_now - r_pt.p_now) < 1e-4
        assert abs(r_onnx.p_fut - r_pt.p_fut) < 1e-4

    def test_multi_turn_equivalence(self, onnx_wrapper, pytorch_wrapper):
        """Both modes produce same results across reset boundaries."""
        frame = _pcm_tone()

        for _ in range(20):
            onnx_wrapper.feed_audio(frame)
            pytorch_wrapper.feed_audio(frame)

        onnx_wrapper.reset()
        pytorch_wrapper.reset()

        for _ in range(20):
            r_onnx = onnx_wrapper.feed_audio(frame)
            r_pt = pytorch_wrapper.feed_audio(frame)

        assert abs(r_onnx.p_now - r_pt.p_now) < 1e-4
        assert abs(r_onnx.p_fut - r_pt.p_fut) < 1e-4


# ---------------------------------------------------------------------------
# Tests: PyTorch mode basic
# ---------------------------------------------------------------------------


class TestPyTorchMode:
    """Verify PyTorch transformer mode also works correctly."""

    def test_inference_produces_result(self, pytorch_wrapper):
        frame = _pcm_tone()
        for _ in range(20):
            result = pytorch_wrapper.feed_audio(frame)
        assert 0.0 <= result.p_now <= 1.0
        assert 0.0 <= result.p_fut <= 1.0

    def test_reset_clears_state(self, pytorch_wrapper):
        for _ in range(20):
            pytorch_wrapper.feed_audio(_pcm_tone())
        pytorch_wrapper.reset()
        result = pytorch_wrapper.feed_audio(_pcm_silence())
        assert result == VAPResult(0.0, 0.0, False)
