"""Stress tests for VAPWrapper with a real VAP model.

Requires:
  - ``vap`` package installed (editable from external/VoiceActivityProjection)
  - VAP_MODEL_PATH: env var pointing to a VAP state_dict ``.pt`` file

Run with:
  uv run pytest -m requires_model voice_pipeline/tests/turn_taking/test_vap_stress.py
"""

from __future__ import annotations

import os
import struct
import time

import pytest

from voice_pipeline.core.types import VAPResult
from voice_pipeline.tts.openai_tts import OpenAITTS

pytestmark = pytest.mark.requires_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 16000
_FRAME_SAMPLES = 480  # 30ms @ 16kHz


def _pcm_tone(n_samples: int = _FRAME_SAMPLES, amplitude: int = 10000) -> bytes:
    return struct.pack(f"<{n_samples}h", *([amplitude] * n_samples))


def _pcm_robot(n_samples: int = 720, amplitude: int = 5000) -> bytes:
    return struct.pack(f"<{n_samples}h", *([amplitude] * n_samples))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_path() -> str:
    path = os.environ.get("VAP_MODEL_PATH", "")
    if not path:
        pytest.skip("VAP_MODEL_PATH not set")
    if not os.path.isfile(path):
        pytest.skip(f"VAP model file not found: {path}")
    return path


@pytest.fixture(scope="module")
def wrapper(model_path: str):
    """Create a VAPWrapper with real model (shared across module tests).

    Class vars are mutated module-wide for the stress test session.
    """
    from voice_pipeline.turn_taking.vap import VAPWrapper

    VAPWrapper._MODEL_PATH = model_path
    VAPWrapper._CONTEXT_SEC = 5.0
    VAPWrapper._STEP_SEC = 0.1
    return VAPWrapper(OpenAITTS.OUTPUT_SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPerformance:
    """Wall-clock performance sanity checks (machine-dependent)."""

    def test_single_inference_under_200ms(self, wrapper):
        """One inference cycle should complete well under 200ms on CPU."""
        wrapper.reset()
        frame = _pcm_tone()

        # Warm up (first inference may be slower)
        for _ in range(10):
            wrapper.feed_audio(frame)

        wrapper.reset()
        # 4 frames (1920 samples >= step_sec 1600) triggers 1 inference
        start = time.perf_counter()
        for _ in range(4):
            wrapper.feed_audio(frame)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.2, f"4 frames took {elapsed:.3f}s, expected < 200ms"

    def test_sustained_streaming(self, wrapper):
        """10 seconds of simulated audio (~333 frames) should process faster than real-time."""
        wrapper.reset()
        frame = _pcm_tone()
        n_frames = 333  # ~10s at 30ms/frame

        start = time.perf_counter()
        for _ in range(n_frames):
            wrapper.feed_audio(frame)
        elapsed = time.perf_counter() - start

        assert elapsed < 10.0, f"333 frames took {elapsed:.3f}s, must be faster than real-time"


class TestRapidResetCycles:
    """Rapid turn churn: many short turns with feed bursts + reset."""

    def test_100_reset_cycles(self, wrapper):
        """100 short turns (5 frames each + reset). No exceptions, valid results."""
        frame = _pcm_tone()
        for _cycle in range(100):
            for _ in range(5):
                result = wrapper.feed_audio(frame)
            assert isinstance(result, VAPResult)
            wrapper.reset()

    def test_rapid_reset_with_stereo(self, wrapper):
        """50 reset cycles alternating mono and stereo input."""
        user_frame = _pcm_tone()
        robot_frame = _pcm_robot()

        for cycle in range(50):
            if cycle % 2 == 0:
                for _ in range(5):
                    result = wrapper.feed_audio(user_frame)
            else:
                for _ in range(5):
                    result = wrapper.feed_audio(user_frame, robot_frame)
            assert isinstance(result, VAPResult)
            wrapper.reset()
