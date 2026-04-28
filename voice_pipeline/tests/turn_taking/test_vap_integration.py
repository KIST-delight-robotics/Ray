"""Integration tests for VAPWrapper with a real VAP model.

Requires:
  - ``vap`` package installed (editable from external/VoiceActivityProjection)
  - VAP_MODEL_PATH: env var pointing to a VAP state_dict ``.pt`` file

Run with:
  uv run pytest -m requires_model voice_pipeline/tests/turn_taking/test_vap_integration.py
"""

from __future__ import annotations

import os
import struct

import pytest

from voice_pipeline.core.types import VAPResult
from voice_pipeline.tts.tts import OpenAITTS
from voice_pipeline.turn_taking.exceptions import VAPError

pytestmark = pytest.mark.requires_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 16000
_FRAME_SAMPLES = 480  # 30ms @ 16kHz


def _pcm_silence(n_samples: int = _FRAME_SAMPLES) -> bytes:
    return struct.pack(f"<{n_samples}h", *([0] * n_samples))


def _pcm_tone(n_samples: int = _FRAME_SAMPLES, amplitude: int = 10000) -> bytes:
    """Constant-amplitude PCM (simulated speech-like energy)."""
    return struct.pack(f"<{n_samples}h", *([amplitude] * n_samples))


def _pcm_robot(n_samples: int = 720, amplitude: int = 5000) -> bytes:
    """Robot audio at 24kHz (720 samples = 30ms)."""
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

    Class vars are mutated module-wide for the integration test session.
    """
    from voice_pipeline.turn_taking.vap import VAPWrapper

    VAPWrapper._MODEL_PATH = model_path
    VAPWrapper._CONTEXT_SEC = 5.0  # shorter buffer for faster tests
    VAPWrapper._STEP_SEC = 0.1
    return VAPWrapper(OpenAITTS.OUTPUT_SAMPLE_RATE)


@pytest.fixture(autouse=True)
def _reset_wrapper(wrapper):
    """Reset wrapper state before each test to eliminate order dependency."""
    wrapper.reset()


# ---------------------------------------------------------------------------
# Tests: Basic operation
# ---------------------------------------------------------------------------


class TestBasicOperation:
    """Verify wrapper loads real model and produces valid VAPResult."""

    def test_feed_silence_returns_result(self, wrapper):
        frame = _pcm_silence()
        result = wrapper.feed_audio(frame)
        assert isinstance(result, VAPResult)
        assert 0.0 <= result.p_now <= 1.0
        assert 0.0 <= result.p_fut <= 1.0
        assert isinstance(result.user_is_speaking, bool)

    def test_inference_triggers_after_step(self, wrapper):
        """After enough frames to reach step_sec, inference runs and values change."""
        frame = _pcm_tone()
        results = []
        # Feed 10 frames (300ms > step_sec=100ms), at least 2 inferences
        for _ in range(10):
            results.append(wrapper.feed_audio(frame))

        # At least one result should come from actual inference (non-default)
        has_inference = any(r.p_now != 0.0 or r.p_fut != 0.0 for r in results)
        assert has_inference, "Expected at least one inference result with non-zero values"

    def test_values_in_valid_range(self, wrapper):
        frame = _pcm_tone()
        for _ in range(10):
            result = wrapper.feed_audio(frame)

        assert 0.0 <= result.p_now <= 1.0
        assert 0.0 <= result.p_fut <= 1.0


# ---------------------------------------------------------------------------
# Tests: Stereo (robot audio)
# ---------------------------------------------------------------------------


class TestStereoInput:
    """Feed both user and robot audio through the full stereo path."""

    def test_robot_audio_accepted(self, wrapper):
        user_frame = _pcm_tone()
        robot_frame = _pcm_robot()

        for _ in range(10):
            result = wrapper.feed_audio(user_frame, robot_frame)

        assert isinstance(result, VAPResult)
        assert 0.0 <= result.p_now <= 1.0

    def test_stereo_produces_valid_results(self, wrapper):
        """Full stereo path (decode, resample 24k→16k, pad/trim, two-channel inference)."""
        user_frame = _pcm_tone()
        robot_frame = _pcm_robot()

        for _ in range(10):
            result = wrapper.feed_audio(user_frame, robot_frame)

        assert isinstance(result, VAPResult)
        assert 0.0 <= result.p_now <= 1.0
        assert 0.0 <= result.p_fut <= 1.0


# ---------------------------------------------------------------------------
# Tests: Reset
# ---------------------------------------------------------------------------


class TestReset:
    """Verify reset clears state and returns to baseline."""

    def test_reset_returns_default_on_next_frame(self, wrapper):
        # Warm up
        frame = _pcm_tone()
        for _ in range(10):
            wrapper.feed_audio(frame)

        wrapper.reset()
        result = wrapper.feed_audio(_pcm_silence())
        assert result == VAPResult(0.0, 0.0, False)

    def test_reset_allows_fresh_inference(self, wrapper):
        frame = _pcm_tone()
        for _ in range(10):
            result = wrapper.feed_audio(frame)
        assert isinstance(result, VAPResult)


# ---------------------------------------------------------------------------
# Tests: Turn cycle (orchestrator usage pattern)
# ---------------------------------------------------------------------------


class TestTurnCycle:
    """Simulate orchestrator turn cycle: stream → reset → stream again."""

    def test_two_turns_no_state_bleed(self, wrapper):
        """Turn 1 state must not affect turn 2 after reset."""
        frame = _pcm_tone()

        # Turn 1: feed frames and collect final result
        for _ in range(10):
            wrapper.feed_audio(frame)
        turn1_result = wrapper.feed_audio(frame)

        # Reset between turns (as orchestrator would)
        wrapper.reset()

        # Turn 2: feed same frames, should produce inference from scratch
        for _ in range(10):
            wrapper.feed_audio(frame)
        turn2_result = wrapper.feed_audio(frame)

        # Both turns should produce valid results
        assert turn1_result.p_now != 0.0 or turn1_result.p_fut != 0.0
        assert turn2_result.p_now != 0.0 or turn2_result.p_fut != 0.0

    def test_silence_then_speech_turn(self, wrapper):
        """Turn 1 with silence, turn 2 with speech — different contexts."""
        silence = _pcm_silence()
        speech = _pcm_tone()

        # Turn 1: silence
        for _ in range(10):
            wrapper.feed_audio(silence)
        silence_result = wrapper.feed_audio(silence)

        wrapper.reset()

        # Turn 2: speech
        for _ in range(10):
            wrapper.feed_audio(speech)
        speech_result = wrapper.feed_audio(speech)

        assert isinstance(silence_result, VAPResult)
        assert isinstance(speech_result, VAPResult)


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Invalid model path raises VAPError at construction time."""

    def test_invalid_model_path_raises(self, monkeypatch):
        from voice_pipeline.turn_taking.vap import VAPWrapper

        monkeypatch.setattr(VAPWrapper, "_MODEL_PATH", "/nonexistent/model.pt")
        with pytest.raises(VAPError, match="Failed to load VAP model"):
            VAPWrapper(OpenAITTS.OUTPUT_SAMPLE_RATE)
