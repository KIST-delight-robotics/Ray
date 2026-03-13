"""Integration tests for TurnGPTWrapper with a real TurnGPT model.

Requires:
  - ``turngpt`` package installed (editable from external/TurnGPT)
  - TURNGPT_CHECKPOINT_PATH: env var pointing to a TurnGPT checkpoint file

Run with:
  uv run pytest -m requires_model voice_pipeline/tests/turn_taking/test_turngpt_integration.py
"""

from __future__ import annotations

import os

import pytest

from voice_pipeline.core.config import TurnGPTConfig
from voice_pipeline.turn_taking.exceptions import TurnGPTError

pytestmark = pytest.mark.requires_model

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def checkpoint_path() -> str:
    path = os.environ.get("TURNGPT_CHECKPOINT_PATH", "")
    if not path:
        pytest.skip("TURNGPT_CHECKPOINT_PATH not set")
    if not os.path.isfile(path):
        pytest.skip(f"TurnGPT checkpoint not found: {path}")
    return path


@pytest.fixture(scope="module")
def wrapper(checkpoint_path: str):
    """Create a TurnGPTWrapper with real model (shared across module tests)."""
    config = TurnGPTConfig(checkpoint_path=checkpoint_path, onnx_model_path="", device="cpu")

    from voice_pipeline.turn_taking.turngpt import TurnGPTWrapper

    return TurnGPTWrapper(config)


@pytest.fixture(autouse=True)
def _reset_wrapper(wrapper):
    """Reset wrapper state before each test to eliminate order dependency."""
    wrapper.reset()


# ---------------------------------------------------------------------------
# Tests: Basic operation
# ---------------------------------------------------------------------------


class TestBasicOperation:
    """Verify wrapper loads real model and produces valid predictions."""

    def test_single_utterance(self, wrapper):
        result = wrapper.predict("hello how are you")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_multi_turn(self, wrapper):
        result = wrapper.predict("hello<ts>I am fine<ts>how about you")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_result_type_is_float(self, wrapper):
        result = wrapper.predict("test input")
        assert type(result) is float


# ---------------------------------------------------------------------------
# Tests: Input variations
# ---------------------------------------------------------------------------


class TestInputVariations:
    """Various text formats and lengths."""

    def test_short_input(self, wrapper):
        result = wrapper.predict("hi")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_long_dialog(self, wrapper):
        turns = [f"turn number {i}" for i in range(10)]
        dialog = "<ts>".join(turns)
        result = wrapper.predict(dialog)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_partial_current_turn(self, wrapper):
        result = wrapper.predict("hello<ts>I think")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_mixed_case_and_punctuation(self, wrapper):
        result = wrapper.predict("Hello!<ts>How are you?<ts>I'm Fine, Thanks.")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Tests: Reset
# ---------------------------------------------------------------------------


class TestReset:
    """Verify reset allows fresh prediction."""

    def test_reset_allows_fresh_prediction(self, wrapper):
        wrapper.predict("hello<ts>world")
        wrapper.reset()
        result = wrapper.predict("new conversation")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_multiple_resets_safe(self, wrapper):
        for _ in range(5):
            wrapper.reset()
        result = wrapper.predict("still works")
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Tests: Turn cycle
# ---------------------------------------------------------------------------


class TestTurnCycle:
    """Simulate orchestrator usage: predict → reset → predict."""

    def test_two_independent_turns(self, wrapper):
        result1 = wrapper.predict("first turn text")
        wrapper.reset()
        result2 = wrapper.predict("second turn text")

        assert isinstance(result1, float)
        assert isinstance(result2, float)
        assert 0.0 <= result1 <= 1.0
        assert 0.0 <= result2 <= 1.0

    def test_growing_dialog_context(self, wrapper):
        results = []
        dialog = "hello"
        for i in range(5):
            results.append(wrapper.predict(dialog))
            dialog += f"<ts>turn {i}"

        assert all(isinstance(r, float) for r in results)
        assert all(0.0 <= r <= 1.0 for r in results)

    def test_incremental_asr_updates(self, wrapper):
        """ASR partial updates — same prefix, growing suffix."""
        base = "hello<ts>"
        for suffix in ["I", "I think", "I think that", "I think that is great"]:
            result = wrapper.predict(base + suffix)
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Invalid checkpoint path raises TurnGPTError."""

    def test_invalid_checkpoint_path_raises(self):
        config = TurnGPTConfig(
            checkpoint_path="/nonexistent/model.ckpt",
            onnx_model_path="",
            device="cpu",
        )

        from voice_pipeline.turn_taking.turngpt import TurnGPTWrapper

        with pytest.raises(TurnGPTError, match="Failed to load TurnGPT model"):
            TurnGPTWrapper(config)
