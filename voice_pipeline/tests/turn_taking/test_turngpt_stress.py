"""Stress tests for TurnGPTWrapper with a real TurnGPT model.

Requires:
  - ``turngpt`` package installed (editable from external/TurnGPT)
  - TURNGPT_CHECKPOINT_PATH: env var pointing to a TurnGPT checkpoint file

Run with:
  uv run pytest -m requires_model voice_pipeline/tests/turn_taking/test_turngpt_stress.py
"""

from __future__ import annotations

import os
import time

import pytest

from voice_pipeline.core.config import TurnGPTConfig

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
    config = TurnGPTConfig(checkpoint_path=checkpoint_path, device="cpu")

    from voice_pipeline.turn_taking.turngpt import TurnGPTWrapper

    return TurnGPTWrapper(config)


@pytest.fixture(autouse=True)
def _reset_wrapper(wrapper):
    """Reset wrapper state before each test."""
    wrapper.reset()


# ---------------------------------------------------------------------------
# Tests: Performance
# ---------------------------------------------------------------------------


class TestPerformance:
    """Wall-clock performance sanity checks (machine-dependent)."""

    def test_single_prediction_under_500ms(self, wrapper):
        """One prediction should complete under 500ms on CPU."""
        # Warm up
        wrapper.predict("warm up text")

        start = time.perf_counter()
        wrapper.predict("hello<ts>how are you doing today")
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5, f"Single prediction took {elapsed:.3f}s, expected < 500ms"

    def test_50_consecutive_predictions(self, wrapper):
        """50 predictions should complete within a reasonable bound."""
        # Warm up
        wrapper.predict("warm up")

        start = time.perf_counter()
        for i in range(50):
            wrapper.predict(f"turn {i}<ts>response {i}")
        elapsed = time.perf_counter() - start

        assert elapsed < 25.0, f"50 predictions took {elapsed:.3f}s, expected < 25s"


# ---------------------------------------------------------------------------
# Tests: Rapid reset cycles
# ---------------------------------------------------------------------------


class TestRapidResetCycles:
    """Rapid predict+reset churn."""

    def test_100_predict_reset_cycles(self, wrapper):
        """100 predict+reset cycles with no exceptions."""
        for i in range(100):
            result = wrapper.predict(f"hello from cycle {i}")
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0
            wrapper.reset()

    def test_varying_dialog_lengths(self, wrapper):
        """Predict+reset with growing and shrinking dialog lengths."""
        for i in range(100):
            n_turns = (i % 10) + 1
            turns = [f"turn {j}" for j in range(n_turns)]
            dialog = "<ts>".join(turns)
            result = wrapper.predict(dialog)
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0
            wrapper.reset()
