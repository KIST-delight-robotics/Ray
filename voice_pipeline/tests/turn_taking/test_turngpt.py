"""Unit tests for TurnGPTWrapper.

All external dependencies (TurnGPT model) are mocked.
The ``turngpt`` package is not installed; we inject mock modules via sys.modules.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest
import torch

from voice_pipeline.core.config import TurnGPTConfig
from voice_pipeline.turn_taking.exceptions import TurnGPTError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_model(trp_prob: float = 0.5) -> MagicMock:
    """Create a mock TurnGPT model with default trp_probs output."""
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model
    mock_model.string_list_to_trp.return_value = {
        "trp_probs": torch.full((1, 5), trp_prob),
    }
    return mock_model


def _build_wrapper(mock_cls: MagicMock, **kwargs) -> tuple:
    """Construct a TurnGPTWrapper with mocked model loading.

    Returns (wrapper, mock_model).
    """
    mock_model = _make_mock_model(kwargs.pop("trp_prob", 0.5))
    mock_cls.load_from_checkpoint.return_value = mock_model

    config = TurnGPTConfig(
        checkpoint_path=kwargs.get("checkpoint_path", "/fake/turngpt.ckpt"),
        device=kwargs.get("device", "cpu"),
    )
    from voice_pipeline.turn_taking.turngpt import TurnGPTWrapper

    wrapper = TurnGPTWrapper(config)
    return wrapper, mock_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _inject_turngpt_module():
    """Inject a fake ``turngpt`` package into sys.modules so local imports work."""
    mock_turngpt_cls = MagicMock()

    turngpt_pkg = types.ModuleType("turngpt")
    turngpt_pkg.TurnGPT = mock_turngpt_cls

    saved = {"turngpt": sys.modules.get("turngpt")}
    sys.modules["turngpt"] = turngpt_pkg

    yield mock_turngpt_cls

    for key, val in saved.items():
        if val is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = val


@pytest.fixture()
def wrapper_and_model(_inject_turngpt_module):
    """Return (TurnGPTWrapper, mock_model) with default config."""
    return _build_wrapper(_inject_turngpt_module)


@pytest.fixture()
def wrapper(wrapper_and_model):
    return wrapper_and_model[0]


@pytest.fixture()
def model(wrapper_and_model):
    return wrapper_and_model[1]


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    """Model loading, device setup, eval mode."""

    def test_model_loaded_and_eval(self, _inject_turngpt_module):
        mock_cls = _inject_turngpt_module
        _, mock_model = _build_wrapper(mock_cls)

        mock_cls.load_from_checkpoint.assert_called_once_with("/fake/turngpt.ckpt")
        mock_model.to.assert_called_once_with("cpu")
        mock_model.eval.assert_called_once()

    def test_load_file_not_found_raises_turngpt_error(self, _inject_turngpt_module):
        mock_cls = _inject_turngpt_module
        mock_cls.load_from_checkpoint.side_effect = FileNotFoundError("no file")

        config = TurnGPTConfig(checkpoint_path="/missing.ckpt")
        from voice_pipeline.turn_taking.turngpt import TurnGPTWrapper

        with pytest.raises(TurnGPTError, match="Failed to load TurnGPT model"):
            TurnGPTWrapper(config)

    def test_load_runtime_error_raises_turngpt_error(self, _inject_turngpt_module):
        mock_cls = _inject_turngpt_module
        mock_cls.load_from_checkpoint.side_effect = RuntimeError("corrupt checkpoint")

        config = TurnGPTConfig(checkpoint_path="/bad.ckpt")
        from voice_pipeline.turn_taking.turngpt import TurnGPTWrapper

        with pytest.raises(TurnGPTError, match="Failed to load TurnGPT model"):
            TurnGPTWrapper(config)

    def test_device_config_respected(self, _inject_turngpt_module):
        mock_cls = _inject_turngpt_module
        _, mock_model = _build_wrapper(mock_cls, device="cuda")
        mock_model.to.assert_called_once_with("cuda")

        # Also test cpu explicitly
        _, mock_model2 = _build_wrapper(mock_cls, device="cpu")
        mock_model2.to.assert_called_once_with("cpu")


# ---------------------------------------------------------------------------
# TestPredict
# ---------------------------------------------------------------------------


class TestPredict:
    """predict() returns correct float probabilities."""

    def test_returns_float_in_range(self, wrapper):
        result = wrapper.predict("hello<ts>how are you")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_calls_string_list_to_trp_with_exact_args(self, wrapper, model):
        wrapper.predict("hello<ts>world")
        model.string_list_to_trp.assert_called_once_with(
            "hello<ts>world", add_post_eos_token=False
        )

    def test_extracts_last_position(self, _inject_turngpt_module):
        mock_cls = _inject_turngpt_module
        wrapper, mock_model = _build_wrapper(mock_cls)

        # Create tensor with different values at each position
        probs = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.9]])
        mock_model.string_list_to_trp.return_value = {"trp_probs": probs}

        result = wrapper.predict("a<ts>b<ts>c")
        assert result == pytest.approx(0.9, abs=1e-5)

    def test_single_turn_input(self, wrapper, model):
        result = wrapper.predict("hello")
        assert isinstance(result, float)
        model.string_list_to_trp.assert_called_once_with("hello", add_post_eos_token=False)

    def test_multi_turn_input(self, wrapper, model):
        result = wrapper.predict("a<ts>b<ts>c")
        assert isinstance(result, float)
        model.string_list_to_trp.assert_called_once_with(
            "a<ts>b<ts>c", add_post_eos_token=False
        )


# ---------------------------------------------------------------------------
# TestEmptyInput
# ---------------------------------------------------------------------------


class TestEmptyInput:
    """Empty/whitespace input returns default without calling model."""

    def test_empty_string_returns_default(self, wrapper, model):
        result = wrapper.predict("")
        assert result == 0.0
        model.string_list_to_trp.assert_not_called()

    def test_whitespace_returns_default(self, wrapper, model):
        result = wrapper.predict("   ")
        assert result == 0.0
        model.string_list_to_trp.assert_not_called()


# ---------------------------------------------------------------------------
# TestEdgeCaseInput
# ---------------------------------------------------------------------------


class TestEdgeCaseInput:
    """Edge cases: trailing separators, malformed input."""

    def test_trailing_ts_separator(self, wrapper, model):
        result = wrapper.predict("hello<ts>")
        assert isinstance(result, float)
        model.string_list_to_trp.assert_called_once()

    def test_only_separators(self, _inject_turngpt_module):
        mock_cls = _inject_turngpt_module
        wrapper, _ = _build_wrapper(mock_cls)

        # Model may succeed or fail on malformed input — either way, valid float
        result = wrapper.predict("<ts><ts><ts>")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# TestReset
# ---------------------------------------------------------------------------


class TestReset:
    """reset() is a no-op but must be safe."""

    def test_reset_does_not_raise(self, wrapper):
        wrapper.reset()

    def test_predict_works_after_reset(self, wrapper):
        wrapper.reset()
        result = wrapper.predict("hello")
        assert isinstance(result, float)

    def test_multiple_resets_safe(self, wrapper):
        for _ in range(10):
            wrapper.reset()
        result = wrapper.predict("hello")
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Runtime errors in predict() return default without raising."""

    def test_runtime_error_returns_default(self, wrapper, model):
        model.string_list_to_trp.side_effect = RuntimeError("CUDA OOM")
        result = wrapper.predict("hello")
        assert result == 0.0

    def test_missing_trp_probs_key_returns_default(self, wrapper, model):
        model.string_list_to_trp.return_value = {"other_key": torch.zeros(1, 5)}
        result = wrapper.predict("hello")
        assert result == 0.0

    def test_import_error_raises_turngpt_error(self):
        """If turngpt package is missing, constructor raises TurnGPTError."""
        import sys

        saved = sys.modules.pop("turngpt", None)
        # Temporarily make import fail
        import builtins

        original_import = builtins.__import__

        def _fail_import(name, *args, **kwargs):
            if name == "turngpt":
                raise ImportError("No module named 'turngpt'")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = _fail_import
        try:
            # Force re-import of the wrapper module
            sys.modules.pop("voice_pipeline.turn_taking.turngpt", None)
            config = TurnGPTConfig(checkpoint_path="/fake.ckpt")
            from voice_pipeline.turn_taking.turngpt import TurnGPTWrapper

            with pytest.raises(TurnGPTError, match="Failed to load TurnGPT model"):
                TurnGPTWrapper(config)
        finally:
            builtins.__import__ = original_import
            if saved is not None:
                sys.modules["turngpt"] = saved

    def test_nan_output_returns_nan_float(self, wrapper, model):
        """Non-finite model output propagates as float (no crash)."""
        model.string_list_to_trp.return_value = {
            "trp_probs": torch.tensor([[float("nan")]]),
        }
        result = wrapper.predict("hello")
        assert isinstance(result, float)

    def test_recovery_after_error(self, wrapper, model):
        # First call fails
        model.string_list_to_trp.side_effect = RuntimeError("fail")
        result1 = wrapper.predict("hello")
        assert result1 == 0.0

        # Second call succeeds
        model.string_list_to_trp.side_effect = None
        model.string_list_to_trp.return_value = {
            "trp_probs": torch.full((1, 5), 0.75),
        }
        result2 = wrapper.predict("hello again")
        assert result2 == pytest.approx(0.75, abs=1e-5)
