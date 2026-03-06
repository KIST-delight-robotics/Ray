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
# Helpers: char-level mock tokenizer
# ---------------------------------------------------------------------------

_EOS_TOKEN_ID = 50256


def _char_tokenize(text: str, **_kwargs) -> dict:
    """Mock tokenizer: each char -> ord(c), '<ts>' -> eos_token_id."""
    ids = []
    i = 0
    while i < len(text):
        if text[i : i + 4] == "<ts>":
            ids.append(_EOS_TOKEN_ID)
            i += 4
        else:
            ids.append(ord(text[i]))
            i += 1
    input_ids = torch.tensor([ids])
    # speaker_ids: 0 before first <ts>, alternating on each <ts>
    speaker = []
    current = 0
    for t in ids:
        speaker.append(current)
        if t == _EOS_TOKEN_ID:
            current = 1 - current
    speaker_ids = torch.tensor([speaker])
    return {"input_ids": input_ids, "speaker_ids": speaker_ids}


def _make_logits(seq_len: int, trp_prob: float = 0.5) -> torch.Tensor:
    """Create logits tensor where softmax at last position gives trp_prob at eos index."""
    # Simple approach: create a tensor that after softmax gives roughly trp_prob
    # at the _EOS_TOKEN_ID index. Use a 2-class simplification via get_trp mock.
    return torch.randn(1, seq_len, 100)


def _make_past_kv(n_layers: int = 2, seq_len: int = 5) -> tuple:
    """Create a fake past_key_values tuple."""
    return tuple(
        (torch.randn(1, 4, seq_len, 16), torch.randn(1, 4, seq_len, 16)) for _ in range(n_layers)
    )


# ---------------------------------------------------------------------------
# Mock model factory
# ---------------------------------------------------------------------------


def _make_mock_model(trp_prob: float = 0.5) -> MagicMock:
    """Create a mock TurnGPT model with tokenizer, forward, and get_trp."""
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model

    # Tokenizer
    mock_model.tokenizer.side_effect = _char_tokenize

    # Forward: model(input_ids, speaker_ids=..., ...) -> dict
    def _forward(input_ids, speaker_ids=None, past_key_values=None, use_cache=False):
        seq_len = input_ids.shape[-1]
        if past_key_values is not None:
            # Total seq_len = past seq + new tokens
            past_seq = past_key_values[0][0].shape[2]
            total_seq = past_seq + seq_len
        else:
            total_seq = seq_len
        return {
            "logits": _make_logits(seq_len, trp_prob),
            "past_key_values": _make_past_kv(seq_len=total_seq),
        }

    mock_model.side_effect = _forward

    # get_trp: extract TRP from probs
    def _get_trp(probs):
        return torch.full(probs.shape[:-1], trp_prob)

    mock_model.get_trp.side_effect = _get_trp

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
        max_context_tokens=kwargs.get("max_context_tokens", 1024),
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

    def test_calls_model_forward(self, wrapper, model):
        wrapper.predict("hello<ts>world")
        model.assert_called_once()

    def test_extracts_last_position(self, _inject_turngpt_module):
        mock_cls = _inject_turngpt_module
        wrapper, mock_model = _build_wrapper(mock_cls, trp_prob=0.9)

        result = wrapper.predict("a<ts>b<ts>c")
        assert result == pytest.approx(0.9, abs=1e-5)

    def test_single_turn_input(self, wrapper, model):
        result = wrapper.predict("hello")
        assert isinstance(result, float)
        model.assert_called_once()

    def test_multi_turn_input(self, wrapper, model):
        result = wrapper.predict("a<ts>b<ts>c")
        assert isinstance(result, float)
        model.assert_called_once()


# ---------------------------------------------------------------------------
# TestEmptyInput
# ---------------------------------------------------------------------------


class TestEmptyInput:
    """Empty/whitespace input returns default without calling model."""

    def test_empty_string_returns_default(self, wrapper, model):
        result = wrapper.predict("")
        assert result == 0.0
        model.assert_not_called()

    def test_whitespace_returns_default(self, wrapper, model):
        result = wrapper.predict("   ")
        assert result == 0.0
        model.assert_not_called()


# ---------------------------------------------------------------------------
# TestEdgeCaseInput
# ---------------------------------------------------------------------------


class TestEdgeCaseInput:
    """Edge cases: trailing separators, malformed input."""

    def test_trailing_ts_separator(self, wrapper, model):
        result = wrapper.predict("hello<ts>")
        assert isinstance(result, float)
        model.assert_called_once()

    def test_only_separators(self, _inject_turngpt_module):
        mock_cls = _inject_turngpt_module
        wrapper, _ = _build_wrapper(mock_cls)

        result = wrapper.predict("<ts><ts><ts>")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# TestReset
# ---------------------------------------------------------------------------


class TestReset:
    """reset() clears cache."""

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

    def test_cache_cleared_on_reset(self, wrapper, model):
        """After reset, next predict does a full forward (no cache reuse)."""
        wrapper.predict("hello")
        model.reset_mock()

        wrapper.reset()
        wrapper.predict("hello")
        # Full forward should be called (not cached return)
        model.assert_called_once()


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Runtime errors in predict() return default without raising."""

    def test_runtime_error_returns_default(self, wrapper, model):
        model.side_effect = RuntimeError("CUDA OOM")
        result = wrapper.predict("hello")
        assert result == 0.0

    def test_tokenizer_error_returns_default(self, wrapper, model):
        model.tokenizer.side_effect = RuntimeError("tokenizer broken")
        result = wrapper.predict("hello")
        assert result == 0.0

    def test_import_error_raises_turngpt_error(self):
        """If turngpt package is missing, constructor raises TurnGPTError."""
        import sys

        saved = sys.modules.pop("turngpt", None)
        import builtins

        original_import = builtins.__import__

        def _fail_import(name, *args, **kwargs):
            if name == "turngpt":
                raise ImportError("No module named 'turngpt'")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = _fail_import
        try:
            sys.modules.pop("voice_pipeline.turn_taking.turngpt", None)
            config = TurnGPTConfig(checkpoint_path="/fake.ckpt")
            from voice_pipeline.turn_taking.turngpt import TurnGPTWrapper

            with pytest.raises(TurnGPTError, match="Failed to load TurnGPT model"):
                TurnGPTWrapper(config)
        finally:
            builtins.__import__ = original_import
            if saved is not None:
                sys.modules["turngpt"] = saved

    def test_recovery_after_error(self, wrapper, model):
        # First call fails
        model.side_effect = RuntimeError("fail")
        result1 = wrapper.predict("hello")
        assert result1 == 0.0

        # Second call succeeds — restore mock
        model.side_effect = _make_mock_model(0.75).side_effect
        model.get_trp.side_effect = lambda p: torch.full(p.shape[:-1], 0.75)
        result2 = wrapper.predict("hello again")
        assert result2 == pytest.approx(0.75, abs=1e-5)


# ---------------------------------------------------------------------------
# TestCache
# ---------------------------------------------------------------------------


class TestCache:
    """KV cache reuse behavior."""

    def test_cache_reuse_on_prefix_match(self, wrapper, model):
        """Growing suffix reuses cache — model receives only new tokens."""
        wrapper.predict("hello")
        model.reset_mock()

        wrapper.predict("hello world")
        # Model should be called with only the new tokens (incremental)
        model.assert_called_once()
        args, kwargs = model.call_args
        new_ids = args[0]
        # "hello" = 5 chars, "hello world" = 11 chars → new = " world" = 6 chars
        assert new_ids.shape[-1] == 6
        assert kwargs.get("past_key_values") is not None

    def test_cache_invalidation_on_prefix_change(self, wrapper, model):
        """Completely different input invalidates cache — full forward."""
        wrapper.predict("hello")
        model.reset_mock()

        wrapper.predict("world")
        model.assert_called_once()
        args, kwargs = model.call_args
        new_ids = args[0]
        # Full forward: all 5 tokens for "world"
        assert new_ids.shape[-1] == 5
        assert kwargs.get("past_key_values") is None

    def test_identical_input_returns_cached(self, wrapper, model):
        """Same input twice — model called only once."""
        wrapper.predict("hello<ts>world")
        model.reset_mock()

        result = wrapper.predict("hello<ts>world")
        model.assert_not_called()
        assert isinstance(result, float)

    def test_shrinking_input_recomputes(self, wrapper, model):
        """ASR correction shortens text — must not return stale cached result."""
        wrapper.predict("hello world")
        model.reset_mock()

        # Shorter input that is a prefix of cached — must recompute
        wrapper.predict("hello")
        model.assert_called_once()

    def test_cache_cleared_on_reset(self, wrapper, model):
        """After reset, identical input triggers full forward."""
        wrapper.predict("hello")
        model.reset_mock()

        wrapper.reset()
        wrapper.predict("hello")
        model.assert_called_once()
        _, kwargs = model.call_args
        # No past_key_values after reset
        assert kwargs.get("past_key_values") is None


# ---------------------------------------------------------------------------
# TestWindow
# ---------------------------------------------------------------------------


class TestWindow:
    """Context window eviction behavior."""

    def test_eviction_triggered_on_overflow(self, _inject_turngpt_module):
        """Long dialog exceeding max_context_tokens triggers eviction."""
        mock_cls = _inject_turngpt_module
        # max_context_tokens=20: very small window
        wrapper, model = _build_wrapper(mock_cls, max_context_tokens=20)

        # "aaaa<ts>bbbb<ts>cccc<ts>dddd" = many tokens
        dialog = "aaaa<ts>bbbb<ts>cccc<ts>dddd<ts>eeee"
        result = wrapper.predict(dialog)
        assert isinstance(result, float)

        # Tokenizer should have been called multiple times (initial + eviction retries)
        assert model.tokenizer.call_count >= 2

    def test_eviction_removes_oldest_turns(self, _inject_turngpt_module):
        """After eviction, the oldest turn(s) are removed."""
        mock_cls = _inject_turngpt_module
        wrapper, model = _build_wrapper(mock_cls, max_context_tokens=20)

        dialog = "first<ts>second<ts>third<ts>last"
        wrapper.predict(dialog)

        # Check the last tokenizer call to see what text was tokenized
        last_call = model.tokenizer.call_args_list[-1]
        tokenized_text = last_call[0][0]
        # "first" should have been evicted
        assert "first" not in tokenized_text
        # "last" should still be present
        assert "last" in tokenized_text

    def test_no_eviction_within_limit(self, _inject_turngpt_module):
        """Short dialog within window — tokenizer called once."""
        mock_cls = _inject_turngpt_module
        wrapper, model = _build_wrapper(mock_cls, max_context_tokens=1024)

        wrapper.predict("hello")
        assert model.tokenizer.call_count == 1

    def test_headroom_prevents_thrashing(self, _inject_turngpt_module):
        """Eviction reduces below headroom target, not just below max."""
        mock_cls = _inject_turngpt_module
        wrapper, model = _build_wrapper(mock_cls, max_context_tokens=30)

        # Dialog that's just over limit — eviction should cut to 80% (24 tokens)
        dialog = "aaaa<ts>bbbb<ts>cccc<ts>dddd"
        wrapper.predict(dialog)

        last_call = model.tokenizer.call_args_list[-1]
        tokenized_text = last_call[0][0]
        final_tokens = _char_tokenize(tokenized_text)["input_ids"]
        assert final_tokens.shape[-1] <= int(30 * 0.8)

    def test_single_long_turn_token_truncation(self, _inject_turngpt_module):
        """Single turn exceeding max_context_tokens is left-truncated at token level."""
        mock_cls = _inject_turngpt_module
        wrapper, model = _build_wrapper(mock_cls, max_context_tokens=10)

        # No <ts> separator — cannot evict turns, must token-truncate
        long_text = "a" * 30
        result = wrapper.predict(long_text)
        assert isinstance(result, float)

        # Model should receive at most max_context_tokens
        args, _ = model.call_args
        assert args[0].shape[-1] <= 10
