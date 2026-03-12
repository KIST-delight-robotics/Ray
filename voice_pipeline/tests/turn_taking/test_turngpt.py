"""Unit tests for TurnGPTWrapper.

All external dependencies (TurnGPT model) are mocked.
The ``turngpt`` package is not installed; we inject mock modules via sys.modules.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
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


# ===========================================================================
# ONNX backend tests
# ===========================================================================

_ONNX_EOS_ID = 50257
_ONNX_SP1_ID = 50258
_ONNX_SP2_ID = 50259
_ONNX_VOCAB = 50260
_ONNX_NL = 12


def _onnx_char_tokenize(text: str, **_kwargs) -> dict:
    """Mock HF tokenizer: char-level, <ts> -> eos_token_id."""
    ids = []
    i = 0
    while i < len(text):
        if text[i : i + 4] == "<ts>":
            ids.append(_ONNX_EOS_ID)
            i += 4
        else:
            ids.append(ord(text[i]) % 500)
            i += 1
    return {"input_ids": torch.tensor([ids])}


def _make_onnx_session_mock(*, has_kv: bool = True) -> MagicMock:
    """Create a mock ORT InferenceSession."""
    mock_sess = MagicMock()

    if has_kv:
        input_names = ["input_ids", "speaker_ids", "position_ids"]
        for i in range(_ONNX_NL):
            input_names += [f"past_key_{i}", f"past_value_{i}"]
    else:
        input_names = ["input_ids", "speaker_ids"]

    mock_inputs = []
    for name in input_names:
        inp = MagicMock()
        inp.name = name
        mock_inputs.append(inp)
    mock_sess.get_inputs.return_value = mock_inputs

    def _run(_output_names, feeds):
        ids = feeds["input_ids"]
        seq_len = ids.shape[1]
        # Logits: put high value at EOS index for predictable TRP
        logits = np.zeros((1, seq_len, _ONNX_VOCAB), dtype=np.float32)
        logits[0, -1, _ONNX_EOS_ID] = 2.0  # will give ~0.73 after softmax? no...
        # Actually let's make it simple: set eos logit high
        logits[0, -1, _ONNX_EOS_ID] = 10.0
        outputs = [logits]

        if has_kv:
            if "past_key_0" in feeds:
                past_len = feeds["past_key_0"].shape[2]
            else:
                past_len = 0
            total_len = past_len + seq_len
            for _i in range(_ONNX_NL):
                outputs.append(np.random.randn(1, 12, total_len, 64).astype(np.float32))
                outputs.append(np.random.randn(1, 12, total_len, 64).astype(np.float32))
        return outputs

    mock_sess.run.side_effect = _run
    return mock_sess


def _build_onnx_wrapper(**kwargs) -> MagicMock:
    """Build TurnGPTWrapper in ONNX mode with mocked dependencies."""

    has_kv = kwargs.pop("has_kv", True)
    mock_sess = _make_onnx_session_mock(has_kv=has_kv)

    mock_hf_tok = MagicMock()
    mock_hf_tok.side_effect = _onnx_char_tokenize
    mock_hf_tok.eos_token_id = _ONNX_EOS_ID
    mock_hf_tok.convert_tokens_to_ids.side_effect = (
        lambda t: {
            "<speaker1>": _ONNX_SP1_ID,
            "<speaker2>": _ONNX_SP2_ID,
        }.get(t, 0)
    )

    mock_so = MagicMock()

    config = TurnGPTConfig(
        onnx_model_path="/fake/model.onnx",
        tokenizer_path="/fake/tokenizer",
        max_context_tokens=kwargs.get("max_context_tokens", 1024),
        onnx_threads=kwargs.get("onnx_threads", 4),
    )

    mock_ort_module = MagicMock()
    mock_ort_module.SessionOptions.return_value = mock_so
    mock_ort_module.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
    mock_ort_module.InferenceSession.return_value = mock_sess

    mock_transformers = MagicMock()
    mock_transformers.GPT2TokenizerFast.from_pretrained.return_value = mock_hf_tok

    import sys
    saved_ort = sys.modules.get("onnxruntime")
    saved_tf = sys.modules.get("transformers")
    sys.modules["onnxruntime"] = mock_ort_module
    sys.modules["transformers"] = mock_transformers

    try:
        from voice_pipeline.turn_taking.turngpt import TurnGPTWrapper

        wrapper = TurnGPTWrapper(config)
    finally:
        if saved_ort is None:
            sys.modules.pop("onnxruntime", None)
        else:
            sys.modules["onnxruntime"] = saved_ort
        if saved_tf is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = saved_tf

    return wrapper, mock_sess


class TestOnnxInit:
    """ONNX backend initialization."""

    def test_backend_set_to_onnx(self):
        wrapper, _ = _build_onnx_wrapper()
        assert wrapper._backend == "onnx"

    def test_onnx_model_path_empty_uses_pytorch(self, _inject_turngpt_module):
        mock_cls = _inject_turngpt_module
        wrapper, _ = _build_wrapper(mock_cls)
        assert wrapper._backend == "pytorch"

    def test_missing_tokenizer_path_raises_turngpt_error(self):
        from voice_pipeline.turn_taking.turngpt import TurnGPTWrapper

        config = TurnGPTConfig(
            onnx_model_path="/fake/model.onnx",
            tokenizer_path="",
        )
        with pytest.raises(TurnGPTError, match="tokenizer_path is required"):
            TurnGPTWrapper(config)

    def test_onnx_init_error_raises_turngpt_error(self):
        config = TurnGPTConfig(
            onnx_model_path="/fake/model.onnx",
            tokenizer_path="/fake/tokenizer",
        )

        # Make onnxruntime import fail
        import builtins

        original_import = builtins.__import__

        def _fail_ort(name, *args, **kwargs):
            if name == "onnxruntime":
                raise ImportError("no onnxruntime")
            return original_import(name, *args, **kwargs)

        saved_ort = sys.modules.pop("onnxruntime", None)
        builtins.__import__ = _fail_ort
        try:
            from voice_pipeline.turn_taking.turngpt import TurnGPTWrapper

            with pytest.raises(TurnGPTError, match="Failed to load ONNX model"):
                TurnGPTWrapper(config)
        finally:
            builtins.__import__ = original_import
            if saved_ort is not None:
                sys.modules["onnxruntime"] = saved_ort

    def test_detects_kv_model(self):
        wrapper, _ = _build_onnx_wrapper(has_kv=True)
        assert wrapper._onnx_has_kv is True

    def test_detects_no_cache_model(self):
        wrapper, _ = _build_onnx_wrapper(has_kv=False)
        assert wrapper._onnx_has_kv is False


class TestOnnxPredict:
    """ONNX backend predict() behavior."""

    def test_returns_float_in_range(self):
        wrapper, _ = _build_onnx_wrapper()
        result = wrapper.predict("hello<ts>how are you")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_empty_input_returns_default(self):
        wrapper, _ = _build_onnx_wrapper()
        assert wrapper.predict("") == 0.0
        assert wrapper.predict("   ") == 0.0

    def test_single_turn(self):
        wrapper, sess = _build_onnx_wrapper()
        result = wrapper.predict("hello")
        assert isinstance(result, float)
        sess.run.assert_called_once()

    def test_multi_turn(self):
        wrapper, sess = _build_onnx_wrapper()
        result = wrapper.predict("a<ts>b<ts>c")
        assert isinstance(result, float)
        sess.run.assert_called_once()

    def test_no_cache_model(self):
        wrapper, sess = _build_onnx_wrapper(has_kv=False)
        result = wrapper.predict("hello<ts>world")
        assert isinstance(result, float)
        sess.run.assert_called_once()
        feeds = sess.run.call_args[0][1]
        assert "input_ids" in feeds
        assert "speaker_ids" in feeds
        assert "position_ids" not in feeds


class TestOnnxCache:
    """ONNX KV cache reuse behavior."""

    def test_cache_reuse_on_prefix_match(self):
        wrapper, sess = _build_onnx_wrapper()
        wrapper.predict("hello")
        sess.run.reset_mock()

        wrapper.predict("hello world")
        sess.run.assert_called_once()
        feeds = sess.run.call_args[0][1]
        # Should have received only new tokens
        assert feeds["input_ids"].shape[1] == 6  # " world" = 6 chars
        # Should have non-empty past
        assert feeds["past_key_0"].shape[2] > 0

    def test_identical_input_returns_cached(self):
        wrapper, sess = _build_onnx_wrapper()
        wrapper.predict("hello<ts>world")
        sess.run.reset_mock()

        result = wrapper.predict("hello<ts>world")
        sess.run.assert_not_called()
        assert isinstance(result, float)

    def test_cache_invalidation_on_prefix_change(self):
        wrapper, sess = _build_onnx_wrapper()
        wrapper.predict("hello")
        sess.run.reset_mock()

        wrapper.predict("world")
        sess.run.assert_called_once()
        feeds = sess.run.call_args[0][1]
        assert feeds["input_ids"].shape[1] == 5  # full "world"
        assert feeds["past_key_0"].shape[2] == 0  # empty past

    def test_cache_cleared_on_reset(self):
        wrapper, sess = _build_onnx_wrapper()
        wrapper.predict("hello")
        sess.run.reset_mock()

        wrapper.reset()
        wrapper.predict("hello")
        sess.run.assert_called_once()
        feeds = sess.run.call_args[0][1]
        assert feeds["past_key_0"].shape[2] == 0

    def test_no_cache_model_no_reuse(self):
        wrapper, sess = _build_onnx_wrapper(has_kv=False)
        wrapper.predict("hello")
        sess.run.reset_mock()

        # Same input — no cache, must call again
        wrapper.predict("hello world")
        sess.run.assert_called_once()


class TestOnnxErrorHandling:
    """ONNX predict error recovery."""

    def test_runtime_error_returns_default(self):
        wrapper, sess = _build_onnx_wrapper()
        sess.run.side_effect = RuntimeError("ORT error")
        result = wrapper.predict("hello")
        assert result == 0.0

    def test_recovery_after_error(self):
        wrapper, sess = _build_onnx_wrapper()
        # Fail first
        sess.run.side_effect = RuntimeError("fail")
        assert wrapper.predict("hello") == 0.0

        # Restore
        sess.run.side_effect = _make_onnx_session_mock(has_kv=True).run.side_effect
        result = wrapper.predict("hello again")
        assert isinstance(result, float)
        assert result > 0.0


# ---------------------------------------------------------------------------
# Test _build_speaker_ids
# ---------------------------------------------------------------------------


class TestOnnxWindow:
    """ONNX backend window eviction behavior."""

    def test_eviction_triggered_on_overflow(self):
        wrapper, sess = _build_onnx_wrapper(max_context_tokens=20)
        dialog = "aaaa<ts>bbbb<ts>cccc<ts>dddd<ts>eeee"
        result = wrapper.predict(dialog)
        assert isinstance(result, float)
        # Should have called run (not crash)
        assert sess.run.call_count >= 1

    def test_eviction_removes_oldest_turns(self):
        wrapper, sess = _build_onnx_wrapper(max_context_tokens=20)
        dialog = "first<ts>second<ts>third<ts>last"
        wrapper.predict(dialog)
        feeds = sess.run.call_args[0][1]
        ids = feeds["input_ids"]
        # After eviction, should fit within max_context_tokens
        assert ids.shape[1] <= 20

    def test_single_long_turn_token_truncation(self):
        wrapper, sess = _build_onnx_wrapper(max_context_tokens=10)
        long_text = "a" * 30
        result = wrapper.predict(long_text)
        assert isinstance(result, float)
        feeds = sess.run.call_args[0][1]
        assert feeds["input_ids"].shape[1] <= 10


class TestExtractTrpNumpy:
    """Test _extract_trp_numpy correctness."""

    def test_returns_eos_probability(self):
        from voice_pipeline.turn_taking.turngpt import _extract_trp_numpy

        # Create logits where EOS token gets a high logit
        vocab_size = 100
        eos_id = 50
        logits = np.zeros((1, 3, vocab_size), dtype=np.float32)
        logits[0, -1, eos_id] = 10.0  # high logit at EOS

        trp = _extract_trp_numpy(logits, eos_id)
        assert isinstance(trp, float)
        assert 0.0 < trp <= 1.0
        # With logit=10 vs 0 for all others, EOS prob should be dominant
        assert trp > 0.5

    def test_uniform_logits_gives_uniform_prob(self):
        from voice_pipeline.turn_taking.turngpt import _extract_trp_numpy

        vocab_size = 100
        eos_id = 50
        logits = np.zeros((1, 1, vocab_size), dtype=np.float32)

        trp = _extract_trp_numpy(logits, eos_id)
        assert trp == pytest.approx(1.0 / vocab_size, abs=1e-5)

    def test_extracts_last_position_only(self):
        from voice_pipeline.turn_taking.turngpt import _extract_trp_numpy

        vocab_size = 100
        eos_id = 50
        logits = np.zeros((1, 3, vocab_size), dtype=np.float32)
        # High EOS at position 0, zero at last position
        logits[0, 0, eos_id] = 100.0
        logits[0, -1, eos_id] = 0.0

        trp = _extract_trp_numpy(logits, eos_id)
        # Should use last position, which has uniform logits
        assert trp == pytest.approx(1.0 / vocab_size, abs=1e-5)


class TestBuildSpeakerIds:
    """Test the speaker_ids construction logic."""

    def test_single_turn_all_speaker1(self):
        from voice_pipeline.turn_taking.turngpt import _build_speaker_ids

        ids = torch.tensor([[10, 20, 30]])
        sp = _build_speaker_ids(ids, _ONNX_EOS_ID, _ONNX_SP1_ID, _ONNX_SP2_ID)
        assert (sp == _ONNX_SP1_ID).all()

    def test_two_turns_alternates(self):
        from voice_pipeline.turn_taking.turngpt import _build_speaker_ids

        # "hello<ts>world" → [h, e, l, l, o, <ts>, w, o, r, l, d]
        ids = torch.tensor([[10, 20, 30, _ONNX_EOS_ID, 40, 50]])
        sp = _build_speaker_ids(ids, _ONNX_EOS_ID, _ONNX_SP1_ID, _ONNX_SP2_ID)
        # Before <ts>: sp1, after <ts>: sp2
        assert sp[0, 0].item() == _ONNX_SP1_ID
        assert sp[0, 3].item() == _ONNX_SP1_ID  # <ts> itself is sp1
        assert sp[0, 4].item() == _ONNX_SP2_ID
        assert sp[0, 5].item() == _ONNX_SP2_ID

    def test_three_turns(self):
        from voice_pipeline.turn_taking.turngpt import _build_speaker_ids

        ids = torch.tensor([[10, _ONNX_EOS_ID, 20, _ONNX_EOS_ID, 30]])
        sp = _build_speaker_ids(ids, _ONNX_EOS_ID, _ONNX_SP1_ID, _ONNX_SP2_ID)
        assert sp[0, 0].item() == _ONNX_SP1_ID  # turn 1
        assert sp[0, 2].item() == _ONNX_SP2_ID  # turn 2
        assert sp[0, 4].item() == _ONNX_SP1_ID  # turn 3
