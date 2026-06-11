"""Unit tests for VAPWrapper.

All external dependencies (VapGPT, torch.load, torchaudio) are mocked.
The ``vap`` package is not installed; we inject mock modules via sys.modules.
"""

from __future__ import annotations

import struct
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

from voice_pipeline.core.types import VAPResult
from voice_pipeline.tts.openai_tts import OpenAITTS
from voice_pipeline.turn_taking.exceptions import VAPError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VAP_MODULE = "voice_pipeline.turn_taking.vap"


def _make_pcm(n_samples: int, amplitude: int = 0) -> bytes:
    """Create 16-bit PCM bytes of constant amplitude."""
    return struct.pack(f"<{n_samples}h", *([amplitude] * n_samples))


def _mock_model_probs(p_now: float = 0.5, p_fut: float = 0.3, vad: float = 0.8):
    """Return a dict mimicking VAP model.probs() output."""
    T = 50
    return {
        "p_now": torch.full((1, T, 2), p_now),
        "p_future": torch.full((1, T, 2), p_fut),
        "vad": torch.full((1, T, 2), vad),
    }


def _make_mock_model():
    """Create a mock VAP model with default probs output."""
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model
    mock_model.probs.return_value = _mock_model_probs()
    return mock_model


def _build_wrapper(mock_vapgpt_cls, mock_torch_load, *, tts_rate: int = OpenAITTS.OUTPUT_SAMPLE_RATE):
    """Construct a VAPWrapper with mocked model loading.

    Class var overrides (`_MODEL_PATH`, `_CONTEXT_SEC`, `_STEP_SEC`, `_TT_TIME`)
    must be applied before this call via `monkeypatch.setattr(VAPWrapper, ...)`.
    The `_patches` fixture ensures `_MODEL_PATH` is "/fake/model.pt" by default.
    """
    mock_model = _make_mock_model()
    mock_vapgpt_cls.return_value = mock_model
    mock_torch_load.return_value = {}

    from voice_pipeline.turn_taking.vap import VAPWrapper

    wrapper = VAPWrapper(tts_rate)
    return wrapper, mock_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _inject_vap_module():
    """Inject a fake ``vap.model`` module into sys.modules so local imports work."""
    mock_vapgpt_cls = MagicMock()
    mock_vapconfig_cls = MagicMock()

    vap_pkg = types.ModuleType("vap")
    vap_model_mod = types.ModuleType("vap.model")
    vap_model_mod.VapGPT = mock_vapgpt_cls
    vap_model_mod.VapConfig = mock_vapconfig_cls

    saved = {
        "vap": sys.modules.get("vap"),
        "vap.model": sys.modules.get("vap.model"),
    }
    sys.modules["vap"] = vap_pkg
    sys.modules["vap.model"] = vap_model_mod

    yield mock_vapgpt_cls, mock_vapconfig_cls

    # Restore
    for key, val in saved.items():
        if val is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = val


@pytest.fixture()
def _patches(_inject_vap_module, monkeypatch):
    """Provide (mock_vapgpt_cls, mock_torch_load); also default _MODEL_PATH to fake."""
    mock_vapgpt_cls, _ = _inject_vap_module
    from voice_pipeline.turn_taking.vap import VAPWrapper

    monkeypatch.setattr(VAPWrapper, "_MODEL_PATH", "/fake/model.pt")
    with patch("torch.load") as mock_torch_load:
        yield mock_vapgpt_cls, mock_torch_load


@pytest.fixture()
def wrapper_and_model(_patches):
    """Return (VAPWrapper, mock_model) with default config."""
    mock_cls, mock_load = _patches
    return _build_wrapper(mock_cls, mock_load)


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
    """Model loading, buffer shape, eval/device setup."""

    def test_model_loaded_and_eval(self, _patches):
        mock_cls, mock_load = _patches
        _, mock_model = _build_wrapper(mock_cls, mock_load)

        mock_model.to.assert_called_once_with("cpu")
        mock_model.eval.assert_called_once()
        mock_model.load_state_dict.assert_called_once()

    def test_model_load_failure_raises_vap_error(self, _patches):
        mock_cls, mock_load = _patches
        mock_load.side_effect = FileNotFoundError("no file")

        from voice_pipeline.turn_taking.vap import VAPWrapper

        with pytest.raises(VAPError, match="Failed to load VAP model"):
            VAPWrapper(OpenAITTS.OUTPUT_SAMPLE_RATE)

    def test_buffer_shape(self, wrapper):
        assert wrapper._buffer.shape == (1, 2, 320000)  # 20s * 16kHz

    def test_weights_only_true(self, _patches):
        mock_cls, mock_load = _patches
        _build_wrapper(mock_cls, mock_load)
        mock_load.assert_called_once_with("/fake/model.pt", map_location="cpu", weights_only=True)

    def test_zero_step_samples_raises(self, _patches, monkeypatch):
        from voice_pipeline.turn_taking.vap import VAPWrapper

        monkeypatch.setattr(VAPWrapper, "_STEP_SEC", 0.00001)
        mock_cls, mock_load = _patches
        with pytest.raises(VAPError, match="All must be >= 1"):
            _build_wrapper(mock_cls, mock_load)

    def test_zero_tt_frames_raises(self, _patches, monkeypatch):
        from voice_pipeline.turn_taking.vap import VAPWrapper

        monkeypatch.setattr(VAPWrapper, "_TT_TIME", 0.001)
        mock_cls, mock_load = _patches
        with pytest.raises(VAPError, match="All must be >= 1"):
            _build_wrapper(mock_cls, mock_load)

    def test_zero_context_raises(self, _patches, monkeypatch):
        from voice_pipeline.turn_taking.vap import VAPWrapper

        monkeypatch.setattr(VAPWrapper, "_CONTEXT_SEC", 0.0)
        mock_cls, mock_load = _patches
        with pytest.raises(VAPError, match="All must be >= 1"):
            _build_wrapper(mock_cls, mock_load)


# ---------------------------------------------------------------------------
# TestFeedAudio
# ---------------------------------------------------------------------------


class TestFeedAudio:
    """feed_audio returns default before first inference, triggers on Nth call."""

    def test_returns_default_before_inference(self, wrapper):
        frame = _make_pcm(480)
        result = wrapper.feed_audio(frame)
        assert result == VAPResult(0.0, 0.0, False)

    def test_inference_triggers_on_step(self, wrapper, model):
        frame = _make_pcm(480)
        model.probs.return_value = _mock_model_probs(p_now=0.7, p_fut=0.4, vad=0.9)

        # 480 * 4 = 1920 >= 1600, so inference fires on 4th call
        for _ in range(3):
            result = wrapper.feed_audio(frame)
        assert result == VAPResult(0.0, 0.0, False)  # still default

        result = wrapper.feed_audio(frame)
        assert model.probs.call_count == 1
        assert result.p_now == pytest.approx(0.7, abs=1e-5)
        assert result.p_fut == pytest.approx(0.4, abs=1e-5)
        assert result.user_is_speaking is True

    def test_cached_result_between_inferences(self, wrapper, model):
        frame = _make_pcm(480)
        model.probs.return_value = _mock_model_probs(p_now=0.6, p_fut=0.2, vad=0.8)

        # Trigger first inference
        for _ in range(4):
            wrapper.feed_audio(frame)

        # Next calls return cached, no new inference
        result = wrapper.feed_audio(frame)
        assert model.probs.call_count == 1
        assert result.p_now == pytest.approx(0.6, abs=1e-5)

    def test_feed_with_robot_audio(self, wrapper, model):
        user_frame = _make_pcm(480)
        robot_frame = _make_pcm(720)  # 24kHz, 30ms
        model.probs.return_value = _mock_model_probs(p_now=0.5, p_fut=0.5, vad=0.6)

        for _ in range(4):
            result = wrapper.feed_audio(user_frame, robot_frame)

        assert model.probs.call_count == 1
        assert result.p_now == pytest.approx(0.5, abs=1e-5)


# ---------------------------------------------------------------------------
# TestBufferRolling
# ---------------------------------------------------------------------------


class TestBufferRolling:
    """User/robot data placement in the rolling buffer."""

    def test_user_data_in_channel_0(self, wrapper):
        user_frame = _make_pcm(480, amplitude=1000)
        wrapper.feed_audio(user_frame)

        tail = wrapper._buffer[0, 0, -480:]
        assert tail.abs().sum() > 0

    def test_robot_silence_in_channel_1(self, wrapper):
        user_frame = _make_pcm(480, amplitude=1000)
        wrapper.feed_audio(user_frame)

        assert wrapper._buffer[0, 1, -480:].abs().sum() == 0

    def test_robot_data_in_channel_1(self, wrapper):
        user_frame = _make_pcm(480, amplitude=0)
        robot_frame = _make_pcm(720, amplitude=5000)  # 24kHz
        wrapper.feed_audio(user_frame, robot_frame)

        tail = wrapper._buffer[0, 1, -480:]
        assert tail.abs().sum() > 0


# ---------------------------------------------------------------------------
# TestRobotAudioResampling
# ---------------------------------------------------------------------------


class TestRobotAudioResampling:
    """Resample from TTS rate to pipeline rate."""

    def test_resample_called_when_rates_differ(self, _patches):
        mock_cls, mock_load = _patches
        wrapper, _ = _build_wrapper(mock_cls, mock_load, tts_rate=24000)

        with patch(f"{_VAP_MODULE}.torchaudio.functional.resample") as mock_resample:
            mock_resample.return_value = torch.zeros(1, 480)
            robot_frame = _make_pcm(720)
            wrapper._decode_and_resample_robot(robot_frame, 480)
            mock_resample.assert_called_once()
            args = mock_resample.call_args
            assert args.kwargs["orig_freq"] == 24000
            assert args.kwargs["new_freq"] == 16000

    def test_no_resample_when_rates_match(self, _patches):
        mock_cls, mock_load = _patches
        wrapper, _ = _build_wrapper(mock_cls, mock_load, tts_rate=16000)

        with patch(f"{_VAP_MODULE}.torchaudio.functional.resample") as mock_resample:
            robot_frame = _make_pcm(480)
            result = wrapper._decode_and_resample_robot(robot_frame, 480)
            mock_resample.assert_not_called()
            assert result.shape[0] == 480

    def test_output_length_matches_target(self, _patches):
        mock_cls, mock_load = _patches
        wrapper, _ = _build_wrapper(mock_cls, mock_load, tts_rate=24000)

        robot_frame = _make_pcm(720, amplitude=100)
        result = wrapper._decode_and_resample_robot(robot_frame, 480)
        assert result.shape[0] == 480


# ---------------------------------------------------------------------------
# TestPCMConversion
# ---------------------------------------------------------------------------


class TestPCMConversion:
    """PCM bytes to float32 tensor conversion."""

    def test_silence_yields_zeros(self, wrapper):
        pcm = _make_pcm(480, amplitude=0)
        tensor = wrapper._pcm_to_tensor(pcm)
        assert tensor.shape[0] == 480
        assert tensor.abs().sum() == 0

    def test_max_amplitude(self, wrapper):
        pcm = _make_pcm(10, amplitude=32767)
        tensor = wrapper._pcm_to_tensor(pcm)
        assert tensor.max().item() == pytest.approx(32767.0 / 32768.0, abs=1e-5)

    def test_dtype_is_float32(self, wrapper):
        pcm = _make_pcm(480)
        tensor = wrapper._pcm_to_tensor(pcm)
        assert tensor.dtype == torch.float32

    def test_sample_count_matches_input(self, wrapper):
        pcm = _make_pcm(100)
        tensor = wrapper._pcm_to_tensor(pcm)
        assert tensor.shape[0] == 100


# ---------------------------------------------------------------------------
# TestReset
# ---------------------------------------------------------------------------


class TestReset:
    """reset() clears buffer, counter, and cached result."""

    def test_buffer_zeroed(self, wrapper, model):
        frame = _make_pcm(480, amplitude=5000)
        model.probs.return_value = _mock_model_probs(p_now=0.9, p_fut=0.8, vad=0.9)
        for _ in range(4):
            wrapper.feed_audio(frame)

        wrapper.reset()
        assert wrapper._buffer.abs().sum() == 0

    def test_counter_cleared(self, wrapper):
        frame = _make_pcm(480)
        wrapper.feed_audio(frame)
        wrapper.feed_audio(frame)
        assert wrapper._samples_since_inference > 0

        wrapper.reset()
        assert wrapper._samples_since_inference == 0

    def test_cached_result_reset(self, wrapper, model):
        frame = _make_pcm(480)
        model.probs.return_value = _mock_model_probs(p_now=0.9, p_fut=0.8, vad=0.9)
        for _ in range(4):
            wrapper.feed_audio(frame)

        wrapper.reset()
        assert wrapper._cached_result == VAPResult(0.0, 0.0, False)


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Runtime errors return default result without raising."""

    def test_inference_error_returns_default(self, wrapper, model):
        frame = _make_pcm(480)
        model.probs.side_effect = RuntimeError("CUDA OOM")

        for _ in range(4):
            result = wrapper.feed_audio(frame)
        assert result == VAPResult(0.0, 0.0, False)

    def test_empty_audio_returns_cached(self, wrapper):
        # Empty audio should return cached result cleanly (no exception path)
        result = wrapper.feed_audio(b"")
        assert result == VAPResult(0.0, 0.0, False)

    def test_pcm_decode_error_returns_default(self, wrapper):
        result = wrapper.feed_audio(b"\x00\x01\x02")
        assert result == VAPResult(0.0, 0.0, False)

    def test_recovery_after_error(self, wrapper, model):
        frame = _make_pcm(480)
        model.probs.side_effect = RuntimeError("fail")
        for _ in range(4):
            wrapper.feed_audio(frame)

        model.probs.side_effect = None
        model.probs.return_value = _mock_model_probs(p_now=0.65, p_fut=0.35, vad=0.7)
        for _ in range(4):
            result = wrapper.feed_audio(frame)
        assert result.p_now == pytest.approx(0.65, abs=1e-5)


# ---------------------------------------------------------------------------
# TestInferenceTiming
# ---------------------------------------------------------------------------


class TestInferenceTiming:
    """Verify computed timing constants."""

    def test_step_samples(self, wrapper):
        assert wrapper._step_samples == 1600  # 0.1 * 16000

    def test_tt_frames(self, wrapper):
        assert wrapper._tt_frames == 25  # 0.5 * 50

    def test_n_samples(self, wrapper):
        assert wrapper._n_samples == 320000  # 20.0 * 16000

    def test_custom_step_sec(self, _patches, monkeypatch):
        from voice_pipeline.turn_taking.vap import VAPWrapper

        monkeypatch.setattr(VAPWrapper, "_STEP_SEC", 0.2)
        mock_cls, mock_load = _patches
        wrapper, _ = _build_wrapper(mock_cls, mock_load)
        assert wrapper._step_samples == 3200  # 0.2 * 16000

    def test_oversized_frame_clamped(self, _patches, monkeypatch):
        """Frame larger than context buffer is clamped to buffer size."""
        from voice_pipeline.turn_taking.vap import VAPWrapper

        # Small context buffer: 0.01s = 160 samples
        monkeypatch.setattr(VAPWrapper, "_CONTEXT_SEC", 0.01)
        monkeypatch.setattr(VAPWrapper, "_STEP_SEC", 0.005)
        mock_cls, mock_load = _patches
        wrapper, model = _build_wrapper(mock_cls, mock_load)
        model.probs.return_value = _mock_model_probs()
        # Feed a frame larger than the buffer (480 > 160)
        frame = _make_pcm(480, amplitude=100)
        result = wrapper.feed_audio(frame)
        # Should not raise; buffer tail should have data
        assert wrapper._buffer[0, 0, -1].abs().item() > 0
        assert isinstance(result, VAPResult)


# ---------------------------------------------------------------------------
# TestVAPResultExtraction
# ---------------------------------------------------------------------------


class TestVAPResultExtraction:
    """Verify p_now/p_fut mapping and vad threshold comparison."""

    def test_p_now_p_fut_mapping(self, wrapper, model):
        frame = _make_pcm(480)
        model.probs.return_value = _mock_model_probs(p_now=0.42, p_fut=0.88, vad=0.1)

        for _ in range(4):
            result = wrapper.feed_audio(frame)

        assert result.p_now == pytest.approx(0.42, abs=1e-5)
        assert result.p_fut == pytest.approx(0.88, abs=1e-5)

    def test_vad_above_threshold_speaking(self, wrapper, model):
        frame = _make_pcm(480)
        model.probs.return_value = _mock_model_probs(vad=0.8)

        for _ in range(4):
            result = wrapper.feed_audio(frame)
        assert result.user_is_speaking is True

    def test_vad_below_threshold_not_speaking(self, wrapper, model):
        frame = _make_pcm(480)
        model.probs.return_value = _mock_model_probs(vad=0.3)

        for _ in range(4):
            result = wrapper.feed_audio(frame)
        assert result.user_is_speaking is False

    def test_vad_at_threshold_not_speaking(self, wrapper, model):
        frame = _make_pcm(480)
        model.probs.return_value = _mock_model_probs(vad=0.5)

        for _ in range(4):
            result = wrapper.feed_audio(frame)
        assert result.user_is_speaking is False  # > threshold, not >=
