"""Unit tests for voice_pipeline.adapters.wakeword (WakewordDetector).

All external dependencies (Silero VAD model, Google STT client) are mocked.
"""

from __future__ import annotations

import logging
import struct
from unittest.mock import MagicMock, patch

import pytest
import torch

from voice_pipeline.adapters.wakeword import WakewordDetector, _State
from voice_pipeline.settings import FRAME_SIZE_SAMPLES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Default audio config: 16kHz, mono, 16-bit, 30ms frames → 480 samples
_FRAME_SAMPLES = FRAME_SIZE_SAMPLES  # 480
_FRAME_BYTES = _FRAME_SAMPLES * 2  # 960 bytes
_VAD_CHUNK_BYTES = WakewordDetector._VAD_CHUNK_BYTES

_CLASS_VAR_KWARGS: dict[str, str] = {
    "keywords": "_KEYWORDS",
    "vad_threshold": "_VAD_THRESHOLD",
    "max_speech_duration_sec": "_MAX_SPEECH_DURATION_SEC",
    "pre_buffer_ms": "_PRE_BUFFER_MS",
    "speech_pad_ms": "_SPEECH_PAD_MS",
    "min_speech_duration_ms": "_MIN_SPEECH_DURATION_MS",
    "stt_timeout_sec": "_STT_TIMEOUT_SEC",
}


def _make_detector(monkeypatch: pytest.MonkeyPatch | None = None, **kwargs) -> WakewordDetector:
    """Build a WakewordDetector. Legacy tuning kwargs translated to class var monkeypatch."""
    for kw, cls_var in _CLASS_VAR_KWARGS.items():
        if kw in kwargs:
            if monkeypatch is None:
                raise AssertionError(f"monkeypatch fixture required for {kw} override")
            monkeypatch.setattr(WakewordDetector, cls_var, kwargs.pop(kw))
    return WakewordDetector(**kwargs)


def _silence_frame() -> bytes:
    """480-sample silence frame."""
    return b"\x00" * _FRAME_BYTES


def _tone_frame(amplitude: int = 10000) -> bytes:
    """480-sample square wave frame."""
    samples = [amplitude if (i % 8) < 4 else -amplitude for i in range(_FRAME_SAMPLES)]
    return struct.pack(f"<{_FRAME_SAMPLES}h", *samples)


def _make_stt_response(transcripts: list[str]) -> MagicMock:
    """Build a mock Google STT RecognizeResponse."""
    alternatives = []
    for t in transcripts:
        alt = MagicMock()
        alt.transcript = t
        alternatives.append(alt)

    result = MagicMock()
    result.alternatives = alternatives

    response = MagicMock()
    response.results = [result]
    return response


def _make_empty_stt_response() -> MagicMock:
    """Build a mock Google STT response with no results."""
    response = MagicMock()
    response.results = []
    return response


def _make_multi_result_response(result_transcripts: list[list[str]]) -> MagicMock:
    """Build a mock STT response with multiple results, each with multiple alternatives."""
    results = []
    for transcripts in result_transcripts:
        alternatives = []
        for t in transcripts:
            alt = MagicMock()
            alt.transcript = t
            alternatives.append(alt)
        result = MagicMock()
        result.alternatives = alternatives
        results.append(result)

    response = MagicMock()
    response.results = results
    return response


@pytest.fixture
def mock_vad_model():
    """Create a mock Silero VAD model."""
    model = MagicMock()
    model.reset_states = MagicMock()
    # Default: return low probability (no speech)
    model.return_value = torch.tensor(0.1)
    return model


@pytest.fixture
def mock_stt_client():
    """Create a mock Google STT client."""
    client = MagicMock()
    client.recognize.return_value = _make_empty_stt_response()
    return client


@pytest.fixture
def detector(mock_vad_model, mock_stt_client):
    """Create a WakewordDetector with mocked dependencies."""
    with (
        patch("voice_pipeline.adapters.wakeword.load_silero_vad", return_value=mock_vad_model),
        patch("voice_pipeline.adapters.wakeword.speech.SpeechClient", return_value=mock_stt_client),
    ):
        return _make_detector()


# ---------------------------------------------------------------------------
# TestVADRechunking
# ---------------------------------------------------------------------------


class TestVADRechunking:
    """Verify 480→512 sample rechunking and residual buffer handling."""

    def test_single_frame_no_vad_call(self, detector, mock_vad_model):
        """One 480-sample frame is insufficient for a 512-sample VAD chunk."""
        detector.feed_audio(_silence_frame())
        # 480 samples < 512: no VAD call yet
        mock_vad_model.assert_not_called()

    def test_two_frames_one_vad_call(self, detector, mock_vad_model):
        """Two 480-sample frames (960 samples) should yield one 512-sample VAD chunk,
        with 448 samples remaining."""
        detector.feed_audio(_silence_frame())
        detector.feed_audio(_silence_frame())
        # 960 samples → 1 chunk of 512, residual 448
        assert mock_vad_model.call_count == 1
        assert len(detector._vad_buffer) == (960 - 512) * 2  # 448 samples * 2 bytes

    def test_residual_carries_over(self, detector, mock_vad_model):
        """Residual bytes from previous calls carry over to produce new chunks."""
        # Feed 3 frames = 1440 samples → 2 chunks (1024), residual 416
        for _ in range(3):
            detector.feed_audio(_silence_frame())
        assert mock_vad_model.call_count == 2
        assert len(detector._vad_buffer) == (1440 - 1024) * 2

    def test_exact_multiple(self, detector, mock_vad_model):
        """512 samples exactly should trigger one VAD call."""
        frame = b"\x00" * (_VAD_CHUNK_BYTES)  # exactly 512 samples
        detector.feed_audio(frame)
        assert mock_vad_model.call_count == 1
        assert len(detector._vad_buffer) == 0

    def test_vad_receives_correct_tensor_shape(self, detector, mock_vad_model):
        """VAD model receives a float32 tensor of 512 samples."""
        # Feed enough for one chunk
        detector.feed_audio(b"\x00" * _VAD_CHUNK_BYTES)
        tensor_arg = mock_vad_model.call_args[0][0]
        assert tensor_arg.shape == (512,)
        assert tensor_arg.dtype == torch.float32


# ---------------------------------------------------------------------------
# TestSpeechDetection
# ---------------------------------------------------------------------------


class TestSpeechDetection:
    """VAD triggers speech state, silence triggers STT recognition."""

    def test_no_speech_no_recognition(self, detector, mock_vad_model, mock_stt_client):
        """All silence → no STT call."""
        mock_vad_model.return_value = torch.tensor(0.1)
        for _ in range(20):
            detector.feed_audio(_silence_frame())
        mock_stt_client.recognize.assert_not_called()

    def test_speech_then_silence_triggers_recognition(self, detector, mock_vad_model, mock_stt_client):
        """Speech frames followed by silence should trigger STT recognition."""
        mock_stt_client.recognize.return_value = _make_empty_stt_response()

        # We need enough frames to produce VAD chunks.
        # Each chunk is 512 samples. We need speech chunks then silence chunks.
        # speech_pad_ms=300, VAD_CHUNK_DURATION_MS=32 → need 300/32 ≈ 10 silence chunks
        # min_speech_duration_ms=100, 32ms per chunk → need ≥ 4 speech chunks

        call_count = 0

        def vad_side_effect(tensor, sr):
            nonlocal call_count
            call_count += 1
            # First 5 chunks: speech, rest: silence
            if call_count <= 5:
                return torch.tensor(0.9)
            return torch.tensor(0.1)

        mock_vad_model.side_effect = vad_side_effect

        # Feed enough frames to produce ~20 VAD chunks
        # 20 chunks * 512 samples = 10240 samples, at 480 samples/frame ≈ 22 frames
        for _ in range(25):
            detector.feed_audio(_tone_frame())

        mock_stt_client.recognize.assert_called_once()

    def test_detection_returns_true_on_keyword_match(self, detector, mock_vad_model, mock_stt_client):
        """feed_audio returns True when STT transcript contains the keyword."""
        mock_stt_client.recognize.return_value = _make_stt_response(["hey ray"])

        call_count = 0

        def vad_side_effect(tensor, sr):
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                return torch.tensor(0.9)
            return torch.tensor(0.1)

        mock_vad_model.side_effect = vad_side_effect

        detected = False
        for _ in range(25):
            if detector.feed_audio(_tone_frame()):
                detected = True
                break

        assert detected

    def test_detection_returns_false_on_no_keyword(self, detector, mock_vad_model, mock_stt_client):
        """feed_audio returns False when STT transcript doesn't contain keyword."""
        mock_stt_client.recognize.return_value = _make_stt_response(["hello world"])

        call_count = 0

        def vad_side_effect(tensor, sr):
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                return torch.tensor(0.9)
            return torch.tensor(0.1)

        mock_vad_model.side_effect = vad_side_effect

        detected = False
        for _ in range(25):
            if detector.feed_audio(_tone_frame()):
                detected = True
                break

        assert not detected


# ---------------------------------------------------------------------------
# TestKeywordMatching
# ---------------------------------------------------------------------------


class TestKeywordMatching:
    """Test keyword matching logic: case-insensitive, word boundary, multi-keyword."""

    def _trigger_recognition(self, detector, mock_vad_model, mock_stt_client, transcript):
        """Helper: run VAD cycle that triggers recognition with given transcript."""
        mock_stt_client.recognize.return_value = _make_stt_response([transcript])

        call_count = 0

        def vad_side_effect(tensor, sr):
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                return torch.tensor(0.9)
            return torch.tensor(0.1)

        mock_vad_model.side_effect = vad_side_effect

        return any(detector.feed_audio(_tone_frame()) for _ in range(25))

    def test_case_insensitive_match(self, detector, mock_vad_model, mock_stt_client):
        """Keyword matching is case-insensitive."""
        assert self._trigger_recognition(detector, mock_vad_model, mock_stt_client, "Hey RAY")

    def test_word_boundary_prevents_substring_match(self, mock_vad_model, mock_stt_client):
        """'array' should NOT match keyword 'ray' due to word boundary."""
        with (
            patch("voice_pipeline.adapters.wakeword.load_silero_vad", return_value=mock_vad_model),
            patch(
                "voice_pipeline.adapters.wakeword.speech.SpeechClient",
                return_value=mock_stt_client,
            ),
        ):
            det = _make_detector()

        assert not self._trigger_recognition(det, mock_vad_model, mock_stt_client, "array of items")

    def test_multiple_keywords(self, mock_vad_model, mock_stt_client, monkeypatch):
        """Multiple keywords: match on any one."""
        keywords = ("ray", "hello")
        with (
            patch("voice_pipeline.adapters.wakeword.load_silero_vad", return_value=mock_vad_model),
            patch(
                "voice_pipeline.adapters.wakeword.speech.SpeechClient",
                return_value=mock_stt_client,
            ),
        ):
            det = _make_detector(monkeypatch, keywords=keywords)

        assert self._trigger_recognition(det, mock_vad_model, mock_stt_client, "hey hello there")

    def test_checks_all_alternatives(self, mock_vad_model, mock_stt_client):
        """Keyword match checks all STT alternatives, not just top-1."""
        # First alternative has no keyword, second does
        response = _make_stt_response(["hello world", "hey ray"])
        mock_stt_client.recognize.return_value = response

        with (
            patch("voice_pipeline.adapters.wakeword.load_silero_vad", return_value=mock_vad_model),
            patch(
                "voice_pipeline.adapters.wakeword.speech.SpeechClient",
                return_value=mock_stt_client,
            ),
        ):
            det = _make_detector()

        call_count = 0

        def vad_side_effect(tensor, sr):
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                return torch.tensor(0.9)
            return torch.tensor(0.1)

        mock_vad_model.side_effect = vad_side_effect

        detected = False
        for _ in range(25):
            if det.feed_audio(_tone_frame()):
                detected = True
                break
        assert detected

    def test_checks_multiple_results(self, mock_vad_model, mock_stt_client):
        """Keyword match checks all result objects, not just the first."""
        response = _make_multi_result_response([["hello"], ["ray"]])
        mock_stt_client.recognize.return_value = response

        with (
            patch("voice_pipeline.adapters.wakeword.load_silero_vad", return_value=mock_vad_model),
            patch(
                "voice_pipeline.adapters.wakeword.speech.SpeechClient",
                return_value=mock_stt_client,
            ),
        ):
            det = _make_detector()

        call_count = 0

        def vad_side_effect(tensor, sr):
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                return torch.tensor(0.9)
            return torch.tensor(0.1)

        mock_vad_model.side_effect = vad_side_effect

        detected = False
        for _ in range(25):
            if det.feed_audio(_tone_frame()):
                detected = True
                break
        assert detected


# ---------------------------------------------------------------------------
# TestStateTransitions
# ---------------------------------------------------------------------------


class TestStateTransitions:
    """Test VAD state machine transitions."""

    def test_starts_in_idle(self, detector):
        """Detector starts in IDLE state."""
        assert detector._state is _State.IDLE

    def test_speech_transitions_to_speech(self, detector, mock_vad_model):
        """High VAD probability transitions from IDLE to SPEECH."""
        mock_vad_model.return_value = torch.tensor(0.9)
        # Feed enough for one VAD chunk
        detector.feed_audio(b"\x00" * _VAD_CHUNK_BYTES)
        assert detector._state is _State.SPEECH

    def test_silence_after_speech_transitions_to_trailing(self, detector, mock_vad_model):
        """Low VAD probability after speech transitions to TRAILING."""
        call_count = 0

        def vad_side_effect(tensor, sr):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return torch.tensor(0.9)  # speech
            return torch.tensor(0.1)  # silence

        mock_vad_model.side_effect = vad_side_effect

        # Two chunks: first speech, second silence → TRAILING
        detector.feed_audio(b"\x00" * _VAD_CHUNK_BYTES)  # → SPEECH
        detector.feed_audio(b"\x00" * _VAD_CHUNK_BYTES)  # → TRAILING
        assert detector._state is _State.TRAILING

    def test_trailing_back_to_speech(self, detector, mock_vad_model):
        """Speech resuming during TRAILING transitions back to SPEECH."""
        call_count = 0

        def vad_side_effect(tensor, sr):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return torch.tensor(0.9)  # → SPEECH
            if call_count == 2:
                return torch.tensor(0.1)  # → TRAILING
            return torch.tensor(0.9)  # → back to SPEECH

        mock_vad_model.side_effect = vad_side_effect

        detector.feed_audio(b"\x00" * _VAD_CHUNK_BYTES)
        detector.feed_audio(b"\x00" * _VAD_CHUNK_BYTES)
        assert detector._state is _State.TRAILING

        detector.feed_audio(b"\x00" * _VAD_CHUNK_BYTES)
        assert detector._state is _State.SPEECH
        assert detector._silence_chunks == 0

    def test_reset_returns_to_idle(self, detector, mock_vad_model):
        """After recognition, state returns to IDLE."""
        mock_vad_model.return_value = torch.tensor(0.9)
        detector.feed_audio(b"\x00" * _VAD_CHUNK_BYTES)
        assert detector._state is _State.SPEECH

        detector._reset()
        assert detector._state is _State.IDLE
        assert len(detector._speech_buffer) == 0
        mock_vad_model.reset_states.assert_called()

    def test_vad_buffer_preserved_across_reset(self, detector, mock_vad_model):
        """Residual VAD buffer is NOT cleared on reset (carries over)."""
        mock_vad_model.return_value = torch.tensor(0.1)
        # Feed partial chunk
        detector.feed_audio(b"\x00" * 100)
        assert len(detector._vad_buffer) == 100

        detector._reset()
        # VAD buffer should be preserved
        assert len(detector._vad_buffer) == 100


# ---------------------------------------------------------------------------
# TestSafetyLimits
# ---------------------------------------------------------------------------


class TestSafetyLimits:
    """Test safety limits: max duration, min duration."""

    def test_max_speech_duration_forces_recognition(self, mock_vad_model, mock_stt_client, monkeypatch):
        """Speech exceeding max_speech_duration_sec forces recognition."""
        mock_stt_client.recognize.return_value = _make_empty_stt_response()

        with (
            patch("voice_pipeline.adapters.wakeword.load_silero_vad", return_value=mock_vad_model),
            patch(
                "voice_pipeline.adapters.wakeword.speech.SpeechClient",
                return_value=mock_stt_client,
            ),
        ):
            det = _make_detector(monkeypatch, max_speech_duration_sec=0.1)  # 100ms

        # All VAD chunks return high probability (continuous speech)
        mock_vad_model.return_value = torch.tensor(0.9)

        # 0.1s at 16kHz = 1600 samples ≈ 3.1 VAD chunks (512 each)
        # 4th chunk → 128ms > 100ms → forces recognition
        for _ in range(4):
            det.feed_audio(b"\x00" * _VAD_CHUNK_BYTES)

        mock_stt_client.recognize.assert_called_once()
        assert det._state is _State.IDLE  # reset after forced recognition

    def test_short_speech_skips_recognition(self, mock_vad_model, mock_stt_client, monkeypatch):
        """Speech shorter than min_speech_duration_ms (100ms) is ignored."""
        # speech_pad_ms=32: just 1 silence chunk triggers recognition check
        # one VAD chunk is 32ms → only 1 chunk of speech (< 100ms default min)
        mock_stt_client.recognize.return_value = _make_empty_stt_response()

        with (
            patch("voice_pipeline.adapters.wakeword.load_silero_vad", return_value=mock_vad_model),
            patch(
                "voice_pipeline.adapters.wakeword.speech.SpeechClient",
                return_value=mock_stt_client,
            ),
        ):
            det = _make_detector(monkeypatch, speech_pad_ms=32)

        call_count = 0

        def vad_side_effect(tensor, sr):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return torch.tensor(0.9)  # 1 speech chunk (32ms < 100ms min)
            return torch.tensor(0.1)  # silence

        mock_vad_model.side_effect = vad_side_effect

        for _ in range(10):
            det.feed_audio(b"\x00" * _VAD_CHUNK_BYTES)

        # Too short → no STT call
        mock_stt_client.recognize.assert_not_called()


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test error handling: init errors raise, runtime errors fail closed."""

    def test_vad_load_failure_raises_wakeword_error(self):
        """Failed VAD model load raises RuntimeError."""
        with (
            patch(
                "voice_pipeline.adapters.wakeword.load_silero_vad",
                side_effect=RuntimeError("model not found"),
            ),
            pytest.raises(RuntimeError, match="Failed to load Silero VAD model"),
        ):
            _make_detector()

    def test_stt_client_creation_failure_raises_wakeword_error(self, mock_vad_model):
        """Failed STT client creation raises RuntimeError."""
        with (
            patch("voice_pipeline.adapters.wakeword.load_silero_vad", return_value=mock_vad_model),
            patch(
                "voice_pipeline.adapters.wakeword.speech.SpeechClient",
                side_effect=RuntimeError("credentials not found"),
            ),
            pytest.raises(RuntimeError, match="Failed to create Google STT client"),
        ):
            _make_detector()

    def test_stt_error_returns_false(self, detector, mock_vad_model, mock_stt_client, caplog):
        """STT recognition error → log warning, return False (fail closed)."""
        mock_stt_client.recognize.side_effect = RuntimeError("network error")

        call_count = 0

        def vad_side_effect(tensor, sr):
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                return torch.tensor(0.9)
            return torch.tensor(0.1)

        mock_vad_model.side_effect = vad_side_effect

        detected = False
        with caplog.at_level(logging.WARNING, logger="voice_pipeline.wakeword"):
            for _ in range(25):
                if detector.feed_audio(_tone_frame()):
                    detected = True
                    break

        assert not detected
        assert "Wakeword STT recognition failed" in caplog.text
        # State should be reset after error
        assert detector._state is _State.IDLE

    def test_recovery_after_stt_error(self, detector, mock_vad_model, mock_stt_client):
        """Detector can detect wakeword after a previous STT error."""
        # First cycle: STT error
        mock_stt_client.recognize.side_effect = RuntimeError("transient error")

        call_count = 0

        def first_cycle(tensor, sr):
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                return torch.tensor(0.9)
            return torch.tensor(0.1)

        mock_vad_model.side_effect = first_cycle

        for _ in range(25):
            detector.feed_audio(_tone_frame())

        assert detector._state is _State.IDLE  # reset after error

        # Second cycle: STT succeeds with keyword
        mock_stt_client.recognize.side_effect = None
        mock_stt_client.recognize.return_value = _make_stt_response(["hey ray"])

        call_count = 0

        def second_cycle(tensor, sr):
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                return torch.tensor(0.9)
            return torch.tensor(0.1)

        mock_vad_model.side_effect = second_cycle

        detected = False
        for _ in range(25):
            if detector.feed_audio(_tone_frame()):
                detected = True
                break

        assert detected


# ---------------------------------------------------------------------------
# TestPhraseHints
# ---------------------------------------------------------------------------


class TestPhraseHints:
    """Verify STT configuration includes phrase hints."""

    def test_recognition_config_includes_speech_context(self, detector):
        """Recognition config should include SpeechContext with keyword phrases."""
        config = detector._recognition_config
        assert len(config.speech_contexts) == 1
        assert "ray" in config.speech_contexts[0].phrases

    def test_recognition_config_max_alternatives(self, detector):
        """Recognition config should request multiple alternatives."""
        assert detector._recognition_config.max_alternatives == 5


# ---------------------------------------------------------------------------
# TestVADFailClosed
# ---------------------------------------------------------------------------


class TestVADFailClosed:
    """Test that VAD inference/reset errors fail closed (return False)."""

    def test_vad_inference_error_returns_false(self, detector, mock_vad_model, caplog):
        """VAD inference error → log warning, reset, return False."""
        mock_vad_model.side_effect = RuntimeError("model error")

        with caplog.at_level(logging.WARNING, logger="voice_pipeline.wakeword"):
            result = detector.feed_audio(b"\x00" * _VAD_CHUNK_BYTES)

        assert not result
        assert "VAD inference failed" in caplog.text
        assert detector._state is _State.IDLE

    def test_vad_reset_states_error_suppressed(self, detector, mock_vad_model, caplog):
        """Error in reset_states is logged but doesn't propagate."""
        mock_vad_model.reset_states.side_effect = RuntimeError("reset error")

        # Put detector in SPEECH state, then trigger reset via _reset()
        mock_vad_model.return_value = torch.tensor(0.9)
        detector.feed_audio(b"\x00" * _VAD_CHUNK_BYTES)
        assert detector._state is _State.SPEECH

        with caplog.at_level(logging.WARNING, logger="voice_pipeline.wakeword"):
            detector._reset()

        assert detector._state is _State.IDLE
        assert "VAD reset_states failed" in caplog.text

    def test_recovery_after_vad_inference_error(self, detector, mock_vad_model):
        """Detector recovers after a transient VAD inference error."""
        call_count = 0

        def vad_side_effect(tensor, sr):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")
            return torch.tensor(0.1)

        mock_vad_model.side_effect = vad_side_effect

        # First call errors → fail closed
        assert not detector.feed_audio(b"\x00" * _VAD_CHUNK_BYTES)
        # Second call succeeds
        assert not detector.feed_audio(b"\x00" * _VAD_CHUNK_BYTES)
        assert call_count == 2


# ---------------------------------------------------------------------------
# TestSpeechPadEdge
# ---------------------------------------------------------------------------


class TestSpeechPadEdge:
    """Test speech_pad_ms <= chunk_duration edge case."""

    def test_speech_pad_equal_to_chunk_duration(self, mock_vad_model, mock_stt_client, monkeypatch):
        """speech_pad_ms == 32 (one chunk) triggers on first silence chunk."""
        mock_stt_client.recognize.return_value = _make_stt_response(["ray"])

        with (
            patch("voice_pipeline.adapters.wakeword.load_silero_vad", return_value=mock_vad_model),
            patch(
                "voice_pipeline.adapters.wakeword.speech.SpeechClient",
                return_value=mock_stt_client,
            ),
        ):
            det = _make_detector(monkeypatch, speech_pad_ms=32, min_speech_duration_ms=32)

        call_count = 0

        def vad_side_effect(tensor, sr):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return torch.tensor(0.9)  # speech
            return torch.tensor(0.1)  # silence

        mock_vad_model.side_effect = vad_side_effect

        # 2 speech chunks + 1 silence chunk → should trigger recognition immediately
        detected = False
        for _ in range(5):
            if det.feed_audio(b"\x00" * _VAD_CHUNK_BYTES):
                detected = True
                break

        mock_stt_client.recognize.assert_called_once()
        assert detected


# ---------------------------------------------------------------------------
# TestClose
# ---------------------------------------------------------------------------


class TestClose:
    """Test resource cleanup via close()."""

    def test_close_closes_transport(self, detector, mock_stt_client):
        """close() calls transport.close() on the STT client."""
        detector.close()
        mock_stt_client.transport.close.assert_called_once()

    def test_close_idempotent(self, detector, mock_stt_client):
        """close() can be called multiple times safely."""
        detector.close()
        detector.close()
        mock_stt_client.transport.close.assert_called_once()

    def test_close_suppresses_transport_error(self, detector, mock_stt_client, caplog):
        """close() suppresses transport.close() errors."""
        mock_stt_client.transport.close.side_effect = RuntimeError("close error")
        with caplog.at_level(logging.DEBUG, logger="voice_pipeline.wakeword"):
            detector.close()
        assert detector._stt_client is None


# ---------------------------------------------------------------------------
# TestReset
# ---------------------------------------------------------------------------


class TestReset:
    """Public reset() — called on every transition into SLEEP."""

    def test_reset_clears_state_and_vad_model(self, detector, mock_vad_model):
        """reset() clears speech state, buffers, and the VAD model state."""
        # Build up state: speech in progress + residual rechunk bytes
        mock_vad_model.return_value = torch.tensor(0.9)
        detector.feed_audio(_tone_frame())
        detector.feed_audio(_tone_frame())  # 1920B → 1 chunk processed, 896B residual
        assert detector._state is _State.SPEECH
        assert len(detector._vad_buffer) > 0

        mock_vad_model.reset_states.reset_mock()
        detector.reset()

        assert detector._state is _State.IDLE
        assert len(detector._vad_buffer) == 0
        assert len(detector._speech_buffer) == 0
        assert len(detector._pre_buffer) == 0
        mock_vad_model.reset_states.assert_called_once()

    def test_reset_suppresses_vad_model_error(self, detector, mock_vad_model):
        """reset() does not raise when the VAD model reset fails."""
        mock_vad_model.reset_states.side_effect = RuntimeError("boom")
        detector.reset()
        assert detector._state is _State.IDLE

    def test_detection_works_after_reset(self, detector, mock_vad_model, mock_stt_client, monkeypatch):
        """A full detect cycle still works after reset()."""
        monkeypatch.setattr(WakewordDetector, "_MIN_SPEECH_DURATION_MS", 0)
        monkeypatch.setattr(WakewordDetector, "_SPEECH_PAD_MS", 0)
        detector.reset()

        mock_stt_client.recognize.return_value = _make_stt_response(["hey ray"])
        mock_vad_model.return_value = torch.tensor(0.9)
        for _ in range(4):
            if detector.feed_audio(_tone_frame()):
                break
        mock_vad_model.return_value = torch.tensor(0.1)
        detected = any(detector.feed_audio(_silence_frame()) for _ in range(4))
        assert detected
