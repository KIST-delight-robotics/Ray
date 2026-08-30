"""Unit tests for OpenAITTS (mocked OpenAI client)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import openai
import pytest

from voice_pipeline.adapters.tts_openai import OpenAITTS
from voice_pipeline.tests.adapters.conftest import create_mock_tts_client

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Legacy kwargs → class var monkeypatch 번역 레이어.
_CLASS_VAR_MAP = {
    "voice": "_VOICE",
    "model": "_MODEL",
    "speed": "_SPEED",
    "instructions": "_INSTRUCTIONS",
}


def _build_tts(
    mock_client: MagicMock,
    monkeypatch: pytest.MonkeyPatch | None = None,
    **kwargs,
) -> OpenAITTS:
    """Build an OpenAITTS with a mock client injected.

    Legacy kwargs (voice/model/speed/instructions) are translated to class var
    monkeypatch — caller must pass ``monkeypatch`` fixture when providing any.
    """
    if kwargs and monkeypatch is None:
        raise TypeError("monkeypatch fixture required when overrides provided")
    for key, value in kwargs.items():
        if key not in _CLASS_VAR_MAP:
            raise TypeError(f"Unknown override: {key}")
        monkeypatch.setattr(OpenAITTS, _CLASS_VAR_MAP[key], value)
    with patch("voice_pipeline.adapters.tts_openai.openai.OpenAI", return_value=mock_client):
        return OpenAITTS()


# ---------------------------------------------------------------------------
# TestSynthesize
# ---------------------------------------------------------------------------


class TestSynthesize:
    def test_yields_pcm_chunks(self) -> None:
        chunks = [b"\x00\x01" * 100, b"\x02\x03" * 100]
        client = create_mock_tts_client(chunks)
        tts = _build_tts(client)

        stream = tts.synthesize("Hello world")
        collected = list(stream)

        assert collected == chunks

    def test_collects_audio_correctly(self) -> None:
        chunks = [b"\x01\x02", b"\x03\x04", b"\x05\x06"]
        client = create_mock_tts_client(chunks)
        tts = _build_tts(client)

        stream = tts.synthesize("Hello world")
        list(stream)

        assert stream.audio == b"\x01\x02\x03\x04\x05\x06"


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_empty_text_raises(self) -> None:
        client = create_mock_tts_client()
        tts = _build_tts(client)

        with pytest.raises(RuntimeError, match="empty"):
            tts.synthesize("")

    def test_whitespace_only_raises(self) -> None:
        client = create_mock_tts_client()
        tts = _build_tts(client)

        with pytest.raises(RuntimeError, match="empty"):
            tts.synthesize("   \n\t  ")

    def test_too_long_text_raises(self) -> None:
        client = create_mock_tts_client()
        tts = _build_tts(client)

        with pytest.raises(RuntimeError, match="4096"):
            tts.synthesize("a" * 4097)

    def test_invalid_speed_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = create_mock_tts_client()
        tts = _build_tts(client, monkeypatch, speed=5.0)

        with pytest.raises(RuntimeError, match="Speed"):
            tts.synthesize("Hello")

    def test_speed_too_low_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = create_mock_tts_client()
        tts = _build_tts(client, monkeypatch, speed=0.1)

        with pytest.raises(RuntimeError, match="Speed"):
            tts.synthesize("Hello")


# ---------------------------------------------------------------------------
# TestModelSpecificParams
# ---------------------------------------------------------------------------


class TestModelSpecificParams:
    def test_tts1_no_instructions_sent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """tts-1 model should not send instructions even if configured."""
        client = create_mock_tts_client([b"\x00"])
        tts = _build_tts(client, monkeypatch, model="tts-1", instructions="Be cheerful")

        stream = tts.synthesize("Hello")
        list(stream)

        call_kwargs = client.audio.speech.with_streaming_response.create.call_args[1]
        assert "instructions" not in call_kwargs

    def test_gpt4o_mini_tts_instructions_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = create_mock_tts_client([b"\x00"])
        tts = _build_tts(client, monkeypatch, model="gpt-4o-mini-tts", instructions="Be cheerful")

        stream = tts.synthesize("Hello")
        list(stream)

        call_kwargs = client.audio.speech.with_streaming_response.create.call_args[1]
        assert call_kwargs["instructions"] == "Be cheerful"

    def test_unsupported_model_logs_warning_for_instructions(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = create_mock_tts_client([b"\x00"])
        tts = _build_tts(client, monkeypatch, model="tts-1-hd", instructions="Be sad")

        with caplog.at_level("WARNING", logger="voice_pipeline.tts"):
            stream = tts.synthesize("Hello")
            list(stream)

        assert "instructions ignored" in caplog.text

    def test_no_instructions_no_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No warning when instructions is empty string."""
        client = create_mock_tts_client([b"\x00"])
        tts = _build_tts(client, monkeypatch, model="tts-1", instructions="")

        with caplog.at_level("WARNING", logger="voice_pipeline.tts"):
            stream = tts.synthesize("Hello")
            list(stream)

        assert "instructions ignored" not in caplog.text


# ---------------------------------------------------------------------------
# TestStreamResult
# ---------------------------------------------------------------------------


class TestStreamResult:
    def test_audio_available_after_iteration(self) -> None:
        client = create_mock_tts_client([b"\xaa\xbb", b"\xcc\xdd"])
        tts = _build_tts(client)

        stream = tts.synthesize("Hello")
        list(stream)

        assert stream.audio == b"\xaa\xbb\xcc\xdd"

    def test_timestamps_empty_for_openai(self) -> None:
        """OpenAI TTS does not support word-level timestamps."""
        client = create_mock_tts_client([b"\x00"])
        tts = _build_tts(client)

        stream = tts.synthesize("Hello")
        list(stream)

        assert stream.timestamps == ()

    def test_result_property(self) -> None:
        client = create_mock_tts_client([b"\x01", b"\x02"])
        tts = _build_tts(client)

        stream = tts.synthesize("Hello")
        list(stream)

        result = stream.result
        assert result.audio == b"\x01\x02"
        assert result.timestamps == ()


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_api_error_wrapped(self) -> None:
        client = create_mock_tts_client(
            side_effect=openai.APIConnectionError(request=MagicMock()),
        )
        tts = _build_tts(client)

        with pytest.raises(RuntimeError):
            tts.synthesize("Hello")

    def test_auth_error_wrapped(self) -> None:
        client = create_mock_tts_client(
            side_effect=openai.AuthenticationError(
                message="bad key",
                response=MagicMock(status_code=401, headers={}),
                body=None,
            ),
        )
        tts = _build_tts(client)

        with pytest.raises(RuntimeError):
            tts.synthesize("Hello")

    def test_rate_limit_error_wrapped(self) -> None:
        client = create_mock_tts_client(
            side_effect=openai.RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429, headers={}),
                body=None,
            ),
        )
        tts = _build_tts(client)

        with pytest.raises(RuntimeError):
            tts.synthesize("Hello")

    def test_streaming_error_wrapped(self) -> None:
        """Error during stream iteration is wrapped in RuntimeError."""
        client = create_mock_tts_client(
            streaming_error=openai.APIConnectionError(request=MagicMock()),
        )
        tts = _build_tts(client)

        stream = tts.synthesize("Hello")
        with pytest.raises(RuntimeError):
            list(stream)


# ---------------------------------------------------------------------------
# TestStreamCleanup
# ---------------------------------------------------------------------------


class TestStreamCleanup:
    def test_cm_exited_after_full_iteration(self) -> None:
        client = create_mock_tts_client([b"\x01", b"\x02"])
        tts = _build_tts(client)
        mock_cm = client.audio.speech.with_streaming_response.create.return_value

        stream = tts.synthesize("Hello")
        list(stream)

        mock_cm.__exit__.assert_called()

    def test_cm_exited_on_partial_iteration(self) -> None:
        """Close after consuming one chunk — CM must still be exited."""
        client = create_mock_tts_client([b"\x01", b"\x02", b"\x03"])
        tts = _build_tts(client)
        mock_cm = client.audio.speech.with_streaming_response.create.return_value

        stream = tts.synthesize("Hello")
        next(stream)
        stream.close()

        mock_cm.__exit__.assert_called()

    def test_cm_exited_on_error(self) -> None:
        client = create_mock_tts_client(
            streaming_error=openai.APIConnectionError(request=MagicMock()),
        )
        tts = _build_tts(client)
        mock_cm = client.audio.speech.with_streaming_response.create.return_value

        stream = tts.synthesize("Hello")
        with pytest.raises(RuntimeError):
            list(stream)

        mock_cm.__exit__.assert_called()

    def test_close_before_first_next(self) -> None:
        """Close before consuming any chunk — close_fn must still fire."""
        client = create_mock_tts_client([b"\x01", b"\x02"])
        tts = _build_tts(client)
        mock_cm = client.audio.speech.with_streaming_response.create.return_value

        stream = tts.synthesize("Hello")
        stream.close()

        mock_cm.__exit__.assert_called()

    def test_cm_exited_exactly_once_after_full_iteration(self) -> None:
        """__exit__ must be called exactly once, not double-exited."""
        client = create_mock_tts_client([b"\x01", b"\x02"])
        tts = _build_tts(client)
        mock_cm = client.audio.speech.with_streaming_response.create.return_value

        stream = tts.synthesize("Hello")
        list(stream)
        stream.close()  # extra close after exhaustion

        # __enter__ called once by synthesize(), __exit__ exactly once
        mock_cm.__enter__.assert_called_once()
        mock_cm.__exit__.assert_called_once()

    def test_cm_exited_exactly_once_on_partial_close(self) -> None:
        """Partial iteration + close: __exit__ exactly once."""
        client = create_mock_tts_client([b"\x01", b"\x02", b"\x03"])
        tts = _build_tts(client)
        mock_cm = client.audio.speech.with_streaming_response.create.return_value

        stream = tts.synthesize("Hello")
        next(stream)
        stream.close()

        mock_cm.__exit__.assert_called_once()

    def test_close_after_exhaustion_is_idempotent(self) -> None:
        client = create_mock_tts_client([b"\x01"])
        tts = _build_tts(client)

        stream = tts.synthesize("Hello")
        list(stream)
        stream.close()
        stream.close()  # double close — should be a no-op


# ---------------------------------------------------------------------------
# TestEnterError
# ---------------------------------------------------------------------------


class TestEnterError:
    def test_enter_error_wrapped_as_tts_error(self) -> None:
        """__enter__ failure is wrapped as RuntimeError."""
        client = create_mock_tts_client([b"\x00"])
        mock_cm = client.audio.speech.with_streaming_response.create.return_value
        mock_cm.__enter__ = MagicMock(side_effect=RuntimeError("connection failed"))
        tts = _build_tts(client)

        with pytest.raises(RuntimeError, match="connection failed"):
            tts.synthesize("Hello")


# ---------------------------------------------------------------------------
# TestMidStreamError
# ---------------------------------------------------------------------------


class TestMidStreamError:
    def test_error_after_partial_chunks(self) -> None:
        """Error mid-stream after some chunks: collected audio is partial, RuntimeError raised."""
        client = create_mock_tts_client()
        mock_cm = client.audio.speech.with_streaming_response.create.return_value
        mock_response = mock_cm.__enter__.return_value

        # Yield one chunk then raise
        def _mixed_iter(**kwargs):  # noqa: ARG001
            yield b"\x01\x02"
            raise openai.APIConnectionError(request=MagicMock())

        mock_response.iter_bytes = _mixed_iter
        tts = _build_tts(client)

        stream = tts.synthesize("Hello")
        first = next(stream)
        assert first == b"\x01\x02"

        with pytest.raises(RuntimeError):
            next(stream)

        mock_cm.__exit__.assert_called_once()


# ---------------------------------------------------------------------------
# TestClientConfig
# ---------------------------------------------------------------------------


class TestClientConfig:
    def test_model_voice_speed_passed_to_create(self) -> None:
        client = create_mock_tts_client([b"\x00"])
        tts = _build_tts(client)

        stream = tts.synthesize("Hello")
        list(stream)

        call_kwargs = client.audio.speech.with_streaming_response.create.call_args[1]
        assert call_kwargs["model"] == "tts-1"
        assert call_kwargs["voice"] == "ash"
        assert call_kwargs["speed"] == 1.0
        assert call_kwargs["response_format"] == "pcm"

    def test_custom_config_values_propagated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = create_mock_tts_client([b"\x00"])
        tts = _build_tts(client, monkeypatch, model="tts-1-hd", voice="nova", speed=1.5)

        stream = tts.synthesize("Hello")
        list(stream)

        call_kwargs = client.audio.speech.with_streaming_response.create.call_args[1]
        assert call_kwargs["model"] == "tts-1-hd"
        assert call_kwargs["voice"] == "nova"
        assert call_kwargs["speed"] == 1.5
