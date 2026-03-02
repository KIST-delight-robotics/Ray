"""Unit tests for OpenAITTS (mocked OpenAI client)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import openai
import pytest

from voice_pipeline.core.config import TTSConfig
from voice_pipeline.tests.tts.conftest import create_mock_client
from voice_pipeline.tts.exceptions import TTSError
from voice_pipeline.tts.tts import OpenAITTS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_tts(mock_client: MagicMock, config: TTSConfig | None = None) -> OpenAITTS:
    """Build an OpenAITTS with a mock client injected."""
    cfg = config or TTSConfig()
    with patch("voice_pipeline.tts.tts.openai.OpenAI", return_value=mock_client):
        return OpenAITTS(cfg)


# ---------------------------------------------------------------------------
# TestSynthesize
# ---------------------------------------------------------------------------


class TestSynthesize:
    def test_yields_pcm_chunks(self, tts_config: TTSConfig) -> None:
        chunks = [b"\x00\x01" * 100, b"\x02\x03" * 100]
        client = create_mock_client(chunks)
        tts = _build_tts(client, tts_config)

        stream = tts.synthesize("Hello world")
        collected = list(stream)

        assert collected == chunks

    def test_collects_audio_correctly(self, tts_config: TTSConfig) -> None:
        chunks = [b"\x01\x02", b"\x03\x04", b"\x05\x06"]
        client = create_mock_client(chunks)
        tts = _build_tts(client, tts_config)

        stream = tts.synthesize("Hello world")
        list(stream)

        assert stream.audio == b"\x01\x02\x03\x04\x05\x06"


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_empty_text_raises(self, tts_config: TTSConfig) -> None:
        client = create_mock_client()
        tts = _build_tts(client, tts_config)

        with pytest.raises(TTSError, match="empty"):
            tts.synthesize("")

    def test_whitespace_only_raises(self, tts_config: TTSConfig) -> None:
        client = create_mock_client()
        tts = _build_tts(client, tts_config)

        with pytest.raises(TTSError, match="empty"):
            tts.synthesize("   \n\t  ")

    def test_too_long_text_raises(self, tts_config: TTSConfig) -> None:
        client = create_mock_client()
        tts = _build_tts(client, tts_config)

        with pytest.raises(TTSError, match="4096"):
            tts.synthesize("a" * 4097)

    def test_invalid_speed_raises(self) -> None:
        config = TTSConfig(speed=5.0)
        client = create_mock_client()
        tts = _build_tts(client, config)

        with pytest.raises(TTSError, match="Speed"):
            tts.synthesize("Hello")

    def test_speed_too_low_raises(self) -> None:
        config = TTSConfig(speed=0.1)
        client = create_mock_client()
        tts = _build_tts(client, config)

        with pytest.raises(TTSError, match="Speed"):
            tts.synthesize("Hello")


# ---------------------------------------------------------------------------
# TestModelSpecificParams
# ---------------------------------------------------------------------------


class TestModelSpecificParams:
    def test_tts1_no_instructions_sent(self, tts_config: TTSConfig) -> None:
        """tts-1 model should not send instructions even if configured."""
        config = TTSConfig(model="tts-1", instructions="Be cheerful")
        client = create_mock_client([b"\x00"])
        tts = _build_tts(client, config)

        stream = tts.synthesize("Hello")
        list(stream)

        call_kwargs = client.audio.speech.with_streaming_response.create.call_args[1]
        assert "instructions" not in call_kwargs

    def test_gpt4o_mini_tts_instructions_passed(self) -> None:
        config = TTSConfig(model="gpt-4o-mini-tts", instructions="Be cheerful")
        client = create_mock_client([b"\x00"])
        tts = _build_tts(client, config)

        stream = tts.synthesize("Hello")
        list(stream)

        call_kwargs = client.audio.speech.with_streaming_response.create.call_args[1]
        assert call_kwargs["instructions"] == "Be cheerful"

    def test_unsupported_model_logs_warning_for_instructions(
        self, tts_config: TTSConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        config = TTSConfig(model="tts-1-hd", instructions="Be sad")
        client = create_mock_client([b"\x00"])
        tts = _build_tts(client, config)

        with caplog.at_level("WARNING", logger="voice_pipeline.tts"):
            stream = tts.synthesize("Hello")
            list(stream)

        assert "instructions ignored" in caplog.text

    def test_no_instructions_no_warning(
        self, tts_config: TTSConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No warning when instructions is empty string."""
        config = TTSConfig(model="tts-1", instructions="")
        client = create_mock_client([b"\x00"])
        tts = _build_tts(client, config)

        with caplog.at_level("WARNING", logger="voice_pipeline.tts"):
            stream = tts.synthesize("Hello")
            list(stream)

        assert "instructions ignored" not in caplog.text


# ---------------------------------------------------------------------------
# TestStreamResult
# ---------------------------------------------------------------------------


class TestStreamResult:
    def test_audio_available_after_iteration(self, tts_config: TTSConfig) -> None:
        client = create_mock_client([b"\xaa\xbb", b"\xcc\xdd"])
        tts = _build_tts(client, tts_config)

        stream = tts.synthesize("Hello")
        list(stream)

        assert stream.audio == b"\xaa\xbb\xcc\xdd"

    def test_timestamps_empty_for_openai(self, tts_config: TTSConfig) -> None:
        """OpenAI TTS does not support word-level timestamps."""
        client = create_mock_client([b"\x00"])
        tts = _build_tts(client, tts_config)

        stream = tts.synthesize("Hello")
        list(stream)

        assert stream.timestamps == ()

    def test_result_property(self, tts_config: TTSConfig) -> None:
        client = create_mock_client([b"\x01", b"\x02"])
        tts = _build_tts(client, tts_config)

        stream = tts.synthesize("Hello")
        list(stream)

        result = stream.result
        assert result.audio == b"\x01\x02"
        assert result.timestamps == ()


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_api_error_wrapped(self, tts_config: TTSConfig) -> None:
        client = create_mock_client(
            side_effect=openai.APIConnectionError(request=MagicMock()),
        )
        tts = _build_tts(client, tts_config)

        with pytest.raises(TTSError):
            tts.synthesize("Hello")

    def test_auth_error_wrapped(self, tts_config: TTSConfig) -> None:
        client = create_mock_client(
            side_effect=openai.AuthenticationError(
                message="bad key",
                response=MagicMock(status_code=401, headers={}),
                body=None,
            ),
        )
        tts = _build_tts(client, tts_config)

        with pytest.raises(TTSError):
            tts.synthesize("Hello")

    def test_rate_limit_error_wrapped(self, tts_config: TTSConfig) -> None:
        client = create_mock_client(
            side_effect=openai.RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429, headers={}),
                body=None,
            ),
        )
        tts = _build_tts(client, tts_config)

        with pytest.raises(TTSError):
            tts.synthesize("Hello")

    def test_streaming_error_wrapped(self, tts_config: TTSConfig) -> None:
        """Error during stream iteration is wrapped in TTSError."""
        client = create_mock_client(
            streaming_error=openai.APIConnectionError(request=MagicMock()),
        )
        tts = _build_tts(client, tts_config)

        stream = tts.synthesize("Hello")
        with pytest.raises(TTSError):
            list(stream)


# ---------------------------------------------------------------------------
# TestStreamCleanup
# ---------------------------------------------------------------------------


class TestStreamCleanup:
    def test_cm_exited_after_full_iteration(self, tts_config: TTSConfig) -> None:
        client = create_mock_client([b"\x01", b"\x02"])
        tts = _build_tts(client, tts_config)
        mock_cm = client.audio.speech.with_streaming_response.create.return_value

        stream = tts.synthesize("Hello")
        list(stream)

        mock_cm.__exit__.assert_called()

    def test_cm_exited_on_partial_iteration(self, tts_config: TTSConfig) -> None:
        """Close after consuming one chunk — CM must still be exited."""
        client = create_mock_client([b"\x01", b"\x02", b"\x03"])
        tts = _build_tts(client, tts_config)
        mock_cm = client.audio.speech.with_streaming_response.create.return_value

        stream = tts.synthesize("Hello")
        next(stream)
        stream.close()

        mock_cm.__exit__.assert_called()

    def test_cm_exited_on_error(self, tts_config: TTSConfig) -> None:
        client = create_mock_client(
            streaming_error=openai.APIConnectionError(request=MagicMock()),
        )
        tts = _build_tts(client, tts_config)
        mock_cm = client.audio.speech.with_streaming_response.create.return_value

        stream = tts.synthesize("Hello")
        with pytest.raises(TTSError):
            list(stream)

        mock_cm.__exit__.assert_called()

    def test_close_before_first_next(self, tts_config: TTSConfig) -> None:
        """Close before consuming any chunk — close_fn must still fire."""
        client = create_mock_client([b"\x01", b"\x02"])
        tts = _build_tts(client, tts_config)
        mock_cm = client.audio.speech.with_streaming_response.create.return_value

        stream = tts.synthesize("Hello")
        stream.close()

        mock_cm.__exit__.assert_called()


# ---------------------------------------------------------------------------
# TestClientConfig
# ---------------------------------------------------------------------------


class TestClientConfig:
    def test_max_retries_passed_to_client(self) -> None:
        config = TTSConfig(max_retries=5)
        with patch("voice_pipeline.tts.tts.openai.OpenAI") as mock_cls:
            OpenAITTS(config)
            mock_cls.assert_called_once_with(max_retries=5, timeout=30.0)

    def test_timeout_passed_to_client(self) -> None:
        config = TTSConfig(timeout_sec=60.0)
        with patch("voice_pipeline.tts.tts.openai.OpenAI") as mock_cls:
            OpenAITTS(config)
            mock_cls.assert_called_once_with(max_retries=2, timeout=60.0)

    def test_model_voice_speed_passed_to_create(self, tts_config: TTSConfig) -> None:
        client = create_mock_client([b"\x00"])
        tts = _build_tts(client, tts_config)

        stream = tts.synthesize("Hello")
        list(stream)

        call_kwargs = client.audio.speech.with_streaming_response.create.call_args[1]
        assert call_kwargs["model"] == "tts-1"
        assert call_kwargs["voice"] == "alloy"
        assert call_kwargs["speed"] == 1.0
        assert call_kwargs["response_format"] == "pcm"

    def test_custom_config_values_propagated(self) -> None:
        config = TTSConfig(model="tts-1-hd", voice="nova", speed=1.5)
        client = create_mock_client([b"\x00"])
        tts = _build_tts(client, config)

        stream = tts.synthesize("Hello")
        list(stream)

        call_kwargs = client.audio.speech.with_streaming_response.create.call_args[1]
        assert call_kwargs["model"] == "tts-1-hd"
        assert call_kwargs["voice"] == "nova"
        assert call_kwargs["speed"] == 1.5


# ---------------------------------------------------------------------------
# TestSaveToFile
# ---------------------------------------------------------------------------


class TestSaveToFile:
    def test_non_streaming_create_called(self, tts_config: TTSConfig, tmp_path) -> None:
        client = create_mock_client()
        tts = _build_tts(client, tts_config)

        out_path = tmp_path / "output.wav"
        tts.save_to_file("Hello", str(out_path))

        client.audio.speech.create.assert_called_once()
        call_kwargs = client.audio.speech.create.call_args[1]
        assert call_kwargs["response_format"] == "wav"
        assert call_kwargs["input"] == "Hello"

    def test_file_written(self, tts_config: TTSConfig, tmp_path) -> None:
        client = create_mock_client()
        mock_response = MagicMock()
        client.audio.speech.create.return_value = mock_response
        tts = _build_tts(client, tts_config)

        out_path = tmp_path / "output.wav"
        tts.save_to_file("Hello", str(out_path))

        mock_response.write_to_file.assert_called_once_with(str(out_path))

    def test_save_to_file_validation(self, tts_config: TTSConfig, tmp_path) -> None:
        client = create_mock_client()
        tts = _build_tts(client, tts_config)

        with pytest.raises(TTSError, match="empty"):
            tts.save_to_file("", str(tmp_path / "output.wav"))

    def test_save_to_file_api_error(self, tts_config: TTSConfig, tmp_path) -> None:
        client = create_mock_client()
        client.audio.speech.create.side_effect = openai.APIConnectionError(request=MagicMock())
        tts = _build_tts(client, tts_config)

        with pytest.raises(TTSError):
            tts.save_to_file("Hello", str(tmp_path / "output.wav"))
