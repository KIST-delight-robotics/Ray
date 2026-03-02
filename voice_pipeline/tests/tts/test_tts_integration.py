"""Integration tests for OpenAITTS with real OpenAI API.

Requires OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import pytest

from voice_pipeline.core.config import TTSConfig
from voice_pipeline.tts.exceptions import TTSError
from voice_pipeline.tts.tts import OpenAITTS

pytestmark = pytest.mark.requires_api


@pytest.fixture
def tts(openai_api_key: str) -> OpenAITTS:  # noqa: ARG001
    """Create an OpenAITTS with default config."""
    return OpenAITTS(TTSConfig())


class TestBasicSynthesis:
    def test_synthesize_short_text(self, tts: OpenAITTS) -> None:
        stream = tts.synthesize("Hello world")
        chunks = list(stream)

        assert len(chunks) > 0
        assert all(isinstance(c, bytes) for c in chunks)
        assert len(stream.audio) > 0

    def test_pcm_audio_size_reasonable(self, tts: OpenAITTS) -> None:
        """PCM 24kHz 16-bit mono: ~48000 bytes/sec. Short phrase should be < 5 sec."""
        stream = tts.synthesize("Hi")
        list(stream)

        assert len(stream.audio) > 100
        assert len(stream.audio) < 48000 * 5


class TestStreamIteration:
    def test_chunks_concatenate_to_audio(self, tts: OpenAITTS) -> None:
        stream = tts.synthesize("Good morning")
        collected = bytearray()
        for chunk in stream:
            collected.extend(chunk)

        assert bytes(collected) == stream.audio

    def test_timestamps_empty(self, tts: OpenAITTS) -> None:
        """OpenAI TTS does not support word-level timestamps."""
        stream = tts.synthesize("Hello there")
        list(stream)

        assert stream.timestamps == ()

    def test_result_matches_audio(self, tts: OpenAITTS) -> None:
        stream = tts.synthesize("Testing result property")
        list(stream)

        result = stream.result
        assert result.audio == stream.audio
        assert result.timestamps == ()


class TestSaveToFile:
    def test_save_creates_wav_file(self, tts: OpenAITTS, tmp_path) -> None:
        out_path = tmp_path / "output.wav"
        tts.save_to_file("Hello world", str(out_path))

        assert out_path.exists()
        content = out_path.read_bytes()
        # WAV files start with RIFF header
        assert content[:4] == b"RIFF"


class TestErrorRecovery:
    def test_invalid_model_propagates_error(self, openai_api_key: str) -> None:  # noqa: ARG002
        tts = OpenAITTS(TTSConfig(model="not-a-real-model-xyz"))
        with pytest.raises(TTSError):
            stream = tts.synthesize("Hello")
            list(stream)

    def test_recovery_after_error(self, openai_api_key: str) -> None:  # noqa: ARG002
        bad_tts = OpenAITTS(TTSConfig(model="not-a-real-model-xyz"))
        with pytest.raises(TTSError):
            stream = bad_tts.synthesize("Hello")
            list(stream)

        good_tts = OpenAITTS(TTSConfig())
        stream = good_tts.synthesize("Hello")
        list(stream)
        assert len(stream.audio) > 0
