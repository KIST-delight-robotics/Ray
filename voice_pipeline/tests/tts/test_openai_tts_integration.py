"""Integration tests for OpenAITTS with real OpenAI API.

Requires OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import wave

import pytest

from voice_pipeline.tts.exceptions import TTSError
from voice_pipeline.tts.greeting_audio import synthesize_to_wav
from voice_pipeline.tts.openai_tts import OpenAITTS

pytestmark = pytest.mark.requires_api


@pytest.fixture
def tts(openai_api_key: str) -> OpenAITTS:  # noqa: ARG001
    """Create an OpenAITTS with default config."""
    return OpenAITTS()


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


class TestSynthesizeToWav:
    def test_creates_valid_wav_file(self, tts: OpenAITTS, tmp_path) -> None:
        out_path = tmp_path / "output.wav"
        synthesize_to_wav(tts, "Hello world", out_path)

        assert out_path.exists()
        with wave.open(str(out_path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 24000
            assert wf.getnframes() > 0


class TestErrorRecovery:
    def test_invalid_model_propagates_error(
        self,
        openai_api_key: str,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(OpenAITTS, "_MODEL", "not-a-real-model-xyz")
        tts = OpenAITTS()
        with pytest.raises(TTSError):
            stream = tts.synthesize("Hello")
            list(stream)

    def test_recovery_after_error(self, openai_api_key: str) -> None:  # noqa: ARG002
        # 한 테스트 함수 내 2개 인스턴스가 서로 다른 `_MODEL` 필요 →
        # monkeypatch teardown이 같은 함수 내에서는 작동하지 않으므로
        # 인스턴스 attr 직접 세팅으로 class var를 가림.
        bad_tts = OpenAITTS()
        bad_tts._MODEL = "not-a-real-model-xyz"
        with pytest.raises(TTSError):
            stream = bad_tts.synthesize("Hello")
            list(stream)

        good_tts = OpenAITTS()
        stream = good_tts.synthesize("Hello")
        list(stream)
        assert len(stream.audio) > 0
