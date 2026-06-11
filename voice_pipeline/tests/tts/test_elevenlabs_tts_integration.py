"""Integration tests for ElevenLabsTTS with real ElevenLabs API.

Requires ELEVENLABS_API_KEY environment variable.
"""

from __future__ import annotations

import wave

import pytest

from voice_pipeline.tts.elevenlabs_tts import ElevenLabsTTS
from voice_pipeline.tts.exceptions import TTSError
from voice_pipeline.tts.greeting_audio import synthesize_to_wav
from voice_pipeline.tts.utterance_truncator import truncate_by_timestamps

pytestmark = pytest.mark.requires_api


@pytest.fixture
def tts(elevenlabs_api_key: str) -> ElevenLabsTTS:  # noqa: ARG001
    """Create an ElevenLabsTTS with default config."""
    return ElevenLabsTTS()


class TestBasicSynthesis:
    def test_synthesize_short_text(self, tts: ElevenLabsTTS) -> None:
        stream = tts.synthesize("Hello world")
        chunks = list(stream)

        assert len(chunks) > 0
        assert all(isinstance(c, bytes) for c in chunks)
        assert len(stream.audio) > 0

    def test_pcm_audio_size_reasonable(self, tts: ElevenLabsTTS) -> None:
        """PCM 24kHz 16-bit mono: ~48000 bytes/sec. Short phrase should be < 5 sec."""
        stream = tts.synthesize("Hi")
        list(stream)

        assert len(stream.audio) > 100
        assert len(stream.audio) < 48000 * 5


class TestStreamIteration:
    def test_chunks_concatenate_to_audio(self, tts: ElevenLabsTTS) -> None:
        stream = tts.synthesize("Good morning")
        collected = bytearray()
        for chunk in stream:
            collected.extend(chunk)

        assert bytes(collected) == stream.audio


class TestWordTimestamps:
    def test_words_match_text_split(self, tts: ElevenLabsTTS) -> None:
        text = "Hello there, how are you today?"
        stream = tts.synthesize(text)
        list(stream)

        assert [ts.word for ts in stream.timestamps] == text.split()

    def test_times_monotonic_and_within_audio(self, tts: ElevenLabsTTS) -> None:
        """Timestamps are absolute from audio start and bounded by audio duration."""
        stream = tts.synthesize("One two three four five")
        list(stream)

        timestamps = stream.timestamps
        assert len(timestamps) == 5
        starts = [ts.start_sec for ts in timestamps]
        assert starts == sorted(starts)

        audio_sec = len(stream.audio) / 48000  # 24kHz × 2 bytes
        assert timestamps[-1].end_sec <= audio_sec + 1.0

    def test_digits_keep_original_text_alignment(self, tts: ElevenLabsTTS) -> None:
        """Alignment must track the original text, not the normalized one."""
        text = "I have 3 apples"
        stream = tts.synthesize(text)
        list(stream)

        assert [ts.word for ts in stream.timestamps] == text.split()

    def test_truncate_by_timestamps_returns_prefix(self, tts: ElevenLabsTTS) -> None:
        """Timestamps must drive barge-in truncation to a whitespace-joined prefix."""
        text = "The quick brown fox jumped over the lazy dog"
        stream = tts.synthesize(text)
        list(stream)

        timestamps = list(stream.timestamps)
        mid_sec = timestamps[len(timestamps) // 2].start_sec
        truncated = truncate_by_timestamps(text, mid_sec, timestamps)

        assert truncated
        assert text.startswith(truncated)


class TestSynthesizeToWav:
    def test_creates_valid_wav_file(self, tts: ElevenLabsTTS, tmp_path) -> None:
        out_path = tmp_path / "output.wav"
        synthesize_to_wav(tts, "Hello world", out_path)

        assert out_path.exists()
        with wave.open(str(out_path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 24000
            assert wf.getnframes() > 0


class TestErrorRecovery:
    def test_invalid_voice_propagates_error(
        self,
        elevenlabs_api_key: str,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SDK is lazy — the error surfaces during iteration, not at synthesize()."""
        monkeypatch.setattr(ElevenLabsTTS, "_VOICE_ID", "not-a-real-voice-xyz")
        tts = ElevenLabsTTS()
        with pytest.raises(TTSError):
            stream = tts.synthesize("Hello")
            list(stream)

    def test_recovery_after_error(self, elevenlabs_api_key: str) -> None:  # noqa: ARG002
        # 한 테스트 함수 내 2개 인스턴스가 서로 다른 `_VOICE_ID` 필요 →
        # monkeypatch teardown이 같은 함수 내에서는 작동하지 않으므로
        # 인스턴스 attr 직접 세팅으로 class var를 가림.
        bad_tts = ElevenLabsTTS()
        bad_tts._VOICE_ID = "not-a-real-voice-xyz"
        with pytest.raises(TTSError):
            stream = bad_tts.synthesize("Hello")
            list(stream)

        good_tts = ElevenLabsTTS()
        stream = good_tts.synthesize("Hello")
        list(stream)
        assert len(stream.audio) > 0
