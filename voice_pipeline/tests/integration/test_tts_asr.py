"""Cross-module integration tests: TTS → ASR round-trip.

Generates audio via OpenAI TTS, then feeds it to Google Cloud ASR and
verifies the transcript matches the original text.  This validates that
the TTS output format is compatible with ASR input expectations, and that
both modules work correctly with real APIs.

Requires:
    - OPENAI_API_KEY env var
    - GOOGLE_APPLICATION_CREDENTIALS env var

Run:
    OPENAI_API_KEY=... GOOGLE_APPLICATION_CREDENTIALS=creds.json \\
        uv run pytest -m requires_api voice_pipeline/tests/integration/ -v
"""

from __future__ import annotations

import subprocess
import time
import wave
from pathlib import Path

import pytest

from voice_pipeline.asr.asr import GoogleCloudASR
from voice_pipeline.core.config import ASRConfig, TTSConfig
from voice_pipeline.tts.tts import OpenAITTS

from .conftest import audio_config_from_wav, make_silence_frames, read_wav_frames

pytestmark = pytest.mark.requires_api


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Target sample rate for ASR (Google STT default).
_ASR_SAMPLE_RATE = 16000

# Silence duration after speech to trigger Google STT end-of-speech detection.
_SILENCE_SEC = 3.0


def _tts_generate_wav(tts: OpenAITTS, text: str, path: Path) -> Path:
    """Generate WAV via TTS and convert to ASR-compatible format.

    OpenAI TTS outputs 24 kHz WAV with potentially malformed headers
    (n_frames = INT_MAX).  ffmpeg re-encodes to 16 kHz mono 16-bit PCM
    with correct headers.
    """
    raw_path = path / "tts_raw.wav"
    tts.save_to_file(text, raw_path)

    fixed_path = path / "tts_fixed.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_path),
            "-ar",
            str(_ASR_SAMPLE_RATE),
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            str(fixed_path),
        ],
        capture_output=True,
        check=True,
    )
    return fixed_path


def _feed_and_transcribe(
    wav_path: Path,
    *,
    language_code: str = "en-US",
    silence_sec: float = _SILENCE_SEC,
) -> tuple[str, list[str]]:
    """Feed WAV to ASR with real-time pacing and silence padding.

    Returns (final_transcript, unique_interims).
    """
    info, frames = read_wav_frames(wav_path)
    audio_cfg = audio_config_from_wav(info)
    frame_sec = 30 / 1000
    frame_bytes = (info.sample_rate * 30 // 1000) * info.sample_width * info.channels

    silence_frames = make_silence_frames(frame_bytes, silence_sec)

    asr = GoogleCloudASR(
        asr_config=ASRConfig(language_code=language_code, interim_results=True),
        audio_config=audio_cfg,
    )
    asr.start()
    try:
        # Small delay for gRPC stream setup
        time.sleep(0.3)

        interims: list[str] = []

        # Feed speech frames at real-time pace
        for frame in frames:
            asr.feed_audio(frame)
            text = asr.get_text()
            if text and (not interims or text != interims[-1]):
                interims.append(text)
            time.sleep(frame_sec)

        # Feed silence to trigger end-of-speech detection
        for frame in silence_frames:
            asr.feed_audio(frame)
            time.sleep(frame_sec)

        final = asr.get_text()
        return final, interims
    finally:
        asr.stop()


def _normalize(text: str) -> str:
    """Normalize text for fuzzy comparison: lowercase, strip punctuation."""
    import re

    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _word_overlap_ratio(expected: str, actual: str) -> float:
    """Fraction of expected words found in the actual transcript."""
    expected_words = set(_normalize(expected).split())
    actual_words = set(_normalize(actual).split())
    if not expected_words:
        return 1.0
    return len(expected_words & actual_words) / len(expected_words)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestTTSToASRRoundTrip:
    """Verify TTS-generated audio is correctly transcribed by ASR."""

    @pytest.fixture
    def tts(self, openai_api_key: str) -> OpenAITTS:
        return OpenAITTS(TTSConfig(model="tts-1", voice="alloy"))

    @pytest.mark.parametrize(
        "text",
        [
            "Hello, my name is Ray. I am a voice assistant.",
            "The quick brown fox jumps over the lazy dog.",
            "Please turn on the lights in the living room.",
        ],
        ids=["greeting", "pangram", "command"],
    )
    def test_round_trip_accuracy(
        self, tts: OpenAITTS, text: str, google_credentials: str, tmp_path: Path
    ) -> None:
        """TTS output should be transcribed with high word overlap."""
        wav_path = _tts_generate_wav(tts, text, tmp_path)
        final, interims = _feed_and_transcribe(wav_path)

        assert len(final) > 0, "Expected non-empty transcript"

        overlap = _word_overlap_ratio(text, final)
        assert overlap >= 0.8, (
            f"Word overlap {overlap:.0%} below 80% threshold.\n"
            f"  Expected: {text!r}\n"
            f"  Got:      {final!r}"
        )

    def test_interim_results_appear_during_streaming(
        self, tts: OpenAITTS, google_credentials: str, tmp_path: Path
    ) -> None:
        """ASR should produce interim results while processing TTS audio."""
        text = "Hello, my name is Ray. I am a voice assistant."
        wav_path = _tts_generate_wav(tts, text, tmp_path)
        final, interims = _feed_and_transcribe(wav_path)

        assert len(interims) > 0, "Expected at least one interim result"
        # Early interims should contain beginning of the text
        first_interim_lower = interims[0].lower()
        assert "hello" in first_interim_lower or "ray" in first_interim_lower, (
            f"First interim should contain early words, got: {interims[0]!r}"
        )

    def test_reset_between_two_phrases(
        self, tts: OpenAITTS, google_credentials: str, tmp_path: Path
    ) -> None:
        """ASR reset should allow transcribing a second TTS phrase cleanly."""
        text1 = "Good morning."
        text2 = "Good night."

        dir1 = tmp_path / "t1"
        dir1.mkdir()
        wav1 = _tts_generate_wav(tts, text1, dir1)

        dir2 = tmp_path / "t2"
        dir2.mkdir()
        wav2 = _tts_generate_wav(tts, text2, dir2)

        info1, frames1 = read_wav_frames(wav1)
        info2, frames2 = read_wav_frames(wav2)
        audio_cfg = audio_config_from_wav(info1)

        frame_bytes = (info1.sample_rate * 30 // 1000) * info1.sample_width
        silence_frames = make_silence_frames(frame_bytes, _SILENCE_SEC)
        frame_sec = 30 / 1000

        asr = GoogleCloudASR(
            asr_config=ASRConfig(language_code="en-US", interim_results=True),
            audio_config=audio_cfg,
        )
        asr.start()
        try:
            time.sleep(0.3)

            # --- Turn 1 ---
            for frame in frames1:
                asr.feed_audio(frame)
                time.sleep(frame_sec)
            for frame in silence_frames:
                asr.feed_audio(frame)
                time.sleep(frame_sec)
            result1 = asr.get_text()

            # --- Reset ---
            asr.reset()
            assert asr.get_text() == ""

            # --- Turn 2 ---
            time.sleep(0.3)
            for frame in frames2:
                asr.feed_audio(frame)
                time.sleep(frame_sec)
            for frame in silence_frames:
                asr.feed_audio(frame)
                time.sleep(frame_sec)
            result2 = asr.get_text()

            assert len(result1) > 0, "Turn 1 should produce a transcript"
            assert len(result2) > 0, "Turn 2 should produce a transcript"

            # Turn 2 should not contain Turn 1 content
            assert "morning" not in result2.lower(), (
                f"Turn 2 should not contain Turn 1 words. Got: {result2!r}"
            )
            assert _word_overlap_ratio("Good night", result2) >= 0.5
        finally:
            asr.stop()

    def test_tts_streaming_to_asr(
        self, tts: OpenAITTS, google_credentials: str, tmp_path: Path
    ) -> None:
        """TTS streaming PCM → write to WAV → ASR should also work.

        Tests the streaming code path (synthesize + iter) as opposed to
        save_to_file.
        """
        text = "Testing one two three four five."
        stream = tts.synthesize(text)
        pcm_data = b""
        for chunk in stream:
            pcm_data += chunk

        # Write PCM to WAV (24kHz, mono, 16-bit — OpenAI PCM format)
        raw_wav = tmp_path / "streaming_raw.wav"
        with wave.open(str(raw_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm_data)

        # Convert to 16kHz for ASR
        fixed_wav = tmp_path / "streaming_fixed.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(raw_wav),
                "-ar",
                str(_ASR_SAMPLE_RATE),
                "-ac",
                "1",
                "-sample_fmt",
                "s16",
                str(fixed_wav),
            ],
            capture_output=True,
            check=True,
        )

        final, _ = _feed_and_transcribe(fixed_wav)
        assert len(final) > 0
        overlap = _word_overlap_ratio(text, final)
        assert overlap >= 0.7, (
            f"Word overlap {overlap:.0%} below 70% threshold.\n"
            f"  Expected: {text!r}\n"
            f"  Got:      {final!r}"
        )
