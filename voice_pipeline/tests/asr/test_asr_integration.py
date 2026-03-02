"""Integration tests for GoogleCloudASR against the real Google Cloud Speech API.

Requires:
    - Valid Google Cloud credentials (GOOGLE_APPLICATION_CREDENTIALS env var)
    - A WAV file containing speech audio

Configuration via environment variables:
    ASR_TEST_WAV   Path to a speech WAV file (required, no default)
    ASR_TEST_LANG  Language code for recognition (default: en-US)

Run integration tests:
    GOOGLE_APPLICATION_CREDENTIALS=creds.json ASR_TEST_WAV=speech.wav \\
        uv run pytest -m requires_api voice_pipeline/tests/asr/ -v

Run all tests including stress:
    GOOGLE_APPLICATION_CREDENTIALS=creds.json ASR_TEST_WAV=speech.wav \\
        uv run pytest -m 'requires_api or requires_stress' voice_pipeline/tests/asr/ -v
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from voice_pipeline.asr.asr import GoogleCloudASR
from voice_pipeline.asr.exceptions import ASRError
from voice_pipeline.core.config import ASRConfig

from .conftest import audio_config_from_wav, read_wav_frames

pytestmark = pytest.mark.requires_api


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestASRIntegration:
    def test_streaming_feed_and_get_text(self, speech_wav: Path, asr_lang: str) -> None:
        """Matches orchestrator loop: feed_audio + get_text per frame.

        Verifies that interim transcripts appear during streaming and that
        a non-empty final transcript is produced.
        """
        info, frames = read_wav_frames(speech_wav)
        frame_interval = info.sample_rate * 30 // 1000  # samples per 30ms
        frame_sec = frame_interval / info.sample_rate
        asr = GoogleCloudASR(
            asr_config=ASRConfig(language_code=asr_lang, interim_results=True),
            audio_config=audio_config_from_wav(info),
        )
        asr.start()
        try:
            got_interim = False
            for frame in frames:
                asr.feed_audio(frame)
                text = asr.get_text()
                assert isinstance(text, str)
                if text:
                    got_interim = True
                time.sleep(frame_sec)

            # Wait for final processing after last frame
            time.sleep(2.0)
            final = asr.get_text()
            assert len(final) > 0, "Expected non-empty transcript for speech audio"
            assert got_interim, "Expected at least one interim result during streaming"
        finally:
            asr.stop()

    def test_reset_between_turns(self, speech_wav: Path, asr_lang: str) -> None:
        """Matches orchestrator turn cycle: stream -> reset -> stream again."""
        info, frames = read_wav_frames(speech_wav)
        frame_sec = 30 / 1000
        asr = GoogleCloudASR(
            asr_config=ASRConfig(language_code=asr_lang),
            audio_config=audio_config_from_wav(info),
        )
        asr.start()
        try:
            # --- Turn 1 ---
            for frame in frames:
                asr.feed_audio(frame)
                time.sleep(frame_sec)
            time.sleep(2.0)

            first_result = asr.get_text()
            assert len(first_result) > 0

            # --- Turn boundary ---
            asr.reset()
            assert asr.get_text() == ""

            # --- Turn 2 ---
            for frame in frames:
                asr.feed_audio(frame)
                time.sleep(frame_sec)
            time.sleep(2.0)

            second_result = asr.get_text()
            assert len(second_result) > 0
        finally:
            asr.stop()

    def test_stop_during_active_stream(self, speech_wav: Path, asr_lang: str) -> None:
        """Stop while gRPC stream is actively receiving audio.

        Feeds frames with real-time pacing, then calls stop() mid-stream.
        Verifies clean shutdown: sentinel sent, thread joined, resources released.
        """
        info, frames = read_wav_frames(speech_wav)
        frame_sec = 30 / 1000
        asr = GoogleCloudASR(
            asr_config=ASRConfig(language_code=asr_lang),
            audio_config=audio_config_from_wav(info),
        )
        asr.start()

        # Feed roughly half the frames at real-time pace
        half = len(frames) // 2
        for frame in frames[:half]:
            asr.feed_audio(frame)
            time.sleep(frame_sec)

        # Stop mid-stream — should not hang or raise
        asr.stop()

        assert not asr._running.is_set()
        assert asr._client is None
        assert asr._audio_queue is None


class TestASRErrorRecovery:
    """Error recovery scenarios against real Google Cloud Speech API."""

    def test_invalid_language_code_propagates_error(self, speech_wav: Path, asr_lang: str) -> None:
        """An invalid language code should surface an ASRError during streaming.

        The error may appear asynchronously via the reader thread — we verify it
        is propagated through get_text() or feed_audio().
        """
        info, frames = read_wav_frames(speech_wav)
        frame_sec = 30 / 1000
        asr = GoogleCloudASR(
            asr_config=ASRConfig(language_code="xx-INVALID", interim_results=True),
            audio_config=audio_config_from_wav(info),
        )
        asr.start()
        try:
            error_raised = False
            for frame in frames:
                try:
                    asr.feed_audio(frame)
                    asr.get_text()
                except ASRError:
                    error_raised = True
                    break
                time.sleep(frame_sec)

            if not error_raised:
                # Give the reader thread time to receive the gRPC error
                time.sleep(3.0)
                with pytest.raises(ASRError):
                    asr.get_text()
        finally:
            asr.stop()

    def test_recovery_after_error_via_stop_start(self, speech_wav: Path, asr_lang: str) -> None:
        """After an error, stop() + start() should restore normal operation."""
        info, frames = read_wav_frames(speech_wav)
        frame_sec = 30 / 1000
        audio_cfg = audio_config_from_wav(info)

        # Phase 1: trigger error with bad language code
        asr = GoogleCloudASR(
            asr_config=ASRConfig(language_code="xx-INVALID"),
            audio_config=audio_cfg,
        )
        asr.start()
        for frame in frames[:20]:
            try:
                asr.feed_audio(frame)
                asr.get_text()
            except ASRError:
                break
            time.sleep(frame_sec)
        asr.stop()

        # Phase 2: restart with valid config — must succeed
        asr_good = GoogleCloudASR(
            asr_config=ASRConfig(language_code=asr_lang, interim_results=True),
            audio_config=audio_cfg,
        )
        asr_good.start()
        try:
            for frame in frames:
                asr_good.feed_audio(frame)
                time.sleep(frame_sec)
            time.sleep(2.0)
            final = asr_good.get_text()
            assert len(final) > 0, "Expected valid transcript after error recovery"
        finally:
            asr_good.stop()

    def test_reset_recovers_from_stale_stream(self, speech_wav: Path, asr_lang: str) -> None:
        """reset() should recover from a potentially stale gRPC stream.

        Simulates a gap where no audio is fed for several seconds (e.g., user
        paused), then reset() creates a fresh stream that works normally.
        """
        info, frames = read_wav_frames(speech_wav)
        frame_sec = 30 / 1000
        asr = GoogleCloudASR(
            asr_config=ASRConfig(language_code=asr_lang, interim_results=True),
            audio_config=audio_config_from_wav(info),
        )
        asr.start()
        try:
            # Feed a few frames then go silent
            for frame in frames[:10]:
                asr.feed_audio(frame)
                time.sleep(frame_sec)

            # Silence gap — stream may become stale
            time.sleep(5.0)

            # Reset should create a fresh stream
            asr.reset()
            assert asr.get_text() == ""

            # New turn should work normally
            for frame in frames:
                asr.feed_audio(frame)
                time.sleep(frame_sec)
            time.sleep(2.0)
            final = asr.get_text()
            assert len(final) > 0, "Expected valid transcript after stale-stream reset"
        finally:
            asr.stop()

    def test_double_stop_is_safe(self, speech_wav: Path, asr_lang: str) -> None:
        """Calling stop() twice should not raise or hang."""
        info, frames = read_wav_frames(speech_wav)
        asr = GoogleCloudASR(
            asr_config=ASRConfig(language_code=asr_lang),
            audio_config=audio_config_from_wav(info),
        )
        asr.start()
        for frame in frames[:5]:
            asr.feed_audio(frame)
        asr.stop()
        asr.stop()  # Second stop — must be safe

        assert not asr._running.is_set()
        assert asr._client is None
