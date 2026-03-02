"""Stress / load tests for GoogleCloudASR against the real Google Cloud Speech API.

These tests exercise the ASR module under heavier conditions than typical
integration tests: rapid reset cycles, sustained streaming, and repeated
start/stop.  They are excluded from the default pytest run.

Requires:
    - Valid Google Cloud credentials (GOOGLE_APPLICATION_CREDENTIALS env var)
    - A WAV file containing speech audio

Configuration via environment variables:
    ASR_TEST_WAV   Path to a speech WAV file (required, no default)
    ASR_TEST_LANG  Language code for recognition (default: en-US)

Run:
    GOOGLE_APPLICATION_CREDENTIALS=creds.json ASR_TEST_WAV=speech.wav \
        uv run pytest -m requires_stress voice_pipeline/tests/asr/ -v
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from voice_pipeline.asr.asr import GoogleCloudASR
from voice_pipeline.core.config import ASRConfig

from .conftest import audio_config_from_wav, read_wav_frames

pytestmark = pytest.mark.requires_stress


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestASRStress:
    def test_rapid_reset_cycles(self, speech_wav: Path, asr_lang: str) -> None:
        """Rapid reset() calls between short bursts of audio.

        Simulates a scenario where the user makes many short utterances in
        quick succession.  Each reset creates a new gRPC stream.  Verifies
        that the module remains functional after many cycles.
        """
        info, frames = read_wav_frames(speech_wav)
        frame_sec = 30 / 1000
        audio_cfg = audio_config_from_wav(info)
        asr = GoogleCloudASR(
            asr_config=ASRConfig(language_code=asr_lang, interim_results=True),
            audio_config=audio_cfg,
        )
        asr.start()
        try:
            n_cycles = 5
            frames_per_burst = max(10, len(frames) // 10)

            for cycle in range(n_cycles):
                for frame in frames[:frames_per_burst]:
                    asr.feed_audio(frame)
                    time.sleep(frame_sec)
                time.sleep(0.5)
                asr.reset()
                assert asr.get_text() == "", f"Transcript not cleared after reset cycle {cycle}"

            # Final turn: full stream should still produce results
            for frame in frames:
                asr.feed_audio(frame)
                time.sleep(frame_sec)
            time.sleep(2.0)
            final = asr.get_text()
            assert len(final) > 0, "Expected transcript after rapid reset cycles"
        finally:
            asr.stop()

    def test_sustained_streaming(self, speech_wav: Path, asr_lang: str) -> None:
        """Stream the same audio file multiple times back-to-back without reset.

        Verifies the gRPC stream handles sustained audio input within a single
        turn.  The total duration may approach the ~5 minute gRPC limit depending
        on the WAV file length; the test passes as long as transcripts are produced.
        """
        info, frames = read_wav_frames(speech_wav)
        frame_sec = 30 / 1000
        asr = GoogleCloudASR(
            asr_config=ASRConfig(language_code=asr_lang, interim_results=True),
            audio_config=audio_config_from_wav(info),
        )
        asr.start()
        try:
            # Stream the audio 3 times consecutively (within one turn)
            repeats = 3
            for _ in range(repeats):
                for frame in frames:
                    asr.feed_audio(frame)
                    time.sleep(frame_sec)

            time.sleep(2.0)
            final = asr.get_text()
            assert len(final) > 0, "Expected transcript after sustained streaming"
        finally:
            asr.stop()

    def test_back_to_back_start_stop(self, speech_wav: Path, asr_lang: str) -> None:
        """Repeated start() / stop() cycles with brief streaming in between.

        Verifies that gRPC client creation/teardown is reliable over multiple
        full lifecycle cycles.
        """
        info, frames = read_wav_frames(speech_wav)
        frame_sec = 30 / 1000
        audio_cfg = audio_config_from_wav(info)
        n_cycles = 3
        frames_per_cycle = max(20, len(frames) // 5)

        for cycle in range(n_cycles):
            asr = GoogleCloudASR(
                asr_config=ASRConfig(language_code=asr_lang, interim_results=True),
                audio_config=audio_cfg,
            )
            asr.start()
            try:
                for frame in frames[:frames_per_cycle]:
                    asr.feed_audio(frame)
                    time.sleep(frame_sec)
                time.sleep(1.0)
                text = asr.get_text()
                assert isinstance(text, str), f"Cycle {cycle}: get_text() returned non-string"
            finally:
                asr.stop()

            assert not asr._running.is_set(), f"Cycle {cycle}: still running after stop"
            assert asr._client is None, f"Cycle {cycle}: client not cleaned up"

        # Final cycle: verify full transcript works
        asr = GoogleCloudASR(
            asr_config=ASRConfig(language_code=asr_lang, interim_results=True),
            audio_config=audio_cfg,
        )
        asr.start()
        try:
            for frame in frames:
                asr.feed_audio(frame)
                time.sleep(frame_sec)
            time.sleep(2.0)
            final = asr.get_text()
            assert len(final) > 0, "Expected transcript after repeated start/stop cycles"
        finally:
            asr.stop()

    def test_reset_without_feeding_audio(self, speech_wav: Path, asr_lang: str) -> None:
        """reset() called immediately without feeding any audio frames.

        Edge case: orchestrator may reset before any audio arrives (e.g.,
        false positive turn detection).
        """
        info, _ = read_wav_frames(speech_wav)
        asr = GoogleCloudASR(
            asr_config=ASRConfig(language_code=asr_lang),
            audio_config=audio_config_from_wav(info),
        )
        asr.start()
        try:
            for _ in range(5):
                asr.reset()
                assert asr.get_text() == ""
                time.sleep(0.2)

            assert asr._running.is_set(), "ASR should still be running after empty resets"
        finally:
            asr.stop()
