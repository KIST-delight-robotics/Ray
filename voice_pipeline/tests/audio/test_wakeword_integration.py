"""Integration tests for WakewordDetector with real VAD model + Google STT.

Requires:
  - silero-vad package (real Silero VAD model)
  - Google Cloud credentials configured
  - WAKEWORD_TEST_WAV: path to a WAV file containing the wakeword
  - WAKEWORD_TEST_SILENCE_WAV (optional): path to a silence WAV file
  - WAKEWORD_TEST_KEYWORD (optional): keyword to detect (default "ray")
"""

from __future__ import annotations

import pytest

from voice_pipeline.audio.wakeword import WakewordDetector
from voice_pipeline.tests.audio.conftest import read_wav_frames

pytestmark = [pytest.mark.requires_api, pytest.mark.requires_model]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detector(keyword: str, monkeypatch: pytest.MonkeyPatch) -> WakewordDetector:
    # max_speech_duration_sec defaults to 3.0 via class var.
    monkeypatch.setattr(WakewordDetector, "_KEYWORDS", (keyword,))
    return WakewordDetector()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWakewordDetection:
    """Integration tests feeding real WAV files to real VAD + STT."""

    def test_detects_wakeword_in_speech(self, speech_wav, wakeword_keyword, monkeypatch):
        """Feed WAV containing the wakeword → returns True at some point."""
        _, frames = read_wav_frames(speech_wav)
        detector = _make_detector(wakeword_keyword, monkeypatch)

        detected = False
        for frame in frames:
            if detector.feed_audio(frame):
                detected = True
                break

        assert detected, f"Wakeword '{wakeword_keyword}' not detected in {speech_wav} ({len(frames)} frames)"

    def test_silence_does_not_trigger(self, silence_wav, wakeword_keyword, monkeypatch):
        """Feed silence → never returns True."""
        if silence_wav is None:
            pytest.skip("WAKEWORD_TEST_SILENCE_WAV not set")

        _, frames = read_wav_frames(silence_wav)
        detector = _make_detector(wakeword_keyword, monkeypatch)

        for frame in frames:
            assert not detector.feed_audio(frame), "Wakeword falsely detected in silence"

    def test_multiple_detection_cycles(self, speech_wav, wakeword_keyword, monkeypatch):
        """Detector can detect the wakeword across multiple cycles."""
        _, frames = read_wav_frames(speech_wav)
        detector = _make_detector(wakeword_keyword, monkeypatch)

        detections = 0
        for _cycle in range(2):
            for frame in frames:
                if detector.feed_audio(frame):
                    detections += 1
                    break

        assert detections == 2, f"Expected 2 detections across 2 cycles, got {detections}"
