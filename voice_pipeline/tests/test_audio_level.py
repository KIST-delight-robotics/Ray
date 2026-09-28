"""Tests for audio_level (재생 파일 출력 레벨 정규화)."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from voice_pipeline import audio_level
from voice_pipeline.audio_level import (
    GAIN_MAX,
    PEAK_CEILING,
    ensure_normalized,
    frame_peaks,
    gain_for,
    measure,
)
from voice_pipeline.settings import CHIME_TARGET_PEAK, SPEECH_TARGET_PEAK

SAMPLE_RATE = 24000


def _tone(amplitude: float, seconds: float = 1.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _write(path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")
    return str(path)


# ---------------------------------------------------------------------------
# frame_peaks / measure
# ---------------------------------------------------------------------------


def test_frame_peaks_drops_silent_frames():
    loud = _tone(0.5, seconds=0.2)
    silence = np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32)
    peaks = frame_peaks(np.concatenate([loud, silence]), SAMPLE_RATE)

    assert peaks.size == 5  # 0.2s / 40ms — 무음 5프레임은 빠진다
    assert np.all(peaks > 0.4)


def test_frame_peaks_empty_when_shorter_than_one_frame():
    assert frame_peaks(_tone(0.5, seconds=0.01), SAMPLE_RATE).size == 0


def test_measure_speech_uses_median():
    samples = _tone(0.3)
    assert measure(samples, SAMPLE_RATE, speech=True) == pytest.approx(0.3, abs=0.01)


def test_measure_chime_uses_p90_so_transient_counts():
    """중앙값은 낮고 꼭대기만 큰 신호 — 알림음 척도는 그 꼭대기를 잡아야 한다."""
    quiet = _tone(0.1, seconds=0.8)
    attack = _tone(0.6, seconds=0.2)
    samples = np.concatenate([attack, quiet])

    assert measure(samples, SAMPLE_RATE, speech=True) == pytest.approx(0.1, abs=0.02)
    assert measure(samples, SAMPLE_RATE, speech=False) > 0.3


def test_measure_silence_returns_zero():
    assert measure(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE, speech=True) == 0.0


# ---------------------------------------------------------------------------
# gain_for
# ---------------------------------------------------------------------------


def test_gain_attenuates_loud_speech_to_target():
    samples = _tone(0.48)  # 목표(0.12)의 4배
    assert gain_for(samples, SAMPLE_RATE, speech=True) == pytest.approx(0.25, rel=0.05)


def test_gain_boosts_quiet_speech_to_target():
    samples = _tone(0.06)
    gain = gain_for(samples, SAMPLE_RATE, speech=True)

    assert gain == pytest.approx(SPEECH_TARGET_PEAK / 0.06, rel=0.05)
    assert gain > 1.0


def test_gain_capped_at_max_for_near_silent_input():
    samples = _tone(0.025)  # VOICED_PEAK_FLOOR 는 넘지만 목표보다 한참 작다
    assert gain_for(samples, SAMPLE_RATE, speech=True) == GAIN_MAX


def test_gain_respects_peak_ceiling():
    """평균은 작은데 순간 피크가 1.0 에 붙은 신호 — 증폭하면 클리핑한다."""
    samples = _tone(0.05)
    samples[100] = 0.99
    gain = gain_for(samples, SAMPLE_RATE, speech=True)

    assert float(np.abs(samples * gain).max()) <= PEAK_CEILING + 1e-6


def test_explicit_target_overrides_default():
    """노래는 통계는 말소리와 같고 목표만 다르다 (scripts/normalize_song_audio.py)."""
    samples = _tone(0.6)

    assert gain_for(samples, SAMPLE_RATE, speech=True, target=0.18) == pytest.approx(0.3, rel=0.05)
    assert gain_for(samples, SAMPLE_RATE, speech=True) == pytest.approx(SPEECH_TARGET_PEAK / 0.6, rel=0.05)


def test_gain_of_silence_is_unity():
    assert gain_for(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE, speech=True) == 1.0


# ---------------------------------------------------------------------------
# ensure_normalized
# ---------------------------------------------------------------------------


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    path = tmp_path / "normalized"
    monkeypatch.setattr(audio_level, "_CACHE_DIR", str(path))
    return path


def test_normalized_copy_hits_speech_target(tmp_path, cache_dir):
    src = _write(tmp_path / "greeting.wav", _tone(0.48))

    dst = ensure_normalized(src, speech=True)
    data, rate = sf.read(str(dst), dtype="float32")

    assert dst.parent == cache_dir
    assert rate == SAMPLE_RATE
    assert measure(data, rate, speech=True) == pytest.approx(SPEECH_TARGET_PEAK, abs=0.01)


def test_normalized_copy_hits_chime_target(tmp_path, cache_dir):
    src = _write(tmp_path / "ready.wav", _tone(0.6))

    data, rate = sf.read(str(ensure_normalized(src, speech=False)), dtype="float32")

    assert measure(data, rate, speech=False) == pytest.approx(CHIME_TARGET_PEAK, abs=0.01)


def test_normalized_copy_preserves_channels_and_rate(tmp_path, cache_dir):
    stereo = np.stack([_tone(0.4, sample_rate=44100), _tone(0.4, sample_rate=44100)], axis=1)
    src = _write(tmp_path / "stereo.wav", stereo, sample_rate=44100)

    data, rate = sf.read(str(ensure_normalized(src, speech=False)), dtype="float32")

    assert rate == 44100
    assert data.ndim == 2 and data.shape[1] == 2


def test_second_call_reuses_cached_copy(tmp_path, cache_dir):
    src = _write(tmp_path / "greeting.wav", _tone(0.48))

    first = ensure_normalized(src, speech=True)
    mtime = first.stat().st_mtime_ns
    second = ensure_normalized(src, speech=True)

    assert second == first
    assert second.stat().st_mtime_ns == mtime  # 다시 쓰지 않았다


def test_changed_source_gets_a_new_copy(tmp_path, cache_dir):
    src = tmp_path / "greeting.wav"
    first = ensure_normalized(_write(src, _tone(0.48)), speech=True)

    _write(src, _tone(0.30, seconds=2.0))
    second = ensure_normalized(src, speech=True)

    assert second != first


def test_speech_and_chime_copies_do_not_collide(tmp_path, cache_dir):
    src = _write(tmp_path / "sound.wav", _tone(0.48))

    assert ensure_normalized(src, speech=True) != ensure_normalized(src, speech=False)


def test_unreadable_source_raises_runtime_error(tmp_path, cache_dir):
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"not audio")

    with pytest.raises(RuntimeError, match="Failed to read audio"):
        ensure_normalized(broken, speech=True)
