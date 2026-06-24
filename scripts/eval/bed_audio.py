"""Shared WAV / MUSAN helpers for the acoustic noise-bed tooling.

Used by prepare_noise_bed.py (build the bed master) and calibrate_noise.py
(measure speech/noise levels). Pure functions — the only I/O is reading and
writing WAVs.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

# 유성 구간 검출 임계 — prepare_audio.trim_trailing_silence와 동일 기준(peak 대비 비율).
# 선행/후행 무음을 RMS 계산에서 제외해야 SNR이 실제 발화 에너지를 기준으로 잡힌다.
_VOICED_PEAK_RATIO = 0.02
_VOICED_MIN_THRESHOLD = 0.005


def index_musan_noise(musan_dir: str | Path) -> list[Path]:
    """Return all MUSAN ``noise`` WAV paths under ``<musan_dir>/noise``.

    MUSAN noise clips are 16 kHz mono recordings of ambient/free-sound noise.
    Raises ``FileNotFoundError`` when the directory is missing or holds no WAVs.
    """
    noise_root = Path(musan_dir) / "noise"
    if not noise_root.is_dir():
        raise FileNotFoundError(f"MUSAN noise dir not found: {noise_root}")
    files = sorted(noise_root.glob("**/*.wav"))
    if not files:
        raise FileNotFoundError(f"No noise WAVs under {noise_root}")
    return files


def _read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit WAV as mono float32 in [-1, 1] plus its sample rate."""
    with wave.open(str(path)) as w:
        params = w.getparams()
        pcm = w.readframes(w.getnframes())
    if params.sampwidth != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got sampwidth={params.sampwidth}")
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if params.nchannels > 1:
        samples = samples.reshape(-1, params.nchannels).mean(axis=1)
    return samples, params.framerate


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write float32 [-1, 1] samples as a 16-bit mono WAV."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def _voiced_rms(speech: np.ndarray) -> float:
    """RMS over the voiced region (samples above a peak-relative threshold)."""
    if len(speech) == 0:
        return 0.0
    amp = np.abs(speech)
    peak = float(amp.max())
    if peak < _VOICED_MIN_THRESHOLD:
        return 0.0
    thresh = max(_VOICED_MIN_THRESHOLD, peak * _VOICED_PEAK_RATIO)
    voiced = speech[amp > thresh]
    if len(voiced) == 0:
        return 0.0
    return float(np.sqrt(np.mean(voiced**2)))
