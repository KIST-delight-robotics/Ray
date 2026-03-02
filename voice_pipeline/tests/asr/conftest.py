"""Shared fixtures and WAV helpers for ASR tests."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import wave
from pathlib import Path

import pytest

from voice_pipeline.core.config import AudioConfig

_SAMPLE_RATE_MIN = 8000
_SAMPLE_RATE_MAX = 48000


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class WavInfo:
    """Properties read from a WAV file header."""

    path: Path
    sample_rate: int
    channels: int
    sample_width: int
    n_frames: int

    @property
    def duration_sec(self) -> float:
        return self.n_frames / self.sample_rate


def read_wav_info(path: Path) -> WavInfo:
    """Read WAV file properties from the header."""
    with wave.open(str(path), "rb") as wf:
        return WavInfo(
            path=path,
            sample_rate=wf.getframerate(),
            channels=wf.getnchannels(),
            sample_width=wf.getsampwidth(),
            n_frames=wf.getnframes(),
        )


def ensure_compatible_wav(path: Path, tmp_path: Path) -> Path:
    """Return a WAV file compatible with Google STT (mono, 16-bit, 8-48kHz).

    Returns the original path if already compatible, otherwise resamples
    via ffmpeg into tmp_path.
    """
    info = read_wav_info(path)
    needs_resample = not (_SAMPLE_RATE_MIN <= info.sample_rate <= _SAMPLE_RATE_MAX)
    needs_convert = info.channels != 1 or info.sample_width != 2

    if not needs_resample and not needs_convert:
        return path

    target_rate = 16000 if needs_resample else info.sample_rate
    out = tmp_path / f"converted_{path.name}"
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(path),
            "-ar", str(target_rate), "-ac", "1", "-sample_fmt", "s16",
            str(out),
        ],
        capture_output=True,
        check=True,
    )
    return out


def read_wav_frames(path: Path, frame_duration_ms: int = 30) -> tuple[WavInfo, list[bytes]]:
    """Read a WAV file and split into pipeline-sized frames."""
    info = read_wav_info(path)
    frame_size_samples = info.sample_rate * frame_duration_ms // 1000
    frame_size_bytes = frame_size_samples * info.sample_width * info.channels

    with wave.open(str(path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())

    frames: list[bytes] = []
    for offset in range(0, len(raw), frame_size_bytes):
        chunk = raw[offset : offset + frame_size_bytes]
        if len(chunk) == frame_size_bytes:
            frames.append(chunk)
    return info, frames


def audio_config_from_wav(info: WavInfo) -> AudioConfig:
    """Build an AudioConfig matching the WAV file's properties."""
    return AudioConfig(
        sample_rate=info.sample_rate,
        channels=info.channels,
        sample_width=info.sample_width,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SKIP_MSG = "ASR_TEST_WAV not set — provide a path to a speech WAV file"


@pytest.fixture
def speech_wav(tmp_path: Path) -> Path:
    """Resolve and validate the speech WAV file.

    Reads ASR_TEST_WAV env var.  Converts to compatible format if needed.
    Skips the test if the env var is not set or the file doesn't exist.
    """
    wav_path_str = os.environ.get("ASR_TEST_WAV")
    if not wav_path_str:
        pytest.skip(_SKIP_MSG)

    wav_path = Path(wav_path_str)
    if not wav_path.exists():
        pytest.fail(f"ASR_TEST_WAV file not found: {wav_path}")

    return ensure_compatible_wav(wav_path, tmp_path)


@pytest.fixture
def asr_lang() -> str:
    """Language code for recognition, from ASR_TEST_LANG or default en-US."""
    return os.environ.get("ASR_TEST_LANG", "en-US")
