"""Shared fixtures and helpers for wakeword module tests."""

from __future__ import annotations

import dataclasses
import os
import struct
import subprocess
import wave
from pathlib import Path

import pytest

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
    """Return a WAV file compatible with Google STT (mono, 16-bit, 16kHz).

    Returns the original path if already compatible, otherwise resamples
    via ffmpeg into tmp_path.
    """
    info = read_wav_info(path)
    needs_resample = info.sample_rate != 16000
    needs_convert = info.channels != 1 or info.sample_width != 2

    if not needs_resample and not needs_convert:
        return path

    out = tmp_path / f"converted_{path.name}"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
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


def make_silence_frame(num_samples: int = 480) -> bytes:
    """Generate a silent audio frame (all zeros)."""
    return b"\x00" * (num_samples * 2)


def make_tone_frame(num_samples: int = 480, amplitude: int = 16000) -> bytes:
    """Generate a simple tone frame (square wave) to trigger VAD."""
    # Alternating positive/negative samples create a simple signal
    samples = []
    for i in range(num_samples):
        val = amplitude if (i % 8) < 4 else -amplitude
        samples.append(val)
    return struct.pack(f"<{num_samples}h", *samples)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SKIP_MSG = "WAKEWORD_TEST_WAV not set — provide a path to a speech WAV file"


@pytest.fixture
def speech_wav(tmp_path: Path) -> Path:
    """Resolve and validate the wakeword speech WAV file.

    Reads WAKEWORD_TEST_WAV env var.  Converts to compatible format if needed.
    Skips the test if the env var is not set or the file doesn't exist.
    """
    wav_path_str = os.environ.get("WAKEWORD_TEST_WAV")
    if not wav_path_str:
        pytest.skip(_SKIP_MSG)

    wav_path = Path(wav_path_str)
    if not wav_path.exists():
        pytest.fail(f"WAKEWORD_TEST_WAV file not found: {wav_path}")

    return ensure_compatible_wav(wav_path, tmp_path)


@pytest.fixture
def silence_wav(tmp_path: Path) -> Path | None:
    """Resolve silence WAV file (optional).

    Reads WAKEWORD_TEST_SILENCE_WAV env var. Returns None if not set.
    """
    wav_path_str = os.environ.get("WAKEWORD_TEST_SILENCE_WAV")
    if not wav_path_str:
        return None

    wav_path = Path(wav_path_str)
    if not wav_path.exists():
        pytest.fail(f"WAKEWORD_TEST_SILENCE_WAV file not found: {wav_path}")

    return ensure_compatible_wav(wav_path, tmp_path)


@pytest.fixture
def wakeword_keyword() -> str:
    """Keyword to detect, from WAKEWORD_TEST_KEYWORD or default 'ray'."""
    return os.environ.get("WAKEWORD_TEST_KEYWORD", "ray")
