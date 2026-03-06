"""Shared fixtures for cross-module integration tests."""

from __future__ import annotations

import os
import wave
from dataclasses import dataclass
from pathlib import Path

import pytest

from voice_pipeline.core.config import AudioConfig

# ---------------------------------------------------------------------------
# WAV helpers (reusable across cross-module tests)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
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


def make_silence_frames(
    frame_bytes: int, duration_sec: float, frame_duration_ms: int = 30
) -> list[bytes]:
    """Generate silence frames to simulate continued mic input after speech."""
    n_frames = int(duration_sec / (frame_duration_ms / 1000))
    silence = b"\x00" * frame_bytes
    return [silence] * n_frames


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def openai_api_key() -> str:
    """Skip if OPENAI_API_KEY not set."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


@pytest.fixture
def google_credentials() -> str:
    """Skip if GOOGLE_APPLICATION_CREDENTIALS not set."""
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds:
        pytest.skip("GOOGLE_APPLICATION_CREDENTIALS not set")
    if not Path(creds).exists():
        pytest.fail(f"GOOGLE_APPLICATION_CREDENTIALS file not found: {creds}")
    return creds
