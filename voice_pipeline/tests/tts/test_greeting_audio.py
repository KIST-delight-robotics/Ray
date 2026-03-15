"""Tests for tts.greeting_audio — greeting/farewell audio pre-generation."""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from voice_pipeline.core.config import GreetingAudioConfig, TTSConfig
from voice_pipeline.core.types import TTSStream
from voice_pipeline.tts.greeting_audio import (
    _cache_key,
    ensure_greeting_audio,
    synthesize_to_wav,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 24000
# 100 samples of silence (16-bit mono)
PCM_SILENCE = struct.pack(f"<{100}h", *([0] * 100))


def _make_tts_mock(pcm_chunks: list[bytes] | None = None) -> MagicMock:
    """Create a mock ITTS whose synthesize() returns a TTSStream."""
    mock = MagicMock()
    chunks = pcm_chunks if pcm_chunks is not None else [PCM_SILENCE]

    def _synthesize(text: str) -> TTSStream:  # noqa: ARG001
        gen = (c for c in chunks)
        return TTSStream(gen)

    mock.synthesize.side_effect = _synthesize
    return mock


def _read_wav(path: Path) -> tuple[int, int, int, bytes]:
    """Read a WAV file, return (channels, sampwidth, framerate, frames)."""
    with wave.open(str(path), "rb") as wf:
        return (
            wf.getnchannels(),
            wf.getsampwidth(),
            wf.getframerate(),
            wf.readframes(wf.getnframes()),
        )


# ---------------------------------------------------------------------------
# synthesize_to_wav
# ---------------------------------------------------------------------------


class TestSynthesizeToWav:
    def test_creates_wav_file(self, tmp_path: Path) -> None:
        tts = _make_tts_mock()
        out = tmp_path / "output.wav"

        synthesize_to_wav(tts, "hello", out, SAMPLE_RATE)

        assert out.exists()
        channels, sampwidth, framerate, frames = _read_wav(out)
        assert channels == 1
        assert sampwidth == 2
        assert framerate == SAMPLE_RATE
        assert frames == PCM_SILENCE

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        tts = _make_tts_mock()
        out = tmp_path / "sub" / "dir" / "output.wav"

        synthesize_to_wav(tts, "hello", out, SAMPLE_RATE)

        assert out.exists()

    def test_collects_multiple_chunks(self, tmp_path: Path) -> None:
        chunk_a = struct.pack("<10h", *([1] * 10))
        chunk_b = struct.pack("<10h", *([2] * 10))
        tts = _make_tts_mock([chunk_a, chunk_b])
        out = tmp_path / "output.wav"

        synthesize_to_wav(tts, "hello", out, SAMPLE_RATE)

        _, _, _, frames = _read_wav(out)
        assert frames == chunk_a + chunk_b

    def test_closes_stream_on_success(self, tmp_path: Path) -> None:
        tts = _make_tts_mock()
        out = tmp_path / "output.wav"

        synthesize_to_wav(tts, "hello", out, SAMPLE_RATE)

        tts.synthesize.assert_called_once_with("hello")

    def test_closes_stream_on_error(self, tmp_path: Path) -> None:
        """Stream is closed even if iteration raises."""
        mock = MagicMock()

        def _bad_gen():
            yield b"\x00\x00"
            raise RuntimeError("boom")

        stream = TTSStream(_bad_gen())
        mock.synthesize.return_value = stream
        out = tmp_path / "output.wav"

        with pytest.raises(RuntimeError, match="boom"):
            synthesize_to_wav(mock, "hello", out, SAMPLE_RATE)

        assert stream._closed


# ---------------------------------------------------------------------------
# ensure_greeting_audio
# ---------------------------------------------------------------------------


class TestEnsureGreetingAudio:
    def test_generates_missing_files(self, tmp_path: Path) -> None:
        tts = _make_tts_mock()
        tts_config = TTSConfig(voice="coral", model="tts-1")
        greeting_config = GreetingAudioConfig(audio_dir=str(tmp_path))

        paths = ensure_greeting_audio(tts, tts_config, greeting_config)

        assert tts.synthesize.call_count == 2
        assert Path(paths.greeting).exists()
        assert Path(paths.farewell).exists()

    def test_skips_existing_files(self, tmp_path: Path) -> None:
        tts = _make_tts_mock()
        tts_config = TTSConfig(voice="coral", model="tts-1")
        greeting_config = GreetingAudioConfig(audio_dir=str(tmp_path))

        # Pre-create the expected files
        for label, text in (
            ("greeting", greeting_config.greeting_text),
            ("farewell", greeting_config.farewell_text),
        ):
            key = _cache_key(tts_config, text)
            (tmp_path / f"{label}_{key}.wav").write_bytes(b"fake")

        ensure_greeting_audio(tts, tts_config, greeting_config)

        tts.synthesize.assert_not_called()

    def test_returns_correct_paths(self, tmp_path: Path) -> None:
        tts = _make_tts_mock()
        tts_config = TTSConfig(voice="alloy", model="tts-1")
        greeting_config = GreetingAudioConfig(audio_dir=str(tmp_path))

        paths = ensure_greeting_audio(tts, tts_config, greeting_config)

        g_key = _cache_key(tts_config, greeting_config.greeting_text)
        f_key = _cache_key(tts_config, greeting_config.farewell_text)
        assert paths.greeting == str(tmp_path / f"greeting_{g_key}.wav")
        assert paths.farewell == str(tmp_path / f"farewell_{f_key}.wav")

    def test_config_change_triggers_regeneration(self, tmp_path: Path) -> None:
        tts = _make_tts_mock()
        greeting_config = GreetingAudioConfig(audio_dir=str(tmp_path))

        # First run with voice "alloy"
        config_v1 = TTSConfig(voice="alloy", model="tts-1")
        paths_v1 = ensure_greeting_audio(tts, config_v1, greeting_config)
        assert tts.synthesize.call_count == 2

        # Second run with voice "coral" — should regenerate
        tts.synthesize.reset_mock()
        config_v2 = TTSConfig(voice="coral", model="tts-1")
        paths_v2 = ensure_greeting_audio(tts, config_v2, greeting_config)
        assert tts.synthesize.call_count == 2

        # Different paths, both exist
        assert paths_v1.greeting != paths_v2.greeting
        assert Path(paths_v1.greeting).exists()
        assert Path(paths_v2.greeting).exists()

    def test_text_change_triggers_regeneration(self, tmp_path: Path) -> None:
        tts = _make_tts_mock()
        tts_config = TTSConfig(voice="alloy", model="tts-1")

        config_v1 = GreetingAudioConfig(
            audio_dir=str(tmp_path), greeting_text="안녕!"
        )
        paths_v1 = ensure_greeting_audio(tts, tts_config, config_v1)
        assert tts.synthesize.call_count == 2

        tts.synthesize.reset_mock()
        config_v2 = GreetingAudioConfig(
            audio_dir=str(tmp_path), greeting_text="반가워요!"
        )
        paths_v2 = ensure_greeting_audio(tts, tts_config, config_v2)
        # greeting regenerated, farewell skipped (same text)
        assert tts.synthesize.call_count == 1
        assert paths_v1.greeting != paths_v2.greeting
        assert paths_v1.farewell == paths_v2.farewell

    def test_tts_error_falls_back(self, tmp_path: Path) -> None:
        tts = MagicMock()
        tts.synthesize.side_effect = RuntimeError("API down")
        tts_config = TTSConfig(voice="alloy", model="tts-1")
        greeting_config = GreetingAudioConfig(audio_dir=str(tmp_path))

        paths = ensure_greeting_audio(tts, tts_config, greeting_config)

        assert paths.greeting == greeting_config.fallback_greeting_path
        assert paths.farewell == greeting_config.fallback_farewell_path

    def test_partial_failure_falls_back_individually(self, tmp_path: Path) -> None:
        """Greeting succeeds, farewell fails → only farewell uses fallback."""
        call_count = 0

        def _fail_on_second(text: str) -> TTSStream:  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return TTSStream((c for c in [PCM_SILENCE]))
            raise RuntimeError("API down")

        tts = MagicMock()
        tts.synthesize.side_effect = _fail_on_second
        tts_config = TTSConfig(voice="alloy", model="tts-1")
        greeting_config = GreetingAudioConfig(audio_dir=str(tmp_path))

        paths = ensure_greeting_audio(tts, tts_config, greeting_config)

        assert Path(paths.greeting).exists()
        assert paths.greeting != greeting_config.fallback_greeting_path
        assert paths.farewell == greeting_config.fallback_farewell_path
