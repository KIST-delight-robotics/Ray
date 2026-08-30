"""Tests for tts.greeting_audio — greeting/farewell audio pre-generation."""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from voice_pipeline import greeting_audio as greeting_audio_module
from voice_pipeline.greeting_audio import (
    _cache_key,
    ensure_greeting_audio,
    synthesize_to_wav,
)
from voice_pipeline.types import TTSStream

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 24000
# 100 samples of silence (16-bit mono)
PCM_SILENCE = struct.pack(f"<{100}h", *([0] * 100))


def _make_tts_mock(
    pcm_chunks: list[bytes] | None = None,
    voice_id: str = "openai|alloy|tts-1|1.0|",
) -> MagicMock:
    """Create a mock ITTS whose synthesize() returns a TTSStream."""
    mock = MagicMock()
    mock.voice_id = voice_id
    mock.output_sample_rate = SAMPLE_RATE
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

        synthesize_to_wav(tts, "hello", out)

        assert out.exists()
        channels, sampwidth, framerate, frames = _read_wav(out)
        assert channels == 1
        assert sampwidth == 2
        assert framerate == SAMPLE_RATE
        assert frames == PCM_SILENCE

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        tts = _make_tts_mock()
        out = tmp_path / "sub" / "dir" / "output.wav"

        synthesize_to_wav(tts, "hello", out)

        assert out.exists()

    def test_collects_multiple_chunks(self, tmp_path: Path) -> None:
        chunk_a = struct.pack("<10h", *([1] * 10))
        chunk_b = struct.pack("<10h", *([2] * 10))
        tts = _make_tts_mock([chunk_a, chunk_b])
        out = tmp_path / "output.wav"

        synthesize_to_wav(tts, "hello", out)

        _, _, _, frames = _read_wav(out)
        assert frames == chunk_a + chunk_b

    def test_closes_stream_on_success(self, tmp_path: Path) -> None:
        tts = _make_tts_mock()
        out = tmp_path / "output.wav"

        synthesize_to_wav(tts, "hello", out)

        tts.synthesize.assert_called_once_with("hello")

    def test_closes_stream_on_error(self, tmp_path: Path) -> None:
        """Stream is closed even if iteration raises."""
        mock = MagicMock()
        mock.output_sample_rate = SAMPLE_RATE

        def _bad_gen():
            yield b"\x00\x00"
            raise RuntimeError("boom")

        stream = TTSStream(_bad_gen())
        mock.synthesize.return_value = stream
        out = tmp_path / "output.wav"

        with pytest.raises(RuntimeError, match="boom"):
            synthesize_to_wav(mock, "hello", out)

        assert stream._closed


# ---------------------------------------------------------------------------
# ensure_greeting_audio
# ---------------------------------------------------------------------------


class TestEnsureGreetingAudio:
    def test_generates_missing_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tts = _make_tts_mock(voice_id="openai|coral|tts-1|1.0|")
        monkeypatch.setattr(greeting_audio_module, "_AUDIO_DIR", str(tmp_path))

        paths = ensure_greeting_audio(tts)

        assert tts.synthesize.call_count == 2
        assert Path(paths.greeting).exists()
        assert Path(paths.farewell).exists()

    def test_skips_existing_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tts = _make_tts_mock(voice_id="openai|coral|tts-1|1.0|")
        monkeypatch.setattr(greeting_audio_module, "_AUDIO_DIR", str(tmp_path))

        # Pre-create the expected files
        for label, text in (
            ("greeting", greeting_audio_module._GREETING_TEXT),
            ("farewell", greeting_audio_module._FAREWELL_TEXT),
        ):
            key = _cache_key(tts, text)
            (tmp_path / f"{label}_{key}.wav").write_bytes(b"fake")

        ensure_greeting_audio(tts)

        tts.synthesize.assert_not_called()

    def test_returns_correct_paths(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tts = _make_tts_mock(voice_id="openai|alloy|tts-1|1.0|")
        monkeypatch.setattr(greeting_audio_module, "_AUDIO_DIR", str(tmp_path))

        paths = ensure_greeting_audio(tts)

        g_key = _cache_key(tts, greeting_audio_module._GREETING_TEXT)
        f_key = _cache_key(tts, greeting_audio_module._FAREWELL_TEXT)
        assert paths.greeting == str(tmp_path / f"greeting_{g_key}.wav")
        assert paths.farewell == str(tmp_path / f"farewell_{f_key}.wav")

    def test_voice_change_triggers_regeneration(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(greeting_audio_module, "_AUDIO_DIR", str(tmp_path))

        # First run with voice "alloy"
        tts_v1 = _make_tts_mock(voice_id="openai|alloy|tts-1|1.0|")
        paths_v1 = ensure_greeting_audio(tts_v1)
        assert tts_v1.synthesize.call_count == 2

        # Second run with voice "coral" — should regenerate
        tts_v2 = _make_tts_mock(voice_id="openai|coral|tts-1|1.0|")
        paths_v2 = ensure_greeting_audio(tts_v2)
        assert tts_v2.synthesize.call_count == 2

        # Different paths, both exist
        assert paths_v1.greeting != paths_v2.greeting
        assert Path(paths_v1.greeting).exists()
        assert Path(paths_v2.greeting).exists()

    def test_text_change_triggers_regeneration(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tts = _make_tts_mock(voice_id="openai|alloy|tts-1|1.0|")
        monkeypatch.setattr(greeting_audio_module, "_AUDIO_DIR", str(tmp_path))

        monkeypatch.setattr(greeting_audio_module, "_GREETING_TEXT", "안녕!")
        paths_v1 = ensure_greeting_audio(tts)
        assert tts.synthesize.call_count == 2

        tts.synthesize.reset_mock()
        monkeypatch.setattr(greeting_audio_module, "_GREETING_TEXT", "반가워요!")
        paths_v2 = ensure_greeting_audio(tts)
        # greeting regenerated, farewell skipped (same text)
        assert tts.synthesize.call_count == 1
        assert paths_v1.greeting != paths_v2.greeting
        assert paths_v1.farewell == paths_v2.farewell

    def test_tts_error_falls_back(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tts = MagicMock()
        tts.voice_id = "openai|alloy|tts-1|1.0|"
        tts.output_sample_rate = SAMPLE_RATE
        tts.synthesize.side_effect = RuntimeError("API down")
        monkeypatch.setattr(greeting_audio_module, "_AUDIO_DIR", str(tmp_path))

        paths = ensure_greeting_audio(tts)

        assert paths.greeting == greeting_audio_module._FALLBACK_GREETING_PATH
        assert paths.farewell == greeting_audio_module._FALLBACK_FAREWELL_PATH

    def test_partial_failure_falls_back_individually(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Greeting succeeds, farewell fails → only farewell uses fallback."""
        call_count = 0

        def _fail_on_second(text: str) -> TTSStream:  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return TTSStream(c for c in [PCM_SILENCE])
            raise RuntimeError("API down")

        tts = MagicMock()
        tts.voice_id = "openai|alloy|tts-1|1.0|"
        tts.output_sample_rate = SAMPLE_RATE
        tts.synthesize.side_effect = _fail_on_second
        monkeypatch.setattr(greeting_audio_module, "_AUDIO_DIR", str(tmp_path))

        paths = ensure_greeting_audio(tts)

        assert Path(paths.greeting).exists()
        assert paths.greeting != greeting_audio_module._FALLBACK_GREETING_PATH
        assert paths.farewell == greeting_audio_module._FALLBACK_FAREWELL_PATH
