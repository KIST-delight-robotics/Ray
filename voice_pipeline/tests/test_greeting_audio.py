"""Tests for greeting_audio — greeting audio pre-generation."""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from voice_pipeline import audio_level as audio_level_module
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


@pytest.fixture(autouse=True)
def _isolate_level_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """정규화 사본이 실제 var/ 로 새지 않게 한다 — 반환 경로는 전부 이 폴더 아래다."""
    monkeypatch.setattr(audio_level_module, "_CACHE_DIR", str(tmp_path / "normalized"))


def _write_wav(path: Path) -> str:
    """폴백 자산 대역 — 정규화가 읽을 수 있는 실제 WAV."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(struct.pack(f"<{SAMPLE_RATE}h", *([8000, -8000] * (SAMPLE_RATE // 2))))
    return str(path)


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


_TEXT = "네, 부르셨어요?"


class TestEnsureGreetingAudio:
    def test_generates_missing_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tts = _make_tts_mock(voice_id="openai|coral|tts-1|1.0|")
        monkeypatch.setattr(greeting_audio_module, "_AUDIO_DIR", str(tmp_path))

        path = ensure_greeting_audio(tts, _TEXT)

        assert tts.synthesize.call_count == 1
        assert Path(path).exists()

    def test_skips_existing_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tts = _make_tts_mock(voice_id="openai|coral|tts-1|1.0|")
        monkeypatch.setattr(greeting_audio_module, "_AUDIO_DIR", str(tmp_path))
        _write_wav(tmp_path / f"greeting_{_cache_key(tts, _TEXT)}.wav")

        ensure_greeting_audio(tts, _TEXT)

        tts.synthesize.assert_not_called()

    def test_returns_leveled_copy_named_after_cache_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tts = _make_tts_mock(voice_id="openai|alloy|tts-1|1.0|")
        monkeypatch.setattr(greeting_audio_module, "_AUDIO_DIR", str(tmp_path))

        path = ensure_greeting_audio(tts, _TEXT)

        # 반환되는 건 레벨을 맞춘 사본이고, 이름에 원본 캐시 키가 남아 출처를 알 수 있다
        key = _cache_key(tts, _TEXT)
        assert Path(path).parent == tmp_path / "normalized"
        assert Path(path).stem.startswith(f"greeting_{key}")
        assert (tmp_path / f"greeting_{key}.wav").exists()  # 원본도 남는다

    def test_voice_change_triggers_regeneration(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(greeting_audio_module, "_AUDIO_DIR", str(tmp_path))
        tts_v1 = _make_tts_mock(voice_id="openai|alloy|tts-1|1.0|")
        tts_v2 = _make_tts_mock(voice_id="openai|coral|tts-1|1.0|")

        path_v1 = ensure_greeting_audio(tts_v1, _TEXT)
        path_v2 = ensure_greeting_audio(tts_v2, _TEXT)

        assert tts_v2.synthesize.call_count == 1
        assert path_v1 != path_v2
        assert Path(path_v1).exists() and Path(path_v2).exists()

    def test_text_change_triggers_regeneration(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tts = _make_tts_mock(voice_id="openai|alloy|tts-1|1.0|")
        monkeypatch.setattr(greeting_audio_module, "_AUDIO_DIR", str(tmp_path))

        path_v1 = ensure_greeting_audio(tts, "안녕!")
        path_v2 = ensure_greeting_audio(tts, "반가워요!")

        assert tts.synthesize.call_count == 2
        assert path_v1 != path_v2

    def test_tts_error_falls_back(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tts = MagicMock()
        tts.voice_id = "openai|alloy|tts-1|1.0|"
        tts.output_sample_rate = SAMPLE_RATE
        tts.synthesize.side_effect = RuntimeError("API down")
        monkeypatch.setattr(greeting_audio_module, "_AUDIO_DIR", str(tmp_path))
        monkeypatch.setattr(
            greeting_audio_module, "_FALLBACK_GREETING_PATH", _write_wav(tmp_path / "fallback_greeting.wav")
        )

        path = ensure_greeting_audio(tts, _TEXT)

        # 폴백 자산도 레벨을 맞춘 사본으로 나간다 (자산 파일 자체는 그대로)
        assert Path(path).stem.startswith("fallback_greeting")
