"""Unit tests for the TTS vendor factory."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from voice_pipeline.adapters.tts_elevenlabs import ElevenLabsTTS
from voice_pipeline.adapters.tts_openai import OpenAITTS
from voice_pipeline.wiring import create_tts


class TestCreateTTS:
    def test_default_is_elevenlabs(self) -> None:
        with (
            patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}),
            patch("voice_pipeline.adapters.tts_elevenlabs.ElevenLabs", return_value=MagicMock()),
        ):
            tts = create_tts()

        assert isinstance(tts, ElevenLabsTTS)

    def test_elevenlabs(self) -> None:
        with (
            patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}),
            patch("voice_pipeline.adapters.tts_elevenlabs.ElevenLabs", return_value=MagicMock()),
        ):
            tts = create_tts("elevenlabs")

        assert isinstance(tts, ElevenLabsTTS)

    def test_openai(self) -> None:
        with patch("voice_pipeline.adapters.tts_openai.openai.OpenAI", return_value=MagicMock()):
            tts = create_tts("openai")

        assert isinstance(tts, OpenAITTS)

    def test_unknown_vendor_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown TTS vendor"):
            create_tts("azure")  # type: ignore[arg-type]
