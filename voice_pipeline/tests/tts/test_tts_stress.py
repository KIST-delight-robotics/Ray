"""Stress tests for OpenAITTS with real OpenAI API.

Requires OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import pytest

from voice_pipeline.core.config import TTSConfig
from voice_pipeline.tts.tts import OpenAITTS

pytestmark = pytest.mark.requires_api


@pytest.fixture
def tts(openai_api_key: str) -> OpenAITTS:  # noqa: ARG001
    """Create an OpenAITTS with default config."""
    return OpenAITTS(TTSConfig())


class TestRapidSequentialCalls:
    def test_five_back_to_back_calls(self, tts: OpenAITTS) -> None:
        phrases = [
            "Hello world",
            "How are you today?",
            "The weather is nice.",
            "I like programming.",
            "Goodbye for now.",
        ]
        for i, phrase in enumerate(phrases):
            stream = tts.synthesize(phrase)
            list(stream)
            assert len(stream.audio) > 0, f"Call {i + 1} produced empty audio"


class TestPartialConsumption:
    def test_consume_one_chunk_then_close(self, tts: OpenAITTS) -> None:
        """Consume only the first chunk and close — no exception, no leak."""
        stream = tts.synthesize("Write a long paragraph about the ocean and its creatures.")
        first = next(stream)
        stream.close()

        assert len(first) > 0

    def test_repeated_partial_consumption(self, tts: OpenAITTS) -> None:
        """Multiple partial-consume cycles in a row."""
        for _ in range(3):
            stream = tts.synthesize("Tell me a long story about a brave knight.")
            next(stream)
            stream.close()


class TestLongText:
    def test_synthesize_longer_paragraph(self, tts: OpenAITTS) -> None:
        text = (
            "The quick brown fox jumped over the lazy dog. "
            "This is a longer paragraph to test whether the TTS system "
            "can handle more substantial amounts of text without issues. "
            "It should produce a reasonable amount of audio data."
        )
        stream = tts.synthesize(text)
        list(stream)

        # Longer text should produce more audio
        assert len(stream.audio) > 48000  # at least ~1 sec of audio
