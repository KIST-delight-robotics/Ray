"""Stress tests for ElevenLabsTTS with real ElevenLabs API.

Requires ELEVENLABS_API_KEY environment variable.
Low subscription tiers cap concurrent requests — these tests stay sequential.
"""

from __future__ import annotations

import pytest

from voice_pipeline.adapters.tts_elevenlabs import ElevenLabsTTS

pytestmark = pytest.mark.requires_api


@pytest.fixture
def tts(elevenlabs_api_key: str) -> ElevenLabsTTS:  # noqa: ARG001
    """Create an ElevenLabsTTS with default config."""
    return ElevenLabsTTS()


class TestRapidSequentialCalls:
    def test_five_back_to_back_calls(self, tts: ElevenLabsTTS) -> None:
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
            assert len(stream.timestamps) > 0, f"Call {i + 1} produced no timestamps"


class TestPartialConsumption:
    def test_consume_one_chunk_then_close(self, tts: ElevenLabsTTS) -> None:
        """Consume only the first chunk and close — no exception, no leak."""
        stream = tts.synthesize("Write a long paragraph about the ocean and its creatures.")
        first = next(stream)
        stream.close()

        assert len(first) > 0

    def test_repeated_partial_consumption(self, tts: ElevenLabsTTS) -> None:
        """Multiple partial-consume cycles in a row."""
        for _ in range(3):
            stream = tts.synthesize("Tell me a long story about a brave knight.")
            next(stream)
            stream.close()


class TestLongText:
    def test_synthesize_longer_paragraph(self, tts: ElevenLabsTTS) -> None:
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
        assert [ts.word for ts in stream.timestamps] == text.split()
