"""Integration tests for SpeechGenerator with real OpenAI LLM + TTS APIs.

Wires up the full pipeline:
ConversationHistory → ContextBuilder → OpenAILLM → OpenAITTS → SpeechGenerator.
Verifies end-to-end streaming behavior with real API calls.

Requires:
    - OPENAI_API_KEY env var

Run:
    OPENAI_API_KEY=... uv run pytest -m requires_api voice_pipeline/tests/generation/ -v
"""

from __future__ import annotations

import os
import time

import pytest

from voice_pipeline.adapters.llm_openai import OpenAILLM
from voice_pipeline.adapters.token_counter import create_token_counter
from voice_pipeline.adapters.tts_openai import OpenAITTS
from voice_pipeline.generator import GeneratorState, SpeechGenerator
from voice_pipeline.history import ConversationHistory, SQLiteStorageBackend
from voice_pipeline.prompt import ContextBuilder

pytestmark = pytest.mark.requires_api

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_POLL_INTERVAL = 0.05  # seconds
_TIMEOUT = 30.0  # seconds — generous for API latency


def _wait_for_state(gen: SpeechGenerator, target: GeneratorState, timeout: float = _TIMEOUT) -> None:
    """Poll until SpeechGenerator reaches the target state or times out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if gen.state == target:
            return
        time.sleep(_POLL_INTERVAL)
    raise TimeoutError(f"Timed out waiting for state {target.value}, current: {gen.state.value}")


def _wait_for_stream_done(gen: SpeechGenerator, timeout: float = _TIMEOUT) -> None:
    """Poll until stream_done becomes True."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if gen.stream_done:
            return
        time.sleep(_POLL_INTERVAL)
    raise TimeoutError("Timed out waiting for stream_done")


def _drain_audio(gen: SpeechGenerator, timeout: float = _TIMEOUT) -> bytes:
    """Collect all audio chunks until stream_done."""
    audio = bytearray()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        chunk = gen.poll_audio()
        if chunk is not None:
            audio.extend(chunk)
        elif gen.stream_done:
            break
        else:
            time.sleep(_POLL_INTERVAL)
    return bytes(audio)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


@pytest.fixture
def speech_generator(openai_api_key: str, monkeypatch: pytest.MonkeyPatch) -> SpeechGenerator:
    """Build a full SpeechGenerator wired to real APIs."""
    monkeypatch.setattr(ContextBuilder, "_MAX_CONTEXT_TOKENS", 2048)
    llm = OpenAILLM(model="gpt-4o-mini", temperature=0.3, max_tokens=128)

    token_counter = create_token_counter(llm.model)
    history = ConversationHistory(SQLiteStorageBackend(":memory:"), token_counter)
    history.new_session("integration-test")
    monkeypatch.setattr(OpenAITTS, "_MODEL", "tts-1")
    monkeypatch.setattr(OpenAITTS, "_VOICE", "alloy")
    tts = OpenAITTS()

    gen = SpeechGenerator(
        llm=llm,
        tts=tts,
        history=history,
        token_counter=token_counter,
        system_prompt="You are Ray, a friendly voice assistant. Keep responses very short (1-2 sentences).",
    )
    yield gen
    gen.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSpeechGeneratorIntegration:
    """End-to-end SpeechGenerator tests with real OpenAI APIs."""

    def test_full_lifecycle(self, speech_generator: SpeechGenerator) -> None:
        """prepare → STREAMING → poll audio → stream_done → get_response_data → IDLE."""
        speech_generator.prepare("Hello, who are you?")

        # Should transition to STREAMING once first TTS chunk arrives
        _wait_for_state(speech_generator, GeneratorState.STREAMING)

        # Drain all audio
        audio = _drain_audio(speech_generator)

        assert len(audio) > 0, "Expected non-empty audio output"
        # PCM 24kHz 16-bit mono: 2 bytes per sample, at least 0.1 sec of audio
        min_bytes = 24000 * 2 * 0.1  # ~4800 bytes
        assert len(audio) > min_bytes, f"Audio too short: {len(audio)} bytes"

        # Text should be available
        text = speech_generator.get_text()
        assert len(text) > 0, "Expected non-empty LLM text"

        # ResponseData should be available
        response_data = speech_generator.get_response_data()
        assert response_data.text == text
        assert len(response_data.audio) == len(audio)

        # Should be back to IDLE
        assert speech_generator.state == GeneratorState.IDLE

    def test_cancel_during_streaming(self, speech_generator: SpeechGenerator) -> None:
        """cancel() during STREAMING should return to IDLE."""
        speech_generator.prepare("Tell me a long story about a robot.")

        _wait_for_state(speech_generator, GeneratorState.STREAMING)

        speech_generator.cancel()
        assert speech_generator.state == GeneratorState.IDLE

    def test_prepare_restart(self, speech_generator: SpeechGenerator) -> None:
        """Calling prepare() again should cancel old run and start new one."""
        speech_generator.prepare("Say hello.")

        # Wait for first run to start streaming
        _wait_for_state(speech_generator, GeneratorState.STREAMING)

        # Restart with a different prompt
        speech_generator.prepare("Say goodbye.")

        # New run should eventually reach STREAMING
        _wait_for_state(speech_generator, GeneratorState.STREAMING)

        audio = _drain_audio(speech_generator)
        assert len(audio) > 0

        text = speech_generator.get_text()
        assert len(text) > 0

        speech_generator.get_response_data()
        assert speech_generator.state == GeneratorState.IDLE

    def test_multiple_sequential_runs(self, speech_generator: SpeechGenerator) -> None:
        """Multiple sequential prepare→consume cycles should work cleanly."""
        for prompt in ["Say hi.", "What is 2+2?", "Say bye."]:
            speech_generator.prepare(prompt)
            _wait_for_state(speech_generator, GeneratorState.STREAMING)

            audio = _drain_audio(speech_generator)
            assert len(audio) > 0, f"No audio for prompt: {prompt}"

            response_data = speech_generator.get_response_data()
            assert len(response_data.text) > 0, f"No text for prompt: {prompt}"
            assert speech_generator.state == GeneratorState.IDLE
