"""Unit tests for ElevenLabsTTS (mocked ElevenLabs client)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import httpx
import pytest
from elevenlabs.core.api_error import ApiError
from elevenlabs.types import VoiceSettings

from voice_pipeline.adapters.tts_elevenlabs import ElevenLabsTTS, _alignment_to_word_timestamps
from voice_pipeline.tests.adapters.conftest import create_mock_elevenlabs_client, make_elevenlabs_chunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_CLASS_VAR_MAP = {
    "voice_id": "_VOICE_ID",
    "model": "_MODEL",
    "voice_settings": "_VOICE_SETTINGS",
}


def _build_tts(
    mock_client: MagicMock,
    monkeypatch: pytest.MonkeyPatch | None = None,
    **kwargs,
) -> ElevenLabsTTS:
    """Build an ElevenLabsTTS with a mock client injected.

    Overrides (voice_id/model/voice_settings) are translated to class var
    monkeypatch — caller must pass ``monkeypatch`` fixture when providing any.
    """
    if kwargs and monkeypatch is None:
        raise TypeError("monkeypatch fixture required when overrides provided")
    for key, value in kwargs.items():
        if key not in _CLASS_VAR_MAP:
            raise TypeError(f"Unknown override: {key}")
        monkeypatch.setattr(ElevenLabsTTS, _CLASS_VAR_MAP[key], value)
    with (
        patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}),
        patch("voice_pipeline.adapters.tts_elevenlabs.ElevenLabs", return_value=mock_client),
    ):
        return ElevenLabsTTS()


def _alignment_for(text: str, sec_per_char: float = 0.1) -> dict[str, list]:
    """Uniform per-character alignment for *text*."""
    return {
        "characters": list(text),
        "starts": [i * sec_per_char for i in range(len(text))],
        "ends": [(i + 1) * sec_per_char for i in range(len(text))],
    }


# ---------------------------------------------------------------------------
# TestSynthesize
# ---------------------------------------------------------------------------


class TestSynthesize:
    def test_yields_decoded_pcm_chunks(self) -> None:
        pcm = [b"\x00\x01" * 100, b"\x02\x03" * 100]
        client = create_mock_elevenlabs_client([make_elevenlabs_chunk(p) for p in pcm])
        tts = _build_tts(client)

        stream = tts.synthesize("Hello world")
        collected = list(stream)

        assert collected == pcm

    def test_collects_audio_correctly(self) -> None:
        pcm = [b"\x01\x02", b"\x03\x04", b"\x05\x06"]
        client = create_mock_elevenlabs_client([make_elevenlabs_chunk(p) for p in pcm])
        tts = _build_tts(client)

        stream = tts.synthesize("Hello world")
        list(stream)

        assert stream.audio == b"\x01\x02\x03\x04\x05\x06"

    def test_empty_audio_chunk_not_yielded(self) -> None:
        """Alignment-only chunks (no audio) must not pollute the audio stream."""
        chunks = [
            make_elevenlabs_chunk(b"", **_alignment_for("Hi")),
            make_elevenlabs_chunk(b"\xaa\xbb"),
        ]
        client = create_mock_elevenlabs_client(chunks)
        tts = _build_tts(client)

        stream = tts.synthesize("Hi")
        collected = list(stream)

        assert collected == [b"\xaa\xbb"]
        assert [ts.word for ts in stream.timestamps] == ["Hi"]


# ---------------------------------------------------------------------------
# TestAlignmentToWordTimestamps — pure function, no mocks
# ---------------------------------------------------------------------------


class TestAlignmentToWordTimestamps:
    def test_basic_two_words(self) -> None:
        ts = _alignment_to_word_timestamps(
            list("hi yo"),
            [0.0, 0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4, 0.5],
        )

        assert [(t.word, t.start_sec, t.end_sec) for t in ts] == [("hi", 0.0, 0.2), ("yo", 0.3, 0.5)]

    def test_matches_text_split(self) -> None:
        text = "Hello world, this is Ray."
        a = _alignment_for(text)
        ts = _alignment_to_word_timestamps(a["characters"], a["starts"], a["ends"])

        assert [t.word for t in ts] == text.split()

    def test_multiple_spaces_and_newlines(self) -> None:
        text = "  Hello \n\t world  "
        a = _alignment_for(text)
        ts = _alignment_to_word_timestamps(a["characters"], a["starts"], a["ends"])

        assert [t.word for t in ts] == ["Hello", "world"]

    def test_start_times_monotonic(self) -> None:
        text = "one two three"
        a = _alignment_for(text)
        ts = _alignment_to_word_timestamps(a["characters"], a["starts"], a["ends"])

        starts = [t.start_sec for t in ts]
        assert starts == sorted(starts)

    def test_empty_returns_empty(self) -> None:
        assert _alignment_to_word_timestamps([], [], []) == ()

    def test_whitespace_only_returns_empty(self) -> None:
        a = _alignment_for("  \n ")
        assert _alignment_to_word_timestamps(a["characters"], a["starts"], a["ends"]) == ()

    def test_length_mismatch_truncates_without_raise(self) -> None:
        """Shorter time lists truncate the character list, never raise."""
        ts = _alignment_to_word_timestamps(list("ab cd"), [0.0, 0.1], [0.1, 0.2])

        assert [t.word for t in ts] == ["ab"]

    def test_inverted_times_clamped(self) -> None:
        """end < start is clamped so WordTimestamp validation passes."""
        ts = _alignment_to_word_timestamps(["a"], [0.5], [0.2])

        assert ts[0].start_sec == 0.5
        assert ts[0].end_sec == 0.5

    def test_negative_times_clamped(self) -> None:
        ts = _alignment_to_word_timestamps(["a"], [-0.3], [-0.1])

        assert ts[0].start_sec == 0.0
        assert ts[0].end_sec == 0.0


# ---------------------------------------------------------------------------
# TestTimestamps — through the stream
# ---------------------------------------------------------------------------


class TestTimestamps:
    def test_word_spanning_chunk_boundary(self) -> None:
        """Alignment split mid-word across chunks must merge into one word."""
        chunks = [
            make_elevenlabs_chunk(b"\x01", characters=list("Hel"), starts=[0.0, 0.1, 0.2], ends=[0.1, 0.2, 0.3]),
            make_elevenlabs_chunk(
                b"\x02",
                characters=list("lo go"),
                starts=[0.3, 0.4, 0.5, 0.6, 0.7],
                ends=[0.4, 0.5, 0.6, 0.7, 0.8],
            ),
        ]
        client = create_mock_elevenlabs_client(chunks)
        tts = _build_tts(client)

        stream = tts.synthesize("Hello go")
        list(stream)

        assert [(t.word, t.start_sec, t.end_sec) for t in stream.timestamps] == [
            ("Hello", 0.0, 0.5),
            ("go", 0.6, 0.8),
        ]

    def test_no_alignment_returns_empty(self) -> None:
        client = create_mock_elevenlabs_client([make_elevenlabs_chunk(b"\x01")])
        tts = _build_tts(client)

        stream = tts.synthesize("Hello")
        list(stream)

        assert stream.timestamps == ()

    def test_timestamps_before_consumption_raises(self) -> None:
        client = create_mock_elevenlabs_client([make_elevenlabs_chunk(b"\x01")])
        tts = _build_tts(client)

        stream = tts.synthesize("Hello")

        with pytest.raises(RuntimeError):
            _ = stream.timestamps

    def test_result_property(self) -> None:
        client = create_mock_elevenlabs_client([make_elevenlabs_chunk(b"\x01\x02", **_alignment_for("Hi"))])
        tts = _build_tts(client)

        stream = tts.synthesize("Hi")
        list(stream)

        result = stream.result
        assert result.audio == b"\x01\x02"
        assert [t.word for t in result.timestamps] == ["Hi"]


# ---------------------------------------------------------------------------
# TestMissingApiKey
# ---------------------------------------------------------------------------


class TestMissingApiKey:
    def test_constructor_fails_fast_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SDK alone would defer the 401 to the first request — we fail at construction."""
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
            ElevenLabsTTS()

    def test_empty_key_also_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ELEVENLABS_API_KEY", "")

        with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
            ElevenLabsTTS()


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_empty_text_raises(self) -> None:
        tts = _build_tts(create_mock_elevenlabs_client())

        with pytest.raises(RuntimeError, match="empty"):
            tts.synthesize("")

    def test_whitespace_only_raises(self) -> None:
        tts = _build_tts(create_mock_elevenlabs_client())

        with pytest.raises(RuntimeError, match="empty"):
            tts.synthesize("   \n\t  ")

    def test_too_long_text_raises(self) -> None:
        tts = _build_tts(create_mock_elevenlabs_client())

        with pytest.raises(RuntimeError, match="4096"):
            tts.synthesize("a" * 4097)


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_api_error_surfaces_at_first_next(self) -> None:
        """SDK is lazy: HTTP errors raise during iteration, not at synthesize()."""
        client = create_mock_elevenlabs_client(call_error=ApiError(status_code=401, body="unauthorized"))
        tts = _build_tts(client)

        stream = tts.synthesize("Hello")  # must not raise

        with pytest.raises(RuntimeError, match="401"):
            next(stream)

    def test_mid_stream_api_error_after_partial_chunks(self) -> None:
        client = create_mock_elevenlabs_client(
            [make_elevenlabs_chunk(b"\x01\x02")],
            streaming_error=ApiError(status_code=500, body="server error"),
        )
        tts = _build_tts(client)

        stream = tts.synthesize("Hello")
        assert next(stream) == b"\x01\x02"

        with pytest.raises(RuntimeError, match="500"):
            next(stream)

    def test_timeout_wrapped(self) -> None:
        client = create_mock_elevenlabs_client(call_error=httpx.ReadTimeout("timed out"))
        tts = _build_tts(client)

        stream = tts.synthesize("Hello")

        with pytest.raises(RuntimeError, match="timeout"):
            next(stream)

    def test_unexpected_error_wrapped(self) -> None:
        client = create_mock_elevenlabs_client(call_error=ValueError("boom"))
        tts = _build_tts(client)

        stream = tts.synthesize("Hello")

        with pytest.raises(RuntimeError, match="boom"):
            next(stream)


# ---------------------------------------------------------------------------
# TestStreamCleanup
# ---------------------------------------------------------------------------


class TestStreamCleanup:
    def test_sdk_stream_closed_after_full_iteration(self) -> None:
        client = create_mock_elevenlabs_client([make_elevenlabs_chunk(b"\x01")])
        tts = _build_tts(client)

        stream = tts.synthesize("Hello")
        list(stream)

        assert client._stream_state["closed"]

    def test_sdk_stream_closed_on_partial_close(self) -> None:
        client = create_mock_elevenlabs_client([make_elevenlabs_chunk(b"\x01"), make_elevenlabs_chunk(b"\x02")])
        tts = _build_tts(client)

        stream = tts.synthesize("Hello")
        next(stream)
        stream.close()

        assert client._stream_state["closed"]

    def test_sdk_stream_closed_on_error(self) -> None:
        client = create_mock_elevenlabs_client(call_error=ApiError(status_code=500, body="err"))
        tts = _build_tts(client)

        stream = tts.synthesize("Hello")
        with pytest.raises(RuntimeError):
            next(stream)

        assert client._stream_state["closed"]

    def test_close_before_first_next_is_noop(self) -> None:
        """No HTTP request was issued yet — close must not raise or start it."""
        client = create_mock_elevenlabs_client([make_elevenlabs_chunk(b"\x01")])
        tts = _build_tts(client)

        stream = tts.synthesize("Hello")
        stream.close()

        assert not client._stream_state["started"]

    def test_double_close_is_idempotent(self) -> None:
        client = create_mock_elevenlabs_client([make_elevenlabs_chunk(b"\x01")])
        tts = _build_tts(client)

        stream = tts.synthesize("Hello")
        list(stream)
        stream.close()
        stream.close()


# ---------------------------------------------------------------------------
# TestRequestParams
# ---------------------------------------------------------------------------


class TestRequestParams:
    def test_default_params_passed(self) -> None:
        client = create_mock_elevenlabs_client([make_elevenlabs_chunk(b"\x01")])
        tts = _build_tts(client)

        stream = tts.synthesize("Hello")
        list(stream)

        call = client.text_to_speech.stream_with_timestamps.call_args
        assert call.args == ("EXAVITQu4vr4xnSDxMaL",)
        assert call.kwargs["text"] == "Hello"
        assert call.kwargs["model_id"] == "eleven_flash_v2_5"
        assert call.kwargs["output_format"] == "pcm_24000"
        assert call.kwargs["request_options"] == {"max_retries": 2}
        assert "voice_settings" not in call.kwargs

    def test_output_format_derived_from_sample_rate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """output_format must follow OUTPUT_SAMPLE_RATE, not a separate constant."""
        client = create_mock_elevenlabs_client([make_elevenlabs_chunk(b"\x01")])
        monkeypatch.setattr(ElevenLabsTTS, "OUTPUT_SAMPLE_RATE", 16000)
        tts = _build_tts(client)

        stream = tts.synthesize("Hello")
        list(stream)

        call = client.text_to_speech.stream_with_timestamps.call_args
        assert call.kwargs["output_format"] == "pcm_16000"

    def test_voice_settings_passed_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = create_mock_elevenlabs_client([make_elevenlabs_chunk(b"\x01")])
        tts = _build_tts(client, monkeypatch, voice_settings={"stability": 0.5})

        stream = tts.synthesize("Hello")
        list(stream)

        settings = client.text_to_speech.stream_with_timestamps.call_args.kwargs["voice_settings"]
        assert isinstance(settings, VoiceSettings)
        assert settings.stability == 0.5

    def test_custom_voice_and_model_propagated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = create_mock_elevenlabs_client([make_elevenlabs_chunk(b"\x01")])
        tts = _build_tts(client, monkeypatch, voice_id="custom-voice", model="eleven_turbo_v2_5")

        stream = tts.synthesize("Hello")
        list(stream)

        call = client.text_to_speech.stream_with_timestamps.call_args
        assert call.args == ("custom-voice",)
        assert call.kwargs["model_id"] == "eleven_turbo_v2_5"


# ---------------------------------------------------------------------------
# TestProperties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_output_sample_rate(self) -> None:
        tts = _build_tts(create_mock_elevenlabs_client())

        assert tts.output_sample_rate == 24000

    def test_model_name(self) -> None:
        tts = _build_tts(create_mock_elevenlabs_client())

        assert tts.model_name == "eleven_flash_v2_5"

    def test_voice_id_format(self) -> None:
        tts = _build_tts(create_mock_elevenlabs_client())

        assert tts.voice_id.startswith("elevenlabs|")
        assert "EXAVITQu4vr4xnSDxMaL" in tts.voice_id
        assert "eleven_flash_v2_5" in tts.voice_id

    def test_voice_id_changes_with_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        plain = _build_tts(create_mock_elevenlabs_client())
        plain_id = plain.voice_id

        tuned = _build_tts(create_mock_elevenlabs_client(), monkeypatch, voice_settings={"stability": 0.5})

        assert tuned.voice_id != plain_id
