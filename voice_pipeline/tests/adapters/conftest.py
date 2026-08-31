"""adapters 테스트 공용 픽스처/헬퍼.

WAV 유틸(ASR·wakeword 통합 테스트), CppBridge 픽스처, OpenAI LLM/TTS mock, ElevenLabs mock.
통합 테스트 입력은 환경변수로 받는다: ASR_TEST_WAV, WAKEWORD_TEST_WAV, OPENAI_API_KEY, ELEVENLABS_API_KEY.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import os
import struct
import subprocess
import threading
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from elevenlabs.types import StreamingAudioChunkWithTimestampsResponse

from voice_pipeline.adapters.cpp_bridge import CppBridge


def ensure_16k_wav(path: Path, tmp_path: Path) -> Path:
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

_WAKEWORD_SKIP_MSG = "WAKEWORD_TEST_WAV not set — provide a path to a speech WAV file"


@pytest.fixture
def wakeword_wav(tmp_path: Path) -> Path:
    """Resolve and validate the wakeword speech WAV file.

    Reads WAKEWORD_TEST_WAV env var.  Converts to compatible format if needed.
    Skips the test if the env var is not set or the file doesn't exist.
    """
    wav_path_str = os.environ.get("WAKEWORD_TEST_WAV")
    if not wav_path_str:
        pytest.skip(_WAKEWORD_SKIP_MSG)

    wav_path = Path(wav_path_str)
    if not wav_path.exists():
        pytest.fail(f"WAKEWORD_TEST_WAV file not found: {wav_path}")

    return ensure_16k_wav(wav_path, tmp_path)


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

    return ensure_16k_wav(wav_path, tmp_path)


@pytest.fixture
def wakeword_keyword() -> str:
    """Keyword to detect, from WAKEWORD_TEST_KEYWORD or default 'ray'."""
    return os.environ.get("WAKEWORD_TEST_KEYWORD", "ray")


_SAMPLE_RATE_MIN = 8000
_SAMPLE_RATE_MAX = 48000


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
    """Return a WAV file compatible with Google STT (mono, 16-bit, 8-48kHz).

    Returns the original path if already compatible, otherwise resamples
    via ffmpeg into tmp_path.
    """
    info = read_wav_info(path)
    needs_resample = not (_SAMPLE_RATE_MIN <= info.sample_rate <= _SAMPLE_RATE_MAX)
    needs_convert = info.channels != 1 or info.sample_width != 2

    if not needs_resample and not needs_convert:
        return path

    target_rate = 16000 if needs_resample else info.sample_rate
    out = tmp_path / f"converted_{path.name}"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-ar",
            str(target_rate),
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


def make_asr_for_wav(info: WavInfo, language_code: str = "en-US"):  # noqa: ARG001
    """Build a GoogleCloudASR. ``info`` is accepted for symmetry with other helpers."""
    from voice_pipeline.adapters.asr_google import GoogleCloudASR

    return GoogleCloudASR(language_code=language_code)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ASR_SKIP_MSG = "ASR_TEST_WAV not set — provide a path to a speech WAV file"


@pytest.fixture
def asr_wav(tmp_path: Path) -> Path:
    """Resolve and validate the speech WAV file.

    Reads ASR_TEST_WAV env var.  Converts to compatible format if needed.
    Skips the test if the env var is not set or the file doesn't exist.
    """
    wav_path_str = os.environ.get("ASR_TEST_WAV")
    if not wav_path_str:
        pytest.skip(_ASR_SKIP_MSG)

    wav_path = Path(wav_path_str)
    if not wav_path.exists():
        pytest.fail(f"ASR_TEST_WAV file not found: {wav_path}")

    return ensure_compatible_wav(wav_path, tmp_path)


@pytest.fixture
def asr_lang() -> str:
    """Language code for recognition, from ASR_TEST_LANG or default en-US."""
    return os.environ.get("ASR_TEST_LANG", "en-US")


@pytest.fixture
def make_bridge(monkeypatch: pytest.MonkeyPatch) -> Callable[..., CppBridge]:
    """테스트용 fast-timeout CppBridge 생성.

    Test-wide class var defaults을 미리 설정한 뒤, 레거시 kwargs(host, port,
    reconnect_attempts, recv/connect/close_timeout_sec)는 class var
    monkeypatch로 변환해 적용한다.
    """

    monkeypatch.setattr(CppBridge, "_RECONNECT_ATTEMPTS", 2)
    monkeypatch.setattr(CppBridge, "_RECV_TIMEOUT_SEC", 0.1)
    monkeypatch.setattr(CppBridge, "_CONNECT_TIMEOUT_SEC", 1.0)
    monkeypatch.setattr(CppBridge, "_CLOSE_TIMEOUT_SEC", 1.0)
    monkeypatch.setattr(CppBridge, "_HOST", "localhost")
    monkeypatch.setattr(CppBridge, "_PORT", 18765)

    _CLASS_VAR_MAP = {
        "host": "_HOST",
        "port": "_PORT",
        "reconnect_attempts": "_RECONNECT_ATTEMPTS",
        "recv_timeout_sec": "_RECV_TIMEOUT_SEC",
        "connect_timeout_sec": "_CONNECT_TIMEOUT_SEC",
        "close_timeout_sec": "_CLOSE_TIMEOUT_SEC",
    }

    def _make(**overrides) -> CppBridge:
        for key, value in overrides.items():
            if key in _CLASS_VAR_MAP:
                monkeypatch.setattr(CppBridge, _CLASS_VAR_MAP[key], value)
            else:
                raise TypeError(f"Unknown override: {key}")
        return CppBridge()

    return _make


@pytest.fixture
def mock_conn() -> MagicMock:
    """A mock websockets ClientConnection."""
    conn = MagicMock()
    conn.close = MagicMock()
    conn.send = MagicMock()
    conn.recv = MagicMock(side_effect=TimeoutError)
    return conn


class FakeServer:
    """Minimal in-test WebSocket message collector.

    Used by unit tests to simulate C++ sending messages back.
    Not a real WebSocket server — just drives the mock's recv side effects.
    """

    def __init__(self) -> None:
        self.received: list[dict] = []
        self._responses: list[str] = []
        self._lock = threading.Lock()

    def queue_response(self, msg: dict) -> None:
        """Queue a JSON response to be returned by mock recv."""
        with self._lock:
            self._responses.append(json.dumps(msg))

    def pop_response(self) -> str | None:
        with self._lock:
            return self._responses.pop(0) if self._responses else None

    def capture_send(self, data: str) -> None:
        """Capture a sent JSON message."""
        self.received.append(json.loads(data))


@pytest.fixture
def fake_server() -> FakeServer:
    return FakeServer()


class FakeStreamEvent:
    """Minimal event object mimicking ResponseStreamEvent."""

    def __init__(
        self,
        event_type: str,
        delta: str = "",
        response: Any = None,
    ) -> None:
        self.type = event_type
        self.delta = delta
        self.response = response


class FakeUsageDetails:
    """Minimal usage details for testing."""

    def __init__(self, cached_tokens: int = 0, reasoning_tokens: int = 0) -> None:
        self.cached_tokens = cached_tokens
        self.reasoning_tokens = reasoning_tokens


class FakeUsage:
    """Minimal usage object for testing."""

    def __init__(
        self,
        input_tokens: int = 50,
        output_tokens: int = 10,
        input_tokens_details: FakeUsageDetails | None = None,
        output_tokens_details: FakeUsageDetails | None = None,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.input_tokens_details = input_tokens_details or FakeUsageDetails()
        self.output_tokens_details = output_tokens_details or FakeUsageDetails()


class FakeCompletedResponse:
    """Minimal completed response for testing."""

    def __init__(
        self,
        model: str = "gpt-4o",
        usage: FakeUsage | None = None,
        output: list[Any] | None = None,
    ) -> None:
        self.model = model
        self.usage = usage or FakeUsage()
        self.output = output or []


def make_stream_events(
    chunks: list[str],
    *,
    include_completed: bool = True,
    completed_response: FakeCompletedResponse | None = None,
) -> list[FakeStreamEvent]:
    """Build a sequence of text delta events with optional completed event."""
    events = [FakeStreamEvent("response.output_text.delta", c) for c in chunks]
    if include_completed:
        resp = completed_response or FakeCompletedResponse()
        events.append(FakeStreamEvent("response.completed", response=resp))
    return events


def create_mock_llm_client(
    stream_events: list[FakeStreamEvent] | None = None,
    side_effect: Exception | None = None,
) -> MagicMock:
    """Create a mock ``openai.OpenAI`` client."""
    client = MagicMock()

    if side_effect is not None:
        client.responses.create.side_effect = side_effect
        return client

    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(return_value=iter(stream_events or []))
    mock_stream.close = MagicMock()
    client.responses.create.return_value = mock_stream
    return client


def create_mock_tts_client(
    chunks: list[bytes] | None = None,
    *,
    side_effect: Exception | None = None,
    streaming_error: Exception | None = None,
) -> MagicMock:
    """Create a mock ``openai.OpenAI`` client for TTS.

    The mock wires up ``audio.speech.with_streaming_response.create()``
    to return a context manager whose ``__enter__`` returns a response
    with ``iter_bytes()`` yielding *chunks*.

    Args:
        chunks: Audio byte chunks the mock stream will yield.
        side_effect: Exception to raise from ``with_streaming_response.create()``.
        streaming_error: Exception to raise during ``iter_bytes()`` iteration.

    Returns:
        A configured ``MagicMock`` that mimics the OpenAI client.
    """
    client = MagicMock()

    if side_effect is not None:
        client.audio.speech.with_streaming_response.create.side_effect = side_effect
        return client

    mock_response = MagicMock()

    if streaming_error is not None:

        def _bad_iter(**kwargs):  # noqa: ARG001
            raise streaming_error

        mock_response.iter_bytes = _bad_iter
    else:
        mock_response.iter_bytes = MagicMock(return_value=iter(chunks or []))

    # Context manager protocol: __enter__ returns mock_response
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_response)
    mock_cm.__exit__ = MagicMock(return_value=False)

    client.audio.speech.with_streaming_response.create.return_value = mock_cm

    return client


def make_elevenlabs_chunk(
    audio: bytes = b"",
    *,
    characters: list[str] | None = None,
    starts: list[float] | None = None,
    ends: list[float] | None = None,
) -> StreamingAudioChunkWithTimestampsResponse:
    """Build a real SDK chunk from wire-format data.

    Using the real pydantic type (validated from the JSON wire format) means
    unit tests catch attribute-name drift across SDK upgrades.

    Args:
        audio: Raw PCM bytes (base64-encoded into the chunk).
        characters: Alignment characters; None omits alignment entirely.
        starts: Per-character start times; defaults to 0.1s per character.
        ends: Per-character end times; defaults to 0.1s per character.

    Returns:
        A validated ``StreamingAudioChunkWithTimestampsResponse``.
    """
    payload: dict[str, Any] = {"audio_base64": base64.b64encode(audio).decode("ascii")}
    if characters is not None:
        if starts is None:
            starts = [i * 0.1 for i in range(len(characters))]
        if ends is None:
            ends = [(i + 1) * 0.1 for i in range(len(characters))]
        payload["alignment"] = {
            "characters": characters,
            "character_start_times_seconds": starts,
            "character_end_times_seconds": ends,
        }
    return StreamingAudioChunkWithTimestampsResponse.model_validate(payload)


def create_mock_elevenlabs_client(
    chunks: list[StreamingAudioChunkWithTimestampsResponse] | None = None,
    *,
    call_error: Exception | None = None,
    streaming_error: Exception | None = None,
) -> MagicMock:
    """Create a mock ``elevenlabs.ElevenLabs`` client for TTS.

    Mirrors the real SDK's laziness: ``stream_with_timestamps()`` returns a
    generator, so *call_error* raises at the first ``next()`` (like an HTTP
    error) and *streaming_error* raises after all *chunks* are yielded.

    The mock records stream lifecycle in ``client._stream_state``
    (``{"started": bool, "closed": bool}``) — ``closed`` becomes True when
    the generator's ``finally`` runs (exhaustion, error, or ``close()``).

    Args:
        chunks: SDK chunk objects the stream will yield (see
            :func:`make_elevenlabs_chunk`).
        call_error: Exception raised before the first chunk.
        streaming_error: Exception raised after the last chunk.

    Returns:
        A configured ``MagicMock`` that mimics the ElevenLabs client.
    """
    client = MagicMock()
    state = {"started": False, "closed": False}

    def _gen() -> Any:
        state["started"] = True
        try:
            if call_error is not None:
                raise call_error
            yield from chunks or []
            if streaming_error is not None:
                raise streaming_error
        finally:
            state["closed"] = True

    client.text_to_speech.stream_with_timestamps.side_effect = lambda *args, **kwargs: _gen()
    client._stream_state = state
    return client


# ---------------------------------------------------------------------------
# Integration test helpers
# ---------------------------------------------------------------------------

_OPENAI_SKIP_MSG = "OPENAI_API_KEY not set"
_ELEVENLABS_SKIP_MSG = "ELEVENLABS_API_KEY not set"


@pytest.fixture(scope="session")
def openai_api_key() -> str:
    """Read OPENAI_API_KEY from env, skip if absent."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip(_OPENAI_SKIP_MSG)
    return key


@pytest.fixture(scope="session")
def elevenlabs_api_key() -> str:
    """Read ELEVENLABS_API_KEY from env, skip if absent."""
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        pytest.skip(_ELEVENLABS_SKIP_MSG)
    return key
