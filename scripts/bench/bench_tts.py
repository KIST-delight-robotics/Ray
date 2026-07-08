"""TTS vendor benchmark — TTFB and total synthesis latency.

Measures per backend, over short/medium/long texts:

  - TTFB (synthesize() call → first audio chunk)
  - Total latency (synthesize() call → stream exhausted)
  - Audio duration and realtime headroom (audio_sec / total_sec)

Backends:

  - ``elevenlabs``      ElevenLabsTTS — production path (stream/with-timestamps)
  - ``elevenlabs-raw``  ElevenLabs plain stream endpoint (no timestamps) —
                        isolates the with-timestamps base64+JSON overhead
  - ``openai``          OpenAITTS — baseline

Usage::

    uv run python scripts/bench/bench_tts.py
    uv run python scripts/bench/bench_tts.py --rounds 5
    uv run python scripts/bench/bench_tts.py --backends elevenlabs openai

Requires: ELEVENLABS_API_KEY and/or OPENAI_API_KEY environment variables.
Note: each round costs ElevenLabs credits (~400 chars/round per backend).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from voice_pipeline.core.interfaces import ITTS
from voice_pipeline.tts.elevenlabs_tts import ElevenLabsTTS
from voice_pipeline.tts.openai_tts import OpenAITTS

_TEXTS = {
    "short": "Sure, I can help you with that.",
    "medium": "The weather today is mostly sunny with a light breeze. It should be a great day for a walk in the park.",
    "long": (
        "Once upon a time, a small lighthouse keeper lived alone on a rocky island. "
        "Every night he climbed the spiral stairs to light the lamp for passing ships. "
        "Although no visitors had come for years, he believed his light still mattered, "
        "and one stormy evening, it guided a lost fishing boat safely home."
    ),
}

_BYTES_PER_SEC = 24000 * 2  # 24kHz 16-bit mono PCM


@dataclass
class CallResult:
    backend: str
    text_key: str
    ttfb_ms: float
    total_ms: float
    audio_sec: float


class _RawElevenLabsStream:
    """ElevenLabs plain stream endpoint wrapper (benchmark-only, no ITTS)."""

    def __init__(self) -> None:
        from elevenlabs import ElevenLabs

        self._client = ElevenLabs(timeout=ElevenLabsTTS._TIMEOUT_SEC)

    def stream(self, text: str):
        return self._client.text_to_speech.stream(
            ElevenLabsTTS._VOICE_ID,
            text=text,
            model_id=ElevenLabsTTS._MODEL,
            output_format=f"pcm_{ElevenLabsTTS.OUTPUT_SAMPLE_RATE}",
        )


def _measure_itts(tts: ITTS, text: str) -> tuple[float, float, float]:
    """Return (ttfb_ms, total_ms, audio_sec) for one ITTS.synthesize() call."""
    total_bytes = 0
    first_ts: float | None = None
    t0 = time.monotonic()
    stream = tts.synthesize(text)
    for chunk in stream:
        if first_ts is None:
            first_ts = time.monotonic()
        total_bytes += len(chunk)
    t_end = time.monotonic()
    assert first_ts is not None, "stream yielded no chunks"
    return (first_ts - t0) * 1000, (t_end - t0) * 1000, total_bytes / _BYTES_PER_SEC


def _measure_raw(raw: _RawElevenLabsStream, text: str) -> tuple[float, float, float]:
    """Return (ttfb_ms, total_ms, audio_sec) for one plain-stream call."""
    total_bytes = 0
    first_ts: float | None = None
    t0 = time.monotonic()
    for chunk in raw.stream(text):
        if first_ts is None:
            first_ts = time.monotonic()
        total_bytes += len(chunk)
    t_end = time.monotonic()
    assert first_ts is not None, "stream yielded no chunks"
    return (first_ts - t0) * 1000, (t_end - t0) * 1000, total_bytes / _BYTES_PER_SEC


def main() -> None:
    parser = argparse.ArgumentParser(description="TTS latency benchmark")
    parser.add_argument("--rounds", type=int, default=3, help="Rounds per backend/text (default: 3)")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["elevenlabs", "elevenlabs-raw", "openai"],
        default=["elevenlabs", "elevenlabs-raw", "openai"],
        help="Backends to benchmark",
    )
    args = parser.parse_args()

    backends: dict[str, object] = {}
    for name in args.backends:
        if name == "elevenlabs":
            backends[name] = ElevenLabsTTS()
        elif name == "elevenlabs-raw":
            backends[name] = _RawElevenLabsStream()
        elif name == "openai":
            backends[name] = OpenAITTS()

    results: list[CallResult] = []
    for rnd in range(1, args.rounds + 1):
        for name, impl in backends.items():
            for text_key, text in _TEXTS.items():
                if isinstance(impl, _RawElevenLabsStream):
                    ttfb, total, audio_sec = _measure_raw(impl, text)
                else:
                    ttfb, total, audio_sec = _measure_itts(impl, text)
                results.append(CallResult(name, text_key, ttfb, total, audio_sec))
                print(
                    f"  round {rnd}  {name:<15} {text_key:<6} "
                    f"ttfb {ttfb:7.1f}ms  total {total:7.1f}ms  audio {audio_sec:5.2f}s"
                )

    print()
    header = (
        f"{'backend':<15} {'text':<6} {'n':>2}   {'ttfb med':>9} {'min':>7} {'max':>7}   "
        f"{'total med':>9} {'min':>7} {'max':>7}   {'audio':>6} {'rt x':>5}"
    )
    print(header)
    print("-" * len(header))
    for name in args.backends:
        for text_key in _TEXTS:
            rows = [r for r in results if r.backend == name and r.text_key == text_key]
            if not rows:
                continue
            ttfbs = [r.ttfb_ms for r in rows]
            totals = [r.total_ms for r in rows]
            audio = statistics.median(r.audio_sec for r in rows)
            # realtime factor: 오디오 길이 / 생성 시간 — 1.0 미만이면 재생 속도보다 느림
            rt = audio / (statistics.median(totals) / 1000)
            print(
                f"{name:<15} {text_key:<6} {len(rows):>2}   "
                f"{statistics.median(ttfbs):>8.1f} {min(ttfbs):>7.1f} {max(ttfbs):>7.1f}   "
                f"{statistics.median(totals):>8.1f} {min(totals):>7.1f} {max(totals):>7.1f}   "
                f"{audio:>5.2f}s {rt:>5.2f}"
            )


if __name__ == "__main__":
    main()
