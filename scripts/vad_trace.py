"""Live VAD/VAP trace — capture mic, log Silero + MaAI VAP per frame, save everything.

Records live mic audio while running, per frame (30ms):
  - Silero VAD on the *production* path (512-sample chunks, 90ms cache, 0.5 thr)
    → ``silero_cached`` (what TurnDetector actually sees) + ``silero_raw`` (no cache)
  - MaAI VAP → ``p_now`` / ``p_fut`` / ``vap_speaking``
  - ``rms`` frame energy (no model) as an acoustic reference line

On exit (Ctrl-C or --seconds) it saves three files next to --out:
  - ``<out>.wav``  captured audio (replay offline later)
  - ``<out>.csv``  per-frame trace
  - ``<out>.png``  Silero/VAP/rms timeline with the 0.5 line + speech shading
and prints a summary quantifying the two suspected lateness causes:
  - trailing hangover (Silero stays >0.5 after speech ends)
  - pause spikes (Silero crosses >0.5 during silence → resets the silence timer)

Usage:
    uv run python scripts/vad_trace.py
    uv run python scripts/vad_trace.py --seconds 30 --out data/vad/run1
"""

from __future__ import annotations

import argparse
import ctypes
import queue
import time
import wave
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
from silero_vad import load_silero_vad

from voice_pipeline.adapters.audio_input import AudioInput
from voice_pipeline.adapters.vap import MaAIVAPModel
from voice_pipeline.settings import (
    CHANNELS,
    FRAME_DURATION_MS,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
)
from voice_pipeline.types import AudioFrame

# Silence ALSA's chatty C-level warnings (mirrors other hardware scripts).
_alsa_error_handler = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
)(lambda *_: None)
try:
    _asound = ctypes.cdll.LoadLibrary("libasound.so.2")
    _asound.snd_lib_error_set_handler(_alsa_error_handler)
except Exception:
    _asound = None

# Production VAD path constants (must mirror __main__.vad_fn).
_SILERO_CHUNK_BYTES = 512 * 2  # 512 samples x 16-bit
_VAD_INFER_INTERVAL = 3  # cache update cadence: every 3rd frame (90ms)
_THRESHOLD = 0.5  # speaking threshold (Silero + VAP)
_FRAME_SEC = FRAME_DURATION_MS / 1000.0
_TTS_SAMPLE_RATE = 24000  # only used for robot-channel resample; robot is silent here


class _SileroProdPath:
    """Replicates __main__.vad_fn: 512-sample chunks, score cached every 3 frames.

    One Silero model instance processes every chunk in order (LSTM state is
    sequence-dependent, not timing-dependent), so ``raw`` is the true latest
    per-chunk score and ``cached`` is the value production would return.
    """

    def __init__(self) -> None:
        self._model = load_silero_vad(onnx=True)
        self._buf = bytearray()
        self._raw = 0.0
        self._cached = 0.0
        self._frame_count = 0

    def feed(self, frame: AudioFrame) -> tuple[float, float]:
        """Feed one 30ms frame, return (cached_score, raw_score)."""
        self._buf.extend(frame)
        self._frame_count += 1
        while len(self._buf) >= _SILERO_CHUNK_BYTES:
            chunk = bytes(self._buf[:_SILERO_CHUNK_BYTES])
            del self._buf[:_SILERO_CHUNK_BYTES]
            samples = torch.frombuffer(bytearray(chunk), dtype=torch.int16).float() / 32768.0
            self._raw = self._model(samples, SAMPLE_RATE).item()
        if self._frame_count % _VAD_INFER_INTERVAL == 0:
            self._cached = self._raw
        return self._cached, self._raw


def _rms(frame: AudioFrame) -> float:
    """Root-mean-square energy of a frame, normalised to [0, 1]."""
    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples**2)))


def _capture(seconds: float | None) -> list[dict]:
    """Run the live capture loop, return the per-frame trace rows."""
    silero = _SileroProdPath()
    vap = MaAIVAPModel(_TTS_SAMPLE_RATE)

    audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=300)
    audio_input = AudioInput(audio_queue)
    if _asound is not None:
        _asound.snd_lib_error_set_handler(None)

    rows: list[dict] = []
    audio_input.start()
    print(f"Recording mic... ({'Ctrl-C to stop' if seconds is None else f'{seconds:.0f}s'})")
    frame_idx = 0
    deadline = None if seconds is None else time.monotonic() + seconds
    try:
        while deadline is None or time.monotonic() < deadline:
            try:
                frame = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            cached, raw = silero.feed(frame)
            # Synchronous single-frame inference for a deterministic trace.
            vr = vap.infer(frame)  # robot channel silent (user-turn context)
            rows.append(
                {
                    "time": frame_idx * _FRAME_SEC,
                    "silero_cached": cached,
                    "silero_raw": raw,
                    "vap_p_now": vr.p_now,
                    "vap_p_fut": vr.p_fut,
                    "vap_speaking": int(vr.user_is_speaking),
                    "rms": _rms(frame),
                    "speaking": int(cached > _THRESHOLD),
                    "_audio": frame,
                }
            )
            frame_idx += 1
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        audio_input.stop()
    return rows


def _energy_threshold(rms: np.ndarray) -> float:
    """Adaptive acoustic-activity gate: noise floor + fraction of the range."""
    if rms.size == 0:
        return 0.0
    floor = float(np.percentile(rms, 20))
    ref = float(np.percentile(rms, 90))
    return floor + 0.15 * max(ref - floor, 1e-6)


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs in *mask* as (start_idx, end_idx_inclusive)."""
    segs: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, len(mask) - 1))
    return segs


def _summarise(rows: list[dict]) -> None:
    t = np.array([r["time"] for r in rows])
    sil = np.array([r["silero_cached"] for r in rows])
    rms = np.array([r["rms"] for r in rows])
    speaking = sil > _THRESHOLD
    acoustic = rms > _energy_threshold(rms)

    speech_segs = _segments(speaking)
    print("\n=== Summary ===")
    print(f"Duration: {t[-1] + _FRAME_SEC:.1f}s, frames: {len(rows)}")
    print(f"Silero speech segments (>{_THRESHOLD}): {len(speech_segs)}")
    print(f"Acoustic-activity gate (rms): {_energy_threshold(rms):.4f}")

    # #1 Trailing hangover: gap between last acoustic frame and Silero dropping <0.5.
    hangovers = []
    for s, e in speech_segs:
        # last acoustically-active frame at or before the Silero speech end
        window = acoustic[s : e + 1]
        if not window.any():
            continue
        last_acoustic = s + int(np.where(window)[0][-1])
        hangovers.append((e - last_acoustic) * _FRAME_SEC)
    if hangovers:
        print(
            f"\n[#1 trailing hangover] n={len(hangovers)} "
            f"mean={np.mean(hangovers) * 1000:.0f}ms max={np.max(hangovers) * 1000:.0f}ms"
        )
        print("    (Silero stays >0.5 this long after acoustic speech ends)")

    # #2 Pause spikes: upward 0.5 crossings while acoustically silent.
    rising = (sil[1:] > _THRESHOLD) & (sil[:-1] <= _THRESHOLD)
    spurious = int(np.sum(rising & ~acoustic[1:]))
    print(f"\n[#2 pause spikes] {spurious} spurious >0.5 crossings during silence")
    print("    (each one resets the silence timer in TurnDetector)")

    # In-speech dips: downward crossings while still acoustically active.
    falling = (sil[1:] <= _THRESHOLD) & (sil[:-1] > _THRESHOLD)
    early_dips = int(np.sum(falling & acoustic[1:]))
    print(f"\n[in-speech dips] {early_dips} drops <0.5 while still speaking")


def _save(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    # WAV
    with wave.open(str(out.with_suffix(".wav")), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(r["_audio"] for r in rows))

    # CSV
    cols = ["time", "silero_cached", "silero_raw", "vap_p_now", "vap_p_fut", "vap_speaking", "rms", "speaking"]
    with open(out.with_suffix(".csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")

    # PNG
    t = np.array([r["time"] for r in rows])
    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.01), 5))
    ax.plot(t, [r["silero_cached"] for r in rows], label="silero_cached", lw=1.2)
    ax.plot(t, [r["silero_raw"] for r in rows], label="silero_raw", lw=0.7, alpha=0.6)
    ax.plot(t, [r["vap_p_now"] for r in rows], label="vap_p_now", lw=0.9, alpha=0.7)
    ax.plot(t, [r["vap_p_fut"] for r in rows], label="vap_p_fut", lw=0.9, alpha=0.7)
    ax.plot(t, [r["rms"] for r in rows], label="rms", lw=0.7, color="gray", alpha=0.5)
    ax.axhline(_THRESHOLD, color="red", ls="--", lw=0.8, alpha=0.6)
    for s, e in _segments(np.array([r["silero_cached"] for r in rows]) > _THRESHOLD):
        ax.axvspan(t[s], t[e], color="green", alpha=0.08)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("score")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=120)
    print(f"\nSaved: {out.with_suffix('.wav')}, {out.with_suffix('.csv')}, {out.with_suffix('.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=None, help="capture duration (default: until Ctrl-C)")
    parser.add_argument("--out", type=str, default=None, help="output path prefix (default: data/vad/trace_<ts>)")
    args = parser.parse_args()

    out = Path(args.out) if args.out else Path("data/vad") / f"trace_{datetime.now():%Y%m%d_%H%M%S}"

    rows = _capture(args.seconds)
    if not rows:
        print("No audio captured.")
        return
    _summarise(rows)
    _save(rows, out)


if __name__ == "__main__":
    main()
