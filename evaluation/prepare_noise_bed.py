"""Build the ambient noise-bed master for the e2e SNR conditions.

Overlays several long MUSAN ``noise`` clips into one short, level-steady loop —
overlay only, never concatenating different clips end to end. Each clip is
cropped to the bed length and RMS-normalized so no single source dominates,
then the layers are summed and the mix is normalized to a reference level
(gain 1.0). calibrate_noise.py later scales this master to the per-condition
playback levels.

Why overlay instead of concatenation: summing many simultaneous sources keeps
the noise floor stationary, so a session's effective SNR stays close to the
calibrated target regardless of which part of the loop it happens to overlap —
a single-source-at-a-time track would let momentary quiet/transient windows
swing the per-session SNR.

Usage:
    uv run python -m evaluation.prepare_noise_bed --musan-dir data/musan
    uv run python -m evaluation.prepare_noise_bed --musan-dir data/musan \\
        --layers 5 --length 60 --out data/eval/noise_bed/bed_master.wav
"""

from __future__ import annotations

import argparse
import random
import wave
from math import gcd
from pathlib import Path

import numpy as np

from evaluation.bed_audio import _read_wav_mono, index_musan_noise, write_wav

_LAYER_RMS = 0.1  # 각 레이어 정규화 RMS — 한 음원이 베드를 지배하지 않게
# gain 1.0 기준 RMS. 베드를 충분히 dense하게 둬야(피크 대비 RMS가 높아야) 캘리브가
# 음성보다 낮은 SNR(시끄러움)까지 클리핑 없이 스케일할 수 있다. 0.1 같은 낮은 값은
# 트랜지언트가 피크를 다 써버려 베드가 음성 대비 충분히 커지지 못한다.
_DEFAULT_MASTER_RMS = 0.30
_PEAK_CEILING = 0.95
_LOOP_FADE_SEC = 0.5  # 루프 이음매 크로스페이드 — gapless 재생 보조


def _soft_limit(x: np.ndarray, ceiling: float) -> np.ndarray:
    """Smoothly compress peaks toward ±ceiling (tanh).

    Raises the RMS-at-fixed-peak (loudness density) without the harsh harmonics
    of hard clipping — the bulk stays near-linear, only the rare transients bend.
    """
    return (ceiling * np.tanh(x / ceiling)).astype(np.float32)


def _duration_sec(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2))) if len(x) else 0.0


def _make_loop_seamless(x: np.ndarray, fade: int) -> np.ndarray:
    """Crossfade the tail into the head so the loop point has no discontinuity."""
    if fade <= 0 or fade * 2 >= len(x):
        return x
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    out = x[:-fade].copy()
    out[:fade] = x[:fade] * ramp + x[-fade:] * (1.0 - ramp)
    return out


def build_bed(
    noise_files: list[Path],
    n_layers: int,
    length_sec: float,
    rate: int,
    seed: str,
    master_rms: float,
) -> tuple[np.ndarray, list[Path]]:
    """Overlay ``n_layers`` clips (each ≥ ``length_sec``) into the bed master."""
    rng = random.Random(seed)
    n = int(length_sec * rate)
    eligible = [f for f in noise_files if _duration_sec(f) >= length_sec]
    if len(eligible) < n_layers:
        raise SystemExit(
            f"Need {n_layers} clips ≥ {length_sec}s but only {len(eligible)} qualify. Lower --layers or --length."
        )
    chosen = rng.sample(eligible, n_layers)
    mix = np.zeros(n, dtype=np.float32)
    for f in chosen:
        x, sr = _read_wav_mono(f)
        if sr != rate:
            g = gcd(rate, sr)
            from scipy.signal import resample_poly

            x = resample_poly(x, rate // g, sr // g).astype(np.float32)
        start = rng.randint(0, len(x) - n)
        seg = x[start : start + n].copy()
        r = _rms(seg)
        if r > 0:
            seg *= _LAYER_RMS / r
        mix += seg

    r = _rms(mix)
    if r > 0:
        mix *= master_rms / r
    # Soft-limit transients toward the ceiling instead of hard-clipping or
    # scaling the whole bed down — keeps the steady RMS (the SNR reference) high.
    mix = _soft_limit(mix, _PEAK_CEILING)
    return _make_loop_seamless(mix, int(_LOOP_FADE_SEC * rate)), chosen


def main() -> None:
    p = argparse.ArgumentParser(description="Build the ambient noise-bed master")
    p.add_argument("--musan-dir", required=True)
    p.add_argument("--layers", type=int, default=5, help="Number of clips to overlay")
    p.add_argument("--length", type=float, default=60.0, help="Bed length in seconds")
    p.add_argument("--rate", type=int, default=16000, help="Output sample rate (MUSAN native = 16k)")
    p.add_argument("--master-rms", type=float, default=_DEFAULT_MASTER_RMS, help="gain-1.0 reference RMS")
    p.add_argument("--seed", default="bed-master", help="Deterministic clip-selection seed")
    p.add_argument("--out", default="data/eval/noise_bed/bed_master.wav")
    args = p.parse_args()

    noise_files = index_musan_noise(args.musan_dir)
    mix, chosen = build_bed(noise_files, args.layers, args.length, args.rate, args.seed, args.master_rms)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_wav(out, mix, args.rate)

    rms = _rms(mix)
    peak = float(np.abs(mix).max())
    print(f"Bed master: {out}")
    print(
        f"  {len(mix) / args.rate:.1f}s @ {args.rate}Hz, {args.layers} layers, "
        f"RMS≈{rms:.3f} peak≈{peak:.3f} crest≈{peak / rms:.1f}x"
    )
    for f in chosen:
        print(f"  layer: {f.relative_to(Path(args.musan_dir))} ({_duration_sec(f):.0f}s)")


if __name__ == "__main__":
    main()
