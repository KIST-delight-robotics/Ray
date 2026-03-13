"""Verify ONNX custom pipeline numerical equivalence against original MaAI
using real conversational audio from the CANDOR dataset.

Feeds stereo audio (L=speaker1, R=speaker2) frame-by-frame and compares
p_now, p_future, vad outputs between the two pipelines over long durations.

Usage:
    uv run python scripts/bench/verify_onnx_equivalence.py \
        --audio CANDOR/raw_media_part_001/23d4ec0e-.../processed/23d4ec0e-...mp3 \
        --max-seconds 120
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from vap_onnx_pipeline import VapOnnxPipeline, create_maai


def load_stereo_audio(path: str, target_sr: int = 16000) -> tuple[np.ndarray, np.ndarray]:
    """Load stereo audio and resample to target_sr. Returns (ch1, ch2) as float32."""
    data, sr = sf.read(path, dtype="float32")

    if data.ndim == 1:
        raise ValueError(f"Expected stereo audio, got mono: {path}")

    ch1 = data[:, 0]
    ch2 = data[:, 1]

    if sr != target_sr:
        # Simple resample via linear interpolation (good enough for verification)
        ratio = target_sr / sr
        n_out = int(len(ch1) * ratio)
        indices = np.linspace(0, len(ch1) - 1, n_out).astype(np.float64)
        idx_lo = indices.astype(np.int64)
        idx_hi = np.minimum(idx_lo + 1, len(ch1) - 1)
        frac = (indices - idx_lo).astype(np.float32)

        ch1 = ch1[idx_lo] * (1 - frac) + ch1[idx_hi] * frac
        ch2 = ch2[idx_lo] * (1 - frac) + ch2[idx_hi] * frac

        print(f"  Resampled: {sr}Hz -> {target_sr}Hz ({len(ch1)} samples)")

    return ch1, ch2


def run_verification(
    audio_path: str,
    frame_rate: int = 10,
    context_len_sec: float = 5.0,
    max_seconds: float | None = None,
):
    print(f"\n{'=' * 70}")
    print("  ONNX Equivalence Verification (Real Audio)")
    print(f"{'=' * 70}")
    print(f"  Audio       : {os.path.basename(audio_path)}")
    print(f"  Frame rate  : {frame_rate}Hz")
    print(f"  Context     : {context_len_sec}s")

    # Load audio
    ch1_full, ch2_full = load_stereo_audio(audio_path)
    total_seconds = len(ch1_full) / 16000
    print(f"  Duration    : {total_seconds:.1f}s")

    if max_seconds and max_seconds < total_seconds:
        n_samples = int(max_seconds * 16000)
        ch1_full = ch1_full[:n_samples]
        ch2_full = ch2_full[:n_samples]
        total_seconds = max_seconds
        print(f"  Truncated to: {total_seconds:.1f}s")

    # Split into frames
    samples_per_frame = 16000 // frame_rate
    n_frames = len(ch1_full) // samples_per_frame
    print(f"  Frames      : {n_frames}")

    # Initialize both pipelines (single-threaded — best config)
    torch.set_num_threads(1)

    print("\n  Loading MaAI (original)...")
    maai = create_maai(frame_rate, context_len_sec)

    print("  Loading Custom ONNX pipeline...")
    custom = VapOnnxPipeline(frame_rate, context_len_sec, ort_threads=1)

    # Run frame-by-frame comparison
    print(f"\n  Running {n_frames} frames...")

    n_compared = 0
    diffs_pnow = []
    diffs_pfuture = []
    diffs_vad = []
    lats_maai = []
    lats_custom = []

    t_start = time.perf_counter()

    for i in range(n_frames):
        start = i * samples_per_frame
        end = start + samples_per_frame
        x1 = ch1_full[start:end]
        x2 = ch2_full[start:end]

        # Original MaAI
        t0 = time.perf_counter()
        maai.process(x1, x2)
        t_maai = time.perf_counter() - t0
        try:
            maai_out = maai.result_dict_queue.get_nowait()
        except Exception:
            # No output yet (buffering)
            custom.process(x1, x2)
            continue

        lats_maai.append(t_maai)

        # Custom ONNX
        t0 = time.perf_counter()
        custom_out = custom.process(x1, x2)
        t_custom = time.perf_counter() - t0
        if custom_out is None:
            continue

        lats_custom.append(t_custom)
        n_compared += 1

        # Compute diffs
        def max_abs_diff(a, b):
            if isinstance(a, list) and isinstance(b, list):
                return max(abs(ai - bi) for ai, bi in zip(a, b, strict=False))
            return abs(a - b)

        d_pnow = max_abs_diff(maai_out["p_now"], custom_out["p_now"])
        d_pfuture = max_abs_diff(maai_out["p_future"], custom_out["p_future"])
        d_vad = max(
            abs(maai_out["vad"][0] - custom_out["vad"][0]),
            abs(maai_out["vad"][1] - custom_out["vad"][1]),
        )

        diffs_pnow.append(d_pnow)
        diffs_pfuture.append(d_pfuture)
        diffs_vad.append(d_vad)

        # Progress every 100 frames
        if n_compared % 100 == 0:
            elapsed = time.perf_counter() - t_start
            maai_avg = np.mean(lats_maai[-100:]) * 1000
            custom_avg = np.mean(lats_custom[-100:]) * 1000
            print(
                f"    frame {n_compared:5d} | "
                f"diff: pnow={max(diffs_pnow):.6f} vad={max(diffs_vad):.6f} | "
                f"lat: maai={maai_avg:.1f}ms custom={custom_avg:.1f}ms | "
                f"{elapsed:.1f}s"
            )

    elapsed = time.perf_counter() - t_start

    # Spike analysis
    if lats_custom:
        custom_arr = np.array(lats_custom) * 1000
        maai_arr = np.array(lats_maai) * 1000
        p90_custom = np.percentile(custom_arr, 90)

        spike_indices = np.where(custom_arr > p90_custom)[0]
        if len(spike_indices) > 0:
            print(f"\n  --- Spike Analysis (Custom > P90={p90_custom:.1f}ms) ---")
            # Show gaps between spikes to detect periodicity
            gaps = np.diff(spike_indices)
            if len(gaps) > 0:
                print(f"  Spike count : {len(spike_indices)} / {len(custom_arr)} frames")
                print(
                    f"  Gap between spikes (frames): "
                    f"mean={gaps.mean():.1f} median={np.median(gaps):.1f} "
                    f"min={gaps.min()} max={gaps.max()}"
                )
            # Top 10 worst frames
            worst = np.argsort(custom_arr)[-10:][::-1]
            print("  Top 10 slowest (Custom):")
            for idx in worst:
                print(
                    f"    frame {idx:5d} ({idx / frame_rate:6.1f}s): "
                    f"custom={custom_arr[idx]:.1f}ms  maai={maai_arr[idx]:.1f}ms"
                )

    # Results
    print(f"\n{'=' * 70}")
    print(f"  Results ({n_compared} frames compared, {elapsed:.1f}s)")
    print(f"{'=' * 70}")

    if n_compared == 0:
        print("  ERROR: No frames compared!")
        return

    def stats(name: str, vals: list[float]):
        arr = np.array(vals)
        print(
            f"  {name:<12}: "
            f"max={arr.max():.7f}  "
            f"mean={arr.mean():.7f}  "
            f"p99={np.percentile(arr, 99):.7f}  "
            f"last={arr[-1]:.7f}"
        )

    print("\n  --- Numerical Equivalence ---")
    stats("p_now", diffs_pnow)
    stats("p_future", diffs_pfuture)
    stats("vad", diffs_vad)

    # Latency stats
    budget_ms = 1000.0 / frame_rate
    print(f"\n  --- Latency (budget={budget_ms:.0f}ms) ---")

    def lat_stats(name: str, vals: list[float]):
        arr = np.array(vals) * 1000  # to ms
        print(
            f"  {name:<12}: "
            f"mean={arr.mean():.1f}ms  "
            f"median={np.median(arr):.1f}ms  "
            f"p95={np.percentile(arr, 95):.1f}ms  "
            f"max={arr.max():.1f}ms"
        )

    lat_stats("MaAI", lats_maai)
    lat_stats("Custom ONNX", lats_custom)

    maai_mean = np.mean(lats_maai) * 1000
    custom_mean = np.mean(lats_custom) * 1000
    speedup = maai_mean / custom_mean if custom_mean > 0 else 0
    rtf = budget_ms / custom_mean if custom_mean > 0 else 0
    print(f"\n  Speedup     : {speedup:.2f}x")
    print(f"  Custom RTF  : {rtf:.2f}x  {'OK' if rtf >= 1.0 else 'TOO SLOW'}")

    # Check for drift (compare first half vs second half)
    if n_compared >= 20:
        mid = n_compared // 2
        first_half_max = max(max(diffs_pnow[:mid]), max(diffs_pfuture[:mid]), max(diffs_vad[:mid]))
        second_half_max = max(
            max(diffs_pnow[mid:]), max(diffs_pfuture[mid:]), max(diffs_vad[mid:])
        )
        drift_ratio = second_half_max / first_half_max if first_half_max > 0 else 1.0
        print("\n  Drift check:")
        print(f"    1st half max: {first_half_max:.7f}")
        print(f"    2nd half max: {second_half_max:.7f}")
        print(f"    Ratio       : {drift_ratio:.2f}x")
        if drift_ratio > 10:
            print("    WARNING: significant drift detected!")
        else:
            print("    OK: no significant drift")

    # Overall verdict
    threshold = 0.01
    overall_max = max(max(diffs_pnow), max(diffs_pfuture), max(diffs_vad))
    print(f"\n  Overall max diff: {overall_max:.7f}")
    if overall_max < threshold:
        print(f"  PASSED (all diffs < {threshold})")
    else:
        print(f"  FAILED (max diff {overall_max:.7f} >= {threshold})")


def run_solo_benchmark(
    audio_path: str,
    frame_rate: int = 10,
    context_len_sec: float = 5.0,
    max_seconds: float | None = None,
    disable_gc: bool = False,
):
    """Benchmark Custom ONNX pipeline alone (no CPU contention)."""
    import gc

    print(f"\n{'=' * 70}")
    print("  Solo Benchmark: Custom ONNX Pipeline (Real Audio)")
    print(f"{'=' * 70}")
    print(f"  Audio       : {os.path.basename(audio_path)}")
    print(f"  Frame rate  : {frame_rate}Hz")
    print(f"  Context     : {context_len_sec}s")
    print(f"  GC          : {'DISABLED' if disable_gc else 'enabled'}")

    ch1_full, ch2_full = load_stereo_audio(audio_path)
    total_seconds = len(ch1_full) / 16000
    print(f"  Duration    : {total_seconds:.1f}s")

    if max_seconds and max_seconds < total_seconds:
        n_samples = int(max_seconds * 16000)
        ch1_full = ch1_full[:n_samples]
        ch2_full = ch2_full[:n_samples]
        total_seconds = max_seconds
        print(f"  Truncated to: {total_seconds:.1f}s")

    samples_per_frame = 16000 // frame_rate
    n_frames = len(ch1_full) // samples_per_frame
    budget_ms = 1000.0 / frame_rate
    warmup = min(50, n_frames // 10)
    print(f"  Frames      : {n_frames} ({warmup} warmup)")

    torch.set_num_threads(1)

    print("\n  Loading Custom ONNX pipeline (pt=1, ort=1)...")
    custom = VapOnnxPipeline(frame_rate, context_len_sec, ort_threads=1)

    # Warmup
    for i in range(warmup):
        start = i * samples_per_frame
        end = start + samples_per_frame
        custom.process(ch1_full[start:end], ch2_full[start:end])

    custom.reset()

    # GC tracking
    gc_events = []  # (frame_idx, generation)

    def gc_callback(phase, info):
        if phase == "start":
            gc_events.append((len(lats), info["generation"]))

    gc.callbacks.append(gc_callback)

    if disable_gc:
        gc.collect()  # clean up before disabling
        gc.disable()

    # Benchmark
    lats = []
    print(f"\n  Running {n_frames - warmup} frames...")
    t_start = time.perf_counter()

    for i in range(warmup, n_frames):
        start = i * samples_per_frame
        end = start + samples_per_frame

        t0 = time.perf_counter()
        custom.process(ch1_full[start:end], ch2_full[start:end])
        lats.append(time.perf_counter() - t0)

        if len(lats) % 200 == 0:
            arr = np.array(lats[-200:]) * 1000
            elapsed = time.perf_counter() - t_start
            print(
                f"    frame {len(lats):5d} | "
                f"mean={arr.mean():.1f}ms  p95={np.percentile(arr, 95):.1f}ms  "
                f"max={arr.max():.1f}ms | {elapsed:.1f}s"
            )

    # Restore GC
    gc.callbacks.remove(gc_callback)
    if disable_gc:
        gc.enable()

    elapsed = time.perf_counter() - t_start
    arr = np.array(lats) * 1000

    print(f"\n{'=' * 70}")
    print(f"  Results ({len(lats)} frames, {elapsed:.1f}s)")
    print(f"{'=' * 70}")
    print(f"  Budget      : {budget_ms:.0f}ms")
    print(f"  Mean        : {arr.mean():.1f}ms")
    print(f"  Median      : {np.median(arr):.1f}ms")
    print(f"  P95         : {np.percentile(arr, 95):.1f}ms")
    print(f"  P99         : {np.percentile(arr, 99):.1f}ms")
    print(f"  Max         : {arr.max():.1f}ms")
    print(f"  Min         : {arr.min():.1f}ms")
    print(
        f"  RTF (mean)  : {budget_ms / arr.mean():.2f}x  "
        f"{'OK' if budget_ms / arr.mean() >= 1.0 else 'TOO SLOW'}"
    )
    print(
        f"  RTF (p95)   : {budget_ms / np.percentile(arr, 95):.2f}x  "
        f"{'OK' if budget_ms / np.percentile(arr, 95) >= 1.0 else 'TOO SLOW'}"
    )

    # Spike analysis
    p90 = np.percentile(arr, 90)
    spike_mask = arr > p90
    spike_indices = np.where(spike_mask)[0]
    if len(spike_indices) > 1:
        gaps = np.diff(spike_indices)
        print(f"\n  Spikes (>{p90:.0f}ms): {spike_mask.sum()} frames")
        print(
            f"    Gap: mean={gaps.mean():.1f} median={np.median(gaps):.1f} "
            f"min={gaps.min()} max={gaps.max()}"
        )

    # GC correlation
    if gc_events:
        gc_frames = set(f for f, _ in gc_events)
        gc_on_spike = sum(1 for idx in spike_indices if idx in gc_frames)
        gen_counts = {}
        for _, gen in gc_events:
            gen_counts[gen] = gen_counts.get(gen, 0) + 1
        print(
            f"\n  GC events   : {len(gc_events)} total "
            f"(gen0={gen_counts.get(0, 0)}"
            f" gen1={gen_counts.get(1, 0)}"
            f" gen2={gen_counts.get(2, 0)})"
        )
        print(
            f"  GC on spike : {gc_on_spike} / {len(spike_indices)} "
            f"({100 * gc_on_spike / len(spike_indices):.0f}%)"
        )

        # Latency of GC frames vs non-GC frames
        gc_lats = [arr[f] for f in gc_frames if f < len(arr)]
        non_gc_lats = [arr[f] for f in range(len(arr)) if f not in gc_frames]
        if gc_lats and non_gc_lats:
            print(
                f"  GC frame lat: mean={np.mean(gc_lats):.1f}ms  "
                f"median={np.median(gc_lats):.1f}ms  max={np.max(gc_lats):.1f}ms"
            )
            print(
                f"  Non-GC lat  : mean={np.mean(non_gc_lats):.1f}ms  "
                f"median={np.median(non_gc_lats):.1f}ms  max={np.max(non_gc_lats):.1f}ms"
            )
    else:
        print(f"\n  GC events   : 0 (GC {'disabled' if disable_gc else 'did not trigger'})")

    # Top 5 worst
    worst = np.argsort(arr)[-5:][::-1]
    gc_frames_set = set(f for f, _ in gc_events)
    print("\n  Top 5 slowest:")
    for idx in worst:
        gc_marker = " [GC]" if idx in gc_frames_set else ""
        t_sec = (idx + warmup) / frame_rate
        print(f"    frame {idx:5d} ({t_sec:6.1f}s): {arr[idx]:.1f}ms{gc_marker}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify ONNX pipeline equivalence with real audio",
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to stereo audio file (CANDOR mp3)",
    )
    parser.add_argument("--frame-rate", type=int, default=10, help="Frame rate (default: 10)")
    parser.add_argument("--context", type=float, default=5.0, help="Context length in seconds")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Max audio duration to test",
    )
    parser.add_argument(
        "--solo",
        action="store_true",
        help="Run Custom ONNX pipeline alone (no MaAI)",
    )
    parser.add_argument(
        "--no-gc",
        action="store_true",
        help="Disable Python GC during benchmark",
    )
    args = parser.parse_args()

    if args.solo:
        run_solo_benchmark(
            audio_path=args.audio,
            frame_rate=args.frame_rate,
            context_len_sec=args.context,
            max_seconds=args.max_seconds,
            disable_gc=args.no_gc,
        )
    else:
        run_verification(
            audio_path=args.audio,
            frame_rate=args.frame_rate,
            context_len_sec=args.context,
            max_seconds=args.max_seconds,
        )


if __name__ == "__main__":
    main()
