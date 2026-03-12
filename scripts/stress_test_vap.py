"""VAP spike analysis: per-stage profiling over sustained real audio.

Profiles each stage (ONNX encoder, transformer, cache trim) per frame
to identify which component causes latency spikes.

Usage:
    uv run python scripts/stress_test_vap.py \
        --audio CANDOR/raw_media_part_001/a29635a0-.../processed/a29635a0-...mp3 \
        --duration 120
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from benchmark_maai_custom_pipeline import VapOnnxPipeline


def load_stereo_audio(path: str) -> tuple[np.ndarray, np.ndarray]:
    data, sr = sf.read(path, dtype="float32")
    ch1, ch2 = data[:, 0], data[:, 1]
    if sr != 16000:
        ratio = 16000 / sr
        n_out = int(len(ch1) * ratio)
        idx = np.linspace(0, len(ch1) - 1, n_out).astype(np.float64)
        lo = idx.astype(np.int64)
        hi = np.minimum(lo + 1, len(ch1) - 1)
        frac = (idx - lo).astype(np.float32)
        ch1 = ch1[lo] * (1 - frac) + ch1[hi] * frac
        ch2 = ch2[lo] * (1 - frac) + ch2[hi] * frac
    return ch1, ch2


class ProfiledPipeline(VapOnnxPipeline):
    """VapOnnxPipeline with per-stage timing instrumentation."""

    def process_profiled(self, x1: np.ndarray, x2: np.ndarray) -> dict:
        """Returns timing dict even when VAP hasn't accumulated enough audio."""
        timings = {}

        # 1. Audio buffering
        self.current_x1 = np.concatenate([self.current_x1, x1])
        self.current_x2 = np.concatenate([self.current_x2, x2])

        if len(self.current_x1) < self.audio_frame_size:
            return {"skipped": True}

        # 2. ONNX encoder
        wav1 = self.current_x1.reshape(1, 1, -1)
        wav2 = self.current_x2.reshape(1, 1, -1)

        t0 = time.perf_counter()
        e1_np, self.h1, self.c1 = self.sess1.run(
            None, {"waveform": wav1, "h_in": self.h1, "c_in": self.c1}
        )
        e2_np, self.h2, self.c2 = self.sess2.run(
            None, {"waveform": wav2, "h_in": self.h2, "c_in": self.c2}
        )
        timings["encoder"] = time.perf_counter() - t0

        # 3. Conversion
        t0 = time.perf_counter()
        e1 = torch.from_numpy(e1_np)
        e2 = torch.from_numpy(e2_np)
        timings["convert"] = time.perf_counter() - t0

        # 4. Transformer
        t0 = time.perf_counter()
        with torch.no_grad():
            out, self.vap_cache = self.vap.forward(e1, e2, cache=self.vap_cache)
        timings["transformer"] = time.perf_counter() - t0

        # 5. Cache trim
        t0 = time.perf_counter()
        if self.vap_cache is not None:
            limit = self.audio_context_len - 1
            new_cache = {}
            for key, (k_list, v_list) in self.vap_cache.items():
                new_cache[key] = (
                    [
                        t[..., -limit:, :] if isinstance(t, torch.Tensor) and t.dim() >= 3 else t
                        for t in k_list
                    ],
                    [
                        t[..., -limit:, :] if isinstance(t, torch.Tensor) and t.dim() >= 3 else t
                        for t in v_list
                    ],
                )
            self.vap_cache = new_cache
        timings["cache_trim"] = time.perf_counter() - t0

        # 6. Buffer trim
        self.current_x1 = self.current_x1[-self.frame_contxt_padding:].copy()
        self.current_x2 = self.current_x2[-self.frame_contxt_padding:].copy()

        timings["total"] = timings["encoder"] + timings["transformer"] + timings["cache_trim"]
        return timings


def measure_memory_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 1024 / 1024


def main():
    parser = argparse.ArgumentParser(description="VAP spike analysis")
    parser.add_argument("--audio", required=True, help="Stereo audio file (CANDOR mp3)")
    parser.add_argument("--duration", type=float, default=120, help="Test duration (seconds)")
    parser.add_argument("--frame-rate", type=int, default=10, help="VAP frame rate (Hz)")
    parser.add_argument("--pt-threads", type=int, default=1, help="PyTorch threads")
    parser.add_argument("--ort-threads", type=int, default=1, help="ONNX Runtime threads")
    args = parser.parse_args()

    budget = 1000.0 / args.frame_rate  # ms per frame

    print("=" * 70)
    print("  VAP Spike Analysis")
    print("=" * 70)
    print(f"  Duration       : {args.duration}s")
    print(f"  Frame rate     : {args.frame_rate}Hz (budget {budget:.0f}ms)")
    print(f"  PT threads     : {args.pt_threads}")
    print(f"  ORT threads    : {args.ort_threads}")

    # Load audio
    print("\n  Loading audio...")
    ch1, ch2 = load_stereo_audio(os.path.abspath(args.audio))
    print(f"  Audio          : {len(ch1) / 16000:.1f}s")

    # Load pipeline
    print("  Loading VAP pipeline...")
    torch.set_num_threads(args.pt_threads)
    pipeline = ProfiledPipeline(
        frame_rate=args.frame_rate, context_len_sec=5.0, ort_threads=args.ort_threads,
    )

    # Warmup
    spf = 16000 // args.frame_rate
    for i in range(50):
        pipeline.process(ch1[i * spf : (i + 1) * spf], ch2[i * spf : (i + 1) * spf])
    pipeline.reset()
    gc.collect()

    mem_start = measure_memory_mb()
    print(f"  RSS at start   : {mem_start:.1f} MB")

    # Run
    n_frames = min(int(args.duration * args.frame_rate), len(ch1) // spf)
    interval = 1.0 / args.frame_rate

    records: list[dict] = []
    gc_counts_start = gc.get_count()
    gc.disable()  # Disable GC to test if it causes spikes

    print(f"\n  Running {args.duration}s ({n_frames} frames)...")
    print(f"  {'─' * 60}")

    t_start = time.perf_counter()

    for i in range(n_frames):
        t_frame = time.perf_counter()
        elapsed = t_frame - t_start

        x1 = ch1[i * spf : (i + 1) * spf]
        x2 = ch2[i * spf : (i + 1) * spf]

        timings = pipeline.process_profiled(x1, x2)
        if "skipped" not in timings:
            timings["elapsed"] = elapsed
            timings["frame"] = i
            records.append(timings)

        # Progress
        if i > 0 and i % (args.frame_rate * 30) == 0:
            m = int(elapsed // 60)
            s = int(elapsed % 60)
            total_ms = timings.get("total", 0) * 1000
            print(f"    [{m:02d}:{s:02d}] frame={i}, last={total_ms:.1f}ms, "
                  f"rss={measure_memory_mb():.1f}MB")

        # Pace to real-time
        frame_elapsed = time.perf_counter() - t_frame
        sleep_time = interval - frame_elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    gc.enable()
    gc.collect()
    total_elapsed = time.perf_counter() - t_start
    mem_end = measure_memory_mb()

    # Analysis
    enc = np.array([r["encoder"] for r in records]) * 1000
    tfm = np.array([r["transformer"] for r in records]) * 1000
    trim = np.array([r["cache_trim"] for r in records]) * 1000
    total = np.array([r["total"] for r in records]) * 1000
    times = np.array([r["elapsed"] for r in records])

    print(f"\n{'=' * 70}")
    print(f"  Results ({total_elapsed:.1f}s, {len(records)} frames)")
    print(f"{'=' * 70}")

    print(f"\n  Per-stage latency (ms):")
    print(f"    {'Stage':<14s} {'Mean':>7s} {'Median':>7s} {'P95':>7s} {'Max':>7s}")
    for name, arr in [("Encoder", enc), ("Transformer", tfm), ("Cache trim", trim), ("Total", total)]:
        print(f"    {name:<14s} {arr.mean():7.1f} {np.median(arr):7.1f} "
              f"{np.percentile(arr, 95):7.1f} {arr.max():7.1f}")

    print(f"\n  Budget ({budget:.0f}ms) analysis:")
    over = total > budget
    print(f"    Over budget  : {over.sum()}/{len(total)} ({100 * over.sum() / len(total):.1f}%)")

    # Spike analysis
    print(f"\n  Spike analysis:")
    for threshold in [budget * 0.8, budget, budget * 1.5]:
        mask = total > threshold
        n = mask.sum()
        if n == 0:
            print(f"    >{threshold:.0f}ms : 0/{len(total)} (0.0%)")
            continue
        pct = 100 * n / len(total)
        spike_total = total[mask]
        spike_enc = enc[mask]
        spike_tfm = tfm[mask]
        spike_trim = trim[mask]
        spike_times = times[mask]
        if n > 1:
            gaps = np.diff(spike_times)
            gap_info = f"  gap mean={gaps.mean():.1f}s min={gaps.min():.2f}s"
        else:
            gap_info = ""
        print(f"    >{threshold:.0f}ms : {n}/{len(total)} ({pct:.1f}%){gap_info}")
        print(f"          enc={spike_enc.mean():.1f}  tfm={spike_tfm.mean():.1f}  "
              f"trim={spike_trim.mean():.1f}ms (avg breakdown)")

    # Distribution
    print(f"\n  Percentile ladder (total ms):")
    for p in [50, 75, 80, 85, 90, 95, 97, 99, 99.5]:
        print(f"    P{p:<5} : {np.percentile(total, p):.1f}ms")

    print(f"\n  Histogram (total ms):")
    bins = list(range(0, int(total.max()) + 20, 10))
    counts, edges = np.histogram(total, bins=bins)
    cum = 0
    for j in range(len(counts)):
        cum += counts[j]
        if counts[j] == 0:
            continue
        bar = "#" * min(counts[j], 80)
        print(f"    {edges[j]:5.0f}–{edges[j+1]:5.0f}ms : {counts[j]:4d} ({100*cum/len(total):5.1f}%) {bar}")

    print(f"\n  Transformer histogram (ms):")
    tfm_bins = list(range(0, int(tfm.max()) + 20, 10))
    tfm_counts, tfm_edges = np.histogram(tfm, bins=tfm_bins)
    cum = 0
    for j in range(len(tfm_counts)):
        cum += tfm_counts[j]
        if tfm_counts[j] == 0:
            continue
        bar = "#" * min(tfm_counts[j], 80)
        print(f"    {tfm_edges[j]:5.0f}–{tfm_edges[j+1]:5.0f}ms : {tfm_counts[j]:4d} ({100*cum/len(tfm):5.1f}%) {bar}")

    # Top spikes detail
    p99 = np.percentile(total, 99)
    outlier_mask = total > p99
    if outlier_mask.sum() > 0:
        print(f"\n  Top spikes (>P99={p99:.0f}ms):")
        print(f"    {'Time':>7s} {'Total':>7s} {'Enc':>7s} {'Tfm':>7s} {'Trim':>7s}")
        idxs = np.where(outlier_mask)[0]
        for i in idxs[:10]:
            print(f"    {times[i]:6.1f}s {total[i]:7.1f} {enc[i]:7.1f} "
                  f"{tfm[i]:7.1f} {trim[i]:7.1f}")

    # Drift per 30s
    print(f"\n  Latency drift (per 30s):")
    print(f"    {'Sec':>6s} {'N':>5s} {'Mean':>7s} {'P95':>7s} {'Max':>7s}")
    for bucket_start in range(0, int(times[-1]) + 1, 30):
        mask = (times >= bucket_start) & (times < bucket_start + 30)
        if mask.sum() == 0:
            continue
        b = total[mask]
        print(f"    {bucket_start:5d}s {mask.sum():5d} {b.mean():7.1f} "
              f"{np.percentile(b, 95):7.1f} {b.max():7.1f}")

    print(f"\n  Memory:")
    print(f"    RSS start    : {mem_start:.1f} MB")
    print(f"    RSS end      : {mem_end:.1f} MB")
    print(f"    RSS delta    : {mem_end - mem_start:+.1f} MB")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
