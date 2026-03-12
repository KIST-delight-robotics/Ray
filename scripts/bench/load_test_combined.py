"""Load test: VAP + TurnGPT concurrent on Raspberry Pi.

Runs both models in parallel threads with synthetic audio, simulating
the real pipeline. Measures per-inference latency and CPU usage,
then prints histograms.

Usage:
    uv run python scripts/bench/load_test_combined.py
    uv run python scripts/bench/load_test_combined.py --duration 30 --vap-no-compile
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
import traceback

import numpy as np
import psutil
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from voice_pipeline.turn_taking.turngpt import TurnGPTWrapper
from voice_pipeline.core.config import TurnGPTConfig

sys.path.insert(0, os.path.dirname(__file__))
from vap_onnx_pipeline import VapOnnxPipeline


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------


def print_histogram(
    lats: np.ndarray, label: str, bin_width: float = 5, max_bar: int = 50
) -> None:
    # Clip to P99 for display, note outliers separately
    p99 = float(np.percentile(lats, 99))
    clipped = lats[lats <= p99]
    n_outliers = len(lats) - len(clipped)

    lo = int(clipped.min() // bin_width) * bin_width
    hi = int(math.ceil(clipped.max() / bin_width)) * bin_width
    bins = list(range(lo, hi + bin_width, bin_width))
    counts = [0] * (len(bins) - 1)
    for v in clipped:
        idx = min(int((v - lo) / bin_width), len(counts) - 1)
        counts[idx] += 1

    max_count = max(counts) if counts else 1

    print(f"\n  {label} (n={len(lats)}, bin={bin_width:.0f}ms)")
    print(f"  {'ms':>8}  {'count':>5}  distribution")
    print(f"  {'-'*8}  {'-'*5}  {'-'*max_bar}")
    for i, c in enumerate(counts):
        bar_len = int(c / max_count * max_bar)
        bar = "#" * bar_len
        rng = f"{bins[i]:>3}-{bins[i+1]:<3}"
        print(f"  {rng:>8}  {c:>5}  {bar}")
    if n_outliers > 0:
        print(f"  {'(tail)':>8}  {n_outliers:>5}  (>{p99:.0f}ms, clipped)")


def print_stats(lats: np.ndarray, label: str, budget_ms: float | None = None) -> None:
    print(f"\n  {label}:")
    print(f"    Count     : {len(lats)}")
    print(f"    Mean      : {lats.mean():.1f}ms")
    print(f"    Median    : {np.median(lats):.1f}ms")
    print(f"    P5        : {np.percentile(lats, 5):.1f}ms")
    print(f"    P95       : {np.percentile(lats, 95):.1f}ms")
    print(f"    Min       : {lats.min():.1f}ms")
    print(f"    Max       : {lats.max():.1f}ms")
    if budget_ms is not None:
        over = (lats > budget_ms).sum()
        print(f"    >{budget_ms:.0f}ms    : {over}/{len(lats)} ({100*over/len(lats):.1f}%)")


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


def vap_worker(
    pipeline: VapOnnxPipeline,
    frame_rate: int,
    duration: float,
    results: dict,
    stop_event: threading.Event,
) -> None:
    try:
        spf = 16000 // frame_rate
        interval = 1.0 / frame_rate
        # Cache already at stable shape from main-thread warmup

        lats = []
        n_frames = int(duration * frame_rate)
        for i in range(n_frames):
            if stop_event.is_set():
                break

            t_frame_start = time.perf_counter()
            x = np.random.randn(spf).astype(np.float32) * 0.01

            t0 = time.perf_counter()
            pipeline.process(x, np.zeros_like(x))
            lats.append((time.perf_counter() - t0) * 1000)

            elapsed = time.perf_counter() - t_frame_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        results["vap"] = np.array(lats) if lats else np.array([])
    except Exception:
        traceback.print_exc()
        results["vap_error"] = traceback.format_exc()


def turngpt_worker(
    turngpt: TurnGPTWrapper,
    duration: float,
    results: dict,
    stop_event: threading.Event,
) -> None:
    try:
        dialogs = [
            "Hello how are you doing today",
            "Hello how are you doing today <ts> I'm doing great thanks for asking",
            "Hello how are you doing today <ts> I'm doing great thanks for asking <ts> That's wonderful",
            "Hello how are you doing today <ts> I'm doing great thanks for asking <ts> That's wonderful <ts> Yeah it's been a really nice day so far",
            "Hello how are you doing today <ts> I'm doing great thanks for asking <ts> That's wonderful <ts> Yeah it's been a really nice day so far <ts> I agree the weather has been perfect",
        ]

        interval = 0.33  # ~3 Hz
        lats = []
        t_start = time.perf_counter()
        turn_idx = 0
        word_idx = 0

        while time.perf_counter() - t_start < duration:
            if stop_event.is_set():
                break

            t_frame_start = time.perf_counter()

            dialog = dialogs[turn_idx % len(dialogs)]
            words = dialog.split()
            partial = " ".join(words[: word_idx + 1])
            word_idx += 1
            if word_idx >= len(words):
                word_idx = 0
                turn_idx += 1

            t0 = time.perf_counter()
            turngpt.predict(partial)
            lats.append((time.perf_counter() - t0) * 1000)

            elapsed = time.perf_counter() - t_frame_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        results["turngpt"] = np.array(lats) if lats else np.array([])
    except Exception:
        traceback.print_exc()
        results["turngpt_error"] = traceback.format_exc()


def cpu_monitor(
    interval: float, duration: float, results: dict, stop_event: threading.Event
) -> None:
    samples = []
    per_cpu_samples = []
    t_start = time.perf_counter()

    while time.perf_counter() - t_start < duration:
        if stop_event.is_set():
            break
        cpu = psutil.cpu_percent(interval=interval, percpu=False)
        per_cpu = psutil.cpu_percent(interval=0, percpu=True)
        samples.append(cpu)
        per_cpu_samples.append(per_cpu)

    results["cpu_overall"] = samples
    results["cpu_per_core"] = per_cpu_samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Load test: VAP + TurnGPT concurrent")
    parser.add_argument("--duration", type=float, default=30, help="Test duration (default: 30s)")
    parser.add_argument("--vap-compile", action="store_true", help="Enable torch.compile for VAP")
    parser.add_argument("--vap-no-compile", action="store_true", help="Disable torch.compile for VAP")
    parser.add_argument("--turngpt-model", default="models/turngpt/turngpt_v2_kvcache_int8.onnx")
    parser.add_argument("--turngpt-threads", type=int, default=2)
    parser.add_argument("--vap-frame-rate", type=int, default=10)
    parser.add_argument("--vap-context", type=float, default=5.0)
    args = parser.parse_args()

    use_compile = not args.vap_no_compile  # default: ON
    if args.vap_compile:
        use_compile = True

    print("=" * 70)
    print("  Load Test: VAP (MaAI) + TurnGPT (ONNX) Concurrent")
    print("=" * 70)
    print(f"  Duration       : {args.duration}s")
    print(f"  CPU cores      : {os.cpu_count()}")
    print(f"  VAP frame rate : {args.vap_frame_rate}Hz")
    print(f"  VAP context    : {args.vap_context}s")
    print(f"  VAP compile    : {use_compile}")
    print(f"  TurnGPT model  : {args.turngpt_model}")
    print(f"  TurnGPT threads: {args.turngpt_threads}")
    print(f"  PyTorch        : {torch.__version__}")

    # Load VAP
    print("\n  Loading VAP...")
    torch.set_num_threads(1)
    vap = VapOnnxPipeline(
        frame_rate=args.vap_frame_rate,
        context_len_sec=args.vap_context,
        ort_threads=1,
    )

    # Load TurnGPT
    print("  Loading TurnGPT...")
    turngpt_config = TurnGPTConfig(
        onnx_model_path=args.turngpt_model,
        tokenizer_path="models/turngpt/tokenizer",
        onnx_threads=args.turngpt_threads,
    )
    turngpt = TurnGPTWrapper(turngpt_config)

    # Compile + warmup VAP (slow first time, do before timed test)
    if use_compile:
        print(f"\n  Compiling + warming up VAP (this takes ~90s on first run)...")
        vap.vap.forward = torch.compile(vap.vap.forward, mode="default")
    else:
        print(f"\n  Warming up VAP...")

    t_warm = time.perf_counter()
    spf = 16000 // args.vap_frame_rate
    dummy = np.random.randn(spf).astype(np.float32) * 0.1
    for _ in range(80):
        vap.process(dummy, np.zeros_like(dummy))
    # Don't reset — keep cache at stable shape to avoid recompilation in worker
    print(f"  Warmup done in {time.perf_counter() - t_warm:.1f}s")

    # Warmup TurnGPT
    turngpt.predict("Hello how are you")
    turngpt.reset()

    results: dict = {}
    stop_event = threading.Event()

    print(f"\n  Running {args.duration}s load test...")
    t_start = time.perf_counter()

    threads = [
        threading.Thread(
            target=vap_worker,
            args=(vap, args.vap_frame_rate, args.duration, results, stop_event),
            name="vap",
        ),
        threading.Thread(
            target=turngpt_worker,
            args=(turngpt, args.duration, results, stop_event),
            name="turngpt",
        ),
        threading.Thread(
            target=cpu_monitor,
            args=(1.0, args.duration + 10, results, stop_event),
            name="cpu_mon",
        ),
    ]

    for t in threads:
        t.start()

    threads[0].join(timeout=args.duration + 60)
    threads[1].join(timeout=args.duration + 30)
    stop_event.set()
    threads[2].join(timeout=5)

    elapsed = time.perf_counter() - t_start

    # Results
    print(f"\n{'=' * 70}")
    print(f"  Results ({elapsed:.1f}s elapsed)")
    print(f"{'=' * 70}")

    budget_vap = 1000.0 / args.vap_frame_rate

    if "vap_error" in results:
        print(f"\n  VAP ERROR:\n{results['vap_error']}")
    if "vap" in results and len(results["vap"]) > 0:
        v = results["vap"]
        print_stats(v, f"VAP ({args.vap_frame_rate}Hz, compile={use_compile})", budget_vap)
        print_histogram(v, "VAP Latency", bin_width=5)
    elif "vap_error" not in results:
        print("\n  VAP: no results collected")

    if "turngpt_error" in results:
        print(f"\n  TurnGPT ERROR:\n{results['turngpt_error']}")
    if "turngpt" in results and len(results["turngpt"]) > 0:
        g = results["turngpt"]
        print_stats(g, "TurnGPT (~3Hz, ONNX KV-cache int8)")
        print_histogram(g, "TurnGPT Latency", bin_width=10)
    elif "turngpt_error" not in results:
        print("\n  TurnGPT: no results collected")

    if "cpu_overall" in results and results["cpu_overall"]:
        cpu = results["cpu_overall"]
        per_cpu = results["cpu_per_core"]
        print(f"\n  CPU Usage (during test):")
        print(f"    Overall   : mean={np.mean(cpu):.1f}%  max={np.max(cpu):.1f}%")
        if per_cpu:
            per_core_means = np.mean(per_cpu, axis=0)
            per_core_maxs = np.max(per_cpu, axis=0)
            for i, (m, mx) in enumerate(zip(per_core_means, per_core_maxs)):
                print(f"    Core {i}    : mean={m:.1f}%  max={mx:.1f}%")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
