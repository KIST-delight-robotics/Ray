"""Load test: VAP (MaAI ONNX) + TurnGPT (ONNX) running concurrently.

Simulates real pipeline load by running both models in parallel threads
with real CANDOR audio input. Measures per-frame latency for each model
and reports CPU utilization.

Usage:
    uv run python scripts/bench/load_test_vap_turngpt.py \
        --audio CANDOR/raw_media_part_001/23d4ec0e-.../processed/23d4ec0e-...mp3 \
        --duration 60
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import numpy as np
import psutil
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from voice_pipeline.turn_taking.turngpt import TurnGPTWrapper
from voice_pipeline.core.config import TurnGPTConfig

sys.path.insert(0, os.path.dirname(__file__))
from vap_onnx_pipeline import VapOnnxPipeline


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


def vap_worker(
    pipeline: VapOnnxPipeline,
    ch1: np.ndarray,
    ch2: np.ndarray,
    frame_rate: int,
    duration: float,
    results: dict,
    stop_event: threading.Event,
):
    """Run VAP inference at real-time pace."""
    spf = 16000 // frame_rate
    n_frames = min(int(duration * frame_rate), len(ch1) // spf)
    interval = 1.0 / frame_rate

    lats = []
    for i in range(n_frames):
        if stop_event.is_set():
            break

        t_frame_start = time.perf_counter()

        x1 = ch1[i * spf : (i + 1) * spf]
        x2 = ch2[i * spf : (i + 1) * spf]

        t0 = time.perf_counter()
        pipeline.process(x1, x2)
        lats.append(time.perf_counter() - t0)

        # Pace to real-time
        elapsed = time.perf_counter() - t_frame_start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    results["vap"] = np.array(lats) * 1000


def turngpt_worker(
    turngpt: TurnGPTWrapper,
    duration: float,
    results: dict,
    stop_event: threading.Event,
):
    """Simulate TurnGPT predictions at ~3 calls/sec (typical ASR interim rate)."""
    # Simulated conversation turns
    dialogs = [
        "Hello how are you doing today",
        "Hello how are you doing today <ts> I'm doing great thanks for asking",
        "Hello how are you doing today <ts> I'm doing great thanks for asking <ts> That's wonderful to hear",
        "Hello how are you doing today <ts> I'm doing great thanks for asking <ts> That's wonderful to hear <ts> Yeah it's been a really nice day so far",
        "Hello how are you doing today <ts> I'm doing great thanks for asking <ts> That's wonderful to hear <ts> Yeah it's been a really nice day so far <ts> I agree the weather has been perfect",
    ]

    interval = 0.33  # ~3 Hz (ASR interim updates)
    lats = []
    t_start = time.perf_counter()

    turn_idx = 0
    word_idx = 0

    while time.perf_counter() - t_start < duration:
        if stop_event.is_set():
            break

        t_frame_start = time.perf_counter()

        # Build incremental text (simulating ASR streaming)
        dialog = dialogs[turn_idx % len(dialogs)]
        words = dialog.split()
        partial = " ".join(words[: word_idx + 1])
        word_idx += 1
        if word_idx >= len(words):
            word_idx = 0
            turn_idx += 1

        t0 = time.perf_counter()
        turngpt.predict(partial)
        lats.append(time.perf_counter() - t0)

        elapsed = time.perf_counter() - t_frame_start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    results["turngpt"] = np.array(lats) * 1000


def cpu_monitor(interval: float, duration: float, results: dict, stop_event: threading.Event):
    """Sample CPU usage periodically."""
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


def main():
    parser = argparse.ArgumentParser(description="Load test: VAP + TurnGPT concurrent")
    parser.add_argument("--audio", required=True, help="Stereo audio file (CANDOR mp3)")
    parser.add_argument("--duration", type=float, default=60, help="Test duration in seconds")
    args = parser.parse_args()

    print("=" * 70)
    print("  Load Test: VAP (MaAI ONNX) + TurnGPT (ONNX) Concurrent")
    print("=" * 70)
    print(f"  Duration    : {args.duration}s")
    print(f"  CPU cores   : {os.cpu_count()}")

    # Load audio
    print("\n  Loading audio...")
    ch1, ch2 = load_stereo_audio(os.path.abspath(args.audio))
    print(f"  Audio       : {len(ch1) / 16000:.1f}s")

    # Load VAP (pt=1, ort=1)
    print("\n  Loading VAP (MaAI ONNX, pt=1, ort=1)...")
    torch.set_num_threads(1)
    vap = VapOnnxPipeline(frame_rate=10, context_len_sec=5.0, ort_threads=1)

    # Warmup VAP
    spf = 1600
    for i in range(50):
        vap.process(ch1[i * spf : (i + 1) * spf], ch2[i * spf : (i + 1) * spf])
    vap.reset()

    # Load TurnGPT (ONNX, 2 threads)
    print("  Loading TurnGPT (ONNX KV-cache, 2 threads)...")
    turngpt_config = TurnGPTConfig(
        onnx_model_path="models/turngpt/turngpt_v2_kvcache.onnx",
        tokenizer_path="models/turngpt/tokenizer",
        onnx_threads=2,
    )
    turngpt = TurnGPTWrapper(turngpt_config)

    # Warmup TurnGPT
    turngpt.predict("Hello how are you")
    turngpt.reset()

    results: dict = {}
    stop_event = threading.Event()

    # Start threads
    print(f"\n  Starting {args.duration}s load test...")
    t_start = time.perf_counter()

    threads = [
        threading.Thread(
            target=vap_worker,
            args=(vap, ch1, ch2, 10, args.duration, results, stop_event),
            name="vap",
        ),
        threading.Thread(
            target=turngpt_worker,
            args=(turngpt, args.duration, results, stop_event),
            name="turngpt",
        ),
        threading.Thread(
            target=cpu_monitor,
            args=(1.0, args.duration + 5, results, stop_event),
            name="cpu_mon",
        ),
    ]

    for t in threads:
        t.daemon = True
        t.start()

    # Wait for workers
    threads[0].join(timeout=args.duration + 30)
    threads[1].join(timeout=args.duration + 30)
    stop_event.set()
    threads[2].join(timeout=5)

    elapsed = time.perf_counter() - t_start

    # Results
    print(f"\n{'=' * 70}")
    print(f"  Results ({elapsed:.1f}s elapsed)")
    print(f"{'=' * 70}")

    if "vap" in results:
        v = results["vap"]
        print(f"\n  VAP (10Hz, budget=100ms):")
        print(f"    Frames    : {len(v)}")
        print(f"    Mean      : {v.mean():.1f}ms")
        print(f"    Median    : {np.median(v):.1f}ms")
        print(f"    P95       : {np.percentile(v, 95):.1f}ms")
        print(f"    P99       : {np.percentile(v, 99):.1f}ms")
        print(f"    Max       : {v.max():.1f}ms")
        over_budget = (v > 100).sum()
        print(f"    >100ms    : {over_budget}/{len(v)} ({100*over_budget/len(v):.1f}%)")

    if "turngpt" in results:
        g = results["turngpt"]
        print(f"\n  TurnGPT (~3Hz):")
        print(f"    Calls     : {len(g)}")
        print(f"    Mean      : {g.mean():.1f}ms")
        print(f"    Median    : {np.median(g):.1f}ms")
        print(f"    P95       : {np.percentile(g, 95):.1f}ms")
        print(f"    Max       : {g.max():.1f}ms")

    if "cpu_overall" in results:
        cpu = results["cpu_overall"]
        per_cpu = results["cpu_per_core"]
        if cpu:
            print(f"\n  CPU Usage:")
            print(f"    Overall   : mean={np.mean(cpu):.1f}%  max={np.max(cpu):.1f}%")
            if per_cpu:
                per_core_means = np.mean(per_cpu, axis=0)
                for i, m in enumerate(per_core_means):
                    print(f"    Core {i}    : mean={m:.1f}%")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
