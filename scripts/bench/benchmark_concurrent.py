"""Concurrent VAP + TurnGPT benchmark simulating real pipeline load.

In the actual pipeline, VAP runs at 10Hz on every audio frame while
TurnGPT runs at ~3Hz when ASR text changes. This script measures
inference latency when both models share CPU resources simultaneously.

Usage:
    uv run python scripts/bench/benchmark_concurrent.py --duration 60
    uv run python scripts/bench/benchmark_concurrent.py --duration 120 --vap-threads 2 --tgpt-threads 2
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from benchmark_compare import (  # noqa: E402
    CONVERSATIONS,
    BenchmarkResult,
    build_incremental_inputs,
    compute_metrics,
    generate_synthetic_stereo,
    get_rss_mb,
    numpy_to_pcm16,
)

from voice_pipeline.core.config import (  # noqa: E402
    AudioConfig,
    MaAIVAPConfig,
    TTSConfig,
    TurnGPTConfig,
)


@dataclass
class ConcurrentResult:
    vap: BenchmarkResult
    turngpt: BenchmarkResult
    total_duration_sec: float


def run_concurrent_benchmark(
    *,
    duration_sec: float,
    vap_ort_threads: int,
    tgpt_onnx_threads: int,
    vap_frame_rate: int = 10,
    tgpt_rate_hz: float = 3.0,
    warmup_sec: float = 5.0,
) -> ConcurrentResult:
    """Run VAP and TurnGPT concurrently, measuring latency under contention."""

    # --- Prepare audio for VAP ---
    sample_rate = 16000
    spf = sample_rate // vap_frame_rate
    total_audio_sec = duration_sec + warmup_sec + 10
    ch1, ch2 = generate_synthetic_stereo(total_audio_sec)

    # --- Prepare text inputs for TurnGPT ---
    all_inputs = build_incremental_inputs(CONVERSATIONS, chain=True)
    n_inputs = len(all_inputs)

    # --- Create models ---
    from voice_pipeline.turn_taking.maai_vap import MaAIVAPWrapper
    from voice_pipeline.turn_taking.turngpt import TurnGPTWrapper

    print("  Loading VAP (maai-full-onnx)...")
    vap_config = MaAIVAPConfig(
        frame_rate=vap_frame_rate,
        ort_threads=vap_ort_threads,
        encoder_onnx_path=MaAIVAPConfig.encoder_onnx_path,
        transformer_onnx_path=MaAIVAPConfig.transformer_onnx_path,
    )
    vap = MaAIVAPWrapper(vap_config, AudioConfig(), TTSConfig())

    print("  Loading TurnGPT (onnx-int8)...")
    tgpt_config = TurnGPTConfig(
        onnx_model_path="models/turngpt/turngpt_v2_kvcache_int8.onnx",
        tokenizer_path="models/turngpt/tokenizer",
        onnx_threads=tgpt_onnx_threads,
    )
    tgpt = TurnGPTWrapper(tgpt_config)
    print("  Models loaded.")

    # --- Warmup both ---
    warmup_vap_frames = int(warmup_sec * vap_frame_rate)
    warmup_tgpt_calls = int(warmup_sec * tgpt_rate_hz)

    print(f"  Warmup (VAP: {warmup_vap_frames} frames, TurnGPT: {warmup_tgpt_calls} calls)...")
    for i in range(warmup_vap_frames):
        idx = i % (len(ch1) // spf)
        s = idx * spf
        pcm1 = numpy_to_pcm16(ch1[s : s + spf])
        pcm2 = numpy_to_pcm16(ch2[s : s + spf])
        vap.feed_audio(pcm1, pcm2)
    for i in range(warmup_tgpt_calls):
        tgpt.predict(all_inputs[i % n_inputs])

    vap.reset()
    tgpt.reset()
    gc.collect()

    # --- Measurement arrays ---
    n_vap = int(duration_sec * vap_frame_rate)
    n_tgpt = int(duration_sec * tgpt_rate_hz)

    vap_latencies = np.zeros(n_vap, dtype=np.float64)
    vap_timestamps = np.zeros(n_vap, dtype=np.float64)
    tgpt_latencies = np.zeros(n_tgpt, dtype=np.float64)
    tgpt_timestamps = np.zeros(n_tgpt, dtype=np.float64)

    start_event = threading.Event()
    vap_actual = [0]
    tgpt_actual = [0]

    # --- VAP thread ---
    def vap_worker():
        interval = 1.0 / vap_frame_rate
        start_event.wait()
        t_start = time.perf_counter()
        audio_offset = warmup_vap_frames
        total_audio_frames = len(ch1) // spf

        for i in range(n_vap):
            idx = (audio_offset + i) % total_audio_frames
            s = idx * spf
            pcm1 = numpy_to_pcm16(ch1[s : s + spf])
            pcm2 = numpy_to_pcm16(ch2[s : s + spf])

            t0 = time.perf_counter()
            vap.feed_audio(pcm1, pcm2)
            t1 = time.perf_counter()

            vap_latencies[i] = (t1 - t0) * 1000.0
            vap_timestamps[i] = t1
            vap_actual[0] += 1

            # Real-time pacing
            elapsed = t1 - t_start
            expected = (i + 1) * interval
            if expected > elapsed:
                time.sleep(expected - elapsed)

    # --- TurnGPT thread ---
    def tgpt_worker():
        interval = 1.0 / tgpt_rate_hz
        start_event.wait()
        t_start = time.perf_counter()
        input_idx = 0

        for i in range(n_tgpt):
            text = all_inputs[input_idx % n_inputs]
            input_idx += 1

            if input_idx > 0 and input_idx % n_inputs == 0:
                tgpt.reset()

            t0 = time.perf_counter()
            tgpt.predict(text)
            t1 = time.perf_counter()

            tgpt_latencies[i] = (t1 - t0) * 1000.0
            tgpt_timestamps[i] = t1
            tgpt_actual[0] += 1

            elapsed = t1 - t_start
            expected = (i + 1) * interval
            if expected > elapsed:
                time.sleep(expected - elapsed)

    # --- Run both threads ---
    mem_start = get_rss_mb()
    t_vap = threading.Thread(target=vap_worker, name="vap-bench")
    t_tgpt = threading.Thread(target=tgpt_worker, name="tgpt-bench")

    t_vap.start()
    t_tgpt.start()

    print(f"  Measuring ({duration_sec}s, VAP@{vap_frame_rate}Hz + TurnGPT@{tgpt_rate_hz}Hz)...", end="", flush=True)
    wall_start = time.perf_counter()
    start_event.set()

    t_vap.join()
    t_tgpt.join()
    wall_end = time.perf_counter()
    mem_end = get_rss_mb()
    print(" done")

    # --- Compute metrics ---
    vap_result = compute_metrics(
        "maai-full-onnx (concurrent)",
        vap_latencies[: vap_actual[0]],
        1000.0 / vap_frame_rate,
        0.0,
        mem_start=mem_start,
        mem_end=mem_end,
        timestamps=vap_timestamps[: vap_actual[0]],
    )
    tgpt_result = compute_metrics(
        "turngpt-int8 (concurrent)",
        tgpt_latencies[: tgpt_actual[0]],
        1000.0 / tgpt_rate_hz,
        0.0,
        mem_start=mem_start,
        mem_end=mem_end,
        timestamps=tgpt_timestamps[: tgpt_actual[0]],
    )

    del vap, tgpt
    gc.collect()

    return ConcurrentResult(
        vap=vap_result,
        turngpt=tgpt_result,
        total_duration_sec=wall_end - wall_start,
    )


def print_concurrent_results(
    solo_vap: dict,
    solo_tgpt: dict,
    concurrent: ConcurrentResult,
    settings: dict,
) -> None:
    """Print comparison of solo vs concurrent performance."""
    sep = "=" * 76
    thin = "-" * 76

    print(f"\n{sep}")
    print("  Concurrent Benchmark: VAP + TurnGPT")
    print(sep)
    print(f"  Duration      : {settings['duration']}s")
    print(f"  VAP rate      : {settings['vap_frame_rate']}Hz (budget {1000 / settings['vap_frame_rate']:.0f}ms)")
    print(f"  TurnGPT rate  : {settings['tgpt_rate']}Hz (budget {1000 / settings['tgpt_rate']:.0f}ms)")
    print(f"  VAP threads   : {settings['vap_threads']}")
    print(f"  TurnGPT threads: {settings['tgpt_threads']}")
    print(sep)

    # VAP comparison
    print("\n  VAP (maai-full-onnx)")
    print(f"  {thin}")
    label_w = 18
    col_w = 22

    header = f"  {'Metric':<{label_w}}{'Solo':>{col_w}}{'Concurrent':>{col_w}}{'Delta':>{col_w}}"
    print(header)
    print(f"  {thin}")

    sv = solo_vap
    cv = concurrent.vap
    rows = [
        ("Calls", f"{sv['n_calls']}", f"{cv.n_calls}", ""),
        ("Mean", f"{sv['mean_ms']:.1f}ms", f"{cv.mean_ms:.1f}ms", f"+{cv.mean_ms - sv['mean_ms']:.1f}ms"),
        ("Median", f"{sv['median_ms']:.1f}ms", f"{cv.median_ms:.1f}ms", f"+{cv.median_ms - sv['median_ms']:.1f}ms"),
        ("P95", f"{sv['p95_ms']:.1f}ms", f"{cv.p95_ms:.1f}ms", f"+{cv.p95_ms - sv['p95_ms']:.1f}ms"),
        ("P99", f"{sv['p99_ms']:.1f}ms", f"{cv.p99_ms:.1f}ms", f"+{cv.p99_ms - sv['p99_ms']:.1f}ms"),
        ("Max", f"{sv['max_ms']:.1f}ms", f"{cv.max_ms:.1f}ms", f"+{cv.max_ms - sv['max_ms']:.1f}ms"),
        (">Budget", f"{sv['over_budget_pct']:.1f}%", f"{cv.over_budget_pct:.1f}%", ""),
    ]
    for label, s_val, c_val, delta in rows:
        print(f"  {label:<{label_w}}{s_val:>{col_w}}{c_val:>{col_w}}{delta:>{col_w}}")

    # TurnGPT comparison
    print("\n  TurnGPT (onnx-int8)")
    print(f"  {thin}")
    header = f"  {'Metric':<{label_w}}{'Solo':>{col_w}}{'Concurrent':>{col_w}}{'Delta':>{col_w}}"
    print(header)
    print(f"  {thin}")

    st = solo_tgpt
    ct = concurrent.turngpt
    rows = [
        ("Calls", f"{st['n_calls']}", f"{ct.n_calls}", ""),
        ("Mean", f"{st['mean_ms']:.1f}ms", f"{ct.mean_ms:.1f}ms", f"+{ct.mean_ms - st['mean_ms']:.1f}ms"),
        ("Median", f"{st['median_ms']:.1f}ms", f"{ct.median_ms:.1f}ms", f"+{ct.median_ms - st['median_ms']:.1f}ms"),
        ("P95", f"{st['p95_ms']:.1f}ms", f"{ct.p95_ms:.1f}ms", f"+{ct.p95_ms - st['p95_ms']:.1f}ms"),
        ("P99", f"{st['p99_ms']:.1f}ms", f"{ct.p99_ms:.1f}ms", f"+{ct.p99_ms - st['p99_ms']:.1f}ms"),
        ("Max", f"{st['max_ms']:.1f}ms", f"{ct.max_ms:.1f}ms", f"+{ct.max_ms - st['max_ms']:.1f}ms"),
        (">Budget", f"{st['over_budget_pct']:.1f}%", f"{ct.over_budget_pct:.1f}%", ""),
    ]
    for label, s_val, c_val, delta in rows:
        print(f"  {label:<{label_w}}{s_val:>{col_w}}{c_val:>{col_w}}{delta:>{col_w}}")

    # Budget headroom summary
    print("\n  Budget Headroom Summary")
    print(f"  {thin}")
    vap_budget = 1000 / settings["vap_frame_rate"]
    tgpt_budget = 1000 / settings["tgpt_rate"]
    vap_headroom = vap_budget - cv.p99_ms
    tgpt_headroom = tgpt_budget - ct.p99_ms
    print(f"  VAP    P99={cv.p99_ms:.1f}ms / budget={vap_budget:.0f}ms → headroom {vap_headroom:.0f}ms ({vap_headroom / vap_budget * 100:.0f}%)")
    print(f"  TurnGPT P99={ct.p99_ms:.1f}ms / budget={tgpt_budget:.0f}ms → headroom {tgpt_headroom:.0f}ms ({tgpt_headroom / tgpt_budget * 100:.0f}%)")

    # Can we raise VAP to 20Hz?
    vap_20hz_budget = 50.0
    vap_20hz_headroom = vap_20hz_budget - cv.p99_ms
    print(f"\n  VAP@20Hz feasibility: P99={cv.p99_ms:.1f}ms / budget=50ms → headroom {vap_20hz_headroom:.0f}ms ({'OK' if vap_20hz_headroom > 0 else 'NO'})")

    print(f"\n{sep}\n")


def main() -> None:
    import json

    parser = argparse.ArgumentParser(description="Concurrent VAP + TurnGPT benchmark")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--vap-threads", type=int, default=1)
    parser.add_argument("--tgpt-threads", type=int, default=2)
    parser.add_argument("--vap-rate", type=int, default=10)
    parser.add_argument("--tgpt-rate", type=float, default=3.0)
    args = parser.parse_args()

    settings = {
        "duration": args.duration,
        "vap_frame_rate": args.vap_rate,
        "tgpt_rate": args.tgpt_rate,
        "vap_threads": args.vap_threads,
        "tgpt_threads": args.tgpt_threads,
    }

    # Load solo results (must run benchmark_compare.py first)
    vap_solo_file = f"bench_vap_t{args.vap_threads}.json"
    tgpt_solo_file = f"bench_tgpt_t{args.tgpt_threads}.json"

    solo_vap = None
    solo_tgpt = None
    for fname, label in [(vap_solo_file, "VAP"), (tgpt_solo_file, "TurnGPT")]:
        if os.path.exists(fname):
            with open(fname) as f:
                data = json.load(f)
            if label == "VAP":
                solo_vap = data["results"][0]
            else:
                solo_tgpt = data["results"][0]
        else:
            print(f"  Warning: {fname} not found. Run solo benchmark first for {label} comparison.")

    # Run concurrent benchmark
    print(f"\n  Concurrent benchmark: VAP@{args.vap_rate}Hz + TurnGPT@{args.tgpt_rate}Hz")
    result = run_concurrent_benchmark(
        duration_sec=args.duration,
        vap_ort_threads=args.vap_threads,
        tgpt_onnx_threads=args.tgpt_threads,
        vap_frame_rate=args.vap_rate,
        tgpt_rate_hz=args.tgpt_rate,
    )

    if solo_vap and solo_tgpt:
        print_concurrent_results(solo_vap, solo_tgpt, result, settings)
    else:
        # Fallback: just print concurrent results
        print(f"\n  VAP (concurrent): mean={result.vap.mean_ms:.1f}ms, P95={result.vap.p95_ms:.1f}ms, P99={result.vap.p99_ms:.1f}ms")
        print(f"  TurnGPT (concurrent): mean={result.turngpt.mean_ms:.1f}ms, P95={result.turngpt.p95_ms:.1f}ms, P99={result.turngpt.p99_ms:.1f}ms")


if __name__ == "__main__":
    main()
