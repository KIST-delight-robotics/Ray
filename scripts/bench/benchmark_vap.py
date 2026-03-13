"""Benchmark VAP inference latency on the current hardware.

Measures per-inference latency under various context buffer sizes and reports
statistics (mean, median, min, max, p95, p99) plus real-time factor.

Usage:
    VAP_MODEL_PATH=/path/to/vap.pt uv run python scripts/bench/benchmark_vap.py

Optional env vars:
    VAP_DEVICE       - torch device (default: "cpu")
    VAP_WARMUP       - number of warmup inferences (default: 5)
    VAP_ITERATIONS   - number of timed inferences per config (default: 50)
"""

from __future__ import annotations

import os
import struct
import sys
import time

import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
FRAME_SAMPLES = 480  # 30ms @ 16kHz
STEP_SAMPLES = 1600  # 0.1s @ 16kHz  (= 100ms step)
FRAMES_PER_INFERENCE = (STEP_SAMPLES + FRAME_SAMPLES - 1) // FRAME_SAMPLES  # ceil


def pcm_tone(n_samples: int = FRAME_SAMPLES, amplitude: int = 10000) -> bytes:
    """Generate a constant-amplitude PCM16 frame."""
    return struct.pack(f"<{n_samples}h", *([amplitude] * n_samples))


def percentile(sorted_vals: list[float], p: float) -> float:
    """Simple percentile on a pre-sorted list."""
    idx = (len(sorted_vals) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def print_stats(label: str, latencies: list[float], step_sec: float) -> None:
    """Print latency statistics for a benchmark run."""
    s = sorted(latencies)
    mean = sum(s) / len(s)
    median = percentile(s, 0.5)
    p95 = percentile(s, 0.95)
    p99 = percentile(s, 0.99)
    realtime_factor = step_sec / mean if mean > 0 else float("inf")

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Inferences : {len(s)}")
    print(f"  Mean        : {mean * 1000:8.1f} ms")
    print(f"  Median      : {median * 1000:8.1f} ms")
    print(f"  Min         : {s[0] * 1000:8.1f} ms")
    print(f"  Max         : {s[-1] * 1000:8.1f} ms")
    print(f"  P95         : {p95 * 1000:8.1f} ms")
    print(f"  P99         : {p99 * 1000:8.1f} ms")
    print(f"  Step budget : {step_sec * 1000:8.1f} ms")
    rt_status = "OK" if realtime_factor >= 1.0 else "TOO SLOW"
    print(f"  RT factor   : {realtime_factor:8.2f}x  {rt_status}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------


def benchmark_raw_inference(
    model: torch.nn.Module,
    device: str,
    context_sec: float,
    step_sec: float,
    warmup: int,
    iterations: int,
) -> list[float]:
    """Benchmark raw model.probs() calls with a pre-filled buffer.

    This isolates model inference from PCM decoding and buffer management.
    """
    n_samples = round(context_sec * SAMPLE_RATE)
    buf = torch.randn(1, 2, n_samples)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model.probs(buf.to(device))

    # Timed runs
    latencies: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        with torch.no_grad():
            model.probs(buf.to(device))
        latencies.append(time.perf_counter() - t0)

    return latencies


def benchmark_feed_audio(
    context_sec: float,
    step_sec: float,
    model_path: str,
    device: str,
    warmup: int,
    iterations: int,
) -> list[float]:
    """Benchmark full feed_audio pipeline (PCM decode + buffer roll + inference).

    Creates a fresh VAPWrapper to test the end-to-end path.
    """
    from voice_pipeline.core.config import AudioConfig, TTSConfig, VAPConfig
    from voice_pipeline.turn_taking.vap import VAPWrapper

    vap_cfg = VAPConfig(
        model_path=model_path,
        context_sec=context_sec,
        step_sec=step_sec,
        device=device,
    )
    audio_cfg = AudioConfig(sample_rate=SAMPLE_RATE, channels=1, frame_duration_ms=30)
    tts_cfg = TTSConfig(output_sample_rate=24000)

    wrapper = VAPWrapper(vap_cfg, audio_cfg, tts_cfg)
    frame = pcm_tone()

    step_samples = round(step_sec * SAMPLE_RATE)
    frames_per_step = max(1, (step_samples + FRAME_SAMPLES - 1) // FRAME_SAMPLES)

    # Warmup
    for _ in range(warmup):
        wrapper.reset()
        for _ in range(frames_per_step):
            wrapper.feed_audio(frame)

    # Timed runs — measure the batch of frames that triggers one inference
    latencies: list[float] = []
    for _ in range(iterations):
        wrapper.reset()
        # Feed frames until just before inference triggers
        for _ in range(frames_per_step - 1):
            wrapper.feed_audio(frame)
        # The last frame triggers inference — measure it
        t0 = time.perf_counter()
        wrapper.feed_audio(frame)
        latencies.append(time.perf_counter() - t0)

    return latencies


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    model_path = os.environ.get("VAP_MODEL_PATH", "")
    if not model_path:
        print("ERROR: Set VAP_MODEL_PATH environment variable.")
        sys.exit(1)
    if not os.path.isfile(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)

    device = os.environ.get("VAP_DEVICE", "cpu")
    warmup = int(os.environ.get("VAP_WARMUP", "5"))
    iterations = int(os.environ.get("VAP_ITERATIONS", "50"))

    print("VAP Benchmark")
    print(f"  Model     : {model_path}")
    print(f"  Device    : {device}")
    print(f"  Warmup    : {warmup}")
    print(f"  Iterations: {iterations}")
    print(f"  Platform  : {sys.platform}")

    # Check for torch optimizations
    print(f"  PyTorch   : {torch.__version__}")
    print(f"  Threads   : {torch.get_num_threads()}")

    # Load model once for raw inference benchmarks
    print("\nLoading VAP model...")
    t0 = time.perf_counter()
    from vap.model import VapConfig, VapGPT

    model = VapGPT(VapConfig())
    sd = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    load_time = time.perf_counter() - t0
    print(f"  Model loaded in {load_time:.2f}s")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # -----------------------------------------------------------------------
    # 1. Raw inference with different context sizes
    # -----------------------------------------------------------------------
    step_sec = 0.1
    context_sizes = [5.0, 10.0, 20.0]

    print(f"\n{'#' * 60}")
    print(f"  Part 1: Raw model.probs() latency (step={step_sec}s)")
    print(f"{'#' * 60}")

    for ctx in context_sizes:
        lats = benchmark_raw_inference(model, device, ctx, step_sec, warmup, iterations)
        print_stats(f"Raw inference | context={ctx}s", lats, step_sec)

    # -----------------------------------------------------------------------
    # 2. Full feed_audio pipeline with default context
    # -----------------------------------------------------------------------
    print(f"\n{'#' * 60}")
    print(f"  Part 2: Full feed_audio() pipeline (step={step_sec}s)")
    print(f"{'#' * 60}")

    for ctx in [5.0, 20.0]:
        lats = benchmark_feed_audio(ctx, step_sec, model_path, device, warmup, iterations)
        print_stats(f"feed_audio() | context={ctx}s", lats, step_sec)

    # -----------------------------------------------------------------------
    # 3. Sustained streaming simulation
    # -----------------------------------------------------------------------
    print(f"\n{'#' * 60}")
    print("  Part 3: Sustained streaming (30s simulated audio)")
    print(f"{'#' * 60}")

    from voice_pipeline.core.config import AudioConfig, TTSConfig, VAPConfig
    from voice_pipeline.turn_taking.vap import VAPWrapper

    ctx = 5.0
    vap_cfg = VAPConfig(model_path=model_path, context_sec=ctx, step_sec=step_sec, device=device)
    audio_cfg = AudioConfig(sample_rate=SAMPLE_RATE, channels=1, frame_duration_ms=30)
    tts_cfg = TTSConfig(output_sample_rate=24000)
    wrapper = VAPWrapper(vap_cfg, audio_cfg, tts_cfg)

    frame = pcm_tone()
    sim_duration = 30.0
    n_frames = int(sim_duration / 0.030)

    inference_latencies: list[float] = []
    total_start = time.perf_counter()
    samples_fed = 0
    for _i in range(n_frames):
        t0 = time.perf_counter()
        wrapper.feed_audio(frame)
        elapsed = time.perf_counter() - t0
        samples_fed += FRAME_SAMPLES
        # Record only frames that triggered inference (elapsed > 1ms heuristic)
        if elapsed > 0.001:
            inference_latencies.append(elapsed)
    total_elapsed = time.perf_counter() - total_start

    audio_duration = samples_fed / SAMPLE_RATE
    overall_rtf = audio_duration / total_elapsed

    print(f"\n  Simulated   : {audio_duration:.1f}s of audio")
    print(f"  Wall clock  : {total_elapsed:.2f}s")
    print(f"  Overall RTF : {overall_rtf:.2f}x  {'OK' if overall_rtf >= 1.0 else 'TOO SLOW'}")
    print(f"  Inferences  : {len(inference_latencies)}")
    if inference_latencies:
        print_stats(
            f"Inference-only frames | context={ctx}s, stream={sim_duration}s",
            inference_latencies,
            step_sec,
        )

    # -----------------------------------------------------------------------
    # 4. Different step_sec values (find feasible step)
    # -----------------------------------------------------------------------
    print(f"\n{'#' * 60}")
    print("  Part 4: Finding feasible step_sec (context=5s)")
    print(f"{'#' * 60}")

    # Use the raw inference result for context=5s as baseline
    lats_5s = benchmark_raw_inference(model, device, 5.0, 0.1, warmup, 20)
    mean_inference = sum(lats_5s) / len(lats_5s)

    step_candidates = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    print(f"\n  Mean inference latency (5s context): {mean_inference * 1000:.1f} ms")
    print(f"\n  {'Step (s)':<10} {'Budget (ms)':<12} {'RTF':<8} {'Feasible'}")
    print(f"  {'-' * 44}")
    for s in step_candidates:
        rtf = s / mean_inference
        ok = "YES" if rtf >= 1.0 else "NO"
        print(f"  {s:<10.2f} {s * 1000:<12.0f} {rtf:<8.2f} {ok}")

    print("\nDone.")


if __name__ == "__main__":
    main()
