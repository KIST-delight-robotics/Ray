"""Unified benchmark for comparing turn-taking model variants.

Compares PyTorch / ONNX / quantized variants side-by-side with consistent
settings. Uses production wrappers directly — no reimplemented inference logic.

VAP variants:
  maai-pytorch     MaAI ONNX encoder + PyTorch transformer (eager)
  maai-compile     MaAI ONNX encoder + torch.compile transformer
  maai-full-onnx   MaAI full ONNX encoder + transformer (recommended)

TurnGPT variants:
  turngpt-pytorch    PyTorch checkpoint
  turngpt-onnx-fp32  ONNX FP32 with KV cache
  turngpt-onnx-int8  ONNX INT8 with KV cache (recommended)

Usage:
    # Compare all MaAI VAP variants with real audio
    uv run python scripts/bench/bench_turn_model.py \\
        --model vap --variants all --audio /path/to/candor.mp3 --duration 60

    # Compare TurnGPT ONNX fp32 vs int8
    uv run python scripts/bench/bench_turn_model.py \\
        --model turngpt --variants turngpt-onnx-fp32 turngpt-onnx-int8

    # Quick synthetic test
    uv run python scripts/bench/bench_turn_model.py \\
        --model vap --variants maai-full-onnx --duration 30
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from voice_pipeline.turn_taking.maai_vap import MaAIVAPWrapper  # noqa: E402

_DEFAULT_TOKENIZER_PATH = "models/turngpt/tokenizer"

# =====================================================================
# Data types
# =====================================================================


@dataclass
class BenchmarkResult:
    """Collected metrics for one variant run."""

    variant: str
    n_calls: int
    mean_ms: float
    median_ms: float
    p5_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    budget_ms: float
    rtf: float
    over_budget_count: int
    over_budget_pct: float
    load_time_sec: float
    mem_start_mb: float | None = None
    mem_end_mb: float | None = None
    mem_delta_mb: float | None = None
    # Per-30s drift buckets: list of (bucket_label, mean_ms, p95_ms)
    drift_buckets: list[tuple[str, float, float]] = field(default_factory=list)


def compute_metrics(
    variant: str,
    latencies_ms: np.ndarray,
    budget_ms: float,
    load_time_sec: float,
    *,
    mem_start: float | None = None,
    mem_end: float | None = None,
    timestamps: np.ndarray | None = None,
) -> BenchmarkResult:
    """Compute standardized metrics from raw latency array."""
    n = len(latencies_ms)
    over = int(np.sum(latencies_ms > budget_ms))
    mean = float(np.mean(latencies_ms))

    # Drift analysis: 30-second buckets
    drift: list[tuple[str, float, float]] = []
    if timestamps is not None and n > 0:
        t0 = timestamps[0]
        bucket_sec = 30.0
        max_t = timestamps[-1] - t0
        n_buckets = max(1, int(np.ceil(max_t / bucket_sec)))
        for b in range(n_buckets):
            lo = t0 + b * bucket_sec
            hi = lo + bucket_sec
            mask = (timestamps >= lo) & (timestamps < hi)
            if np.any(mask):
                bucket_lats = latencies_ms[mask]
                label = f"{b * 30}-{(b + 1) * 30}s"
                drift.append((label, float(np.mean(bucket_lats)), float(np.percentile(bucket_lats, 95))))

    return BenchmarkResult(
        variant=variant,
        n_calls=n,
        mean_ms=mean,
        median_ms=float(np.median(latencies_ms)),
        p5_ms=float(np.percentile(latencies_ms, 5)),
        p95_ms=float(np.percentile(latencies_ms, 95)),
        p99_ms=float(np.percentile(latencies_ms, 99)),
        min_ms=float(np.min(latencies_ms)),
        max_ms=float(np.max(latencies_ms)),
        std_ms=float(np.std(latencies_ms)),
        budget_ms=budget_ms,
        rtf=budget_ms / mean if mean > 0 else float("inf"),
        over_budget_count=over,
        over_budget_pct=100.0 * over / n if n > 0 else 0.0,
        load_time_sec=load_time_sec,
        mem_start_mb=mem_start,
        mem_end_mb=mem_end,
        mem_delta_mb=(mem_end - mem_start) if mem_start is not None and mem_end is not None else None,
        drift_buckets=drift,
    )


# =====================================================================
# Audio utilities
# =====================================================================


def load_stereo_audio(path: str, sample_rate: int = 16000) -> tuple[np.ndarray, np.ndarray]:
    """Load audio file and return (ch1, ch2) as float32 arrays at target sample rate."""
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)

    # Resample if needed
    if sr != sample_rate:
        try:
            import soxr

            data = soxr.resample(data, sr, sample_rate)
        except ImportError:
            # Linear interpolation fallback
            ratio = sample_rate / sr
            n_out = int(len(data) * ratio)
            indices = np.linspace(0, len(data) - 1, n_out)
            lo = indices.astype(np.int64)
            hi = np.minimum(lo + 1, len(data) - 1)
            frac = (indices - lo).astype(np.float32)[:, np.newaxis]
            data = data[lo] * (1 - frac) + data[hi] * frac

    if data.shape[1] == 1:
        # Mono: duplicate to both channels
        return data[:, 0], data[:, 0].copy()
    return data[:, 0], data[:, 1]


def numpy_to_pcm16(arr: np.ndarray) -> bytes:
    """Convert float32 numpy array to 16-bit PCM bytes."""
    int16 = (arr * 32768.0).clip(-32768, 32767).astype(np.int16)
    return int16.tobytes()


def generate_synthetic_stereo(duration_sec: float, sample_rate: int = 16000) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic stereo audio (low-amplitude noise)."""
    n = int(duration_sec * sample_rate)
    rng = np.random.default_rng(42)
    ch1 = rng.standard_normal(n).astype(np.float32) * 0.01
    ch2 = rng.standard_normal(n).astype(np.float32) * 0.01
    return ch1, ch2


# =====================================================================
# TurnGPT dialog data
# =====================================================================

CONVERSATIONS = [
    [
        "hello how are you doing today",
        "i'm doing great thanks for asking how about you",
        "pretty good just been busy with work lately",
        "oh yeah what kind of work do you do",
        "i work in software engineering mostly backend stuff",
        "that sounds interesting do you enjoy it",
        "yeah i really do it's challenging but rewarding",
        "i can imagine what languages do you use",
        "mostly python and go sometimes rust for performance critical stuff",
        "nice i've been wanting to learn rust actually",
    ],
    [
        "hey did you watch the game last night",
        "no i missed it who won",
        "the home team pulled it off in overtime it was incredible",
        "oh man i wish i had seen that what was the final score",
        "it was three to two they scored with just seconds left",
        "that must have been so exciting to watch live",
        "it really was the whole stadium went crazy",
    ],
    [
        "what are you planning for the weekend",
        "i was thinking about going hiking if the weather is nice",
        "oh that sounds fun where would you go",
        "there's a trail about an hour north of here with amazing views",
        "i've been looking for good hiking spots can i come along",
        "of course the more the merrier we usually leave around eight",
        "perfect i'll bring some snacks and water",
        "great let's meet at the parking lot by the trailhead",
    ],
    [
        "i just got back from vacation",
        "oh nice where did you go",
        "we went to japan for two weeks it was amazing",
        "wow two weeks that's a proper trip what was your favorite part",
        "probably the food honestly everything we ate was incredible",
        "i've always wanted to try authentic ramen there",
        "the ramen was unreal we had it almost every day and never got tired of it",
        "you're making me hungry now did you visit tokyo",
        "yeah tokyo osaka kyoto and a few smaller towns",
        "sounds like the perfect itinerary i need to plan a trip",
        "you definitely should i can share our whole route with you",
        "that would be amazing thanks so much",
    ],
    [
        "have you tried that new restaurant downtown",
        "not yet but i heard good things about it",
        "the pasta there is honestly some of the best i've ever had",
        "really i'm always looking for good pasta recommendations",
        "you should go on a weeknight though weekends are packed",
        "good tip i'll try to go this thursday",
    ],
]


def build_incremental_inputs(conversations: list[list[str]], *, chain: bool = False) -> list[str]:
    """Build incremental dialog strings simulating ASR streaming.

    Words arrive one by one within each utterance, utterances accumulate
    with <ts> separators.

    Args:
        chain: If True, dialog accumulates across all conversations
            (exercises eviction). If False, resets between conversations.
    """
    inputs: list[str] = []
    dialog_parts: list[str] = []

    for conv in conversations:
        if not chain:
            dialog_parts = []
        for utterance in conv:
            words = utterance.split()
            for w_idx in range(1, len(words) + 1):
                partial = " ".join(words[:w_idx])
                if dialog_parts:
                    full = " <ts> ".join(dialog_parts) + " <ts> " + partial
                else:
                    full = partial
                inputs.append(full)
            dialog_parts.append(utterance)

    return inputs


# =====================================================================
# Memory helper
# =====================================================================


def get_rss_mb() -> float:
    """Current process RSS in MB."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


# =====================================================================
# VAP variant factories
# =====================================================================

VAP_VARIANTS = ["maai-pytorch", "maai-compile", "maai-full-onnx"]


def create_vap_variant(
    variant: str,
    *,
    ort_threads: int = 1,
    pt_threads: int = 1,
    frame_rate: int = 10,
    context_len_sec: float = 5.0,
    encoder_onnx: str = "",
    transformer_onnx: str = "",
):
    """Create a VAP wrapper for the given variant."""
    import torch

    torch.set_num_threads(pt_threads)
    from voice_pipeline.tts.openai_tts import OpenAITTS

    tts_sample_rate = OpenAITTS.OUTPUT_SAMPLE_RATE

    # MaAI variants — override class vars (MaAIVAP has no per-instance init args beyond tts_sample_rate).
    MaAIVAPWrapper._FRAME_RATE = frame_rate
    MaAIVAPWrapper._CONTEXT_LEN_SEC = context_len_sec
    MaAIVAPWrapper._ORT_THREADS = ort_threads
    MaAIVAPWrapper._PT_THREADS = pt_threads
    if encoder_onnx:
        MaAIVAPWrapper.ENCODER_ONNX_PATH = encoder_onnx

    if variant == "maai-pytorch":
        MaAIVAPWrapper._USE_ONNX_TRANSFORMER = False
        MaAIVAPWrapper._USE_TORCH_COMPILE = False
    elif variant == "maai-compile":
        MaAIVAPWrapper._USE_ONNX_TRANSFORMER = False
        MaAIVAPWrapper._USE_TORCH_COMPILE = True
    elif variant == "maai-full-onnx":
        MaAIVAPWrapper._USE_ONNX_TRANSFORMER = True
        if transformer_onnx:
            MaAIVAPWrapper.TRANSFORMER_ONNX_PATH = transformer_onnx
    else:
        raise ValueError(f"Unknown VAP variant: {variant}")

    return MaAIVAPWrapper(tts_sample_rate)


# =====================================================================
# TurnGPT variant factories
# =====================================================================

TURNGPT_VARIANTS = ["turngpt-pytorch", "turngpt-onnx-fp32", "turngpt-onnx-int8"]


_TURNGPT_ONNX_PATHS = {
    "turngpt-onnx-fp32": "models/turngpt/turngpt_v2_kvcache.onnx",
    "turngpt-onnx-int8": "models/turngpt/turngpt_v2_kvcache_int8.onnx",
}


def create_turngpt_variant(
    variant: str,
    *,
    onnx_threads: int = 2,
    checkpoint_path: str = "",
    turngpt_onnx: str = "",
    tokenizer_path: str = "",
):
    """Create a TurnGPT wrapper for the given variant."""
    from voice_pipeline.turn_taking.turngpt import TurnGPTWrapper

    tok = tokenizer_path or _DEFAULT_TOKENIZER_PATH

    # scripts: variant 분기마다 class var 명시적 재할당 (이전 variant 잔존 방지)
    if variant == "turngpt-pytorch":
        path = checkpoint_path or os.environ.get("TURNGPT_CHECKPOINT_PATH", "")
        if not path:
            raise RuntimeError("PyTorch variant requires --checkpoint-path or TURNGPT_CHECKPOINT_PATH env var")
        TurnGPTWrapper._BACKEND = "pytorch"
        TurnGPTWrapper._CHECKPOINT_PATH = path
        return TurnGPTWrapper()
    if variant in _TURNGPT_ONNX_PATHS:
        TurnGPTWrapper._BACKEND = "onnx"
        TurnGPTWrapper._ONNX_MODEL_PATH = turngpt_onnx or _TURNGPT_ONNX_PATHS[variant]
        TurnGPTWrapper._TOKENIZER_PATH = tok
        TurnGPTWrapper._ONNX_THREADS = onnx_threads
        return TurnGPTWrapper()
    raise ValueError(f"Unknown TurnGPT variant: {variant}")


# =====================================================================
# Benchmark runners
# =====================================================================


def run_vap_benchmark(
    variant: str,
    ch1: np.ndarray,
    ch2: np.ndarray,
    *,
    frame_rate: int,
    duration_sec: float,
    warmup_frames: int,
    track_memory: bool,
    **factory_kwargs,
) -> BenchmarkResult:
    """Benchmark one VAP variant."""
    sample_rate = 16000
    spf = sample_rate // frame_rate  # samples per frame
    budget_ms = 1000.0 / frame_rate
    interval = 1.0 / frame_rate

    # Total audio frames available (loop if needed)
    total_audio_frames = len(ch1) // spf
    if total_audio_frames == 0:
        raise ValueError("Audio too short for frame_rate")

    n_measure_frames = int(duration_sec * frame_rate)

    # Load model (includes __init__ warmup)
    print(f"    Loading {variant}...")
    t_load = time.perf_counter()
    wrapper = create_vap_variant(variant, frame_rate=frame_rate, **factory_kwargs)
    load_time = time.perf_counter() - t_load
    print(f"    Loaded in {load_time:.1f}s")

    # feed_audio is an async command (buffers for the wrapper's own thread);
    # reach the internal _infer for synchronous single-inference measurement.
    infer = wrapper._infer

    # Warmup with real audio to fill KV cache for steady-state measurement
    print(f"    Warmup ({warmup_frames} frames)...", end="", flush=True)
    for i in range(warmup_frames):
        idx = i % total_audio_frames
        s = idx * spf
        pcm1 = numpy_to_pcm16(ch1[s : s + spf])
        pcm2 = numpy_to_pcm16(ch2[s : s + spf])
        infer(pcm1, pcm2)
    print(" done")

    wrapper.reset()
    gc.collect()

    # Measure
    mem_start = get_rss_mb() if track_memory else None

    latencies = np.zeros(n_measure_frames, dtype=np.float64)
    timestamps = np.zeros(n_measure_frames, dtype=np.float64)
    actual_count = 0

    print(f"    Measuring ({duration_sec}s, {n_measure_frames} frames)...", end="", flush=True)
    t_start = time.perf_counter()

    for i in range(n_measure_frames):
        idx = (warmup_frames + i) % total_audio_frames
        s = idx * spf
        pcm1 = numpy_to_pcm16(ch1[s : s + spf])
        pcm2 = numpy_to_pcm16(ch2[s : s + spf])

        t0 = time.perf_counter()
        infer(pcm1, pcm2)
        t1 = time.perf_counter()

        latencies[i] = (t1 - t0) * 1000.0
        timestamps[i] = t1
        actual_count += 1

        # Real-time pacing
        elapsed = t1 - t_start
        expected = (i + 1) * interval
        if expected > elapsed:
            time.sleep(expected - elapsed)

    print(" done")
    mem_end = get_rss_mb() if track_memory else None

    # Cleanup
    del wrapper
    gc.collect()

    return compute_metrics(
        variant,
        latencies[:actual_count],
        budget_ms,
        load_time,
        mem_start=mem_start,
        mem_end=mem_end,
        timestamps=timestamps[:actual_count],
    )


def run_turngpt_benchmark(
    variant: str,
    *,
    duration_sec: float,
    rate_hz: float,
    warmup_calls: int,
    track_memory: bool,
    **factory_kwargs,
) -> BenchmarkResult:
    """Benchmark one TurnGPT variant."""
    budget_ms = 1000.0 / rate_hz
    interval = 1.0 / rate_hz

    # Prepare inputs
    all_inputs = build_incremental_inputs(CONVERSATIONS, chain=True)
    n_inputs = len(all_inputs)

    n_measure_calls = int(duration_sec * rate_hz)

    # Load model
    print(f"    Loading {variant}...")
    t_load = time.perf_counter()
    wrapper = create_turngpt_variant(variant, **factory_kwargs)
    load_time = time.perf_counter() - t_load
    print(f"    Loaded in {load_time:.1f}s")

    # Warmup
    print(f"    Warmup ({warmup_calls} calls)...", end="", flush=True)
    for i in range(warmup_calls):
        wrapper.predict(all_inputs[i % n_inputs])
    print(" done")

    wrapper.reset()
    gc.collect()

    # Measure
    mem_start = get_rss_mb() if track_memory else None

    latencies = np.zeros(n_measure_calls, dtype=np.float64)
    timestamps = np.zeros(n_measure_calls, dtype=np.float64)
    actual_count = 0

    print(f"    Measuring ({duration_sec}s, {n_measure_calls} calls @ {rate_hz}Hz)...", end="", flush=True)
    t_start = time.perf_counter()
    input_idx = 0

    for i in range(n_measure_calls):
        text = all_inputs[input_idx % n_inputs]
        input_idx += 1

        # Reset between conversations (when input wraps)
        if input_idx > 0 and input_idx % n_inputs == 0:
            wrapper.reset()

        t0 = time.perf_counter()
        wrapper.predict(text)
        t1 = time.perf_counter()

        latencies[i] = (t1 - t0) * 1000.0
        timestamps[i] = t1
        actual_count += 1

        # Real-time pacing
        elapsed = t1 - t_start
        expected = (i + 1) * interval
        if expected > elapsed:
            time.sleep(expected - elapsed)

    print(" done")
    mem_end = get_rss_mb() if track_memory else None

    # Cleanup
    del wrapper
    gc.collect()

    return compute_metrics(
        variant,
        latencies[:actual_count],
        budget_ms,
        load_time,
        mem_start=mem_start,
        mem_end=mem_end,
        timestamps=timestamps[:actual_count],
    )


# =====================================================================
# Output formatting
# =====================================================================


def print_comparison(results: list[BenchmarkResult], settings: dict) -> None:
    """Print side-by-side comparison table."""
    sep = "=" * 70
    thin = "-" * 70

    print(f"\n{sep}")
    print("  Turn-Taking Model Benchmark")
    print(sep)
    print(f"  Model family  : {settings['model']}")
    print(f"  Variants      : {', '.join(r.variant for r in results)}")
    print(f"  Audio source  : {settings.get('audio_source', 'N/A')}")
    print(f"  Duration      : {settings['duration']}s")
    print(f"  Budget        : {results[0].budget_ms:.0f}ms")

    if settings["model"] == "vap":
        print(f"  Frame rate    : {settings['frame_rate']}Hz")
        print(f"  ORT threads   : {settings['ort_threads']}")
        print(f"  PT threads    : {settings['pt_threads']}")
    else:
        print(f"  Rate          : {settings['rate']}Hz")
        print(f"  ONNX threads  : {settings['onnx_threads']}")

    print(sep)

    # Column widths
    label_w = 18
    col_w = max(16, max(len(r.variant) for r in results) + 2)

    # Header
    header = f"  {'Metric':<{label_w}}"
    for r in results:
        header += f"{r.variant:>{col_w}}"
    print(f"\n{header}")
    print(f"  {thin}")

    # Rows
    rows = [
        ("Calls", lambda r: f"{r.n_calls}"),
        ("Load time", lambda r: f"{r.load_time_sec:.1f}s"),
        ("Mean", lambda r: f"{r.mean_ms:.1f}ms"),
        ("Median", lambda r: f"{r.median_ms:.1f}ms"),
        ("P5", lambda r: f"{r.p5_ms:.1f}ms"),
        ("P95", lambda r: f"{r.p95_ms:.1f}ms"),
        ("P99", lambda r: f"{r.p99_ms:.1f}ms"),
        ("Max", lambda r: f"{r.max_ms:.1f}ms"),
        ("Std", lambda r: f"{r.std_ms:.1f}ms"),
        ("RTF", lambda r: f"{r.rtf:.2f}x"),
        (">Budget", lambda r: f"{r.over_budget_count}/{r.n_calls} ({r.over_budget_pct:.1f}%)"),
    ]

    for label, fmt_fn in rows:
        line = f"  {label:<{label_w}}"
        for r in results:
            line += f"{fmt_fn(r):>{col_w}}"
        print(line)

    # Memory (if tracked)
    if any(r.mem_start_mb is not None for r in results):
        print(f"\n  {'Memory (MB)':<{label_w}}")
        print(f"  {thin}")
        for label, attr in [("RSS start", "mem_start_mb"), ("RSS end", "mem_end_mb"), ("RSS delta", "mem_delta_mb")]:
            line = f"  {label:<{label_w}}"
            for r in results:
                val = getattr(r, attr)
                if val is not None:
                    line += f"{val:>{col_w}.1f}"
                else:
                    line += f"{'N/A':>{col_w}}"
            print(line)

    # Drift (if duration >= 60s)
    if any(len(r.drift_buckets) > 1 for r in results):
        print("\n  Drift Analysis (mean / P95 per 30s window):")
        print(f"  {thin}")
        # Find the variant with most buckets for labels
        max_buckets = max(len(r.drift_buckets) for r in results)
        for b_idx in range(max_buckets):
            label = ""
            line_parts = []
            for r in results:
                if b_idx < len(r.drift_buckets):
                    lbl, mean, p95 = r.drift_buckets[b_idx]
                    if not label:
                        label = lbl
                    line_parts.append(f"{mean:.1f} / {p95:.1f}")
                else:
                    line_parts.append("—")
            line = f"  {label:<{label_w}}"
            for part in line_parts:
                line += f"{part:>{col_w}}"
            print(line)

    print(f"\n{sep}\n")


def write_json(results: list[BenchmarkResult], settings: dict, path: str) -> None:
    """Write results as JSON."""
    output = {
        "timestamp": datetime.now(UTC).isoformat(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "settings": settings,
        "results": [],
    }
    for r in results:
        d = asdict(r)
        output["results"].append(d)

    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to {path}")


# =====================================================================
# CLI
# =====================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare turn-taking model variants (VAP / TurnGPT)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model",
        required=True,
        choices=["vap", "turngpt"],
        help="Model family to benchmark",
    )
    p.add_argument(
        "--variants",
        nargs="+",
        default=["all"],
        help="Variants to compare (or 'all')",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Measurement duration in seconds (default: 60)",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Warmup frames/calls (default: 50 for VAP, 10 for TurnGPT)",
    )
    p.add_argument(
        "--memory",
        action="store_true",
        help="Track RSS memory usage",
    )
    p.add_argument(
        "--json",
        type=str,
        default=None,
        help="Save results to JSON file",
    )

    # VAP options
    vap_group = p.add_argument_group("VAP options")
    vap_group.add_argument(
        "--audio",
        type=str,
        default="synthetic",
        help="Audio source: file path or 'synthetic' (default: synthetic)",
    )
    vap_group.add_argument(
        "--frame-rate",
        type=int,
        default=10,
        help="VAP frame rate in Hz (default: 10)",
    )
    vap_group.add_argument(
        "--ort-threads",
        type=int,
        default=1,
        help="ONNX Runtime intra-op threads (default: 1)",
    )
    vap_group.add_argument(
        "--pt-threads",
        type=int,
        default=1,
        help="PyTorch threads (default: 1)",
    )
    vap_group.add_argument(
        "--context-len",
        type=float,
        default=5.0,
        help="VAP context length in seconds (default: 5.0)",
    )
    vap_group.add_argument(
        "--encoder-onnx",
        type=str,
        default=MaAIVAPWrapper.ENCODER_ONNX_PATH,
        help=f"MaAI encoder ONNX path (default: {MaAIVAPWrapper.ENCODER_ONNX_PATH})",
    )
    vap_group.add_argument(
        "--transformer-onnx",
        type=str,
        default=MaAIVAPWrapper.TRANSFORMER_ONNX_PATH,
        help=f"MaAI transformer ONNX path (default: {MaAIVAPWrapper.TRANSFORMER_ONNX_PATH})",
    )

    # TurnGPT options
    tgpt_group = p.add_argument_group("TurnGPT options")
    tgpt_group.add_argument(
        "--rate",
        type=float,
        default=3.0,
        help="TurnGPT prediction rate in Hz (default: 3.0)",
    )
    tgpt_group.add_argument(
        "--onnx-threads",
        type=int,
        default=2,
        help="ONNX Runtime threads for TurnGPT (default: 2)",
    )
    tgpt_group.add_argument(
        "--checkpoint-path",
        type=str,
        default="",
        help="Path to TurnGPT PyTorch checkpoint (for turngpt-pytorch variant)",
    )
    tgpt_group.add_argument(
        "--turngpt-onnx",
        type=str,
        default="",
        help="TurnGPT ONNX model path (overrides variant default)",
    )
    tgpt_group.add_argument(
        "--tokenizer-path",
        type=str,
        default=_DEFAULT_TOKENIZER_PATH,
        help=f"TurnGPT tokenizer path (default: {_DEFAULT_TOKENIZER_PATH})",
    )

    return p


def main() -> None:
    args = build_parser().parse_args()

    # Resolve variants
    if args.model == "vap":
        all_variants = VAP_VARIANTS
        default_warmup = 50
    else:
        all_variants = TURNGPT_VARIANTS
        default_warmup = 10

    if "all" in args.variants:
        variants = all_variants
    else:
        for v in args.variants:
            if v not in all_variants:
                print(f"Error: unknown variant '{v}'. Available: {all_variants}")
                sys.exit(1)
        variants = args.variants

    warmup = args.warmup if args.warmup is not None else default_warmup

    # Settings dict for output
    settings: dict = {
        "model": args.model,
        "duration": args.duration,
        "warmup": warmup,
    }

    results: list[BenchmarkResult] = []

    if args.model == "vap":
        settings["frame_rate"] = args.frame_rate
        settings["ort_threads"] = args.ort_threads
        settings["pt_threads"] = args.pt_threads
        settings["context_len_sec"] = args.context_len
        settings["audio_source"] = args.audio

        # Load audio once
        print(f"\n  Audio: {args.audio}")
        if args.audio == "synthetic":
            ch1, ch2 = generate_synthetic_stereo(args.duration + warmup / args.frame_rate + 10)
        else:
            ch1, ch2 = load_stereo_audio(args.audio)
            dur = len(ch1) / 16000
            print(f"  Duration: {dur:.1f}s, will loop if needed")

        for variant in variants:
            print(f"\n  --- {variant} ---")
            try:
                result = run_vap_benchmark(
                    variant,
                    ch1,
                    ch2,
                    frame_rate=args.frame_rate,
                    duration_sec=args.duration,
                    warmup_frames=warmup,
                    track_memory=args.memory,
                    ort_threads=args.ort_threads,
                    pt_threads=args.pt_threads,
                    context_len_sec=args.context_len,
                    encoder_onnx=args.encoder_onnx,
                    transformer_onnx=args.transformer_onnx,
                )
                results.append(result)
            except Exception as e:
                print(f"    SKIPPED: {e}")

    else:  # turngpt
        settings["rate"] = args.rate
        settings["onnx_threads"] = args.onnx_threads

        for variant in variants:
            print(f"\n  --- {variant} ---")
            try:
                result = run_turngpt_benchmark(
                    variant,
                    duration_sec=args.duration,
                    rate_hz=args.rate,
                    warmup_calls=warmup,
                    track_memory=args.memory,
                    onnx_threads=args.onnx_threads,
                    checkpoint_path=args.checkpoint_path,
                    turngpt_onnx=args.turngpt_onnx,
                    tokenizer_path=args.tokenizer_path,
                )
                results.append(result)
            except Exception as e:
                print(f"    SKIPPED: {e}")

    if not results:
        print("\n  No variants completed successfully.")
        sys.exit(1)

    print_comparison(results, settings)

    if args.json:
        write_json(results, settings, args.json)


if __name__ == "__main__":
    main()
