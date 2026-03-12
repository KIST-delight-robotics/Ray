"""Quantize MaAI VAP ONNX encoder to int8 and benchmark.

Applies onnxruntime dynamic quantization (weight-only int8) to the CPC
encoder ONNX model and compares:
  - File size (fp32 vs int8)
  - Numerical accuracy (max diff, MAE)
  - Encoder-only latency
  - Full pipeline latency (ONNX encoder + PyTorch transformer)

Usage:
    uv run python scripts/tools/quantize_vap_encoder.py
    uv run python scripts/tools/quantize_vap_encoder.py --frame-rates 5 10 --iterations 30
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def percentile(sorted_vals: list[float], p: float) -> float:
    idx = (len(sorted_vals) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def print_stats(label: str, latencies: list[float], budget_ms: float) -> None:
    s = sorted(latencies)
    mean = sum(s) / len(s)
    median = percentile(s, 0.5)
    p95 = percentile(s, 0.95)
    p99 = percentile(s, 0.99)
    rtf = (budget_ms / 1000) / mean if mean > 0 else float("inf")

    print(f"  {label}")
    print(f"    Mean: {mean * 1000:7.1f} ms | Median: {median * 1000:7.1f} ms | "
          f"P95: {p95 * 1000:7.1f} ms | P99: {p99 * 1000:7.1f} ms")
    print(f"    Min: {s[0] * 1000:7.1f} ms | Max: {s[-1] * 1000:7.1f} ms | "
          f"Budget: {budget_ms:.0f} ms | RTF: {rtf:.2f}x {'OK' if rtf >= 1.0 else 'SLOW'}")


def make_session(onnx_path: str, threads: int = 1) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = threads
    return ort.InferenceSession(onnx_path, opts)


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


def quantize_encoder(fp32_path: str, int8_path: str) -> None:
    """Apply dynamic int8 quantization to ONNX encoder."""
    print(f"\nQuantizing: {fp32_path}")
    print(f"  Output:   {int8_path}")

    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
    )

    # Validate
    model = onnx.load(int8_path)
    onnx.checker.check_model(model)

    op_counts: dict[str, int] = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    fp32_size = os.path.getsize(fp32_path)
    int8_size = os.path.getsize(int8_path)
    ratio = int8_size / fp32_size * 100

    print(f"  FP32 size: {fp32_size / 1024 / 1024:.1f} MB")
    print(f"  INT8 size: {int8_size / 1024 / 1024:.1f} MB ({ratio:.0f}%)")
    print(f"  Ops: {dict(sorted(op_counts.items()))}")


# ---------------------------------------------------------------------------
# Numerical accuracy
# ---------------------------------------------------------------------------


def verify_accuracy(
    fp32_path: str, int8_path: str, frame_rate: int, n_trials: int = 100
) -> None:
    """Compare fp32 vs int8 encoder outputs."""
    print(f"\n--- Numerical accuracy ({n_trials} random inputs) ---")

    sess_fp32 = make_session(fp32_path)
    sess_int8 = make_session(int8_path)

    samples_per_frame = 16000 // frame_rate
    input_size = 320 + samples_per_frame

    diffs_emb: list[float] = []
    diffs_h: list[float] = []
    diffs_c: list[float] = []

    h = np.zeros((1, 1, 256), dtype=np.float32)
    c = np.zeros((1, 1, 256), dtype=np.float32)

    for _ in range(n_trials):
        wav = np.random.randn(1, 1, input_size).astype(np.float32)
        feeds = {"waveform": wav, "h_in": h, "c_in": c}

        out_fp32 = sess_fp32.run(None, feeds)
        out_int8 = sess_int8.run(None, feeds)

        diffs_emb.append(np.abs(out_fp32[0] - out_int8[0]).max())
        diffs_h.append(np.abs(out_fp32[1] - out_int8[1]).max())
        diffs_c.append(np.abs(out_fp32[2] - out_int8[2]).max())

    print(f"  Embedding max diff: mean={np.mean(diffs_emb):.6f}, max={np.max(diffs_emb):.6f}")
    print(f"  Hidden    max diff: mean={np.mean(diffs_h):.6f}, max={np.max(diffs_h):.6f}")
    print(f"  Cell      max diff: mean={np.mean(diffs_c):.6f}, max={np.max(diffs_c):.6f}")

    # Accumulated drift test (simulate streaming)
    print(f"\n  Accumulated drift (30s streaming at {frame_rate}Hz):")
    n_frames = 30 * frame_rate

    h_fp32 = np.zeros((1, 1, 256), dtype=np.float32)
    c_fp32 = np.zeros((1, 1, 256), dtype=np.float32)
    h_int8 = np.zeros((1, 1, 256), dtype=np.float32)
    c_int8 = np.zeros((1, 1, 256), dtype=np.float32)

    emb_drifts = []
    for i in range(n_frames):
        wav = np.random.randn(1, 1, input_size).astype(np.float32) * 0.1

        e_fp32, h_fp32, c_fp32 = sess_fp32.run(
            None, {"waveform": wav, "h_in": h_fp32, "c_in": c_fp32}
        )
        e_int8, h_int8, c_int8 = sess_int8.run(
            None, {"waveform": wav, "h_in": h_int8, "c_in": c_int8}
        )
        emb_drifts.append(np.abs(e_fp32 - e_int8).max())

    print(f"    Frame   1: {emb_drifts[0]:.6f}")
    print(f"    Frame  10: {emb_drifts[min(9, len(emb_drifts) - 1)]:.6f}")
    print(f"    Frame  50: {emb_drifts[min(49, len(emb_drifts) - 1)]:.6f}")
    print(f"    Frame 150: {emb_drifts[min(149, len(emb_drifts) - 1)]:.6f}")
    print(f"    Final ({n_frames}): {emb_drifts[-1]:.6f}")
    print(f"    Max drift: {max(emb_drifts):.6f}")


# ---------------------------------------------------------------------------
# Encoder-only benchmark
# ---------------------------------------------------------------------------


def benchmark_encoder(
    fp32_path: str,
    int8_path: str,
    frame_rate: int,
    warmup: int,
    iterations: int,
    threads: int,
) -> None:
    """Benchmark encoder-only latency: fp32 vs int8."""
    print(f"\n--- Encoder-only latency ({iterations} iterations, {threads} thread(s)) ---")

    samples_per_frame = 16000 // frame_rate
    input_size = 320 + samples_per_frame
    budget_ms = 1000.0 / frame_rate

    for label, path in [("FP32", fp32_path), ("INT8", int8_path)]:
        sess = make_session(path, threads)

        wav = np.random.randn(1, 1, input_size).astype(np.float32)
        h = np.zeros((1, 1, 256), dtype=np.float32)
        c = np.zeros((1, 1, 256), dtype=np.float32)

        for _ in range(warmup):
            sess.run(None, {"waveform": wav, "h_in": h, "c_in": c})

        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            sess.run(None, {"waveform": wav, "h_in": h, "c_in": c})
            latencies.append(time.perf_counter() - t0)

        print_stats(f"Encoder {label}", latencies, budget_ms)


# ---------------------------------------------------------------------------
# Full pipeline benchmark (ONNX encoder + PyTorch transformer)
# ---------------------------------------------------------------------------


def benchmark_pipeline(
    fp32_path: str,
    int8_path: str,
    frame_rate: int,
    context_len_sec: float,
    warmup: int,
    iterations: int,
    threads: int,
) -> None:
    """Benchmark full pipeline: fp32 vs int8 encoder + shared PyTorch transformer."""
    import torch

    torch.set_num_threads(threads)

    print(f"\n--- Full pipeline ({frame_rate}Hz, context={context_len_sec}s, "
          f"{iterations} iters, pt_threads={threads}) ---")

    from scripts.bench.vap_onnx_pipeline import VapOnnxPipeline

    # Create pipeline (uses fp32 encoder by default)
    pipe = VapOnnxPipeline(
        frame_rate=frame_rate,
        context_len_sec=context_len_sec,
        lang="en",
        device="cpu",
        ort_threads=threads,
    )

    samples_per_frame = 16000 // frame_rate
    budget_ms = 1000.0 / frame_rate

    for label, onnx_path in [("FP32", fp32_path), ("INT8", int8_path)]:
        # Replace encoder sessions
        pipe.sess1 = make_session(onnx_path, threads)
        pipe.sess2 = make_session(onnx_path, threads)

        # Simulate streaming: feed frames and measure
        dummy_frame = np.random.randn(samples_per_frame).astype(np.float32) * 0.1

        # Warmup
        for _ in range(warmup):
            pipe.reset()
            pipe.process(dummy_frame, np.zeros_like(dummy_frame))

        # Timed runs
        latencies: list[float] = []
        pipe.reset()
        for i in range(warmup + iterations):
            t0 = time.perf_counter()
            out = pipe.process(dummy_frame, np.zeros_like(dummy_frame))
            elapsed = time.perf_counter() - t0

            if out is not None and i >= warmup:
                latencies.append(elapsed)

        if latencies:
            print_stats(f"Pipeline {label}", latencies, budget_ms)
        else:
            print(f"  Pipeline {label}: no inferences triggered")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize and benchmark VAP ONNX encoder")
    parser.add_argument("--frame-rates", nargs="+", type=int, default=[5, 10],
                        help="Frame rates to test (default: 5 10)")
    parser.add_argument("--context", type=float, default=5.0,
                        help="Context length in seconds (default: 5.0)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations (default: 10)")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Timed iterations (default: 50)")
    parser.add_argument("--threads", type=int, default=1,
                        help="Threads for ORT and PyTorch (default: 1)")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Skip full pipeline benchmark (requires MaAI)")
    args = parser.parse_args()

    models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("  VAP ONNX Encoder Quantization & Benchmark")
    print("=" * 60)
    print(f"  Frame rates : {args.frame_rates}")
    print(f"  Context     : {args.context}s")
    print(f"  Warmup      : {args.warmup}")
    print(f"  Iterations  : {args.iterations}")
    print(f"  Threads     : {args.threads}")
    print(f"  ORT version : {ort.__version__}")

    for fr in args.frame_rates:
        fp32_path = os.path.join(models_dir, f"maai_encoder_{fr}hz.onnx")
        int8_path = os.path.join(models_dir, f"maai_encoder_{fr}hz_int8.onnx")

        if not os.path.isfile(fp32_path):
            print(f"\nWARNING: {fp32_path} not found, skipping {fr}Hz")
            print("  Run: uv run python scripts/tools/convert_maai_encoder_onnx.py")
            continue

        print(f"\n{'#' * 60}")
        print(f"  {fr}Hz Encoder")
        print(f"{'#' * 60}")

        # 1. Quantize
        quantize_encoder(fp32_path, int8_path)

        # 2. Accuracy
        verify_accuracy(fp32_path, int8_path, fr)

        # 3. Encoder benchmark
        benchmark_encoder(fp32_path, int8_path, fr, args.warmup, args.iterations, args.threads)

        # 4. Full pipeline benchmark
        if not args.skip_pipeline:
            try:
                benchmark_pipeline(
                    fp32_path, int8_path, fr, args.context,
                    args.warmup, args.iterations, args.threads,
                )
            except Exception as e:
                print(f"\n  Pipeline benchmark failed: {e}")
                print("  Use --skip-pipeline to skip, or install MaAI")

    print(f"\n{'=' * 60}")
    print("  Done.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
