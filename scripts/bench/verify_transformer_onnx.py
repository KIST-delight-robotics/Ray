"""Verify VAP transformer ONNX export numerical equivalence.

Compares original PyTorch VapGPT.forward() against TransformerONNXWrapper
(run via ORT) frame-by-frame using synthetic audio or real CANDOR audio.

Usage:
    # Synthetic (quick sanity check)
    uv run python scripts/bench/verify_transformer_onnx.py

    # Real audio
    uv run python scripts/bench/verify_transformer_onnx.py \
        --audio CANDOR/raw_media_part_001/.../processed/....mp3 \
        --max-seconds 60
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from voice_pipeline.turn_taking.onnx_export import export_transformer_onnx


def load_maai(frame_rate: int = 10, context_len_sec: float = 5.0):
    """Create a MaAI instance and return (maai, vap)."""
    from maai import Maai, MaaiInput

    ch1 = MaaiInput.Chunk()
    ch2 = MaaiInput.Chunk()
    maai = Maai(
        mode="vap",
        lang="en",
        frame_rate=frame_rate,
        context_len_sec=context_len_sec,
        audio_ch1=ch1,
        audio_ch2=ch2,
        device="cpu",
        use_kv_cache=True,
    )
    return maai


def create_ort_session(onnx_path: str, threads: int = 1) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = threads
    return ort.InferenceSession(onnx_path, opts)


def run_pytorch_frame(
    vap, x1: torch.Tensor, x2: torch.Tensor, cache: dict | None
) -> tuple[dict, dict]:
    """Run one frame through original PyTorch VapGPT."""
    with torch.inference_mode():
        out, new_cache = vap.forward(x1, x2, cache=cache)
    return out, new_cache


def pytorch_cache_to_numpy(
    cache: dict | None, n_ch_layers: int, n_cross_layers: int, num_heads: int, head_dim: int
) -> dict[str, np.ndarray]:
    """Convert PyTorch dict cache → flat numpy arrays for ORT."""
    def _stack_or_empty(cache_entry, n_layers):
        if cache_entry is None:
            return (
                np.zeros((n_layers, 1, num_heads, 0, head_dim), dtype=np.float32),
                np.zeros((n_layers, 1, num_heads, 0, head_dim), dtype=np.float32),
            )
        k_list, v_list = cache_entry
        return (
            np.stack([k.numpy() for k in k_list]),
            np.stack([v.numpy() for v in v_list]),
        )

    if cache is None:
        empty_ch = np.zeros((n_ch_layers, 1, num_heads, 0, head_dim), dtype=np.float32)
        empty_cr = np.zeros((n_cross_layers, 1, num_heads, 0, head_dim), dtype=np.float32)
        return {
            "ar1_k": empty_ch, "ar1_v": empty_ch.copy(),
            "ar2_k": empty_ch.copy(), "ar2_v": empty_ch.copy(),
            "cross1_k": empty_cr, "cross1_v": empty_cr.copy(),
            "cross2_k": empty_cr.copy(), "cross2_v": empty_cr.copy(),
            "cross1_c_k": empty_cr.copy(), "cross1_c_v": empty_cr.copy(),
            "cross2_c_k": empty_cr.copy(), "cross2_c_v": empty_cr.copy(),
        }

    ar1_k, ar1_v = _stack_or_empty(cache.get("ar1"), n_ch_layers)
    ar2_k, ar2_v = _stack_or_empty(cache.get("ar2"), n_ch_layers)
    c1_k, c1_v = _stack_or_empty(cache.get("cross1"), n_cross_layers)
    c2_k, c2_v = _stack_or_empty(cache.get("cross2"), n_cross_layers)
    c1c_k, c1c_v = _stack_or_empty(cache.get("cross1_c"), n_cross_layers)
    c2c_k, c2c_v = _stack_or_empty(cache.get("cross2_c"), n_cross_layers)

    return {
        "ar1_k": ar1_k, "ar1_v": ar1_v,
        "ar2_k": ar2_k, "ar2_v": ar2_v,
        "cross1_k": c1_k, "cross1_v": c1_v,
        "cross2_k": c2_k, "cross2_v": c2_v,
        "cross1_c_k": c1c_k, "cross1_c_v": c1c_v,
        "cross2_c_k": c2c_k, "cross2_c_v": c2c_v,
    }


def run_ort_frame(
    sess: ort.InferenceSession,
    x1: np.ndarray,
    x2: np.ndarray,
    cache: dict[str, np.ndarray],
) -> tuple[dict, dict[str, np.ndarray]]:
    """Run one frame through ONNX transformer."""
    inputs = {"x1": x1, "x2": x2, **cache}
    outputs = sess.run(None, inputs)

    out_names = [o.name for o in sess.get_outputs()]
    result = dict(zip(out_names, outputs))

    scalar_out = {
        "p_now": result["p_now"].tolist(),      # list of 2 floats
        "p_future": result["p_future"].tolist(), # list of 2 floats
        "vad": [float(result["vad1"]), float(result["vad2"])],
    }

    new_cache = {
        "ar1_k": result["out_ar1_k"], "ar1_v": result["out_ar1_v"],
        "ar2_k": result["out_ar2_k"], "ar2_v": result["out_ar2_v"],
        "cross1_k": result["out_cross1_k"], "cross1_v": result["out_cross1_v"],
        "cross2_k": result["out_cross2_k"], "cross2_v": result["out_cross2_v"],
        "cross1_c_k": result["out_cross1_c_k"], "cross1_c_v": result["out_cross1_c_v"],
        "cross2_c_k": result["out_cross2_c_k"], "cross2_c_v": result["out_cross2_c_v"],
    }

    return scalar_out, new_cache


def get_encoder_embeddings(maai, x1_audio: np.ndarray, x2_audio: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Get encoder embeddings from audio using MaAI's encoder."""
    vap = maai.vap
    with torch.inference_mode():
        wav1 = torch.from_numpy(x1_audio.reshape(1, 1, -1))
        wav2 = torch.from_numpy(x2_audio.reshape(1, 1, -1))
        e1 = vap.encoder1(wav1)
        e2 = vap.encoder2(wav2)
    return e1, e2


def run_verification(
    n_frames: int = 100,
    audio_path: str | None = None,
    frame_rate: int = 10,
    context_len_sec: float = 5.0,
    max_seconds: float | None = None,
):
    print(f"\n{'=' * 70}")
    print(f"  VAP Transformer ONNX Equivalence Verification")
    print(f"{'=' * 70}")
    print(f"  Mode        : {'Real audio' if audio_path else 'Synthetic'}")
    print(f"  Frame rate  : {frame_rate}Hz")
    print(f"  Context     : {context_len_sec}s")

    torch.set_num_threads(1)

    # Load MaAI
    print("\n  Loading MaAI...")
    maai = load_maai(frame_rate, context_len_sec)
    vap = maai.vap
    vap.eval()

    conf = vap.conf
    num_heads = conf.num_heads
    head_dim = conf.dim // num_heads
    n_ch_layers = len(list(vap.ar_channel.layers))
    n_cross_layers = len(list(vap.ar.layers))
    print(f"  Model       : dim={conf.dim}, heads={num_heads}, "
          f"ch_layers={n_ch_layers}, cross_layers={n_cross_layers}")

    # Export ONNX transformer
    print("  Exporting transformer ONNX...")
    t0 = time.perf_counter()
    onnx_path = export_transformer_onnx(vap)
    export_time = time.perf_counter() - t0
    onnx_size = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"  Export time : {export_time:.1f}s")
    print(f"  ONNX size   : {onnx_size:.1f}MB")

    # Load ORT session
    print("  Loading ORT session...")
    sess = create_ort_session(onnx_path, threads=1)
    os.unlink(onnx_path)

    # Prepare audio frames
    if audio_path:
        import soundfile as sf
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim == 1:
            raise ValueError(f"Expected stereo: {audio_path}")
        ch1_full, ch2_full = data[:, 0], data[:, 1]
        if sr != 16000:
            ratio = 16000 / sr
            n_out = int(len(ch1_full) * ratio)
            idx = np.linspace(0, len(ch1_full) - 1, n_out).astype(np.float64)
            lo = idx.astype(np.int64)
            hi = np.minimum(lo + 1, len(ch1_full) - 1)
            frac = (idx - lo).astype(np.float32)
            ch1_full = ch1_full[lo] * (1 - frac) + ch1_full[hi] * frac
            ch2_full = ch2_full[lo] * (1 - frac) + ch2_full[hi] * frac

        total_sec = len(ch1_full) / 16000
        if max_seconds and max_seconds < total_sec:
            n_samples = int(max_seconds * 16000)
            ch1_full = ch1_full[:n_samples]
            ch2_full = ch2_full[:n_samples]
            total_sec = max_seconds

        # We need encoder embeddings, not raw audio.
        # Feed audio through encoder in chunks matching frame size.
        padding = 320
        samples_per_frame = 16000 // frame_rate
        audio_frame_size = samples_per_frame + padding
        n_frames = len(ch1_full) // samples_per_frame
        print(f"  Audio       : {total_sec:.1f}s, {n_frames} frames")
    else:
        n_frames = n_frames
        print(f"  Frames      : {n_frames} (synthetic embeddings)")

    # Run comparison
    print(f"\n  Running {n_frames} frames...")

    pt_cache: dict | None = None
    ort_cache = pytorch_cache_to_numpy(None, n_ch_layers, n_cross_layers, num_heads, head_dim)

    diffs_pnow = []
    diffs_pfut = []
    diffs_vad = []
    lats_pt = []
    lats_ort = []

    # For real audio, track encoder state
    if audio_path:
        buf1 = np.zeros(padding, dtype=np.float32)
        buf2 = np.zeros(padding, dtype=np.float32)

    t_start = time.perf_counter()

    for i in range(n_frames):
        # Get embeddings
        if audio_path:
            start = i * samples_per_frame
            end = start + samples_per_frame
            x1_chunk = ch1_full[start:end]
            x2_chunk = ch2_full[start:end]
            buf1_frame = np.concatenate([buf1, x1_chunk])
            buf2_frame = np.concatenate([buf2, x2_chunk])
            e1, e2 = get_encoder_embeddings(maai, buf1_frame, buf2_frame)
            buf1 = buf1_frame[-padding:]
            buf2 = buf2_frame[-padding:]
        else:
            e1 = torch.randn(1, 1, conf.dim)
            e2 = torch.randn(1, 1, conf.dim)

        # PyTorch forward
        t0 = time.perf_counter()
        pt_out, pt_cache = run_pytorch_frame(vap, e1, e2, pt_cache)
        lats_pt.append(time.perf_counter() - t0)

        # ORT forward (same embeddings)
        e1_np = e1.numpy()
        e2_np = e2.numpy()

        t0 = time.perf_counter()
        ort_out, ort_cache = run_ort_frame(sess, e1_np, e2_np, ort_cache)
        lats_ort.append(time.perf_counter() - t0)

        # Compare (p_now/p_future are lists of 2 floats)
        def _max_diff(a, b):
            if isinstance(a, list) and isinstance(b, list):
                return max(abs(ai - bi) for ai, bi in zip(a, b))
            return abs(a - b)

        d_pnow = _max_diff(pt_out["p_now"], ort_out["p_now"])
        d_pfut = _max_diff(pt_out["p_future"], ort_out["p_future"])
        d_vad = _max_diff(pt_out["vad"], ort_out["vad"])

        diffs_pnow.append(d_pnow)
        diffs_pfut.append(d_pfut)
        diffs_vad.append(d_vad)

        # Cache trimming for PyTorch (match production behavior)
        context_limit = int(context_len_sec * frame_rate) - 1
        if pt_cache is not None:
            new_pt_cache: dict = {}
            for key, (k_list, v_list) in pt_cache.items():
                new_pt_cache[key] = (
                    [t[..., -context_limit:, :] if t.dim() >= 3 else t for t in k_list],
                    [t[..., -context_limit:, :] if t.dim() >= 3 else t for t in v_list],
                )
            pt_cache = new_pt_cache

        # Cache trimming for ORT
        for name, arr in ort_cache.items():
            if arr.ndim >= 4 and arr.shape[3] > context_limit:
                ort_cache[name] = arr[:, :, :, -context_limit:, :]

        # Progress
        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t_start
            print(
                f"    frame {i+1:5d} | "
                f"pnow={max(diffs_pnow):.7f} pfut={max(diffs_pfut):.7f} "
                f"vad={max(diffs_vad):.7f} | {elapsed:.1f}s"
            )

    elapsed = time.perf_counter() - t_start

    # Results
    print(f"\n{'=' * 70}")
    print(f"  Results ({len(diffs_pnow)} frames, {elapsed:.1f}s)")
    print(f"{'=' * 70}")

    def stats(name: str, vals: list[float]):
        arr = np.array(vals)
        print(
            f"  {name:<12}: "
            f"max={arr.max():.7f}  "
            f"mean={arr.mean():.7f}  "
            f"p99={np.percentile(arr, 99):.7f}"
        )

    print(f"\n  --- Numerical Equivalence ---")
    stats("p_now", diffs_pnow)
    stats("p_future", diffs_pfut)
    stats("vad", diffs_vad)

    # Latency
    budget_ms = 1000.0 / frame_rate
    pt_arr = np.array(lats_pt) * 1000
    ort_arr = np.array(lats_ort) * 1000
    print(f"\n  --- Latency (transformer only, budget={budget_ms:.0f}ms) ---")
    print(f"  {'PyTorch':<12}: mean={pt_arr.mean():.1f}ms  "
          f"median={np.median(pt_arr):.1f}ms  p95={np.percentile(pt_arr, 95):.1f}ms")
    print(f"  {'ONNX ORT':<12}: mean={ort_arr.mean():.1f}ms  "
          f"median={np.median(ort_arr):.1f}ms  p95={np.percentile(ort_arr, 95):.1f}ms")

    if ort_arr.mean() > 0:
        speedup = pt_arr.mean() / ort_arr.mean()
        print(f"\n  Speedup     : {speedup:.2f}x")

    # Drift check
    if len(diffs_pnow) >= 20:
        mid = len(diffs_pnow) // 2
        first_max = max(max(diffs_pnow[:mid]), max(diffs_pfut[:mid]), max(diffs_vad[:mid]))
        second_max = max(max(diffs_pnow[mid:]), max(diffs_pfut[mid:]), max(diffs_vad[mid:]))
        drift = second_max / first_max if first_max > 0 else 1.0
        print(f"\n  Drift check:")
        print(f"    1st half max: {first_max:.7f}")
        print(f"    2nd half max: {second_max:.7f}")
        print(f"    Ratio       : {drift:.2f}x {'WARNING' if drift > 10 else 'OK'}")

    # Verdict
    threshold = 0.001
    overall_max = max(max(diffs_pnow), max(diffs_pfut), max(diffs_vad))
    print(f"\n  Overall max diff: {overall_max:.7f}")
    if overall_max < threshold:
        print(f"  PASSED (all diffs < {threshold})")
    else:
        print(f"  FAILED (max diff {overall_max:.7f} >= {threshold})")

    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Verify VAP transformer ONNX equivalence")
    parser.add_argument("--audio", default=None, help="Path to stereo audio file")
    parser.add_argument("--frames", type=int, default=100, help="Frames for synthetic mode")
    parser.add_argument("--frame-rate", type=int, default=10)
    parser.add_argument("--context", type=float, default=5.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args()

    run_verification(
        n_frames=args.frames,
        audio_path=args.audio,
        frame_rate=args.frame_rate,
        context_len_sec=args.context,
        max_seconds=args.max_seconds,
    )


if __name__ == "__main__":
    main()
