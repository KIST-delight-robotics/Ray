"""Convert MaAI CPC Encoder to ONNX with explicit LSTM hidden state.

Loads the encoder through MaAI so that trained downsample weights from
the VAP state dict are properly applied (not random init).

Usage:
    uv run python scripts/tools/convert_maai_encoder_onnx.py
    uv run python scripts/tools/convert_maai_encoder_onnx.py --lang jp --context 20

Outputs:
    models/maai_encoder_5hz.onnx
    models/maai_encoder_10hz.onnx
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from voice_pipeline.turn_taking.onnx_export import EncoderONNXWrapper


def load_encoder_from_maai(frame_rate: int, context_len_sec: float, lang: str):
    """Load encoder through MaAI to get properly trained downsample weights."""
    from maai import Maai, MaaiInput

    ch1 = MaaiInput.Chunk()
    ch2 = MaaiInput.Chunk()
    maai = Maai(
        mode="vap",
        lang=lang,
        frame_rate=frame_rate,
        context_len_sec=context_len_sec,
        audio_ch1=ch1,
        audio_ch2=ch2,
        device="cpu",
        use_kv_cache=True,
    )
    # encoder1 and encoder2 have identical weights
    return maai.vap.encoder1


def convert(
    frame_rate: int,
    context_len_sec: float,
    lang: str,
    output_path: str,
) -> None:
    samples_per_frame = 16000 // frame_rate
    input_size = 320 + samples_per_frame

    print(f"\n{'=' * 50}")
    print(f"Converting encoder for {frame_rate}Hz (input={input_size} samples)")
    print(f"  lang={lang}, context={context_len_sec}s")
    print(f"{'=' * 50}")

    # Load encoder with trained weights via MaAI
    encoder = load_encoder_from_maai(frame_rate, context_len_sec, lang)
    encoder.eval()

    # Log downsample weight stats to confirm trained (not random)
    ds_w = encoder.downsample[1].weight
    print(f"Downsample conv weight: mean={ds_w.mean():.6f}, std={ds_w.std():.6f}")

    wrapper = EncoderONNXWrapper(encoder)
    wrapper.eval()

    # Dummy inputs
    dummy_wav = torch.randn(1, 1, input_size)
    dummy_h = torch.zeros(1, 1, 256)
    dummy_c = torch.zeros(1, 1, 256)

    # Verify wrapper matches original encoder
    encoder.encoder.gAR.hidden = None
    with torch.no_grad():
        pt_out, pt_h, pt_c = wrapper(dummy_wav, dummy_h, dummy_c)
        orig_out = encoder(dummy_wav)

    orig_diff = (pt_out - orig_out).abs().max().item()
    print(f"PyTorch output: {list(pt_out.shape)}, h: {list(pt_h.shape)}")
    print(f"Wrapper vs original encoder max diff: {orig_diff:.6f}")
    assert orig_diff < 1e-5, f"Wrapper output diverges from original: {orig_diff}"

    # Export
    torch.onnx.export(
        wrapper,
        (dummy_wav, dummy_h, dummy_c),
        output_path,
        input_names=["waveform", "h_in", "c_in"],
        output_names=["embedding", "h_out", "c_out"],
        dynamic_axes={
            "waveform": {2: "n_samples"},
            "embedding": {1: "n_frames"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported to {output_path}")

    # Validate ONNX model
    model = onnx.load(output_path)
    onnx.checker.check_model(model)

    # Count ops
    op_counts: dict[str, int] = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    transpose_count = op_counts.get("Transpose", 0)
    print(f"ONNX validated OK — Transpose ops: {transpose_count}")

    # Verify ONNX output with optimized session
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = 4

    sess = ort.InferenceSession(output_path, sess_opts)

    # Reset LSTM state for fair comparison
    encoder.encoder.gAR.hidden = None
    with torch.no_grad():
        pt_out2, pt_h2, pt_c2 = wrapper(dummy_wav, dummy_h, dummy_c)

    onnx_out, onnx_h, onnx_c = sess.run(
        None,
        {
            "waveform": dummy_wav.numpy(),
            "h_in": dummy_h.numpy(),
            "c_in": dummy_c.numpy(),
        },
    )

    diff_out = np.abs(pt_out2.numpy() - onnx_out).max()
    diff_h = np.abs(pt_h2.numpy() - onnx_h).max()
    diff_c = np.abs(pt_c2.numpy() - onnx_c).max()
    print(f"Max diff — embedding: {diff_out:.6f}, h: {diff_h:.6f}, c: {diff_c:.6f}")

    if diff_out < 1e-4 and diff_h < 1e-4 and diff_c < 1e-4:
        print("Numerical verification PASSED")
    else:
        print("WARNING: Numerical difference detected!")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Convert MaAI encoder to ONNX")
    parser.add_argument("--lang", default="en", help="Language (default: en)")
    parser.add_argument(
        "--context",
        type=float,
        default=20,
        help="Context length in seconds (default: 20)",
    )
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)

    for fr in [5, 10]:
        convert(fr, args.context, args.lang, f"models/maai_encoder_{fr}hz.onnx")

    print("\nDone.")


if __name__ == "__main__":
    main()
