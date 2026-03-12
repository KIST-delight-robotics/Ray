"""View VAP predictions on CANDOR audio using our ONNX pipeline.

Shows p_now (turn-shift probability) and VAD as console bars.

Usage:
    uv run python scripts/vap_console_view.py \
        --audio CANDOR/raw_media_part_001/a29635a0-.../processed/a29635a0-...mp3 \
        --duration 60
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from benchmark_maai_custom_pipeline import VapOnnxPipeline


def load_stereo_16k(path: str) -> tuple[np.ndarray, np.ndarray]:
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


def bar(value: float, width: int = 20) -> str:
    n = int(value * width)
    return "\u2588" * n + "\u2591" * (width - n)


def main():
    parser = argparse.ArgumentParser(description="VAP console viewer (ONNX pipeline)")
    parser.add_argument("--audio", required=True, help="Stereo audio file")
    parser.add_argument("--start", type=float, default=0, help="Start offset (seconds)")
    parser.add_argument("--duration", type=float, default=60, help="Duration (seconds)")
    parser.add_argument("--frame-rate", type=int, default=10, help="Frame rate (Hz)")
    parser.add_argument("--realtime", action="store_true", help="Pace to real-time")
    args = parser.parse_args()

    print("Loading audio...")
    ch1, ch2 = load_stereo_16k(os.path.abspath(args.audio))

    print("Loading VAP pipeline (ONNX)...")
    torch.set_num_threads(1)
    pipeline = VapOnnxPipeline(
        frame_rate=args.frame_rate, context_len_sec=5.0, ort_threads=1,
    )

    spf = 16000 // args.frame_rate
    start_frame = int(args.start * args.frame_rate)
    end_frame = min(start_frame + int(args.duration * args.frame_rate), len(ch1) // spf)
    interval = 1.0 / args.frame_rate

    # Feed audio from the beginning up to start_frame so LSTM state is warm
    print(f"Warming up LSTM state ({args.start:.0f}s of audio)...")
    for i in range(start_frame):
        pipeline.process(ch1[i * spf : (i + 1) * spf], ch2[i * spf : (i + 1) * spf])

    # Pre-compute target frames
    print("Pre-computing VAP predictions...")
    results = []
    for i in range(start_frame, end_frame):
        x1 = ch1[i * spf : (i + 1) * spf]
        x2 = ch2[i * spf : (i + 1) * spf]
        out = pipeline.process(x1, x2)
        if out is not None:
            results.append((i, out))
    print(f"Done: {len(results)} frames\n")

    # Mix to mono for playback (both speakers audible on any speaker)
    mix_path = "/tmp/candor_mix.wav"
    seg_start = start_frame * spf
    seg_end = end_frame * spf
    mono = (ch1[seg_start:seg_end] + ch2[seg_start:seg_end]) * 0.5
    sf.write(mix_path, mono, 16000)

    print(f"{'Time':>6s}  {'p_shift':>7s} {'bar':<22s} {'vad1':>5s} {'vad2':>5s}  status")
    print("-" * 75)

    # Start audio playback
    audio_proc = subprocess.Popen(
        ["aplay", "-q", mix_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    t_start = time.perf_counter()

    try:
        idx = 0
        while idx < len(results) and audio_proc.poll() is None:
            frame_i, out = results[idx]
            t_sec = frame_i / args.frame_rate  # absolute time for display
            playback_sec = (frame_i - start_frame) / args.frame_rate  # relative to playback start

            # Wait until audio reaches this frame's time
            while time.perf_counter() - t_start < playback_sec:
                time.sleep(0.005)

            p_shift = out["p_now"][1]
            vad1 = out["vad"][0]
            vad2 = out["vad"][1]

            if vad1 > 0.5 and vad2 < 0.5:
                status = "SP1 speaking"
            elif vad2 > 0.5 and vad1 < 0.5:
                status = "SP2 speaking"
            elif vad1 > 0.5 and vad2 > 0.5:
                status = "OVERLAP"
            else:
                status = "silence"

            if p_shift > 0.6:
                shift_marker = " << SHIFT"
            elif p_shift > 0.4:
                shift_marker = " < maybe"
            else:
                shift_marker = ""

            print(f"{t_sec:6.1f}s  {p_shift:7.3f} {bar(p_shift)}  {vad1:.2f}  {vad2:.2f}  {status}{shift_marker}")
            idx += 1
    except KeyboardInterrupt:
        pass
    finally:
        audio_proc.terminate()
        audio_proc.wait()


if __name__ == "__main__":
    main()
