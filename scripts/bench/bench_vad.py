"""Benchmark VAD accuracy: MaAI VAP vad vs Silero VAD on AVA-Speech.

Compares raw continuous scores (threshold-independent) using AUC-ROC
and precision-recall curves. Also reports optimal thresholds.

Ground truth: AVA-Speech labels (segment-level → frame-level).
  - SPEECH = CLEAN_SPEECH | SPEECH_WITH_MUSIC | SPEECH_WITH_NOISE
  - SILENCE = NO_SPEECH

Robot channel is fed silence (matches turn-shift context in production).

Setup:
    1. Download AVA-Speech labels + audio:
         curl -o data/ava_speech/ava_speech_labels_v1.csv \\
             https://research.google.com/ava/download/ava_speech_labels_v1.csv
         # Then download WAVs (16kHz mono, 15:00–30:00) via yt-dlp + ffmpeg

    2. Run:
         uv run python scripts/bench/bench_vad.py --data-dir data/ava_speech

    3. Optional flags:
         --max-files 3          Process only N files (quick test)
         --frame-dur-ms 30      Pipeline frame duration (default 30ms)
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

SPEECH_LABELS = {"CLEAN_SPEECH", "SPEECH_WITH_MUSIC", "SPEECH_WITH_NOISE"}
LABEL_OFFSET_SEC = 900.0  # audio is clipped from 15:00 (900s)


@dataclass
class Segment:
    start: float
    end: float
    is_speech: bool


def load_labels(csv_path: Path, video_id: str) -> list[Segment]:
    segments: list[Segment] = []
    with open(csv_path) as f:
        for row in csv.reader(f):
            if row[0] != video_id:
                continue
            start = float(row[1]) - LABEL_OFFSET_SEC
            end = float(row[2]) - LABEL_OFFSET_SEC
            if end <= 0:
                continue
            start = max(0.0, start)
            segments.append(Segment(start, end, row[3] in SPEECH_LABELS))
    segments.sort(key=lambda s: s.start)
    return segments


def segments_to_frame_labels(segments: list[Segment], n_frames: int, frame_dur_sec: float) -> np.ndarray:
    """Convert segment labels to per-frame binary ground truth."""
    labels = np.full(n_frames, -1, dtype=np.int8)  # -1 = unlabeled
    for seg in segments:
        i_start = int(seg.start / frame_dur_sec)
        i_end = int(seg.end / frame_dur_sec)
        i_start = max(0, min(i_start, n_frames))
        i_end = max(0, min(i_end, n_frames))
        labels[i_start:i_end] = 1 if seg.is_speech else 0
    return labels


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------


def load_wav_frames(wav_path: Path, frame_samples: int) -> list[bytes]:
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        raw = wf.readframes(wf.getnframes())
    frames = []
    stride = frame_samples * 2  # 16-bit
    for i in range(0, len(raw) - stride + 1, stride):
        frames.append(raw[i : i + stride])
    return frames


# ---------------------------------------------------------------------------
# VAD runners
# ---------------------------------------------------------------------------


@dataclass
class VADScores:
    """Per-frame VAD scores from a single model on a single file."""

    video_id: str
    model_name: str
    scores: np.ndarray  # float32, one per frame
    elapsed_sec: float = 0.0


def run_maai_vap(frames: list[bytes], video_id: str) -> VADScores:
    from voice_pipeline.turn_taking.maai_vap import MaAIVAPModel

    vap = MaAIVAPModel(tts_sample_rate=24000)

    scores = []
    last_score = 0.0
    t0 = time.monotonic()
    for frame in frames:
        x1 = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        x2 = np.zeros(len(x1), dtype=np.float32)

        vap._buf_x1 = np.concatenate([vap._buf_x1, x1])
        vap._buf_x2 = np.concatenate([vap._buf_x2, x2])

        if len(vap._buf_x1) < vap._audio_frame_size:
            scores.append(last_score)
            continue

        wav1 = vap._buf_x1.reshape(1, 1, -1)
        wav2 = vap._buf_x2.reshape(1, 1, -1)
        e1, vap._h1, vap._c1 = vap._sess1.run(None, {"waveform": wav1, "h_in": vap._h1, "c_in": vap._c1})
        e2, vap._h2, vap._c2 = vap._sess2.run(None, {"waveform": wav2, "h_in": vap._h2, "c_in": vap._c2})
        if vap._use_onnx_transformer:
            out = vap._process_transformer_onnx(e1, e2)
        else:
            out = vap._process_transformer_pytorch(e1, e2)

        vap._buf_x1 = vap._buf_x1[-vap._FRAME_CTX_PADDING :].copy()
        vap._buf_x2 = vap._buf_x2[-vap._FRAME_CTX_PADDING :].copy()

        last_score = float(out["vad"][0])
        scores.append(last_score)

    elapsed = time.monotonic() - t0
    return VADScores(video_id, "maai_vap", np.array(scores, dtype=np.float32), elapsed)


def run_silero_vad(frames: list[bytes], video_id: str, sample_rate: int) -> VADScores:
    """Run Silero VAD. Silero requires 512 samples minimum at 16kHz,
    so we feed 512-sample windows and interpolate scores back to
    per-pipeline-frame resolution."""
    import torch

    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)

    # Concatenate all frames into one array
    all_pcm = np.concatenate([np.frombuffer(f, dtype=np.int16) for f in frames]).astype(np.float32) / 32768.0

    # Silero accepts 512 samples at 16kHz
    silero_window = 512
    silero_scores: list[float] = []

    t0 = time.monotonic()
    for i in range(0, len(all_pcm) - silero_window + 1, silero_window):
        chunk = all_pcm[i : i + silero_window]
        tensor = torch.from_numpy(chunk)
        prob = model(tensor, sample_rate).item()
        silero_scores.append(prob)
    elapsed = time.monotonic() - t0

    # Map silero scores back to pipeline frames via nearest-neighbor
    frame_samples = len(np.frombuffer(frames[0], dtype=np.int16))
    n_frames = len(frames)
    scores = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        frame_center = (i + 0.5) * frame_samples
        silero_idx = int(frame_center / silero_window)
        silero_idx = min(silero_idx, len(silero_scores) - 1)
        scores[i] = silero_scores[silero_idx]

    return VADScores(video_id, "silero", scores, elapsed)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class Metrics:
    model_name: str
    auc_roc: float
    auc_pr: float
    optimal_threshold: float
    f1_at_optimal: float
    precision_at_optimal: float
    recall_at_optimal: float
    n_frames: int = 0


def compute_metrics(gt: np.ndarray, scores: np.ndarray, model_name: str) -> Metrics:
    from sklearn.metrics import (
        auc,
        precision_recall_curve,
        roc_auc_score,
    )

    mask = gt >= 0
    gt_valid = gt[mask]
    sc_valid = scores[mask]

    auc_roc = roc_auc_score(gt_valid, sc_valid)

    precision, recall, thresholds_pr = precision_recall_curve(gt_valid, sc_valid)
    auc_pr = auc(recall, precision)

    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    opt_t = float(thresholds_pr[best_idx]) if best_idx < len(thresholds_pr) else 0.5

    return Metrics(
        model_name=model_name,
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        optimal_threshold=opt_t,
        f1_at_optimal=float(f1_scores[best_idx]),
        precision_at_optimal=float(precision[best_idx]),
        recall_at_optimal=float(recall[best_idx]),
        n_frames=int(mask.sum()),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="VAD benchmark: MaAI VAP vs Silero")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/ava_speech"),
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--frame-dur-ms", type=int, default=30)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    label_csv = args.data_dir / "ava_speech_labels_v1.csv"
    if not label_csv.exists():
        print(f"Labels not found: {label_csv}")
        sys.exit(1)

    wav_files = sorted(args.data_dir.glob("*.wav"))
    if not wav_files:
        print(f"No WAV files in {args.data_dir}")
        sys.exit(1)
    if args.max_files:
        wav_files = wav_files[: args.max_files]

    frame_samples = args.sample_rate * args.frame_dur_ms // 1000
    frame_dur_sec = args.frame_dur_ms / 1000.0

    out_dir = args.data_dir / "results"
    out_dir.mkdir(exist_ok=True)

    all_gt: list[np.ndarray] = []
    all_vap: list[np.ndarray] = []
    all_silero: list[np.ndarray] = []

    for wav_path in wav_files:
        video_id = wav_path.stem
        print(f"\n{'=' * 60}")
        print(f"Processing: {video_id}")

        segments = load_labels(label_csv, video_id)
        if not segments:
            print("  No labels found, skipping")
            continue

        frames = load_wav_frames(wav_path, frame_samples)
        n_frames = len(frames)
        gt = segments_to_frame_labels(segments, n_frames, frame_dur_sec)

        labeled_ratio = (gt >= 0).sum() / n_frames * 100
        speech_ratio = (gt == 1).sum() / max((gt >= 0).sum(), 1) * 100
        print(f"  Frames: {n_frames}, labeled: {labeled_ratio:.0f}%, speech: {speech_ratio:.0f}%")

        print("  Running MaAI VAP...")
        vap_scores = run_maai_vap(frames, video_id)
        print(f"    Done in {vap_scores.elapsed_sec:.1f}s")

        print("  Running Silero VAD...")
        silero_scores = run_silero_vad(frames, video_id, args.sample_rate)
        print(f"    Done in {silero_scores.elapsed_sec:.1f}s")

        # Per-file metrics
        for s in [vap_scores, silero_scores]:
            m = compute_metrics(gt, s.scores, s.model_name)
            print(
                f"  [{m.model_name}] AUC-ROC={m.auc_roc:.4f}  AUC-PR={m.auc_pr:.4f}  "
                f"opt_t={m.optimal_threshold:.3f}  F1={m.f1_at_optimal:.4f}  "
                f"P={m.precision_at_optimal:.4f}  R={m.recall_at_optimal:.4f}"
            )

        # Save per-file raw scores
        csv_path = out_dir / f"{video_id}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "time_sec", "gt", "maai_vap", "silero"])
            for i in range(n_frames):
                w.writerow(
                    [
                        i,
                        f"{i * frame_dur_sec:.3f}",
                        int(gt[i]),
                        f"{vap_scores.scores[i]:.6f}",
                        f"{silero_scores.scores[i]:.6f}",
                    ]
                )
        print(f"  Saved: {csv_path}")

        all_gt.append(gt)
        all_vap.append(vap_scores.scores)
        all_silero.append(silero_scores.scores)

    # Aggregate metrics
    if not all_gt:
        print("No files processed")
        sys.exit(1)

    gt_all = np.concatenate(all_gt)
    vap_all = np.concatenate(all_vap)
    silero_all = np.concatenate(all_silero)

    print(f"\n{'=' * 60}")
    print(f"AGGREGATE ({len(all_gt)} files)")
    print(f"{'=' * 60}")
    for name, scores in [("maai_vap", vap_all), ("silero", silero_all)]:
        m = compute_metrics(gt_all, scores, name)
        print(f"  [{m.model_name}]")
        print(f"    AUC-ROC:    {m.auc_roc:.4f}")
        print(f"    AUC-PR:     {m.auc_pr:.4f}")
        print(f"    Optimal t:  {m.optimal_threshold:.3f}")
        print(f"    F1:         {m.f1_at_optimal:.4f}")
        print(f"    Precision:  {m.precision_at_optimal:.4f}")
        print(f"    Recall:     {m.recall_at_optimal:.4f}")
        print(f"    Frames:     {m.n_frames}")


if __name__ == "__main__":
    main()
