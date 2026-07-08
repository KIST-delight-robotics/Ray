"""Plot VAD benchmark results from saved CSVs.

Usage:
    uv run python scripts/bench/plot_vad.py
    uv run python scripts/bench/plot_vad.py --results-dir data/ava_speech/results
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def load_all(results_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_gt, all_vap, all_silero = [], [], []
    for csv_path in sorted(results_dir.glob("*.csv")):
        with open(csv_path) as f:
            r = csv.DictReader(f)
            for row in r:
                gt = int(row["gt"])
                if gt < 0:
                    continue
                all_gt.append(gt)
                all_vap.append(float(row["maai_vap"]))
                all_silero.append(float(row["silero"]))

    gt = np.array(all_gt)
    vap = np.array(all_vap)
    silero = np.array(all_silero)

    # Forward-fill VAP zeros
    last = 0.0
    for i in range(len(vap)):
        if vap[i] == 0.0:
            vap[i] = last
        else:
            last = vap[i]

    return gt, vap, silero


def load_single(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times, gts, vaps, sileros = [], [], [], []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            times.append(float(row["time_sec"]))
            gts.append(int(row["gt"]))
            vaps.append(float(row["maai_vap"]))
            sileros.append(float(row["silero"]))

    vap = np.array(vaps)
    last = 0.0
    for i in range(len(vap)):
        if vap[i] == 0.0:
            vap[i] = last
        else:
            last = vap[i]

    return np.array(times), np.array(gts), vap, np.array(sileros)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("data/ava_speech/results"))
    parser.add_argument("--out", type=Path, default=Path("data/ava_speech/results/vad_comparison.png"))
    args = parser.parse_args()

    gt, vap, silero = load_all(args.results_dir)

    # Pick first file for timeline
    first_csv = sorted(args.results_dir.glob("*.csv"))[0]
    t_time, t_gt, t_vap, t_silero = load_single(first_csv)

    fig, axes = plt.subplots(4, 1, figsize=(12, 14))

    # --- 1. ROC Curve ---
    ax = axes[0]
    for name, scores, color in [("MaAI VAP", vap, "#e74c3c"), ("Silero VAD", silero, "#2980b9")]:
        fpr, tpr, _ = roc_curve(gt, scores)
        auc_val = roc_auc_score(gt, scores)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # --- 2a. Score Distribution — MaAI VAP ---
    bins = np.linspace(0, 1, 51)
    ax = axes[1]
    ax.hist(vap[gt == 0], bins=bins, alpha=0.6, color="#95a5a6", label="Silence", density=True)
    ax.hist(vap[gt == 1], bins=bins, alpha=0.6, color="#e74c3c", label="Speech", density=True)
    ax.set_xlabel("VAD Score")
    ax.set_ylabel("Density")
    ax.set_title("MaAI VAP — Score Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- 2b. Score Distribution — Silero VAD ---
    ax = axes[2]
    ax.hist(silero[gt == 0], bins=bins, alpha=0.6, color="#95a5a6", label="Silence", density=True)
    ax.hist(silero[gt == 1], bins=bins, alpha=0.6, color="#2980b9", label="Speech", density=True)
    ax.set_xlabel("VAD Score")
    ax.set_ylabel("Density")
    ax.set_title("Silero VAD — Score Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- 3. Timeline (first file, 60s window) ---
    ax = axes[3]
    window = (t_time >= 60) & (t_time < 120)
    tw = t_time[window]
    ax.fill_between(tw, 0, 1, where=(t_gt[window] == 1), alpha=0.15, color="green", label="GT speech")
    ax.plot(tw, t_vap[window], color="#e74c3c", lw=1, alpha=0.8, label="MaAI VAP")
    ax.plot(tw, t_silero[window], color="#2980b9", lw=1, alpha=0.8, label="Silero VAD")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("VAD Score")
    ax.set_title(f"Timeline — {first_csv.stem} (60s–120s)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(tw[0], tw[-1])
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
