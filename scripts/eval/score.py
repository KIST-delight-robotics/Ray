"""Evaluate ASR accuracy and turn-taking latency from eval results.

Reads the report JSON produced by report.py and computes:
  - ASR: Word Error Rate (WER) per question and overall
  - Turn-taking: latency statistics (mean, median, p95)

Usage:
    uv run python scripts/eval/report.py data/eval/results
    uv run python scripts/eval/score.py data/eval/results/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# ASR — Word Error Rate
# ---------------------------------------------------------------------------


def _normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into words."""
    cleaned = ""
    for ch in text.lower():
        if ch.isalnum() or ch == " ":
            cleaned += ch
        else:
            cleaned += " "
    return cleaned.split()


def _levenshtein(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Compute word-level edit distance. Returns (substitutions, deletions, insertions)."""
    n, m = len(ref), len(hyp)
    # dp[i][j] = (cost, subs, dels, ins)
    dp = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub = dp[i - 1][j - 1]
                dele = dp[i - 1][j]
                ins = dp[i][j - 1]
                candidates = [
                    (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                    (dele[0] + 1, dele[1], dele[2] + 1, dele[3]),
                    (ins[0] + 1, ins[1], ins[2], ins[3] + 1),
                ]
                dp[i][j] = min(candidates, key=lambda x: x[0])

    _, s, d, i = dp[n][m]
    return s, d, i


def compute_wer(reference: str, hypothesis: str) -> dict:
    """Compute Word Error Rate between reference and hypothesis text."""
    ref_words = _normalize(reference)
    hyp_words = _normalize(hypothesis)

    if not ref_words:
        return {"wer": 0.0 if not hyp_words else 1.0, "ref_words": 0, "errors": len(hyp_words)}

    subs, dels, ins = _levenshtein(ref_words, hyp_words)
    total_errors = subs + dels + ins
    wer = total_errors / len(ref_words)

    return {
        "wer": round(wer, 4),
        "ref_words": len(ref_words),
        "hyp_words": len(hyp_words),
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "errors": total_errors,
    }


# ---------------------------------------------------------------------------
# Turn-taking — Latency statistics
# ---------------------------------------------------------------------------


def _percentile(values: list[float], p: float) -> float:
    """Compute percentile (0-100) using nearest-rank method."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = max(0, min(int(len(sorted_v) * p / 100), len(sorted_v) - 1))
    return sorted_v[k]


def compute_latency_stats(latencies: list[float]) -> dict:
    """Compute summary statistics for a list of latency values (ms)."""
    if not latencies:
        return {}
    return {
        "count": len(latencies),
        "mean_ms": round(sum(latencies) / len(latencies), 1),
        "median_ms": round(_percentile(latencies, 50), 1),
        "p95_ms": round(_percentile(latencies, 95), 1),
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_report(report: dict) -> dict:
    """Add evaluation scores to a report."""
    asr_scores = []
    vap_detection_delays: list[float] = []
    latency_values: dict[str, list[float]] = {
        "turn_shift_to_playback_ms": [],
        "llm_ttft_ms": [],
        "tts_ttfc_ms": [],
        "bridge_ms": [],
    }

    for turn in report["turns"]:
        # ASR scoring
        if turn["success"] and turn["asr_text"]:
            wer = compute_wer(turn["input_text"], turn["asr_text"])
            turn["asr_score"] = wer
            asr_scores.append(wer["wer"])

        # VAP detection delay (pre-computed in run.py)
        vap_delay = turn.get("vap_detection_delay_ms")
        if vap_delay is not None and vap_delay > 0:
            vap_detection_delays.append(vap_delay)

        # Pipeline latency collection
        latency = turn.get("latency", {})
        for key in latency_values:
            val = latency.get(key)
            if val and val > 0:
                latency_values[key].append(val)

    # Aggregate ASR
    asr_summary = {}
    if asr_scores:
        asr_summary = {
            "mean_wer": round(sum(asr_scores) / len(asr_scores), 4),
            "perfect_count": sum(1 for w in asr_scores if w == 0.0),
            "total_scored": len(asr_scores),
        }

    # Aggregate latency
    latency_summary = {key: compute_latency_stats(vals) for key, vals in latency_values.items() if vals}
    if vap_detection_delays:
        latency_summary["vap_detection_delay_ms"] = compute_latency_stats(vap_detection_delays)

    # Success rate
    total = len(report["turns"])
    successful = sum(1 for t in report["turns"] if t["success"])

    return {
        **report,
        "scores": {
            "success_rate": round(successful / total, 4) if total else 0.0,
            "asr": asr_summary,
            "latency": latency_summary,
        },
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_scores(scored: dict) -> None:
    """Print a human-readable score summary."""
    scores = scored["scores"]
    print(f"\n{'=' * 60}")
    print("Evaluation Scores")
    print(f"{'=' * 60}")

    print(f"\nSuccess rate: {scores['success_rate']:.1%}")

    asr = scores.get("asr", {})
    if asr:
        print("\nASR (Word Error Rate):")
        print(f"  Mean WER:     {asr['mean_wer']:.2%}")
        print(f"  Perfect:      {asr['perfect_count']}/{asr['total_scored']}")

    latency = scores.get("latency", {})
    if latency:
        print("\nTurn-taking latency:")
        for key, stats in latency.items():
            label = key.removesuffix("_ms")
            print(f"  {label}:")
            print(f"    mean={stats['mean_ms']:.0f}ms  median={stats['median_ms']:.0f}ms  p95={stats['p95_ms']:.0f}ms")

    # Per-question ASR details
    turns_with_asr = [t for t in scored["turns"] if "asr_score" in t]
    if turns_with_asr:
        print("\nPer-question ASR:")
        for turn in turns_with_asr:
            wer = turn["asr_score"]["wer"]
            status = "OK" if wer == 0.0 else f"WER={wer:.0%}"
            print(f"  [{status:>8}] {turn['question_id']}: {turn['input_text'][:50]}")
            if wer > 0:
                print(f"             ASR: {turn['asr_text'][:50]}")

    print(f"\n{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score eval results")
    parser.add_argument("report", help="Path to report.json")
    parser.add_argument("--output", default=None, help="Output scored JSON (default: <dir>/scored.json)")
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"Error: {report_path} not found", file=sys.stderr)
        sys.exit(1)

    report = json.loads(report_path.read_text())
    scored = score_report(report)

    output_path = Path(args.output) if args.output else report_path.parent / "scored.json"
    output_path.write_text(json.dumps(scored, indent=2, ensure_ascii=False))
    print(f"Scored report saved: {output_path}")

    print_scores(scored)


if __name__ == "__main__":
    main()
