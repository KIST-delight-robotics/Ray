"""Generate eval report from sessions.json + eval DB.

Joins session mapping with pipeline_traces and messages tables
to produce a unified result JSON.

Usage:
    uv run python scripts/eval/report.py data/eval/results
    uv run python scripts/eval/report.py data/eval/results --output report.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

_LATENCY_COLUMNS = (
    "memory_ms",
    "context_ms",
    "llm_ms",
    "llm_ttft_ms",
    "tts_ms",
    "tts_ttfc_ms",
    "prepare_to_streaming_ms",
    "turn_shift_to_playback_ms",
    "speculative_ms",
    "bridge_ms",
)


def _extract_text(messages: list[tuple[str]], role: str) -> str:
    """Extract text for a given role from message item_json rows."""
    texts = []
    for (item_json,) in messages:
        item = json.loads(item_json)
        if item.get("role") == role:
            texts.append(item.get("content", ""))
    return " ".join(texts)


def _extract_latency(row: sqlite3.Row | None) -> dict[str, float]:
    """Extract latency metrics from a trace row."""
    if row is None:
        return {}
    return {col: row[col] for col in _LATENCY_COLUMNS if row[col]}


def build_report(results_dir: Path) -> dict:
    """Build a report from sessions.json and eval.db."""
    sessions_path = results_dir / "sessions.json"
    db_path = results_dir / "eval.db"

    if not sessions_path.exists():
        print(f"Error: {sessions_path} not found", file=sys.stderr)
        sys.exit(1)
    if not db_path.exists():
        print(f"Error: {db_path} not found", file=sys.stderr)
        sys.exit(1)

    session_data = json.loads(sessions_path.read_text())
    entries = session_data["sessions"]

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    turns = []
    for entry in entries:
        sid = entry["session_id"]

        messages = conn.execute(
            "SELECT item_json FROM messages WHERE session_id = ? ORDER BY msg_id",
            (sid,),
        ).fetchall()

        trace = conn.execute(
            "SELECT * FROM pipeline_traces WHERE session_id = ? AND outcome = 'completed' ORDER BY id DESC LIMIT 1",
            (sid,),
        ).fetchone()

        asr_text = _extract_text(messages, "user")
        response_text = _extract_text(messages, "assistant")
        latency = _extract_latency(trace)

        turns.append(
            {
                "suite_name": entry["suite_name"],
                "question_id": entry["question_id"],
                "input_text": entry["input_text"],
                "asr_text": asr_text,
                "response_text": response_text,
                "latency": latency,
                "success": entry["success"],
                "error": entry.get("error"),
                "vap_detection_delay_ms": entry.get("vap_detection_delay_ms"),
            }
        )

    conn.close()

    successful = sum(1 for t in turns if t["success"])
    return {
        "started_at": session_data.get("started_at", ""),
        "finished_at": session_data.get("finished_at", ""),
        "total": len(turns),
        "successful": successful,
        "failed": len(turns) - successful,
        "turns": turns,
    }


def print_summary(report: dict) -> None:
    """Print a human-readable summary to stdout."""
    print(f"\n{'=' * 60}")
    print(f"Eval Report: {report['started_at']} — {report['finished_at']}")
    print(f"Total: {report['total']} | Success: {report['successful']} | Failed: {report['failed']}")
    print(f"{'=' * 60}")

    for turn in report["turns"]:
        status = "OK" if turn["success"] else "FAIL"
        latency = turn.get("latency", {})
        ts_pb = latency.get("turn_shift_to_playback_ms", 0)

        print(f"\n[{status}] {turn['question_id']} ({turn['suite_name']})")
        print(f"  Input:    {turn['input_text']}")
        print(f"  ASR:      {turn['asr_text']}")
        print(f"  Response: {turn['response_text'][:80]}")
        if ts_pb:
            llm_ttft = latency.get("llm_ttft_ms", 0)
            tts_ttfc = latency.get("tts_ttfc_ms", 0)
            print(f"  Latency:  ts→pb={ts_pb:.0f}ms  llm_ttft={llm_ttft:.0f}ms  tts_ttfc={tts_ttfc:.0f}ms")
        if turn.get("error"):
            print(f"  Error:    {turn['error']}")

    print(f"\n{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate eval report")
    parser.add_argument("results_dir", help="Directory with sessions.json and eval.db")
    parser.add_argument("--output", default=None, help="Output JSON path (default: <results_dir>/report.json)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    report = build_report(results_dir)

    output_path = Path(args.output) if args.output else results_dir / "report.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report saved: {output_path}")

    print_summary(report)


if __name__ == "__main__":
    main()
