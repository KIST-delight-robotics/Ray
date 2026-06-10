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
    "interrupt_latency_ms",
)


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

    # Check if call_records table exists
    has_call_records = bool(
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='call_records'"
        ).fetchone()
    )

    # Pre-fetch messages and traces per session
    session_ids = list({e["session_id"] for e in entries})
    msg_cache: dict[str, list[dict]] = {}
    trace_cache: dict[str, list] = {}
    call_cache: dict[str, dict] = {}
    for sid in session_ids:
        rows = conn.execute(
            "SELECT item_json FROM messages WHERE session_id = ? ORDER BY msg_id",
            (sid,),
        ).fetchall()
        msg_pairs: list[dict] = []
        i = 0
        items = [json.loads(r[0]) for r in rows]
        while i < len(items):
            pair: dict = {"user": "", "assistant": ""}
            if i < len(items) and items[i].get("role") == "user":
                pair["user"] = items[i].get("content", "")
                i += 1
            if i < len(items) and items[i].get("role") == "assistant":
                pair["assistant"] = items[i].get("content", "")
                i += 1
            msg_pairs.append(pair)
        msg_cache[sid] = msg_pairs

        traces = conn.execute(
            "SELECT * FROM pipeline_traces WHERE session_id = ? ORDER BY id",
            (sid,),
        ).fetchall()
        trace_cache[sid] = traces

        if has_call_records:
            rows = conn.execute(
                "SELECT module, status FROM call_records WHERE session_id = ? AND status != 'ok'",
                (sid,),
            ).fetchall()
            retry_count = sum(1 for r in rows if r["status"] == "retry")
            error_count = sum(1 for r in rows if r["status"] in ("error", "timeout"))
            if retry_count or error_count:
                call_cache[sid] = {"retry_count": retry_count, "error_count": error_count}

    # Track turn index within each session for multi-turn
    session_turn_idx: dict[str, int] = {}

    turns = []
    for entry in entries:
        sid = entry["session_id"]
        idx = session_turn_idx.get(sid, 0)
        session_turn_idx[sid] = idx + 1

        msg_pairs = msg_cache.get(sid, [])
        traces = trace_cache.get(sid, [])

        if idx < len(msg_pairs):
            system_text = msg_pairs[idx]["user"]
            response_text = msg_pairs[idx]["assistant"]
        else:
            system_text = ""
            response_text = ""

        trace = traces[idx] if idx < len(traces) else None
        latency = _extract_latency(trace)
        outcome = trace["outcome"] if trace else None
        asr_text = entry.get("asr_text") or system_text

        call_issues = call_cache.get(sid)

        turn_data = {
            "suite_name": entry["suite_name"],
            "session_id": sid,
            "question_id": entry["question_id"],
            "input_text": entry["input_text"],
            "asr_text": asr_text,
            "system_text": system_text,
            "response_text": response_text,
            "latency": latency,
            "outcome": outcome,
            "success": entry["success"],
            "error": entry.get("error"),
            "turn_detection_delay_ms": entry.get("turn_detection_delay_ms"),
        }
        if call_issues:
            turn_data["call_issues"] = call_issues
        for key in (
            "scenario_id",
            "voice",
            "interrupt_audio",
            "interrupt_delay_sec",
            "interrupt_played",
            "text_mode",
            "turn_shift_reason",
            "retrieved_episodes",
            "target_sessions",
            "target_episode_ids",
        ):
            if key in entry:
                turn_data[key] = entry[key]

        turns.append(turn_data)

    conn.close()

    successful = sum(1 for t in turns if t["success"])
    report: dict = {
        "started_at": session_data.get("started_at", ""),
        "finished_at": session_data.get("finished_at", ""),
        "total": len(turns),
        "successful": successful,
        "failed": len(turns) - successful,
        "eval_db": str(db_path),
        "seed_session_map": session_data.get("seed_session_map"),
        "seed_file": session_data.get("seed_file"),
        "turns": turns,
    }
    if session_data.get("config"):
        report["config"] = session_data["config"]
    return report


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
