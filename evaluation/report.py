"""Generate eval report from sessions.json + eval DB.

Joins session mapping with pipeline_traces and messages tables
to produce a unified result JSON.

Usage:
    uv run python -m evaluation.report data/eval/results
    uv run python -m evaluation.report data/eval/results --output report.json
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


def _summarize_stage_calls(bucket: dict) -> tuple[dict, dict | None]:
    """Reduce a per-turn call bucket to a compact ``stage_calls`` dict + issues.

    vap/turngpt are collapsed to count/avg/max (raw per-frame rows would bloat
    the report); tts keeps its few per-call rows; ``counts`` is the collapsed
    module.operation tally for the ⑦ timeline overview. Returns
    ``(stage_calls, call_issues_or_None)``.
    """
    stage_calls: dict = {}
    for mod, bad_status in (("vap", "overrun"), ("turngpt", "slow")):
        samples = bucket[mod]
        if not samples:
            continue
        ms = [m for m, _ in samples]
        summary = {"count": len(ms), "avg_ms": round(sum(ms) / len(ms), 1), "max_ms": round(max(ms), 1)}
        bad = sum(1 for _, s in samples if s == bad_status)
        if bad:
            summary[bad_status] = bad
        stage_calls[mod] = summary
    if bucket["tts"]:
        stage_calls["tts"] = bucket["tts"]
    if bucket["counts"]:
        stage_calls["counts"] = bucket["counts"]
    issues = None
    if bucket["retry_count"] or bucket["error_count"]:
        issues = {"retry_count": bucket["retry_count"], "error_count": bucket["error_count"]}
    return stage_calls, issues


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
        conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='call_records'").fetchone()
    )
    # Pre-column DBs (turn_index added later) fall back to legacy gate-only attribution.
    has_turn_index = has_call_records and bool(
        conn.execute("SELECT 1 FROM pragma_table_info('call_records') WHERE name='turn_index'").fetchone()
    )

    # Pre-fetch messages and traces per session
    session_ids = list({e["session_id"] for e in entries})
    msg_cache: dict[str, list[dict]] = {}
    trace_cache: dict[str, list] = {}
    stage_cache: dict[str, dict] = {}  # sid -> {turn_index -> raw call bucket}
    gate_cache: dict[str, dict] = {}  # sid -> {turn_index -> [gate events]}
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

        rows = conn.execute(
            "SELECT * FROM pipeline_traces WHERE session_id = ? ORDER BY id",
            (sid,),
        ).fetchall()
        # cancelled trace는 잠정 turn_shift가 취소된 흔적이라 턴과 1:1이 아님 —
        # 조인에서 제외하고 직후 확정 trace의 턴에 cancelled_turn_shifts로 귀속.
        # 잔여(trailing) cancel은 확정 trace 없이 끝난 턴(감지 타임아웃 등)의 것.
        aligned: list[tuple[sqlite3.Row, int]] = []
        pending_cancels = 0
        for row in rows:
            if row["outcome"] == "cancelled":
                pending_cancels += 1
            else:
                aligned.append((row, pending_cancels))
                pending_cancels = 0
        trace_cache[sid] = (aligned, pending_cancels)

        if has_call_records and has_turn_index:
            # One pass over the session's call records, bucketed by the
            # turn_index column → per-turn stage summaries, similarity-gate
            # events, and API issue counts. (turn_index is the exchange the
            # call belongs to; see SQLiteCallStore.current_turn_index.)
            rows = conn.execute(
                "SELECT module, operation, elapsed_ms, status, metadata, turn_index "
                "FROM call_records WHERE session_id = ? ORDER BY id",
                (sid,),
            ).fetchall()
            per_turn: dict[int, dict] = {}
            gate_by_turn: dict[int, list[dict]] = {}
            for r in rows:
                ti = r["turn_index"]
                bucket = per_turn.setdefault(
                    ti,
                    {"vap": [], "turngpt": [], "tts": [], "counts": {}, "retry_count": 0, "error_count": 0},
                )
                op_key = f"{r['module']}.{r['operation']}"
                bucket["counts"][op_key] = bucket["counts"].get(op_key, 0) + 1
                if r["status"] == "retry":
                    bucket["retry_count"] += 1
                elif r["status"] in ("error", "timeout"):
                    bucket["error_count"] += 1
                mod = r["module"]
                if mod == "vap":
                    bucket["vap"].append((r["elapsed_ms"], r["status"]))
                elif mod == "turngpt":
                    bucket["turngpt"].append((r["elapsed_ms"], r["status"]))
                elif mod == "tts":
                    entry = {"operation": r["operation"], "elapsed_ms": r["elapsed_ms"], "status": r["status"]}
                    # The meaningful TTS timing lives on the stream op (synthesize is
                    # ~0ms for lazy-generator vendors); lift ttfc/audio out of metadata.
                    if r["operation"] == "stream" and r["metadata"]:
                        try:
                            m = json.loads(r["metadata"])
                            entry["ttfc_ms"] = m.get("ttfc_ms")
                            entry["audio_sec"] = m.get("audio_sec")
                        except json.JSONDecodeError:
                            pass
                    bucket["tts"].append(entry)
                elif mod == "similarity_gate":
                    try:
                        meta = json.loads(r["metadata"]) if r["metadata"] else {}
                    except json.JSONDecodeError:
                        meta = {}
                    gate_by_turn.setdefault(ti, []).append(
                        {"operation": r["operation"], "elapsed_ms": r["elapsed_ms"], **meta}
                    )
            stage_cache[sid] = per_turn
            gate_cache[sid] = gate_by_turn
        elif has_call_records:
            # Legacy DB without the turn_index column: recover similarity-gate
            # attribution from metadata (no per-turn stage summaries available).
            rows = conn.execute(
                "SELECT operation, elapsed_ms, metadata FROM call_records "
                "WHERE session_id = ? AND module = 'similarity_gate' ORDER BY id",
                (sid,),
            ).fetchall()
            gate_by_turn = {}
            for r in rows:
                try:
                    meta = json.loads(r["metadata"]) if r["metadata"] else {}
                except json.JSONDecodeError:
                    meta = {}
                gate_by_turn.setdefault(meta.get("turn_index", 0), []).append(
                    {"operation": r["operation"], "elapsed_ms": r["elapsed_ms"], **meta}
                )
            gate_cache[sid] = gate_by_turn

    # Track turn index within each session for multi-turn
    session_turn_idx: dict[str, int] = {}

    turns = []
    for entry in entries:
        sid = entry["session_id"]
        idx = session_turn_idx.get(sid, 0)
        session_turn_idx[sid] = idx + 1

        msg_pairs = msg_cache.get(sid, [])
        traces, trailing_cancels = trace_cache.get(sid, ([], 0))

        if idx < len(msg_pairs):
            system_text = msg_pairs[idx]["user"]
            response_text = msg_pairs[idx]["assistant"]
        else:
            system_text = ""
            response_text = ""

        if idx < len(traces):
            trace, cancelled_shifts = traces[idx]
        else:
            trace = None
            cancelled_shifts = trailing_cancels if idx == len(traces) else 0
        latency = _extract_latency(trace)
        outcome = trace["outcome"] if trace else None
        asr_text = entry.get("asr_text") or system_text

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
        # sqlite3.Row: `in trace` checks values, not column names — keys() required
        if trace is not None and "speculative_attempts" in trace.keys():  # noqa: SIM118
            turn_data["speculative_attempts"] = trace["speculative_attempts"]
        if cancelled_shifts:
            turn_data["cancelled_turn_shifts"] = cancelled_shifts
        gate_events = gate_cache.get(sid, {}).get(idx, [])
        if gate_events:
            turn_data["similarity_events"] = gate_events
        bucket = stage_cache.get(sid, {}).get(idx)
        if bucket is not None:
            stage_calls, call_issues = _summarize_stage_calls(bucket)
            if stage_calls:
                turn_data["stage_calls"] = stage_calls
            if call_issues:
                turn_data["call_issues"] = call_issues
        for key in (
            "scenario_id",
            "voice",
            "snr",
            "condition",
            "interrupt_audio",
            "interrupt_delay_sec",
            "interrupt_played",
            "text_mode",
            "turn_shift_reason",
            "expect_wait",
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
