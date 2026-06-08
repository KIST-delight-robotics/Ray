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
import logging
import re
import sqlite3
import sys
from pathlib import Path

from num2words import num2words

logger = logging.getLogger("eval.score")

# ---------------------------------------------------------------------------
# ASR — Word Error Rate
# ---------------------------------------------------------------------------


def _digits_to_words(text: str) -> str:
    """Convert digit sequences in text to word form for fair WER comparison."""

    def _replace(m: re.Match[str]) -> str:
        s = m.group(0)
        if re.match(r"\d+(st|nd|rd|th)$", s):
            n = int(re.match(r"\d+", s).group())  # type: ignore[union-attr]
            return num2words(n, to="ordinal")
        if ":" in s:
            parts = s.split(":")
            return num2words(int(parts[0])) + " " + num2words(int(parts[1]))
        if "." in s:
            return num2words(float(s))
        return num2words(int(s))

    return re.sub(r"\d+[:.]\d+|\d+(st|nd|rd|th)|\d+", _replace, text)


def _normalize(text: str) -> list[str]:
    """Lowercase, normalize digits to words, strip punctuation, split."""
    text = _digits_to_words(text.lower())
    cleaned = ""
    for ch in text:
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
# LLM Response Quality — Judge evaluation
# ---------------------------------------------------------------------------

_JUDGE_MODEL = "gpt-5.5"

_QUALITY_SUITES: dict[str, dict[str, str]] = {
    "lq_factual": {
        "criterion": "correctness",
        "rubric": (
            "1: Completely incorrect\n"
            "2: Major factual errors\n"
            "3: Partially correct with notable inaccuracies\n"
            "4: Mostly correct with minor imprecision\n"
            "5: Completely accurate"
        ),
    },
    "lq_advice": {
        "criterion": "helpfulness",
        "rubric": (
            "1: Useless or harmful\n"
            "2: Vague, generic, little practical value\n"
            "3: Somewhat helpful but lacking specificity\n"
            "4: Practical and actionable\n"
            "5: Excellent — specific, actionable, well-prioritized"
        ),
    },
    "lq_casual": {
        "criterion": "engagement",
        "rubric": (
            "1: Kills the conversation or non-sequitur\n"
            "2: Minimal, disinterested response\n"
            "3: Adequate but doesn't advance the conversation\n"
            "4: Engages naturally, shows genuine interest\n"
            "5: Warmly responds and naturally continues dialogue"
        ),
    },
    "lq_empathy": {
        "criterion": "empathy",
        "rubric": (
            "1: Dismissive or tone-deaf\n"
            "2: Acknowledges situation but lacks warmth\n"
            "3: Basic empathy, somewhat formulaic\n"
            "4: Warm and emotionally attuned\n"
            "5: Deeply empathetic — validates feelings with genuine care"
        ),
    },
    "lq_voice_adaptation": {
        "criterion": "format_adaptation",
        "rubric": (
            "1: Uses lists, bullet points, code blocks, or markdown\n"
            "2: Partially text-formatted (numbered steps, markdown)\n"
            "3: Mixed — some voice-friendly restructuring with text remnants\n"
            "4: Mostly voice-friendly, conveys structured info conversationally\n"
            "5: Perfectly adapted — all information in natural spoken form"
        ),
    },
    "lq_multi_turn": {
        "criterion": "context_coherence",
        "rubric": (
            "1: Completely ignores prior conversation\n"
            "2: Minimal reference to prior turns\n"
            "3: Some context usage but misses key details\n"
            "4: Good context integration, references prior discussion naturally\n"
            "5: Seamless — builds on prior context as a human would"
        ),
    },
    "lq_wrong_premise": {
        "criterion": "correction_quality",
        "rubric": (
            "1: Accepts and reinforces the false premise\n"
            "2: Vaguely hints something might be wrong\n"
            "3: Corrects but awkwardly or condescendingly\n"
            "4: Tactfully corrects with accurate information\n"
            "5: Gracefully corrects — informative, respectful, accurate"
        ),
    },
    "lq_impossible": {
        "criterion": "boundary_communication",
        "rubric": (
            "1: Pretends to fulfill the request\n"
            "2: Confusing or unclear refusal\n"
            "3: Refuses but offers no help or alternatives\n"
            "4: Honestly communicates limitation with helpful context\n"
            "5: Transparent about limitation, suggests practical alternatives"
        ),
    },
}

_COMMON_RUBRIC = """\
relevance (1-5):
  1: Completely ignores or misunderstands the input
  2: Partially related but misses the core intent
  3: Addresses the topic but misses key nuances
  4: Directly addresses the input with appropriate detail
  5: Perfectly addresses the input

voice_appropriateness (1-5):
  1: Completely unsuitable — very long, uses markdown/lists/code/URLs
  2: Mostly unsuitable — too long or contains visual formatting
  3: Acceptable but could be more concise or spoken-friendly
  4: Well-suited for voice — appropriate length, natural structure
  5: Perfectly concise and structured for spoken delivery

naturalness (1-5):
  1: Robotic, overly formal, or template-generated
  2: Somewhat stiff or unnatural phrasing
  3: Acceptable but noticeably AI-like
  4: Natural conversational tone with minor stiffness
  5: Completely natural, indistinguishable from human conversation"""


def _build_judge_messages(suite_name: str, turns: list[dict], *, multi_turn: bool = False) -> list[dict]:
    cfg = _QUALITY_SUITES[suite_name]
    criterion = cfg["criterion"]

    if multi_turn:
        system = (
            "You are evaluating a multi-turn conversation from a voice conversation robot.\n"
            "The robot speaks to users through a physical speaker — "
            "responses must be suitable for listening, not reading.\n\n"
            "Evaluate the ENTIRE conversation as a whole on these criteria:\n\n"
            f"{_COMMON_RUBRIC}\n\n"
            f"{criterion} (1-5):\n{cfg['rubric']}\n\n"
            "Return a JSON object with: relevance, voice_appropriateness, "
            f"naturalness, {criterion}, reasoning (one sentence in Korean)."
        )
        parts = [
            f"User: {t['input_text']}\nResponse: {t['response_text']}"
            for t in turns
        ]
        user = "Evaluate this multi-turn conversation as a whole:\n\n" + "\n\n".join(parts)
    else:
        system = (
            "You are evaluating responses from a voice conversation robot.\n"
            "The robot speaks to users through a physical speaker — "
            "responses must be suitable for listening, not reading.\n\n"
            f"Score each response on these criteria:\n\n"
            f"{_COMMON_RUBRIC}\n\n"
            f"{criterion} (1-5):\n{cfg['rubric']}\n\n"
            'Return a JSON object with an "evaluations" array. '
            "Each element must have: question_id, relevance, voice_appropriateness, "
            f"naturalness, {criterion}, reasoning (one sentence in Korean)."
        )
        parts = [f"[{t['question_id']}]\nUser: {t['input_text']}\nResponse: {t['response_text']}" for t in turns]
        user = "Evaluate these responses:\n\n" + "\n\n".join(parts)

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _call_judge(messages: list[dict]) -> dict | None:
    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=_JUDGE_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        logger.error("Judge API call failed", exc_info=True)
        return None


def _apply_judge_result(
    suite_name: str,
    turns: list[dict],
    result: dict,
    criterion_agg: dict[str, list[float]],
) -> None:
    criterion = _QUALITY_SUITES[suite_name]["criterion"]
    eval_by_id = {e["question_id"]: e for e in result["evaluations"]}
    score_keys = ["relevance", "voice_appropriateness", "naturalness", criterion]

    for turn in turns:
        ev = eval_by_id.get(turn["question_id"])
        if not ev:
            continue
        scores = {}
        for key in score_keys:
            val = ev.get(key)
            if isinstance(val, (int, float)) and 1 <= val <= 5:
                scores[key] = val
                criterion_agg.setdefault(key, []).append(val)
        turn["quality_scores"] = scores
        turn["quality_reasoning"] = ev.get("reasoning", "")


def _score_quality(turns: list[dict]) -> dict:
    """Score response quality for quality-suite turns. Mutates turns. Returns summary."""
    by_suite: dict[str, list[dict]] = {}
    for turn in turns:
        suite = turn["suite_name"]
        if suite in _QUALITY_SUITES and turn.get("response_text"):
            by_suite.setdefault(suite, []).append(turn)

    if not by_suite:
        return {}

    criterion_agg: dict[str, list[float]] = {}
    suite_summaries: dict[str, dict] = {}

    for suite_name, suite_turns in by_suite.items():
        if suite_name == "lq_multi_turn":
            criterion = _QUALITY_SUITES[suite_name]["criterion"]
            score_keys = ["relevance", "voice_appropriateness", "naturalness", criterion]
            by_session: dict[str, list[dict]] = {}
            for t in suite_turns:
                by_session.setdefault(t.get("session_id", ""), []).append(t)
            for session_turns in by_session.values():
                messages = _build_judge_messages(suite_name, session_turns, multi_turn=True)
                result = _call_judge(messages)
                if not result:
                    logger.error("No valid judge result for %s", suite_name)
                    continue
                scores: dict[str, float] = {}
                for key in score_keys:
                    val = result.get(key)
                    if isinstance(val, (int, float)) and 1 <= val <= 5:
                        scores[key] = val
                        criterion_agg.setdefault(key, []).append(val)
                reasoning = result.get("reasoning", "")
                for turn in session_turns:
                    turn["quality_scores"] = scores
                    turn["quality_reasoning"] = reasoning
        else:
            messages = _build_judge_messages(suite_name, suite_turns)
            result = _call_judge(messages)
            if result and "evaluations" in result:
                _apply_judge_result(suite_name, suite_turns, result, criterion_agg)
            else:
                logger.error("No valid judge result for %s", suite_name)

        scored = [t for t in suite_turns if "quality_scores" in t]
        if scored:
            vals = [v for t in scored for v in t["quality_scores"].values()]
            suite_summaries[suite_name] = {
                "mean_score": round(sum(vals) / len(vals), 2),
                "turn_count": len(scored),
            }

    all_vals = [v for vals in criterion_agg.values() for v in vals]
    if not all_vals:
        return {}

    return {
        "mean_score": round(sum(all_vals) / len(all_vals), 2),
        "by_criterion": {name: round(sum(vals) / len(vals), 2) for name, vals in criterion_agg.items()},
        "by_suite": suite_summaries,
    }


# ---------------------------------------------------------------------------
# Memory — Writer, Retriever Recall, Memory Quality
# ---------------------------------------------------------------------------


def _score_writer(report: dict) -> dict:
    """Score memory writer quality by judging extracted episodes against seed sessions."""
    seed_file = report.get("seed_file")
    seed_session_map = report.get("seed_session_map")
    eval_db = report.get("eval_db")

    if not seed_file or not seed_session_map:
        return {}

    seed_path = Path(seed_file)
    if not seed_path.exists():
        logger.error("Seed file not found: %s", seed_file)
        return {}

    seed_data = json.loads(seed_path.read_text())
    seed_sessions = seed_data.get("sessions", [])

    db_path = Path(eval_db) if eval_db else None
    if not db_path or not db_path.exists():
        logger.error("Eval DB not found: %s", eval_db)
        return {}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    by_session: list[dict] = []
    all_scores: list[float] = []

    for idx_str, session_id in seed_session_map.items():
        session_index = int(idx_str)
        if session_index >= len(seed_sessions):
            continue

        seed_session = seed_sessions[session_index]
        utterances = seed_session.get("utterances", [])

        rows = conn.execute(
            "SELECT id, text, importance FROM episodes WHERE session_id = ?",
            (session_id,),
        ).fetchall()

        episodes_text = (
            "\n".join(f"- [importance={r['importance']}] {r['text']}" for r in rows)
            if rows
            else "(no episodes extracted)"
        )

        utterances_text = "\n".join(f"- {u['role']}: {u['text']}" for u in utterances)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are evaluating a memory extraction system. Given a conversation session "
                    "and the episodes extracted from it, evaluate:\n"
                    "- completeness (1-5): Are all important facts and events captured?\n"
                    "- accuracy (1-5): Are the extracted episodes factually faithful to the conversation?\n"
                    "- granularity (1-5): Is the level of detail appropriate? "
                    "(1=too coarse or too fine, 5=well-balanced)\n\n"
                    "Return JSON with: completeness, accuracy, granularity, reasoning (one sentence in Korean)."
                ),
            },
            {
                "role": "user",
                "content": (f"Session utterances:\n{utterances_text}\n\nExtracted episodes:\n{episodes_text}"),
            },
        ]

        result = _call_judge(messages)
        if not result:
            continue

        entry: dict = {
            "session_index": session_index,
            "episode_count": len(rows),
            "completeness": result.get("completeness", 0),
            "accuracy": result.get("accuracy", 0),
            "granularity": result.get("granularity", 0),
            "reasoning": result.get("reasoning", ""),
            "utterances": [{"role": u["role"], "text": u["text"]} for u in utterances],
            "episodes": [{"id": r["id"], "text": r["text"], "importance": r["importance"]} for r in rows],
        }
        by_session.append(entry)

        scores = [entry["completeness"], entry["accuracy"], entry["granularity"]]
        valid = [s for s in scores if isinstance(s, (int, float)) and 1 <= s <= 5]
        all_scores.extend(valid)

    conn.close()

    if not all_scores:
        return {}

    return {
        "mean_score": round(sum(all_scores) / len(all_scores), 2),
        "by_session": by_session,
    }


def _compute_retriever_recall(turns: list[dict]) -> dict:
    """Compute recall for turns that have target_episode_ids. No LLM needed."""
    per_probe: list[dict] = []

    for turn in turns:
        target_ids = turn.get("target_episode_ids")
        if not target_ids:
            continue

        retrieved = turn.get("retrieved_episodes", [])
        retrieved_ids = set()
        for ep in retrieved:
            if isinstance(ep, dict):
                ep_id = ep.get("id") or ep.get("episode_id")
                if ep_id is not None:
                    retrieved_ids.add(ep_id)
            else:
                retrieved_ids.add(ep)

        found = sum(1 for tid in target_ids if tid in retrieved_ids)
        recall = found / len(target_ids) if target_ids else 0.0
        recall = round(recall, 4)

        turn["retriever_recall"] = recall

        per_probe.append(
            {
                "question_id": turn.get("question_id", ""),
                "recall": recall,
                "found": found,
                "total_targets": len(target_ids),
            }
        )

    if not per_probe:
        return {}

    mean_recall = round(sum(p["recall"] for p in per_probe) / len(per_probe), 4)

    return {
        "mean_recall": mean_recall,
        "per_probe": per_probe,
    }


def _score_memory_quality(turns: list[dict]) -> dict:
    """Score memory-augmented response quality for mem_* suite turns."""
    all_scores: dict[str, list[float]] = {}
    precision_values: list[float] = []
    scored_count = 0

    criteria = [
        "response_relevance",
        "memory_appropriateness",
        "factual_accuracy",
        "naturalness",
    ]

    for turn in turns:
        suite = turn.get("suite_name", "")
        if not suite.startswith("mem_") or not turn.get("response_text"):
            continue

        retrieved = turn.get("retrieved_episodes", [])
        if retrieved:
            ep_lines = []
            for i, ep in enumerate(retrieved, 1):
                if isinstance(ep, dict):
                    ep_text = ep.get("text", str(ep))
                    ep_id = ep.get("id") or ep.get("episode_id", "?")
                    ep_lines.append(f"{i}. [id={ep_id}] {ep_text}")
                else:
                    ep_lines.append(f"{i}. {ep}")
            episodes_formatted = "\n".join(ep_lines)
        else:
            episodes_formatted = "(no episodes retrieved)"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are evaluating a memory-augmented conversational AI. "
                    "Given the user's question, retrieved memory episodes, and the AI's response, evaluate:\n"
                    "(1) episode_relevance — for each retrieved episode, is it relevant to the question? "
                    "Return an array of booleans, one per episode, in order.\n"
                    "(2) response_relevance (1-5): Does the response address the user's question?\n"
                    "(3) memory_appropriateness (1-5): Does the response use memory naturally "
                    "without over-sharing or ignoring relevant memories?\n"
                    "(4) factual_accuracy (1-5): Is the response factually consistent with the episodes?\n"
                    "(5) naturalness (1-5): Does the response sound natural and conversational?\n\n"
                    "Return JSON with keys: episode_relevance (array of booleans), "
                    "response_relevance (int), memory_appropriateness (int), "
                    "factual_accuracy (int), naturalness (int), reasoning (one sentence in Korean)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User question: {turn['input_text']}\n\n"
                    f"Retrieved episodes:\n{episodes_formatted}\n\n"
                    f"AI response: {turn['response_text']}"
                ),
            },
        ]

        result = _call_judge(messages)
        if not result:
            continue

        # Store scores on turn
        turn_scores: dict[str, int | float] = {}
        for c in criteria:
            val = result.get(c)
            if isinstance(val, (int, float)) and 1 <= val <= 5:
                turn_scores[c] = val
                all_scores.setdefault(c, []).append(val)

        turn["memory_scores"] = turn_scores
        turn["memory_reasoning"] = result.get("reasoning", "")

        # Compute precision from episode_relevance
        ep_relevance = result.get("episode_relevance", [])
        if ep_relevance and retrieved:
            relevant_count = sum(1 for r in ep_relevance if r)
            precision = relevant_count / len(retrieved)
            turn["retriever_precision"] = round(precision, 4)
            precision_values.append(turn["retriever_precision"])
        elif not retrieved:
            turn["retriever_precision"] = None
        else:
            turn["retriever_precision"] = None

        scored_count += 1

    if not all_scores:
        return {}

    flat = [v for vals in all_scores.values() for v in vals]
    by_criterion = {name: round(sum(vals) / len(vals), 2) for name, vals in all_scores.items()}

    return {
        "mean_score": round(sum(flat) / len(flat), 2),
        "by_criterion": by_criterion,
        "mean_precision": round(sum(precision_values) / len(precision_values), 4) if precision_values else None,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _asr_suite_summary(wers: list[float]) -> dict:
    if not wers:
        return {}
    return {
        "mean_wer": round(sum(wers) / len(wers), 4),
        "perfect_count": sum(1 for w in wers if w == 0.0),
        "total_scored": len(wers),
    }


def score_report(report: dict) -> dict:
    """Add evaluation scores to a report."""
    asr_scores: list[float] = []
    asr_by_suite: dict[str, list[float]] = {}
    turn_detection_delays: list[float] = []
    latency_values: dict[str, list[float]] = {
        "turn_shift_to_playback_ms": [],
        "llm_ttft_ms": [],
        "tts_ttfc_ms": [],
        "bridge_ms": [],
    }

    for turn in report["turns"]:
        is_text_mode = turn.get("text_mode", False)
        is_interruption = "interrupt_delay_sec" in turn

        # ASR scoring (text mode, interruption 제외)
        if not is_text_mode and not is_interruption and turn["success"] and turn["asr_text"]:
            wer = compute_wer(turn["input_text"], turn["asr_text"])
            turn["asr_score"] = wer
            asr_scores.append(wer["wer"])
            suite = turn["suite_name"]
            asr_by_suite.setdefault(suite, []).append(wer["wer"])

        # Turn detection delay (text mode 제외)
        if not is_text_mode:
            td_delay = turn.get("turn_detection_delay_ms")
            if td_delay is not None and td_delay > 0:
                turn_detection_delays.append(td_delay)

        # Pipeline latency collection (text mode 제외)
        if not is_text_mode:
            latency = turn.get("latency", {})
            for key in latency_values:
                val = latency.get(key)
                if val and val > 0:
                    latency_values[key].append(val)

    # Aggregate ASR
    asr_summary = {}
    if asr_scores:
        asr_summary = {
            **_asr_suite_summary(asr_scores),
            "by_suite": {suite: _asr_suite_summary(wers) for suite, wers in asr_by_suite.items()},
        }

    # Aggregate latency
    latency_summary = {key: compute_latency_stats(vals) for key, vals in latency_values.items() if vals}
    if turn_detection_delays:
        latency_summary["turn_detection_delay_ms"] = compute_latency_stats(turn_detection_delays)

    # Interruption scoring
    int_turns = [t for t in report["turns"] if "interrupt_delay_sec" in t]
    int_summary = {}
    if int_turns:
        by_delay: dict[float, dict[str, int]] = {}
        int_latencies: list[float] = []
        for t in int_turns:
            delay = t["interrupt_delay_sec"]
            outcome = t.get("outcome", "unknown")
            was_played = t.get("interrupt_played", False)

            if delay not in by_delay:
                by_delay[delay] = {"total": 0, "truncated": 0, "completed": 0, "cancelled": 0, "na": 0}
            bucket = by_delay[delay]
            bucket["total"] += 1

            if not was_played:
                bucket["na"] += 1
            elif outcome == "truncated":
                bucket["truncated"] += 1
            elif outcome == "cancelled":
                bucket["cancelled"] += 1
            else:
                bucket["completed"] += 1

            lat = t.get("latency", {}).get("interrupt_latency_ms")
            if lat and lat > 0 and outcome == "truncated":
                int_latencies.append(lat)

        testable = sum(b["total"] - b["na"] for b in by_delay.values())
        detected = sum(b["truncated"] + b["cancelled"] for b in by_delay.values())
        int_summary = {
            "detection_rate": round(detected / testable, 4) if testable else 0.0,
            "detected": detected,
            "testable": testable,
            "by_delay": {
                str(d): {
                    "detected": b["truncated"] + b["cancelled"],
                    "testable": b["total"] - b["na"],
                    "truncated": b["truncated"],
                    "completed": b["completed"],
                    "na": b["na"],
                }
                for d, b in sorted(by_delay.items())
            },
            "latency": compute_latency_stats(int_latencies),
        }

    # Quality scoring
    quality_summary = _score_quality(report["turns"])

    # Memory scoring
    memory_summary = {}
    memory_turns = [t for t in report["turns"] if t.get("suite_name", "").startswith("mem_")]
    if memory_turns:
        writer_summary = _score_writer(report)
        recall_summary = _compute_retriever_recall(memory_turns)
        quality_summary_mem = _score_memory_quality(report["turns"])
        memory_summary = {
            "writer": writer_summary,
            "retriever_recall": recall_summary,
            "quality": quality_summary_mem,
        }

    # Success rate
    total = len(report["turns"])
    successful = sum(1 for t in report["turns"] if t["success"])

    return {
        **report,
        "scores": {
            "success_rate": round(successful / total, 4) if total else 0.0,
            "asr": asr_summary,
            "latency": latency_summary,
            "interruption": int_summary,
            "quality": quality_summary,
            "memory": memory_summary,
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

    interruption = scores.get("interruption", {})
    if interruption:
        print(
            f"\nInterruption (detection rate: {interruption['detected']}/{interruption['testable']}"
            f" = {interruption['detection_rate']:.0%}):"
        )
        for delay_str, b in interruption.get("by_delay", {}).items():
            testable = b["testable"]
            if testable == 0:
                print(f"  {delay_str}s: n/a (response ended before interrupt)")
            else:
                rate = b["detected"] / testable
                print(
                    f"  {delay_str}s: {b['detected']}/{testable} ({rate:.0%})"
                    f"  truncated={b['truncated']} completed={b['completed']}" + (f" na={b['na']}" if b["na"] else "")
                )
        int_lat = interruption.get("latency", {})
        if int_lat:
            print(
                f"  Detection latency: mean={int_lat['mean_ms']:.0f}ms"
                f"  median={int_lat['median_ms']:.0f}ms  p95={int_lat['p95_ms']:.0f}ms"
            )

    quality = scores.get("quality", {})
    if quality:
        print(f"\nLLM Response Quality (mean: {quality['mean_score']:.1f}/5):")
        for name, val in quality.get("by_criterion", {}).items():
            print(f"  {name}: {val:.1f}")
        by_suite = quality.get("by_suite", {})
        if by_suite:
            print()
            for suite, stats in by_suite.items():
                print(f"  {suite}: {stats['mean_score']:.1f}/5 ({stats['turn_count']} turns)")

    quality_turns = [t for t in scored["turns"] if "quality_scores" in t]
    if quality_turns:
        print("\nPer-question Quality:")
        for turn in quality_turns:
            qs = turn["quality_scores"]
            avg = sum(qs.values()) / len(qs) if qs else 0
            scores_str = " ".join(f"{k}={v}" for k, v in qs.items())
            print(f"  [{avg:.1f}] {turn['question_id']}: {scores_str}")
            if turn.get("quality_reasoning"):
                print(f"        {turn['quality_reasoning'][:80]}")

    # Memory evaluation
    memory = scores.get("memory", {})
    if memory:
        writer = memory.get("writer", {})
        if writer:
            print(f"\nMemory Writer Quality (mean: {writer['mean_score']:.1f}/5):")
            for s in writer.get("by_session", []):
                print(
                    f"  session {s['session_index']}: "
                    f"completeness={s['completeness']} accuracy={s['accuracy']} "
                    f"granularity={s['granularity']} ({s['episode_count']} episodes)"
                )
                if s.get("reasoning"):
                    print(f"        {s['reasoning'][:80]}")

        recall = memory.get("retriever_recall", {})
        if recall:
            print(f"\nRetriever Recall (mean: {recall['mean_recall']:.2%}):")
            for p in recall.get("per_probe", []):
                print(f"  {p['question_id']}: {p['recall']:.0%} ({p['found']}/{p['total_targets']})")

        mem_quality = memory.get("quality", {})
        if mem_quality:
            print(f"\nMemory Usage Quality (mean: {mem_quality['mean_score']:.1f}/5):")
            if mem_quality.get("mean_precision") is not None:
                print(f"  Retriever precision: {mem_quality['mean_precision']:.2%}")
            for name, val in mem_quality.get("by_criterion", {}).items():
                print(f"  {name}: {val:.1f}")

        mem_turns = [t for t in scored["turns"] if "memory_scores" in t]
        if mem_turns:
            print("\nPer-question Memory:")
            for turn in mem_turns:
                ms = turn["memory_scores"]
                avg = sum(ms.values()) / len(ms) if ms else 0
                parts = [f"{k}={v}" for k, v in ms.items()]
                recall_str = ""
                if "retriever_recall" in turn:
                    recall_str = f" recall={turn['retriever_recall']:.0%}"
                prec_str = ""
                if turn.get("retriever_precision") is not None:
                    prec_str = f" precision={turn['retriever_precision']:.0%}"
                print(f"  [{avg:.1f}] {turn.get('question_id', '?')}: {' '.join(parts)}{recall_str}{prec_str}")
                if turn.get("memory_reasoning"):
                    print(f"        {turn['memory_reasoning'][:80]}")

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
