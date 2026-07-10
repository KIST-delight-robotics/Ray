"""CLI entry point for the memory benchmark harness.

Usage:
    uv run python -m evaluation.memory_bench ingest --data data/eval/locomo/locomo10.json \\
        --run-dir data/eval/locomo/runs/r1 [--conversations conv-26,conv-30] [--workers 4]
    uv run python -m evaluation.memory_bench answer --run-dir data/eval/locomo/runs/r1 \\
        [--conversations conv-26] [--workers 8] [--model gpt-4o-mini]
    uv run python -m evaluation.memory_bench score --run-dir data/eval/locomo/runs/r1 \\
        [--workers 8] [--judge-model gpt-4o-mini]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def _parse_sample_ids(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [s.strip() for s in value.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(prog="evaluation.memory_bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Replay sessions through MemoryWriter into per-perspective DBs")
    p_ingest.add_argument("--data", required=True, help="Path to locomo10.json")
    p_ingest.add_argument("--run-dir", required=True)
    p_ingest.add_argument("--conversations", help="Comma-separated sample_ids (default: all)")
    p_ingest.add_argument("--workers", type=int, default=4)
    p_ingest.add_argument("--writer-model", default=None, help="Extraction LLM (default: production model)")

    p_answer = sub.add_parser("answer", help="Retrieve memories and answer benchmark questions")
    p_answer.add_argument("--run-dir", required=True)
    p_answer.add_argument("--data", help="Path to locomo10.json (default: from run config)")
    p_answer.add_argument("--conversations", help="Comma-separated sample_ids (default: all)")
    p_answer.add_argument("--workers", type=int, default=8)
    p_answer.add_argument("--model", default=None, help="Answer LLM")
    p_answer.add_argument(
        "--half-life-days",
        type=float,
        default=None,
        help="Experimental override for the retriever recency-decay half-life (default: production value)",
    )

    p_score = sub.add_parser("score", help="Judge answers, attribute failures, write scores.json")
    p_score.add_argument("--run-dir", required=True)
    p_score.add_argument("--workers", type=int, default=8)
    p_score.add_argument("--judge-model", default=None, help="Judge LLM")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    run_dir = Path(args.run_dir)

    if args.command == "ingest":
        from evaluation.memory_bench.common import DEFAULT_WRITER_MODEL
        from evaluation.memory_bench.ingest import ingest_run

        ingest_run(
            data_path=args.data,
            run_dir=run_dir,
            sample_ids=_parse_sample_ids(args.conversations),
            workers=args.workers,
            writer_model=args.writer_model or DEFAULT_WRITER_MODEL,
        )
    elif args.command == "answer":
        from evaluation.memory_bench.answer import answer_run
        from evaluation.memory_bench.common import DEFAULT_ANSWER_MODEL

        answer_run(
            run_dir=run_dir,
            data_path=args.data,
            sample_ids=_parse_sample_ids(args.conversations),
            workers=args.workers,
            answer_model=args.model or DEFAULT_ANSWER_MODEL,
            half_life_days=args.half_life_days,
        )
    elif args.command == "score":
        from evaluation.memory_bench.common import DEFAULT_JUDGE_MODEL
        from evaluation.memory_bench.score import format_summary, score_run

        scores = score_run(
            run_dir=run_dir,
            workers=args.workers,
            judge_model=args.judge_model or DEFAULT_JUDGE_MODEL,
        )
        print()
        print(format_summary(scores))


if __name__ == "__main__":
    main()
