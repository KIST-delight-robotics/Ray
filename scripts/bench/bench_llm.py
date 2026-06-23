"""LLM standalone speed benchmark — model axis × input-content axis.

Measures LLM latency in isolation (no ASR/TTS/turn-taking), answering two
questions:

  1. Model axis    — does speed differ across models? (TTFT, total latency)
  2. Input axis    — does speed differ by how much context goes in?
                     (history depth, memory on/off → input token count)

Builds a realistic conversation prompt via ContextBuilder (system prompt,
profile, session summaries, history turns, memory) and runs a full
``models × input-variants`` matrix, reporting per cell:

  - TTFT (time to first token) — dominated by prefill, the cleanest signal
    for the *input-content* effect.
  - Total latency — also depends on output length, so read it alongside TTFT.
  - Prompt caching effectiveness (cached_tokens / input_tokens)
  - Token usage

Models and input variants are configured IN CODE below (see ``MODELS`` and
``INPUT_VARIANTS``), not via CLI flags. Edit those lists to change the run.
Reasoning models use ``model:effort`` syntax (e.g. ``gpt-5.4:low``); plain
names are non-reasoning. Both can be mixed freely.

Usage::

    # Run the matrix defined in MODELS × INPUT_VARIANTS, 5 rounds each (default)
    uv run python scripts/bench/bench_llm.py

    # More rounds per cell for tighter medians
    uv run python scripts/bench/bench_llm.py --rounds 10

Requires: OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field

from voice_pipeline.context.context_builder import ContextBuilder
from voice_pipeline.core.types import LLMMetrics
from voice_pipeline.history.conversation_history import ConversationHistory
from voice_pipeline.history.storage_backend import MemoryStorageBackend
from voice_pipeline.llm.llm import OpenAILLM
from voice_pipeline.llm.prompts import DEFAULT_SYSTEM_PROMPT
from voice_pipeline.llm.token_counter import create_token_counter
from voice_pipeline.memory.types import Episode, MemoryReadResult, Profile

# ---------------------------------------------------------------------------
# Benchmark configuration — EDIT HERE
# ---------------------------------------------------------------------------

# Models to compare. Reasoning models use "model:effort" (e.g. "gpt-5.4:low",
# "gpt-5.4:none"); plain names are non-reasoning. Mix freely.
MODELS: list[str] = [
    "gpt-5.5:none",  # newest reasoning model, effort off
    "gpt-5.4:none",  # current production default (reasoning model, effort off)
    "gpt-5.4-mini:none",  # smaller reasoning model, effort off
    "gpt-4o",  # non-reasoning, larger
    "gpt-4o-mini",  # non-reasoning, fast tier
]

# Cap on output tokens per call. Keeps total-latency comparisons fair across
# models. Note: for reasoning models this budget also covers reasoning tokens.
MAX_TOKENS = 256


@dataclass(frozen=True)
class InputVariant:
    """One point on the input-content axis.

    The model is held fixed while these knobs change the *input* size and
    composition, isolating how much the prompt content affects speed.
    """

    label: str
    history_turns: int  # number of prior turns injected (0 = none)
    include_memory: bool  # whether the retrieved-memory block is present


# Input variants to compare (model held fixed). Ordered small → large so the
# input-axis effect reads top-to-bottom.
INPUT_VARIANTS: list[InputVariant] = [
    # Single variant: mirrors production (6 history turns + memory block).
    InputVariant("typical", history_turns=6, include_memory=True),
]

# ---------------------------------------------------------------------------
# Realistic test data
# ---------------------------------------------------------------------------

_PROFILES = [
    Profile(None, "basic_info", "name", "재헌", "2026-04-01 10:00:00"),
    Profile(None, "basic_info", "age", "25", "2026-04-01 10:00:00"),
    Profile(None, "interest", "movie", "SF, 특히 놀란 감독 작품들", "2026-04-10 15:00:00"),
    Profile(None, "interest", "music", "재즈, Lo-fi", "2026-04-05 12:00:00"),
    Profile(None, "personality", "humor", "약간 드라이한 유머 좋아함", "2026-04-08 18:00:00"),
]

_EPISODES = [
    Episode(
        id=1,
        text="사용자가 인터스텔라를 보고 울었다고 함. 비 오는 날에 혼자 봤다고.",
        timestamp="2026-03-15 14:00:00",
        session_id="s1",
        importance=0.8,
        last_cited_at="2026-03-15 14:00:00",
    ),
    Episode(
        id=2,
        text="사용자가 듄 2편이 1편보다 훨씬 낫다고 평가함.",
        timestamp="2026-03-20 10:00:00",
        session_id="s2",
        importance=0.6,
        last_cited_at="2026-03-20 10:00:00",
    ),
    Episode(
        id=3,
        text="사용자가 요즘 빌 에반스 앨범을 자주 듣고 있다고 함. 특히 'Waltz for Debby'.",
        timestamp="2026-04-01 20:00:00",
        session_id="s3",
        importance=0.7,
        last_cited_at="2026-04-01 20:00:00",
    ),
]

_SESSION_SUMMARIES = [
    "[2026-04-18 세션] 사용자가 주말에 영화 마라톤을 했다고 함. "
    "테넷을 다시 봤는데 여전히 어렵다고. 놀란 감독 영화 중 가장 좋아하는 건 인터스텔라.",
    "[2026-04-19 세션] 카페에서 재즈 라이브 공연을 봤다고 함. "
    "피아노 트리오 구성이었고, 빌 에반스 곡들을 많이 연주했다고.",
]

# Varying user inputs to simulate realistic multi-turn benchmark.
_USER_INPUTS = [
    "오늘 날씨가 좋아서 산책했어",
    "요즘 볼만한 영화 있을까?",
    "저번에 말한 그 재즈 앨범 또 들었어",
    "주말에 뭐 할지 고민이야",
    "놀란 감독 신작 나온다던데 알아?",
]

# Base history turns; repeated/truncated to reach a variant's desired depth.
_HISTORY_TURNS = [
    ("user", "안녕, 오늘 하루 어땠어?"),
    ("assistant", "안녕! 오늘도 이것저것 이야기 나눌 준비 됐어. 뭐 재밌는 일 있었어?"),
    ("user", "회사에서 좀 바빴는데, 퇴근하고 카페 갔어"),
    ("assistant", "카페! 좋다. 어떤 카페 갔어? 분위기 좋은 데?"),
    ("user", "응, 조용한 데. 재즈 틀어주는 곳이야"),
    ("assistant", "오 재즈 카페! 빌 에반스 같은 거 나왔으면 딱이었겠다."),
]


def _history_for(turns: int) -> list[tuple[str, str]]:
    """Build a history list of *turns* entries from the base turns."""
    if turns <= 0:
        return []
    repeats = (turns // len(_HISTORY_TURNS)) + 1
    return (_HISTORY_TURNS * repeats)[:turns]


# ---------------------------------------------------------------------------
# Benchmark data types
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Result of a single LLM call."""

    round_idx: int
    ttft_ms: int
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    response_text: str
    model_returned: str


@dataclass
class CellReport:
    """Aggregated results for one (model, input-variant) cell."""

    model: str
    variant: str
    runs: list[RunResult] = field(default_factory=list)

    @property
    def warm_runs(self) -> list[RunResult]:
        """Rounds excluding the first (cold) one, where cache is warm."""
        return self.runs[1:] if len(self.runs) > 1 else self.runs

    @property
    def warm_ttft(self) -> list[int]:
        return [r.ttft_ms for r in self.warm_runs]

    @property
    def warm_latency(self) -> list[int]:
        return [r.latency_ms for r in self.warm_runs]

    @property
    def avg_input_tokens(self) -> float:
        if not self.runs:
            return 0.0
        return statistics.mean(r.input_tokens for r in self.runs)


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------


def _build_context(variant: InputVariant) -> tuple[ContextBuilder, MemoryReadResult | None]:
    """Build a ContextBuilder + memory result for one input variant.

    Token counting for assembly uses a fixed gpt-4o counter — the assembled
    message dicts are model-agnostic and reused across all models.
    """
    token_counter = create_token_counter("gpt-4o")

    backend = MemoryStorageBackend()
    history = ConversationHistory(backend, token_counter)
    history.new_session("bench")

    for role, text in _history_for(variant.history_turns):
        if role == "user":
            history.add_user_message(text)
        else:
            history.add_assistant_message(text)

    cb = ContextBuilder(
        history,
        DEFAULT_SYSTEM_PROMPT,
        token_counter,
        profiles=_PROFILES,
        session_summaries=_SESSION_SUMMARIES,
    )

    memory_result: MemoryReadResult | None = None
    if variant.include_memory:
        memory_result = MemoryReadResult(
            episodes=_EPISODES,
            scores=[0.85, 0.72, 0.68],
            index_to_id={1: 1, 2: 2, 3: 3},
        )

    return cb, memory_result


def _run_single(llm: OpenAILLM, messages: list[dict], round_idx: int) -> RunResult:
    """Run a single LLM call and extract metrics."""
    stream = llm.generate(messages, tools=[])
    chunks: list[str] = []
    for chunk in stream:
        chunks.append(chunk)

    result = stream.result
    text = "".join(chunks)
    metrics: LLMMetrics | None = result.metrics

    if metrics is None:
        # Fallback — shouldn't happen with OpenAI
        return RunResult(
            round_idx=round_idx,
            ttft_ms=0,
            latency_ms=0,
            input_tokens=0,
            output_tokens=0,
            cached_tokens=0,
            reasoning_tokens=0,
            response_text=text,
            model_returned="unknown",
        )

    return RunResult(
        round_idx=round_idx,
        ttft_ms=metrics.ttft_ms,
        latency_ms=metrics.latency_ms,
        input_tokens=metrics.usage.input_tokens,
        output_tokens=metrics.usage.output_tokens,
        cached_tokens=metrics.usage.cached_tokens,
        reasoning_tokens=metrics.usage.reasoning_tokens,
        response_text=text,
        model_returned=metrics.model,
    )


def _make_llm(model_spec: str) -> OpenAILLM:
    """Build an OpenAILLM from a 'model' or 'model:effort' spec."""
    if ":" in model_spec:
        model_name, reasoning_effort = model_spec.split(":", 1)
    else:
        model_name, reasoning_effort = model_spec, None

    return OpenAILLM(
        model=model_name,
        temperature=0.7,
        max_tokens=MAX_TOKENS,
        tools=[],
        reasoning_effort=reasoning_effort,
    )


def run_benchmark(
    models: list[str],
    variants: list[InputVariant],
    rounds: int,
) -> list[CellReport]:
    """Run the full model × input-variant matrix."""
    # Build each variant's context once; reuse across all models.
    contexts = {v.label: _build_context(v) for v in variants}

    reports: list[CellReport] = []

    for model_spec in models:
        llm = _make_llm(model_spec)
        print(f"\n{'#' * 70}")
        print(f"  MODEL: {model_spec}")
        print(f"{'#' * 70}")

        for variant in variants:
            cb, memory_result = contexts[variant.label]
            report = CellReport(model=model_spec, variant=variant.label)

            print(f"\n  ── input='{variant.label}' (history={variant.history_turns}, memory={variant.include_memory})")

            for i in range(rounds):
                user_input = _USER_INPUTS[i % len(_USER_INPUTS)]
                messages = cb.build(user_input, memory_result=memory_result)

                result = _run_single(llm, messages, i)
                report.runs.append(result)

                cache_pct = result.cached_tokens / result.input_tokens * 100 if result.input_tokens > 0 else 0
                reasoning_str = f"  reasoning={result.reasoning_tokens}" if result.reasoning_tokens > 0 else ""
                print(
                    f"    [{i + 1}/{rounds}]  "
                    f"TTFT={result.ttft_ms:>4d}ms  "
                    f"total={result.latency_ms:>5d}ms  "
                    f"in={result.input_tokens:>5d}  "
                    f"cached={result.cached_tokens:>5d} ({cache_pct:4.1f}%)  "
                    f"out={result.output_tokens:>3d}"
                    f"{reasoning_str}"
                )

                if i < rounds - 1:
                    time.sleep(0.5)

            reports.append(report)
            time.sleep(0.5)
        time.sleep(1.0)

    return reports


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_stats(values: list[int] | list[float], unit: str = "ms") -> str:
    """Format min/median/mean/max stats."""
    if not values:
        return "no data"
    mn = min(values)
    mx = max(values)
    med = statistics.median(values)
    avg = statistics.mean(values)
    return f"min={mn:.0f}  med={med:.0f}  avg={avg:.0f}  max={mx:.0f} {unit}"


def _med(values: list[int] | list[float]) -> float:
    return statistics.median(values) if values else 0.0


def print_summary(
    reports: list[CellReport],
    models: list[str],
    variants: list[InputVariant],
) -> None:
    """Print two cross-tabs: model axis (input fixed) and input axis (model fixed)."""
    by_key = {(r.model, r.variant): r for r in reports}
    variant_labels = [v.label for v in variants]

    print(f"\n{'=' * 72}")
    print("  SUMMARY — warm rounds (2+); TTFT is the cleanest input-size signal")
    print(f"{'=' * 72}")

    # --- Axis 1: MODEL comparison, one table per input variant ---
    print("\n  [모델별] 같은 입력에서 모델 간 비교 (median, warm)")
    for v in variants:
        print(f"\n    input='{v.label}'")
        print(f"    {'model':<16s} {'TTFT':>8s} {'total':>8s} {'in_tok':>7s} {'cache%':>7s}")
        print(f"    {'-' * 50}")
        for m in models:
            r = by_key.get((m, v.label))
            if not r or not r.runs:
                continue
            ttft = _med(r.warm_ttft)
            lat = _med(r.warm_latency)
            cache = _med([rr.cached_tokens / rr.input_tokens * 100 if rr.input_tokens else 0 for rr in r.warm_runs])
            print(f"    {m:<16s} {ttft:>6.0f}ms {lat:>6.0f}ms {r.avg_input_tokens:>7.0f} {cache:>6.1f}%")

    # --- Axis 2: INPUT comparison, one table per model ---
    print("\n  [입력별] 같은 모델에서 입력 내용 간 비교 (median, warm)")
    for m in models:
        print(f"\n    model='{m}'")
        print(f"    {'input':<12s} {'in_tok':>7s} {'TTFT':>8s} {'total':>8s} {'cache%':>7s}")
        print(f"    {'-' * 50}")
        for label in variant_labels:
            r = by_key.get((m, label))
            if not r or not r.runs:
                continue
            ttft = _med(r.warm_ttft)
            lat = _med(r.warm_latency)
            cache = _med([rr.cached_tokens / rr.input_tokens * 100 if rr.input_tokens else 0 for rr in r.warm_runs])
            print(f"    {label:<12s} {r.avg_input_tokens:>7.0f} {ttft:>6.0f}ms {lat:>6.0f}ms {cache:>6.1f}%")

    # --- Per-cell detail (cold vs warm TTFT, reasoning) ---
    print(f"\n{'=' * 72}")
    print("  PER-CELL DETAIL")
    print(f"{'=' * 72}")
    for r in reports:
        if not r.runs:
            continue
        print(f"\n  {r.model}  /  input='{r.variant}'  ({len(r.runs)} rounds)")
        print(f"    TTFT:   {_fmt_stats([rr.ttft_ms for rr in r.runs])}")
        print(f"    Total:  {_fmt_stats([rr.latency_ms for rr in r.runs])}")
        if len(r.runs) >= 2:
            cold = r.runs[0].ttft_ms
            warm = _med(r.warm_ttft)
            print(f"    TTFT cold→warm: {cold}ms → {warm:.0f}ms (Δ={cold - warm:+.0f}ms, caching)")
        total_reasoning = sum(rr.reasoning_tokens for rr in r.runs)
        if total_reasoning > 0:
            print(f"    Reasoning: avg={total_reasoning / len(r.runs):.0f} tokens/call")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM standalone speed benchmark (model × input-content axes). "
        "Edit MODELS and INPUT_VARIANTS in the script to configure the run.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Rounds per (model, input) cell (default: 5; round 1 is the cold/cache-miss one)",
    )
    args = parser.parse_args()

    print("LLM Standalone Speed Benchmark")
    print(f"  Models ({len(MODELS)}): {', '.join(MODELS)}")
    print(f"  Input variants ({len(INPUT_VARIANTS)}): {', '.join(v.label for v in INPUT_VARIANTS)}")
    print(f"  Rounds per cell: {args.rounds}")
    print(f"  Max output tokens: {MAX_TOKENS}")

    try:
        reports = run_benchmark(MODELS, INPUT_VARIANTS, args.rounds)
        print_summary(reports, MODELS, INPUT_VARIANTS)
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
