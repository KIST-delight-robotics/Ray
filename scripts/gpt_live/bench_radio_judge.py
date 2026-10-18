"""라디오 판정기 벤치 — 지금 판정기(gpt-5.4-mini, Responses)와 Decisions API(gpt-6-luna)를 같은 입력으로 비교한다.

실시간 세션은 없다. 입력은 지난 라디오 실험 결과(``var/experiments/radio_*/``)에 남은 판정기 호출의 (읽던 대사,
끼어든 뒤 대화)이고, 정답은 그 회차에 끼어든 발화로 정한다(``exp_radio_flow.UTTERANCES`` — 짧은 반응은 on, 나머지는
paused, 옛 off 기대값은 지금 기준대로 paused). 같은 입력은 한 번만 쓴다. 사용자 발화는 TTS 라 정확도는 참고치다.

모델:
    mini       운영 판정기 그대로 (``RadioJudge`` 와 같은 지시문·구조화 출력, wiring 과 같은 설정)
    luna       Decisions choice 질문 — 판정기 지시문의 첫 문단을 질문 지시로, on/paused 항목을 선택지 설명으로
    luna_full  Decisions choice 질문 — 판정기 지시문 전체를 질문 지시로, 선택지는 값만

    uv run --with openai==3.35.1 python scripts/gpt_live/bench_radio_judge.py accuracy --tag j1
    uv run --with openai==3.35.1 python scripts/gpt_live/bench_radio_judge.py latency --tag j1
    uv run --with openai==3.35.1 python scripts/gpt_live/bench_radio_judge.py tokens --tag j1
    uv run python scripts/gpt_live/bench_radio_judge.py report j1

accuracy 는 입력마다 모델별 ``--repeats`` 회를 병렬로, latency 는 뽑은 입력에 모델을 번갈아 하나씩 차례로 부른다
(병렬은 지연을 부풀린다). tokens 는 질문 수(1·2·4)에 따라 Decisions 입력 토큰이 어떻게 느는지 본다. 끊겨도 다시 돌리면
이미 한 호출은 건너뛴다. Decisions API 는 openai 3.26 이상이 필요해 프로젝트 의존성(3.14)은 두고 ``--with`` 로 돌린다.

결과: var/experiments/radio_judge/<번호>_<태그>/ — inputs.json(입력·정답), calls.jsonl(호출 하나에 한 줄),
misses.txt(report 가 쓰는 오답 목록), code/(이 결과를 낸 스크립트).
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import shutil
import statistics
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from exp_dirs import EXPERIMENTS, find_result_dir, result_dir
from exp_radio_flow import UTTERANCES

from voice_pipeline.adapters.llm_openai import OpenAILLM
from voice_pipeline.engines.gpt_live.instructions import (
    RADIO_JUDGE_FORMAT,
    RADIO_JUDGE_INSTRUCTIONS,
    build_radio_judge_prompt,
)
from voice_pipeline.engines.gpt_live.radio import letters

EXPERIMENT = "radio_judge"
MODELS = ("mini", "luna", "luna_full")
DECISIONS_MODEL = "gpt-6-luna"
COMPLETE_RATIO = 0.8  # 사용자 전사가 발화 글자의 이만큼 이상과 맞으면 "완전한 입력"
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# 1M 토큰당 달러 (pricing 페이지·Decisions 가이드, 2026-10-19). Decisions 는 입력만 과금한다.
PRICE_MINI = {"input": 0.75, "cached": 0.075, "output": 4.50}
PRICE_DECISIONS_INPUT = 0.10


# ---------------------------------------------------------------------------
# 입력
# ---------------------------------------------------------------------------


def _classifier_lists(obj: Any) -> Iterator[tuple[dict[str, Any], list[Any]]]:
    """결과 JSON 안의 ``classifier`` 목록과 그것을 가진 회차 dict."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "classifier" and isinstance(value, list):
                yield obj, value
            elif isinstance(value, (dict, list)):
                yield from _classifier_lists(value)
    elif isinstance(obj, list):
        for value in obj:
            if isinstance(value, (dict, list)):
                yield from _classifier_lists(value)


def _complete(exchange: list[list[str]], utterance: str) -> bool:
    user = letters("".join(text for who, text in exchange if who == "user"))
    target = letters(UTTERANCES[utterance][1])
    blocks = difflib.SequenceMatcher(None, target, user, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks) >= COMPLETE_RATIO * len(target)


def load_inputs() -> list[dict[str, Any]]:
    """지난 실험의 판정기 입력 — 같은 (대사, 대화)는 하나로 묶고 나온 횟수를 센다."""
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(EXPERIMENTS.glob("radio_*/*/*.json")):
        if path.name == "config.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for run, calls in _classifier_lists(data):
            utterance = run.get("utterance")
            if utterance not in UTTERANCES:
                continue
            for call in calls:
                if not isinstance(call, dict) or "line" not in call or "exchange" not in call:
                    continue
                key = (call["line"], json.dumps(call["exchange"], ensure_ascii=False))
                if key in seen:
                    seen[key]["count"] += 1
                    continue
                expected = UTTERANCES[utterance][2]
                seen[key] = {
                    "id": len(seen),
                    "line": call["line"],
                    "exchange": call["exchange"],
                    "utterance": utterance,
                    "label": "paused" if expected == "off" else expected,
                    "complete": _complete(call["exchange"], utterance),
                    "by": call.get("by", "user"),
                    "source": str(path.relative_to(EXPERIMENTS).parent),
                    "count": 1,
                }
    return list(seen.values())


# ---------------------------------------------------------------------------
# 판정기
# ---------------------------------------------------------------------------


def _split_judge_instructions() -> tuple[str, list[dict[str, str]]]:
    """판정기 지시문을 (첫 문단, on/paused 선택지와 설명)으로 나눈다 — 운영 문구와 함께 바뀌게."""
    head, *items = RADIO_JUDGE_INSTRUCTIONS.strip().split("\n- ")
    choices = []
    for item in items:
        value, _, description = item.partition(":")
        choices.append({"value": value.strip(), "description": " ".join(description.split())})
    if [c["value"] for c in choices] != ["on", "paused"]:
        raise RuntimeError(f"unexpected judge instructions: {[c['value'] for c in choices]}")
    return " ".join(head.split()), choices


def build_question(model: str, name: str = "state") -> dict[str, Any]:
    if model == "luna":
        instructions, choices = _split_judge_instructions()
        return {"type": "choice", "name": name, "instructions": instructions, "choices": choices}
    return {
        "type": "choice",
        "name": name,
        "instructions": RADIO_JUDGE_INSTRUCTIONS.strip(),
        "choices": [{"value": "on"}, {"value": "paused"}],
    }


class Judges:
    """모델 이름 → 판정 한 번. 결과는 상태·확률·토큰·지연을 담은 dict."""

    def __init__(self) -> None:
        import openai

        # wiring 의 radio_judge 와 같은 설정
        self._mini = OpenAILLM(model="gpt-5.4-mini", temperature=0.0, reasoning_effort="none", max_tokens=120, tools=[])
        self._client = openai.OpenAI(max_retries=2, timeout=30.0)
        self._questions = {m: build_question(m) for m in ("luna", "luna_full")}

    def run(self, model: str, item: dict[str, Any], questions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        prompt = build_radio_judge_prompt(item["line"], [tuple(t) for t in item["exchange"]])
        start = time.monotonic()
        try:
            if model == "mini":
                out = self._run_mini(prompt)
            else:
                out = self._run_decisions(prompt, questions or [self._questions[model]])
        except Exception as exc:  # noqa: BLE001 — 실패도 기록한다 (운영은 on 으로 본다)
            out = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        out["latency_ms"] = round((time.monotonic() - start) * 1000.0)
        return out

    def _run_mini(self, prompt: str) -> dict[str, Any]:
        stream = self._mini.generate(
            [{"role": "system", "content": RADIO_JUDGE_INSTRUCTIONS}, {"role": "user", "content": prompt}],
            tools=[],
            response_format=RADIO_JUDGE_FORMAT,
        )
        parsed = json.loads("".join(stream))
        metrics = stream.result.metrics
        usage = metrics.usage if metrics is not None else None
        return {
            "state": str(parsed.get("state", "")),
            "reason": str(parsed.get("reason", "")),
            "input_tokens": usage.input_tokens if usage else None,
            "cached_tokens": usage.cached_tokens if usage else None,
            "output_tokens": usage.output_tokens if usage else None,
        }

    def _run_decisions(self, prompt: str, questions: list[dict[str, Any]]) -> dict[str, Any]:
        resp = self._client.decisions.create(model=DECISIONS_MODEL, input=prompt, questions=questions)
        answer = resp.answers[0]
        usage = resp.usage
        out: dict[str, Any] = {
            "input_tokens": usage.input_tokens,
            "cached_tokens": usage.input_tokens_details.cached_tokens,
            "output_tokens": usage.output_tokens,
        }
        if answer.type == "refusal":
            out["refusal"] = True
            return out
        probs = {str(p.value): p.probability for p in answer.probabilities}
        out.update(state=str(answer.choice), p_paused=probs.get("paused"), confidence=answer.confidence)
        return out


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------


class CallLog:
    """``calls.jsonl`` — 여러 스레드가 한 줄씩 덧붙인다."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _prepare(tag: str) -> tuple[Path, list[dict[str, Any]], CallLog]:
    out_dir = result_dir(EXPERIMENT, tag)
    inputs_path = out_dir / "inputs.json"
    if inputs_path.exists():
        inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    else:
        inputs = load_inputs()
        inputs_path.write_text(json.dumps(inputs, ensure_ascii=False, indent=1), encoding="utf-8")
    code_dir = out_dir / "code"
    code_dir.mkdir(exist_ok=True)
    shutil.copy(Path(__file__), code_dir / Path(__file__).name)
    return out_dir, inputs, CallLog(out_dir / "calls.jsonl")


def _parallel(jobs: list[Callable[[], None]], workers: int) -> None:
    done = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(workers) as pool:
        for _ in pool.map(lambda job: job(), jobs):
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} ({time.monotonic() - t0:.0f}s)", flush=True)


def cmd_accuracy(args: argparse.Namespace) -> None:
    out_dir, inputs, log = _prepare(args.tag)
    items = inputs[: args.limit] if args.limit else inputs
    have = {(r["model"], r["input"], r["rep"]) for r in log.read() if r["pass"] == "accuracy"}
    judges = Judges()

    def job(model: str, item: dict[str, Any], rep: int) -> Callable[[], None]:
        def run() -> None:
            out = judges.run(model, item)
            log.append({"pass": "accuracy", "model": model, "input": item["id"], "rep": rep, **out})

        return run

    jobs = [
        job(model, item, rep)
        for rep in range(args.repeats)
        for item in items
        for model in args.models
        if (model, item["id"], rep) not in have
    ]
    print(f"{out_dir.name}: 입력 {len(items)}, 호출 {len(jobs)} (이미 한 것 {len(have)})", flush=True)
    _parallel(jobs, args.workers)


def cmd_latency(args: argparse.Namespace) -> None:
    out_dir, inputs, log = _prepare(args.tag)
    picked = random.Random(args.seed).sample(inputs, min(args.n, len(inputs)))
    judges = Judges()
    print(f"{out_dir.name}: 입력 {len(picked)} × 모델 {len(args.models)}, 차례로", flush=True)
    for i, item in enumerate(picked):
        order = args.models[i % len(args.models) :] + args.models[: i % len(args.models)]
        for model in order:
            out = judges.run(model, item)
            log.append({"pass": "latency", "model": model, "input": item["id"], "rep": i, **out})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(picked)}", flush=True)


def cmd_tokens(args: argparse.Namespace) -> None:
    out_dir, inputs, log = _prepare(args.tag)
    picked = random.Random(args.seed).sample(inputs, min(args.n, len(inputs)))
    judges = Judges()
    print(f"{out_dir.name}: 입력 {len(picked)} × 질문 수 1·2·4", flush=True)
    for item in picked:
        for count in (1, 2, 4):
            questions = [build_question("luna", f"state_{k}") for k in range(count)]
            out = judges.run("luna", item, questions)
            log.append({"pass": "tokens", "model": "luna", "input": item["id"], "questions": count, **out})


# ---------------------------------------------------------------------------
# 보고
# ---------------------------------------------------------------------------


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _cost_per_1k(model: str, rows: list[dict[str, Any]]) -> float | None:
    usable = [r for r in rows if r.get("input_tokens") is not None]
    if not usable:
        return None
    if model == "mini":
        total = sum(
            (r["input_tokens"] - (r.get("cached_tokens") or 0)) * PRICE_MINI["input"]
            + (r.get("cached_tokens") or 0) * PRICE_MINI["cached"]
            + (r.get("output_tokens") or 0) * PRICE_MINI["output"]
            for r in usable
        )
    else:
        total = sum(r["input_tokens"] * PRICE_DECISIONS_INPUT for r in usable)
    return total / 1e6 / len(usable) * 1000


def _dialog(item: dict[str, Any]) -> str:
    return " | ".join(f"{'User' if who == 'user' else 'Ray'}: {text}" for who, text in item["exchange"])


def cmd_report(args: argparse.Namespace) -> None:
    out_dir = find_result_dir(EXPERIMENT, args.tag)
    if out_dir is None:
        raise SystemExit(f"no result for tag {args.tag!r}")
    inputs = {i["id"]: i for i in json.loads((out_dir / "inputs.json").read_text(encoding="utf-8"))}
    calls = CallLog(out_dir / "calls.jsonl").read()
    acc = [r for r in calls if r["pass"] == "accuracy"]
    models = [m for m in MODELS if any(r["model"] == m for r in acc)]

    labels = Counter(i["label"] for i in inputs.values())
    complete = sum(i["complete"] for i in inputs.values())
    print(f"{out_dir.name}: 입력 {len(inputs)} (on {labels['on']}, paused {labels['paused']}, 완전한 입력 {complete})")

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in acc:
        by_model[r["model"]].append(r)

    print("\n## 정확도 (호출 단위 / 입력별 다수결)")
    votes: dict[str, dict[int, str]] = {}
    for model in models:
        rows = by_model[model]
        ok_rows = [r for r in rows if r.get("state") in ("on", "paused")]
        errors = sum("error" in r for r in rows)
        refusals = sum(bool(r.get("refusal")) for r in rows)

        def hit(r: dict[str, Any]) -> bool:
            return r.get("state") == inputs[r["input"]]["label"]

        per_input: dict[int, list[str]] = defaultdict(list)
        for r in ok_rows:
            per_input[r["input"]].append(r["state"])
        votes[model] = {k: Counter(v).most_common(1)[0][0] for k, v in per_input.items()}
        stable = sum(len(set(v)) == 1 for v in per_input.values())
        vote_ok = sum(votes[model][k] == inputs[k]["label"] for k in votes[model])
        complete_rows = [r for r in ok_rows if inputs[r["input"]]["complete"]]
        recall = {
            lab: sum(hit(r) for r in ok_rows if inputs[r["input"]]["label"] == lab)
            / max(1, sum(inputs[r["input"]]["label"] == lab for r in ok_rows))
            for lab in ("on", "paused")
        }
        print(
            f"  {model:9s} 호출 {len(rows)} (오류 {errors}, 거절 {refusals}) | 정확도 {sum(map(hit, ok_rows))}/"
            f"{len(ok_rows)} = {sum(map(hit, ok_rows)) / max(1, len(ok_rows)):.3f}, 완전한 입력 "
            f"{sum(map(hit, complete_rows)) / max(1, len(complete_rows)):.3f} | on 재현 {recall['on']:.3f}, "
            f"paused 재현 {recall['paused']:.3f} | 다수결 {vote_ok}/{len(votes[model])} | 반복 일치 "
            f"{stable}/{len(per_input)}"
        )

    print("\n## 발화별 다수결 정확도")
    utterances = sorted({i["utterance"] for i in inputs.values()}, key=lambda u: (UTTERANCES[u][2], u))
    print(f"  {'발화':14s} {'정답':6s} {'입력':>4s}  " + "  ".join(f"{m:>9s}" for m in models))
    for utterance in utterances:
        ids = [k for k, i in inputs.items() if i["utterance"] == utterance]
        label = inputs[ids[0]]["label"]
        cells = []
        for model in models:
            judged = [k for k in ids if k in votes[model]]
            cells.append(f"{sum(votes[model][k] == label for k in judged):>4d}/{len(judged):<4d}")
        print(f"  {utterance:14s} {label:6s} {len(ids):>4d}  " + "  ".join(f"{c:>9s}" for c in cells))

    for model in models:
        rows = [r for r in by_model[model] if r.get("p_paused") is not None]
        if not rows:
            continue
        print(f"\n## {model} 문턱별 (P(paused) ≥ 문턱이면 paused) — 호출 단위, 전체 / 완전한 입력")
        for t in THRESHOLDS:
            cells = []
            for subset in (rows, [r for r in rows if inputs[r["input"]]["complete"]]):
                false_pause = sum(r["p_paused"] >= t and inputs[r["input"]]["label"] == "on" for r in subset)
                miss = sum(r["p_paused"] < t and inputs[r["input"]]["label"] == "paused" for r in subset)
                accuracy = 1 - (false_pause + miss) / len(subset)
                cells.append(f"잘못 멈춤 {false_pause:4d}, 놓침 {miss:4d}, 정확도 {accuracy:.3f}")
            print(f"  {t:.1f}: " + " / ".join(cells))

    lat_rows = [r for r in calls if r["pass"] == "latency" and "error" not in r]
    source = "latency (차례로)" if lat_rows else "accuracy (병렬 — 부풀려짐)"
    lat_rows = lat_rows or [r for r in acc if "error" not in r]
    print(f"\n## 지연 ms — {source}, 이 PC 기준")
    for model in models:
        values = [r["latency_ms"] for r in lat_rows if r["model"] == model]
        if values:
            print(
                f"  {model:9s} n {len(values)} | p50 {statistics.median(values):.0f} | p90 {_pct(values, 0.9):.0f} | "
                f"p99 {_pct(values, 0.99):.0f} | 최대 {max(values):.0f}"
            )

    print("\n## 토큰·비용 (accuracy 호출 평균)")
    for model in models:
        rows = [r for r in by_model[model] if r.get("input_tokens") is not None]
        if not rows:
            continue
        cost = _cost_per_1k(model, rows)
        print(
            f"  {model:9s} 입력 {statistics.mean(r['input_tokens'] for r in rows):.0f} (캐시 "
            f"{statistics.mean(r.get('cached_tokens') or 0 for r in rows):.0f}), 출력 "
            f"{statistics.mean(r.get('output_tokens') or 0 for r in rows):.0f} | 판정 1,000회 ${cost:.4f}"
        )

    tok_rows = [r for r in calls if r["pass"] == "tokens" and r.get("input_tokens") is not None]
    if tok_rows:
        print("\n## 질문 수에 따른 Decisions 입력 토큰 (같은 입력 평균)")
        for count in (1, 2, 4):
            values = [r["input_tokens"] for r in tok_rows if r["questions"] == count]
            if values:
                print(f"  질문 {count}: {statistics.mean(values):.0f}")

    lines = []
    for k, item in sorted(inputs.items()):
        judged = {m: votes[m].get(k) for m in models}
        if all(v in (None, item["label"]) for v in judged.values()):
            continue
        p = {
            m: statistics.mean(r["p_paused"] for r in by_model[m] if r["input"] == k and r.get("p_paused") is not None)
            for m in models
            if any(r["input"] == k and r.get("p_paused") is not None for r in by_model[m])
        }
        reasons = [r.get("reason", "") for r in by_model["mini"] if r["input"] == k and r.get("reason")]
        lines.append(
            f"#{k} {item['utterance']} 정답 {item['label']} ({'완전' if item['complete'] else '이른'}, {item['by']}, "
            f"{item['source']}) — "
            + ", ".join(f"{m} {v}" for m, v in judged.items())
            + ("".join(f" | P({m}) {v:.2f}" for m, v in p.items()))
            + f"\n  대사: {item['line']}\n  {_dialog(item)}"
            + (f"\n  mini 이유: {reasons[0]}" if reasons else "")
        )
    (out_dir / "misses.txt").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n다수결이 정답과 다른 입력 {len(lines)}개 → {out_dir / 'misses.txt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("accuracy")
    p.add_argument("--tag", required=True)
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="앞에서 이만큼의 입력만 (시험용)")
    p = sub.add_parser("latency")
    p.add_argument("--tag", required=True)
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("tokens")
    p.add_argument("--tag", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("report")
    p.add_argument("tag")
    args = parser.parse_args()
    {"accuracy": cmd_accuracy, "latency": cmd_latency, "tokens": cmd_tokens, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    main()
