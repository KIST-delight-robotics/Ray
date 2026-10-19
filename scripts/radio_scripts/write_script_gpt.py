"""GPT 모델이 지시서로 라디오 대본 한 편을 쓴다 (Responses API + 웹 검색, 실시간 세션 없음).

대본 자동 생성은 운영과 같은 OpenAI 를 써야 해서 만든 비교용 도구. 지시서와 곡 목록을 넘기고, 답에서 대본과 근거를 꺼내
`var/radio_script_drafts/<키>_<suffix>.md` 로 저장한다. 형식 검사에 걸리면 고치게 한다. 응답 id, 추론·검색(검색어)·답의
순서는 `logs/<이름>.json` 에 남는다. 기록은 dj-mode-worknote.md 관찰 "대본 쓰기 — 모델과 지시서".

    uv run python scripts/radio_scripts/write_script_gpt.py --effort high --suffix gpt61
    uv run python scripts/radio_scripts/write_script_gpt.py --minimal --brief brief_minimal_scene.md --effort high
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from openai import OpenAI

from voice_pipeline.engines.gpt_live.radio import _parse_frontmatter, check_script

HERE = Path(__file__).resolve().parent
DRAFTS = HERE.parents[1] / "var" / "radio_script_drafts"
INSTRUCTIONS = """\
You write one Korean radio script for a small desk robot named Ray, following the brief below exactly. The brief is in
Korean. Use web search to verify every fact you use (years, people, anecdotes, and anything you say about the chosen
song, including who sings the listed version); leave out what you cannot verify. If dropping a fact breaks the story's
logic, take the story another way instead of compressing the gap. Read the script once as if hearing it aloud and fix
anything that sounds flat or list-like before answering.

Answer in exactly this shape and nothing else:
KEY: <short lowercase English snake_case name for the subject>
=== SCRIPT ===
<the script file: frontmatter and lines, exactly as the brief's format>
=== SOURCES ===
<the sources file in Korean: conclusion sentence on top, then the chosen song and why, then one row per line that has a
fact: line number, the fact, source URL; then facts you dropped because you could not verify them>
"""
# --minimal: 답의 모양만 정한다 (지시는 지시서 파일에만)
MINIMAL_INSTRUCTIONS = """\
Write the radio script the brief below asks for. The brief is in Korean.

Answer in exactly this shape and nothing else:
KEY: <short lowercase English snake_case name for the subject>
=== SCRIPT ===
<the script file>
=== SOURCES ===
<in Korean: for each line that has a fact, the line number and source URL>
"""


def trace(r) -> dict:
    """응답 하나의 출력 순서 — 추론·검색(검색어)·답이 일어난 차례."""
    steps = []
    for o in r.output:
        if o.type == "web_search_call":
            action = getattr(o, "action", None)
            a = action.model_dump(exclude_none=True) if action is not None else {}
            steps.append({"search": a.get("query") or a.get("queries") or a.get("url") or a})
        else:
            steps.append({"type": o.type})
    usage = r.usage.model_dump(exclude_none=True) if r.usage else {}
    return {"id": r.id, "model": r.model, "steps": steps, "usage": usage}


def parse(text: str) -> tuple[str, str, str]:
    m = re.search(r"KEY:\s*([a-z0-9_]+)\s*=== SCRIPT ===\s*(.*?)\s*=== SOURCES ===\s*(.*)", text, re.S)
    if m is None:
        raise RuntimeError("answer is not in the expected shape")
    return m.group(1), m.group(2).strip() + "\n", m.group(3).strip() + "\n"


def problems(script: str, minimal: bool = False) -> list[str]:
    _, body = _parse_frontmatter(script)
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    found = check_script(lines)
    if minimal:  # 추적에 꼭 필요한 형식만
        return found
    if not 14 <= len(lines) <= 20:
        found.append(f"{len(lines)} lines (must be 14-20)")
    if re.search(r"(대|래)요[.?!]?(\s|$)", body):
        found.append("contracted hearsay ending ~대요/~래요")
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--suffix", default="gpt61")
    parser.add_argument("--subject", default=None, help="소재를 이것으로 고정한다 (지시서의 '이번에 쓸 것'보다 우선)")
    parser.add_argument("--research", action="store_true", help="쓰기 전에 여러 출처로 충분히 조사하라는 지시를 더한다")
    parser.add_argument("--brief", default="brief.md", help="지시서 파일 (이 폴더 기준)")
    parser.add_argument("--minimal", action="store_true", help="instructions 는 답 모양만, 형식 검사는 check_script 만")
    args = parser.parse_args()
    brief = (HERE / args.brief).read_text(encoding="utf-8")
    songs = (HERE / "songs.md").read_text(encoding="utf-8")
    task = f"{brief}\n\n---\n\n아래는 지시서가 말하는 곡 목록 파일(songs.md)의 내용이다.\n\n{songs}"
    if args.subject:
        task += f"\n\n---\n\n이번 소재는 정해져 있다: {args.subject}. 지시서의 소재 고르기보다 이것이 우선한다."
    instructions = MINIMAL_INSTRUCTIONS if args.minimal else INSTRUCTIONS
    if args.research:
        instructions += (
            "\nBefore writing, research the subject widely: search several different sources and gather enough "
            "anecdotes, scenes, quotes and facts to choose from. A story needs material; do not write from a single "
            "source.\n"
        )
    client = OpenAI()
    start = time.monotonic()
    r = client.responses.create(
        model=args.model,
        instructions=instructions,
        input=task,
        tools=[{"type": "web_search"}],
        reasoning={"effort": args.effort},
    )
    searches = sum(1 for o in r.output if o.type == "web_search_call")
    log = {"args": vars(args), "instructions": instructions, "task": task, "responses": [trace(r)], "rounds": []}
    key, script, sources = parse(r.output_text)
    for round_ in range(3):
        found = problems(script, args.minimal)
        print(f"round {round_}: {key} {len(found)} problems {found}")
        log["rounds"].append(found)
        if not found:
            break
        r = client.responses.create(
            model=args.model,
            previous_response_id=r.id,
            instructions=instructions,  # 앞 응답의 instructions 는 이어지지 않는다
            input="The format check found these problems. Fix them and answer again in the same shape:\n"
            + "\n".join(f"- {p}" for p in found),
            tools=[{"type": "web_search"}],
            reasoning={"effort": args.effort},
        )
        searches += sum(1 for o in r.output if o.type == "web_search_call")
        log["responses"].append(trace(r))
        key, script, sources = parse(r.output_text)
    name = f"{key}_{args.suffix}"
    DRAFTS.mkdir(parents=True, exist_ok=True)
    (DRAFTS / f"{name}.md").write_text(script, encoding="utf-8")
    (DRAFTS / f"{name}.sources.md").write_text(sources, encoding="utf-8")
    log["seconds"] = round(time.monotonic() - start)
    (DRAFTS / "logs").mkdir(exist_ok=True)
    (DRAFTS / "logs" / f"{name}.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {name}.md, .sources.md, logs/{name}.json — {log['seconds']} s, searches {searches}, model {r.model}")
    for i, resp in enumerate(log["responses"]):
        print(
            f"  response {i} {resp['id']}: "
            + " → ".join(f"search {st['search']!r}" if "search" in st else st["type"] for st in resp["steps"])
        )


if __name__ == "__main__":
    main()
