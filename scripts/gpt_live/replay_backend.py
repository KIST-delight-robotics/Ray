"""실기 백엔드 응답을 같은 맥락에서 다시 돌린다 — 지시문을 바꿔 툴 선택을 비교 (실시간 세션 없음).

운영 로그(``Backend response … resp_…``)의 응답 id 를 주면, 그 응답의 바로 앞 응답에 ``previous_response_id`` 를 걸고
그 응답이 받은 새 입력 항목(SRT 조각·앱 instructions·툴 결과)을 그대로 보내 n 회 다시 부른다. 앞 맥락(이전 위임의
항목·추론·웹 검색 결과)은 서버에 저장된 그대로라 ``bench_backend_tools.py`` 의 흉내보다 정확하다. 모델·툴·라이브
지시문(장기기억 포함)은 그 응답의 것을 쓰고, 백엔드 지시문만 ``--backend current`` 로 지금 코드의 것으로 바꿀 수 있다
(라디오 상태는 원래 지시문에서 읽는다). 저장된 응답은 OpenAI 보관 기간(30일) 안에서만 꺼낼 수 있다.

    uv run python scripts/gpt_live/replay_backend.py --list var/log/pipeline/20261014_172714.log
    uv run python scripts/gpt_live/replay_backend.py resp_… [resp_…] --n 5 --backend current
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from openai import OpenAI

from voice_pipeline.engines.gpt_live import songs
from voice_pipeline.engines.gpt_live.instructions import SONG_BACKEND_INSTRUCTIONS, build_backend_instructions

_BACKEND_OPEN = "<backend_delegation_instructions>\n"
_BACKEND_CLOSE = "</backend_delegation_instructions>"
_RESPONSE_RE = re.compile(r"Backend response .*calls=(.*)\) (resp_[0-9a-f]+)")
_USER_RE = re.compile(r"live_session\s+INFO\s+user: (.*)")


def list_log(path: Path) -> None:
    """로그의 백엔드 응답 id 를 그 앞 사용자 전사와 함께 보인다."""
    user = ""
    for text in path.read_text(encoding="utf-8").splitlines():
        if m := _USER_RE.search(text):
            user = m.group(1)
        elif m := _RESPONSE_RE.search(text):
            print(f"{text[11:19]} {m.group(2)} {m.group(1):22s} ← {user[:50]}")


def current_backend(original: str) -> str:
    """원래 백엔드 지시문 자리에 넣을 지금 코드의 지시문 (곡 중 정지 분류기면 그것)."""
    if original.lstrip().startswith(SONG_BACKEND_INSTRUCTIONS.split("\n", 1)[0]):
        return SONG_BACKEND_INSTRUCTIONS
    status = m.group(1) if (m := re.search(r"Radio talk status: (\w+)", original)) else None
    return build_backend_instructions(songs.load_song_catalog(), status)


def swap_backend(instructions: str) -> tuple[str, bool]:
    """(백엔드 지시문을 지금 코드의 것으로 바꾼 지시문, 원래와 같았는지)."""
    head, sep, rest = instructions.partition(_BACKEND_OPEN)
    original, sep2, tail = rest.partition(_BACKEND_CLOSE)
    if not sep or not sep2:
        raise RuntimeError("backend instructions section not found")
    new = current_backend(original).strip()
    return f"{head}{_BACKEND_OPEN}{new}\n\n{_BACKEND_CLOSE}{tail}", new == original.strip()


def as_input(item: dict[str, Any]) -> dict[str, Any]:
    """조회한 입력 항목 → 요청에 다시 넣을 항목."""
    kind = item["type"]
    if kind == "message":
        part = "output_text" if item["role"] == "assistant" else "input_text"
        content = [{"type": part, "text": c.get("text", "")} for c in item["content"]]
        return {"type": "message", "role": item["role"], "content": content}
    if kind == "function_call":
        return {k: item[k] for k in ("type", "call_id", "name", "arguments")}
    if kind == "function_call_output":
        return {k: item[k] for k in ("type", "call_id", "output")}
    raise RuntimeError(f"cannot resend input item of type {kind!r}")


def input_items(client: OpenAI, response_id: str) -> list[dict[str, Any]]:
    """응답이 받은 입력 항목 전부, 오래된 것부터 — SDK 자동 페이지 넘김은 개수가 틀려 직접 넘긴다."""
    items: list[dict[str, Any]] = []
    after: dict[str, str] = {}
    while True:
        page = client.responses.input_items.list(response_id, order="asc", limit=100, **after)
        items += [it.model_dump(exclude_none=True) for it in page.data]
        if not page.has_more or not page.data:
            return items
        after = {"after": page.data[-1].id}


def outcome(r: Any) -> str:
    """응답 하나의 결과 한 줄: 툴 호출 / 웹 검색 / 글 답 / 빈 답."""
    calls = [o for o in r.output if o.type == "function_call"]
    calls = [c.name + (f"({c.arguments})" if c.arguments not in ("", "{}") else "") for c in calls]
    if calls:
        return " + ".join(calls)
    if any(o.type == "web_search_call" for o in r.output):
        return "web_search"
    text = (r.output_text or "").strip().replace("\n", " ")
    return f"text: {text[:70]}" if text else "empty"


def replay(client: OpenAI, response_id: str, n: int, backend: str) -> None:
    r = client.responses.retrieve(response_id)
    if r.previous_response_id is None:
        raise RuntimeError(f"{response_id}: no previous response to replay from")
    prev = client.responses.retrieve(r.previous_response_id)
    seen = len(input_items(client, prev.id)) + len(prev.output)
    items = input_items(client, response_id)
    if prev.output and items[seen - 1].get("id") != prev.output[-1].id:
        raise RuntimeError(f"{response_id}: input items do not continue the previous response")
    new = [as_input(it) for it in items[seen:]]
    if not new:
        raise RuntimeError(f"{response_id}: no new input items after the previous response")
    instructions = r.instructions
    note = ""
    if backend == "current":
        instructions, same = swap_backend(instructions)
        note = " (지금 코드 = 원래)" if same else " (지금 코드 ≠ 원래)"
    last = new[-1]["content"][0]["text"] if new and new[-1]["type"] == "message" else json.dumps(new[-1:])
    print(f"== {response_id}  새 항목 {len(new)}  백엔드 지시문 {backend}{note}")
    print(f"   마지막 입력 끝: …{last[-120:]!r}")
    print(f"   원래: {outcome(r)}  (입력 {r.usage.input_tokens})")

    def one(_: int) -> str:
        out = client.responses.create(
            model=r.model,
            previous_response_id=prev.id,
            instructions=instructions,
            tools=[t.model_dump(exclude_none=True) for t in r.tools],
            reasoning=r.reasoning.model_dump(exclude_none=True),
            input=new,
            store=False,
        )
        return f"{outcome(out)}  (입력 {out.usage.input_tokens})"

    with ThreadPoolExecutor(min(n, 8)) as ex:
        results = list(ex.map(one, range(n)))
    for result, count in Counter(results).most_common():
        print(f"   {count}× {result}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("response_ids", nargs="*")
    parser.add_argument("--list", type=Path, metavar="LOG", help="로그의 백엔드 응답 id 를 보이고 끝낸다")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--backend", choices=("original", "current"), default="original")
    args = parser.parse_args()
    if args.list:
        list_log(args.list)
        return
    client = OpenAI()
    for response_id in args.response_ids:
        replay(client, response_id, args.n, args.backend)


if __name__ == "__main__":
    main()
