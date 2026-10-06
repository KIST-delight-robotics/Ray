"""responses 위임 백엔드의 툴 선택 벤치 — 실제 백엔드 입력 구조를 흉내 낸 Responses 호출 (실시간 세션 없음).

백엔드는 라이브 세션 지시문을 developer 메시지 안에 함께 받고 전사는 SRT 형식 user 메시지 하나로 받는다
(FINDINGS §4). 그 구조로 케이스마다 n 회 호출해 첫 툴 선택을 센다. 라이브 지시문을 바꾼 뒤 백엔드 회귀 확인용.

    uv run python scripts/gpt_live/bench_backend_tools.py              # 현재 코드의 지시문
    uv run python scripts/gpt_live/bench_backend_tools.py --live f.txt # 라이브 지시문을 파일 내용으로 바꿔서
    uv run python scripts/gpt_live/bench_backend_tools.py --n 30 --case 날씨
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from voice_pipeline.engines.gpt_live.dj import load_segments
from voice_pipeline.engines.gpt_live.instructions import build_backend_instructions, build_live_instructions
from voice_pipeline.engines.gpt_live.songs import load_song_catalog
from voice_pipeline.engines.gpt_live.tools import DEFAULT_TOOLS, DJ_TOOLS, LOOP_TOOLS, make_play_song_tool_def
from voice_pipeline.settings import DJ_SEGMENTS_DIR
from voice_pipeline.wiring import _LIVE_BACKEND_MODEL, _LIVE_BACKEND_REASONING  # 실작동과 같은 설정

SRT_HEAD = (
    "The conversation below is an SRT-style transcript of the recent live spoken conversation. The blocks are ordered "
    "and role-labeled; use it to answer the latest user turn for the downstream voice assistant.\n\n"
    "[Spoken conversation transcript]\n\n"
)
_HANDLED = {"search_memory", "adjust_volume", "set_brightness", "get_device_settings"}  # wiring 이 핸들러를 붙이는 툴

DENTAL = [
    ("user", "이빨이 좀 아픈 거 같더라고"),
    ("assistant", "어유. 이가 시큰거려요? 어디가 어떻게 아픈지 말해줄래요?"),
    ("user", "근처에 치과 어디 있더라"),
    ("assistant", "성북구 월곡역 기준으로 여러 치과가 떠요. 가까운 곳부터 정리해드릴까요?"),
]
DJ_TALK = [
    ("user", "DJ 모드 틀어줘"),
    ("assistant", "네, 시작할게요."),
    (
        "assistant",
        "오늘은 서랍 이야기를 해볼게요. 서랍은 집에서 가장 느리게 바뀌는 자리예요. 책상 위는 매일 달라지는데, "
        "서랍 안은 몇 년째 그대로죠. 어쩌면 서랍은 우리 대신 기억을 맡아 두는 자리인지도 모르겠어요. "
        "그런 마음으로 고른 곡, 십센치의 '서랍' 들려드릴게요.",
    ),
]
# 이름: (전사, DJ 중인가, 기대 툴). web_search 는 내장 툴 호출, text 는 툴 없이 답한 경우.
CASES = {
    "첫 턴 DJ": ([("user", "DJ 모드 틀어줘")], False, "start_dj"),
    "작별": ([("user", "오늘은 여기까지 할게, 잘 가")], False, "end_conversation"),
    "볼륨": ([("user", "볼륨 좀 올려줘")], False, "adjust_volume"),
    "날씨": ([("user", "오늘 서울 날씨 알려줘")], False, "web_search"),
    "곡 지정": ([("user", "아이브 아이엠 틀어줘")], False, "play_song"),
    "3턴 '아니 됐어' 뒤 DJ": (
        DENTAL + [("user", "아니 됐어"), ("assistant", "네, 알겠어요."), ("user", "DJ 모드 틀어줘")],
        False,
        "start_dj",
    ),
    "3턴 '고마워' 뒤 DJ": (
        DENTAL + [("user", "고마워"), ("assistant", "천만에요. 또 필요하면 말해줘요."), ("user", "DJ 모드 틀어줘")],
        False,
        "start_dj",
    ),
    "DJ 곡 소개 뒤 위임": (DJ_TALK, True, "play_dj_song"),
    "DJ 중 그만": (DJ_TALK[:2] + [("assistant", DJ_TALK[2][1][:60]), ("user", "그만할래")], True, "stop_dj"),
}


def developer_message(live: str, backend: str) -> str:
    return (
        "You are a backend model supporting a realtime frontend assistant during a live spoken conversation. "
        "Shared session instructions\n"
        + live
        + "\n<backend_delegation_instructions>\n"
        + backend
        + "\n</backend_delegation_instructions>"
    )


def srt(blocks: list[tuple[str, str]]) -> str:
    body = SRT_HEAD
    for i, (role, text) in enumerate(blocks, 1):
        body += f"{i}\n[{'User' if role == 'user' else 'Assistant'} speech transcript]\n{text}\n\n"
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--live", type=Path, help="라이브 지시문을 이 파일 내용으로 바꾼다")
    parser.add_argument("--case", action="append", help="이 케이스만 (여러 번 지정 가능)")
    args = parser.parse_args()

    catalog = load_song_catalog()
    segment = load_segments(DJ_SEGMENTS_DIR, catalog)["drawer"]  # DJ_TALK 전사가 이 소재 기준
    tools = [
        {"type": "web_search"},
        *[t for t in DEFAULT_TOOLS if t["name"] in LOOP_TOOLS or t["name"] in _HANDLED],
        make_play_song_tool_def(catalog),
        *DJ_TOOLS,
    ]
    live = args.live.read_text() if args.live else build_live_instructions()
    backend = {False: build_backend_instructions(catalog), True: build_backend_instructions(catalog, segment)}
    client = OpenAI()

    def one(blocks: list[tuple[str, str]], in_dj: bool) -> str:
        messages = [
            {"role": "developer", "content": developer_message(live, backend[in_dj])},
            {"role": "user", "content": srt(blocks)},
        ]
        for attempt in range(3):
            try:
                r = client.responses.create(
                    model=_LIVE_BACKEND_MODEL,
                    tools=tools,
                    reasoning={"effort": _LIVE_BACKEND_REASONING},
                    input=messages,
                )
                break
            except Exception as exc:
                if attempt == 2:
                    return f"ERR:{type(exc).__name__}"
                time.sleep(1.5 * (attempt + 1))
        calls = [o.name for o in r.output if o.type == "function_call"]
        if calls:
            return calls[0]
        return "web_search" if any(o.type == "web_search_call" for o in r.output) else "text"

    t0 = time.monotonic()
    total = ok = 0
    with ThreadPoolExecutor(16) as ex:
        for name, (blocks, in_dj, expect) in CASES.items():
            if args.case and name not in args.case:
                continue
            counts = Counter(ex.map(lambda _, b=blocks, d=in_dj: one(b, d), range(args.n)))
            miss = ", ".join(f"{k} {v}" for k, v in counts.items() if k != expect)
            total += args.n
            ok += counts[expect]
            print(f"  {name:22s} {counts[expect]:3d}/{args.n}" + (f"  ({miss})" if miss else ""), flush=True)
    print(f"합계 {ok}/{total}  ({time.monotonic() - t0:.0f}s)")


if __name__ == "__main__":
    main()
