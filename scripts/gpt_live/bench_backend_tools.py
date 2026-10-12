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

from voice_pipeline.engines.gpt_live import songs
from voice_pipeline.engines.gpt_live.instructions import build_backend_instructions, build_live_instructions
from voice_pipeline.engines.gpt_live.tools import (
    DEFAULT_TOOLS,
    LOOP_TOOLS,
    RADIO_TOOLS,
    make_play_song_tool_def,
)
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
# 라디오를 시작해 읽다가 사용자가 끼어들어 멈춘(paused) 뒤까지의 전사
RADIO_PAUSED = [
    ("user", "라디오 해 줘"),
    ("assistant", "네, 라디오 시작할게요."),
    (
        "assistant",
        "오늘은 책상 서랍 이야기예요. 늘 곁에 있는데도 좀처럼 열어 보지 않는 곳이죠. 서랍은 집에서 가장 느리게 "
        "바뀌는 자리예요.",
    ),
    ("user", "근데 너는 서랍에 뭐 넣어 둬?"),
    ("assistant", "저는 서랍이 없지만, 있다면 작은 메모지를 넣어 둘 것 같아요."),
]
# 이름: (전사, 라디오 상태, 기대). web_search 는 내장 툴 호출, empty/text 는 툴 없이 빈 답/글 답.
CASES = {
    "첫 턴 라디오": ([("user", "라디오 해 줘")], "off", "start_radio"),
    "디제이": ([("user", "디제이 모드 틀어줘")], "off", "start_radio"),
    "작별": ([("user", "오늘은 여기까지 할게, 잘 가")], "off", "end_conversation"),
    "볼륨": ([("user", "볼륨 좀 올려줘")], "off", "adjust_volume"),
    "날씨": ([("user", "오늘 서울 날씨 알려줘")], "off", "web_search"),
    "곡 지정": ([("user", "아이브 아이엠 틀어줘")], "off", "play_song"),
    "3턴 '아니 됐어' 뒤 라디오": (
        DENTAL + [("user", "아니 됐어"), ("assistant", "네, 알겠어요."), ("user", "라디오 해 줘")],
        "off",
        "start_radio",
    ),
    "3턴 '고마워' 뒤 라디오": (
        DENTAL + [("user", "고마워"), ("assistant", "천만에요. 또 필요하면 말해줘요."), ("user", "라디오 해 줘")],
        "off",
        "start_radio",
    ),
    "멈춘 뒤 이야기 계속해 줘": (RADIO_PAUSED + [("user", "이야기 계속해 줘")], "paused", "resume_radio"),
    "멈춘 뒤 라디오 이어서": (RADIO_PAUSED + [("user", "라디오 다시 이어서 해 줘")], "paused", "resume_radio"),
    "멈춘 뒤 라디오 해 줘": (RADIO_PAUSED + [("user", "라디오 해 줘")], "paused", "start_radio"),
    "멈춘 뒤 대화": (RADIO_PAUSED + [("user", "메모지에는 뭘 적을 건데?")], "paused", "text"),
}
TEXT_SAMPLES: dict[str, list[str]] = {}  # 툴 없이 글로 답한 경우의 예 (케이스별)


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

    songs._SONG_FILES = tuple(f for f in songs._SONG_FILES if f[0] != "audio/music")  # 음원 없이 목록만
    catalog = songs.load_song_catalog()
    tools = [
        {"type": "web_search"},
        *[t for t in DEFAULT_TOOLS if t["name"] in LOOP_TOOLS or t["name"] in _HANDLED],
        make_play_song_tool_def(catalog),
        *RADIO_TOOLS,
    ]
    live = args.live.read_text() if args.live else build_live_instructions()
    client = OpenAI()

    def one(name: str, blocks: list[tuple[str, str]], status: str) -> str:
        backend = build_backend_instructions(catalog, status)
        messages = [
            {"role": "developer", "content": developer_message(live, backend)},
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
        if any(o.type == "web_search_call" for o in r.output):
            return "web_search"
        text = (r.output_text or "").strip()
        if text:
            TEXT_SAMPLES.setdefault(name, []).append(text)
        return "text" if text else "empty"

    t0 = time.monotonic()
    total = ok = 0
    with ThreadPoolExecutor(16) as ex:
        for name, (blocks, status, expect) in CASES.items():
            if args.case and name not in args.case:
                continue
            run = lambda _, n=name, b=blocks, st=status: one(n, b, st)  # noqa: E731
            counts = Counter(ex.map(run, range(args.n)))
            miss = ", ".join(f"{k} {v}" for k, v in counts.items() if k != expect)
            total += args.n
            ok += counts[expect]
            print(f"  {name:22s} {counts[expect]:3d}/{args.n}" + (f"  ({miss})" if miss else ""), flush=True)
    print(f"합계 {ok}/{total}  ({time.monotonic() - t0:.0f}s)")
    for name, texts in TEXT_SAMPLES.items():
        print(f"  글 답 예 [{name}]: " + " | ".join(t[:60] for t in texts[:2]))


if __name__ == "__main__":
    main()
