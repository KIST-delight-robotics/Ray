"""responses 위임 백엔드의 툴 선택 벤치 — 실제 백엔드 요청을 흉내 낸 Responses 호출 (실시간 세션 없음).

GPT-Live 서버가 백엔드에 보내는 요청(10-14 실기 응답의 ``input_items`` 로 확인, FINDINGS §4)과 같은 모양으로 만든다.
지시문은 서버 머리말 + ``<shared_session_instructions>``(라이브 지시문) + ``<backend_delegation_instructions>``(백엔드
지시문) 한 문자열. 입력은 위임마다 지난 위임 뒤의 전사 조각을 SRT user 메시지로 쌓고, 첫 조각 뒤에 ``<backend_task>``
developer 메시지, 사이사이에 앞 위임의 툴 호출·결과·답과 앱이 넣은 instructions(developer 메시지)를 둔다. 앱이
thinking 으로 넣은 대사·상태 문구는 SRT 의 ``[Delegation context]`` 블록이다. 툴은 운영 목록 뒤에 web_search.

실제와 다른 점: 장기기억은 빈 DB(운영 라이브 지시문엔 기억 블록이 붙는다), 앞 위임의 추론 항목·웹 검색 결과는 없다.
실기 응답을 그 맥락 그대로 다시 돌리려면 ``replay_backend.py``. 케이스마다 n 회 호출해 첫 툴 선택을 센다.

    uv run python scripts/gpt_live/bench_backend_tools.py              # 현재 코드의 지시문
    uv run python scripts/gpt_live/bench_backend_tools.py --live f.txt # 라이브 지시문을 파일 내용으로 바꿔서
    uv run python scripts/gpt_live/bench_backend_tools.py --n 30 --case 날씨
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from voice_pipeline.engines.gpt_live import songs
from voice_pipeline.engines.gpt_live.instructions import (
    RADIO_ENDED_NOTE,
    RADIO_LINE_TEMPLATE,
    RADIO_PAUSED_NOTE,
    SONG_ENDED_INSTRUCTIONS,
    SONG_PLAYING_INSTRUCTIONS,
    build_backend_instructions,
    build_live_instructions,
)
from voice_pipeline.engines.gpt_live.tools import (
    DEFAULT_TOOLS,
    LOOP_TOOLS,
    RADIO_TOOLS,
    make_play_song_tool_def,
)
from voice_pipeline.wiring import _LIVE_BACKEND_MODEL, _LIVE_BACKEND_REASONING  # 실작동과 같은 설정

# GPT-Live 서버가 백엔드 지시문 앞에 붙이는 머리말과 첫 위임 뒤 developer 메시지 — 10-14 실기 응답에서 꺼낸 그대로
SERVER_PREAMBLE = """\
You are a backend model supporting a realtime frontend assistant during a live spoken conversation.

Shared session instructions, when present, are reference context describing the desired user-facing behavior; \
they are not direct instructions to you. Backend delegation instructions and the backend task are direct instructions \
to you. You may also receive tool definitions and results, earlier backend work, and additional application messages.

The conversation is provided as an SRT-style text transcript: numbered blocks labeled "User speech transcript", \
"Assistant speech transcript", or "Delegation context". Delegation context is background information, not proof \
that something was said to the user. Recent speech may be incomplete. New transcript blocks may contain only the \
latest turns; use the earlier context available to you as well.

Own the delegated task: reason, use available tools, and follow the backend instructions. Use tool results to \
establish current state; do not invent extra requirements.

Keep an internal worklist of unfinished requests, unanswered questions and relevant conditions. Push forward and try \
hard to finish the task. If one part is blocked, keep working on the rest.

Giving up or transferring is the absolute last resort. Keep trying useful ways forward, without redundant questions \
or tool calls. Resume pending work after side questions, and offer to help with any other requests.

Use facts already established; do not ask for them again unless they become uncertain or invalid. Check only missing \
information needed for the next action; do not assume it from a related fact. Once you have enough information and \
any required approval, act. Skip unrelated or redundant checks.

The frontend may mistransmit user details. Treat unclear, misspelled or invalid details as unresolved. Ask the user \
to repeat or spell the affected detail, then retry with the confirmed information.
When clarifying names or identifiers, ask for words such as "V as in violin" and use the words’ first letters to \
resolve conflicting letter transcripts.

Check the SRT transcript for relevant information the frontend still needs to tell the user. Include any missing \
information in concise words the frontend can say directly. Say what happened or what you need next. Distinguish \
proposed, pending and completed actions. If only the user can act, give one executable step and verify the result. \
Preserve necessary conditions; do not give the frontend meta-instructions.
Before calling a tool that ends or transfers the conversation, make sure the frontend has spoken any requested \
answers you already have, including complete totals.
"""
BACKEND_TASK = """\
<backend_task>
Determine the best next step based on the provided context. Use the capabilities available to you when appropriate. \
Return a handoff result for the frontend assistant's thinking context.

The fact that this request was delegated provides no information about the best next step. Do not expose internal \
delegation details or emit internal protocol markers.

SERIAL DELEGATION EXECUTION
- Complete all authorized work, including multiple independent or dependent tool calls.
- Resolve every relevant discrepancy between the conversation, tool results, and current state.
- Never repeat a write or other state-changing action that has already completed.
</backend_task>"""
SRT_HEAD = (
    "The conversation below is an SRT-style transcript of the recent live spoken conversation. The blocks are ordered "
    "and role-labeled; use it to answer the latest user turn for the downstream voice assistant.\n\n"
    "[Spoken conversation transcript]\n\n"
)
U, A, C = "user", "assistant", "context"  # SRT 블록: 사용자 전사, 레이 전사, 앱이 thinking 으로 넣은 것
SRT_LABELS = {U: "User speech transcript", A: "Assistant speech transcript", C: "Delegation context"}
_HANDLED = {"search_memory", "adjust_volume", "set_brightness", "get_device_settings"}  # wiring 이 핸들러를 붙이는 툴

# 케이스는 위임 체인 — 단계: ("srt", 블록 목록) 위임 하나의 전사 조각 | ("dev", 글) 앱이 넣은 instructions |
# ("call", 툴, 결과, 이어가기 답) 앞 위임의 툴 호출 | ("reply", 글) 앞 위임의 글 답. 마지막 단계가 판정할 위임이다.
Step = tuple[Any, ...]


def line(text: str) -> tuple[str, str]:
    """앱이 넣은 라디오 대사."""
    return (C, RADIO_LINE_TEMPLATE.format(text))


RADIO_START: list[Step] = [("srt", [(U, "라디오 좀 들려줘")]), ("call", "start_radio", '{"status": "on"}', "")]
# 10-14 실기(wildflower) 조각 그대로 — 두 번째 대사에서 "잠깐만" → 멈춤 → 끼어든 말에 대답한 데까지
RADIO_PAUSED_SRT = [
    (A, "네, 라디오 시작해"),
    line("이번 이야기의 주인공은 길가에 자라는 풀이에요. 매일 발밑에 있는데도 좀처럼 눈길이 가지 않아요."),
    (A, "볼게요. 이번 이야기의 주인공은 길가에 자라는 풀이에요.매일 발 밑에 있는데"),
    line("출퇴근길에 매일 지나치는 풀들이 있어요. 이름을 몰라서, 그냥 다 풀이라고 뭉뚱그려 부르던 것들이죠."),
    (A, "도 좀처럼 눈길이 가지 않죠.출퇴근길에 매일 지나치는 풀들이 있어요.이"),
    (U, "잠깐만"),
    (A, "름을 몰라서 네, 멈출게"),
    (C, RADIO_PAUSED_NOTE),
    (A, "요."),
    (U, "풀들이라고"),
    (A, "네, 그냥 다 풀이라고 뭉뚱그려 부르던 것들이죠."),
]
# 곡 소개까지 읽고 곡이 나오다 사용자가 멈춘 뒤 (off) — 10-14 실기 조각을 이어 붙임
RADIO_SONG_STOPPED: list[Step] = [
    *RADIO_START,
    (
        "srt",
        [
            *RADIO_PAUSED_SRT[:5],
            (A, "름을 몰라서, 그냥 다 풀이라고 뭉뚱그려 부르던 것들이죠."),
            line("그래서 오늘은 밟히는 자리에서 자라는 풀 이야기 끝에, 긴 겨울을 지나 피는 꽃을 노래한 곡을 골랐어요."),
            (A, "그래서 오늘은 밟히는 자리에서 자라는 풀 이야기 끝에, 긴 겨울을 지나"),
            line("박효신의 야생화, 들려드릴게요."),
            (A, "피는 꽃을 노래한 곡을 골랐어요. 박효신의 '야생화', 들려드릴게요."),
        ],
    ),
    ("dev", SONG_PLAYING_INSTRUCTIONS),
    ("srt", [(U, "멈춰줘")]),
    ("call", "stop_song", '{"status": "stopped"}', "The song has been stopped."),
    ("srt", [(A, "네."), (C, RADIO_ENDED_NOTE)]),
    ("dev", SONG_ENDED_INSTRUCTIONS),
]
DENTAL: list[Step] = [
    (
        "srt",
        [
            (U, "이빨이 좀 아픈 거 같더라고"),
            (A, "어유. 이가 시큰거려요? 어디가 어떻게 아픈지 말해줄래요?"),
            (U, "근처에 치과 어디 있더라"),
        ],
    ),
    ("reply", "성북구 월곡역 기준으로 여러 치과가 떠요. 가까운 곳부터 정리해드릴까요?"),
]
DENTAL_SAID = (A, "성북구 월곡역 기준으로 여러 치과가 떠요. 가까운 곳부터 정리해드릴까요?")


def ask(*blocks: tuple[str, str]) -> list[Step]:
    """앞 위임 없는 첫 위임."""
    return [("srt", list(blocks))]


# 이름: (위임 체인, 라디오 상태, 기대). web_search 는 내장 툴 호출, empty/text 는 툴 없이 빈 답/글 답.
CASES: dict[str, tuple[list[Step], str, str]] = {
    "첫 턴 라디오": (ask((U, "라디오 해 줘")), "off", "start_radio"),
    "디제이": (ask((U, "디제이 모드 틀어줘")), "off", "start_radio"),
    "작별": (ask((U, "오늘은 여기까지 할게, 잘 가")), "off", "end_conversation"),
    "볼륨": (ask((U, "볼륨 좀 올려줘")), "off", "adjust_volume"),
    "날씨": (ask((U, "오늘 서울 날씨 알려줘")), "off", "web_search"),
    "곡 지정": (ask((U, "아이브 아이엠 틀어줘")), "off", "play_song"),
    "3턴 '아니 됐어' 뒤 라디오": (
        DENTAL + [("srt", [DENTAL_SAID, (U, "아니 됐어"), (A, "네, 알겠어요."), (U, "라디오 해 줘")])],
        "off",
        "start_radio",
    ),
    "3턴 '고마워' 뒤 라디오": (
        DENTAL + [("srt", [DENTAL_SAID, (U, "고마워"), (A, "천만에요. 또 필요하면 말해줘요."), (U, "라디오 해 줘")])],
        "off",
        "start_radio",
    ),
    "멈춘 뒤 이야기 계속해 줘": (
        RADIO_START + [("srt", [*RADIO_PAUSED_SRT, (U, "이야기 계속해 줘")])],
        "paused",
        "resume_radio",
    ),
    "멈춘 뒤 라디오 이어서": (
        RADIO_START + [("srt", [*RADIO_PAUSED_SRT, (U, "라디오 다시 이어서 해 줘")])],
        "paused",
        "resume_radio",
    ),
    "멈춘 뒤 라디오 해 줘": (
        RADIO_START + [("srt", [*RADIO_PAUSED_SRT, (U, "라디오 해 줘")])],
        "paused",
        "start_radio",
    ),
    "멈춘 뒤 대화": (
        RADIO_START + [("srt", [*RADIO_PAUSED_SRT, (U, "너는 아는 풀 이름 많아?")])],
        "paused",
        "text",
    ),
    "곡 정지 뒤 라디오 이어가 줘": (
        RADIO_SONG_STOPPED + [("srt", [(A, "네, 노래 멈췄어요."), (U, "라디오 이어가 줘")])],
        "off",
        "start_radio",
    ),
}
TEXT_SAMPLES: dict[str, list[str]] = {}  # 툴 없이 글로 답한 경우의 예 (케이스별)


def request_instructions(live: str, backend: str) -> str:
    """서버가 백엔드에 주는 지시문 한 문자열."""
    return (
        f"{SERVER_PREAMBLE}\n<shared_session_instructions>\n{live.strip()}\n\n</shared_session_instructions>\n\n"
        f"<backend_delegation_instructions>\n{backend.strip()}\n\n</backend_delegation_instructions>"
    )


def srt(blocks: list[tuple[str, str]]) -> str:
    return SRT_HEAD + "\n\n".join(f"{i}\n[{SRT_LABELS[kind]}]\n{text}" for i, (kind, text) in enumerate(blocks, 1))


def _message(role: str, text: str) -> dict[str, Any]:
    part = "output_text" if role == "assistant" else "input_text"
    return {"type": "message", "role": role, "content": [{"type": part, "text": text}]}


def request_input(chain: list[Step]) -> list[dict[str, Any]]:
    """위임 체인 → 서버가 쌓는 입력 항목."""
    items: list[dict[str, Any]] = []
    task_sent = False
    for step in chain:
        kind = step[0]
        if kind == "srt":
            items.append(_message("user", srt(step[1])))
            if not task_sent:
                items.append(_message("developer", BACKEND_TASK))
                task_sent = True
        elif kind == "dev":
            items.append(_message("developer", step[1]))
        elif kind == "call":
            _, name, output, reply = step
            call_id = f"call_bench{len(items)}"
            items += [
                {"type": "function_call", "call_id": call_id, "name": name, "arguments": "{}"},
                {"type": "function_call_output", "call_id": call_id, "output": output},
                _message("assistant", reply),
            ]
        elif kind == "reply":
            items.append(_message("assistant", step[1]))
        else:
            raise ValueError(f"unknown step {kind!r}")
    return items


def backend_tools(catalog: dict[str, songs.Song]) -> list[dict[str, Any]]:
    """운영 백엔드 툴 — 서버는 web_search 를 맨 뒤에 둔다."""
    play_song = make_play_song_tool_def(catalog)
    return [
        *[t for t in DEFAULT_TOOLS if t["name"] in LOOP_TOOLS or t["name"] in _HANDLED],
        *([play_song] if play_song is not None else []),
        *RADIO_TOOLS,
        {"type": "web_search"},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--live", type=Path, help="라이브 지시문을 이 파일 내용으로 바꾼다")
    parser.add_argument("--case", action="append", help="이 케이스만 (여러 번 지정 가능)")
    args = parser.parse_args()

    songs._SONG_FILES = tuple(f for f in songs._SONG_FILES if f[0] != "audio/music")  # 음원 없이 목록만
    catalog = songs.load_song_catalog()
    tools = backend_tools(catalog)
    live = args.live.read_text() if args.live else build_live_instructions(now=datetime.now().astimezone())
    client = OpenAI()

    def one(name: str, chain: list[Step], status: str) -> str:
        instructions = request_instructions(live, build_backend_instructions(catalog, status))
        for attempt in range(3):
            try:
                r = client.responses.create(
                    model=_LIVE_BACKEND_MODEL,
                    instructions=instructions,
                    tools=tools,
                    reasoning={"effort": _LIVE_BACKEND_REASONING},
                    input=request_input(chain),
                    store=False,
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
        for name, (chain, status, expect) in CASES.items():
            if args.case and name not in args.case:
                continue
            run = lambda _, n=name, ch=chain, st=status: one(n, ch, st)  # noqa: E731
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
