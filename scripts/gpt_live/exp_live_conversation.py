"""대화 모델 지시문 평가 — 실작동 경로로 기본 대화·위임 경우를 돌려 위임 여부·백엔드 툴·대답을 기록한다.

라이브 지시문만 바꿔 비교한다. ``--live current`` 는 운영 지시문, ``simple`` 은 ``live_instructions_simple.txt``.
백엔드 지시문·툴은 운영 그대로. 장기기억은 운영처럼 켜되 실험 폴더의 빈 DB 를 쓴다.
사용자 발화는 TTS 로 넣고, 대답이 끝나면 다음 발화를 넣는다 — 발화가 끝난 뒤 모델 음성이 나왔고, 그 뒤 받은 음성·
위임·백엔드 활동이 ``QUIET_SEC`` 동안 없을 때. 음성이 ``NO_REPLY_SEC`` 동안 없으면 대답 없음으로 넘어간다.

판정(발화마다): 위임 여부가 기대와 같고, 기대한 툴이 모두 불렸고, 금지한 툴이 불리지 않았으면 통과.
위임·툴은 그 발화부터 다음 발화 전까지 생긴 것을 센다.

    uv run python scripts/gpt_live/exp_live_conversation.py --live simple --runs 3 --tag t1
    uv run python scripts/gpt_live/exp_live_conversation.py --live current --runs 3 --tag t1   # 다른 터미널에서 병렬
    uv run python scripts/gpt_live/exp_live_conversation.py --compare t1

결과: var/experiments/live_conversation/<번호>_<태그>/<조건>/ — results.json, pipeline.log, instructions.txt,
<경우>_<회차>.wav(왼쪽 사용자, 오른쪽 모델) + .labels.txt(Audacity 레이블: 사용자 발화, 위임, 툴, 모델 전사).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pipeline_harness as h
from exp_dirs import find_result_dir

from voice_pipeline.engines.gpt_live import instructions as instructions_mod
from voice_pipeline.engines.gpt_live.loop import LiveSessionLoop
from voice_pipeline.settings import SAMPLE_RATE

SIMPLE_PATH = Path(__file__).resolve().parent / "live_instructions_simple.txt"
QUIET_SEC = 3.5  # 대답 끝 — 위임을 기다리는 동안의 쉼(1~2 s)보다 길게
NO_REPLY_SEC = 12.0
FIRST_SAY_DELAY_SEC = 1.0  # run_session 이 연결 뒤 이만큼 무음을 넣고 첫 발화를 흘린다
SONG_SEC = 2.0
ACTIVITY = ("delegation_created", "backend_call", "backend_reply", "submit_function_output", "song_play")


@dataclass(frozen=True)
class Turn:
    say: str
    delegate: bool  # 위임해야 하나
    tools: tuple[str, ...] = ()  # 모두 불려야 하는 백엔드 툴 (web_search 포함)
    forbid: tuple[str, ...] = ()  # 불리면 안 되는 툴


# 날씨·시각은 지역을 넣는다 — 실험 DB 가 비어 사는 곳을 모르면 백엔드가 지역부터 되묻는다(지시문대로).
CASES: dict[str, tuple[Turn, ...]] = {
    "weather": (Turn("서울 날씨 어때?", True, ("web_search",)),),
    "time": (Turn("지금 서울은 몇 시야?", True, ("web_search",)),),
    "volume_up": (Turn("소리 좀 키워줘", True, ("adjust_volume",)),),
    "light_on": (Turn("불 켜져 있어?", True, ("get_device_settings",)),),
    "music_any": (Turn("노래 틀어줘", True, forbid=("play_song",)),),
    "music_named": (Turn("십센치 서랍 틀어줘", True, ("play_song",)),),
    "memory": (Turn("지난번에 내가 뭐 좋아한다고 했었지?", True, ("search_memory",)),),
    "goodbye": (Turn("이제 그만할게, 다음에 봐", True, ("end_conversation",)),),
    "greeting": (Turn("안녕", False),),
    "tired": (Turn("오늘 좀 피곤하네", False),),
    "backchannel": (Turn("요즘 잠을 잘 못 자", False), Turn("음, 그렇구나", False)),
    "clarify": (Turn("그거 좀 바꿔줘", False),),
    "reuse": (Turn("서울 날씨 어때?", True, ("web_search",)), Turn("아까 몇 도라고 했지?", False)),
}


class Script:
    """발화를 차례로 넣는다. :meth:`hook` 은 루프 스레드에서 프레임마다 불린다."""

    def __init__(self, rec: h.Recorder, turns: tuple[Turn, ...], speech: list[bytes]) -> None:
        self.rec = rec
        self.turns = turns
        self.speech = speech
        self.spans: list[tuple[float, float]] = []  # 발화마다 (시작, 끝) — 넣은 시각 + 길이로 잡는다
        self.done_at: float | None = None

    def hook(self, loop: LiveSessionLoop, now: float) -> None:
        if self.rec.connected_at is None or self.done_at is not None:
            return
        if not self.spans:  # 첫 발화는 run_session 이 흘린다
            self._start(0, self.rec.connected_at + FIRST_SAY_DELAY_SEC)
            return
        if not self._reply_finished(now):
            return
        nxt = len(self.spans)
        if nxt >= len(self.turns):
            self.done_at = now
            return
        self.rec.inputs.put(self.speech[nxt])
        self._start(nxt, now)

    def _start(self, k: int, start: float) -> None:
        self.spans.append((start, start + len(self.speech[k]) / (SAMPLE_RATE * 2)))
        self.rec.events.append({"t": start, "name": "user_say", "text": self.turns[k].say})

    def _reply_finished(self, now: float) -> bool:
        _, end = self.spans[-1]
        if now < end:
            return False
        voices = [t for t, _, voiced in self.rec.audio if voiced and t > end]
        if not voices:
            return now - end >= NO_REPLY_SEC
        last = max([voices[-1], *(e["t"] + (SONG_SEC if e["name"] == "song_play" else 0.0) for e in self._activity())])
        return now - last >= QUIET_SEC

    def _activity(self) -> list[dict[str, Any]]:
        start = self.spans[-1][0]
        return [e for e in self.rec.events if e["name"] in ACTIVITY and e["t"] >= start]


def judge(rec: h.Recorder, script: Script, end_t: float) -> list[dict[str, Any]]:
    rows = []
    for k, turn in enumerate(script.turns):
        if k >= len(script.spans):
            rows.append({"say": turn.say, "fed": False, "ok": False})
            continue
        start, said_end = script.spans[k]
        stop = script.spans[k + 1][0] if k + 1 < len(script.spans) else end_t

        def within(t: float, start: float = start, stop: float = stop) -> bool:
            return start <= t < stop

        delegated = any(within(e["t"]) for e in rec.all("delegation_created"))
        tools = [e["fn"] for e in rec.all("backend_call") if within(e["t"])]
        ok = (
            delegated == turn.delegate
            and all(t in tools for t in turn.tools)
            and not any(t in tools for t in turn.forbid)
        )
        segs = sorted(rec.transcripts.values(), key=lambda s: s["t"])
        voice = next((t for t, _, voiced in rec.audio if voiced and t > said_end), None)
        rows.append(
            {
                "say": turn.say,
                "fed": True,
                "ok": ok,
                "delegated": delegated,
                "tools": tools,
                "first_voice_sec": round(voice - said_end, 2) if voice is not None and voice < stop else None,
                "heard": " / ".join(s["text"] for s in segs if s["speaker"] == "user" and within(s["t"])),
                "said": " / ".join(s["text"] for s in segs if s["speaker"] == "assistant" and within(s["t"])),
                "backend_replies": [e["text"] for e in rec.all("backend_reply") if within(e["t"]) and e["text"]],
            }
        )
    return rows


def save_audio(rec: h.Recorder, path: Path) -> None:
    """녹음(왼쪽 사용자, 오른쪽 모델)과 Audacity 레이블. 레이블 위치는 그 시각까지 받은 오디오 길이."""
    h.save_conversation_wav(rec, path)

    def pos(t: float) -> float:
        return h.audio_pos(rec, t)

    # 사용자 발화는 실제로 흘리기 시작한 시각(녹음 왼쪽 채널과 같은 위치)으로
    says = rec.all("user_say")
    starts = [t for t, _ in rec.user_clips] + [e["t"] for e in says[len(rec.user_clips) :]]
    labels = [(pos(t), f"USER: {e['text']}") for e, t in zip(says, starts, strict=True)]
    labels += [(pos(e["t"]), "delegation") for e in rec.all("delegation_created")]
    labels += [(pos(e["t"]), f"tool: {e['fn']}") for e in rec.all("backend_call")]
    labels += [(pos(e["t"]), f"volume {e['percent']}%") for e in rec.all("volume_set")]
    labels += [(pos(e["t"]), f"song: {e['song']}") for e in rec.all("song_play")]
    labels += [(pos(s["t"]), f"RAY: {s['text']}") for s in rec.transcripts.values() if s["speaker"] == "assistant"]
    labels.sort()
    path.with_suffix(".labels.txt").write_text("".join(f"{p:.3f}\t{p:.3f}\t{text}\n" for p, text in labels))


def run(args: argparse.Namespace) -> None:
    if args.live == "simple":
        instructions_mod.DEFAULT_LIVE_INSTRUCTIONS = SIMPLE_PATH.read_text(encoding="utf-8")
    comps, out_dir = h.setup("live_conversation", args.tag, subdir=args.live)
    (out_dir / "instructions.txt").write_text(instructions_mod.DEFAULT_LIVE_INSTRUCTIONS)
    cases = {k: v for k, v in CASES.items() if not args.case or k in args.case}
    speech = {turn.say: h.tts(turn.say) for turns in cases.values() for turn in turns}

    results: list[dict[str, Any]] = []
    for rnd in range(args.runs):
        for name, turns in cases.items():
            rec = h.Recorder(f"{name}#{rnd}")
            script = Script(rec, turns, [speech[t.say] for t in turns])
            rec.frame_hook = script.hook

            def stop_when(rec: h.Recorder, loop: LiveSessionLoop, now: float, script: Script = script) -> bool:
                return script.done_at is not None and now - script.done_at > 0.5

            h.run_session(
                comps,
                rec,
                speech=speech[turns[0].say],
                stop_when=stop_when,
                max_sec=args.max,
                song_sec=SONG_SEC,
                memory_enabled=True,
            )
            rows = judge(rec, script, time.monotonic())
            save_audio(rec, out_dir / f"{name}_{rnd}.wav")
            results.append({"case": name, "run": rnd, "turns": rows})
            h.save_json(out_dir / "results.json", {"live": args.live, "results": results})
            for r in rows:
                mark = "ok  " if r["ok"] else "FAIL"
                tools = ",".join(r.get("tools") or []) or "-"
                said = (r.get("said") or "")[:70]
                print(f"[{name}#{rnd}] {mark} {r['say']!r} deleg={r.get('delegated')} tools={tools} | {said}")
            time.sleep(1.5)
    print_summary(results, args.live)
    print(f"results: {out_dir}")


def tally(results: list[dict[str, Any]]) -> dict[tuple[str, int], tuple[int, int]]:
    """(경우, 발화 순번) → (통과, 회차). 발화를 넣지 못한 회차(세션 시작 실패 등)는 세지 않는다."""
    out: dict[tuple[str, int], tuple[int, int]] = {}
    for r in results:
        for k, t in enumerate(r["turns"]):
            if not t["fed"]:
                continue
            ok, n = out.get((r["case"], k), (0, 0))
            out[(r["case"], k)] = (ok + t["ok"], n + 1)
    return out


def print_summary(results: list[dict[str, Any]], live: str) -> None:
    print(f"\n[{live}] 통과")
    for (case, k), (ok, n) in tally(results).items():
        print(f"  {case:12s} {CASES[case][k].say:28s} {ok}/{n}")


def compare(tag: str) -> None:
    root = find_result_dir("live_conversation", tag)
    if root is None:
        raise SystemExit(f"결과 폴더 없음: {tag}")
    tallies = {}
    for live in ("current", "simple"):
        path = root / live / "results.json"
        tallies[live] = tally(json.loads(path.read_text())["results"]) if path.exists() else {}
    keys = list(dict.fromkeys([*tallies["current"], *tallies["simple"]]))
    print(f"{'case':12s} {'say':28s} {'current':>8s} {'simple':>8s}")
    for case, k in keys:
        cells = []
        for live in ("current", "simple"):
            ok, n = tallies[live].get((case, k), (0, 0))
            cells.append(f"{ok}/{n}" if n else "-")
        print(f"{case:12s} {CASES[case][k].say:28s} {cells[0]:>8s} {cells[1]:>8s}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", choices=("current", "simple"), default="simple")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--case", nargs="*", choices=list(CASES), help="이 경우만")
    parser.add_argument("--max", type=float, default=90.0, help="세션당 상한 초")
    parser.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--compare", metavar="TAG", help="두 조건 결과를 나란히 보여주고 끝낸다")
    args = parser.parse_args()
    if args.compare:
        compare(args.compare)
    else:
        run(args)


if __name__ == "__main__":
    main()
