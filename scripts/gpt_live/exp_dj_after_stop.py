"""노래를 멈춘 뒤 DJ 때 일을 기억하는지 — 실작동 경로 재현 (2026-10-07 실기: "기억이 안 난다").

순서: "디제이 모드 틀어줘" → 이야기 → 종료 문구 → 곡(스텁, 길게) → 3 s 뒤 "노래 멈춰줘" → 대답이 끝나면
"아까 무슨 노래 틀었어?" → 대답이 끝나면 "아까 무슨 얘기 했었지?" → 대답이 끝나면 종료.

변형(--variant):
    base     지금 그대로
    signal   DJ 종료 신호에 "방금의 이야기와 노래도 이 대화의 일부" 한 문장 추가
    modes    라이브 지시문에서 "Follow only the rules of the current mode." 제거

판정(질문마다, 대충 분류 — 답 원문을 함께 남긴다): remembers(곡명·가수 / 소재 단어가 나옴), denies(기억 안 남·모름),
other. 질문이 위임됐는지와 백엔드 답도 남긴다.

    uv run python scripts/gpt_live/exp_dj_after_stop.py --variant base --rounds 3

결과: var/experiments/dj_after_stop/<태그>/ (results.json, pipeline.log)
"""

from __future__ import annotations

import argparse
import re
import time
from typing import Any

import pipeline_harness as h

from voice_pipeline.engines.gpt_live import instructions as instructions_mod
from voice_pipeline.engines.gpt_live import loop as loop_mod
from voice_pipeline.engines.gpt_live.loop import LiveSessionLoop, SongState

SEGMENT = "wildflower"  # 실기와 같은 소재
REQUEST = "디제이 모드 틀어줘"
STOP = "노래 멈춰줘"
QUESTIONS = (("song", "아까 무슨 노래 틀었어?"), ("talk", "아까 무슨 얘기 했었지?"))
SONG_WORDS = ("야생화", "박효신")
TALK_WORDS = ("풀", "길가", "출퇴근", "질경이", "민들레", "이름")
DENIAL = re.compile(r"기억.{0,6}(안|못|않)|모르|떠오르지|떠올리지|떠올릴 수가 없")
SIGNAL_EXTRA = " The talk and the song you just played are part of this conversation; you can talk about them."
MODES_SENTENCE = " Follow only the rules of the current mode."
QUIET_AFTER_ANSWER = 3.5  # 대답이 끝났다고 보는 무음 (위임 대기 중 쉼보다 길게)


def apply_variant(variant: str) -> None:
    if variant == "signal":
        loop_mod.DJ_ENDED_INSTRUCTIONS += SIGNAL_EXTRA  # 루프가 이름으로 가져다 쓴다
    elif variant == "modes":
        text = instructions_mod.DEFAULT_LIVE_INSTRUCTIONS
        assert text.count(MODES_SENTENCE) == 1, "지시문에서 모드 문장을 못 찾음"
        instructions_mod.DEFAULT_LIVE_INSTRUCTIONS = text.replace(MODES_SENTENCE, "")


def classify(text: str, words: tuple[str, ...]) -> str:
    if any(w in text for w in words):
        return "remembers"
    return "denies" if DENIAL.search(text) else "other"


def answered(rec: h.Recorder, loop: LiveSessionLoop, since: float, now: float) -> bool:
    """``since`` 뒤에 모델이 말했고, 지금 조용하며, 진행 중인 위임·툴이 없다."""
    last = loop._last_voice_time or 0.0
    busy = loop._inflight or loop._pending_calls or loop._live._active_responses
    return last > since and now - last >= QUIET_AFTER_ANSWER and not busy


def summarize(rec: h.Recorder, ended_signal: str) -> dict[str, Any]:
    marks = {name: rec.first(name) for name in ("dj_begin", "song_play", "stop_queued", "song", "talk", "done")}
    segs = sorted(rec.transcripts.values(), key=lambda s: s["t"])

    def said_between(a: dict | None, b: dict | None) -> str:
        if a is None:
            return ""
        end = b["t"] if b else float("inf")
        return " ".join(s["text"] for s in segs if s["speaker"] == "assistant" and a["t"] < s["t"] < end)

    def backend_between(a: dict | None, b: dict | None) -> dict[str, list[str]]:
        if a is None:
            return {"calls": [], "replies": [], "delegations": 0}
        end = b["t"] if b else float("inf")
        return {
            "calls": [e["fn"] for e in rec.all("backend_call", after=a["t"]) if e["t"] < end],
            "replies": [e["text"] for e in rec.all("backend_reply", after=a["t"]) if e["t"] < end],
            "delegations": len([e for e in rec.all("delegation_created", after=a["t"]) if e["t"] < end]),
        }

    song_answer = said_between(marks["song"], marks["talk"])
    talk_answer = said_between(marks["talk"], marks["done"])
    return {
        "reached": {k: v is not None for k, v in marks.items()},
        "stopped_by_request": any(e["fn"] == "stop_song" for e in rec.all("backend_call")),  # 아니면 회차 무효
        "after_stop": said_between(marks["stop_queued"], marks["song"]),
        "ended_signal_sent": any(e["text"] == ended_signal for e in rec.all("append_instructions")),
        "song_q": {
            "verdict": classify(song_answer, SONG_WORDS),
            "answer": song_answer,
            **backend_between(marks["song"], marks["talk"]),
        },  # fmt: skip
        "talk_q": {
            "verdict": classify(talk_answer, TALK_WORDS),
            "answer": talk_answer,
            **backend_between(marks["talk"], marks["done"]),
        },  # fmt: skip
        "user": rec.said("user"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=["base", "signal", "modes"], required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    apply_variant(args.variant)
    comps, out_dir = h.setup("dj_after_stop", args.tag or args.variant)
    picker = comps.dj_picker
    assert picker is not None, "DJ 소재가 없다"
    segment = picker.segments[SEGMENT]
    picker.pick = lambda: segment  # type: ignore[method-assign]
    request, stop = h.tts(REQUEST), h.tts(STOP)
    questions = [(name, h.tts(text)) for name, text in QUESTIONS]
    ended_signal = loop_mod.DJ_ENDED_INSTRUCTIONS

    def hook(loop: LiveSessionLoop, now: float, rec: h.Recorder) -> None:
        play = rec.first("song_play")
        if play is None:
            return
        stop_mark = rec.first("stop_queued")
        if stop_mark is None:
            if now - play["t"] > 3.0:
                rec.inputs.put(stop)
                rec.mark("stop_queued")
            return
        if loop._song_state is not SongState.NONE:
            return
        prev = stop_mark
        for name, pcm in questions:
            mark = rec.first(name)
            if mark is None:
                if answered(rec, loop, prev["t"], now):
                    rec.inputs.put(pcm)
                    rec.mark(name)
                return
            prev = mark
        if rec.first("done") is None and answered(rec, loop, prev["t"], now):
            rec.mark("done")

    def stop_when(rec: h.Recorder, loop: LiveSessionLoop, now: float) -> bool:
        if rec.first("dj_begin") is None:
            return rec.connected_at is not None and now - rec.connected_at > 30.0
        return rec.first("done") is not None

    results = []
    for rnd in range(args.rounds):
        rec = h.Recorder(f"#{rnd}")
        rec.frame_hook = lambda loop, now, rec=rec: hook(loop, now, rec)
        h.run_session(comps, rec, speech=request, stop_when=stop_when, max_sec=200.0, song_sec=60.0)
        row = {"round": rnd, "variant": args.variant, **summarize(rec, ended_signal)}
        results.append(row)
        h.save_json(out_dir / "results.json", results)
        print(
            f"[{args.variant} #{rnd}] stopped={row['stopped_by_request']} done={row['reached']['done']} "
            f"song_q={row['song_q']['verdict']} talk_q={row['talk_q']['verdict']}",
            flush=True,
        )
        time.sleep(1.5)
    print(f"results: {out_dir}")


if __name__ == "__main__":
    main()
