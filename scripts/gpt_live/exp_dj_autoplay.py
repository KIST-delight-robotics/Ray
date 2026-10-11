"""DJ 곡 자동 재생 실험 — 실작동 경로로 "디제이 모드 틀어줘" 부터 곡 N 개까지 돌리고 이야기 끝 판정을 기록한다.

판정 경로 (곡마다):
    closing   종료 문구 + 무음으로 앱이 바로 틀었다
    check     문구 없이 멈춰 백엔드 판정 요청 → play_dj_song
    request   그 밖의 play_dj_song (대화 모델이 스스로 위임)

    # 자연 경로: 진입 소재 → 곡 → 다음 소재 → 곡 (노래는 스텁이 2 s 로 흉내)
    uv run python scripts/gpt_live/exp_dj_autoplay.py --segment drawer --rounds 2
    # 문구 없는 경로: 라이브 지시문에서 종료 문구 규칙을 빼 판정 요청을 일부러 일으킨다
    uv run python scripts/gpt_live/exp_dj_autoplay.py --segment drawer --no-closing-line --songs 1

결과: var/experiments/dj_autoplay/<번호>_<태그>/ (results.json, 회차별 WAV, pipeline.log).
시각은 진입(dj_begin) 기준 초.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import pipeline_harness as h

from voice_pipeline.engines.gpt_live import instructions as instructions_mod
from voice_pipeline.engines.gpt_live.instructions import DJ_RETURN_INSTRUCTIONS
from voice_pipeline.engines.gpt_live.loop import DJWatch, LiveSessionLoop

REQUEST = "디제이 모드 틀어줘"
CLOSING_RULE = (
    '- Finish with this exact closing line and say nothing after it: "<artist>의 <title>, 들려드릴게요." '
    'For example: "악뮤의 소문의 낙원, 들려드릴게요."\n'
)


def drop_closing_rule() -> None:
    """라이브 지시문에서 종료 문구 규칙만 뺀다 (이 프로세스 안에서만). build_live_instructions 가 호출 때 읽는다."""
    text = instructions_mod.DEFAULT_LIVE_INSTRUCTIONS
    assert text.count(CLOSING_RULE) == 1, "지시문에서 종료 문구 규칙을 못 찾음"
    instructions_mod.DEFAULT_LIVE_INSTRUCTIONS = text.replace(CLOSING_RULE, "")


def record_watch() -> None:
    """이야기 끝 판정 지점에 기록 훅을 건다 (동작은 바꾸지 않는다)."""
    orig_play = LiveSessionLoop._dj_play_now
    orig_check = LiveSessionLoop._dj_send_check
    orig_finished = LiveSessionLoop._dj_check_finished

    def play_now(self: LiveSessionLoop, quiet: float) -> None:
        if h.CURRENT is not None:
            h.CURRENT.mark("dj_closing", quiet=round(quiet, 2), tail=self._dj_talk_text[-80:])
        orig_play(self, quiet)

    def send_check(self: LiveSessionLoop, now: float, quiet: float) -> None:
        if h.CURRENT is not None:
            h.CURRENT.mark("dj_check", quiet=round(quiet, 2), tail=self._dj_talk_text[-80:])
        orig_check(self, now, quiet)

    def check_finished(self: LiveSessionLoop, reason: str) -> None:
        if h.CURRENT is not None and self._dj_watch is DJWatch.CHECKING and not self._dj_check_done:
            h.CURRENT.mark("dj_check_done", reason=reason)  # 루프가 실제로 '곡 없이 끝남'으로 받을 때만
        orig_finished(self, reason)

    LiveSessionLoop._dj_play_now = play_now  # type: ignore[method-assign]
    LiveSessionLoop._dj_send_check = send_check  # type: ignore[method-assign]
    LiveSessionLoop._dj_check_finished = check_finished  # type: ignore[method-assign]


def summarize(rec: h.Recorder) -> dict[str, Any]:
    begin = rec.first("dj_begin")
    t0 = begin["t"] if begin else None

    def rel(t: float | None) -> float | None:
        return round(t - t0, 1) if t is not None and t0 is not None else None

    songs = []
    for play in rec.all("song_play"):
        last_voice = max((t for t, _, voiced in rec.audio if voiced and t < play["t"]), default=None)
        closing = [e for e in rec.all("dj_closing") if e["t"] <= play["t"]]
        checks = [e for e in rec.all("dj_check") if e["t"] <= play["t"]]
        calls = [e for e in rec.all("backend_call") if e["fn"] == "play_dj_song" and e["t"] <= play["t"]]
        if closing and (not calls or closing[-1]["t"] > calls[-1]["t"]):
            path = "closing"
        elif checks and calls and calls[-1]["t"] > checks[-1]["t"]:
            path = "check"
        else:
            path = "request"
        songs.append(
            {
                "at": rel(play["t"]),
                "path": path,
                "last_voice_to_song": round(play["t"] - last_voice, 2) if last_voice else None,
            }
        )
    checks = []
    for e in rec.all("dj_check"):
        after = [s for s in rec.transcripts.values() if s["speaker"] == "assistant" and s["t"] > e["t"]]
        checks.append(
            {
                "at": rel(e["t"]),
                "tail": e["tail"],
                "backend_calls": [c["fn"] for c in rec.all("backend_call", after=e["t"])][:3],
                "backend_replies": [r["text"] for r in rec.all("backend_reply", after=e["t"])][:3],
                "done": [d["reason"] for d in rec.all("dj_check_done", after=e["t"])][:1],
                "assistant_after": [s["text"] for s in sorted(after, key=lambda s: s["t"])][:2],
            }
        )
    returns = [e for e in rec.all("append_instructions") if e["text"] == DJ_RETURN_INSTRUCTIONS]
    return {
        "entered": begin is not None,
        "songs": songs,
        "closing_lines": [{"at": rel(e["t"]), "tail": e["tail"]} for e in rec.all("dj_closing")],
        "checks": checks,
        "return_signals": [rel(e["t"]) for e in returns],
        "calls": [e["fn"] for e in rec.all("backend_call")],
        "transcripts": [
            {"speaker": s["speaker"], "t": rel(s["t"]), "text": s["text"]}
            for s in sorted(rec.transcripts.values(), key=lambda s: s["t"])
        ],
        "voice_segments": [[rel(a), rel(b)] for a, b in rec.voice_segments()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--segment", default="drawer", help="첫 소재 키. 다음 소재는 나머지를 이름순으로")
    parser.add_argument("--songs", type=int, default=2, help="이만큼 곡이 나오면 1 s 뒤 끝낸다")
    parser.add_argument("--no-closing-line", action="store_true", help="종료 문구 규칙을 빼 판정 요청 경로를 본다")
    parser.add_argument("--max", type=float, default=300.0, help="연결 뒤 상한 초")
    parser.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    if args.no_closing_line:
        drop_closing_rule()
    record_watch()
    comps, out_dir = h.setup("dj_autoplay", args.tag)
    picker = comps.dj_picker
    assert picker is not None, "DJ 소재가 없다 (재생 가능한 곡이 없거나 소재 폴더가 비었다)"
    order = [picker.segments[args.segment], *(s for k, s in sorted(picker.segments.items()) if k != args.segment)]
    speech = h.tts(REQUEST)

    def stop_when(rec: h.Recorder, loop: LiveSessionLoop, now: float) -> bool:
        if rec.first("dj_begin") is None:
            return rec.connected_at is not None and now - rec.connected_at > 30.0
        plays = rec.all("song_play")
        return len(plays) >= args.songs and now - plays[-1]["t"] > 1.0

    results = []
    for rnd in range(args.rounds):
        queue = iter(order)
        picker.pick = lambda queue=queue: next(queue, None)  # type: ignore[method-assign]
        rec = h.Recorder(f"#{rnd}")
        h.run_session(
            comps, rec, speech=speech, stop_when=stop_when, max_sec=args.max, wav_path=out_dir / f"run_{rnd}.wav"
        )
        row = {"round": rnd, **summarize(rec)}
        results.append(row)
        h.save_json(out_dir / "results.json", results)
        paths = ", ".join(f"{s['path']}@{s['at']}(+{s['last_voice_to_song']}s)" for s in row["songs"]) or "-"
        print(f"[#{rnd}] entered={row['entered']} songs: {paths} checks={len(row['checks'])}", flush=True)
        time.sleep(1.5)
    print(f"results: {out_dir}")


if __name__ == "__main__":
    main()
