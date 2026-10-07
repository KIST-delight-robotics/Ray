"""DJ 곡 소개 뒤 위임 실험 — 실작동 경로로 "DJ 모드 틀어줘" 부터 곡 소개 뒤까지 돌리고 play_dj_song 위임을 센다.

판정 (회차마다):
    SPONTANEOUS   첫 복귀 신호 전에 play_dj_song
    AFTER_SIGNAL  복귀 신호 뒤 --after-signal 초 안에 play_dj_song
    NONE          위임 없음
    NO_ENTRY      DJ 모드에 들어가지 못함 (진입 위임 누락·요청 못 들음)

    # 실작동 그대로 (6 s 무음 복귀 신호, 신호 뒤 20 s)
    uv run python scripts/gpt_live/exp_dj_song_delegation.py --rounds 6
    # 짧은 소재, 신호 없이 이야기 끝 3 s 무음에서 끊기
    uv run python scripts/gpt_live/exp_dj_song_delegation.py --segment short --quiet 60 --end-silence 3
    # 복귀 신호 문구·시점 바꾸기
    uv run python scripts/gpt_live/exp_dj_song_delegation.py --segment short --quiet 3 --after-signal 6 \\
        --signal "DJ: If you have just introduced the song, delegate now to play it, without speaking. ..."

결과: var/experiments/dj_song_delegation/<태그>/ (results.json, 회차별 WAV, pipeline.log)
"""

from __future__ import annotations

import argparse
import time

import pipeline_harness as h

from voice_pipeline.engines.gpt_live import loop as loop_mod
from voice_pipeline.engines.gpt_live.dj import Segment
from voice_pipeline.engines.gpt_live.loop import LiveSessionLoop

REQUEST = "디제이 모드 틀어줘"
SHORT_NOTES = (
    "- 서랍은 집에서 가장 느리게 바뀌는 자리. 몇 년을 그대로 감\n"
    "- 곡 연결: 10CM(읽기: 십센치) '서랍'은 헤어진 뒤 남은 물건과 마음을 서랍에 넣어 두는 노래"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--segment", default="drawer", help="소재 키, 또는 short (노트 2줄짜리 서랍)")
    parser.add_argument("--signal", help="복귀 신호 문구 (기본: 실작동 DJ_RETURN_INSTRUCTIONS)")
    parser.add_argument("--quiet", type=float, help="복귀 신호까지 무음 초 (기본: 실작동 _DJ_QUIET_SEC)")
    parser.add_argument("--after-signal", type=float, default=20.0, help="첫 복귀 신호 뒤 지켜볼 초")
    parser.add_argument("--end-silence", type=float, help="복귀 신호 전, 이야기 뒤 이만큼 무음이면 끊는다")
    parser.add_argument("--max", type=float, default=150.0, help="연결 뒤 상한 초")
    parser.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    comps, out_dir = h.setup("dj_song_delegation", args.tag)
    assert comps.dj_picker is not None, "DJ 소재가 없다 (재생 가능한 곡이 없거나 소재 폴더가 비었다)"
    if args.segment == "short":
        segment = Segment("drawer_short", "서랍 속에 남아 있는 것들", comps.song_catalog["Drawer_10cm"], SHORT_NOTES)
    else:
        segment = comps.dj_picker.segments[args.segment]
    comps.dj_picker.pick = lambda: segment  # type: ignore[method-assign]
    if args.signal:
        loop_mod.DJ_RETURN_INSTRUCTIONS = args.signal
    if args.quiet is not None:
        LiveSessionLoop._DJ_QUIET_SEC = args.quiet
    signal = loop_mod.DJ_RETURN_INSTRUCTIONS
    speech = h.tts(REQUEST)

    def signal_at(rec: h.Recorder) -> float | None:
        return next((e["t"] for e in rec.all("append_instructions") if e["text"] == signal), None)

    def stop_when(rec: h.Recorder, loop: LiveSessionLoop, now: float) -> bool:
        begin = rec.first("dj_begin")
        if begin is None:
            return rec.connected_at is not None and now - rec.connected_at > 30.0
        song = rec.first("song_play")
        if song is not None:
            return now - song["t"] > 1.0
        delegations = rec.all("delegation_created", after=begin["t"])
        if delegations:
            return now - delegations[-1]["t"] > 10.0  # 위임했는데 노래가 안 나옴
        sig = signal_at(rec)
        if sig is not None:
            return now - sig > args.after_signal
        last_voice = loop._last_voice_time or 0.0
        talked = last_voice > begin["t"]
        return args.end_silence is not None and talked and now - last_voice > args.end_silence

    results = []
    for rnd in range(args.rounds):
        rec = h.Recorder(f"#{rnd}")
        h.run_session(
            comps, rec, speech=speech, stop_when=stop_when, max_sec=args.max, wav_path=out_dir / f"run_{rnd}.wav"
        )
        begin = rec.first("dj_begin")
        t0 = begin["t"] if begin else None
        song_call = next((e for e in rec.all("backend_call") if e["fn"] == "play_dj_song"), None)
        sig = signal_at(rec)
        if t0 is None:
            verdict = "NO_ENTRY"
        elif song_call is None:
            verdict = "NONE"
        elif sig is None or song_call["t"] < sig:
            verdict = "SPONTANEOUS"
        else:
            verdict = "AFTER_SIGNAL"

        def rel(t: float | None, t0: float | None = t0) -> float | None:
            return round(t - t0, 1) if t is not None and t0 is not None else None

        said = rec.said()
        results.append(
            {
                "round": rnd,
                "verdict": verdict,
                "play_dj_song_at": rel(song_call["t"] if song_call else None),
                "signal_at": rel(sig),
                "calls": [e["fn"] for e in rec.all("backend_call")],
                "assistant": said,
                "user": rec.said("user"),
                # 진입(dj_begin) 기준 초. 쉼 길이는 조각 도착 시각 기준이라 네트워크 정지도 쉼으로 잡힌다
                "voice_segments": [[rel(a), rel(b)] for a, b in rec.voice_segments()],
                "transcripts": [
                    {"speaker": s["speaker"], "t": rel(s["t"]), "text": s["text"]}
                    for s in sorted(rec.transcripts.values(), key=lambda s: s["t"])
                ],
            }
        )
        h.save_json(out_dir / "results.json", results)
        tail = said[-1][-120:] if said else "-"
        print(
            f"[#{rnd}] {verdict:12s} play_dj_song@{rel(song_call['t'] if song_call else None)} "
            f"signal@{rel(sig)}\n    …{tail}",
            flush=True,
        )
        time.sleep(1.5)

    counts = {k: sum(r["verdict"] == k for r in results) for k in ("SPONTANEOUS", "AFTER_SIGNAL", "NONE", "NO_ENTRY")}
    print(counts)
    print(f"results: {out_dir}")


if __name__ == "__main__":
    main()
