"""사용자 발화 없이 앱 신호(instructions.append)만으로 위임이 생기는지 확인.

일반 대화 모드에서 사용자가 "안녕" → 모델 대답 → 출력 무음 3 s 뒤 신호 하나를 보내고, 8 s 동안 위임 이벤트와 모델
발화를 센다. 대화가 시작되기 전에 보낸 append 에는 모델이 반응하지 않으므로(FINDINGS: 먼저 말하지 않음)
대화 중에 보낸다.

    uv run python scripts/gpt_live/exp_delegation_signal.py --rounds 4
    uv run python scripts/gpt_live/exp_delegation_signal.py --signal "..." --signal "..."

결과: var/experiments/delegation_signal/<태그>/
"""

from __future__ import annotations

import argparse
import time

import pipeline_harness as h

from voice_pipeline.engines.gpt_live.loop import LiveSessionLoop

DEFAULT_SIGNALS = [
    "Delegate to the backend now to play the song 서랍 by 10CM, "
    "and say a short line such as '서랍 틀어드릴게요' as you do.",
    "Delegate to the backend now to play the song 서랍 by 10CM, without speaking.",
]
QUIET_BEFORE_SIGNAL_SEC = 3.0
WATCH_SEC = 8.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--signal", action="append", help="보낼 신호 문구 (여러 번 지정 가능, 회차마다 번갈아)")
    parser.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    signals = args.signal or DEFAULT_SIGNALS

    comps, out_dir = h.setup("delegation_signal", args.tag)
    speech = h.tts("안녕")
    results = []
    for rnd in range(args.rounds):
        for i, text in enumerate(signals):
            rec = h.Recorder(f"s{i}#{rnd}")

            def send_after_reply(loop: LiveSessionLoop, now: float, rec: h.Recorder = rec, text: str = text) -> None:
                replied = loop._last_voice_time is not None and loop._last_voice_time > (rec.connected_at or 0.0)
                if replied and rec.first("signal") is None and now - loop._last_voice_time > QUIET_BEFORE_SIGNAL_SEC:
                    rec.mark("signal")
                    loop._live.append_instructions(text)

            rec.frame_hook = send_after_reply

            def stop_when(rec: h.Recorder, loop: LiveSessionLoop, now: float) -> bool:
                sig = rec.first("signal")
                return sig is not None and now - sig["t"] > WATCH_SEC

            h.run_session(comps, rec, speech=speech, stop_when=stop_when, max_sec=30.0)
            sig = rec.first("signal")
            after = sig["t"] if sig else float("inf")
            delegated = bool(rec.all("delegation_created", after=after))
            spoke_after = any(t > after and v for t, _, v in rec.audio)
            results.append(
                {
                    "signal": i,
                    "round": rnd,
                    "sent": sig is not None,
                    "delegated": delegated,
                    "spoke_after_signal": spoke_after,
                    "assistant": rec.said(),
                }
            )
            h.save_json(out_dir / "results.json", results)
            print(f"[s{i}#{rnd}] sent={sig is not None} delegated={delegated} spoke_after={spoke_after}", flush=True)
            time.sleep(1.0)

    for i, text in enumerate(signals):
        rs = [r for r in results if r["signal"] == i and r["sent"]]
        print(
            f"s{i}: delegated {sum(r['delegated'] for r in rs)}/{len(rs)}, "
            f"spoke {sum(r['spoke_after_signal'] for r in rs)}/{len(rs)} — {text[:70]}"
        )
    print(f"results: {out_dir}")


if __name__ == "__main__":
    main()
