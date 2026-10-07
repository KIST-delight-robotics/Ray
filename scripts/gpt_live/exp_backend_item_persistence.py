"""앱이 백엔드 대화에 넣은 메시지(response.item.create)가 다음 위임에도 남는지 — 실작동 경로.

순서: "안녕" → 모델 대답 → 1 s 무음 → 앱이 표식 메시지 + response.create → 그 응답이 끝나고 2 s 뒤
"오늘 서울 날씨 어때?" → 날씨 위임의 백엔드 답을 본다.

판정 근거 두 가지:
    probe   실험용 백엔드 지시문 끝의 프로브 — 날씨 위임 답이 표식 코드(MARKER-7Q3)로 시작하면 남아 있다,
            NONE 이면 없다.
            표식 응답 자신은 빈 응답으로 끝내게 해 대화 모델이 코드를 소리 내지 않게 한다(전사로 새면 판정이 오염된다).
    chain   날씨 위임의 response.created 에 붙은 previous_response_id 가 표식 응답의 id 인가.
조건(--role): developer / user (표식 메시지 역할), none (표식 없음, 대조).

    uv run python scripts/gpt_live/exp_backend_item_persistence.py --role developer --rounds 3

결과: var/experiments/backend_item_persistence/<태그>/ (results.json, pipeline.log)
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from typing import Any

import pipeline_harness as h

from voice_pipeline.adapters.gpt_live import GPTLiveSession
from voice_pipeline.engines.gpt_live import instructions as instructions_mod

CODE = "MARKER-7Q3"
MARKER_NOTE = f"Application note {CODE}: no action needed."
PROBE = (
    "\n## Probe (experiment)\n"
    "If the latest message in your input is an application note, reply with an empty message.\n"
    "Otherwise, begin your reply with the code of any application note in your input (it looks like MARKER- "
    "followed by three characters), or with NONE if your input has no such note. Then answer as usual.\n"
)
MARKER_EVENT = "probe_marker"


def feed(q: queue.Queue[bytes], speech: bytes, rec: h.Recorder, stop: threading.Event) -> None:
    """연결 뒤 1 s 무음 → ``speech`` → 무음. 이후 ``rec.inputs`` 에 넣은 음성을 이어서 흘린다 (실시간 속도)."""
    while not stop.is_set() and rec.connected_at is None:
        time.sleep(0.01)
    rec.inputs = queue.Queue()  # type: ignore[attr-defined]
    frames: list[bytes] = [bytes(h.FRAME_SIZE_BYTES)] * int(1.0 / h.FRAME_SEC)

    def chunks(pcm: bytes) -> list[bytes]:
        pcm += bytes((-len(pcm)) % h.FRAME_SIZE_BYTES)
        return [pcm[i : i + h.FRAME_SIZE_BYTES] for i in range(0, len(pcm), h.FRAME_SIZE_BYTES)]

    frames += chunks(speech)
    next_t = time.monotonic()
    while not stop.is_set():
        if not frames:
            try:
                frames = chunks(rec.inputs.get_nowait())  # type: ignore[attr-defined]
            except queue.Empty:
                frames = [bytes(h.FRAME_SIZE_BYTES)]
        try:
            q.put_nowait(frames.pop(0))
        except queue.Full:
            pass
        next_t += h.FRAME_SEC
        time.sleep(max(0.0, next_t - time.monotonic()))


def record_raw() -> None:
    """위임·백엔드 응답의 id 와 previous_response_id, 백엔드 답, 오류를 기록한다 (동작은 바꾸지 않는다)."""
    orig = GPTLiveSession._dispatch

    def dispatch(self: GPTLiveSession, event: Any) -> None:
        rec = h.CURRENT
        t = getattr(event, "type", None)
        if rec is not None and t == "session.delegation.created":
            d = getattr(event, "delegation", None)
            rec.mark("deleg", id=getattr(d, "id", None), client_event_id=getattr(event, "client_event_id", None))
        elif rec is not None and t == "response.event":
            inner = event.event
            it = inner.get("type")
            resp = inner.get("response") or {}
            if it == "response.created":
                rec.mark("resp_created", delegation=event.delegation_id, id=resp.get("id"),
                         previous=resp.get("previous_response_id"))  # fmt: skip
            elif it in ("response.completed", "response.failed", "response.incomplete"):
                rec.mark("resp_done", delegation=event.delegation_id, status=it)
            elif it == "response.output_item.done" and (inner.get("item") or {}).get("type") == "message":
                parts = inner["item"].get("content") or []
                text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict)).strip()
                rec.mark("reply", delegation=event.delegation_id, text=text)
        elif rec is not None and t == "error":
            err = getattr(event, "error", None)
            rec.mark("api_error", message=str(getattr(err, "message", err)))
        orig(self, event)

    GPTLiveSession._dispatch = dispatch  # type: ignore[method-assign]


def summarize(rec: h.Recorder, role: str) -> dict[str, Any]:
    delegs = rec.all("deleg")
    marker = next((d["id"] for d in delegs if d["client_event_id"] == MARKER_EVENT), None)
    queued = rec.first("weather_queued")
    weather = next((d["id"] for d in delegs if queued and d["t"] > queued["t"] and not d["client_event_id"]), None)
    created = rec.all("resp_created")
    marker_ids = [c["id"] for c in created if c["delegation"] == marker]
    weather_created = next((c for c in created if c["delegation"] == weather), None)
    weather_reply = " ".join(r["text"] for r in rec.all("reply") if r["delegation"] == weather)
    head = weather_reply[:20]
    probe = "FOUND" if head.startswith(CODE) else "NONE" if head.upper().startswith("NONE") else "OTHER"
    texts = [s["text"] for s in rec.transcripts.values()]
    return {
        "role": role,
        "marker_delegation": marker,
        "marker_replies": [r["text"] for r in rec.all("reply") if marker and r["delegation"] == marker],
        "weather_delegation": weather,
        "weather_reply": weather_reply[:200],
        "probe": probe if weather else "NO_DELEGATION",
        "chain_previous": weather_created["previous"] if weather_created else None,
        "marker_response_ids": marker_ids,
        "chain_to_marker": bool(weather_created and weather_created["previous"] in marker_ids),
        "code_in_transcript": any(CODE[-3:] in t for t in texts),  # 코드가 전사로 샜으면 프로브 판정은 오염
        "api_errors": [e["message"] for e in rec.all("api_error")],
        "assistant": rec.said(),
        "user": rec.said("user"),
        "all_responses": [{k: c[k] for k in ("delegation", "id", "previous")} for c in created],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=["developer", "user", "none"], required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    instructions_mod.DEFAULT_BACKEND_INSTRUCTIONS += PROBE  # 세션 생성 때 build_backend_instructions 가 읽는다
    h._feed = feed  # type: ignore[assignment]
    comps, out_dir = h.setup("backend_item_persistence", args.tag or args.role)
    record_raw()
    greeting = h.tts("안녕")
    weather = h.tts("오늘 서울 날씨 어때?")

    def hook(loop: Any, now: float, rec: h.Recorder) -> None:
        step = rec.first("marker_step")
        if step is None:
            last = loop._last_voice_time
            if last is None or last <= (rec.connected_at or 0.0) or now - last < 1.0 or loop._live._active_responses:
                return
            if args.role != "none":
                item = {"type": "message", "role": args.role, "content": [{"type": "input_text", "text": MARKER_NOTE}]}
                loop._live._conn.response.item.create(item=item, event_id=f"{MARKER_EVENT}_item")
                loop._live._conn.response.create(event_id=MARKER_EVENT)
            rec.mark("marker_step", sent=args.role != "none")
            return
        if rec.first("weather_queued") is not None:
            return
        if step["sent"]:
            marker = next((d["id"] for d in rec.all("deleg") if d["client_event_id"] == MARKER_EVENT), None)
            done = next((d for d in rec.all("resp_done") if marker and d["delegation"] == marker), None)
            if done is None or now - done["t"] < 2.0:
                return
        elif now - step["t"] < 3.0:
            return
        rec.inputs.put(weather)  # type: ignore[attr-defined]
        rec.mark("weather_queued")

    def stop_when(rec: h.Recorder, loop: Any, now: float) -> bool:
        queued = rec.first("weather_queued")
        if queued is None:
            return False
        weather_delegs = [d["id"] for d in rec.all("deleg", after=queued["t"]) if not d["client_event_id"]]
        done = next((d for d in rec.all("resp_done") if d["delegation"] in weather_delegs), None)
        return (done is not None and now - done["t"] > 1.5) or now - queued["t"] > 30.0

    results = []
    for rnd in range(args.rounds):
        rec = h.Recorder(f"#{rnd}")
        rec.frame_hook = lambda loop, now, rec=rec: hook(loop, now, rec)
        h.run_session(comps, rec, speech=greeting, stop_when=stop_when, max_sec=75.0)
        row = {"round": rnd, **summarize(rec, args.role)}
        results.append(row)
        h.save_json(out_dir / "results.json", results)
        print(
            f"[{args.role} #{rnd}] probe={row['probe']} chain_to_marker={row['chain_to_marker']} "
            f"leak={row['code_in_transcript']} errors={len(row['api_errors'])} reply={row['weather_reply'][:60]!r}",
            flush=True,
        )
        time.sleep(1.5)
    print(f"results: {out_dir}")


if __name__ == "__main__":
    main()
