"""라디오 추적 트리거 비용 실험 — 운영 루프의 긴 라디오(exp_radio_pipeline ``long``)에 출력 전사 트리거를 걸어 잰다.

라디오가 on 이 되면 ``session.update`` 로 출력 전사 트리거를 걸고 paused·off 면 끈다(``--trigger-at start`` 면 세션
시작부터 계속). 트리거 질문은 둘을 한 번에 평가한다 — 지금 읽는 대사에서 어디까지 왔나(choice), 대사를 읽는 중인가
(predicate). 판정 문맥은 ``decision.item`` 하나로 "지금 읽는 대사(+ 이미 넣은 다음 대사)"를 두고, Tracker 의 끝낸 대사
수·넣은 대사 수가 바뀔 때 새로 넣고 앞 것을 지운다. 진행은 운영 Tracker 가 그대로 하고 트리거 답은 기록만 한다.

재는 것: 평가 수(전사 조각 대비), 평가당 토큰과 세션 시간에 따른 증가, 원인 조각 도착 → 판정 도착 지연, 순서 뒤바뀜,
Live 연결 시간·백엔드 토큰. 비용은 토큰·시간에 단가를 곱한다 — 트리거 판정 단가는 문서에 없어 Decisions API 단가
(입력 $0.10/1M, 출력 무료)로 가정한다.

    uv run --with openai==3.35.1 python scripts/gpt_live/exp_radio_trigger.py --tag t1 --index 0
    uv run python scripts/gpt_live/exp_radio_trigger.py --report t1

결과: var/experiments/radio_trigger/<번호>_<태그>/long_<회차>/ — run.json(라디오 기록 + ``trigger``), run.wav(왼쪽
사용자, 오른쪽 모델) + run.labels.txt, timeline.txt(--report 가 쓰는 조각별 판정), pipeline.log. 프로세스마다 하위
폴더 하나라 병렬로 돌린다(하네스는 프로세스당 약 850 MB).
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import exp_radio_pipeline as epp
import pipeline_harness as h
from exp_dirs import find_result_dir

from voice_pipeline.adapters import gpt_live
from voice_pipeline.engines.gpt_live.loop import LiveSessionLoop
from voice_pipeline.settings import BRIDGE_SAMPLE_RATE

EXPERIMENT = "radio_trigger"

# 1M 토큰당 달러 (pricing 페이지·Decisions 가이드, 2026-10-19)
PRICE_LIVE_PER_MIN = 0.05
PRICE_BACKEND = {"input": 0.75, "cached": 0.075, "output": 4.50}  # gpt-5.4-mini
PRICE_DECISIONS_INPUT = 0.10  # 트리거 판정도 이 단가라고 가정한다

TRIGGER: dict[str, Any] = {
    "type": "decisions",
    "event_type": "session.output_transcript.delta",
    "decisions": {
        "questions": [
            {
                "type": "choice",
                "name": "progress",
                "instructions": (
                    "Ray, a small desk robot, is reading prepared radio lines aloud. The latest application message "
                    "gives the line Ray is reading now. From Ray's latest speech in the transcript, how far has Ray "
                    "got in that line?"
                ),
                "choices": [
                    {"value": "not_started", "description": "Ray has not started saying this line yet."},
                    {"value": "middle", "description": "Ray is saying this line, before its last sentence."},
                    {"value": "last_sentence", "description": "Ray is saying the last sentence of this line."},
                    {"value": "finished", "description": "Ray has said the whole line."},
                ],
            },
            {
                "type": "predicate",
                "name": "reading",
                "instructions": (
                    "Ray's latest words come from the given line, read as written or nearly so, rather than a reply "
                    "to the user or other talk."
                ),
            },
        ]
    },
}


def context_text(lines: tuple[str, ...]) -> str:
    """판정 문맥 — 넣었지만 아직 다 읽지 않은 대사들 (보통 하나, 다음 대사를 넣은 직후엔 둘)."""
    text = f"Line Ray is reading now: «{lines[0]}»"
    if len(lines) > 1:
        text += "\nNext line, already given to Ray: «" + " ".join(lines[1:]) + "»"
    return text


# ---------------------------------------------------------------------------
# 트리거 제어·기록
# ---------------------------------------------------------------------------


class TriggerControl:
    """프레임마다 라디오 상태를 보고 트리거와 판정 문맥을 맞춘다. 받은 이벤트는 :meth:`on_event` 로 기록한다.

    보내기는 루프 스레드(프레임 훅)에서, 기록은 수신 스레드에서 한다 — 기록은 목록에 덧붙이기만 한다.
    """

    def __init__(self, always: bool) -> None:
        self.always = always
        self.on = always
        self.loop: LiveSessionLoop | None = None
        self.ctx = ""
        self.item_event: str | None = None  # 지금 문맥 항목을 만든 event_id
        self.item_ids: dict[str, str] = {}  # 만든 event_id → 서버 item id
        self.pending_delete: list[str] = []
        self._n = 0
        self.log: dict[str, list[dict[str, Any]]] = {
            k: [] for k in ("deltas", "decisions", "sent", "acks", "errors", "usage", "backend")
        }

    def _eid(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}_{self._n}"

    def _send(self, kind: str, fn: Any, **kwargs: Any) -> None:
        fields = {k: v for k, v in kwargs.items() if k not in ("item", "session")}
        self.log["sent"].append({"t": time.monotonic(), "kind": kind, **fields})
        try:
            fn(**kwargs)
        except Exception as exc:  # noqa: BLE001 — 실험 기록
            self.log["errors"].append({"t": time.monotonic(), "code": "send_failed", "message": str(exc)[:300]})

    def step(self, loop: LiveSessionLoop, now: float) -> None:
        self.loop = loop
        conn = loop._live._conn
        if conn is None:
            return
        r = loop._radio
        want = self.always or (r is not None and r.status == "on")
        if want != self.on:
            triggers = [TRIGGER] if want else []
            self._send(
                "triggers_on" if want else "triggers_off",
                conn.session.update,
                session={"triggers": triggers},
                event_id=self._eid("trig"),
            )
            self.on = want
        desired = ""
        if r is not None and r.status == "on" and r.tracker is not None:
            tr = r.tracker
            unread = tr.lines[tr.done : tr.pushed]
            desired = context_text(unread) if unread else ""
        if desired != self.ctx:
            if self.item_event is not None:
                self.pending_delete.append(self.item_event)
                self.item_event = None
            if desired:
                eid = self._eid("ctx")
                item = {"type": "message", "role": "user", "content": desired}
                self._send("item_create", conn.decision.item.create, item=item, event_id=eid)
                self.log["sent"][-1]["text"] = desired
                self.item_event = eid
            self.ctx = desired
        for eid in list(self.pending_delete):
            item_id = self.item_ids.get(eid)
            if item_id is not None:
                self._send("item_delete", conn.decision.item.delete, item_id=item_id, event_id=self._eid("del"))
                self.pending_delete.remove(eid)

    def on_event(self, event: Any) -> None:
        t = time.monotonic()
        etype = getattr(event, "type", None)
        if etype == "session.output_transcript.delta":
            self.log["deltas"].append(
                {"t": t, "id": event.event_id, "text": event.delta, "start_ms": event.start_ms, "end_ms": event.end_ms}
            )
        elif etype == "decision.event":
            tracker = self.loop._radio.tracker if self.loop is not None and self.loop._radio is not None else None
            self.log["decisions"].append(
                {
                    "t": t,
                    "trigger_id": event.trigger_id,
                    "delegation_id": event.delegation_id,
                    "source": event.source_event_id,
                    "source_type": event.source_event_type,
                    "decision": event.event.decision.model_dump(mode="json"),
                    # 수신 스레드라 계산하지 않는 내부 값만 읽는다
                    "tracker": {"done": tracker._done, "pushed": len(tracker._pushes)} if tracker else None,
                }
            )
        elif etype == "decision.item.created":
            if event.client_event_id:
                self.item_ids[event.client_event_id] = event.item.id
            self.log["acks"].append({"t": t, "type": etype, "client_event_id": event.client_event_id})
        elif etype in ("decision.item.deleted", "session.updated"):
            ack: dict[str, Any] = {"t": t, "type": etype, "client_event_id": event.client_event_id}
            if etype == "session.updated":
                ack["triggers"] = len(event.session.triggers or [])
            self.log["acks"].append(ack)
        elif etype == "error":
            err = event.error
            self.log["errors"].append(
                {"t": t, "code": err.code, "message": err.message, "client_event_id": err.client_event_id}
            )
        elif etype == "session.usage.updated":
            ratio = event.context_window.usage_ratio if event.context_window is not None else None
            self.log["usage"].append({"t": t, "seconds": event.usage.seconds, "ratio": ratio})
        elif etype == "response.event":
            inner = event.event
            if inner.get("type") == "response.completed":
                resp = inner.get("response") or {}
                self.log["backend"].append({"t": t, "id": resp.get("id"), "usage": resp.get("usage")})


CONTROL: TriggerControl | None = None
_hooked = False


def install_trigger_hooks() -> None:
    """수신 이벤트를 :data:`CONTROL` 에 넘긴다 (한 번만, 동작은 그대로)."""
    global _hooked
    if _hooked:
        return
    _hooked = True
    orig_dispatch = gpt_live.GPTLiveSession._dispatch

    def dispatch(self: gpt_live.GPTLiveSession, event: Any) -> None:
        if CONTROL is not None:
            try:
                CONTROL.on_event(event)
            except Exception:  # noqa: BLE001 — 기록 실패가 세션을 멈추지 않게
                h.logger.warning("trigger record failed", exc_info=True)
        orig_dispatch(self, event)

    gpt_live.GPTLiveSession._dispatch = dispatch  # type: ignore[method-assign]


def install_start_trigger() -> None:
    """``--trigger-at start`` — 세션 시작 설정에 트리거를 넣는다."""
    orig = gpt_live.LiveSessionConfig.to_session_dict

    def to_session_dict(self: gpt_live.LiveSessionConfig) -> dict[str, Any]:
        cfg = orig(self)
        cfg["triggers"] = [TRIGGER]
        return cfg

    gpt_live.LiveSessionConfig.to_session_dict = to_session_dict  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    global CONTROL
    epp.install_radio_hooks()
    comps, out_dir = h.setup(EXPERIMENT, args.tag, subdir=f"long_{args.index}")
    assert comps.radio_picker is not None, "radio scripts not loaded"
    install_trigger_hooks()
    if args.trigger_at == "start":
        install_start_trigger()
    code_dir = out_dir.parent / "code"
    code_dir.mkdir(exist_ok=True)
    for name in ("exp_radio_trigger.py", "exp_radio_pipeline.py"):
        shutil.copy(Path(__file__).with_name(name), code_dir / name)

    rec = h.Recorder(f"long#{args.index}")
    rec.snapshots = []  # type: ignore[attr-defined]
    driver = epp.Driver(rec, SimpleNamespace(scenario="long", at=2.0, songs=args.songs), [], [])
    CONTROL = control = TriggerControl(always=args.trigger_at == "start")

    def frame(loop: LiveSessionLoop, now: float) -> None:
        driver.hook(loop, now)
        control.step(loop, now)

    rec.frame_hook = frame
    h.run_session(
        comps,
        rec,
        speech=h.tts(epp.START_ASK),
        stop_when=lambda rec, loop, now: driver.done,
        max_sec=args.max,
        song_sec=args.song_sec,
        wav_path=out_dir / "run.wav",
    )
    CONTROL = None
    if not driver.done:
        driver.finish("timeout")
    data = {
        "result": driver.result,
        "index": args.index,
        "t0": rec.connected_at or 0.0,
        "t_end": time.monotonic(),
        "events": rec.events,
        "snapshots": rec.snapshots,  # type: ignore[attr-defined]
        "audio": [(t, n, v) for t, n, v in rec.audio],
        "trigger": control.log,
        "trigger_config": TRIGGER,
        "config": {k: v for k, v in vars(args).items() if k != "report"},
    }
    h.save_json(out_dir / "run.json", data)
    labels = [(h.audio_pos(rec, e["t"]), e["name"] + f" {e.get('index', '')}") for e in rec.events]
    labels += [(h.audio_pos(rec, s["t"]), f"{s['speaker']}: {s['text']}") for s in rec.transcripts.values()]
    for d in control.log["decisions"]:
        answers = {a.get("name"): a for a in d["decision"]["answers"]}
        progress = answers.get("progress", {}).get("choice", "?")
        reading = answers.get("reading", {}).get("probability")
        labels.append((h.audio_pos(rec, d["t"]), f"trig {progress} r={reading}"))
    (out_dir / "run.labels.txt").write_text("".join(f"{p:.3f}\t{p:.3f}\t{x}\n" for p, x in sorted(labels)))
    n = len(control.log["decisions"])
    print(f"[long #{args.index}] {driver.result}, 트리거 판정 {n}, 오류 {len(control.log['errors'])} → {out_dir}")


# ---------------------------------------------------------------------------
# 분석
# ---------------------------------------------------------------------------


def _on_spans(data: dict[str, Any]) -> list[tuple[float, float]]:
    """트리거가 켜져 있던 구간 (보낸 시각 기준)."""
    log = data["trigger"]
    end = data["t_end"]
    if data["config"].get("trigger_at") == "start":
        return [(data["t0"], end)]
    spans, start = [], None
    for s in log["sent"]:
        if s["kind"] == "triggers_on" and start is None:
            start = s["t"]
        elif s["kind"] == "triggers_off" and start is not None:
            spans.append((start, s["t"]))
            start = None
    if start is not None:
        spans.append((start, end))
    return spans


def _inside(t: float, spans: list[tuple[float, float]]) -> bool:
    return any(a <= t <= b for a, b in spans)


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """최소제곱 직선 (절편, 기울기)."""
    mx, my = statistics.mean(xs), statistics.mean(ys)
    var = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / var if var else 0.0
    return my - slope * mx, slope


def analyze(data: dict[str, Any]) -> dict[str, Any]:
    log, t0 = data["trigger"], data["t0"]
    spans = _on_spans(data)
    on_sec = sum(b - a for a, b in spans)
    deltas = [d for d in log["deltas"] if _inside(d["t"], spans)]
    voiced = sum(n for t, n, v in data["audio"] if v and _inside(t, spans)) / (BRIDGE_SAMPLE_RATE * 2)
    decisions = [d for d in log["decisions"] if d.get("trigger_id")]
    tokens = [d["decision"]["usage"]["input_tokens"] for d in decisions]
    cached = [d["decision"]["usage"]["input_tokens_details"]["cached_tokens"] for d in decisions]
    output = [d["decision"]["usage"]["output_tokens"] for d in decisions]
    minutes = [(d["t"] - t0) / 60 for d in decisions]
    delta_at = {d["id"]: (i, d["t"]) for i, d in enumerate(log["deltas"])}
    latency = [d["t"] - delta_at[d["source"]][1] for d in decisions if d["source"] in delta_at]
    order = [delta_at[d["source"]][0] for d in decisions if d["source"] in delta_at]
    out_of_order = sum(b < a for a, b in zip(order, order[1:], strict=False))
    live_sec = max((u["seconds"] for u in log["usage"]), default=data["t_end"] - t0)
    backend_cost = 0.0
    for b in log["backend"]:
        u = b.get("usage") or {}
        cached_in = (u.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
        backend_cost += (
            ((u.get("input_tokens") or 0) - cached_in) * PRICE_BACKEND["input"]
            + cached_in * PRICE_BACKEND["cached"]
            + (u.get("output_tokens") or 0) * PRICE_BACKEND["output"]
        ) / 1e6
    progress: dict[str, int] = {}
    reading = []
    for d in decisions:
        for a in d["decision"]["answers"]:
            if a.get("name") == "progress":
                key = a.get("choice") or a.get("type")
                progress[key] = progress.get(key, 0) + 1
            elif a.get("name") == "reading" and a.get("probability") is not None:
                reading.append(a["probability"])
    pushes = [e for e in data["events"] if e["name"] == "radio_push"]
    return {
        "result": data["result"],
        "on_min": on_sec / 60,
        "voiced_min": voiced / 60,
        "deltas": len(deltas),
        "decisions": len(decisions),
        "tokens": tokens,
        "cached": cached,
        "output": output,
        "minutes": minutes,
        "latency": latency,
        "out_of_order": out_of_order,
        "errors": log["errors"],
        "live_sec": live_sec,
        "backend_cost": backend_cost,
        "trigger_cost": sum(tokens) * PRICE_DECISIONS_INPUT / 1e6,
        "progress": progress,
        "reading": reading,
        "pushes": {by: sum(e.get("by") == by for e in pushes) for by in ("start", "text", "stall", "resume")},
        "repush": len([e for e in data["events"] if e["name"] == "radio_repush"]),
        "songs": len([e for e in data["events"] if e["name"] == "song_play"]),
        "acks": log["acks"],
        "sent": log["sent"],
    }


def _answer_text(decision: dict[str, Any]) -> str:
    parts = []
    for a in decision["answers"]:
        if a.get("type") == "choice":
            parts.append(f"{a.get('name')}={a.get('choice')}({a.get('confidence', 0):.2f})")
        elif a.get("type") == "predicate":
            parts.append(f"{a.get('name')}={a.get('probability', 0):.2f}")
        else:
            parts.append(f"{a.get('name')}={a.get('type')}")
    return " ".join(parts)


def timeline(data: dict[str, Any]) -> list[str]:
    """레이 출력 전사 조각을 도착 순으로 늘어놓고, 판정을 일으킨 조각 옆에 그 판정을 단다.

    판정이 붙지 않은 조각은 합쳐져 다음 판정에 들어갔거나 트리거가 꺼져 있던 때다. 문맥 갱신·트리거 켜고 끔·대사
    넣기도 시각 순으로 끼운다. 시각은 연결 뒤 초.
    """
    log, t0 = data["trigger"], data["t0"]
    by_source: dict[str, list[dict[str, Any]]] = {}
    for d in log["decisions"]:
        by_source.setdefault(d["source"], []).append(d)
    known = {x["id"] for x in log["deltas"]}
    rows: list[tuple[float, str]] = []
    for x in log["deltas"]:
        line = f"  «{x['text']}»"
        for d in by_source.get(x["id"], []):
            usage = d["decision"]["usage"]
            line += (
                f"  → 판정 +{d['t'] - x['t']:.2f}s {_answer_text(d['decision'])} 입력 {usage['input_tokens']}"
                f" (Tracker 끝낸 {d['tracker']['done'] if d['tracker'] else '-'}"
                f"/넣은 {d['tracker']['pushed'] if d['tracker'] else '-'})"
            )
        rows.append((x["t"], line))
    for d in log["decisions"]:
        if d["source"] not in known:
            rows.append((d["t"], f"  ? 원인 조각 모름 ({d['source']}) {_answer_text(d['decision'])}"))
    for s in log["sent"]:
        if s["kind"] == "item_create":
            rows.append((s["t"], f"[문맥] {s.get('text', '')}".replace("\n", " / ")))
        elif s["kind"].startswith("triggers"):
            rows.append((s["t"], f"[{s['kind']}]"))
    for e in data["events"]:
        if e["name"] == "radio_push":
            rows.append((e["t"], f"[대사 {e['index'] + 1} 넣음 ({e['by']})]"))
        elif e["name"] == "song_play":
            rows.append((e["t"], "[곡]"))
    return [f"{t - t0:7.2f} {text}" for t, text in sorted(rows, key=lambda r: r[0])]


def report(tag: str) -> None:
    root = find_result_dir(EXPERIMENT, tag)
    if root is None:
        raise SystemExit(f"no result for tag {tag!r}")
    runs = []
    for path in sorted(root.glob("long_*/run.json")):
        data = json.loads(path.read_text())
        (path.parent / "timeline.txt").write_text("\n".join(timeline(data)) + "\n")
        runs.append((path.parent.name, analyze(data)))
    if not runs:
        raise SystemExit("no runs")
    print(f"{root.name}: 회차 {len(runs)}")
    for name, a in runs:
        print(f"\n## {name} — {a['result']}, 곡 {a['songs']}, 넣기 {a['pushes']}, 다시 넣기 {a['repush']}")
        updates = [s for s in a["sent"] if s["kind"].startswith("triggers")]
        update_acks = [x for x in a["acks"] if x["type"] == "session.updated"]
        print(
            f"  트리거 갱신 보냄 {len(updates)}, session.updated {len(update_acks)} "
            f"(트리거 수 {[x.get('triggers') for x in update_acks]}), 오류 {len(a['errors'])}"
        )
        for e in a["errors"][:5]:
            print(f"    오류 {e.get('code')}: {e.get('message')} ({e.get('client_event_id')})")
        if not a["decisions"]:
            print("  트리거 판정 없음")
            continue
        print(
            f"  켜진 시간 {a['on_min']:.1f}분 (모델 말소리 {a['voiced_min']:.1f}분), 전사 조각 {a['deltas']}, 판정 "
            f"{a['decisions']} (조각당 {a['decisions'] / max(1, a['deltas']):.2f}, 켜진 분당 "
            f"{a['decisions'] / a['on_min']:.0f}, 말소리 분당 {a['decisions'] / max(0.01, a['voiced_min']):.0f})"
        )
        t = a["tokens"]
        intercept, slope = _fit(a["minutes"], t) if len(t) > 1 else (t[0], 0.0)
        print(
            f"  입력 토큰/판정 평균 {statistics.mean(t):.0f} (p50 {statistics.median(t):.0f}, 최소 {min(t)}, 최대 "
            f"{max(t)}), 캐시 평균 {statistics.mean(a['cached']):.0f}, 출력 합 {sum(a['output'])} | 시간 따라 "
            f"{intercept:.0f} + {slope:.0f}/분"
        )
        if a["latency"]:
            lat = a["latency"]
            print(
                f"  지연(원인 조각 → 판정) p50 {statistics.median(lat):.2f}s p90 {_pct(lat, 0.9):.2f}s 최대 "
                f"{max(lat):.2f}s, 순서 뒤바뀜 {a['out_of_order']}"
            )
        reading = statistics.mean(a["reading"]) if a["reading"] else 0.0
        print(f"  답: progress {a['progress']}, reading 평균 {reading:.2f}")
        print(
            f"  비용: Live {a['live_sec'] / 60:.1f}분 ${a['live_sec'] / 60 * PRICE_LIVE_PER_MIN:.3f}, 백엔드 "
            f"${a['backend_cost']:.4f}, 트리거 ${a['trigger_cost']:.4f} (가정 단가)"
        )

    usable = [a for _, a in runs if a["decisions"]]
    if not usable:
        return
    per_min = statistics.mean(a["decisions"] / a["on_min"] for a in usable)
    xs = [m for a in usable for m in a["minutes"]]
    ys = [v for a in usable for v in a["tokens"]]
    intercept, slope = _fit(xs, ys)
    print(f"\n## 추정 — 켜진 분당 판정 {per_min:.0f}회, 판정당 입력 토큰 {intercept:.0f} + {slope:.0f}×(세션 분)")
    for minute in (5, 10, 30):
        cost = per_min * (intercept + slope * minute) * PRICE_DECISIONS_INPUT / 1e6
        total = sum(per_min * (intercept + slope * (m + 0.5)) for m in range(minute)) * PRICE_DECISIONS_INPUT / 1e6
        print(
            f"  {minute:2d}분째: 트리거 분당 ${cost:.4f} (Live 분당 ${PRICE_LIVE_PER_MIN:.2f}의 "
            f"{cost / PRICE_LIVE_PER_MIN:.0%}), {minute}분 누적 트리거 ${total:.3f} "
            f"vs Live ${minute * PRICE_LIVE_PER_MIN:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--trigger-at",
        choices=("radio", "start"),
        default="radio",
        help="radio: 라디오 on 동안만 session.update 로 / start: 세션 시작부터 계속",
    )
    parser.add_argument("--songs", type=int, default=2, help="곡을 이만큼 튼 뒤 끝낸다")
    parser.add_argument("--song-sec", type=float, default=15.0)
    parser.add_argument("--max", type=float, default=600.0, help="세션 상한 초")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--report", metavar="TAG")
    args = parser.parse_args()
    if args.report:
        report(args.report)
        return
    run(args)


if __name__ == "__main__":
    main()
