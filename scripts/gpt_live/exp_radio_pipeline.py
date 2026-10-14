"""라디오 운영 경로 실험 — wiring 으로 조립한 운영 루프(라디오 포함)를 합성 음성·C++ 스텁으로 돌린다.

지시문·루프·라디오·판정기·백엔드는 운영 그대로다(:mod:`pipeline_harness`). 사용자는 TTS 로 "라디오 해 줘." 하고
시작한다. 라디오의 넣기·다시 넣기·판정·곡은 기록 훅으로만 남긴다(동작은 바꾸지 않는다).

시나리오(``--scenario``):

- ``long``: 끼어들기 없이 대본 둘 — 곡 소개 → 곡(스텁 ``--song-sec``) → 다음 대본 → 곡까지.
- ``interrupt``: 첫 대본(``--first``)의 ``--at-line`` 번째 대사를 읽기 시작하고 ``--at`` 초 뒤, 모델이 말하는 중에
  끼어든다(``exp_radio_flow.UTTERANCES`` 의 발화). paused 면 대답이 끝난 뒤 ``FOLLOW_UP``, 이어 재개 요청(회차마다
  "라디오 다시 이어서 해 줘." / "이야기 계속해 줘." 번갈아). 재개 뒤 곡이 시작될 때까지 지켜본다.
- ``multi``: 한 세션에서 끼어들기·재개를 ``--cycles`` 번 — 대사 ``MULTI_LINES`` 번째에서 질문 → 대답이 끝나면 재개
  요청(번갈아). 마지막 재개 뒤 곡까지.
- ``chat``: ``--at-line`` 번째 대사에서 레이에게 직접 묻고, 멈춘 뒤 재개 없이 대화를 이어 간다(``CHAT_TURNS``, 대답이
  끝날 때마다 하나씩). 멈춘 뒤 대화가 라디오와 상관없이 평소처럼 흘러가는지 본다.
- ``window``: 사용자 목소리 시작 → 첫 사용자 전사(보류) 사이의 틈(약 1.2~1.4 s)에 대사가 들어가게 끼어든다. 앞뒤 무음을
  자른 발화를 넣고 ``--watch`` 초 지켜본 뒤 끝낸다(재개 없음).
  ``--window text``: ``--at-line`` 번째 이후 대사의 넣기 지점(마지막 문장 끝 ``PUSH_REMAIN`` 자)까지 ``--lead`` 자
  남았을 때 끼어든다 — 평소 넣기가 틈에 걸린다.
  ``--window stall``: ``--at-line`` 번째 대사 뒤의 앞서 넣기를 막아(실험 훅) 받아 둔 대사 없이 조용해지게 하고, 조용해진
  지 ``--quiet`` 초에 끼어든다 — 무음 폴백(``STALL_SEC``)의 넣기가 틈에 걸린다.
  ``--text``/``--label`` 로 ``UTTERANCES`` 밖의 문장을 넣을 수 있다. 리포트는 운영 라디오가 넣은 "대답 먼저" 문구
  (``RADIO_LATE_LINE_NOTE``)의 시각도 보인다.

    uv run python scripts/gpt_live/exp_radio_pipeline.py --scenario long --tag p1 --index 0
    uv run python scripts/gpt_live/exp_radio_pipeline.py --scenario interrupt --tag p2 --utterance content_key
    uv run python scripts/gpt_live/exp_radio_pipeline.py --scenario window --window stall --utterance what --tag w1
    uv run python scripts/gpt_live/exp_radio_pipeline.py --report p1

결과: var/experiments/radio_pipeline/<번호>_<태그>/<시나리오>_<발화>_<회차>/ — run.json(이벤트·전사 조각·넣기·판정),
run.wav(왼쪽 사용자, 오른쪽 모델) + run.labels.txt, pipeline.log. 프로세스마다 하위 폴더 하나라 병렬로 돌린다.
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pipeline_harness as h
import radio_replay as rr
from exp_dirs import find_result_dir
from exp_radio_flow import FOLLOW_UP, RESUME_ASKS, UTTERANCES

from voice_pipeline.engines.gpt_live import radio
from voice_pipeline.engines.gpt_live.instructions import RADIO_LATE_LINE_NOTE
from voice_pipeline.engines.gpt_live.loop import LiveSessionLoop

START_ASK = "라디오 해 줘."
START_TIMEOUT_SEC = 12.0  # 요청 목소리 끝 뒤 이만큼 start_radio 가 없으면 진입 실패
REPLY_QUIET_SEC = 1.5  # 모델 대답이 끝나고 이만큼 조용하면 사용자가 다음 말을 한다
REPLY_MAX_WAIT_SEC = 40.0  # 깊은 질문엔 대답이 20 s 를 넘기도 한다 — 끊고 들어가지 않게 넉넉히
RESUME_WAIT_SEC = 15.0  # 재개 요청 뒤 이만큼 라디오가 on 이 아니면 재개 실패
WATCH_MAX_SEC = 150.0  # 재개 뒤 곡까지 지켜보는 상한
MULTI_LINES = (3, 7, 11)  # multi: 이 대사(1부터)를 읽기 시작하고 --at 초 뒤 끼어든다
MULTI_QUESTIONS = ("잠깐만, 너는 어떻게 생각해?", "근데 그게 무슨 뜻이야?", "아 그거 나도 궁금했는데, 왜 그런 거야?")
SPEAKING_SEC = 0.2  # 이 안에 모델 말소리가 있었으면 말하는 중
TRIM_PAD_SEC = 0.1  # window: 발화 앞뒤로 남기는 무음 — 넣는 순간 ≈ 목소리 시작
TRIM_RMS = 200  # window: 이 rms(20 ms) 이상이 목소리
CHAT_INTERRUPT = "근데 너는 버리지 못하는 물건 같은 거 있어?"
CHAT_TURNS = (
    "나는 옛날 편지를 아직도 못 버리겠더라. 그걸 다시 읽으면 좀 슬퍼지거든.",
    "사람들은 왜 지난 일을 붙잡고 있는 걸까? 그냥 잊으면 편할 텐데.",
    "너는 기억이라는 게 뭐라고 생각해? 너한테도 기억이 있는 거야?",
    "그럼 우리가 오늘 나눈 얘기도 나중에 기억할 수 있어?",
    "고마워. 덕분에 생각이 좀 정리됐어.",
)


# ---------------------------------------------------------------------------
# 기록 훅
# ---------------------------------------------------------------------------

_hooked = False


def install_radio_hooks() -> None:
    """라디오의 전사 입력·넣기·판정·다시 넣기·곡 끝을 기록한다 (한 번만, 동작은 그대로)."""
    global _hooked
    if _hooked:
        return
    _hooked = True

    def wrap(cls: Any, name: str, mark: Any) -> None:
        orig = getattr(cls, name)

        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            out = orig(self, *args, **kwargs)
            if h.CURRENT is not None:
                mark(h.CURRENT, self, args, kwargs, out)
            return out

        setattr(cls, name, wrapper)

    def snapshot(rec: h.Recorder, _self: Any, args: tuple, _kw: dict, _out: Any) -> None:
        now, segment_id, speaker, text = args
        rec.snapshots.append((now, segment_id, speaker, text))  # type: ignore[attr-defined]

    wrap(radio.Radio, "on_transcript", snapshot)
    wrap(radio.Radio, "_push", lambda rec, s, a, k, o: rec.mark("radio_push", index=a[0].index, by=a[0].by))
    wrap(radio.Radio, "_pause", lambda rec, s, a, k, o: rec.mark("radio_decision", state="paused", applied=bool(o)))
    wrap(radio.Radio, "_release", lambda rec, s, a, k, o: rec.mark("radio_release", why=a[1]))
    wrap(
        radio.Radio,
        "_begin_script",
        lambda rec, s, a, k, o: rec.mark("radio_script", key=a[0].key, lines=list(s.tracker.lines)),
    )
    wrap(radio.Tracker, "mark_repushed", lambda rec, s, a, k, o: rec.mark("radio_repush"))
    wrap(radio.Radio, "on_song_end", lambda rec, s, a, k, o: rec.mark("radio_song_end", completed=k.get("completed")))
    wrap(
        radio.Radio,
        "start",
        lambda rec, s, a, k, o: rec.mark("radio_start", output=o[0], script=s.script.key if s.script else None),
    )
    wrap(radio.Radio, "resume", lambda rec, s, a, k, o: rec.mark("radio_resume", output=o[0]))


_block_push: int | None = None  # window stall: 이 번호(0부터) 대사는 앞서 넣지 않는다 — 무음 폴백으로만 들어간다


def install_block_hook() -> None:
    """``_block_push`` 대사의 글자 일치 넣기를 막는다 (실험 훅). 무음 폴백 분기에 닿도록 넣기 문턱만 끈다."""
    orig = radio.Tracker.next_action

    def next_action(self: radio.Tracker, t: float, quiet: float) -> radio.Action | None:
        if _block_push is not None and self.pushed == _block_push:
            self.PUSH_REMAIN = -1  # type: ignore[misc]
        else:
            self.__dict__.pop("PUSH_REMAIN", None)
        return orig(self, t, quiet)

    radio.Tracker.next_action = next_action  # type: ignore[method-assign]


def trim_voice(pcm: bytes) -> bytes:
    """16 kHz 발화의 앞뒤 무음을 자르고 ``TRIM_PAD_SEC`` 만 남긴다 (TTS 는 앞에 1 s 넘는 무음을 붙이기도 한다)."""
    hop = int(0.02 * 16000)
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    n = len(x) // hop
    idx = np.flatnonzero(np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1)) >= TRIM_RMS)
    if not len(idx):
        return pcm
    pad = int(TRIM_PAD_SEC / 0.02)
    a, b = max(0, int(idx[0]) - pad) * hop, min(n, int(idx[-1]) + 1 + pad) * hop
    return pcm[a * 2 : b * 2]


# ---------------------------------------------------------------------------
# 진행
# ---------------------------------------------------------------------------


class Driver:
    """프레임마다 라디오 상태를 보고 사용자 말을 넣는다. ``plan`` 은 끼어들기 차례 [(대사 번호, 말)]."""

    def __init__(self, rec: h.Recorder, args: argparse.Namespace, plan: list[tuple[int, str]], resume_asks: list[str]):
        self.rec, self.args = rec, args
        self.plan = list(plan)
        self.resume_asks = list(resume_asks)
        self.state = "start"
        self.line_seen: float | None = None
        self.since = 0.0
        self.last_voice = 0.0
        self.audio_i = 0
        self.speech_end = 0.0
        self.done = False
        self.result = "running"
        self.songs = 0
        self.chat = list(CHAT_TURNS) if args.scenario == "chat" else []

    def say(self, text: str, now: float, clip: bytes | None = None) -> None:
        clip = clip if clip is not None else h.tts(text)
        self.rec.inputs.put(clip)
        self.rec.mark("user_say", text=text)
        self.speech_end = now + len(clip) / (16000 * 2) + 0.1

    def _voice(self, now: float) -> None:
        audio = self.rec.audio
        for t, _, voiced in audio[self.audio_i :]:
            if voiced:
                self.last_voice = t
        self.audio_i = len(audio)

    def _quiet_after(self, start: float, now: float) -> bool:
        """``start`` 뒤 모델이 말을 했고 그 뒤 조용하다 (또는 상한)."""
        spoke = self.last_voice > start
        return (spoke and now - self.last_voice >= REPLY_QUIET_SEC) or now - start >= REPLY_MAX_WAIT_SEC

    def hook(self, loop: LiveSessionLoop, now: float) -> None:
        self._voice(now)
        r = loop._radio
        if r is None or self.done:
            return
        songs = len(self.rec.all("song_play"))
        if self.state == "start":
            if self.rec.first("radio_start") is not None:
                self.state = "reading"
            elif self.rec.connected_at is not None and now - self.rec.connected_at > 2.0 + START_TIMEOUT_SEC:
                self.finish("no_start")
            return
        if self.args.scenario == "window":
            self._window(r, now)
            return
        if self.state == "reading":
            if self.plan:
                line, text = self.plan[0]
                tr = r.tracker
                reading = r.status == "on" and tr is not None and tr.pushed >= line and tr.done >= line - 1
                if reading and self.line_seen is None:
                    self.line_seen = now
                speaking = now - self.last_voice < SPEAKING_SEC
                if self.line_seen is not None and now - self.line_seen >= self.args.at and speaking:
                    self.plan.pop(0)
                    self.say(text, now)
                    self.state, self.since, self.line_seen = "interrupted", now, None
            elif songs >= self.args.songs:
                self.finish("done")
            if self.plan and songs >= 1:
                self.finish("missed")  # 끼어들 대사를 지나 곡까지 갔다
            elif now - self.since > WATCH_MAX_SEC and self.since:
                self.finish("watch_timeout")
            return
        if self.state == "interrupted":  # 판정을 기다린다 — 사용자 말이 끝난 뒤
            if r.status == "paused":
                self.state, self.since = "paused", now
            elif r.status == "off":
                self.state, self.since = "ended", now
            elif self.rec.all("radio_release", after=self.since):
                self.state, self.since = "reading", now  # on — 그대로 진행
            elif now - self.since > 20.0:
                self.finish("no_decision")
            return
        if self.state == "paused" and self.args.scenario == "chat":
            if r.status == "on":
                self.rec.mark("radio_back_on")  # 재개 요청 없이 라디오가 돌아왔다
            if now >= self.speech_end and self._quiet_after(max(self.since, self.speech_end), now):
                if not self.chat:
                    self.finish("done")
                    return
                self.say(self.chat.pop(0), now)
                self.since = now
            return
        if self.state == "paused":
            if now >= self.speech_end and self._quiet_after(max(self.since, self.speech_end), now):
                if self.args.scenario == "interrupt" and not self.rec.all("user_say", after=self.since):
                    self.say(FOLLOW_UP, now)
                    self.since = now
                    return
                ask = self.resume_asks.pop(0) if self.resume_asks else RESUME_ASKS[0]
                self.say(ask, now)
                self.state, self.since = "resume_ask", now
            return
        if self.state == "resume_ask":
            if r.status == "on":
                self.state, self.since = "reading", now
            elif now - self.since > RESUME_WAIT_SEC:
                self.finish("no_resume")
            return
        if self.state == "ended":
            if now >= self.speech_end and self._quiet_after(self.since, now):
                self.finish("ended")

    def _window(self, r: radio.Radio, now: float) -> None:
        if self.state == "watch":
            if now - self.since >= self.args.watch:
                self.finish("done")
            return
        if self.rec.all("song_play"):
            self.finish("missed")  # 끼어들 자리를 지나 곡까지 갔다
            return
        tr = r.tracker
        if r.status != "on" or tr is None:
            return
        line = self.args.at_line  # 1부터
        if self.args.window == "stall":
            # line 번째 대사를 다 읽었고 다음 대사는 막혀 있다 — 조용해진 지 --quiet 초에 끼어든다
            quiet = now - self.last_voice
            if tr.done >= line and tr.pushed == line and self.last_voice > 0 and quiet >= self.args.quiet:
                self._fire(now, line=line, quiet=round(quiet, 2))
            return
        # text: line 번째 이후 대사를 읽는 중, 넣기 지점까지 --lead 자 남았을 때
        last = tr.pushed - 1
        if last < line - 1 or tr.done != last or now - self.last_voice >= SPEAKING_SEC:
            return
        reach, _ = tr._progress()
        remain = tr._offsets[last + 1] - reach
        if reach > tr._last_start[last] and tr.PUSH_REMAIN < remain <= tr.PUSH_REMAIN + self.args.lead:
            self._fire(now, line=last + 1, remain=remain)

    def _fire(self, now: float, **info: Any) -> None:
        text = self.args.text or UTTERANCES[self.args.utterance][1]
        self.rec.mark("window_fire", **info)
        self.say(text, now, clip=trim_voice(h.tts(text)))
        self.state, self.since = "watch", now

    def finish(self, result: str) -> None:
        self.result, self.done = result, True
        self.rec.mark("finish", result=result)


def run(args: argparse.Namespace) -> None:
    global _block_push
    install_radio_hooks()
    key = args.label or args.utterance if args.scenario in ("interrupt", "window") else "-"
    scenario = f"{args.scenario}-{args.window}" if args.scenario == "window" else args.scenario
    comps, out_dir = h.setup("radio_pipeline", args.tag, subdir=f"{scenario}_{key}_{args.index}")
    if args.scenario == "window" and args.window == "stall":
        install_block_hook()
        _block_push = args.at_line  # at_line 번째(1부터) 다음 대사의 번호(0부터)
    assert comps.radio_picker is not None, "radio scripts not loaded"
    if args.first:  # 첫 대본을 고정한다 (끼어드는 말이 대본 내용을 묻는다)
        picker = comps.radio_picker
        orig_pick, first = picker.pick, [args.first]

        def pick() -> radio.RadioScript | None:
            if first:
                key = first.pop()
                picker._log.record(key)
                picker._session_used.add(key)
                return picker.scripts[key]
            return orig_pick()

        picker.pick = pick  # type: ignore[method-assign]
    if args.scenario == "interrupt":
        plan = [(args.at_line, UTTERANCES[args.utterance][1])]
        asks = [RESUME_ASKS[args.index % len(RESUME_ASKS)]]
    elif args.scenario == "chat":
        plan, asks = [(args.at_line, CHAT_INTERRUPT)], []
    elif args.scenario == "multi":
        plan = list(zip(MULTI_LINES[: args.cycles], MULTI_QUESTIONS, strict=False))
        asks = [RESUME_ASKS[(args.index + i) % len(RESUME_ASKS)] for i in range(args.cycles)]
    else:
        plan, asks = [], []
    rec = h.Recorder(f"{args.scenario}#{args.index}")
    rec.snapshots = []  # type: ignore[attr-defined]
    driver = Driver(rec, args, plan, asks)
    rec.frame_hook = driver.hook
    h.run_session(
        comps,
        rec,
        speech=h.tts(START_ASK),
        stop_when=lambda rec, loop, now: driver.done,
        max_sec=args.max,
        song_sec=args.song_sec,
        wav_path=out_dir / "run.wav",
    )
    if not driver.done:
        driver.finish("timeout")
    t0 = rec.connected_at or 0.0
    data = {
        "scenario": args.scenario,
        "utterance": key,
        # 운영 판정기는 off 를 내지 않는다 — 그만두자는 말도 paused (exp_radio_flow 의 기대값은 직접 실험 판정기 기준)
        "expected": {"off": "paused"}.get(UTTERANCES[key][2], UTTERANCES[key][2]) if key in UTTERANCES else None,
        "index": args.index,
        "result": driver.result,
        "t0": t0,
        "events": rec.events,
        "snapshots": rec.snapshots,  # type: ignore[attr-defined]
        "audio": [(t, n, v) for t, n, v in rec.audio],
        "user_clips": [(t, len(c)) for t, c in rec.user_clips],
        "scripts": {k: list(s.lines) for k, s in comps.radio_picker.scripts.items()},
        "songs": {k: s.song.key for k, s in comps.radio_picker.scripts.items()},
        "config": {k: v for k, v in vars(args).items() if k not in ("report",)},
    }
    h.save_json(out_dir / "run.json", data)
    labels = [
        (h.audio_pos(rec, e["t"]), e["name"] + (f" {e.get('index', '')}{e.get('state', '')}")) for e in rec.events
    ]
    labels += [(h.audio_pos(rec, s["t"]), f"{s['speaker']}: {s['text']}") for s in rec.transcripts.values()]
    (out_dir / "run.labels.txt").write_text("".join(f"{p:.3f}\t{p:.3f}\t{x}\n" for p, x in sorted(labels)))
    print(f"[{args.scenario} {key} #{args.index}] {driver.result} → {out_dir}")


# ---------------------------------------------------------------------------
# 분석
# ---------------------------------------------------------------------------


def _deltas(snapshots: list[list[Any]]) -> list[tuple[float, str, str]]:
    texts: dict[str, str] = {}
    out = []
    for t, seg, who, text in snapshots:
        old = texts.get(seg, "")
        delta = text[len(old) :] if text.startswith(old) else text
        texts[seg] = text
        if delta.strip():
            out.append((t, who, delta))
    return out


def _segments(data: dict[str, Any]) -> list[dict[str, Any]]:
    """대본마다 {키, 줄(곡 뒤면 첫머리 포함), 시작, 끝(곡 시작·다음 대본·끝), 곡, 넣기}."""
    events = data["events"]
    scripts = [e for e in events if e["name"] == "radio_script"]
    songs = [e for e in events if e["name"] == "song_play"]
    out = []
    for i, sc in enumerate(scripts):
        nxt = scripts[i + 1]["t"] if i + 1 < len(scripts) else float("inf")
        song = next((x for x in songs if sc["t"] < x["t"] < nxt), None)
        end = min(nxt, song["t"] if song else float("inf"))
        pushes = [e for e in events if e["name"] == "radio_push" and sc["t"] <= e["t"] < nxt]
        out.append({"key": sc["key"], "lines": sc["lines"], "t0": sc["t"], "t1": end, "song": song, "pushes": pushes})
    return out


def _gt(data: dict[str, Any], seg: dict[str, Any], t0: float, t1: float, u0: int = 0) -> dict[int, Any]:
    run = SimpleNamespace(deltas=_deltas(data["snapshots"]), units=seg["lines"])
    return rr.span_gt(run, t0, t1, u0, len(seg["lines"]))  # type: ignore[arg-type]


def _voice_after(data: dict[str, Any], t: float) -> float | None:
    return next((a for a, _, v in data["audio"] if v and a > t), None)


def report(tag: str) -> None:
    root = find_result_dir("radio_pipeline", tag)
    if root is None:
        raise SystemExit(f"결과 폴더 없음: {tag}")
    runs = [json.loads(p.read_text()) for p in sorted(root.glob("*/run.json"))]
    entered = [d for d in runs if any(e["name"] == "radio_start" for e in d["events"])]
    print(f"{root.name}: 회차 {len(runs)}, 결과 {dict(collections.Counter(d['result'] for d in runs))}")
    print(f"  진입(start_radio) {len(entered)}/{len(runs)}")
    places: collections.Counter[str] = collections.Counter()
    leads: list[float] = []
    closing, after_song, first_ok = [], [], [0, 0]
    for d in entered:
        users = [e["t"] for e in d["events"] if e["name"] == "user_say" and e["text"] != START_ASK]
        for seg in _segments(d):
            cut = min([u for u in users if u > seg["t0"]] + [seg["t1"]])  # 끼어들기 전까지
            gts = _gt(d, seg, seg["t0"], cut)
            for p in seg["pushes"]:
                if p["by"] in ("text", "stall") and p["t"] < cut:
                    place, lead = rr.place(p["t"], gts.get(p["index"] - 1))
                    places[place] += 1
                    if place == "in" and lead is not None:
                        leads.append(lead)
            g0 = gts.get(0)
            if g0 is not None:
                first_ok[0] += g0.said[-1] >= 0.8 and g0.said[0] >= 0.8
                first_ok[1] += 1
            if seg["song"] is not None:
                g = _gt(d, seg, seg["t0"], seg["t1"]).get(len(seg["lines"]) - 1)
                if g is not None and g.end is not None:
                    closing.append(round(seg["song"]["t"] - g.end, 1))
        for e in d["events"]:
            if e["name"] == "radio_song_end" and e.get("completed"):
                v = _voice_after(d, e["t"])
                if v is not None:
                    after_song.append(round(v - e["t"], 1))
    repush = sum(1 for d in entered for e in d["events"] if e["name"] == "radio_repush")
    stall = sum(1 for d in entered for e in d["events"] if e["name"] == "radio_push" and e["by"] == "stall")
    lead = f"{np.median(leads):.2f}" if leads else "-"
    print(f"  넣기 위치(끼어들기 전): {dict(places)}, 여유 중앙 {lead} s, 무음 넣기 {stall}, 다시 넣기 {repush}")
    print(f"  첫 대사 그대로(80%) {first_ok[0]}/{first_ok[1]}")
    if closing:
        print(f"  곡 소개 끝 → 곡 {closing} (먼저 {sum(c < 0 for c in closing)})")
    if after_song:
        print(f"  곡 끝 → 다음 대본 첫 음성 {after_song}")
    inter = [d for d in entered if d["scenario"] in ("interrupt", "multi")]
    if inter:
        _report_interrupts(inter)
    window = [d for d in entered if d["scenario"] == "window"]
    if window:
        _report_window(window)
    for d in entered:
        if d["scenario"] == "chat":
            _report_chat(d)


def _report_window(runs: list[dict[str, Any]]) -> None:
    """틈 끼어들기 — 목소리 시작 기준으로 넣기·첫 사용자 전사(보류)·판정, 그 뒤 레이 말."""
    print("  틈 끼어들기 (목소리 시작 기준 s, 넣기가 첫 사용자 전사보다 먼저면 틈 안)")
    for d in runs:
        ev = d["events"]
        fire = next((e for e in ev if e["name"] == "window_fire"), None)
        if fire is None:
            print(f"    {d['config']['window']} {d['utterance']} #{d['index']}: 끼어들지 못함 ({d['result']})")
            continue
        clip_t, clip_n = next(((t, n) for t, n in d["user_clips"] if t >= fire["t"] - 0.05), (fire["t"], 0))
        v0, v1 = clip_t + TRIM_PAD_SEC, clip_t + clip_n / (16000 * 2) - TRIM_PAD_SEC
        push = next((e for e in ev if e["name"] == "radio_push" and e["t"] > fire["t"] - 0.05), None)
        deltas = _deltas(d["snapshots"])
        hold = next((t for t, who, _ in deltas if who == "user" and t > v0), None)
        dec = next((e for e in ev if e["name"] in ("radio_decision", "radio_release") and e["t"] > v0), None)
        note = next(
            (e for e in ev if e["name"] == "append_thinking" and e.get("text") == RADIO_LATE_LINE_NOTE and e["t"] > v0),
            None,
        )
        overlap = sum(n / (24000 * 2) for t, n, v in d["audio"] if v and v0 <= t < v1)
        said = "".join(x for t, who, x in deltas if who == "assistant" and t > v0).strip()

        def rel(t: float | None, v0: float = v0) -> str:
            return f"{t - v0:+.2f}" if t is not None else "-"

        inside = push is not None and hold is not None and push["t"] < hold
        line = d["scripts"][d["config"]["first"]][push["index"]] if push else ""
        decision = f"{dec['name'].split('_')[1]} {rel(dec['t'])}" if dec else "-"
        print(
            f"    {d['config']['window']} {d['utterance']} #{d['index']} {fire.get('remain', fire.get('quiet'))}: "
            f"넣기 {rel(push['t'] if push else None)}({push['by'] if push else '-'}) 전사 {rel(hold)} "
            f"{'틈 안' if inside else '틈 밖'} | 판정 {decision} | 겹침 {overlap:.1f}s / 목소리 {v1 - v0:.1f}s"
            + (f" | 대답 먼저 문구 {rel(note['t'])}" if note else "")
        )
        print(f"      넣은 대사: {line}")
        print(f"      레이: {said[:160]}")


def _report_chat(d: dict[str, Any]) -> None:
    """멈춘 뒤 대화 — 차례마다 사용자·레이 전사, 위임·툴, 대본 낭독·방송 어휘."""
    ev = d["events"]
    pause = next((e for e in ev if e["name"] == "radio_decision"), None)
    print(f"  대화 ({d['result']}) — 판정 {pause['state'] if pause else '없음'}")
    t0 = pause["t"] - 6.0 if pause else 0.0
    turns: list[list[Any]] = []
    for t, who, delta in _deltas(d["snapshots"]):
        if t < t0:
            continue
        if turns and turns[-1][0] == who:
            turns[-1][1] += delta
        else:
            turns.append([who, delta, t])
    seg = _segments(d)[0]
    script = rr._letters("".join(seg["lines"]))
    for who, text, t in turns:
        calls = [e["fn"] for e in ev if e["name"] == "backend_call" and t - 1 <= e["t"] < t + 8]
        said = rr._letters(text)
        blocks = rr.difflib.SequenceMatcher(None, said, script, autojunk=False).get_matching_blocks()
        own = sum(b.size for b in blocks if b.size >= 8)
        flags = f"  [대본 {own}자]" if who == "assistant" and own else ""
        flags += f"  [툴 {calls}]" if calls and who == "user" else ""
        print(f"    {'사용자' if who == 'user' else '레이  '}: {text.strip()}{flags}")
    back = [e for e in ev if e["name"] == "radio_back_on"]
    print(f"    위임 {sum(1 for e in ev if e['name'] == 'delegation_created' and e['t'] > t0)}회, "
          f"재개 요청 없이 라디오 복귀 {len(back) > 0}")  # fmt: skip


def _report_interrupts(runs: list[dict[str, Any]]) -> None:
    print("  끼어들기:")
    judged: collections.Counter[tuple[str, str]] = collections.Counter()
    resumed = collections.Counter()
    for d in runs:
        ev = d["events"]
        seg = _segments(d)[0]
        clips = d["user_clips"][1:]  # 첫 음성은 시작 요청 (user_say 기록은 그 뒤 말부터)
        says = [e["text"] for e in ev if e["name"] == "user_say"]
        line = []
        for k, (start, n) in enumerate(clips[: len(says)]):
            end_voice = start + n / (16000 * 2)
            say = says[k]
            if say in RESUME_ASKS or say == FOLLOW_UP:
                continue
            dec = next((e for e in ev if e["name"] in ("radio_decision", "radio_release") and e["t"] > start), None)
            state = "none" if dec is None else ("on" if dec["name"] == "radio_release" else dec["state"])
            delay = f"{dec['t'] - end_voice:+.1f}" if dec else "-"
            if d["scenario"] == "interrupt":
                judged[(d["expected"], state)] += 1
            item = f"{state}({delay}s)"
            if state == "paused":
                # 멈춘 뒤 재개 전에 대본을 스스로 읽었는지, 재개가 안 읽힌 첫 대사부터인지
                res = next((e for e in ev if e["name"] == "radio_resume" and e["t"] > dec["t"]), None)
                t_res = res["t"] if res else float("inf")
                said = rr._letters(
                    "".join(x for t, w, x in _deltas(d["snapshots"]) if w == "assistant" and dec["t"] < t < t_res)
                )
                script = rr._letters("".join(seg["lines"]))
                own = sum(
                    b.size
                    for b in rr.difflib.SequenceMatcher(None, said, script, autojunk=False).get_matching_blocks()
                    if b.size >= 8
                )
                gts = _gt(d, seg, seg["t0"], start)
                expect = next(
                    (u for u in range(len(seg["lines"])) if (g := gts.get(u)) is None or g.said[-1] < 0.5), None
                )
                got = next(
                    (
                        e["index"]
                        for e in ev
                        if e["name"] == "radio_push" and e["by"] == "resume" and abs(e["t"] - t_res) < 0.5
                    ),
                    None,
                )
                ok = '"on"' in res["output"] if res else False
                resumed[ok] += 1
                item += f" 혼자읽음{own} 재개{'ok' if ok else 'X'} 재개대사 {got}/{expect}"
            line.append(item)
        print(f"    {d['scenario']:9s} {d['utterance']:14s} #{d['index']} {d['result']:13s} " + " | ".join(line))
    if judged:
        print("  판정(기대 → 결과):", {f"{a}→{b}": n for (a, b), n in sorted(judged.items())})
    print(f"  재개 성공 {resumed[True]}/{resumed[True] + resumed[False]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=("long", "interrupt", "multi", "chat", "window"), default="long")
    parser.add_argument("--window", choices=("text", "stall"), default="stall", help="window: 틈에 걸릴 넣기")
    parser.add_argument("--lead", type=int, default=4, help="window text: 넣기 지점까지 이만큼(자) 남았을 때 끼어든다")
    parser.add_argument("--quiet", type=float, default=0.6, help="window stall: 모델이 조용해진 지 이만큼 뒤 끼어든다")
    parser.add_argument("--watch", type=float, default=15.0, help="window: 끼어든 뒤 이만큼 지켜보고 끝낸다")
    parser.add_argument("--text", default=None, help="window: UTTERANCES 대신 이 문장으로 끼어든다")
    parser.add_argument("--label", default=None, help="--text 를 쓸 때 결과 폴더·리포트에 쓸 이름")
    parser.add_argument("--utterance", choices=list(UTTERANCES), default="content_key")
    parser.add_argument("--first", default=None, help="첫 대본 키 (interrupt·multi 는 drawer 기본)")
    parser.add_argument("--at-line", type=int, default=4, help="interrupt: 이 대사(1부터)에서 끼어든다")
    parser.add_argument("--at", type=float, default=2.0, help="그 대사를 읽기 시작하고 이만큼 뒤")
    parser.add_argument("--cycles", type=int, default=3, help="multi: 끼어들기·재개 횟수")
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
    if args.first is None and args.scenario != "long":
        args.first = "drawer"
    if args.scenario != "long" and args.songs == 2:
        args.songs = 1
    run(args)


if __name__ == "__main__":
    main()
