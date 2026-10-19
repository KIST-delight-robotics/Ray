"""라디오 스코프 — 기록된 라디오 세션을 운영 :class:`Radio` 그대로 다시 돌려, 매 순간 추적이 어떻게 되고 있었는지 HTML
한 장으로 본다. API 를 부르지 않는다.

입력은 운영 경로 하네스 실험(``exp_radio_pipeline.py``)의 회차 폴더다 — ``run.json``(라디오에 들어간 전사 스냅샷,
출력 오디오 조각, 이벤트), ``experiment.db``(판정 호출 기록), ``pipeline.log``, ``run.wav``.

재생: 기록된 입력(전사 스냅샷·모델 음성·무음·툴·곡 끝)을 시각 순서대로 운영 ``Radio`` 에 넣고, 입력마다 그리고
:data:`TICK_SEC` 마다 ``step`` 한다. 고정하는 것은 둘 — 대본 고르기는 기록된 대본으로, 판정기 둘은 기록된 결과를
기록된 걸린 시간 뒤에 돌려준다(호출 순서대로). 같은 입력이면 같은 결정이므로 재생의 넣기·지시·판정·로그를 기록과
맞대 "재생 = 기록" 인지 보인다 — 어긋나면 그 시각 뒤 상태는 믿지 않는다.

매 틱 상태는 ``Radio`` / ``Tracker`` 의 비공개 속성을 읽어 뽑는다(운영 코드는 그대로). 그래서 radio.py 가 바뀌면 여기를
같이 고쳐야 한다 — 바깥에서 다시 계산한 진행(일치 덩어리)과 "왜" 설명의 분기는 운영 값과 맞대 어긋나면 표시한다.

    uv run python scripts/gpt_live/radio_scope.py build var/experiments/radio_pipeline
    uv run python scripts/gpt_live/radio_scope.py serve var/experiments/radio_pipeline

``build`` 는 회차마다 ``var/radio_scope/<태그>/<회차>/scope.html`` 을(녹음은 원본을 가리키는 링크), 그 위에 회차 목록
``scope_index.html`` 을 쓴다(``--out`` 으로 바꾼다). 실험 폴더는 읽기만 한다. ``serve`` 는 그 폴더를
로컬 HTTP 로 연다(VS Code 원격이면 포트가 전달된다). 파일로 바로 열어도 된다 — 녹음은 같은 폴더의 ``run.wav``.
"""

from __future__ import annotations

import argparse
import difflib
import functools
import html
import http.server
import json
import logging
import math
import re
import sqlite3
import subprocess
import wave
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from voice_pipeline.engines.gpt_live import radio
from voice_pipeline.engines.gpt_live.instructions import (
    RADIO_ENDED_NOTE,
    RADIO_LATE_LINE_NOTE,
    RADIO_ON_NOTE,
    RADIO_PAUSED_NOTE,
)
from voice_pipeline.engines.gpt_live.songs import Song
from voice_pipeline.settings import BRIDGE_SAMPLE_RATE

TEMPLATE = Path(__file__).with_name("radio_scope.html")
TICK_SEC = 0.02  # step 간격 — 운영 루프는 마이크 프레임(30 ms)마다 돈다. 입력이 올 때도 step 한다
ENVELOPE_SEC = 0.04  # 파형 해상도
BACKEND_BUSY_MAX_SEC = 20.0  # LiveSessionLoop._BACKEND_BUSY_MAX_SEC 와 같게
INF = float("inf")
LOG_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) (\S+)\s+(\w+)\s+(.*)$")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
LINE_RE = re.compile(r"^You are now saying: «(.*)»$", re.DOTALL)
NOTES = {
    RADIO_LATE_LINE_NOTE: "대답 먼저",
    RADIO_PAUSED_NOTE: "멈춤 알림",
    RADIO_ON_NOTE: "시작·재개 알림",
    "Radio talk: resumed. Read the application's lines as before.": "재개 알림",  # 10-19 전 로그
    RADIO_ENDED_NOTE: "종료 알림",
}
SHOWN_LOGGERS = ("voice_pipeline.radio", "voice_pipeline.live_session", "voice_pipeline.gpt_live")
# 보이는 상수 — 계기판 문턱. 클래스 변수에서 읽는다
TRACKER_CONSTS = (
    "TRACK_BLOCK", "DONE_SLACK", "PUSH_REMAIN", "STALL_SEC", "CUT_LETTERS", "REPUSH_WAIT_SEC", "CLOSING_QUIET_SEC",
    "CLOSING_FALLBACK_SEC", "CLOSING_CAP_SEC",
)  # fmt: skip
RADIO_CONSTS = (
    "USER_QUIET_SEC", "RECHECK_LETTERS", "DUE_WAIT_SEC", "JUDGE_LATE_MAX_SEC", "HOLD_MAX_QUIET_SEC",
    "RESUME_QUIET_SEC", "RESUME_JUDGE_LEAD_SEC", "RESUME_LATER_SEC",
)  # fmt: skip


# ---------------------------------------------------------------------------
# 기록
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeRecord:
    """판정 호출 기록 한 건 (``call_records``)."""

    result: str
    reason: str
    elapsed: float
    ok: bool


@dataclass
class Tape:
    """회차 하나의 기록. 시각은 ``run.json`` 의 단조 시각."""

    run_dir: Path
    data: dict[str, Any]
    judges: dict[str, list[JudgeRecord]]
    log: list[tuple[float, str, str]]  # (단조 시각, 로거, 메시지) — 기준점을 못 찾으면 빈 목록
    busy: list[tuple[float, float]]  # 백엔드 위임이 진행 중이던 구간
    block_push: int | None  # window stall 실험 훅이 앞서 넣기를 막은 대사(0부터)

    @property
    def t0(self) -> float:
        return float(self.data["t0"])

    @property
    def name(self) -> str:
        return f"{self.run_dir.parent.name}/{self.run_dir.name}"

    @property
    def end(self) -> float:
        times = [e["t"] for e in self.data["events"]] + [a[0] for a in self.data["audio"]]
        times += [s[0] for s in self.data["snapshots"]]
        return max(times)


def load_tape(run_dir: Path) -> Tape:
    data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    raw_log = _read_log(run_dir / "pipeline.log")
    offset = _log_offset(raw_log, data["events"])
    log = [(w + offset, name, msg) for w, name, msg in raw_log] if offset is not None else []
    cfg = data.get("config") or {}
    block = cfg.get("at_line") if cfg.get("scenario") == "window" and cfg.get("window") == "stall" else None
    return Tape(
        run_dir=run_dir,
        data=data,
        judges=_judge_records(run_dir / "experiment.db"),
        log=log,
        busy=_busy_spans(log, data["events"]),
        block_push=block,
    )


def _read_log(path: Path) -> list[tuple[float, str, str]]:
    """``pipeline.log`` 의 (벽시계, 로거, 메시지). 여러 줄 메시지의 이어진 줄은 버린다."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LOG_RE.match(line)
        if m:
            wall = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f").timestamp()
            out.append((wall, m.group(2), m.group(4)))
    return out


def _log_offset(log: list[tuple[float, str, str]], events: list[dict[str, Any]]) -> float | None:
    """벽시계 → 단조 시각. 첫 대사 넣기(로그 한 줄과 기록 훅 한 건)를 기준점으로 맞춘다."""
    push = next((e for e in events if e["name"] == "radio_push"), None)
    wall = next((w for w, name, msg in log if name == "voice_pipeline.radio" and msg.startswith("Radio: line ")), None)
    if push is None or wall is None:
        return None
    return float(push["t"]) - wall


def _busy_spans(log: list[tuple[float, str, str]], events: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """백엔드 위임이 진행 중이던 구간 — 루프처럼 위임 생성부터 툴 호출 없는 응답 완료까지(상한 20 s).

    로그가 없으면 기록 훅의 위임 생성 → 다음 백엔드 메시지로 근사한다.
    """
    spans: list[tuple[float, float]] = []
    opened: list[float] = []
    if log:
        for t, name, msg in log:
            if name != "voice_pipeline.gpt_live":
                continue
            if msg.startswith("Delegation created"):
                opened.append(t)
            elif msg.startswith("Backend response") and opened:
                status = re.search(r"\((\w+)", msg)
                if "calls=-" in msg or (status and status.group(1) in ("failed", "incomplete")):
                    spans.append((opened.pop(0), t))
    else:
        for e in sorted(events, key=lambda e: e["t"]):
            if e["name"] == "delegation_created":
                opened.append(e["t"])
            elif e["name"] == "backend_reply" and opened:
                spans.append((opened.pop(0), e["t"]))
    spans += [(t, INF) for t in opened]
    return [(a, min(b, a + BACKEND_BUSY_MAX_SEC)) for a, b in spans]


def _judge_records(db: Path) -> dict[str, list[JudgeRecord]]:
    out: dict[str, list[JudgeRecord]] = {"judge": [], "resume_judge": []}
    if not db.exists():
        return out
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT operation, elapsed_ms, status, metadata FROM call_records WHERE module = 'radio' ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    for op, ms, status, meta in rows:
        fields = json.loads(meta or "{}")
        result = str(fields.get("state") or fields.get("decision") or "")
        out.setdefault(op, []).append(JudgeRecord(result, str(fields.get("reason", "")), ms / 1000.0, status == "ok"))
    return out


# ---------------------------------------------------------------------------
# 재생 대역 — 대본 고르기·판정기·executor
# ---------------------------------------------------------------------------


class _Clock:
    now = 0.0


class _TapePicker:
    """기록된 대본을 기록된 순서로 고른다."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._keys = [e["key"] for e in data["events"] if e["name"] == "radio_script"]
        self.scripts = {
            key: radio.RadioScript(
                key=key,
                topic="",
                song=Song(key=data["songs"][key], title=data["songs"][key], artist=""),
                lines=tuple(lines),
            )
            for key, lines in data["scripts"].items()
        }

    def pick(self) -> radio.RadioScript | None:
        return self.scripts[self._keys.pop(0)] if self._keys else None

    def new_session(self) -> None:
        pass


@dataclass
class _Call:
    """판정 호출 하나 — 재생이 만든 입력과 기록된 결과."""

    kind: str  # judge / resume_judge
    inputs: dict[str, Any]
    record: JudgeRecord | None
    sent: float = 0.0
    due: float = 0.0
    log_index: int = -1  # 재생 로그에서 이 호출을 부른 줄 (판정 입력)


class _Executor:
    """판정을 그 자리에서 기록 결과로 정하고, 기록된 걸린 시간이 지나야 끝나는 Future 를 준다."""

    def __init__(self, clock: _Clock) -> None:
        self._clock = clock
        self._pending: list[tuple[float, Future[str], str | None, Exception | None, _Call | None]] = []
        self.calls: list[_Call] = []
        self.current: _Call | None = None

    def submit(self, fn: Callable[[], str]) -> Future[str]:
        future: Future[str] = Future()
        self.current = None
        value: str | None = None
        exc: Exception | None = None
        try:
            value = fn()
        except Exception as e:  # noqa: BLE001 — 기록된 실패도 Future 로 넘긴다
            exc = e
        call, now = self.current, self._clock.now
        due = now + (call.record.elapsed if call is not None and call.record is not None else 0.0)
        if call is not None:
            call.sent, call.due = now, due
        self._pending.append((due, future, value, exc, call))
        return future

    def settle(self, now: float) -> None:
        for item in list(self._pending):
            due, future, value, exc, call = item
            if due > now + 1e-9:
                continue
            self._pending.remove(item)
            if call is not None and call.record is not None and call.record.ok:
                rec = call.record
                label = "Radio judge" if call.kind == "judge" else "Radio resume judge"
                logging.getLogger("voice_pipeline.radio").info(
                    "%s: %s %.2fs (%s)", label, rec.result, rec.elapsed, rec.reason
                )
            if exc is not None:
                future.set_exception(exc)
            else:
                future.set_result(value or "")


class _Judge:
    """``RadioJudge`` / ``RadioResumeJudge`` 대역 — 호출 순서대로 기록된 결과를 낸다."""

    def __init__(self, kind: str, records: list[JudgeRecord], executor: _Executor, log_size: Callable[[], int]):
        self._kind = kind
        self._records = list(records)
        self._executor = executor
        self._log_size = log_size

    def classify(self, line: str, exchange: Any, earlier: Any = ()) -> str:
        return self._answer(
            {"line": line, "exchange": [list(x) for x in exchange], "earlier": [list(x) for x in earlier]}
        )

    def decide(self, line: str, interruption: str, conversation: Any, *, second: bool = False) -> str:
        inputs = {"line": line, "interruption": interruption, "conversation": [list(x) for x in conversation]}
        return self._answer({**inputs, "second": second})

    def _answer(self, inputs: dict[str, Any]) -> str:
        record = self._records.pop(0) if self._records else None
        call = _Call(self._kind, inputs, record, log_index=self._log_size() - 1)
        self._executor.current = call
        self._executor.calls.append(call)
        if record is None:
            raise RuntimeError("no recorded judgment")
        if not record.ok:
            raise RuntimeError("recorded judgment failed")
        return record.result


class _Capture(logging.Handler):
    """라디오 로그를 재생 시각과 함께 모은다."""

    def __init__(self, clock: _Clock) -> None:
        super().__init__(logging.INFO)
        self._clock = clock
        self.records: list[tuple[float, str, tuple[Any, ...], str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        args = record.args if isinstance(record.args, tuple) else ()
        self.records.append((self._clock.now, str(record.msg), args, record.getMessage()))


# ---------------------------------------------------------------------------
# 설명 — Radio.step / Tracker.next_action 의 분기를 그대로 따라간다
# ---------------------------------------------------------------------------

ACTION_OF = {"push": "Push", "push-stall": "Push", "repush": "Repush", "closing": "Closing"}
ACTION_OF.update({"closing-quiet": "Closing", "closing-cap": "Closing"})


def explain(r: radio.Radio, t: float, quiet: float) -> tuple[str, str]:
    """(분기 코드, 한 줄 설명) — 이 틱의 상태에서 라디오가 무엇을 기다리거나 하는지."""
    if r.status == "off":
        return "off", "라디오 꺼짐"
    if r.playing_song:
        return "song", "곡 재생 중 — 라디오는 쉰다"
    if r.status == "paused":
        return _explain_paused(r, t)
    tr = r.tracker
    if tr is None:
        return "none", "대본 없음"
    if r._hold is not None:
        return _explain_hold(r, r._hold, t, quiet)
    return explain_tracker(tr, t, quiet)


def explain_tracker(tr: radio.Tracker, t: float, quiet: float) -> tuple[str, str]:
    if not tr._pushes:
        return "idle", "넣은 대사 없음"
    tr._update_done()
    last = len(tr._pushes) - 1
    reach, said = tr._progress()
    voiced = tr._first_voice is not None
    stalled = voiced and quiet >= tr.STALL_SEC
    n = len(tr.lines)
    remain = tr._offsets[last + 1] - reach
    if len(tr._pushes) >= n:  # 곡 소개까지 넣었다
        if tr._done >= n and quiet >= tr.CLOSING_QUIET_SEC:
            return "closing", f"곡 소개 끝, 조용 {quiet:.1f} ≥ {tr.CLOSING_QUIET_SEC} s → 곡"
        unread = tr._unread(reach, said) if stalled else []
        if unread and tr._repushed != last:
            return "repush", f"곡 소개에서 멈춤(조용 {quiet:.1f} s) — 끊긴 문장 다시 넣기"
        waited = tr._repushed != last or t - tr._repushed_at >= tr.REPUSH_WAIT_SEC
        if voiced and quiet >= tr.CLOSING_FALLBACK_SEC and waited:
            return "closing-quiet", f"곡 소개 끝을 못 찾고 조용 {quiet:.1f} ≥ {tr.CLOSING_FALLBACK_SEC} s → 곡"
        if t - tr._pushes[-1] >= tr.CLOSING_CAP_SEC:
            return "closing-cap", f"곡 소개를 넣은 지 {tr.CLOSING_CAP_SEC:.0f} s → 곡"
        if tr._done >= n:
            return "closing-wait", f"곡 소개 다 읽음 — 조용 {quiet:.1f} / {tr.CLOSING_QUIET_SEC} s 면 곡"
        return "closing-read", f"곡 소개 읽는 중 — 끝까지 {remain}자"
    if reach > tr._last_start[last] and remain <= tr.PUSH_REMAIN:
        return "push", f"{last + 1}번 대사 마지막 문장 안, 남은 {remain}자 ≤ {tr.PUSH_REMAIN} → 다음 대사"
    if not stalled:
        if not voiced:
            return "wait-voice", f"{last + 1}번 대사를 넣었다 — 레이 목소리를 기다린다"
        if reach <= tr._last_start[last]:
            text = f"{last + 1}번 대사 읽는 중 — 마지막 문장까지 {tr._last_start[last] - reach}자"
        elif tr.PUSH_REMAIN < 0:
            text = f"{last + 1}번 대사 마지막 문장 안 — 실험 훅이 이 대사의 앞서 넣기를 막았다"
        else:
            text = f"{last + 1}번 대사 마지막 문장 안 — 남은 {remain}자 > {tr.PUSH_REMAIN}"
        if quiet >= 0.3:
            text += f" · 조용 {quiet:.1f} / {tr.STALL_SEC} s"
        return "reading", text
    unread = tr._unread(reach, said)
    if unread and tr._repushed != last:
        half = remain / 2
        return (
            "repush",
            f"멈춤(조용 {quiet:.1f} s) — 끝까지 {remain}자, 그 뒤 말한 {said}자 < {half:.0f} → 끊긴 문장 다시 넣기",
        )
    if unread and t - tr._repushed_at < tr.REPUSH_WAIT_SEC:
        return "repush-wait", f"다시 넣은 뒤 기다림 {t - tr._repushed_at:.1f} / {tr.REPUSH_WAIT_SEC:.0f} s"
    if unread:
        why = "다시 넣어도 조용"
    elif remain < tr.CUT_LETTERS:
        why = f"끝까지 {remain}자 < {tr.CUT_LETTERS} — 어미는 넘어간다"
    else:
        why = f"그 뒤 말한 {said}자 ≥ 남은 {remain}자의 절반 — 바꿔 말하며 읽었다"
    return "push-stall", f"멈춤(조용 {quiet:.1f} s) — {why} → 다음 대사"


def _explain_hold(r: radio.Radio, hold: Any, t: float, quiet: float) -> tuple[str, str]:
    R = radio.Radio
    last_user = r._last_user_text
    if last_user is not None and last_user > hold.judged_user and t - last_user < R.USER_QUIET_SEC:
        code, text = "hold-listen", f"보류 — 사용자 말 받는 중 (전사가 {R.USER_QUIET_SEC} s 멈추면 판정)"
    elif r._waiting():
        call = next(c for c in r._calls if not c.future.done() and c.hold_start == r._last_hold_start)
        code, text = "hold-judging", f"보류 — 판정 기다리는 중 ({t - call.sent_at:.1f} s)"
    elif hold.latest_on is not None and r._stale(hold.latest_on, hold):
        code, text = "hold-recheck", "보류 — on 판정 뒤 새 말·반응이 있어 다시 판정한다"
    elif hold.latest_on is not None:
        code, text = "hold-on", "on 판정 — 보류를 푼다"
    else:
        code, text = "hold-quiet", f"보류 — 판정 없이 레이가 {R.HOLD_MAX_QUIET_SEC:.0f} s 조용하면 푼다"
    if hold.due_since is not None:
        if hold.last_sent is None:
            text += " · 넣을 때가 됐다 (판정 호출 전)"
        else:
            waited = t - max(hold.due_since, hold.last_sent)
            text += f" · 넣을 때가 됐다 — 판정 기다림 {waited:.1f} / {R.DUE_WAIT_SEC:.0f} s"
    return code, text


PAUSE_MARKS = ("멈춘 때", "사용자 전사", "레이 말소리", "백엔드 위임")


def _lull(r: radio.Radio) -> tuple[float, int]:
    marks = (r._paused_at, r._last_user_text, r._last_voice, r._last_busy)
    i = max((i for i, m in enumerate(marks) if m is not None), key=lambda i: marks[i] or 0.0)
    return float(marks[i] or 0.0), i


def _explain_paused(r: radio.Radio, t: float) -> tuple[str, str]:
    R = radio.Radio
    if not r.auto_resume or r._resume_judge is None or r.tracker is None:
        return "paused-manual", "멈춤 — 자동 복귀 꺼짐, 이어 달라는 요청으로만 돌아간다"
    lull, mark = _lull(r)
    if r._given_up_lull == lull:
        return "paused-givenup", "멈춤 — 복귀 판정이 거듭 실패해 이 잠잠함에선 더 묻지 않는다"
    later = r._later_lull == lull
    due = R.RESUME_LATER_SEC if later else R.RESUME_QUIET_SEC
    quiet_for = f"잠잠 {t - lull:.1f} s ({PAUSE_MARKS[mark]}부터)"
    check = r._resume_check
    second = " · 답 기다림 뒤 다시" if later else ""
    if check is None or check.lull != lull:
        ask = due - R.RESUME_JUDGE_LEAD_SEC
        return "paused-lull", f"멈춤 — {quiet_for}, {ask:.1f} s 에 복귀 판정, {due:.0f} s 에 적용{second}"
    if not check.future.done():
        return "paused-asking", f"멈춤 — 복귀 판정 기다리는 중 · {quiet_for}{second}"
    if check.future.exception() is not None:
        return "paused-failed", "멈춤 — 복귀 판정 실패, 다시 묻는다"
    return "paused-ready", f"멈춤 — 복귀 판정 {check.future.result()} · {quiet_for}, {due:.0f} s 에 적용"


# ---------------------------------------------------------------------------
# 진행 다시 계산 — Tracker._progress 와 같고, 일치 덩어리를 밖으로 낸다
# ---------------------------------------------------------------------------


def match_frame(tr: radio.Tracker) -> dict[str, Any] | None:
    if not tr._pushes:
        return None
    base = min(max(tr._track_unit, tr._done - 1), len(tr._pushes) - 1)
    since = max(tr._track_from, tr._pushes[base])
    raw, at, pos = "", [], []
    off = 0
    first: int | None = None
    for t, d in tr._out:
        if t >= since:
            first = off if first is None else first
            raw += d
            at += [t] * len(d)
            pos += range(off, off + len(d))
        off += len(d)
    idx = [i for i, ch in enumerate(raw) if radio.letters(ch)]
    text = "".join(raw[i] for i in idx)
    script = "".join(tr._letters[base : len(tr._pushes)])
    blocks = [
        b
        for b in difflib.SequenceMatcher(None, text, script, autojunk=False).get_matching_blocks()
        if b.size >= tr.TRACK_BLOCK
    ]
    reach = reach_t = 0
    if blocks:
        reach, reach_t = blocks[-1].b + blocks[-1].size, blocks[-1].a + blocks[-1].size
    spans = [(at[idx[b.a]], at[idx[b.a + b.size - 1]]) for b in blocks]
    excluded = []
    for a, e in tr._holds:
        if e != INF and not any(t0 <= e <= t1 for t0, t1 in spans):
            e = min((t0 for t0, _ in spans if t0 >= e), default=INF)
        excluded.append((a, e))
    said = 0
    if reach_t:
        said = sum(1 for i in idx[reach_t:] if not any(a <= at[i] < e for a, e in excluded))
    base_off = tr._offsets[base]
    return {
        "since": since,
        "from": first if first is not None else off,
        "base": base,
        "blocks": [[b.a, base_off + b.b, b.size] for b in blocks],
        "excl": [[a, e] for a, e in excluded],
        "reach": base_off + reach,
        "reachAt": reach_t,
        "said": said,
        "pairs": [(pos[idx[b.a + k]], base_off + b.b + k) for b in blocks for k in range(b.size)],
    }


# ---------------------------------------------------------------------------
# 재생
# ---------------------------------------------------------------------------


@dataclass
class _Gen:
    """대본 하나를 읽는 추적기 (곡 뒤·새 시작마다 새로 생긴다)."""

    index: int
    tracker: radio.Tracker
    key: str
    song: str
    t: float
    stream_len: int = 0  # 추적기에 들어간 출력 전사 글자 수
    align: dict[int, int] = field(default_factory=dict)  # 전사 글자 위치 → 대본 글자 위치 (마지막 일치)


class Replay:
    """기록을 운영 Radio 로 다시 돈다. :meth:`run` 이 뷰어 데이터를 돌려준다."""

    def __init__(self, tape: Tape) -> None:
        self.tape = tape
        self.clock = _Clock()
        self.executor = _Executor(self.clock)
        self.capture = _Capture(self.clock)
        log_size = lambda: len(self.capture.records)  # noqa: E731
        self.judge = _Judge("judge", tape.judges.get("judge", []), self.executor, log_size)
        self.resume_judge = _Judge("resume_judge", tape.judges.get("resume_judge", []), self.executor, log_size)
        self.radio = radio.Radio(
            _TapePicker(tape.data),
            self.judge,  # type: ignore[arg-type]
            self.executor,
            self.resume_judge,  # type: ignore[arg-type]
        )
        self.quiet = 0.0
        self.song = False
        self.gens: list[_Gen] = []
        self.thinking: list[dict[str, Any]] = []
        self.songs: list[dict[str, Any]] = []
        self.links: list[dict[str, Any]] = []
        self.frames: list[dict[str, Any]] = []
        self.snaps: dict[str, list[Any]] = {k: [] for k in SNAP_KEYS}
        self.strings: list[str] = []
        self._string_ids: dict[str, int] = {}
        self._frame_key: tuple[Any, ...] | None = None
        self.progress_mismatch = 0
        self.explain_mismatch: list[tuple[float, str, str]] = []

    # -- 진행 ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        logger = logging.getLogger("voice_pipeline.radio")
        old_level, old_propagate = logger.level, logger.propagate
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.addHandler(self.capture)
        restore = self._install_block_hook()
        try:
            self._loop()
        finally:
            logger.removeHandler(self.capture)
            logger.setLevel(old_level)
            logger.propagate = old_propagate
            restore()
        return self._view()

    def _install_block_hook(self) -> Callable[[], None]:
        """window stall 실험의 넣기 막기 훅을 이 프로세스에서만 똑같이 건다 (exp_radio_pipeline.install_block_hook)."""
        block = self.tape.block_push
        if block is None:
            return lambda: None
        orig = radio.Tracker.next_action

        def next_action(tr: radio.Tracker, t: float, quiet: float) -> radio.Action | None:
            _apply_block(tr, block)
            return orig(tr, t, quiet)

        radio.Tracker.next_action = next_action  # type: ignore[method-assign]

        def restore() -> None:
            radio.Tracker.next_action = orig  # type: ignore[method-assign]

        return restore

    def _inputs(self) -> list[tuple[float, int, str, Any]]:
        d = self.tape.data
        out: list[tuple[float, int, str, Any]] = []
        out += [(float(t), 0, "audio", (n, v)) for t, n, v in d["audio"]]
        out += [(float(s[0]), 1, "transcript", s) for s in d["snapshots"]]
        for e in d["events"]:
            if e["name"] == "radio_start":
                out.append((float(e["t"]), 2, "start", None))
            elif e["name"] == "radio_resume":
                out.append((float(e["t"]), 2, "resume", None))
            elif e["name"] == "radio_song_end":
                out.append((float(e["t"]), 2, "song_end", bool(e.get("completed"))))
        return sorted(out, key=lambda x: (x[0], x[1]))

    def _loop(self) -> None:
        inputs = self._inputs()
        end = self.tape.end
        grid = inputs[0][0] if inputs else self.tape.t0
        i = 0
        while True:
            nxt = inputs[i][0] if i < len(inputs) else INF
            if nxt <= grid:
                t = nxt
                while i < len(inputs) and inputs[i][0] == t:
                    self._apply(inputs[i])
                    i += 1
            else:
                t = grid
                grid += TICK_SEC
            if t > end:
                break
            self.clock.now = t
            self.executor.settle(t)
            if not self.song:
                self._run(self.radio.step(t, self.quiet, backend_busy=self._busy(t)))
            self._capture(t)

    def _apply(self, item: tuple[float, int, str, Any]) -> None:
        t, _, kind, payload = item
        self.clock.now = t
        r = self.radio
        if kind == "audio":
            n, voiced = payload
            if self.song:
                return  # 곡 중엔 스트림이 닫혀 루프가 음성·무음을 세지 않는다
            if voiced:
                self.quiet = 0.0
                r.on_voice(t)
            else:
                self.quiet += n / (BRIDGE_SAMPLE_RATE * 2)
        elif kind == "transcript":
            _, seg, speaker, text = payload
            tr = r.tracker
            before = len(tr._out) if tr is not None else 0
            old = r._transcripts.texts.get(seg, "")
            r.on_transcript(t, seg, speaker, text)
            if tr is not None and len(tr._out) > before:
                gen = self._gen(tr, t)
                delta = tr._out[-1][1]
                start = len(old) if text.startswith(old) else 0
                self.links.append(
                    {"seg": seg, "off": start, "len": len(delta), "gen": gen.index, "raw": gen.stream_len}
                )
                gen.stream_len += len(delta)
        elif kind == "start":
            self._run(r.start(t)[1])
        elif kind == "resume":
            self._run(r.resume(t)[1])
        elif kind == "song_end":
            self.song = False
            self._run(r.on_song_end(t, completed=payload))
        if r.tracker is not None:
            self._gen(r.tracker, t)

    def _run(self, commands: list[radio.Command]) -> None:
        t = self.clock.now
        for c in commands:
            if isinstance(c, radio.Think):
                m = LINE_RE.match(c.text)
                if m:
                    self.thinking.append({"t": t, "kind": "line", "text": m.group(1)})
                else:
                    self.thinking.append({"t": t, "kind": "note", "label": NOTES.get(c.text, "지시"), "text": c.text})
            elif isinstance(c, radio.PlaySong):
                self.song = True
                self.songs.append({"t": t, "song": c.song.key})
        if self.radio.tracker is not None:
            self._gen(self.radio.tracker, t)

    def _busy(self, t: float) -> bool:
        return any(a <= t < b for a, b in self.tape.busy)

    def _gen(self, tr: radio.Tracker, t: float) -> _Gen:
        for g in self.gens:
            if g.tracker is tr:
                return g
        script = self.radio.script
        gen = _Gen(len(self.gens), tr, script.key if script else "?", script.song.key if script else "", t)
        self.gens.append(gen)
        return gen

    # -- 틱마다 상태 -------------------------------------------------------------

    def _string(self, text: str) -> int:
        if text not in self._string_ids:
            self._string_ids[text] = len(self.strings)
            self.strings.append(text)
        return self._string_ids[text]

    def _capture(self, t: float) -> None:
        r, tr, q = self.radio, self.radio.tracker, self.quiet
        if tr is not None and self.tape.block_push is not None:
            _apply_block(tr, self.tape.block_push)
        code, text = explain(r, t, q)
        row: dict[str, Any] = dict.fromkeys(SNAP_KEYS)
        row["t"] = t
        row["st"] = {"off": 0, "on": 1, "paused": 2}[r.status] if not r.playing_song else 3
        row["hold"] = 1 if r._hold is not None else 0
        row["q"] = q
        row["why"] = self._string(text)
        row["code"] = self._string(code)
        if tr is not None and tr._pushes:
            gen = self._gen(tr, t)
            row["gen"] = gen.index
            last = len(tr._pushes) - 1
            reach, said = tr._progress()
            row.update(pu=len(tr._pushes), dn=tr.done, rd=tr.reading()[0], rc=reach, sa=said)
            row["rem"] = tr._offsets[last + 1] - reach
            row["tl"] = tr._last_start[last] - reach
            row["fv"] = 1 if tr._first_voice is not None else 0
            row["rp"] = t - tr._repushed_at if tr._repushed == last else None
            self._frame(tr, gen, t)
            if r.status == "on" and r._hold is None and not r.playing_song:
                action = tr.next_action(t, q)
                got = type(action).__name__ if action is not None else "None"
                want = ACTION_OF.get(code, "None")
                if got != want and len(self.explain_mismatch) < 50:
                    self.explain_mismatch.append((t, code, got))
        hold = r._hold
        if hold is not None:
            if r._last_user_text is not None and r._last_user_text > hold.judged_user:
                row["uq"] = t - r._last_user_text
            pending = [c for c in r._calls if not c.future.done() and c.hold_start == r._last_hold_start]
            row["ja"] = t - pending[0].sent_at if pending else None
            if hold.due_since is not None and hold.last_sent is not None:
                row["da"] = t - max(hold.due_since, hold.last_sent)
            if tr is not None and tr._pushes:
                row["rx"] = r._reaction(hold) - hold.judged_reaction
        if r.status == "paused" and r.auto_resume and r.tracker is not None:
            lull, _ = _lull(r)
            later = r._later_lull == lull
            row["lull"] = t - lull
            row["ldue"] = radio.Radio.RESUME_LATER_SEC if later else radio.Radio.RESUME_QUIET_SEC
        for k, v in row.items():
            self.snaps[k].append(_round(v))

    def _frame(self, tr: radio.Tracker, gen: _Gen, t: float) -> None:
        key = (gen.index, len(tr._out), len(tr._pushes), tr._done, tr._track_from, str(tr._holds))
        if key == self._frame_key:
            return
        self._frame_key = key
        frame = match_frame(tr)
        if frame is None:
            return
        if (frame["reach"], frame["said"]) != tr._progress():
            self.progress_mismatch += 1
        for raw, pos in frame.pop("pairs"):
            gen.align[raw] = pos
        frame["excl"] = [[a, None if e == INF else e] for a, e in frame["excl"]]
        self.frames.append({"t": t, "gen": gen.index, **frame})

    # -- 결과 ----------------------------------------------------------------

    def _view(self) -> dict[str, Any]:
        tape, t0 = self.tape, self.tape.t0
        rel = functools.partial(_rel, t0)
        replay_log = [
            {"t": rel(t), "src": "replay", "msg": msg}
            for t, _, _, msg in self.capture.records
            if not msg.startswith("Radio scripts:")
        ]
        pushes = [
            {"t": rel(t), "index": args[0] - 1, "n": args[1], "by": args[2]}
            for t, fmt, args, _ in self.capture.records
            if fmt.startswith("Radio: line %d/%d pushed") and len(args) == 3
        ]
        repushes = [
            {"t": rel(t), "text": args[0]}
            for t, fmt, args, _ in self.capture.records
            if fmt.startswith("Radio: quiet after speaking") and args
        ]
        verify = self._verify(pushes, replay_log)
        return {
            "meta": self._meta(),
            "verify": verify,
            "consts": {
                **{k: getattr(radio.Tracker, k) for k in TRACKER_CONSTS},
                **{k: getattr(radio.Radio, k) for k in RADIO_CONSTS},
            },
            "gens": [self._gen_view(g, t0) for g in self.gens],
            "snaps": {k: [_rel(t0, x) for x in v] if k == "t" else v for k, v in self.snaps.items()},
            "strings": self.strings,
            "frames": [_frame_view(f, t0) for f in self.frames],
            "segments": _segments(tape.data, t0),
            "links": self.links,
            "pushes": pushes,
            "repushes": repushes,
            "thinking": [{**x, "t": rel(x["t"])} for x in self.thinking],
            "songs": [{**x, "t": rel(x["t"])} for x in self.songs],
            "judges": [self._call_view(c, t0) for c in self.executor.calls],
            "busy": [[rel(a), rel(b)] for a, b in tape.busy],
            "recorded": _recorded(tape, t0),
            "logs": replay_log
            + [
                {"t": rel(t), "src": "record", "msg": msg, "logger": name}
                for t, name, msg in tape.log
                if name in SHOWN_LOGGERS
            ],
            **_audio_view(tape),
        }

    def _meta(self) -> dict[str, Any]:
        d = self.tape.data
        cfg = d.get("config") or {}
        recorded = _recorded_at(self.tape.run_dir)
        return {
            "recordedAt": recorded.strftime("%m-%d %H:%M") if recorded else None,
            "codeChanges": _code_changes(recorded) if recorded else None,
            "name": self.tape.name,
            "scenario": d.get("scenario"),
            "utterance": d.get("utterance"),
            "result": d.get("result"),
            "expected": d.get("expected"),
            "at": cfg.get("at_line"),
            "window": cfg.get("window") if d.get("scenario") == "window" else None,
            "blockPush": self.tape.block_push,
            "end": _rel(self.tape.t0, self.tape.end),
            "said": [e.get("text") for e in d["events"] if e["name"] == "user_say"],
        }

    def _gen_view(self, g: _Gen, t0: float) -> dict[str, Any]:
        tr = g.tracker
        stream, deltas, off = "", [], 0
        for t, d in tr._out:
            deltas.append([_rel(t0, t), off])
            stream += d
            off += len(d)
        times = np.repeat([x[0] for x in deltas], [len(d) for _, d in tr._out]) if deltas else np.array([])
        dots = sorted((float(times[raw]), pos, raw) for raw, pos in g.align.items() if raw < len(times))
        return {
            "key": g.key,
            "song": g.song,
            "t": _rel(t0, g.t),
            "lines": list(tr.lines),
            "masks": [_mask(line) for line in tr.lines],
            "sentenceEnds": [_sentence_ends(line) for line in tr.lines],
            "offsets": tr._offsets,
            "lastStart": tr._last_start,
            "stream": stream,
            "smask": _mask(stream),
            "deltas": deltas,
            "dots": [
                [round(t, 3), p, raw] for t, p, raw in dots
            ],  # 대본과 일치한 전사 글자 (시각, 대본 위치, 전사 위치)
        }

    def _call_view(self, c: _Call, t0: float) -> dict[str, Any]:
        rec = c.record
        asked = self.capture.records[c.log_index][3] if 0 <= c.log_index < len(self.capture.records) else ""
        return {
            "kind": c.kind,
            "sent": _rel(t0, c.sent),
            "due": _rel(t0, c.due),
            "inputs": c.inputs,
            "recorded": rec is not None,
            "result": rec.result if rec else "",
            "reason": rec.reason if rec else "",
            "elapsed": rec.elapsed if rec else 0.0,
            "ok": rec.ok if rec else False,
            "asked": asked,
            "askedMatch": self._asked_in_record(asked),
        }

    def _asked_in_record(self, asked: str) -> bool | None:
        """재생의 판정 입력 로그 줄이 기록 로그에도 있는가 (기록 로그가 없으면 None)."""
        if not self.tape.log or not asked:
            return None
        return any(msg == asked for _, name, msg in self.tape.log if name == "voice_pipeline.radio")

    def _verify(self, pushes: list[dict[str, Any]], replay_log: list[dict[str, Any]]) -> dict[str, Any]:
        tape, t0 = self.tape, self.tape.t0
        rec_pushes = [e for e in tape.data["events"] if e["name"] == "radio_push"]
        same_push = 0
        max_dt = 0.0
        diverge: float | None = None
        for k in range(max(len(pushes), len(rec_pushes))):
            a = pushes[k] if k < len(pushes) else None
            b = rec_pushes[k] if k < len(rec_pushes) else None
            if a and b and a["index"] == b["index"] and a["by"] == b["by"] and abs(a["t"] - _rel(t0, b["t"])) <= 0.25:
                same_push += 1
                max_dt = max(max_dt, abs(a["t"] - _rel(t0, b["t"])))
            elif diverge is None:
                diverge = min(x for x in (a and a["t"], b and _rel(t0, b["t"])) if x is not None)
        rec_think_ev = [e for e in tape.data["events"] if e["name"] == "append_thinking"]
        rec_think = [e.get("text") or "" for e in rec_think_ev]
        mine = [x["text"] if x["kind"] == "note" else f"You are now saying: «{x['text']}»" for x in self.thinking]
        # 지시는 순서대로 맞춰 본다 — 하나가 끼어들거나 빠져도 뒤가 전부 어긋난 것으로 세지 않게
        think_sm = difflib.SequenceMatcher(None, mine, rec_think, autojunk=False)
        same_think = sum(b.size for b in think_sm.get_matching_blocks())
        think_diffs = []
        for op, a0, a1, b0, b1 in think_sm.get_opcodes():
            if op == "equal":
                continue
            times = [_rel(t0, self.thinking[a0]["t"]) if a0 < a1 else None]
            times.append(_rel(t0, rec_think_ev[b0]["t"]) if b0 < b1 else None)
            first = min(x for x in times if x is not None)
            diverge = first if diverge is None else min(diverge, first)
            think_diffs.append({"t": first, "replay": mine[a0:a1], "record": rec_think[b0:b1]})
        rec_lines = [msg for _, name, msg in tape.log if name == "voice_pipeline.radio"]
        rec_lines = [m for m in rec_lines if not m.startswith("Radio scripts:")]
        norm_a = [NUM_RE.sub("#", x["msg"]) for x in replay_log]
        norm_b = [NUM_RE.sub("#", m) for m in rec_lines]
        sm = difflib.SequenceMatcher(None, norm_a, norm_b, autojunk=False)
        diffs = []
        for op, a0, a1, b0, b1 in sm.get_opcodes():
            if op != "equal":
                diffs.append({"replay": [x["msg"] for x in replay_log[a0:a1]], "record": rec_lines[b0:b1]})
        calls = self.executor.calls
        return {
            "pushes": {"same": same_push, "replay": len(pushes), "record": len(rec_pushes), "maxDt": round(max_dt, 3)},
            "thinking": {"same": same_think, "replay": len(mine), "record": len(rec_think), "diffs": think_diffs[:20]},
            "judges": {
                kind: {
                    "replay": sum(1 for c in calls if c.kind == kind),
                    "record": len(tape.judges.get(kind, [])),
                    "unrecorded": sum(1 for c in calls if c.kind == kind and c.record is None),
                }
                for kind in ("judge", "resume_judge")
            },
            "log": {
                "same": sum(b.size for b in sm.get_matching_blocks()),
                "replay": len(norm_a),
                "record": len(norm_b),
                "available": bool(tape.log),
                "diffs": diffs[:30],
            },
            "progressMismatch": self.progress_mismatch,
            "explainMismatch": [[_rel(t0, t), c, g] for t, c, g in self.explain_mismatch],
            "diverge": diverge,
        }


SNAP_KEYS = (
    "t", "st", "hold", "q", "why", "code", "gen", "pu", "dn", "rd", "rc", "sa", "rem", "tl", "fv", "rp",
    "uq", "ja", "da", "rx", "lull", "ldue",
)  # fmt: skip


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "var" / "radio_scope"
RADIO_CODE = ("voice_pipeline/engines/gpt_live/radio.py", "voice_pipeline/engines/gpt_live/instructions.py")


def _recorded_at(run_dir: Path) -> datetime | None:
    """회차를 기록한 시각 — ``pipeline.log`` 첫 줄, 없으면 ``run.json`` 수정 시각."""
    log = run_dir / "pipeline.log"
    if log.exists():
        with log.open(encoding="utf-8", errors="replace") as f:
            m = LOG_RE.match(f.readline())
        if m:
            return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")
    run = run_dir / "run.json"
    return datetime.fromtimestamp(run.stat().st_mtime) if run.exists() else None


@functools.lru_cache(maxsize=64)
def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout


def _code_changes(recorded: datetime) -> dict[str, Any] | None:
    """기록 뒤 바뀐 라디오 코드 — 커밋과 커밋 안 된 변경. 재생은 지금 코드로 돌므로 기록과 다를 수 있는 이유."""
    since = recorded.strftime("%Y-%m-%d %H:%M:%S")
    log = _git("log", f"--since={since}", "--format=%h %ad %s", "--date=format:%m-%d %H:%M", "--", *RADIO_CODE)
    dirty = _git("status", "--porcelain", "--", *RADIO_CODE)
    if log is None:
        return None
    return {"commits": [line for line in log.splitlines() if line.strip()], "dirty": bool(dirty and dirty.strip())}


def _apply_block(tr: radio.Tracker, block: int) -> None:
    if tr.pushed == block:
        tr.PUSH_REMAIN = -1  # type: ignore[misc]
    else:
        tr.__dict__.pop("PUSH_REMAIN", None)


def _round(v: Any) -> Any:
    if isinstance(v, float):
        return None if math.isinf(v) else round(v, 3)
    return v


def _rel(t0: float, t: float | None) -> float | None:
    if t is None or math.isinf(t):
        return None
    return round(t - t0, 3)


def _mask(text: str) -> str:
    return "".join("1" if radio.letters(ch) else "0" for ch in text)


def _sentence_ends(line: str) -> list[int]:
    """대사 안 문장마다 끝 글자 위치(대사 안 글자 수 누적)."""
    out, n = [], 0
    for s in radio.sentences(line):
        n += len(radio.letters(s))
        out.append(n)
    return out


def _frame_view(f: dict[str, Any], t0: float) -> dict[str, Any]:
    return {
        "t": _rel(t0, f["t"]),
        "gen": f["gen"],
        "since": _rel(t0, f["since"]),
        "from": f["from"],
        "base": f["base"],
        "blocks": f["blocks"],
        "excl": [[_rel(t0, a), _rel(t0, e)] for a, e in f["excl"]],
        "reach": f["reach"],
        "reachAt": f["reachAt"],
        "said": f["said"],
    }


def _segments(data: dict[str, Any], t0: float) -> list[dict[str, Any]]:
    """전사 세그먼트마다 (화자, 스냅샷 [(시각, 글)]) — 바뀐 것만."""
    segs: dict[str, dict[str, Any]] = {}
    for t, seg, speaker, text in data["snapshots"]:
        s = segs.setdefault(seg, {"id": seg, "who": speaker, "snaps": []})
        if not s["snaps"] or s["snaps"][-1][1] != text:
            s["snaps"].append([_rel(t0, t), text])
    return sorted(segs.values(), key=lambda s: s["snaps"][0][0])


def _recorded(tape: Tape, t0: float) -> dict[str, Any]:
    ev = tape.data["events"]
    return {
        "pushes": [
            {"t": _rel(t0, e["t"]), "index": e["index"], "by": e["by"]} for e in ev if e["name"] == "radio_push"
        ],
        "userSay": [{"t": _rel(t0, e["t"]), "text": e.get("text", "")} for e in ev if e["name"] == "user_say"],
        "delegations": [_rel(t0, e["t"]) for e in ev if e["name"] == "delegation_created"],
        "tools": [{"t": _rel(t0, e["t"]), "fn": e.get("fn")} for e in ev if e["name"] == "backend_call"],
    }


def _audio_view(tape: Tape) -> dict[str, Any]:
    """파형(단조 시각 격자)과 단조 시각 ↔ 녹음 위치 대응. 녹음은 받은 모델 오디오를 이어 붙인 것이라 시간축이 다르다."""
    t0 = tape.t0
    audio = tape.data["audio"]
    if not audio:
        return {"audioMap": None, "env": None, "wav": None}
    times = np.array([a[0] for a in audio], dtype=np.float64) - t0
    pos = np.cumsum([a[1] for a in audio]) / (BRIDGE_SAMPLE_RATE * 2)
    audio_map = {"t": [round(float(x), 3) for x in times], "pos": [round(float(x), 3) for x in pos]}
    wav_path = tape.run_dir / "run.wav"
    if not wav_path.exists():
        return {"audioMap": audio_map, "env": None, "wav": None}
    with wave.open(str(wav_path)) as w:
        rate, channels = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, channels).astype(np.float32)
    hop = int(rate * ENVELOPE_SEC)
    n = len(pcm) // hop
    rms = np.sqrt(np.mean(pcm[: n * hop].reshape(n, hop, channels) ** 2, axis=1))  # (n, ch)
    db = 20 * np.log10(rms / 32768.0 + 1e-9)
    level = np.clip((db + 60.0) / 54.0, 0.0, 1.0)  # -60 dBFS → 0, -6 dBFS → 1
    grid = np.arange(0.0, _rel(t0, tape.end) or 0.0, ENVELOPE_SEC)
    frames = np.clip((np.interp(grid, times, pos, left=0.0) / ENVELOPE_SEC).astype(int), 0, max(n - 1, 0))
    env = {
        "hop": ENVELOPE_SEC,
        "user": [int(x) for x in np.round(level[frames, 0] * 100)],
        "ray": [int(x) for x in np.round(level[frames, min(1, channels - 1)] * 100)],
    }
    return {"audioMap": audio_map, "env": env, "wav": "run.wav"}


# ---------------------------------------------------------------------------
# 쓰기
# ---------------------------------------------------------------------------


def find_runs(paths: list[Path]) -> list[Path]:
    runs: list[Path] = []
    for p in paths:
        if (p / "run.json").exists():
            runs.append(p)
        else:
            runs += sorted(f.parent for f in p.rglob("run.json"))
    return runs


def build_run(run_dir: Path, out_root: Path) -> dict[str, Any] | None:
    """회차 하나 → ``out_root/<태그>/<회차>/scope.html`` (+ 녹음 링크, 목록용 요약). 라디오를 시작하지 않은 회차는
    건너뛴다 (None)."""
    tape = load_tape(run_dir)
    if not any(e["name"] == "radio_start" for e in tape.data["events"]):
        return None
    view = Replay(tape).run()
    payload = json.dumps(view, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.read_text(encoding="utf-8").replace("__SCOPE_DATA__", payload)
    out_dir = out_root / tape.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scope.html").write_text(page, encoding="utf-8")
    wav = out_dir / "run.wav"
    if wav.is_symlink() or wav.exists():
        wav.unlink()
    if (run_dir / "run.wav").exists():
        wav.symlink_to(run_dir / "run.wav")
    summary = {"meta": view["meta"], "verify": view["verify"]}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    return {"dir": out_dir, **summary}


def _verdict(v: dict[str, Any]) -> tuple[bool, str]:
    p, th = v["pushes"], v["thinking"]
    ok = (
        p["same"] == p["replay"] == p["record"]
        and th["same"] == th["replay"] == th["record"]
        and all(j["replay"] == j["record"] and not j["unrecorded"] for j in v["judges"].values())
        and not v["progressMismatch"]
        and not v["explainMismatch"]
    )
    text = f"넣기 {p['same']}/{max(p['replay'], p['record'])} · 지시 {th['same']}/{max(th['replay'], th['record'])}"
    return ok, text


def _changes_cell(changes: dict[str, Any] | None) -> str:
    """기록 뒤 바뀐 라디오 코드 — 재생이 기록과 다를 수 있는 이유."""
    if not changes:
        return ""
    n = len(changes["commits"])
    text = (f"커밋 {n}개" if n else "") + (" + 미커밋" if changes["dirty"] else "")
    title = html.escape("\n".join(changes["commits"]))
    return f'<span title="{title}">{html.escape(text) or "없음"}</span>'


def write_index(root: Path) -> Path:
    """``root`` 아래 만들어 둔 회차 전부의 목록 (이번에 만든 것만이 아니라)."""
    built = []
    for f in sorted(root.rglob("summary.json")):
        built.append({"dir": f.parent, **json.loads(f.read_text(encoding="utf-8"))})
    rows = []
    for b in built:
        m, v = b["meta"], b["verify"]
        ok, text = _verdict(v)
        href = b["dir"].relative_to(root).as_posix() + "/scope.html"
        cells = [
            f'<a href="{html.escape(href)}">{html.escape(m["name"])}</a>',
            html.escape(str(m["scenario"]) + (f" · {m['window']}" if m.get("window") else "")),
            html.escape(str(m["utterance"] or "")),
            html.escape(str(m["result"] or "")),
            f"{m['end']:.0f} s",
            html.escape(m.get("recordedAt") or ""),
            _changes_cell(m.get("codeChanges")),
            f'<span class="{"ok" if ok else "warn"}">{"✓" if ok else "⚠"} {html.escape(text)}</span>',
        ]
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    page = INDEX_HTML.replace("__ROWS__", "\n".join(rows)).replace("__ROOT__", html.escape(root.as_posix()))
    out = root / "scope_index.html"
    out.write_text(page, encoding="utf-8")
    return out


INDEX_HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>라디오 스코프</title>
<style>
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--ok:#006300;--warn:#a35f00}
@media (prefers-color-scheme:dark){:root{--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;
--ok:#0ca30c;--warn:#fab219}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:32px 16px}
h1{font-size:20px;font-weight:600;margin:0 0 4px}p{color:var(--ink2);margin:0 0 24px}
table{width:100%;border-collapse:collapse;background:var(--surface);border-radius:8px;overflow:hidden}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--grid);white-space:nowrap}
th{font-weight:500;color:var(--muted);font-size:12px}td{font-variant-numeric:tabular-nums}
a{color:inherit;text-underline-offset:3px}.ok{color:var(--ok)}.warn{color:var(--warn)}
</style></head><body><main>
<h1>라디오 스코프</h1><p>__ROOT__ — 회차를 누르면 재생 화면이 열린다. ✓ 는 재생의 넣기·지시·판정이 기록과 같다는 뜻.</p>
<table><thead><tr><th>회차</th><th>시나리오</th><th>발화</th><th>결과</th><th>길이</th>
<th>기록</th><th>그 뒤 라디오 코드</th><th>재생 = 기록</th></tr></thead>
<tbody>
__ROWS__
</tbody></table></main></body></html>
"""


def cmd_build(args: argparse.Namespace) -> None:
    paths = [Path(p).resolve() for p in args.paths]
    out_root = Path(args.out).resolve()
    built = []
    for run_dir in find_runs(paths):
        try:
            result = build_run(run_dir, out_root)
        except Exception as exc:  # noqa: BLE001 — 한 회차가 깨져도 나머지는 만든다
            print(f"  ✗ {run_dir}: {exc!r}")
            continue
        if result is None:
            print(f"  - {run_dir.name}: 라디오 시작 없음, 건너뜀")
            continue
        ok, text = _verdict(result["verify"])
        print(f"  {'✓' if ok else '⚠'} {result['meta']['name']}  ({text})")
        built.append(result)
    if built:
        print(f"목록: {write_index(out_root)}")


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """``Range`` 요청을 받는 정적 파일 서버 — 브라우저가 녹음 중간으로 건너뛸 수 있게 (기본 핸들러는 늘 전체를 준다)."""

    def send_head(self):  # type: ignore[no-untyped-def]
        self._remaining: int | None = None
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        path = Path(self.translate_path(self.path))
        if m is None or not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        start = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2) or 0))
        end = min(int(m.group(2)), size - 1) if m.group(1) and m.group(2) else size - 1
        if start >= size:
            self.send_error(416, "Requested Range Not Satisfiable")
            return None
        f = path.open("rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):  # type: ignore[no-untyped-def]
        remaining = getattr(self, "_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        while remaining > 0:
            chunk = source.read(min(65536, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


def cmd_serve(args: argparse.Namespace) -> None:
    root = Path(args.path).resolve()
    handler = functools.partial(_RangeHandler, directory=str(root))
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        page = "scope_index.html" if (root / "scope_index.html").exists() else "scope.html"
        print(f"http://127.0.0.1:{args.port}/{page}  (Ctrl+C 로 끝)")
        server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="실험 폴더(들)의 회차마다 scope.html 과 목록 scope_index.html 을 --out 에")
    b.add_argument("paths", nargs="+")
    b.add_argument("--out", default=str(OUT_DIR), help="결과 폴더 (기본 var/radio_scope)")
    b.set_defaults(fn=cmd_build)
    s = sub.add_parser("serve", help="폴더를 로컬 HTTP 로 연다")
    s.add_argument("path", nargs="?", default=str(OUT_DIR))
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(fn=cmd_serve)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
