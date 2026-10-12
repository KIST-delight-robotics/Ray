"""라디오 추적 규칙 재생 평가 — 기록된 전사로 B′ 추적기를 다시 돌려 사후 정답과 맞대 본다. API 를 부르지 않는다.

대상은 지금 넣기 규칙으로 모은 결과다: ``radio_ahead`` 의 l4~l11c(짧은 대본, 끼어들기 있음)와 ``radio_long`` 의
long_t~long_v2(긴 대본). 각 회차의 전사 조각(도착 위치), 넣기·보류·판정·재개 이벤트, 녹음(오른쪽 모델)을 쓴다.

- **사후 정답**: 읽기 구간(첫 넣기 ~ 끼어든 사용자 목소리, 재개 ~ 끝, 긴 대본은 소재마다 첫 대사 ~ 곡)의 모델 전사
  전체를 대본에 아핀 갭 반전역 정렬(양 끝 빈칸 무료)로 맞춰, 대사마다 첫 글자·마지막 문장 첫 글자·마지막 글자가
  들린 시각을 얻는다. 추적기(``difflib`` 일치 덩어리)와 다른 방식이라 추적기를 채점하는 기준으로 쓴다.
- **추적기 재현**: ``exp_radio_ahead.AheadSession`` 의 ``_progress`` / ``_update_done`` / 넣기 판단을 규칙
  (``Rule``)으로 바꿔 쓸 수 있게 옮긴 것. 기록된 이벤트를 그대로 따라가며 돌린다. ``validate`` 가 원래 메서드·기록과
  같은지 확인한다.

하위 명령:

- ``validate``: 재현한 진행 계산이 원래 메서드와 같은지, 재현한 넣기 시각이 기록과 같은지.
- ``push``: 넣기마다 그 규칙이 처음 "넣을 때"라고 할 시각을 사후 정답과 비교 — 마지막 문장 안 / 이름(마지막 문장
  전) / 늦음(대사 끝 뒤) / 앞서 넣지 못함(무음 규칙으로 넘어감). 규칙 변형별로.
- ``stall``: 멈춘 뒤 무엇을 다시 넣을지 — 앞서 넣지 못한 경우(모델이 말을 마친 시점)와 실제 다시 넣기 시점에서,
  실험 규칙(남은 2자)과 정한 규칙(확실할 때만, on 판정 뒤 대답은 뺌) 및 그 사이 변형의 판단을 사후 정답과 비교.
- ``pauses``: 대사 안 무음 길이 분포와 무음 문턱별로 잘못 걸렸을 쉼(앱이 아무것도 넣지 않았는데 모델이 스스로 이음).
- ``done``: 끝남 판정 시각(일치 최소 글자 5 / 8)과 곡 시작이 곡 소개 끝 뒤였는지.

    uv run python scripts/gpt_live/radio_replay.py push
"""

from __future__ import annotations

import argparse
import collections
import copy
import difflib
import json
import re
import wave
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import exp_radio_ahead as ea
import numpy as np
from exp_dirs import EXPERIMENTS
from exp_radio_tts import sentences

SHORT = [
    "11_l4", "13_l5n", "14_l5", "15_l6c", "16_l6g", "17_l7", "18_l7d", "19_l8c", "20_l8r",
    "21_l9", "22_l9n", "23_l10", "24_l10n", "25_l11", "26_l11c",
]  # fmt: skip
LONG = ["02_long_t", "03_long_s", "04_long_r", "05_long_v1", "06_long_v2"]
FIXED_PUNCT = ("05_long_v1", "06_long_v2")  # 부호 신호 시작점 고친 뒤의 실행
GROUPS = {"11_l4": "l4", **{f: "long_t~r" for f in LONG[:3]}, **{f: "long_v" for f in LONG[3:]}}  # 나머지는 l5~l11c
FRAME = 0.02
RATE = 24000
QUIET_RMS = 30
TERMINATOR = re.compile(r"[.?!…]+")
DECIDED_TYPES = ("text", "letters", "punct", "stall")  # 넣기 판단으로 생긴 넣기 (시작·재개·소재 첫 대사 말고)
NEG = -1e9
SAID = 0.5  # stall: 문장 글자의 이만큼 이상이 글자 그대로 맞으면 그 문장을 말한 것으로 친다


def _letters(text: str) -> str:
    return re.sub(r"[\W_]+", "", text)


# ---------------------------------------------------------------------------
# 규칙
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """추적·넣기·멈춤 규칙. 기본값은 정한 규칙(10-12)."""

    block: int = 8  # 진행으로 칠 전사·대본 일치 최소 길이
    slack: int = 4  # 대사 끝 이만큼 안까지 일치하면 끝난 것
    remain: int = 13  # 마지막 문장에 들어서 남은 글자가 이만큼 이하면 다음 대사
    drift: bool = False  # 넣기 판단에 마지막 일치 뒤 더 말한 글자를 더한다
    punct: str | None = None  # 문장 끝 부호 신호 — None / "old"(시작점 고치기 전) / "fix"
    cut: int = 8  # 멈췄을 때 문장 끝까지 이만큼 이상 남은 문장만 다시 넣는다
    said_check: bool = True  # 끊김으로 보려면 마지막 일치 뒤 말한 글자가 남은 양의 절반 미만이어야 한다
    said_check_user: bool = (
        True  # 이 대사를 읽는 동안 사용자가 말했어도 위 확인을 한다 (끄면 그 대사는 남은 cut 자만 본다)
    )
    exclude_reply: bool = True  # 보류가 풀린 뒤 모델 말도 대본과 다시 일치할 때까지 진행에서 뺀다 (on 판정 뒤 대답)


DECIDED = Rule()
EXPERIMENT = Rule(block=5, drift=True, punct="fix", cut=2, said_check=False)  # exp_radio_ahead.py 지금 규칙


def as_run(cond: str) -> Rule:
    """그 실행이 쓴 규칙 (부호 신호 시작점만 실행마다 다르다)."""
    return replace(EXPERIMENT, punct="fix" if cond in FIXED_PUNCT else "old")


# ---------------------------------------------------------------------------
# 기록
# ---------------------------------------------------------------------------


@dataclass
class Run:
    name: str
    kind: str  # short / long
    cond: str
    units: list[str]
    deltas: list[tuple[float, str, str]]
    events: list[dict[str, Any]]
    wav: Path
    raw: dict[str, Any]
    seg_first: list[int] | None = None
    songs: list[dict[str, Any]] | None = None
    _voiced: np.ndarray | None = field(default=None, repr=False)

    @property
    def group(self) -> str:
        return GROUPS.get(self.cond, "l5~l11c")

    def voiced(self) -> np.ndarray:
        """모델 채널 20 ms 마다 rms 30 이상인지 (앱의 무음 판정과 같은 문턱)."""
        if self._voiced is None:
            with wave.open(str(self.wav)) as w:
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, 2)[:, 1]
            x = pcm.astype(np.float32)
            hop = int(RATE * FRAME)
            n = len(x) // hop
            self._voiced = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1)) >= QUIET_RMS
        return self._voiced

    def pushes(self) -> list[dict[str, Any]]:
        return sorted((e for e in self.events if e["name"] == "push"), key=lambda e: e["pos"])


def load_runs() -> list[Run]:
    runs = []
    for folder in SHORT:
        d = EXPERIMENTS / "radio_ahead" / folder
        for f in sorted(d.glob("*.json")):
            if f.name == "config.json":
                continue
            for r in json.load(open(f))["results"]:
                wav = d / f"{f.stem}_{r['run']}.wav"
                if wav.exists() and r.get("deltas"):
                    runs.append(_run(f"{folder}/{f.stem}_{r['run']}", "short", folder, r, wav))
    for folder in LONG:
        for f in sorted((EXPERIMENTS / "radio_long" / folder).glob("run_*.json")):
            r = json.load(open(f))["result"]
            run = _run(f"{folder}/{f.stem}", "long", folder, r, f.with_suffix(".wav"))
            run.seg_first, run.songs = r["seg_first"], r["songs"]
            runs.append(run)
    return runs


def _run(name: str, kind: str, cond: str, r: dict[str, Any], wav: Path) -> Run:
    return Run(name, kind, cond, [t for _, t in r["units"]], [tuple(x) for x in r["deltas"]], r["events"], wav, r)


# ---------------------------------------------------------------------------
# 사후 정답
# ---------------------------------------------------------------------------


def align(
    t: str, s: str, ma: float = 2.0, mi: float = -1.0, go: float = -3.0, ge: float = -0.5
) -> list[tuple[int, int]]:
    """전사 글자열 ``t`` 와 대본 글자열 ``s`` 의 정렬 쌍 (전사 자리, 대본 자리). 아핀 갭, 양 끝 빈칸은 무료.

    바꿔 말한 글자는 어긋남으로, 덧붙인 말·건너뛴 대본은 빈칸으로 들어간다.
    """
    n, m = len(t), len(s)
    if not n or not m:
        return []
    tv = np.array([ord(c) for c in t])
    sv = np.array([ord(c) for c in s])
    M = np.full((n + 1, m + 1), NEG)  # t[i-1] 과 s[j-1] 을 맞춤
    X = np.full((n + 1, m + 1), NEG)  # s[j-1] 을 건너뜀
    Y = np.full((n + 1, m + 1), NEG)  # t[i-1] 을 덧붙임
    M[0, 0], X[0, 1:], Y[1:, 0] = 0.0, 0.0, 0.0
    ks = np.arange(m + 1, dtype=np.float64)
    for i in range(1, n + 1):
        hp = np.maximum(np.maximum(M[i - 1], X[i - 1]), Y[i - 1])
        M[i, 1:] = hp[:-1] + np.where(sv == tv[i - 1], ma, mi)
        Y[i, 1:] = np.maximum(np.maximum(M[i - 1, 1:] + go, Y[i - 1, 1:] + ge), X[i - 1, 1:] + go)
        # X[i, j] = max_{k<j} (max(M, Y)[i, k] + go + ge*(j-1-k)) — 누적 최대로 한 번에
        X[i, 1:] = ge * ks[:-1] + np.maximum.accumulate(np.maximum(M[i], Y[i]) + go - ge * ks)[:-1]
    H = np.maximum(np.maximum(M, X), Y)
    j_row, i_col = int(np.argmax(H[n])), int(np.argmax(H[:, m]))
    i, j = (n, j_row) if H[n, j_row] >= H[i_col, m] else (i_col, m)

    def state(i: int, j: int, v: float) -> str:
        return "M" if abs(M[i, j] - v) < 1e-6 else "Y" if abs(Y[i, j] - v) < 1e-6 else "X"

    st = state(i, j, H[i, j])
    pairs = []
    while i > 0 and j > 0:
        if st == "M":
            pairs.append((i - 1, j - 1))
            v = M[i, j] - (ma if t[i - 1] == s[j - 1] else mi)
            i, j = i - 1, j - 1
            st = state(i, j, v)
        elif st == "Y":
            v = Y[i, j]
            st = "M" if abs(M[i - 1, j] + go - v) < 1e-6 else "Y" if abs(Y[i - 1, j] + ge - v) < 1e-6 else "X"
            i -= 1
        else:
            v = X[i, j]
            st = "X" if abs(X[i, j - 1] + ge - v) < 1e-6 else "M" if abs(M[i, j - 1] + go - v) < 1e-6 else "Y"
            j -= 1
    return pairs[::-1]


@dataclass
class UnitGT:
    first: float | None  # 대사 첫 글자가 들린 시각
    last_start: float | None  # 마지막 문장 첫 글자가 들린 시각
    end: float | None  # 대사 마지막 글자가 들린 시각
    last_exact: float  # 마지막 문장 글자 중 글자 그대로 맞은 비율
    last_said: float  # 마지막 문장 글자 중 무언가와 맞춰진 비율 (바꿔 말함 포함)
    said: list[float]  # 문장마다 글자 그대로 맞은 비율 — 바꿔 말하면 낮아지고, 엉뚱한 말이면 0 에 가깝다


def span_gt(run: Run, t0: float, t1: float, u0: int, u1: int) -> dict[int, UnitGT]:
    """[t0, t1) 에 도착한 모델 전사를 대사 u0..u1-1 에 맞춘 사후 정답."""
    text, at = "", []
    for p, sp, d in run.deltas:
        if sp == "assistant" and t0 <= p < t1:
            for ch in d:
                if _letters(ch):
                    text += ch
                    at.append(p)
    script = "".join(_letters(run.units[u]) for u in range(u0, u1))
    hit: list[float | None] = [None] * len(script)
    exact = [False] * len(script)
    for a, b in align(text, script):
        hit[b], exact[b] = at[a], text[a] == script[b]
    out, lo = {}, 0
    for u in range(u0, u1):
        sents = [len(_letters(x)) for x in sentences(run.units[u])]
        hi = lo + sum(sents)
        ls = hi - sents[-1]
        idx = [q for q in range(lo, hi) if hit[q] is not None]
        lidx = [q for q in range(ls, hi) if hit[q] is not None]
        said, a = [], lo
        for k in sents:
            said.append(sum(exact[a : a + k]) / max(1, k))
            a += k
        out[u] = UnitGT(
            first=hit[idx[0]] if idx else None,
            last_start=hit[lidx[0]] if lidx else None,
            end=max(hit[q] for q in idx) if idx else None,
            last_exact=sum(exact[ls:hi]) / max(1, hi - ls),
            last_said=len(lidx) / max(1, hi - ls),
            said=said,
        )
        lo = hi
    return out


def spans(run: Run) -> list[tuple[float, float, int, int]]:
    """사후 정답을 낼 읽기 구간 (t0, t1, u0, u1)."""
    pushes = run.pushes()
    if not pushes:
        return []
    if run.kind == "long":
        out = []
        for s in range(len(run.seg_first) - 1):
            first = [e for e in pushes if e["unit"] - 1 == run.seg_first[s]]
            if not first:
                break
            t1 = run.songs[s]["start"] if s < len(run.songs) else float("inf")
            out.append((first[0]["pos"], t1, run.seg_first[s], run.seg_first[s + 1]))
        return out
    turns = run.raw.get("user_turns") or []
    if not any(e["name"] == "decision" and e.get("state") in ("paused", "off") for e in run.events):
        return [(pushes[0]["pos"], float("inf"), 0, len(run.units))]  # on 판정·끼어들기 없음 — 대답은 덧붙임으로 맞춘다
    out = [(pushes[0]["pos"], turns[0]["start"] + turns[0]["voice"][0], 0, len(run.units))]
    resumed = [e for e in pushes if e.get("by") == "resume"]
    if resumed:
        out.append((resumed[0]["pos"], float("inf"), resumed[0]["unit"] - 1, len(run.units)))
    return out


def run_gt(run: Run) -> dict[tuple[int, int], UnitGT]:
    return {(si, u): g for si, sp in enumerate(spans(run)) for u, g in span_gt(run, *sp).items()}


def span_of(run: Run, t: float, unit: int) -> int | None:
    return next((i for i, (t0, t1, u0, u1) in enumerate(spans(run)) if t0 <= t < t1 and u0 <= unit < u1), None)


# ---------------------------------------------------------------------------
# 추적기 재현
# ---------------------------------------------------------------------------


@dataclass
class State:
    units: list[str]
    deltas: list[tuple[float, str, str]] = field(default_factory=list)
    appends: list[float] = field(default_factory=list)
    done: int = 0
    track_from: float = 0.0
    track_unit: int = 0
    excluded: list[list[float]] = field(default_factory=list)
    status: str = "on"
    holding: bool = False
    song: bool = False
    seg_end: int = 10**9
    repushed: int = -1
    repushed_at: float = 0.0
    first_voice: float | None = None
    quiet: float = 0.0
    t: float = 0.0

    def __post_init__(self) -> None:
        self.unit_letters = [_letters(t) for t in self.units]
        self.offsets = [0]
        for x in self.unit_letters:
            self.offsets.append(self.offsets[-1] + len(x))
        self.unit_sentences = [len(sentences(t)) for t in self.units]
        last = [_letters(sentences(t)[-1]) for t in self.units]
        self.last_len = [len(s) for s in last]
        self.last_start = [self.offsets[i + 1] - len(s) for i, s in enumerate(last)]
        self._cache: tuple[Any, dict[str, Any]] | None = None

    def clone(self) -> State:
        c = copy.copy(self)
        c.deltas, c.appends = list(self.deltas), list(self.appends)
        c.excluded = [list(x) for x in self.excluded]
        c._cache = None
        return c


def progress(st: State, rule: Rule) -> dict[str, Any]:
    """``AheadSession._progress`` 와 같은 계산. ``rule.punct`` 가 "old" 면 고치기 전 시작점."""
    key = (
        len(st.deltas), len(st.appends), st.done, st.track_from, str(st.excluded), rule.block, rule.punct,
        rule.exclude_reply,
    )  # fmt: skip
    if st._cache is not None and st._cache[0] == key:
        return st._cache[1]
    base = min(max(st.track_unit, st.done - 1), len(st.appends) - 1)
    since = max(st.track_from, st.appends[base])
    raw, at = "", []
    for p, sp, d in st.deltas:
        if sp == "assistant" and p >= since:
            raw += d
            at += [p] * len(d)
    idx = [i for i, ch in enumerate(raw) if _letters(ch)]
    text = "".join(raw[i] for i in idx)
    script = "".join(st.unit_letters[base : len(st.appends)])
    last = len(st.appends) - 1
    reach = reach_t = 0
    last_from = None
    blocks = [
        b
        for b in difflib.SequenceMatcher(None, text, script, autojunk=False).get_matching_blocks()
        if b.size >= rule.block
    ]
    excluded = st.excluded
    if rule.exclude_reply:
        # 보류가 풀린 뒤의 말은 대본과 다시 일치하는 곳(그때 읽던 일치가 이어지면 풀린 때)까지 뺀다
        spans_t = [(at[idx[b.a]], at[idx[b.a + b.size - 1]]) for b in blocks]
        excluded = []
        for a, e in st.excluded:
            if e != float("inf") and not any(t0 <= e <= t1 for t0, t1 in spans_t):
                e = min((t0 for t0, _ in spans_t if t0 >= e), default=float("inf"))
            excluded.append([a, e])
    free = [not any(a <= q < b for a, b in excluded) for q in at]
    for b in blocks:
        reach, reach_t = b.b + b.size, b.a + b.size
        at_script = st.offsets[base] + b.b
        if last_from is None and at_script + b.size > st.offsets[last]:
            if rule.punct == "old":
                last_from = max(0, b.a + st.offsets[last] - at_script)
            else:
                last_from = b.a + max(0, st.offsets[last] - at_script)
    drift = sum(free[i] for i in idx[reach_t:]) if reach_t else 0
    punct_remain = None
    n = st.unit_sentences[last]
    if rule.punct and last_from is not None and n > 1 and last_from < len(idx):
        ends = [m.end() for m in TERMINATOR.finditer(raw, idx[last_from]) if free[m.start()]]
        if len(ends) >= n - 1:
            punct_remain = st.last_len[last] - sum(free[i] for i in idx if i >= ends[n - 2])
    out = {
        "reach": st.offsets[base] + reach,
        "est": st.offsets[base] + reach + drift,
        "drift": drift,
        "punct_remain": punct_remain,
    }
    st._cache = (key, out)
    return out


def update_done(st: State, rule: Rule) -> list[int]:
    if not st.appends:
        return []
    reach = progress(st, rule)["reach"]
    newly = []
    while st.done < len(st.appends) and reach >= st.offsets[st.done + 1] - rule.slack:
        st.done += 1
        newly.append(st.done)
    return newly


def push_signal(st: State, rule: Rule) -> str | None:
    """앞서 넣을 때면 근거(text / letters / punct), 아니면 None. 무음 규칙은 빼고."""
    n = len(st.appends)
    if not n or n >= min(len(st.units), st.seg_end):
        return None
    last = n - 1
    prog = progress(st, rule)
    est = prog["est"] if rule.drift else prog["reach"]
    if est > st.last_start[last] and st.offsets[last + 1] - est <= rule.remain:
        return "text" if prog["drift"] <= rule.slack or not rule.drift else "letters"
    if prog["punct_remain"] is not None and prog["punct_remain"] <= rule.remain:
        return "punct"
    return None


def stall_repush(st: State, rule: Rule) -> list[str]:
    """멈췄을 때(넣은 뒤 말하고 조용함) 다시 넣을 문장. 빈 목록이면 다음 대사.

    마지막으로 넣은 대사와 그 앞 대사에서 문장 끝까지 ``rule.cut`` 자 이상 남은 문장(끊긴 문장은 처음부터).
    ``rule.said_check`` 면 남은 글자가 ``rule.cut`` 보다 적거나, 마지막 일치 뒤로 말한 글자가 남은 양의 절반 이상일
    때 — 바꿔 말하며 다 읽었을 때 — 는 다음 대사로 본다. on 판정 뒤 대답은 ``rule.exclude_reply`` 로 말한 글자에서
    빠진다. ``rule.said_check_user`` 를 끄면 이 대사를 읽는 동안 사용자가 말했을 때 이 확인을 하지 않는다.
    ``EXPERIMENT`` 규칙이면 ``AheadSession._unread`` 와 같다.
    """
    prog = progress(st, rule)
    last, reach = len(st.appends) - 1, prog["reach"]
    remain = st.offsets[last + 1] - reach
    base = min(max(st.track_unit, st.done - 1), last)
    user_spoke = any(a >= st.appends[base] for a, _ in st.excluded)
    if (
        rule.said_check
        and (rule.said_check_user or not user_spoke)
        and (remain < rule.cut or 2 * prog["drift"] >= remain)
    ):
        return []
    out = []
    for u in range(max(0, last - 1), last + 1):
        end = st.offsets[u]
        for s in sentences(st.units[u]):
            end += len(_letters(s))
            if end - reach >= rule.cut:
                out.append(s)
    return out


def can_push(st: State) -> bool:
    return st.status == "on" and not st.holding and not st.song and 0 < len(st.appends) < min(len(st.units), st.seg_end)


def apply_event(st: State, e: dict[str, Any], seg_first: list[int] | None) -> None:
    name, pos = e["name"], e["pos"]
    if name == "push":
        u = e["unit"]
        st.appends = st.appends[: u - 1] + [pos]  # 재개 때 읽히지 않은 대사를 지우고 다시 넣은 것까지
        st.first_voice = None
        if e.get("by") == "segment" and seg_first is not None:
            st.done = max(st.done, u - 1)
            st.track_from, st.track_unit = pos, u - 1
            st.song = False
            st.seg_end = next((x for x in seg_first if x > u - 1), len(st.units))
        elif e.get("by") == "resume":
            st.track_from, st.track_unit = pos, st.done
    elif name == "repush":
        st.repushed, st.repushed_at = e["unit"] - 1, pos
    elif name == "hold":
        st.holding = True
        st.excluded.append([pos, float("inf")])
    elif name in ("decision", "hold_release"):
        st.holding = False
        if st.excluded and st.excluded[-1][1] == float("inf"):
            st.excluded[-1][1] = pos
    elif name == "status":
        st.status = e["status"]
    elif name == "start":
        st.status = "on"
    elif name == "song_start":
        st.song = True


def initial_state(run: Run) -> State:
    st = State(units=run.units)
    if run.kind == "long":
        st.seg_end = run.seg_first[1]
    if run.raw.get("start_ask") is not None:
        st.status = "off"
    return st


def _after_decision(e: dict[str, Any]) -> bool:
    """원래 실험에서 넣기 판단 뒤에 생기는 이벤트 — 같은 위치면 판단보다 나중에 적용한다."""
    return (e["name"] == "push" and e.get("by") in DECIDED_TYPES) or e["name"] == "repush"


def snapshots(run: Run, rule: Rule) -> list[tuple[dict[str, Any], State]]:
    """기록된 넣기마다 (넣기 이벤트, 넣은 직후 상태)."""
    st = initial_state(run)
    deltas = sorted(run.deltas, key=lambda x: x[0])
    out, di = [], 0
    for e in sorted(run.events, key=lambda e: (e["pos"], _after_decision(e))):
        while di < len(deltas) and deltas[di][0] <= e["pos"] + 1e-9:
            st.deltas.append(deltas[di])
            di += 1
        st.t = e["pos"]
        update_done(st, rule)
        apply_event(st, e, run.seg_first)
        if e["name"] == "push":
            snap = st.clone()
            snap.t = e["pos"]
            out.append((e, snap))
    return out


def advance(run: Run, st: State, t: float, rule: Rule) -> None:
    """넣기 없이 ``t`` 까지 전사와 보류·판정 이벤트만 더한다."""
    for e in sorted(run.events, key=lambda e: e["pos"]):
        if st.t < e["pos"] <= t and e["name"] in ("hold", "decision", "hold_release", "status"):
            apply_event(st, e, None)
    st.deltas += sorted((d for d in run.deltas if st.t < d[0] <= t + 1e-9), key=lambda x: x[0])
    st.t = t
    update_done(st, rule)


def first_push(run: Run, st0: State, rule: Rule, t_stop: float) -> tuple[float, str] | None:
    """넣은 직후 상태 ``st0`` 에서 전사를 하나씩 더하며, 규칙이 처음 앞서 넣을 때라고 하는 (시각, 근거)."""
    st = st0.clone()
    if can_push(st) and (by := push_signal(st, rule)):
        return st.t, by
    deltas = sorted((d for d in run.deltas if st0.t < d[0] <= t_stop), key=lambda x: x[0])
    events = sorted(
        (
            e
            for e in run.events
            if st0.t < e["pos"] <= t_stop and e["name"] in ("hold", "decision", "hold_release", "status")
        ),
        key=lambda e: e["pos"],
    )
    di = ei = 0
    while di < len(deltas):
        t = deltas[di][0]
        while ei < len(events) and events[ei]["pos"] <= t:
            apply_event(st, events[ei], None)
            ei += 1
        while di < len(deltas) and deltas[di][0] <= t + 1e-9:
            st.deltas.append(deltas[di])
            di += 1
        st.t = t
        update_done(st, rule)
        if can_push(st) and (by := push_signal(st, rule)):
            return t, by
    return None


def timeline(run: Run, rule: Rule, stall_sec: float, stall_mode: str) -> dict[str, Any]:
    """기록된 넣기를 따라가며 프레임마다 넣기 판단을 돌린다 — 넣은 대사 수마다 규칙이 처음 넣자고 한 (시각, 근거)
    와 끝남 시각. 무음 규칙은 실험의 ``--stall`` 과 같게."""
    st = initial_state(run)
    voiced = run.voiced()
    deltas = sorted(run.deltas, key=lambda x: x[0])
    events = sorted(run.events, key=lambda e: e["pos"])
    early = [e for e in events if not _after_decision(e)]
    late = [e for e in events if _after_decision(e)]
    di = ei = li = 0
    fired: dict[int, tuple[float, str]] = {}
    done_at: dict[int, float] = {}
    for f in range(len(voiced)):
        t = round(f * FRAME, 2)
        st.t = t
        while ei < len(early) and early[ei]["pos"] <= t + 1e-9:
            apply_event(st, early[ei], run.seg_first)
            ei += 1
        while di < len(deltas) and deltas[di][0] <= t + 1e-9:
            st.deltas.append(deltas[di])
            di += 1
        if voiced[f]:
            if st.appends and st.first_voice is None and t >= st.appends[-1]:
                st.first_voice = t
            st.quiet = 0.0
        else:
            st.quiet += FRAME
        for u in update_done(st, rule):
            done_at.setdefault(u, t)
        n = len(st.appends)
        if can_push(st) and n not in fired:
            by = push_signal(st, rule)
            if by is None and st.first_voice is not None and st.quiet >= stall_sec - 1e-9:
                last = n - 1
                if stall_mode == "sentence":
                    unread = stall_repush(st, rule)
                    if not unread or (st.repushed == last and t - st.repushed_at >= ea.REPUSH_WAIT_SEC):
                        by = "stall"
                elif progress(st, rule)["est"] > st.last_start[last] or st.quiet >= 2 * stall_sec - 1e-9:
                    by = "stall"
            if by is not None:
                fired[n] = (t, by)
        while li < len(late) and late[li]["pos"] <= t + 1e-9:
            apply_event(st, late[li], run.seg_first)
            li += 1
    return {"fired": fired, "done_at": done_at}


# ---------------------------------------------------------------------------
# 넣기 위치 판정
# ---------------------------------------------------------------------------

PLACE = {"in": "마지막 문장 안", "early": "이름", "late": "늦음", "never": "앞서 못 넣음", "unknown": "모름"}


def place(t: float | None, g: UnitGT | None) -> tuple[str, float | None]:
    """앞 대사(읽던 대사)의 사후 정답과 비교한 넣기 위치와 여유(대사 끝까지 초)."""
    if g is None or g.end is None or g.last_start is None:
        return "unknown", None
    if t is None:
        return "never", None
    lead = round(g.end - t, 2)
    return ("early" if t < g.last_start - 1e-6 else "late" if t > g.end + 1e-6 else "in"), lead


def opportunities(run: Run) -> list[dict[str, Any]]:
    """넣기 판단으로 생긴 기록된 넣기마다: 읽던 대사의 사후 정답과 그 대사를 넣은 직후 상태."""
    gts = run_gt(run)
    snaps = snapshots(run, as_run(run.cond))
    out = []
    for e in run.pushes():
        if e.get("by") not in DECIDED_TYPES:
            continue
        k = e["unit"] - 1  # 넣은 대사. 읽던 대사는 k-1
        si = span_of(run, e["pos"], k - 1)
        prev = [(pe, st) for pe, st in snaps if pe["unit"] == k and pe["pos"] <= e["pos"]]
        if si is None or not prev:
            continue
        g = gts.get((si, k - 1))
        holds = [x["pos"] for x in run.events if x["name"] == "hold" and x["pos"] > prev[-1][0]["pos"]]
        t_stop = min([(g.end if g and g.end else e["pos"]) + 3.0, *holds])
        out.append({"event": e, "unit": k, "gt": g, "state": prev[-1][1], "t_stop": t_stop})
    return out


# ---------------------------------------------------------------------------
# 하위 명령
# ---------------------------------------------------------------------------


def _map(fn: Any, runs: list[Run]) -> list[Any]:
    with ProcessPoolExecutor() as ex:
        return list(ex.map(fn, runs, chunksize=2))


def _validate_run(run: Run) -> dict[str, Any]:
    rule = as_run(run.cond)
    same = total = 0
    for _, st in snapshots(run, rule):
        if not st.appends:
            continue
        obj = object.__new__(ea.AheadSession)
        obj.units = [(0, t) for t in run.units]
        for k in ("unit_letters", "offsets", "unit_sentences", "last_len", "last_start", "deltas", "appends", "done"):
            setattr(obj, k, getattr(st, k))
        obj.track_from, obj.track_unit, obj.excluded = st.track_from, st.track_unit, st.excluded
        obj._progress_cache = None
        mine = progress(st, replace(rule, punct="fix"))
        st._cache = None
        total += 1
        same += ea.AheadSession._progress(obj) == mine
    cfg = json.load(open(run.wav.parent / "config.json"))
    if run.kind == "long":
        stall_mode, stall_sec = cfg.get("stall", "remainder"), cfg.get("stall_sec", 2.0)
    else:
        stall_mode, stall_sec = cfg.get("stall_mode") or "next", cfg.get("stall", 2.0)
    fired = timeline(run, rule, stall_sec, stall_mode)["fired"]
    rows = []
    for i, e in enumerate(run.pushes()):
        if e.get("by") not in DECIDED_TYPES:
            continue
        f = fired.get(e["unit"] - 1)
        # 재개 뒤 같은 대사 수가 다시 오면 앞 구간의 판단이 남아 있다 — 같은 대사 수의 첫 넣기만 맞댄다
        if any(p["unit"] == e["unit"] and p["by"] in DECIDED_TYPES for p in run.pushes()[:i]):
            continue
        rows.append((e["by"], f is not None and abs(f[0] - e["pos"]) <= 0.11))
    return {"same": same, "total": total, "rows": rows}


def cmd_validate(runs: list[Run]) -> None:
    res = _map(_validate_run, runs)
    print(f"진행 계산이 원래 메서드와 같음: {sum(r['same'] for r in res)}/{sum(r['total'] for r in res)}")
    c = collections.Counter((by, ok) for r in res for by, ok in r["rows"])
    for by in DECIDED_TYPES:
        print(f"  넣기 {by:7s}: 0.1 s 안에서 재현 {c[(by, True)]}/{c[(by, True)] + c[(by, False)]}")


VARIANTS = {
    "실험 규칙(실행 그대로)": None,
    "실험 규칙(부호 시작점 고침)": EXPERIMENT,
    "실험 규칙, 4자": replace(EXPERIMENT, block=4),
    "부호 뺌": replace(EXPERIMENT, punct=None),
    "글자 수 뺌": replace(EXPERIMENT, drift=False),
    "둘 다 뺌 (5자)": replace(EXPERIMENT, drift=False, punct=None),
    "정한 규칙 (둘 다 뺌, 8자)": DECIDED,
    "둘 다 뺌, 10자": replace(DECIDED, block=10),
    "둘 다 뺌, 4자": replace(DECIDED, block=4),
    "정한 규칙, 남은 11자": replace(DECIDED, remain=11),
    "정한 규칙, 남은 15자": replace(DECIDED, remain=15),
}


def _push_run(run: Run) -> list[dict[str, Any]]:
    rows = []
    for o in opportunities(run):
        row = {"run": run.name, "group": run.group, "unit": o["unit"], "rec_by": o["event"]["by"], "g": o["gt"]}
        row["rec"] = place(o["event"]["pos"], o["gt"])
        for name, rule in VARIANTS.items():
            fp = first_push(run, o["state"], rule or as_run(run.cond), o["t_stop"])
            row[name] = (*place(fp[0] if fp else None, o["gt"]), fp[1] if fp else None)
        rows.append(row)
    return rows


def cmd_push(runs: list[Run]) -> None:
    rows = [r for rs in _map(_push_run, runs) for r in rs]
    keys = ["in", "early", "late", "never", "unknown"]
    print(f"넣기 판단 {len(rows)}회 — 앞 대사 사후 정답과 비교 (앞서 못 넣음 = 무음 규칙으로 넘어감)")
    print(f"{'':28s}" + "".join(f"{PLACE[k]:>12s}" for k in keys) + f"{'여유<1s':>10s}")

    def line(label: str, vals: list[tuple[str, float | None]]) -> None:
        c = collections.Counter(v[0] for v in vals)
        tight = sum(1 for v in vals if v[0] == "in" and v[1] < 1.0)
        print(f"{label:28s}" + "".join(f"{c[k]:12d}" for k in keys) + f"{tight:10d}")

    line("기록된 넣기", [r["rec"] for r in rows])
    for name in VARIANTS:
        line(name, [r[name] for r in rows])
    print("\n기록된 넣기의 근거별:")
    for by in DECIDED_TYPES:
        line(f"  {by}", [r["rec"] for r in rows if r["rec_by"] == by])
    print("\n묶음별 (실험 규칙 → 정한 규칙):")
    for grp in ("l4", "l5~l11c", "long_t~r", "long_v"):
        a = collections.Counter(r["실험 규칙(부호 시작점 고침)"][0] for r in rows if r["group"] == grp)
        b = collections.Counter(r["정한 규칙 (둘 다 뺌, 8자)"][0] for r in rows if r["group"] == grp)
        print(f"  {grp:9s} " + ", ".join(f"{PLACE[k]} {a[k]}→{b[k]}" for k in keys if a[k] or b[k]))
    print("\n실험 규칙과 정한 규칙의 판정이 다른 넣기:")
    for r in rows:
        a, b = r["실험 규칙(부호 시작점 고침)"], r["정한 규칙 (둘 다 뺌, 8자)"]
        if a[0] != b[0]:
            print(f"  {r['run']:26s} {r['unit']:2d}번 대사: {PLACE[a[0]]}({a[2]}) → {PLACE[b[0]]}")


def _stall_run(run: Run) -> list[dict[str, Any]]:
    rows = []
    # 정한 규칙이 앞서 넣지 못하는 경우 — 모델이 말을 마친 시점(사후 정답의 대사 끝)에서 판단
    for o in opportunities(run):
        g = o["gt"]
        if g is None or g.end is None or first_push(run, o["state"], DECIDED, o["t_stop"]):
            continue
        verdicts = {}
        for name, rule in STALL_RULES.items():
            st = o["state"].clone()
            advance(run, st, g.end, rule)
            verdicts[name] = stall_repush(st, rule)
        rows.append({"kind": "missed", "run": run.name, "unit": o["unit"] - 1, "said": g.said, **verdicts})
    # 실제 다시 넣기 — 그 순간까지 말한 것을 사후 정답으로 (이후 다시 읽은 것과 섞이지 않게)
    snaps = snapshots(run, as_run(run.cond))
    for e in run.events:
        if e["name"] != "repush":
            continue
        u = e["unit"] - 1
        prev = [st for pe, st in snaps if pe["unit"] == e["unit"] and pe["pos"] <= e["pos"]]
        si = span_of(run, e["pos"], u)
        if not prev or si is None:
            continue
        t0, _, u0, u1 = spans(run)[si]
        verdicts = {}
        for name, rule in STALL_RULES.items():
            st = prev[-1].clone()
            advance(run, st, e["pos"], rule)
            verdicts[name] = stall_repush(st, rule)
        first = sentences(e["text"])[0]
        # 다시 넣은 글은 앞 대사의 끊긴 문장부터일 수 있다
        said_first = 0.0
        for uu in (u - 1, u):
            gg = span_gt(run, t0, e["pos"] + 1e-6, u0, u1).get(uu) if uu >= u0 else None
            sents = sentences(run.units[uu]) if uu >= 0 else []
            if gg and first in sents:
                said_first = gg.said[sents.index(first)]
                break
        rows.append({
            "kind": "repush", "run": run.name, "unit": u, "first": first, "said_first": said_first,
            **verdicts,
        })  # fmt: skip
    return rows


STALL_RULES = {
    "experiment": EXPERIMENT,
    "decided_always": replace(DECIDED, exclude_reply=False),
    "decided": replace(DECIDED, said_check_user=False, exclude_reply=False),
    "reply": DECIDED,
}
STALL_LABELS = {
    "experiment": "① 실험 규칙(남은 2자)",
    "decided_always": "② 확실할 때만",
    "decided": "③ ② + on 판정이 난 대사는 ①",
    "reply": "④ ② + on 판정 뒤 대답은 다시 일치할 때까지 진행에서 뺌 (정한 규칙)",
}


def cmd_stall(runs: list[Run]) -> None:
    rows = [r for rs in _map(_stall_run, runs) for r in rs]
    missed = [r for r in rows if r["kind"] == "missed"]
    print(f"정한 규칙이 앞서 넣지 못한 {len(missed)}회 — 모델이 말을 마친 시점의 멈춤 판단")
    print(
        f"  (말함 = 사후 정답에서 마지막 문장 글자의 {SAID:.0%} 이상이 글자 그대로 맞음 — 바꿔 말한 문장은 들고, "
        "대답 같은 엉뚱한 말은 빠진다)"
    )
    for name, label in STALL_LABELS.items():
        c = collections.Counter(("말함" if r["said"][-1] >= SAID else "안 말함", bool(r[name])) for r in missed)
        print(
            f"  {label}: 말한 문장을 다시 넣음 {c[('말함', True)]}, 말함→다음 {c[('말함', False)]}, "
            f"안 말한 문장을 다시 넣음 {c[('안 말함', True)]}, 안 말함→다음 {c[('안 말함', False)]}"
        )
    for r in missed:
        verdicts = {name: "다시 넣기" if r[name] else "다음" for name in STALL_RULES}
        said = r["said"][-1] >= SAID
        if len(set(verdicts.values())) > 1 or (said and r["reply"]) or (not said and not r["reply"]):
            print(f"    {r['run']:26s} {r['unit'] + 1:2d}번 대사 마지막 문장 글자 그대로 {r['said'][-1]:.0%}: ", end="")
            print(" / ".join(f"{STALL_LABELS[n][0]} {v}" for n, v in verdicts.items()))
    rep = [r for r in rows if r["kind"] == "repush"]
    print(f"\n실제 다시 넣기 {len(rep)}회 — 다시 넣은 첫 문장을 그 전에 말했는지와 각 판단")
    for name, label in STALL_LABELS.items():
        c = collections.Counter(("말함" if r["said_first"] >= SAID else "안 말함", bool(r[name])) for r in rep)
        print(
            f"  {label}: 안 말함→다시 넣음 {c[('안 말함', True)]}, 안 말함→다음 {c[('안 말함', False)]}, "
            f"말함→다시 넣음 {c[('말함', True)]}, 말함→다음 {c[('말함', False)]}"
        )
    for r in rep:
        if not r["reply"] or r["said_first"] >= SAID:
            print(f"    {r['run']:26s} {r['unit'] + 1:2d}번 대사 «{r['first'][:30]}» 글자 그대로 {r['said_first']:.0%}")


def _pause_run(run: Run) -> list[tuple[float, bool]]:
    v = run.voiced()
    pushes = [e["pos"] for e in run.events if e["name"] in ("push", "repush")]
    out = []
    for sp in spans(run):
        for g in span_gt(run, *sp).values():
            if g.first is None or g.end is None:
                continue
            a = int(g.first / FRAME)
            seg = v[a : int(g.end / FRAME)]
            idx = np.flatnonzero(seg)
            if len(idx) < 2:
                continue
            cur = start = 0
            for i, s in enumerate(seg[idx[0] : idx[-1] + 1]):
                if not s:
                    start = i if cur == 0 else start
                    cur += 1
                elif cur:
                    ts, te = (a + idx[0] + start) * FRAME, (a + idx[0] + i) * FRAME
                    out.append((round(cur * FRAME, 2), any(ts <= p <= te + 0.2 for p in pushes)))
                    cur = 0
    return out


def cmd_pauses(runs: list[Run]) -> None:
    rows = [r for rs in _map(_pause_run, runs) for r in rs]
    d = np.array([r[0] for r in rows])
    print(f"대사 안 무음 {len(d)}개: 95% {np.percentile(d, 95):.2f} s, 99% {np.percentile(d, 99):.2f} s, ", end="")
    print(f"99.9% {np.percentile(d, 99.9):.2f} s, 최대 {d.max():.2f} s")
    for thr in (0.8, 1.0, 1.2, 1.5, 2.0):
        sel = [r for r in rows if r[0] >= thr]
        print(
            f"  {thr:.1f} s 이상 {len(sel):3d} — 그동안 앱이 넣음 {sum(r[1] for r in sel):3d}, 모델이 스스로 이음 ",
            end="",
        )
        print(sum(not r[1] for r in sel))


def _done_run(run: Run) -> list[tuple[Any, ...]]:
    gts = run_gt(run)
    by_unit: dict[int, UnitGT] = {}
    for (_, u), g in gts.items():
        by_unit.setdefault(u, g)
    times = {name: timeline(run, rule, 99.0, "next")["done_at"] for name, rule in (("5", EXPERIMENT), ("8", DECIDED))}
    out = []
    for n, t5 in times["5"].items():
        g = by_unit.get(n - 1)
        closing = run.kind == "long" and n in run.seg_first[1:]
        out.append(("done", closing, t5, times["8"].get(n), g.end if g else None))
    for s, song in enumerate(run.songs or []):
        g = gts.get((s, run.seg_first[s + 1] - 1))
        if g and g.end is not None:
            out.append(("song", song["by"], song["start"] - g.end))
    return out


def cmd_done(runs: list[Run]) -> None:
    rows = [r for rs in _map(_done_run, runs) for r in rs]
    done = [r for r in rows if r[0] == "done" and r[4] is not None]
    for i, label in ((2, "5자"), (3, "8자")):
        d = np.array([r[i] - r[4] for r in done if r[i] is not None])
        print(f"끝남 판정 − 대사 끝 ({label}): 중앙 {np.median(d):+.2f} s, 1~95% {np.percentile(d, 1):+.2f}~", end="")
        print(f"{np.percentile(d, 95):+.2f} s, 판정 {len(d)}")
    closing = [r for r in done if r[1]]
    later = sum(1 for r in closing if r[3] is None or r[3] - r[2] > 0.2)
    print(f"곡 소개 대사 끝남 {len(closing)}회 중 8자에서 0.2 s 넘게 늦어짐 {later}")
    songs = [r for r in rows if r[0] == "song"]
    gaps = np.array([r[2] for r in songs])
    print(f"곡 시작 {len(songs)}회: 곡 소개 끝보다 먼저 {int(np.sum(gaps < 0))}, ", end="")
    print(f"곡 소개 끝 → 곡 {gaps.min():.1f}~{gaps.max():.1f} s")


COMMANDS = {"validate": cmd_validate, "push": cmd_push, "stall": cmd_stall, "pauses": cmd_pauses, "done": cmd_done}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=list(COMMANDS))
    args = parser.parse_args()
    runs = load_runs()
    short = sum(r.kind == "short" for r in runs)
    print(f"회차 {len(runs)} (짧은 대본 {short}, 긴 대본 {len(runs) - short})")
    COMMANDS[args.command](runs)


if __name__ == "__main__":
    main()
