"""라디오 앞서 넣기 실험 — 대사를 thinking 으로 한 줄 앞서 넣어 모델이 늘 다음 줄을 가진 채 읽게 한다.

``exp_radio_flow`` 와 같은 흐름(판정기, 멈춘 뒤 대화, 위임 재개)에서 대사를 넣는 방법만 바꾼다. 쉬고 있는 모델에게
thinking 한 줄을 주면 그것을 소재로 이야기를 지어냈지만(덧붙임 57%), 말하는 도중 다음 줄을 미리 주면 대사를 거의
그대로 차례로 읽었다(``exp_dj_commentary_chunks --channel thinking``, th_a·th_b 대 th_ahead) — 받은 것을 다 말했는데
다음 것이 없으면 지어낸다. 그래서 다음 단위는 모델이 마지막으로 넣은 단위의 마지막 문장에 들어서고 남은 글자가
``PUSH_REMAIN_LETTERS``(약 2 s) 이하일 때 넣는다. 더 일찍 넣으면 읽지 않은 단위가 둘인 채 문장 경계를 넘게 되는데,
시작 구간에서는 그때 앞 단위를 건너뛰었다(s3·l3). 단위는 기본이 대사(``--unit line``).

진행은 지금까지 넣은 단위들과 모델 전사의 일치(``TRACK_BLOCK`` 글자 이상)로 따라간다 — 앞 단위를 건너뛰고 다음
단위를 읽으면 앞 단위는 끝난 것이다. 바꿔 말해 일치가 끊겨도 늦지 않게, 마지막 일치 뒤에 더 말한 글자만큼 나아간
것으로 치고, 문장 끝 부호 수로도 마지막 문장을 알아본다. 모델이 넣은 뒤 말을 하고 ``STALL_SEC`` 조용하면 그냥
넣는다. 사용자 말을 받는 동안(보류)의 모델 말은 진행에서 뺀다. ``read`` 안내는 넣지 않는다.

대사 문구(``LINE_TEMPLATE``)에는 "radio" 를 넣지 않는다 — "On the radio you are now saying" 은 모델이 라디오를 따로
트는 방송처럼 전하게 했다. ``--definition`` 은 기본 지시문 Backend tools 의 Radio 줄을 라디오가 레이 자신이 진행하는
것이라고 알려 주는 줄(``RADIO_DEFINITION_LINE``)로 바꾼다.

끼어들기(``--interrupt``): ``mid`` 는 모델이 ``--chunk`` 번째 줄을 읽기 시작하고 ``--at`` 초 뒤 말소리가 날 때, ``gap``
은 그 줄을 다 읽고 ``--gap-at`` 초 쉴 때. 사용자 전사가 오면 줄 넣기를 멈추고 판정한다(``exp_radio_flow`` 규칙).
on 이면 줄 넣기를 다시 시작한다 — 대기 중인 줄은 모델이 알아서 읽는다. 모델 말 가운데 읽던 줄·대기 줄과 맞지 않는
글자를 반응으로 센다.

지표: 끼어들기 전 대사 밖 글자 비율(덧붙임), 질문에 답하기 전에 대사를 먼저 읽었는지(``script_before``), 멈춘 뒤
스스로 대사를 이어 읽었는지(``self_continue``), 구간별 방송 어휘(``BROADCAST``), 재개 한마디와 재개 뒤 대사 밖 글자,
그 밖에 ``exp_radio_flow`` 와 같은 판정·재개 지표. 넣기 시점(``push_timing``): 끼어들기 전 넣기마다 앞 단위 마지막
문장의 시작·끝과 비교한 위치와 여유. 대본 충실도·방송 어휘·재개·넣기 시점 지표는 ``--report`` 가 전사에서 다시
계산하므로 이전 결과 폴더에도 쓸 수 있다.

    uv run python scripts/gpt_live/exp_radio_ahead.py --tag a1 --utterances content_why content_key
    uv run python scripts/gpt_live/exp_radio_ahead.py --report a1

결과: var/experiments/radio_ahead/<번호>_<태그>/ — 구성은 ``exp_radio_flow`` 와 같다.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from exp_barge_in import user_clip, voiced_offsets
from exp_dirs import find_result_dir, result_dir
from exp_dj_commentary_chunks import CHUNKS, tts24
from exp_radio_flow import (
    BACKEND_INSTRUCTIONS,
    CLASSIFIER_INSTRUCTIONS,
    CLASSIFIER_MODEL,
    END_NOTE,
    FOLLOW_UP,
    HOLD_MAX_QUIET_SEC,
    LIVE_RADIO_INSTRUCTIONS,
    NEXT_TURN_MAX_WAIT_SEC,
    NEXT_TURN_QUIET_SEC,
    PAUSE_NOTE,
    RADIO_TOOLS_LINE,
    RESUME_ASKS,
    RESUME_NOTE,
    TURN_WATCH_SEC,
    UTTERANCES,
    VOICED_SEC,
    Session,
    _voiced,
)
from exp_radio_tts import sentences

from voice_pipeline.adapters.llm_openai import OpenAILLM
from voice_pipeline.engines.gpt_live.instructions import LIVE_VOICE

# 모델이 문구의 틀을 첫머리에 말로 옮긴다: "On the radio you are now saying" → "지금 라디오에서 이렇게 말하고 있어요"
# (s2·l2 21/38, 라디오 줄 없는 지시문의 th_ahead 2/4), "Next line of your radio talk" → "이어서 한마디 더 할게요"
# (a1·ag1 12/57). 그래서 방송도 다음 줄도 아닌, 지금 하는 말로만 적는다.
LINE_TEMPLATE = "You are now saying: «{}»"
# --definition: 라디오가 따로 트는 방송이 아니라 레이가 진행하는 것이라고 알린다. 상태 thinking 과 같은 "radio talk".
RADIO_DEFINITION_LINE = (
    "- Radio: your own radio talk, which you host and speak yourself. Starting it, or resuming it after a pause.\n"
)
assert RADIO_TOOLS_LINE in LIVE_RADIO_INSTRUCTIONS
DEFINED_INSTRUCTIONS = LIVE_RADIO_INSTRUCTIONS.replace(RADIO_TOOLS_LINE, RADIO_DEFINITION_LINE, 1)
# 라디오를 따로 트는 방송처럼 전하는 말 — "지금 라디오에서는 …", "이어서 틀어볼게요", "재생할게요"
BROADCAST = re.compile(r"라디오|방송|재생|틀(?:어|게|께|었|고|겠)")
PLACES = {"early": "마지막 문장 전", "in_last": "마지막 문장 중", "late": "끝난 뒤", "unknown": "모름"}
ACK_MATCH = 8  # 재개 한마디의 끝 — 대사와 이만큼 이어 겹치는 첫 곳 ("해 볼게요." 같은 짧은 겹침은 넘긴다)
PUSH_REMAIN_LETTERS = 13  # 지금 단위에 남은 글자가 이만큼(말 속도 중앙 6.5자/초로 약 2 s) 이하면 다음 단위
TRACK_BLOCK = 5  # 진행으로 칠 전사·대본 일치 최소 길이
DONE_SLACK_LETTERS = 4  # 단위 끝 이만큼 안까지 일치하면 끝난 것 (끝말 "요" 같은 차이)
TERMINATOR = re.compile(r"[.?!…]+")
STALL_SEC = 2.0  # 넣은 뒤 말을 하고 이만큼 조용하면 전사와 상관없이 다음 단위
MATCH_LETTERS = 4  # 대사로 칠 일치 구간 최소 길이
REACT_LETTERS = 6  # 대답 시작으로 칠 대사 밖 글자 길이
FRAME_SEC = 0.02
SENTENCES = [s for line in CHUNKS for s in sentences(line)]

logger = logging.getLogger("exp_radio_ahead")


def _letters(text: str) -> str:
    return re.sub(r"[\W_]+", "", text)


def script_mask(text: str, script: str, min_size: int = MATCH_LETTERS) -> list[bool]:
    """``text``(글자열)의 각 글자가 ``script`` 와 ``min_size`` 이상 일치하는 구간에 드는지."""
    mask = [False] * len(text)
    for b in difflib.SequenceMatcher(None, text, script, autojunk=False).get_matching_blocks():
        if b.size >= min_size:
            mask[b.a : b.a + b.size] = [True] * b.size
    return mask


def build_units(unit: str) -> list[tuple[int, str]]:
    """넣을 단위 — (대사 번호, 글). ``sentence`` 면 대사를 문장으로 쪼갠다."""
    if unit == "line":
        return list(enumerate(CHUNKS))
    return [(i, s) for i, line in enumerate(CHUNKS) for s in sentences(line)]


class AheadSession(Session):
    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.preamble = None
        self.units = build_units(self.args.unit)
        self.unit_letters = [_letters(t) for _, t in self.units]
        self.offsets = [0]  # 단위 시작의 글자 위치 (단위 전체를 이은 대본 기준)
        for letters in self.unit_letters:
            self.offsets.append(self.offsets[-1] + len(letters))
        # 단위마다 문장 수와 마지막 문장의 글자 위치·길이
        self.unit_sentences = [len(sentences(t)) for _, t in self.units]
        last = [_letters(sentences(t)[-1]) for _, t in self.units]
        self.last_len = [len(s) for s in last]
        self.last_start = [self.offsets[i + 1] - len(s) for i, s in enumerate(last)]
        self.done = 0  # 끝까지 읽힌 단위 수 (앞에서부터, 건너뛴 단위 포함)
        self.done_at: list[float] = []  # 단위마다 끝까지 읽힌 위치
        self.line_started: dict[int, float] = {}  # 대사 번호 → 그 대사 첫 단위를 읽기 시작한(앞 단위가 끝난) 위치
        self.track_from = 0.0  # 진행 추적에 쓰는 전사의 시작 위치 — 재개 때 옮긴다
        self.track_unit = 0  # 진행 추적에 쓰는 첫 단위
        self.excluded: list[list[float]] = []  # 보류 구간 [시작, 끝] — 이 동안의 말은 더 말한 글자·부호로 세지 않는다
        self.push_by = "start"
        self._progress_cache: tuple[Any, dict[str, Any]] | None = None

    def live_instructions(self) -> str:
        return DEFINED_INSTRUCTIONS if self.args.definition else LIVE_RADIO_INSTRUCTIONS

    # -- 넣기 ----------------------------------------------------------------

    def _append(self) -> None:
        self.appends.append(self.pos())
        self.first_voice = None
        self._event("push", unit=len(self.appends), by=self.push_by)
        self.live.append_thinking(LINE_TEMPLATE.format(self.units[len(self.appends) - 1][1]))

    def _progress(self) -> dict[str, Any]:
        """모델이 대본에서 와 있는 곳 (글자 위치는 단위 전체를 이은 대본 기준).

        ``reach``: 전사와 대본이 ``TRACK_BLOCK`` 글자 이상 일치한 가장 먼 곳. ``est``: ``reach`` 에 그 일치 뒤 더 말한
        글자 수를 더한 것 — 바꿔 말해도 진행을 놓치지 않게. ``punct_remain``: 마지막으로 넣은 단위의 시작 뒤 문장 끝
        부호로 센 마지막 문장의 남은 글자 (마지막 문장에 아직 안 들어섰거나 모르면 None). 일치는 ``done`` 바로 앞
        단위부터 넣은 단위까지와, 그 단위를 넣은 뒤(재개 뒤면 재개 뒤) 전사로 본다.
        """
        key = (len(self.deltas), len(self.appends), self.done, self.track_from, str(self.excluded))
        if self._progress_cache is not None and self._progress_cache[0] == key:
            return self._progress_cache[1]
        base = min(max(self.track_unit, self.done - 1), len(self.appends) - 1)
        since = max(self.track_from, self.appends[base])
        raw, at = "", []
        for p, sp, d in self.deltas:
            if sp == "assistant" and p >= since:
                raw += d
                at += [p] * len(d)
        free = [not any(a <= p < b for a, b in self.excluded) for p in at]
        idx = [i for i, ch in enumerate(raw) if _letters(ch)]
        text = "".join(raw[i] for i in idx)
        script = "".join(self.unit_letters[base : len(self.appends)])
        last = len(self.appends) - 1
        reach = reach_t = 0
        last_from: int | None = None  # 마지막 단위가 시작된 전사 글자 자리
        for b in difflib.SequenceMatcher(None, text, script, autojunk=False).get_matching_blocks():
            if b.size < TRACK_BLOCK:
                continue
            reach, reach_t = b.b + b.size, b.a + b.size
            at_script = self.offsets[base] + b.b
            if last_from is None and at_script + b.size > self.offsets[last]:
                last_from = max(0, b.a + self.offsets[last] - at_script)
        drift = sum(free[i] for i in idx[reach_t:])
        punct_remain = None
        n = self.unit_sentences[last]
        if last_from is not None and n > 1 and last_from < len(idx):
            ends = [m.end() for m in TERMINATOR.finditer(raw, idx[last_from]) if free[m.start()]]
            if len(ends) >= n - 1:
                said = sum(free[i] for i in idx if i >= ends[n - 2])
                punct_remain = self.last_len[last] - said
        out = {
            "reach": self.offsets[base] + reach,
            "est": self.offsets[base] + reach + drift,
            "drift": drift,
            "punct_remain": punct_remain,
        }
        self._progress_cache = (key, out)
        return out

    def _update_done(self) -> None:
        if self.excluded and self.excluded[-1][1] == float("inf") and self.hold_start is None:
            self.excluded[-1][1] = self.pos()  # 보류가 풀렸다 (on·paused·off)
        if not self.appends:
            return
        reach = self._progress()["reach"]
        while self.done < len(self.appends) and reach >= self.offsets[self.done + 1] - DONE_SLACK_LETTERS:
            self.done += 1
            self.done_at.append(self.pos())
            if self.done < len(self.units):
                self.line_started.setdefault(self.units[self.done][0], self.pos())
            self._event("unit_done", unit=self.done)

    def _current_line(self) -> str:
        return CHUNKS[self.units[min(self.done, len(self.appends) - 1)][0]]

    def _ready_for_next(self) -> bool:
        """마지막으로 넣은 단위의 마지막 문장에 들어서 남은 글자가 ``PUSH_REMAIN_LETTERS`` 이하인지. 근거는 기록."""
        n = len(self.appends)
        if n >= len(self.units):
            return False
        last = n - 1
        prog = self._progress()
        est, end = prog["est"], self.offsets[last + 1]
        if est > self.last_start[last] and end - est <= PUSH_REMAIN_LETTERS:
            self.push_by = "text" if prog["drift"] <= DONE_SLACK_LETTERS else "letters"
            return True
        if prog["punct_remain"] is not None and prog["punct_remain"] <= PUSH_REMAIN_LETTERS:
            self.push_by = "punct"
            return True
        # 넣은 뒤 말을 하고 조용해졌다 — 마지막 문장까지 왔으면 바로, 아니면 조금 더 기다려서
        if self.first_voice is not None and self.quiet >= STALL_SEC:
            if est > self.last_start[last] or self.quiet >= 2 * STALL_SEC:
                self.push_by = "stall"
                return True
        return False

    def _on_user_text(self) -> None:
        held = self.hold_start is not None
        super()._on_user_text()
        if not held and self.hold_start is not None:
            self.excluded.append([self.hold_start, float("inf")])

    def _reaction_letters(self) -> int:
        since = self.hold_start if self.hold_start is not None else self.pos()
        text = _letters("".join(d for p, sp, d in self.deltas if sp == "assistant" and p >= since))
        script = _letters("".join(t for _, t in self.units[max(0, self.done - 1) : len(self.appends)]))
        return sum(not m for m in script_mask(text, script))

    def _on_call(self, call: Any) -> None:
        if call.name != "resume_radio" or self.status != "paused":
            super()._on_call(call)
            return
        self.pending_calls.append(call)
        self._event("call", tool=call.name)
        self.outputs[call.call_id] = json.dumps({"status": "on"})
        del self.appends[self.done :]  # 넣었지만 읽히지 않은 단위는 다시 넣는다 — 재개 때 건너뛰지 않게
        self._set_status("on")
        self.track_from, self.track_unit = self.pos(), self.done  # 다시 넣는 단위부터 새로 따라간다
        if len(self.appends) < len(self.units):
            self.push_by = "resume"
            self._append()

    def _hold_step(self, p: float, can_append: bool) -> None:
        """on 이면 보류만 푼다 — 대기 단위는 모델이 읽고, 다음 단위는 앞서 넣기 규칙이 넣는다."""
        self._maybe_classify(p)
        on = self.latest_on
        if on is not None and not self._waiting() and not self._stale(on):
            on["applied"] = True
            self.hold_start = None
            self._event("decision", state="on", by="timeout" if on.get("timeout") else "classifier")
        elif not self._waiting() and self.quiet >= HOLD_MAX_QUIET_SEC:
            self.hold_start = None
            self._event("hold_release", by="max")

    # -- 진행 ----------------------------------------------------------------

    def _step(self, since_preamble: float) -> None:
        p = self.pos()
        self._update_done()
        if not self.appends:
            if since_preamble >= 1.0:
                self._append()
            return
        if not self.user_turns:  # 끼어들기 전
            target = self.args.chunk - 1
            idx = [i for i, (line, _) in enumerate(self.units) if line == target]
            started = self.line_started.get(target)
            if self.args.interrupt == "mid":
                reading = idx[0] <= self.done <= idx[-1] and started is not None
                if reading and p - started >= self.args.at and self.quiet < VOICED_SEC:
                    self._interrupt()
            elif self.done == idx[-1] + 1 and self.quiet >= self.args.gap_at:
                self._interrupt()  # 그 대사를 다 읽고 쉬는 틈
            if not self.user_turns and self._ready_for_next():
                self._append()
            if self.done >= len(self.units) or (len(self.appends) == len(self.units) and self.quiet >= 3.0):
                self.result = "missed"  # 끼어들 대사를 전사로 찾지 못한 채 다 읽었다 (건너뛰기·압축 등)
            return
        last = len(self.user_turns) - 1
        since_last = p - self._turn_end(last)
        if self.status_pos is None:
            if self.hold_start is not None:
                self._hold_step(p, False)
            elif self._ready_for_next():
                self._append()
            if since_last >= self.args.watch:
                self.result = "done"
            return
        if self.user_pcm:
            return
        nxt = len(self.user_turns)
        if self.status != "on" and nxt - 1 < len(self.plan):
            spoke = self.last_voice is not None and self.last_voice > self._turn_end(last)
            if (spoke and self.quiet >= NEXT_TURN_QUIET_SEC) or since_last >= NEXT_TURN_MAX_WAIT_SEC:
                text = self.plan[nxt - 1]
                self.turns.append((text, user_clip(text, trim=True), voiced_offsets(text, trim=True)))
                self._say(nxt)
                return
        if self.status == "on" and self._ready_for_next():
            self._append()  # 재개 뒤 진행
        if since_last >= TURN_WATCH_SEC:
            self.result = "done"

    def raw(self) -> dict[str, Any]:
        return {
            **super().raw(),
            "chunk": self.args.chunk,
            "unit": self.args.unit,
            "units": [list(u) for u in self.units],
            "done_at": [round(x, 2) for x in self.done_at],
        }


# ---------------------------------------------------------------------------
# 분석
# ---------------------------------------------------------------------------


def _text(raw: dict[str, Any], a: float, b: float) -> str:
    return "".join(d for p, sp, d in raw["deltas"] if sp == "assistant" and a <= p < b)


def _pause_max(voiced: np.ndarray, a: float, b: float) -> float:
    seg = voiced[int(a / FRAME_SEC) : int(b / FRAME_SEC)]
    idx = np.flatnonzero(seg)
    if len(idx) < 2:
        return 0.0
    body = seg[idx[0] : idx[-1] + 1]
    best = cur = 0
    for v in body:
        cur = 0 if v else cur + 1
        best = max(best, cur)
    return round(best * FRAME_SEC, 2)


def fidelity(raw: dict[str, Any]) -> dict[str, Any]:
    """대본 충실도 — 끼어들기 전(끼어들지 못했으면 전부) 모델 말. 모든 회차에 잰다."""
    turns = raw.get("user_turns") or []
    cut = turns[0]["start"] if turns else float("inf")
    spoken = _letters(_text(raw, 0.0, cut))
    hits = [i for i, s in enumerate(SENTENCES) if sum(script_mask(_letters(s), spoken)) >= 0.6 * len(_letters(s))]
    upto = max(hits) if hits else -1
    first = _letters(_text(raw, 0.0, cut))[:6]
    return {
        "skipped": [i + 1 for i in range(upto + 1) if i not in hits],  # 마지막으로 말한 문장 앞에서 빠진 문장
        "lead_in": not first.startswith(_letters(SENTENCES[0])[:4]),  # 첫 문장 앞에 군말
        "off_script_all": round(
            sum(not m for m in script_mask(spoken, _letters("".join(CHUNKS)))) / max(1, len(spoken)), 2
        ),
        "opening": _text(raw, 0.0, cut).strip()[:40],
    }


def push_timing(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """끼어들기 전 넣기마다, 그때 읽던 단위(앞 단위)의 마지막 문장과 비교한 위치.

    전사 전체를 대본 전체와 맞춰 단위마다 마지막 문장 첫 글자·마지막 글자가 들린 시각을 잡는다. ``early`` 는 마지막
    문장이 들리기 전(읽지 않은 단위가 둘인 채 문장 경계를 넘는다), ``late`` 는 앞 단위가 끝난 뒤(그 사이를 지어낼
    수 있다). ``lead`` 는 넣은 때부터 앞 단위가 끝날 때까지.
    """
    turns = raw.get("user_turns") or []
    cut = turns[0]["start"] + turns[0]["voice"][0] if turns else float("inf")
    units = [t for _, t in raw["units"]]
    unit_letters = [_letters(t) for t in units]
    offsets = [0]
    for s in unit_letters:
        offsets.append(offsets[-1] + len(s))
    script = "".join(unit_letters)
    text, at = "", []
    for p, sp, d in raw["deltas"]:
        if sp == "assistant" and p < cut:
            for ch in d:
                if _letters(ch):
                    text += ch
                    at.append(p)
    where: list[float | None] = [None] * len(script)  # 대본 글자 → 들린 시각
    hit: list[int | None] = [None] * len(script)  # 대본 글자 → 전사 글자 자리
    for b in difflib.SequenceMatcher(None, text, script, autojunk=False).get_matching_blocks():
        if b.size >= MATCH_LETTERS:
            for q in range(b.size):
                where[b.b + q], hit[b.b + q] = at[b.a + q], b.a + q
    rows = []
    pushes = [e for e in raw["events"] if e["name"] == "push"]
    for k, p in enumerate(raw["appends"]):
        if k == 0 or p >= cut:
            continue
        prev = k - 1
        last_start = offsets[prev + 1] - len(_letters(sentences(units[prev])[-1]))
        in_last = [where[q] for q in range(last_start, offsets[prev + 1]) if where[q] is not None]
        tail = [q for q in range(offsets[prev], offsets[prev + 1]) if where[q] is not None]
        ended = bool(tail) and tail[-1] >= offsets[prev + 1] - DONE_SLACK_LETTERS
        row: dict[str, Any] = {"unit": k + 1, "by": pushes[k]["by"] if k < len(pushes) else "?"}
        if not in_last or not ended:
            row["place"] = "unknown"  # 앞 단위 마지막 문장을 전사로 못 찾음 (건너뜀·바꿔 말함)
            rows.append(row)
            continue
        t_last, t_end = min(in_last), where[tail[-1]]
        row["place"] = "early" if p < t_last else "late" if p > t_end else "in_last"
        row["lead"] = round(t_end - p, 2)
        nxt = [q for q in range(offsets[k], offsets[k + 1]) if where[q] is not None]
        if nxt and hit[nxt[0]] > hit[tail[-1]]:
            row["gap"] = round(where[nxt[0]] - t_end, 2)  # 앞 단위 끝 → 이 단위 첫 글자 (전사 도착 기준)
            row["between"] = text[hit[tail[-1]] + 1 : hit[nxt[0]]]
        rows.append(row)
    return rows


def wording(raw: dict[str, Any]) -> dict[str, Any]:
    """구간별 방송 어휘와 재개 — 끼어들기 전, 끼어든 뒤 대답, 이어 말함 대답, 재개 한마디, 재개 뒤 대사."""
    turns = raw.get("user_turns") or []
    starts = [t["start"] + t["voice"][0] for t in turns] + [float("inf")]
    ends = [t["start"] + t["voice"][1] for t in turns]
    spans = {"lead": (0.0, starts[0])}
    for i, name in enumerate(("reply", "follow")):
        if i < len(ends):
            spans[name] = (ends[i], starts[i + 1])
    row: dict[str, Any] = {f"bc_{k}": BROADCAST.findall(_text(raw, a, b)) for k, (a, b) in spans.items()}
    resumed = len(ends) > 2 and any(
        e["name"] == "status" and e.get("status") == "on" and e["pos"] >= ends[2] - 1.0 for e in raw["events"]
    )
    if resumed:
        text = _text(raw, ends[2], float("inf"))
        mask = script_mask(text, " ".join(CHUNKS), ACK_MATCH)
        cut = next((i for i, m in enumerate(mask) if m), len(text))
        rest = _letters(text[cut:])
        off = sum(not m for m in script_mask(rest, _letters("".join(CHUNKS))))
        row["resume_ack"] = text[:cut].strip()
        row["resume_off"] = round(off / max(1, len(rest)), 2)
        row["bc_ack"] = BROADCAST.findall(text[:cut])
        row["bc_resumed"] = BROADCAST.findall(text[cut:])
    return row


def analyze(raw: dict[str, Any], pcm: bytes) -> dict[str, Any]:
    if raw["result"] != "done":
        return {}
    units = [(line, text) for line, text in raw["units"]]
    turns = raw["user_turns"]
    v0, v1 = turns[0]["start"] + turns[0]["voice"][0], turns[0]["start"] + turns[0]["voice"][1]
    starts = [t["start"] + t["voice"][0] for t in turns] + [float("inf")]
    ends = [t["start"] + t["voice"][1] for t in turns]
    decisions = [e for e in raw["events"] if e["name"] == "decision"]
    first = decisions[0] if decisions else None
    voiced = _voiced(pcm)
    n_done = len([x for x in raw["done_at"] if x < v0])
    script_all = _letters("".join(CHUNKS))
    # 끼어들기 전에 읽었어야 할 대사 — mid 는 끼어든 대사 앞까지, gap 은 끼어든 대사까지 — 중 실제로 말한 비율
    target = raw["chunk"] - 1
    expected = _letters(
        "".join(t for line, t in units if line < target or (raw["interrupt"] == "gap" and line == target))
    )
    before = _letters(_text(raw, 0.0, v0))
    coverage = sum(script_mask(expected, before)) / max(1, len(expected))
    # 끼어든 뒤 다음 사용자 말 전까지: 대사 / 대사 밖 순서. 끼어들 때 아직 시작하지 않은 단위 — mid 는 읽던 단위
    # 다음부터, gap 은 다 읽은 단위 다음부터 — 을 대답 전에 읽었으면 질문이 대사에 묻힌 것이다(B 의 분량 사이 실패).
    unread = _letters("".join(t for _, t in units[n_done + (1 if raw["interrupt"] == "mid" else 0) :]))
    after = _letters(_text(raw, v1, starts[1]))
    amask = script_mask(after, script_all)
    react_at = next(
        (i for i in range(len(amask)) if not any(amask[i : i + REACT_LETTERS]) and i + REACT_LETTERS <= len(amask)),
        None,
    )
    nmask = script_mask(after, unread)
    row: dict[str, Any] = {
        "units_done_before": n_done,
        "coverage_before": round(coverage, 2),
        "pause_max_before": _pause_max(voiced, 3.0, v0),
        "heard": "".join(
            d for p, sp, d in raw["deltas"] if sp == "user" and turns[0]["start"] <= p < starts[1]
        ).strip(),
        "decision": first["state"] if first else "none",
        "decision_by": first["by"] if first else "",
        "decision_at": round(first["pos"] - v1, 2) if first else None,
        "decision_reason": next((c["reason"] for c in raw["classifier"] if c.get("applied")), ""),
        "judge_states": [c["state"] for c in raw["classifier"]],
        "judge_latency": [c["latency"] for c in raw["classifier"]],
        "after_text": _text(raw, v1, starts[1]).strip(),
        "script_before": sum(amask[:react_at]) if react_at is not None else sum(amask),  # 반응 전 대사 (읽던 문장 포함)
        "next_before": sum(nmask[:react_at]) if react_at is not None else sum(nmask),  # 그중 아직 시작 안 한 단위
        "reacted": react_at is not None,
    }
    if first and first["state"] != "on":
        tail = _letters(_text(raw, first["pos"] + 0.5, starts[1]))
        row["self_continue"] = sum(script_mask(tail, unread))  # 멈춘 뒤 스스로 읽은 새 단위
    delegations = [e["pos"] for e in raw["events"] if e["name"] == "delegation"]
    status_on = [e["pos"] for e in raw["events"] if e["name"] == "status" and e.get("status") == "on"]
    for i, name in ((1, "follow"), (2, "resume")):
        if i < len(turns):
            row[f"{name}_text"] = turns[i]["text"]
            row[f"{name}_delegations"] = [
                round(p - ends[i], 2) for p in delegations if starts[i] - 0.5 <= p < starts[i + 1]
            ]
            row[f"{name}_reply"] = _text(raw, starts[i], starts[i + 1]).strip()
    if len(turns) > 2:
        row["resumed_at"] = round(status_on[0] - ends[2], 2) if status_on else None
        if status_on:  # 재개 뒤 처음 읽은 문장 — 끊긴 곳부터 이어 읽는지
            resumed = _letters(_text(raw, status_on[0], float("inf")))
            row["resumed_from"] = next(
                (i + 1 for i, s in enumerate(SENTENCES) if sum(script_mask(_letters(s), resumed[:40])) >= 4), None
            )
    return row


def format_row(r: dict[str, Any]) -> list[str]:
    fid = f"건너뛴 문장 {r['skipped'] or '-'}" + (f", 첫머리 군말 「{r['opening'][:20]}」" if r["lead_in"] else "")
    fid += f", 대사 밖 {r['off_script_all']:.0%}"
    if r.get("push_timing"):
        places = [t["place"] for t in r["push_timing"]]
        fid += ", 넣기 " + "·".join(PLACES[p] for p in places)
    bc = {k[3:]: v for k, v in r.items() if k.startswith("bc_") and v}
    if bc:
        fid += ", 방송 어휘 " + " ".join(f"{k}:{'·'.join(v)}" for k, v in bc.items())
    if r["result"] != "done":
        return [f"  #{r['run']} {r['result']} | {fid}"]
    at = f"{r['decision_at']:+.1f}s" if r["decision_at"] is not None else "-"
    states = "/".join(r["judge_states"]) or "-"
    head = f"  #{r['run']} 판정 {r['decision']} @{at} (호출: {states}) | {fid}, 쉼 최대 {r['pause_max_before']}s"
    head += f" | 반응 전 대사 {r['script_before']}자(새 단위 {r['next_before']}자)"
    head += "" if r["reacted"] else " (반응 없음)"
    if r.get("self_continue"):
        head += f" | 멈춘 뒤 스스로 읽음 {r['self_continue']}자"
    lines = [head, f"     들림: {r['heard'] or '-'} | 모델: {r['after_text'][:150] or '-'}"]
    if "follow_text" in r:
        d = " ".join(f"{t:+.1f}" for t in r["follow_delegations"]) or "없음"
        lines.append(f"     이어 말함 → 위임 {d}: {r['follow_reply'][:110] or '-'}")
    if "resume_text" in r:
        d = " ".join(f"{t:+.1f}" for t in r["resume_delegations"]) or "없음"
        if r.get("resumed_at") is not None:
            res = f"재개 {r['resumed_at']:+.1f}s, {r.get('resumed_from')}번 문장부터"
        else:
            res = "재개 없음"
        lines.append(f'     "{r["resume_text"]}" → 위임 {d}, {res}: {r["resume_reply"][:90] or "-"}')
    return lines


def report(tag: str) -> None:
    root = find_result_dir("radio_ahead", tag)
    if root is None:
        raise SystemExit(f"결과 폴더 없음: {tag}")
    lines: list[str] = []
    every: list[dict[str, Any]] = []
    allrows: list[tuple[str, dict[str, Any]]] = []
    for key, (kind, text, expect) in UTTERANCES.items():
        path = root / f"{key}.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text())["results"]
        for r in rows:
            if "deltas" in r:
                r.update(fidelity(r), **wording(r))
                if "units" in r:
                    r["push_timing"] = push_timing(r)
        every += [r for r in rows if "skipped" in r]
        valid = [r for r in rows if r["result"] == "done"]
        allrows += [(expect, r) for r in valid]
        lines.append(
            f"\n[{kind}] {text} ({key}) 기대 {expect} — 맞음 {sum(r['decision'] == expect for r in valid)}/"
            f"{len(valid)} (유효 {len(valid)}/{len(rows)})"
        )
        for r in rows:
            lines.extend(format_row(r))
    rs = [r for _, r in allrows]
    paused = [r for e, r in allrows if e != "on"]
    lines.append("\n전체")
    lines.append(
        f"  대본 충실도(모든 회차 {len(every)}): 문장 건너뜀 {sum(bool(r['skipped']) for r in every)}, "
        f"첫머리 군말 {sum(r['lead_in'] for r in every)}, "
        f"대사 밖 글자 중앙 {float(np.median([r['off_script_all'] for r in every])):.0%} "
        f"(10% 넘음 {sum(r['off_script_all'] > 0.1 for r in every)}), "
        f"끼어들지 못함 {len(every) - len(rs)}"
    )
    timing = [t for r in every for t in r.get("push_timing", [])]
    if timing:
        places = ", ".join(f"{PLACES[k]} {sum(t['place'] == k for t in timing)}" for k in PLACES)
        leads = [t["lead"] for t in timing if "lead" in t]
        gaps = [t["gap"] for t in timing if "gap" in t]
        by = " ".join(f"{k} {sum(t['by'] == k for t in timing)}" for k in sorted({t["by"] for t in timing}))
        lines.append(f"  넣기 시점(끼어들기 전 {len(timing)}회, 앞 단위 마지막 문장 기준): {places}")
        if leads:
            lines.append(
                f"    앞 단위 끝까지 여유 중앙 {float(np.median(leads)):.1f}s (최소 {min(leads)}, 1 s 미만 "
                f"{sum(x < 1.0 for x in leads)}), 앞 단위 끝 → 다음 단위 중앙 {float(np.median(gaps)):.2f}s, "
                f"사이 끼워 넣은 말(3자 넘음) {sum(len(t.get('between', '')) > 3 for t in timing)} | 근거: {by}"
            )
    spans = (("lead", "끼어들기 전"), ("reply", "끼어든 뒤 대답"), ("follow", "이어 말함"), ("ack", "재개 한마디"))
    spans += (("resumed", "재개 뒤"),)
    counts = []
    for key, name in spans:
        has = [r for r in every if f"bc_{key}" in r]
        counts.append(f"{name} {sum(bool(r[f'bc_{key}']) for r in has)}/{len(has)}")
    lines.append("  방송 어휘(라디오·방송·재생·틀-) 든 회차: " + ", ".join(counts))
    resumed = [r for r in every if "resume_ack" in r]
    if resumed:
        lines.append(
            f"  재개 뒤 대사 밖 글자 중앙 {float(np.median([r['resume_off'] for r in resumed])):.0%} "
            f"(10% 넘음 {sum(r['resume_off'] > 0.1 for r in resumed)}/{len(resumed)})"
        )
        lines.append("  재개 한마디: " + " / ".join(f"「{r['resume_ack'][-40:]}」" for r in resumed))
    lines.append(f"  판정 맞음 {sum(r['decision'] == e for e, r in allrows)}/{len(rs)}")
    for e in ("on", "paused", "off"):
        sub = [r for x, r in allrows if x == e]
        if sub:
            got = " ".join(f"{s} {sum(r['decision'] == s for r in sub)}" for s in ("on", "paused", "off", "none"))
            lines.append(f"    기대 {e}: {got}")
    if rs:
        lines.append(f"  끼어들기 전 쉼 최대 중앙 {float(np.median([r['pause_max_before'] for r in rs])):.2f}s")
    lines.append(
        f"  말 걸기(기대 paused·off): 대답 전에 새 단위를 10자 넘게 읽음 {sum(r['next_before'] > 10 for r in paused)}/"
        f"{len(paused)} (읽던 문장 마저 포함 {sum(r['script_before'] > 10 for r in paused)}), "
        f"반응 없음 {sum(not r['reacted'] for r in paused)}/{len(paused)}, "
        f"멈춘 뒤 스스로 대사를 10자 넘게 읽음 {sum(r.get('self_continue', 0) > 10 for r in paused)}/{len(paused)}"
    )
    follow = [r for r in rs if "follow_text" in r]
    lines.append(f"  이어 말한 뒤 위임(없어야 함) {sum(bool(r['follow_delegations']) for r in follow)}/{len(follow)}")
    for ask in RESUME_ASKS:
        sub = [r for r in rs if r.get("resume_text") == ask]
        if sub:
            lines.append(f'  "{ask}" 재개 {sum(r.get("resumed_at") is not None for r in sub)}/{len(sub)}')
    out = "\n".join(lines)
    print(out)
    (root / "report.txt").write_text(out + "\n")


def run(args: argparse.Namespace) -> None:
    out_dir = result_dir("radio_ahead", args.tag)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-28s %(levelname)-5s %(message)s",
        handlers=[logging.FileHandler(out_dir / f"pipeline_{os.getpid()}.log")],
    )
    (out_dir / "instructions.txt").write_text(DEFINED_INSTRUCTIONS if args.definition else LIVE_RADIO_INSTRUCTIONS)
    config = {
        "definition": RADIO_DEFINITION_LINE if args.definition else None,
        "interrupt": args.interrupt,
        "chunk": args.chunk,
        "at": args.at,
        "gap_at": args.gap_at,
        "unit": args.unit,
        "line_template": LINE_TEMPLATE,
        "push_remain_letters": PUSH_REMAIN_LETTERS,
        "track_block": TRACK_BLOCK,
        "stall": STALL_SEC,
        "voice": LIVE_VOICE,
        "notes": {"paused": PAUSE_NOTE, "on": RESUME_NOTE, "off": END_NOTE},
        "backend_instructions": BACKEND_INSTRUCTIONS,
        "classifier": {"model": CLASSIFIER_MODEL, "instructions": CLASSIFIER_INSTRUCTIONS},
        "chunks": list(CHUNKS),
        "utterances": UTTERANCES,
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1))
    for text in (FOLLOW_UP, *RESUME_ASKS, *(UTTERANCES[k][1] for k in args.utterances)):
        tts24(text)
    llm = OpenAILLM(model=CLASSIFIER_MODEL, temperature=0.0, reasoning_effort="none", max_tokens=120, tools=[])
    pool = ThreadPoolExecutor(max_workers=4)
    rows: dict[str, list[dict[str, Any]]] = {key: [] for key in args.utterances}
    for n in range(args.runs):
        for key in args.utterances:
            session = AheadSession(key, n, args, llm, pool)
            try:
                session.run()
            except RuntimeError as exc:
                row = {"run": n, "utterance": key, "result": f"failed: {exc}", "events": []}
            else:
                raw = session.raw()
                row = {"run": n, **raw, **fidelity(raw), **wording(raw), **analyze(raw, bytes(session.pcm))}
                row["push_timing"] = push_timing(raw)
                session.save(out_dir / f"{key}_{n}.wav")
            rows[key].append(row)
            print(f"[{key} #{n}] {UTTERANCES[key][1]} (기대 {UTTERANCES[key][2]})", flush=True)
            print("\n".join(format_row(row)), flush=True)
            (out_dir / f"{key}.json").write_text(
                json.dumps({"config": config, "results": rows[key]}, ensure_ascii=False, indent=1)
            )
            time.sleep(1.5)
    pool.shutdown()
    print(f"results: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--utterances", nargs="*", choices=list(UTTERANCES), default=list(UTTERANCES))
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--interrupt", choices=("mid", "gap"), default="mid")
    parser.add_argument("--unit", choices=("sentence", "line"), default="line", help="한 번에 넣는 단위")
    parser.add_argument("--definition", action="store_true", help="Radio 줄을 라디오가 무엇인지 알려 주는 줄로")
    parser.add_argument("--chunk", type=int, default=4, help="이 대사(1부터)에서 끼어든다")
    parser.add_argument("--at", type=float, default=2.0, help="mid: 그 대사를 읽기 시작하고 이 초 뒤")
    parser.add_argument("--gap-at", type=float, default=0.2, help="gap: 그 대사를 다 읽고 이 초 쉴 때")
    parser.add_argument("--gap", type=float, default=1.0)
    parser.add_argument("--midline-wait", type=float, default=1.5)
    parser.add_argument("--watch", type=float, default=20.0, help="on 경로에서 사용자 목소리 끝 뒤 지켜볼 초")
    parser.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--report", metavar="TAG", help="발화별로 모아 보여주고 끝낸다")
    args = parser.parse_args()
    if args.report:
        report(args.report)
    else:
        run(args)


if __name__ == "__main__":
    main()
