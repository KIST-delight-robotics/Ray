"""라디오 앞서 넣기 실험 — 대사를 thinking 으로 한 줄 앞서 넣어 모델이 늘 다음 줄을 가진 채 읽게 한다.

``exp_radio_flow`` 와 같은 흐름(판정기, 멈춘 뒤 대화, 위임 재개)에서 대사를 넣는 방법만 바꾼다. 쉬고 있는 모델에게
thinking 한 줄을 주면 그것을 소재로 이야기를 지어냈지만(덧붙임 57%), 말하는 도중 다음 줄을 미리 주면 대사를 거의
그대로 차례로 읽었다(``exp_dj_commentary_chunks --channel thinking``, th_a·th_b 대 th_ahead). 그래서 모델이 단위 n−1
을 읽기 시작하면(전사로 단위 n−2 가 끝까지 읽혔으면) 단위 n 을 넣는다. 단위는 기본이 문장(``--unit sentence``) —
대사 단위로 넣으면 두 문장짜리 대사의 둘째 문장을 건너뛰었다(57회 중 15회, a1·ag1): 모델은 문장 경계에서 새로 온
것으로 넘어간다. 전사가 어긋나 멈추지 않게, 모델이 ``STALL_SEC`` 조용하면 그냥 다음 단위를 넣는다. ``read`` 안내는
넣지 않는다.

끼어들기(``--interrupt``): ``mid`` 는 모델이 ``--chunk`` 번째 줄을 읽기 시작하고 ``--at`` 초 뒤 말소리가 날 때, ``gap``
은 그 줄을 다 읽고 ``--gap-at`` 초 쉴 때. 사용자 전사가 오면 줄 넣기를 멈추고 판정한다(``exp_radio_flow`` 규칙).
on 이면 줄 넣기를 다시 시작한다 — 대기 중인 줄은 모델이 알아서 읽는다. 모델 말 가운데 읽던 줄·대기 줄과 맞지 않는
글자를 반응으로 센다.

지표: 끼어들기 전 대사 밖 글자 비율(덧붙임), 질문에 답하기 전에 대사를 먼저 읽었는지(``script_before``), 멈춘 뒤
스스로 대사를 이어 읽었는지(``self_continue``), 그 밖에 ``exp_radio_flow`` 와 같은 판정·재개 지표.

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
from exp_dj_interrupt_reply import split_reading
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

# 이 문구로 모델이 대사를 거의 그대로 읽었다. "Next line of your radio talk" 는 첫머리에 "이어서 한마디 더 할게요" 같은
# 군말을 붙였다(57회 중 12회, a1·ag1).
LINE_TEMPLATE = "On the radio you are now saying: «{}»"
FIRST_AHEAD_SEC = 1.0  # 첫 단위 첫 음성 이만큼 뒤 둘째 단위
STALL_SEC = 2.0  # 전사로 진행을 못 따라가도 모델이 이만큼 조용하면 다음 단위
MATCH_LETTERS = 4  # 대사로 칠 일치 구간 최소 길이
REACT_LETTERS = 6  # 대답 시작으로 칠 대사 밖 글자 길이
FRAME_SEC = 0.02
SENTENCES = [s for line in CHUNKS for s in sentences(line)]

logger = logging.getLogger("exp_radio_ahead")


def _letters(text: str) -> str:
    return re.sub(r"[\W_]+", "", text)


def script_mask(text: str, script: str) -> list[bool]:
    """``text``(글자열)의 각 글자가 ``script`` 와 ``MATCH_LETTERS`` 이상 일치하는 구간에 드는지."""
    mask = [False] * len(text)
    for b in difflib.SequenceMatcher(None, text, script, autojunk=False).get_matching_blocks():
        if b.size >= MATCH_LETTERS:
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
        self.done = 0  # 전사로 끝까지 읽힌 단위 수 (앞에서부터)
        self.done_at: list[float] = []  # 단위마다 끝까지 읽힌 위치
        self.line_started: dict[int, float] = {}  # 대사 번호 → 그 대사 첫 단위를 읽기 시작한(앞 단위가 끝난) 위치

    # -- 넣기 ----------------------------------------------------------------

    def _append(self) -> None:
        self.appends.append(self.pos())
        self.first_voice = None
        self.live.append_thinking(LINE_TEMPLATE.format(self.units[len(self.appends) - 1][1]))

    def _update_done(self) -> None:
        while self.done < len(self.appends):
            text = "".join(d for p, sp, d in self.deltas if sp == "assistant" and p >= self.appends[self.done])
            if not split_reading(text, self.units[self.done][1])[1]:
                return
            self.done += 1
            self.done_at.append(self.pos())
            if self.done < len(self.units):
                self.line_started.setdefault(self.units[self.done][0], self.pos())
            self._event("unit_done", unit=self.done)

    def _current_line(self) -> str:
        return CHUNKS[self.units[min(self.done, len(self.appends) - 1)][0]]

    def _ready_for_next(self) -> bool:
        n = len(self.appends)
        if n >= len(self.units):
            return False
        if n == 1:
            return self.first_voice is not None and self.pos() - self.first_voice >= FIRST_AHEAD_SEC
        # 바로 앞에 넣은 단위를 모델이 읽기 시작했고(그 뒤 첫 음성) 그 앞 단위가 끝났을 때. 안전장치는 무음 한 번에
        # 한 단위만 — 무음이 이어지는 동안 프레임마다 넣으면 남은 단위가 한꺼번에 들어가 모델이 요약·건너뛰었다.
        reading = self.first_voice is not None and self.done >= n - 1
        stalled = self.quiet >= STALL_SEC and self.pos() - self.appends[-1] >= STALL_SEC
        return reading or stalled

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
        if len(self.appends) < len(self.units):
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
    (out_dir / "instructions.txt").write_text(LIVE_RADIO_INSTRUCTIONS)
    config = {
        "interrupt": args.interrupt,
        "chunk": args.chunk,
        "at": args.at,
        "gap_at": args.gap_at,
        "unit": args.unit,
        "line_template": LINE_TEMPLATE,
        "first_ahead": FIRST_AHEAD_SEC,
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
                row = {"run": n, **raw, **fidelity(raw), **analyze(raw, bytes(session.pcm))}
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
    parser.add_argument("--unit", choices=("sentence", "line"), default="sentence", help="한 번에 넣는 단위")
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
