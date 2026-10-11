"""라디오 위임 실험 — 분량 주입 낭독 도중 사용자가 말을 걸면 대화 모델이 위임하고, 백엔드가 일시정지를 부르는지 본다.

대화 모델이 새로 아는 것: 위임 규칙(``--rule``: ``base`` 는 기본 지시문 위임 목록 한 줄 ``RADIO_DELEGATION`` + Backend
tools 한 줄, ``preamble`` 은 라디오 시작 안내 끝에 한 문장 ``RADIO_PREAMBLE_RULE``)과 thinking 으로 넣는 라디오 상태
(``status_note``: on / paused / off). 일시정지·종료 판단은 백엔드 몫이다(``BACKEND_INSTRUCTIONS``, 툴
``pause_radio``·``stop_radio``). 백엔드는 thinking 을 못 보므로 백엔드 지시문의 상태 줄은 상태가 바뀔 때
``update_backend`` 로 바꾼다.

흐름은 ``exp_dj_interrupt_reply`` 와 같다(정리본 지시문 + 위 두 줄, ``read`` 안내, 분량 주입, ``--chunk`` 번째 분량 첫
음성 ``--at`` 초 뒤 말하는 도중 끼어들기). 다른 점: 앱은 사용자 전사가 아니라 **백엔드의 pause_radio·stop_radio 호출**을
받아야 분량 넣기를 멈춘다. 사용자 전사가 오면 분량 넣기를 잠시 보류해 모델이 반응할 시간을 둔다(``HOLD_*``,
``--no-hold`` 로 끈다) — 위임은 모델 대답보다 늦게 와서, 보류가 없으면 다음 분량이 대답보다 먼저 들어간다. 툴 결과는
그 백엔드 응답이 끝나면 돌려주고 이어가게 한다(빈 응답을 기대). 멈춘 뒤 모델이 말하고 ``FOLLOW_UP_QUIET_SEC`` 조용해지면
사용자가 한 번 더 말한다(``FOLLOW_UP``) — 상태가 paused/off 이므로 위임이 없어야 한다. 끝: 마지막 사용자 목소리 뒤
``--watch`` 초(이어 말한 경우 ``FOLLOW_UP_WATCH_SEC``).

기록(시각은 첫 사용자 목소리 끝 기준): 위임 생성·툴 호출 시각, 사용자 목소리가 시작된 뒤 툴 호출 전에 들어간 분량(늦은
신호), 낭독과 나눈 대답·빈말("잠시만요" 류)·대답 첫 전사 도착, 이어 말한 뒤의 위임과 대답, 끼어들기 전 분량의 덧붙임.

    # 프로세스 여럿으로 나눠 병렬로 (같은 태그 폴더에 발화별 파일로 모인다)
    uv run python scripts/gpt_live/exp_radio_delegation.py --tag d1 --utterances content_why content_key
    uv run python scripts/gpt_live/exp_radio_delegation.py --report d1

결과: var/experiments/radio_delegation/<번호>_<태그>/ — <발화>.json, <발화>_<회차>.wav(왼쪽 사용자, 오른쪽 모델) +
.labels.txt, config.json, instructions.txt, backend_instructions.txt, pipeline_<pid>.log, report.txt(--report).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from exp_audio import read_model_channel, write_conversation_wav
from exp_barge_in import BIN_SEC, speech_bins, speech_segments, user_clip, voiced_offsets
from exp_dirs import find_result_dir, result_dir
from exp_dj_commentary_chunks import CHUNKS, LIVE_INSTRUCTIONS, PREAMBLES, QUIET_RMS, _rms, tts24
from exp_dj_interrupt_reply import UTTERANCES as REPLY_UTTERANCES
from exp_dj_interrupt_reply import script_reuse, split_reading

from voice_pipeline.adapters.gpt_live import (
    GPTLiveSession,
    LiveAudio,
    LiveClosed,
    LiveDelegationCreated,
    LiveError,
    LiveFunctionCall,
    LiveResponseDone,
    LiveSessionConfig,
    LiveTranscript,
)
from voice_pipeline.engines.gpt_live.instructions import LIVE_VOICE
from voice_pipeline.settings import BRIDGE_SAMPLE_RATE

# 발화 키 → (종류, 문장, 기대하는 백엔드 툴). r1 의 14개 + 멈춤·종료 요청 + 짧은 반응 둘.
UTTERANCES: dict[str, tuple[str, str, str]] = {
    **{
        k: (kind, text, "none" if kind == "짧은 반응" else "pause_radio")
        for k, (kind, text) in REPLY_UTTERANCES.items()
    },
    "wait": ("멈춤·종료", "잠깐만.", "pause_radio"),
    "stop": ("멈춤·종료", "그만하자.", "stop_radio"),
    "bored_short": ("시큰둥", "지루해.", "pause_radio"),
    "react_ok": ("짧은 반응", "그렇구나.", "none"),
    "react_oh": ("짧은 반응", "오, 그래?", "none"),
}
FOLLOW_UP = "그렇구나. 나는 요즘 좀 피곤해."  # 멈춘 뒤 한 번 더 — 어느 발화 뒤에도 어색하지 않게

# --rule base: 정리본 지시문에 두 줄. 라디오가 꺼져 있으면 상태가 on 이 아니므로 걸리지 않는다.
RADIO_TOOLS_LINE = "- Radio: pausing or stopping the radio talk.\n"
RADIO_DELEGATION = (
    "- Radio talk status is on, and the user says something to you, more than a brief reaction such as 응 or 아 진짜. "
    "Reply to the user yourself at the same time.\n"
)
_TOOLS_ANCHOR = "- End of conversation:"
_DELEGATION_ANCHOR = "Say a short goodbye yourself at the same time.\n"
PREAMBLE = PREAMBLES["read"]


def radio_instructions(base: str) -> str:
    assert _TOOLS_ANCHOR in base and _DELEGATION_ANCHOR in base
    out = base.replace(_TOOLS_ANCHOR, RADIO_TOOLS_LINE + _TOOLS_ANCHOR, 1)
    return out.replace(_DELEGATION_ANCHOR, _DELEGATION_ANCHOR + RADIO_DELEGATION, 1)


LIVE_RADIO_INSTRUCTIONS = radio_instructions(LIVE_INSTRUCTIONS)

# --rule preamble: 기본 지시문은 정리본 그대로 두고 라디오 시작 안내(``read``) 끝에 한 문장. "even when you can answer
# yourself" 는 기본 지시문의 "Do not delegate when: You can answer from the conversation" 과 부딪히지 않게.
RADIO_PREAMBLE_RULE = (
    " While the radio talk status is on and the user says more than a brief reaction such as 응 or 아 진짜, "
    "also delegate to the backend at the same time, even when you can answer yourself."
)
RULES = {  # 이름 → (세션 지시문, 라디오 시작 안내)
    "base": (LIVE_RADIO_INSTRUCTIONS, PREAMBLE),
    "preamble": (LIVE_INSTRUCTIONS, PREAMBLE + RADIO_PREAMBLE_RULE),
}


def status_note(status: str) -> str:
    """thinking 으로 넣는 상태 한 줄. thinking 은 누적되므로 앞의 상태를 대신한다고 적는다."""
    return f"Radio talk status: {status}. This replaces any earlier radio talk status."


BACKEND_MODEL = "gpt-5.4-mini"  # 운영과 같게 (wiring._LIVE_BACKEND_MODEL)
BACKEND_REASONING = "low"
BACKEND_INSTRUCTIONS = """\
# Backend for Ray
Ray is a small desk robot talking with the user by voice. Ray delegates requests to you and speaks your answer.

## Radio
Ray hosts a radio talk: the application sends lines and Ray reads them aloud.
Radio talk status: {status}

When Ray delegates while the radio talk status is on:
- If the user wants to stop or end the radio, call stop_radio.
- Otherwise, if the user says anything to Ray (a question, a comment, a complaint, a request, or asks Ray to wait), \
call pause_radio.
- If the user only reacted briefly (for example 응, 아 진짜, 맞아), call no tool.
Ray answers the user directly. After the radio tool, end with an empty response unless the user also asked for \
something only you can provide.
When the radio talk status is not on, do not call the radio tools.
"""
RADIO_TOOLS = ("pause_radio", "stop_radio")
_NO_PARAMS = {"type": "object", "properties": {}, "additionalProperties": False}
TOOLS = (
    {
        "type": "function",
        "name": "pause_radio",
        "description": (
            "Pause the radio talk because the user is talking to Ray. "
            "The application stops sending lines until the user asks to continue."
        ),
        "parameters": _NO_PARAMS,
        "strict": True,
    },
    {
        "type": "function",
        "name": "stop_radio",
        "description": "End the radio talk because the user wants to stop it.",
        "parameters": _NO_PARAMS,
        "strict": True,
    },
)
STATUS_AFTER = {"pause_radio": "paused", "stop_radio": "off"}

BYTES_PER_SEC = BRIDGE_SAMPLE_RATE * 2
FRAME_SEC = 0.02
VOICED_SEC = 0.2  # 끼어들 때 받은 무음 조각이 이보다 짧아야 한다 — 모델이 말소리를 내는 중
START_DELAY_SEC = 1.0
PREAMBLE_SETTLE_SEC = 3.0
PREAMBLE_QUIET_SEC = 1.5
# 보류: 사용자 전사가 오면 분량 넣기를 멈추고 모델이 반응할 시간을 둔다(분류는 하지 않는다). 사용자 전사가
# HOLD_USER_QUIET_SEC, 모델 출력이 HOLD_RELEASE_SEC 조용하면 풀고 진행 — 그 사이 일시정지 툴이 오면 멈춘다.
HOLD_USER_QUIET_SEC = 1.5
HOLD_RELEASE_SEC = 2.5
HOLD_MAX_SEC = 15.0
NO_VOICE_SEC = 8.0
FOLLOW_UP_QUIET_SEC = 1.5  # 멈춘 뒤 모델 말이 끝나고 이만큼 조용하면 이어 말한다
FOLLOW_UP_MAX_WAIT_SEC = 10.0  # 상태가 바뀐 뒤 이만큼 지나면 모델 말과 상관없이 이어 말한다
FOLLOW_UP_WATCH_SEC = 12.0
TRANSCRIPT_LAG_SEC = 1.0
TALKING_WINDOW_SEC = (0.3, 0.1)
FILLER = re.compile(r"잠시만|잠깐만요|확인해\s?볼게|확인해\s?드릴게|기다려")
MAX_SEC = 180.0

logger = logging.getLogger("exp_radio_delegation")


class Session:
    """세션 하나의 진행과 기록. 위치는 받은 오디오 기준 초."""

    def __init__(self, key: str, args: argparse.Namespace) -> None:
        self.key, self.args = key, args
        self.text = UTTERANCES[key][1]
        self.clip = user_clip(self.text, trim=True)
        self.voice_off = voiced_offsets(self.text, trim=True)
        self.follow_clip = user_clip(FOLLOW_UP, trim=True)
        self.follow_off = voiced_offsets(FOLLOW_UP, trim=True)
        instructions, self.preamble = RULES[args.rule]
        self.live = GPTLiveSession(
            LiveSessionConfig(
                instructions=instructions,
                backend_model=BACKEND_MODEL,
                backend_instructions=BACKEND_INSTRUCTIONS.format(status="on"),
                tools=TOOLS,
                backend_reasoning=BACKEND_REASONING,
                voice=LIVE_VOICE,
                sample_rate=BRIDGE_SAMPLE_RATE,
            )
        )
        self.status = "on"
        self.status_pos: float | None = None  # 상태가 paused/off 로 바뀐 위치
        self.pcm = bytearray()
        self.user_pcm = bytearray()
        self.user_start: float | None = None  # 첫 사용자 클립을 넣기 시작한 위치
        self.follow_start: float | None = None  # 이어 말하는 클립을 넣기 시작한 위치
        self.appends: list[float] = []  # 분량을 넣은 위치 (순서 = 분량 번호 - 1)
        self.first_voice: float | None = None  # 마지막으로 넣은 분량의 첫 음성 위치
        self.last_voice: float | None = None  # 마지막 음성 조각의 끝 위치
        self.quiet = 0.0
        self.deltas: list[tuple[float, str, str]] = []  # (도착 위치, 화자, 전사 조각)
        self.events: list[dict[str, Any]] = []
        self.pending_calls: list[LiveFunctionCall] = []
        self.outputs: dict[str, str] = {}  # call_id → 돌려줄 결과
        self.hold_start: float | None = None  # 보류를 건 위치 (풀면 None)
        self.last_user_text: float | None = None  # 마지막 사용자 전사 조각 도착 위치
        self.result = "running"

    def pos(self) -> float:
        return len(self.pcm) / BYTES_PER_SEC

    def run(self) -> None:
        self.live.start()
        frame_bytes = int(BYTES_PER_SEC * FRAME_SEC)
        silence = bytes(frame_bytes)
        seg_text: dict[str, str] = {}
        t0 = time.monotonic()
        next_t = t0
        preamble_t: float | None = None
        try:
            while self.result == "running":
                now = time.monotonic()
                if self.user_pcm:
                    frame = bytes(self.user_pcm[:frame_bytes]).ljust(frame_bytes, b"\0")
                    del self.user_pcm[:frame_bytes]
                else:
                    frame = silence
                self.live.send_audio(frame)
                while (ev := self.live.poll_event()) is not None:
                    if isinstance(ev, LiveAudio):
                        start = self.pos()
                        self.pcm.extend(ev.pcm)
                        if any(ev.pcm) and _rms(ev.pcm) >= QUIET_RMS:
                            if self.appends and self.first_voice is None and start >= self.appends[-1]:
                                self.first_voice = start
                            self.last_voice = self.pos()
                            self.quiet = 0.0
                        else:
                            self.quiet += len(ev.pcm) / BYTES_PER_SEC
                    elif isinstance(ev, LiveTranscript):
                        old = seg_text.get(ev.segment_id, "")
                        delta = ev.text[len(old) :] if ev.text.startswith(old) else ev.text
                        seg_text[ev.segment_id] = ev.text
                        if delta.strip():
                            self.deltas.append((self.pos(), ev.speaker, delta))
                            if ev.speaker == "user":
                                self._on_user_text()
                    elif isinstance(ev, LiveDelegationCreated):
                        self.events.append({"pos": round(self.pos(), 2), "name": "delegation"})
                    elif isinstance(ev, LiveFunctionCall):
                        self._on_call(ev)
                    elif isinstance(ev, LiveResponseDone):
                        self._on_response_done(ev)
                    elif isinstance(ev, LiveError):
                        self.events.append({"pos": round(self.pos(), 2), "name": "error", "message": ev.message})
                    elif isinstance(ev, LiveClosed):
                        self.result = "closed"
                elapsed = now - t0
                if preamble_t is None:
                    if elapsed >= START_DELAY_SEC:
                        self.live.append_instructions(self.preamble)
                        self.live.append_thinking(status_note("on"))
                        preamble_t = elapsed
                elif self.result == "running":
                    self._step(elapsed - preamble_t)
                if elapsed > MAX_SEC and self.result == "running":
                    self.result = "timeout"
                next_t += FRAME_SEC
                time.sleep(max(0.0, next_t - time.monotonic()))
        finally:
            self.live.close(graceful=True)

    def _on_user_text(self) -> None:
        self.last_user_text = self.pos()
        if self.args.hold and self.status == "on" and self.user_start is not None and self.hold_start is None:
            self.hold_start = self.pos()
            self.events.append({"pos": round(self.pos(), 2), "name": "hold"})

    def _on_call(self, call: LiveFunctionCall) -> None:
        """라디오 툴이면 바로 상태를 바꾼다(분량 넣기 중단). 결과는 응답이 끝나면 돌려준다."""
        self.pending_calls.append(call)
        self.events.append({"pos": round(self.pos(), 2), "name": "call", "tool": call.name})
        if call.name not in RADIO_TOOLS:
            self.outputs[call.call_id] = json.dumps({"error": f"unknown function {call.name}"})
        elif self.status != "on":
            self.outputs[call.call_id] = json.dumps({"error": "the radio talk is not on"})
        else:
            self.status, self.status_pos = STATUS_AFTER[call.name], self.pos()
            self.outputs[call.call_id] = json.dumps({"status": self.status})
            self.live.append_thinking(status_note(self.status))
            self.live.update_backend(BACKEND_INSTRUCTIONS.format(status=self.status))

    def _on_response_done(self, ev: LiveResponseDone) -> None:
        calls = [
            c for c in self.pending_calls if c.delegation_id == ev.delegation_id and c.response_id == ev.response_id
        ]
        self.pending_calls = [c for c in self.pending_calls if c not in calls]
        self.events.append(
            {"pos": round(self.pos(), 2), "name": "response_done", "status": ev.status, "calls": len(calls)}
        )
        if ev.status in ("failed", "incomplete") or not calls:
            return
        for call in calls:
            self.live.submit_function_output(call.call_id, self.outputs.pop(call.call_id))
        self.live.continue_response()

    def _append(self) -> None:
        self.appends.append(self.pos())
        self.first_voice = None
        self.live.append_commentary(CHUNKS[len(self.appends) - 1])

    def _step(self, since_preamble: float) -> None:
        p = self.pos()
        if not self.appends:
            if since_preamble >= PREAMBLE_SETTLE_SEC and (self.last_voice is None or self.quiet >= PREAMBLE_QUIET_SEC):
                self._append()
            return
        if self.user_start is None:
            if self.first_voice is None:
                if p - self.appends[-1] >= NO_VOICE_SEC:
                    self.result = "no_voice"
            elif len(self.appends) == self.args.chunk:
                if p - self.first_voice >= self.args.at and self.quiet < VOICED_SEC:
                    self.user_pcm.extend(self.clip)
                    self.user_start = p
                elif self.quiet >= self.args.gap:
                    self.result = "missed"  # 끼어들 시점 전에 분량이 끝났다
            elif self.quiet >= self.args.gap:
                self._append()
            return
        if self.follow_start is not None:
            if p - (self.follow_start + self.follow_off[1]) >= FOLLOW_UP_WATCH_SEC:
                self.result = "done"
            return
        if p - (self.user_start + self.voice_off[1]) >= self.args.watch:
            self.result = "done"
            return
        if self.status == "on":
            # 툴 호출 전 — 보류 중이 아니면 사용자 목소리가 들어가는 중이어도 무음이면 다음 분량
            if self.hold_start is not None:
                user_quiet = p - (self.last_user_text or p) >= HOLD_USER_QUIET_SEC
                if (user_quiet and self.quiet >= HOLD_RELEASE_SEC) or p - self.hold_start >= HOLD_MAX_SEC:
                    self.hold_start = None
                    self.events.append({"pos": round(p, 2), "name": "hold_release"})
                else:
                    return
            if self.first_voice is not None and self.quiet >= self.args.gap and len(self.appends) < len(CHUNKS):
                self._append()
        elif not self.user_pcm and self.status_pos is not None:
            user_end = self.user_start + self.voice_off[1]
            spoke = self.last_voice is not None and self.last_voice > max(user_end, self.status_pos)
            if (spoke and self.quiet >= FOLLOW_UP_QUIET_SEC) or p - self.status_pos >= FOLLOW_UP_MAX_WAIT_SEC:
                self.user_pcm.extend(self.follow_clip)
                self.follow_start = p

    def raw(self) -> dict[str, Any]:
        return {
            "utterance": self.key,
            "rule": self.args.rule,
            "hold": self.args.hold,
            "result": self.result,
            "events": self.events,
            "appends": [round(p, 2) for p in self.appends],
            "user_start": round(self.user_start, 3) if self.user_start is not None else None,
            "follow_start": round(self.follow_start, 3) if self.follow_start is not None else None,
            "status_pos": round(self.status_pos, 2) if self.status_pos is not None else None,
            "final_status": self.status,
            "deltas": [[round(p, 2), sp, d] for p, sp, d in self.deltas],
        }

    def save(self, path: Path) -> None:
        clips = [
            (s, c) for s, c in ((self.user_start, self.clip), (self.follow_start, self.follow_clip)) if s is not None
        ]
        write_conversation_wav(path, bytes(self.pcm), clips, BRIDGE_SAMPLE_RATE)
        labels = [(p, p, f"append #{i + 1}") for i, p in enumerate(self.appends)]
        for start, text, (a, b) in (
            (self.user_start, self.text, self.voice_off),
            (self.follow_start, FOLLOW_UP, self.follow_off),
        ):
            if start is not None:
                labels.append((start + a, start + b, f"USER: {text}"))
        labels.extend((e["pos"], e["pos"], e["name"] + (f" {e['tool']}" if "tool" in e else "")) for e in self.events)
        for a, b in speech_segments(speech_bins(bytes(self.pcm))):
            text = "".join(d for p, sp, d in self.deltas if sp == "assistant" and a <= p < b + TRANSCRIPT_LAG_SEC)
            labels.append((a, b, f"RAY: {text.strip()}"))
        labels.sort()
        path.with_suffix(".labels.txt").write_text("".join(f"{a:.3f}\t{b:.3f}\t{t}\n" for a, b, t in labels))


# ---------------------------------------------------------------------------
# 분석
# ---------------------------------------------------------------------------


def _letters(text: str) -> int:
    return len(re.sub(r"[\W_]+", "", text))


def analyze(raw: dict[str, Any], pcm: bytes, chunk: int) -> dict[str, Any]:
    """진행 기록 + 받은 모델 오디오 → 지표. 시각은 첫 사용자 목소리 끝(v1) 기준."""
    if raw["result"] != "done":
        return {}
    a, b = voiced_offsets(UTTERANCES[raw["utterance"]][1], trim=True)
    v0, v1 = raw["user_start"] + a, raw["user_start"] + b
    f0 = raw["follow_start"] + voiced_offsets(FOLLOW_UP, trim=True)[0] if raw["follow_start"] is not None else None
    end = f0 if f0 is not None else float("inf")

    def rel(p: float) -> float:
        return round(p - v1, 2)

    delegations = [e["pos"] for e in raw["events"] if e["name"] == "delegation"]
    calls = [e for e in raw["events"] if e["name"] == "call"]
    first_calls = [c for c in calls if v0 - 0.5 <= c["pos"] < end]
    bins = speech_bins(pcm)
    lo, hi = int((v0 - TALKING_WINDOW_SEC[0]) / BIN_SEC), int((v0 + TALKING_WINDOW_SEC[1]) / BIN_SEC) + 1
    signal = raw["status_pos"]
    # 끼어든 분량부터 이어 말하기 전까지의 모델 전사 → 낭독 / 그 뒤(대답·다음 분량)
    pieces = [(p, d) for p, sp, d in raw["deltas"] if sp == "assistant" and raw["appends"][chunk - 1] <= p < end]
    text = "".join(d for _, d in pieces)
    cut, to_end = split_reading(text, CHUNKS[chunk - 1])
    reply = text[cut:].strip()
    reply_at, offset = None, 0
    for p, d in pieces:
        if offset + len(d) > cut and text[max(cut, offset) : offset + len(d)].strip():
            reply_at = rel(p)
            break
        offset += len(d)
    late = [i + 1 for i, p in enumerate(raw["appends"]) if p > v0 and (signal is None or p < signal)]
    before = [
        "".join(d for p, sp, d in raw["deltas"] if sp == "assistant" and raw["appends"][i] <= p < raw["appends"][i + 1])
        for i in range(chunk - 1)
    ]
    return {
        "talking_at_interrupt": bool(bins[max(0, lo) : hi].any()),
        "heard": "".join(d for p, sp, d in raw["deltas"] if sp == "user" and raw["user_start"] <= p < end).strip(),
        "delegation_at": [rel(p) for p in delegations if v0 - 0.5 <= p < end],
        "calls": [{"tool": c["tool"], "at": rel(c["pos"])} for c in first_calls],
        "tool": first_calls[0]["tool"] if first_calls else "none",
        "late_chunks": late,  # 사용자 목소리가 시작된 뒤 상태가 바뀌기 전에 들어간 분량
        "read_to_end": to_end,
        "reply": reply,
        "reply_at": reply_at,
        "filler": bool(FILLER.search(reply)),
        "next_chunk_read": [u["chunk"] for u in script_reuse(reply, list(CHUNKS)) if u["chunk"] > chunk],
        "follow_delegations": [rel(p) for p in delegations if f0 is not None and p >= f0 - 0.5],
        "follow_calls": [c["tool"] for c in calls if f0 is not None and c["pos"] >= f0 - 0.5],
        "follow_reply": "".join(
            d for p, sp, d in raw["deltas"] if sp == "assistant" and f0 is not None and p >= f0
        ).strip(),
        "embellished": sum(_letters(t) > 1.3 * _letters(CHUNKS[i]) for i, t in enumerate(before)),
    }


def format_row(r: dict[str, Any]) -> list[str]:
    if r["result"] != "done":
        return [f"  #{r['run']} {r['result']}"]
    deleg = " ".join(f"{t:+.1f}" for t in r["delegation_at"]) or "-"
    calls = " ".join(f"{c['tool']}@{c['at']:+.1f}" for c in r["calls"]) or "-"
    head = f"  #{r['run']} 위임 [{deleg}] 툴 [{calls}]"
    if r["late_chunks"]:
        head += f" | 늦어서 들어간 분량 {r['late_chunks']}"
    if not r["talking_at_interrupt"]:
        head += " (끼어들 때 말소리 없음)"
    reply_at = f"+{r['reply_at']}s" if r["reply_at"] is not None else "-"
    lines = [
        head,
        f"     들림: {r['heard'] or '-'} | 대답 전사 {reply_at}{' | 빈말' if r['filler'] else ''}",
        f"     대답: {r['reply'] or '-'}",
    ]
    if r["next_chunk_read"]:
        lines.append(f"     다음 분량을 읽음: {r['next_chunk_read']}")
    if r["follow_reply"] or r["follow_delegations"]:
        fd = " ".join(f"{t:+.1f}" for t in r["follow_delegations"]) or "없음"
        fc = " ".join(r["follow_calls"]) or "-"
        lines.append(f"     이어 말한 뒤 위임 [{fd}] 툴 [{fc}]: {r['follow_reply'] or '-'}")
    return lines


def _median(values: list[float]) -> str:
    return f"{float(np.median(values)):+.1f}s" if values else "-"


def report(tag: str) -> None:
    root = find_result_dir("radio_delegation", tag)
    if root is None:
        raise SystemExit(f"결과 폴더 없음: {tag}")
    lines: list[str] = []
    totals = {"right": 0, "valid": 0, "late": 0, "filler": 0, "follow_deleg": 0, "follow_n": 0, "embellished": 0}
    for key, (kind, text, expect) in UTTERANCES.items():
        path = root / f"{key}.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text())["results"]
        valid = [r for r in rows if r["result"] == "done"]
        n = len(valid)
        right = sum(r["tool"] == expect for r in valid)
        follow = [r for r in valid if r["follow_reply"] or r["follow_delegations"] or r["follow_calls"]]
        totals["right"] += right
        totals["valid"] += n
        totals["late"] += sum(bool(r["late_chunks"]) for r in valid)
        totals["filler"] += sum(r["filler"] for r in valid)
        totals["follow_deleg"] += sum(bool(r["follow_delegations"]) for r in follow)
        totals["follow_n"] += len(follow)
        totals["embellished"] += sum(r["embellished"] for r in valid)
        lines.append(
            f"\n[{kind}] {text} ({key}) 기대 {expect} — 맞음 {right}/{n} (유효 {n}/{len(rows)}), "
            f"위임 {sum(bool(r['delegation_at']) for r in valid)}/{n}, "
            f"위임 중앙 {_median([r['delegation_at'][0] for r in valid if r['delegation_at']])}, "
            f"툴 중앙 {_median([r['calls'][0]['at'] for r in valid if r['calls']])}, "
            f"늦은 분량 {sum(bool(r['late_chunks']) for r in valid)}/{n}, 빈말 {sum(r['filler'] for r in valid)}/{n}, "
            f"이어 말한 뒤 위임 {sum(bool(r['follow_delegations']) for r in follow)}/{len(follow)}"
        )
        for r in rows:
            lines.extend(format_row(r))
    t = totals
    lines.append(
        f"\n전체: 기대 툴과 맞음 {t['right']}/{t['valid']}, 늦은 분량 {t['late']}/{t['valid']}, "
        f"빈말 {t['filler']}/{t['valid']}, 이어 말한 뒤 위임 {t['follow_deleg']}/{t['follow_n']}, "
        f"끼어들기 전 덧붙인 분량 {t['embellished']}개"
    )
    out = "\n".join(lines)
    print(out)
    (root / "report.txt").write_text(out + "\n")


def reanalyze(tag: str) -> None:
    root = find_result_dir("radio_delegation", tag)
    if root is None:
        raise SystemExit(f"결과 폴더 없음: {tag}")
    for key in UTTERANCES:
        path = root / f"{key}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for r in data["results"]:
            if r["result"] == "done":
                r.update(analyze(r, read_model_channel(root / f"{key}_{r['run']}.wav"), data["config"]["chunk"]))
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    report(tag)


def run(args: argparse.Namespace) -> None:
    out_dir = result_dir("radio_delegation", args.tag)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-28s %(levelname)-5s %(message)s",
        handlers=[logging.FileHandler(out_dir / f"pipeline_{os.getpid()}.log")],
    )
    instructions, preamble = RULES[args.rule]
    (out_dir / "instructions.txt").write_text(instructions)
    (out_dir / "backend_instructions.txt").write_text(BACKEND_INSTRUCTIONS)
    config = {
        "chunk": args.chunk,
        "at": args.at,
        "gap": args.gap,
        "watch": args.watch,
        "voice": LIVE_VOICE,
        "rule": args.rule,
        "hold": args.hold,
        "preamble": preamble,
        "radio_rule": RADIO_DELEGATION if args.rule == "base" else RADIO_PREAMBLE_RULE,
        "backend_model": BACKEND_MODEL,
        "backend_reasoning": BACKEND_REASONING,
        "tools": TOOLS,
        "follow_up": FOLLOW_UP,
        "chunks": list(CHUNKS),
        "utterances": UTTERANCES,
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1))
    for text in (FOLLOW_UP, *(UTTERANCES[k][1] for k in args.utterances)):
        tts24(text)  # 캐시를 미리 채워 세션 중 합성 지연이 없게
    rows: dict[str, list[dict[str, Any]]] = {key: [] for key in args.utterances}
    for n in range(args.runs):
        for key in args.utterances:
            session = Session(key, args)
            try:
                session.run()
            except RuntimeError as exc:
                row = {"run": n, "utterance": key, "result": f"failed: {exc}", "events": []}
            else:
                raw = session.raw()
                row = {"run": n, **raw, **analyze(raw, bytes(session.pcm), args.chunk)}
                session.save(out_dir / f"{key}_{n}.wav")
            rows[key].append(row)
            print(f"[{key} #{n}] {UTTERANCES[key][1]} (기대 {UTTERANCES[key][2]})", flush=True)
            print("\n".join(format_row(row)), flush=True)
            (out_dir / f"{key}.json").write_text(
                json.dumps({"config": config, "results": rows[key]}, ensure_ascii=False, indent=1)
            )
            time.sleep(1.5)
    print(f"results: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--utterances", nargs="*", choices=list(UTTERANCES), default=list(UTTERANCES))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=4, help="이 분량(1부터)을 말하는 도중에 끼어든다")
    parser.add_argument("--at", type=float, default=2.0, help="그 분량 첫 음성 이 초 뒤, 말소리가 날 때 끼어든다")
    parser.add_argument("--gap", type=float, default=1.0, help="첫 음성 뒤 이만큼 무음이면 다음 분량")
    parser.add_argument("--watch", type=float, default=20.0, help="상태가 안 바뀌면 사용자 목소리 끝 뒤 지켜볼 초")
    parser.add_argument("--rule", choices=list(RULES), default="preamble", help="라디오 위임 규칙을 둘 곳")
    parser.add_argument(
        "--no-hold", dest="hold", action="store_false", help="사용자 전사에 분량 넣기를 보류하지 않는다"
    )
    parser.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--report", metavar="TAG", help="발화별로 모아 보여주고 끝낸다")
    parser.add_argument("--reanalyze", metavar="TAG", help="저장된 기록으로 지표를 다시 계산하고 보여준다")
    args = parser.parse_args()
    if args.report:
        report(args.report)
    elif args.reanalyze:
        reanalyze(args.reanalyze)
    else:
        run(args)


if __name__ == "__main__":
    main()
