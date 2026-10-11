"""라디오 흐름 실험 — 끼어들기는 앱의 판정기(LLM)가, 멈춘 뒤 재개는 대화 모델의 위임이 맡는다.

대화 모델은 끼어들기에 대해 아무것도 모른다(``read`` 안내뿐). 앱은 사용자 전사가 오면 분량 넣기를 보류하고 판정기
(``classify``, ``adapters/llm_openai`` 의 ``OpenAILLM``)에 "지금 라디오가 어떤 상태여야 하는가"를 묻는다 — on(진행) /
paused(일시정지, 대화) / off(종료). 입력은 낭독 중이던 대사와 끼어든 뒤의 대화(사용자·레이 전사).
판정은 모델의 반응을 기다리지 않는다: 사용자 전사가 ``USER_QUIET_SEC`` 멈추면 부르고, 그 뒤 모델이 대사가 아닌 말을
``RECHECK_LETTERS`` 글자 넘게 더 하면 그 말까지 넣어 다시 부른다(대사인지는 ``split_reading``). paused·off 는 오는 즉시
적용한다. on 은 그 판정 뒤 새 사용자 말도, 새 반응(``RECHECK_LETTERS`` 글자 이상)도 없을 때만 — 모델 무음 ``--gap``
에 다음 분량을 넣는 것으로 — 적용한다. ``CLASSIFY_TIMEOUT_SEC`` 안에 결과가 없으면 on 으로 보고, 늦게 온
paused·off 는 적용한다.

멈추면 thinking 으로 상태를 알린다(``PAUSE_NOTE`` — 나머지 이야기는 모델에게 없고, 이어 달라고 하면 위임하라).
재개는 위임 → 백엔드 ``resume_radio`` → 앱이 곧바로 다음 분량을 넣는다(모델이 말하는 중이어도). 기본 지시문에는
Backend tools 와 위임 목록에 라디오 한 줄씩(``RADIO_TOOLS_LINE``, ``RADIO_DELEGATION`` — 운영에서도 라디오 시작은 이
목록으로 받는다).

다음 분량은 모델 무음 ``--gap`` 에 넣되, 지금 대사가 아직 끝까지 전사되지 않았으면(모델이 대사 도중에 멈췄다 —
사용자가 끼어들었을 수 있다) ``--midline-wait`` 를 더 기다린다. 보류 중 on 판정으로 넣을 때는 적용하지 않는다.

흐름: ``--interrupt mid`` 는 ``--chunk`` 번째 분량 첫 음성 ``--at`` 초 뒤 말하는 도중 끼어든다. ``gap`` 은 그 대사를
끝까지 말하고 모델이 ``--gap-at`` 초 조용할 때 끼어든다 — 앱은 전사가 오기 전에는 모르므로 그대로 다음 분량을 넣는다.
멈추면(paused) 모델 말이 끝나고 ``NEXT_TURN_QUIET_SEC`` 조용할 때마다 사용자가 차례로 말한다 — 일반 대화
``FOLLOW_UP``(위임이 없어야 한다), 이어 재개 요청(``RESUME_ASKS`` 중 하나, 회차마다 번갈아). 종료(off)면
``FOLLOW_UP`` 만. 진행(on)이면 분량이 계속 들어가고 사용자 목소리 끝 ``--watch`` 초 뒤 끝낸다.

    # 프로세스 여럿으로 나눠 병렬로 (같은 태그 폴더에 발화별 파일로 모인다)
    uv run python scripts/gpt_live/exp_radio_flow.py --tag f1 --utterances content_why content_key
    uv run python scripts/gpt_live/exp_radio_flow.py --report f1

결과: var/experiments/radio_flow/<번호>_<태그>/ — <발화>.json(판정기 호출의 입력·출력 포함), <발화>_<회차>.wav(왼쪽
사용자, 오른쪽 모델) + .labels.txt, config.json, instructions.txt, pipeline_<pid>.log, report.txt(--report).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from exp_audio import read_model_channel, write_conversation_wav
from exp_barge_in import BIN_SEC, speech_bins, speech_segments, user_clip, voiced_offsets
from exp_dirs import find_result_dir, result_dir
from exp_dj_commentary_chunks import CHUNKS, LIVE_INSTRUCTIONS, PREAMBLES, QUIET_RMS, _rms, tts24
from exp_dj_interrupt_reply import script_reuse, split_reading
from exp_radio_delegation import FILLER, FOLLOW_UP
from exp_radio_delegation import UTTERANCES as DELEGATION_UTTERANCES

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
from voice_pipeline.adapters.llm_openai import OpenAILLM
from voice_pipeline.engines.gpt_live.instructions import LIVE_VOICE
from voice_pipeline.settings import BRIDGE_SAMPLE_RATE

_EXPECT = {"none": "on", "pause_radio": "paused", "stop_radio": "off"}
# 발화 키 → (종류, 문장, 기대 상태). 위임 실험과 같은 19개.
UTTERANCES: dict[str, tuple[str, str, str]] = {
    k: (kind, text, _EXPECT[tool]) for k, (kind, text, tool) in DELEGATION_UTTERANCES.items()
}
RESUME_ASKS = ("라디오 다시 이어서 해 줘.", "이야기 계속해 줘.")

# 대화 모델 ---------------------------------------------------------------
PREAMBLE = PREAMBLES["read"]
RADIO_TOOLS_LINE = "- Radio: starting or resuming the radio talk.\n"
RADIO_DELEGATION = "- The user asks to start the radio, or to continue a paused radio talk.\n"
_TOOLS_ANCHOR = "- End of conversation:"
_DELEGATION_ANCHOR = "Say a short goodbye yourself at the same time.\n"


def radio_instructions(base: str) -> str:
    assert _TOOLS_ANCHOR in base and _DELEGATION_ANCHOR in base
    out = base.replace(_TOOLS_ANCHOR, RADIO_TOOLS_LINE + _TOOLS_ANCHOR, 1)
    return out.replace(_DELEGATION_ANCHOR, _DELEGATION_ANCHOR + RADIO_DELEGATION, 1)


LIVE_RADIO_INSTRUCTIONS = radio_instructions(LIVE_INSTRUCTIONS)
# thinking 으로 넣는 상태. 재개를 모델 혼자 할 수 없는 일로 적는다 — 위임은 그런 일에서만 걸렸다.
PAUSE_NOTE = (
    "Radio talk: paused, because the user started talking to you. Talk with the user now. "
    "You do not have the rest of the talk; if the user asks to continue the talk or the radio, "
    "delegate to the backend to resume it."
)
RESUME_NOTE = "Radio talk: resumed. Read the application's lines as before."
END_NOTE = "Radio talk: ended. Talk with the user now."

# 백엔드 -----------------------------------------------------------------
BACKEND_MODEL = "gpt-5.4-mini"  # 운영과 같게 (wiring._LIVE_BACKEND_MODEL)
BACKEND_REASONING = "low"
BACKEND_INSTRUCTIONS = """\
# Backend for Ray
Ray is a small desk robot talking with the user by voice. Ray delegates requests to you and speaks your answer.

## Radio
Ray hosts a radio talk: the application sends lines and Ray reads them aloud.
Radio talk status: {status}

- If the radio talk status is paused and the user asks to continue the talk or the radio, call resume_radio. \
While it is paused, 계속해, 이어서 해 줘, 이야기 계속해 줘 and 라디오 이어서 해 줘 all ask to continue the radio talk.
- Do not call resume_radio in any other case.
Ray answers the user directly. After resume_radio, end with an empty response.
"""
TOOLS = (
    {
        "type": "function",
        "name": "resume_radio",
        "description": "Resume the paused radio talk. The application sends Ray the next line.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
)

# 판정기 -----------------------------------------------------------------
CLASSIFIER_MODEL = "gpt-5.4-mini"
CLASSIFIER_INSTRUCTIONS = """\
Ray, a small desk robot, is hosting a radio talk: it reads prepared lines aloud, one at a time. \
The user spoke while Ray was reading. Decide what the radio should do now, from what the user said and \
how Ray responded. The transcripts come from speech recognition and may contain errors.

- on: the user only reacted briefly (for example 응, 아 진짜, 맞아, 그렇구나), or Ray ignored it and kept reading. \
The talk goes on with the next line.
- paused: the user is talking to Ray: a question (also about the talk itself), a comment, a complaint, a request, \
or asking Ray to wait. Ray stops the talk and talks with the user.
- off: the user wants to stop or end the radio.
"""
CLASSIFIER_FORMAT = {
    "type": "json_schema",
    "name": "radio_state",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "One short sentence."},
            "state": {"type": "string", "enum": ["on", "paused", "off"]},
        },
        "required": ["reason", "state"],
        "additionalProperties": False,
    },
}
CLASSIFY_TIMEOUT_SEC = 2.0

# 진행 -------------------------------------------------------------------
BYTES_PER_SEC = BRIDGE_SAMPLE_RATE * 2
FRAME_SEC = 0.02
VOICED_SEC = 0.2  # 끼어들 때 받은 무음 조각이 이보다 짧아야 한다 — 모델이 말소리를 내는 중
START_DELAY_SEC = 1.0
PREAMBLE_SETTLE_SEC = 3.0
PREAMBLE_QUIET_SEC = 1.5
NO_VOICE_SEC = 8.0
USER_QUIET_SEC = 0.6  # 사용자 전사가 이만큼 멈추면 판정
RECHECK_LETTERS = 6  # 판정 뒤 모델이 대사가 아닌 말을 이만큼(공백·부호 제외) 더 하면 다시 판정
HOLD_MAX_QUIET_SEC = 10.0  # 판정을 기다리지 않는데 모델이 이만큼 조용하면 보류를 푼다 (안전장치)
NEXT_TURN_QUIET_SEC = 1.5  # 멈춘 뒤 모델 말이 끝나고 이만큼 조용하면 사용자가 다음 말을 한다
NEXT_TURN_MAX_WAIT_SEC = 10.0
TURN_WATCH_SEC = 15.0  # 마지막 사용자 말 뒤 지켜볼 초 (멈춘 경로)
TRANSCRIPT_LAG_SEC = 1.0
TALKING_WINDOW_SEC = (0.3, 0.1)
MAX_SEC = 200.0

logger = logging.getLogger("exp_radio_flow")


def classify(llm: OpenAILLM, line: str, exchange: list[tuple[str, str]]) -> dict[str, Any]:
    """판정기 한 번. ``exchange`` 는 끼어든 뒤의 (화자, 전사) 차례."""
    dialog = "\n".join(f"{'User' if who == 'user' else 'Ray'}: {text}" for who, text in exchange)
    prompt = f"Line Ray was reading: {line}\n\nSince the user spoke:\n{dialog}"
    stream = llm.generate(
        [{"role": "system", "content": CLASSIFIER_INSTRUCTIONS}, {"role": "user", "content": prompt}],
        tools=[],
        response_format=CLASSIFIER_FORMAT,
    )
    return json.loads("".join(stream))


def group_turns(deltas: list[tuple[float, str, str]], since: float) -> list[tuple[str, str]]:
    """``since`` 뒤에 도착한 전사 조각을 화자가 바뀔 때마다 묶는다."""
    turns: list[list[str]] = []
    for p, who, d in deltas:
        if p < since:
            continue
        if turns and turns[-1][0] == who:
            turns[-1][1] += d
        else:
            turns.append([who, d])
    return [(who, text.strip()) for who, text in turns if text.strip()]


class Session:
    """세션 하나의 진행과 기록. 위치는 받은 오디오 기준 초."""

    def __init__(self, key: str, run: int, args: argparse.Namespace, llm: OpenAILLM, pool: ThreadPoolExecutor) -> None:
        self.key, self.args, self.llm, self.pool = key, args, llm, pool
        self.text = UTTERANCES[key][1]
        self.resume_ask = RESUME_ASKS[(run + list(UTTERANCES).index(key)) % len(RESUME_ASKS)]
        self.turns = [(self.text, user_clip(self.text, trim=True), voiced_offsets(self.text, trim=True))]
        self.live = GPTLiveSession(
            LiveSessionConfig(
                instructions=LIVE_RADIO_INSTRUCTIONS,
                backend_model=BACKEND_MODEL,
                backend_instructions=BACKEND_INSTRUCTIONS.format(status="on"),
                tools=TOOLS,
                backend_reasoning=BACKEND_REASONING,
                voice=LIVE_VOICE,
                sample_rate=BRIDGE_SAMPLE_RATE,
            )
        )
        self.status = "on"
        self.status_pos: float | None = None  # 판정으로 처음 멈추거나 끝난 위치 — 그 뒤는 사용자 차례 진행
        self.plan: list[str] = []  # 멈춘 뒤 사용자가 차례로 할 말
        self.pcm = bytearray()
        self.user_pcm = bytearray()
        self.user_turns: list[dict[str, Any]] = []  # 사용자가 말한 차례 {"text", "start"}
        self.appends: list[float] = []
        self.first_voice: float | None = None
        self.last_voice: float | None = None
        self.quiet = 0.0
        self.deltas: list[tuple[float, str, str]] = []  # (도착 위치, 화자, 전사 조각)
        self.events: list[dict[str, Any]] = []
        self.pending_calls: list[LiveFunctionCall] = []
        self.outputs: dict[str, str] = {}
        self.hold_start: float | None = None
        self.last_user_text: float | None = None
        self.script_from = 0.0  # 보류 때 낭독 중이던 분량을 넣은 위치 — 반응 글자는 그 뒤 전사로 센다
        self.judged_user = -1.0  # 마지막 판정 호출이 본 사용자 전사 위치
        self.judged_reaction = 0  # 마지막 판정 호출이 본 반응 글자 수
        self.latest_on: dict[str, Any] | None = None  # 가장 최근의 on 판정 (시간 초과 대체 포함)
        self.midline_noted = 0  # 대사 중간 멈춤 대기를 기록한 분량 번호
        self.interrupt_quiet: float | None = None  # 끼어들 때 모델 무음 길이
        self.inflight: list[tuple[Future[dict[str, Any]], dict[str, Any]]] = []
        self.calls: list[dict[str, Any]] = []  # 판정 기록
        self.preamble: str | None = PREAMBLE  # 라디오 시작 때 instructions.append (없으면 넣지 않는다)
        self.result = "running"

    def pos(self) -> float:
        return len(self.pcm) / BYTES_PER_SEC

    # -- 루프 ----------------------------------------------------------------

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
                        if self.preamble:
                            self.live.append_instructions(self.preamble)
                        preamble_t = elapsed
                elif self.result == "running":
                    self._poll_classifier()
                    self._step(elapsed - preamble_t)
                if elapsed > MAX_SEC and self.result == "running":
                    self.result = "timeout"
                next_t += FRAME_SEC
                time.sleep(max(0.0, next_t - time.monotonic()))
        finally:
            self.live.close(graceful=True)

    def _event(self, name: str, **fields: Any) -> None:
        self.events.append({"pos": round(self.pos(), 2), "name": name, **fields})

    def _say(self, i: int) -> None:
        text, clip, _ = self.turns[i]
        self.user_pcm.extend(clip)
        self.user_turns.append({"text": text, "start": round(self.pos(), 3)})

    def _append(self) -> None:
        self.appends.append(self.pos())
        self.first_voice = None
        self.live.append_commentary(CHUNKS[len(self.appends) - 1])

    def _set_status(self, status: str) -> None:
        self.status = status
        self._event("status", status=status)
        self.live.append_thinking({"paused": PAUSE_NOTE, "on": RESUME_NOTE, "off": END_NOTE}[status])
        self.live.update_backend(BACKEND_INSTRUCTIONS.format(status=status))

    # -- 판정기 --------------------------------------------------------------

    def _on_user_text(self) -> None:
        self.last_user_text = self.pos()
        if self.status == "on" and self.status_pos is None and self.user_turns and self.hold_start is None:
            self.hold_start = self.pos()
            self.script_from = self.appends[-1]
            self._event("hold")

    def _reaction_letters(self) -> int:
        """보류 때 낭독 중이던 대사를 벗어난 모델 전사 글자 수."""
        text = "".join(d for p, sp, d in self.deltas if sp == "assistant" and p >= self.script_from)
        cut, _ = split_reading(text, CHUNKS[len(self.appends) - 1])
        return _letters(text[cut:])

    def _stale(self, record: dict[str, Any]) -> bool:
        new_user = self.last_user_text is not None and self.last_user_text > record["upto_user"]
        return new_user or self._reaction_letters() - record["upto_reaction"] >= RECHECK_LETTERS

    def _waiting(self) -> bool:
        return any(not r.get("timeout") for _, r in self.inflight)

    def _current_line(self) -> str:
        """판정기에 줄 '낭독 중이던 대사' — 마지막으로 넣은 분량."""
        return CHUNKS[len(self.appends) - 1]

    def _maybe_classify(self, p: float) -> None:
        if self.last_user_text is None or self._waiting():
            return
        new_user = self.last_user_text > self.judged_user
        if new_user and p - self.last_user_text < USER_QUIET_SEC:
            return  # 사용자 전사가 아직 들어오는 중
        reaction = self._reaction_letters()
        if not new_user and reaction - self.judged_reaction < RECHECK_LETTERS:
            return
        exchange = group_turns(self.deltas, self.user_turns[0]["start"])
        line = self._current_line()
        record = {
            "pos": round(p, 2),
            "upto_user": self.last_user_text,
            "upto_reaction": reaction,
            "by": "user" if new_user else "reaction",
            "line": line,
            "exchange": exchange,
        }
        self.judged_user, self.judged_reaction = self.last_user_text, reaction
        self.inflight.append((self.pool.submit(classify, self.llm, line, exchange), record))
        self._event("classify", by=record["by"])

    def _poll_classifier(self) -> None:
        p = self.pos()
        for item in list(self.inflight):
            future, record = item
            if future.done():
                self.inflight.remove(item)
                record["latency"] = round(p - record["pos"], 2)
                try:
                    out = future.result()
                    record["state"], record["reason"] = out["state"], out["reason"]
                except Exception as exc:  # noqa: BLE001 — 판정 실패는 on 으로
                    record["state"], record["reason"] = "on", f"error: {exc}"
                    record["error"] = True
                record["stale"] = self._stale(record)
                if record["state"] != "on":
                    record["applied"] = self._decide(record["state"], "classifier")
                elif not record.get("timeout"):
                    self.latest_on = record  # 시간 초과로 이미 on 으로 둔 호출이면 그대로
                self.calls.append(record)
            elif p - record["pos"] >= CLASSIFY_TIMEOUT_SEC and not record.get("timeout"):
                record["timeout"] = True  # 결과는 계속 기다린다 — 늦게 온 paused·off 는 적용
                self.latest_on = record

    def _decide(self, state: str, by: str) -> bool:
        """paused·off 적용 — 라디오가 진행 중이면 언제나. 적용했으면 True."""
        if self.status != "on" or self.status_pos is not None:
            return False
        self.hold_start = None
        self.status_pos = self.pos()
        self.plan = [FOLLOW_UP] + ([self.resume_ask] if state == "paused" else [])
        self._set_status(state)
        self._event("decision", state=state, by=by)
        return True

    def _hold_step(self, p: float, can_append: bool) -> None:
        """보류 중: 판정을 부르고, 최신 on 판정이 유효하면 모델 무음 ``--gap`` 에 다음 분량을 넣고 보류를 푼다."""
        self._maybe_classify(p)
        on = self.latest_on
        if on is not None and can_append and not self._waiting() and not self._stale(on):
            on["applied"] = True
            self.hold_start = None
            self._event("decision", state="on", by="timeout" if on.get("timeout") else "classifier")
            self._append()
        elif not self._waiting() and self.quiet >= HOLD_MAX_QUIET_SEC:
            self.hold_start = None
            self._event("hold_release", by="max")

    # -- 백엔드 --------------------------------------------------------------

    def _on_call(self, call: LiveFunctionCall) -> None:
        self.pending_calls.append(call)
        self._event("call", tool=call.name)
        if call.name != "resume_radio":
            self.outputs[call.call_id] = json.dumps({"error": f"unknown function {call.name}"})
        elif self.status != "paused":
            self.outputs[call.call_id] = json.dumps({"error": "the radio talk is not paused"})
        else:
            self.outputs[call.call_id] = json.dumps({"status": "on"})
            self._set_status("on")
            if len(self.appends) < len(CHUNKS):
                self._append()  # 모델이 말하는 중이어도 곧바로 — 지어내던 말을 버리고 대사로 넘어가게

    def _on_response_done(self, ev: LiveResponseDone) -> None:
        mine = (ev.delegation_id, ev.response_id)
        calls = [c for c in self.pending_calls if (c.delegation_id, c.response_id) == mine]
        self.pending_calls = [c for c in self.pending_calls if c not in calls]
        self._event("response_done", status=ev.status, calls=len(calls))
        if ev.status in ("failed", "incomplete") or not calls:
            return
        for call in calls:
            self.live.submit_function_output(call.call_id, self.outputs.pop(call.call_id))
        self.live.continue_response()

    # -- 진행 ----------------------------------------------------------------

    def _turn_end(self, i: int) -> float:
        return self.user_turns[i]["start"] + self.turns[i][2][1]

    def _line_done(self) -> bool:
        """마지막으로 넣은 대사가 끝까지 전사됐는지."""
        text = "".join(d for p, sp, d in self.deltas if sp == "assistant" and p >= self.appends[-1])
        return split_reading(text, CHUNKS[len(self.appends) - 1])[1]

    def _ready_for_next(self) -> bool:
        """평소 진행에서 다음 분량을 넣을 때인지 — 무음 ``--gap``, 대사 도중 멈췄으면 ``--midline-wait`` 더."""
        if self.first_voice is None or len(self.appends) >= len(CHUNKS) or self.quiet < self.args.gap:
            return False
        if self._line_done() or self.quiet >= self.args.gap + self.args.midline_wait:
            return True
        if self.midline_noted != len(self.appends):
            self.midline_noted = len(self.appends)
            self._event("midline_wait")
        return False

    def _interrupt(self) -> None:
        self.interrupt_quiet = round(self.quiet, 2)
        self._say(0)

    def _step(self, since_preamble: float) -> None:
        p = self.pos()
        if not self.appends:
            if since_preamble >= PREAMBLE_SETTLE_SEC and (self.last_voice is None or self.quiet >= PREAMBLE_QUIET_SEC):
                self._append()
            return
        if not self.user_turns:  # 끼어들기 전
            if self.first_voice is None:
                if p - self.appends[-1] >= NO_VOICE_SEC:
                    self.result = "no_voice"
            elif len(self.appends) == self.args.chunk:
                if self.args.interrupt == "mid":
                    if p - self.first_voice >= self.args.at and self.quiet < VOICED_SEC:
                        self._interrupt()
                    elif self.quiet >= self.args.gap:
                        self.result = "missed"  # 끼어들 시점 전에 분량이 끝났다
                elif self.quiet >= self.args.gap_at and self._line_done():
                    self._interrupt()  # 대사를 마치고 조용해진 틈
                elif self.quiet >= self.args.gap + self.args.midline_wait:
                    self.result = "missed"
            elif self._ready_for_next():
                self._append()
            return
        last = len(self.user_turns) - 1
        since_last = p - self._turn_end(last)
        can_append = self.first_voice is not None and self.quiet >= self.args.gap and len(self.appends) < len(CHUNKS)
        if self.status_pos is None:
            # 판정 전이거나 on — 보류 중이면 판정을 기다리고, 아니면 무음마다 다음 분량. 앱은 전사가 오기 전에는
            # 사용자 목소리를 모르므로 사용자 클립이 들어가는 중에도 넣는다.
            if self.hold_start is not None:
                self._hold_step(p, can_append)
            elif self._ready_for_next():
                self._append()
            if since_last >= self.args.watch:
                self.result = "done"
            return
        # 멈춘(또는 끝난) 뒤: 사용자가 차례로 말한다 — 일반 대화, 그다음 재개 요청(멈춘 경우만)
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

    # -- 기록 ----------------------------------------------------------------

    def raw(self) -> dict[str, Any]:
        return {
            "utterance": self.key,
            "result": self.result,
            "resume_ask": self.resume_ask,
            "interrupt": self.args.interrupt,
            "gap_at": self.args.gap_at if self.args.interrupt == "gap" else None,
            "interrupt_quiet": self.interrupt_quiet,
            "events": self.events,
            "appends": [round(p, 2) for p in self.appends],
            "user_turns": [{**t, "voice": list(self.turns[i][2])} for i, t in enumerate(self.user_turns)],
            "classifier": self.calls,
            "final_status": self.status,
            "deltas": [[round(p, 2), sp, d] for p, sp, d in self.deltas],
        }

    def save(self, path: Path) -> None:
        clips = [(t["start"], self.turns[i][1]) for i, t in enumerate(self.user_turns)]
        write_conversation_wav(path, bytes(self.pcm), clips, BRIDGE_SAMPLE_RATE)
        labels = [(p, p, f"append #{i + 1}") for i, p in enumerate(self.appends)]
        for i, t in enumerate(self.user_turns):
            a, b = self.turns[i][2]
            labels.append((t["start"] + a, t["start"] + b, f"USER: {t['text']}"))
        for e in self.events:
            extra = " ".join(str(v) for k, v in e.items() if k not in ("pos", "name"))
            labels.append((e["pos"], e["pos"], f"{e['name']} {extra}".strip()))
        for c in self.calls:
            done = c["pos"] + c["latency"]
            labels.append((done, done, f"judge {c['state']} ({c['reason']})"))
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


def _voiced(pcm: bytes) -> np.ndarray:
    """20 ms 단위 말소리 여부 — 앱의 무음 판정과 같은 문턱(``QUIET_RMS``)."""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    hop = int(BRIDGE_SAMPLE_RATE * FRAME_SEC)
    n = len(x) // hop
    return np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1)) >= QUIET_RMS


def _gap_at(voiced: np.ndarray, p: float) -> float | None:
    """``p`` 에 넣은 분량 앞뒤 무음 — 그 전 마지막 말소리 끝부터 그 뒤 첫 말소리까지."""
    i = int(p / FRAME_SEC)
    before, after = np.flatnonzero(voiced[:i]), np.flatnonzero(voiced[i:])
    if not len(before) or not len(after):
        return None
    return round((i + int(after[0]) - int(before[-1]) - 1) * FRAME_SEC, 2)


def _assistant(raw: dict[str, Any], a: float, b: float) -> str:
    return "".join(d for p, sp, d in raw["deltas"] if sp == "assistant" and a <= p < b).strip()


def analyze(raw: dict[str, Any], pcm: bytes, chunk: int) -> dict[str, Any]:
    """진행 기록 + 받은 모델 오디오 → 지표. 시각은 첫 사용자 목소리 끝(v1) 기준."""
    if raw["result"] != "done":
        return {}
    turns = raw["user_turns"]
    v0, v1 = turns[0]["start"] + turns[0]["voice"][0], turns[0]["start"] + turns[0]["voice"][1]
    starts = [t["start"] + t["voice"][0] for t in turns] + [float("inf")]
    ends = [t["start"] + t["voice"][1] for t in turns]
    status = [e for e in raw["events"] if e["name"] == "status"]
    decisions = [e for e in raw["events"] if e["name"] == "decision"]
    first = decisions[0] if decisions else None
    reason = next((c["reason"] for c in raw["classifier"] if c.get("applied")), "")
    bins = speech_bins(pcm)
    lo, hi = int((v0 - TALKING_WINDOW_SEC[0]) / BIN_SEC), int((v0 + TALKING_WINDOW_SEC[1]) / BIN_SEC) + 1
    # 첫 끼어들기: 끼어든 분량부터 다음 사용자 말 전까지 → 낭독 / 그 뒤
    pieces = [(p, d) for p, sp, d in raw["deltas"] if sp == "assistant" and raw["appends"][chunk - 1] <= p < starts[1]]
    text = "".join(d for _, d in pieces)
    cut, to_end = split_reading(text, CHUNKS[chunk - 1])
    reply = text[cut:].strip()
    decision_pos = first["pos"] if first else None
    voiced = _voiced(pcm)
    row: dict[str, Any] = {
        "talking_at_interrupt": bool(bins[max(0, lo) : hi].any()),
        "heard": "".join(
            d for p, sp, d in raw["deltas"] if sp == "user" and turns[0]["start"] <= p < starts[1]
        ).strip(),
        "decision": first["state"] if first else "none",
        "decision_by": first["by"] if first else "",
        "decision_reason": reason,
        "decision_at": round(decision_pos - v1, 2) if decision_pos is not None else None,
        "judge_latency": [c["latency"] for c in raw["classifier"]],
        "judge_calls": len(raw["classifier"]),
        "judge_stale": sum(bool(c.get("stale")) for c in raw["classifier"]),
        "judge_timeout": sum(bool(c.get("timeout")) for c in raw["classifier"]),
        "judge_states": [c["state"] for c in raw["classifier"]],
        "late_chunks": [
            i + 1 for i, p in enumerate(raw["appends"]) if p > v0 and (decision_pos is None or p < decision_pos)
        ],
        "read_to_end": to_end,
        "reply": reply,
        "filler": bool(FILLER.search(reply)),
        "next_read": [u["chunk"] for u in script_reuse(reply, list(CHUNKS)) if u["chunk"] > chunk],
        "midline_waits": sum(e["name"] == "midline_wait" for e in raw["events"]),
        "normal_gaps": [g for i in range(1, chunk) if (g := _gap_at(voiced, raw["appends"][i])) is not None],
        "on_gap": _gap_at(voiced, decision_pos) if first and first["state"] == "on" else None,
        "embellished": sum(
            _letters(_assistant(raw, raw["appends"][i], raw["appends"][i + 1])) > 1.3 * _letters(CHUNKS[i])
            for i in range(chunk - 1)
        ),
    }
    delegations = [e["pos"] for e in raw["events"] if e["name"] == "delegation"]
    for i, name in ((1, "follow"), (2, "resume")):
        if i >= len(turns):
            continue
        row[f"{name}_text"] = turns[i]["text"]
        row[f"{name}_delegations"] = [
            round(p - ends[i], 2) for p in delegations if starts[i] - 0.5 <= p < starts[i + 1]
        ]
        row[f"{name}_reply"] = _assistant(raw, starts[i], starts[i + 1])
    resumed = [e for e in status if e["status"] == "on"]
    if len(turns) > 2:
        if resumed:
            r = resumed[0]["pos"]
            row["resumed_at"] = round(r - ends[2], 2)
            row["before_resume"] = _assistant(
                raw, starts[2], r + 0.5
            )  # 다음 분량을 넣기 전까지 모델이 한 말 (전사 지연)
            nxt = next((i for i, p in enumerate(raw["appends"]) if p >= r - 0.05), None)
            after = _assistant(raw, r, float("inf"))
            row["resumed_chunk"] = nxt + 1 if nxt is not None else None
            row["resumed_read"] = nxt is not None and any(
                u["chunk"] == nxt + 1 for u in script_reuse(after, list(CHUNKS))
            )
        else:
            row["resumed_at"] = None
    return row


def format_row(r: dict[str, Any]) -> list[str]:
    if r["result"] != "done":
        return [f"  #{r['run']} {r['result']}"]
    at = f"{r['decision_at']:+.1f}s" if r["decision_at"] is not None else "-"
    lat = " ".join(f"{x:.1f}" for x in r["judge_latency"]) or "-"
    by = " (시간초과 대체)" if r["decision_by"] == "timeout" else ""
    states = "/".join(r["judge_states"]) or "-"
    head = f"  #{r['run']} 판정 {r['decision']}{by} @{at} (호출 {r['judge_calls']}: {states}, 응답 {lat}s"
    head += f", 낡음 {r['judge_stale']}" if r["judge_stale"] else ""
    head += f", 시간초과 {r['judge_timeout']}" if r["judge_timeout"] else ""
    head += ")"
    if r["late_chunks"]:
        head += f" | 판정 전 들어간 분량 {r['late_chunks']}"
    if r.get("interrupt", "mid") == "mid" and not r["talking_at_interrupt"]:
        head += " (끼어들 때 말소리 없음)"
    lines = [
        head,
        f"     들림: {r['heard'] or '-'} | 이유: {r['decision_reason']}",
        f"     대답: {r['reply'] or '-'}{' [빈말]' if r['filler'] else ''}",
    ]
    if r.get("next_read"):
        lines.append(f"     다음 분량을 읽음: {r['next_read']}")
    if "follow_text" in r:
        d = " ".join(f"{t:+.1f}" for t in r["follow_delegations"]) or "없음"
        lines.append(f"     이어 말함 → 위임 {d}: {r['follow_reply'] or '-'}")
    if "resume_text" in r:
        d = " ".join(f"{t:+.1f}" for t in r["resume_delegations"]) or "없음"
        if r.get("resumed_at") is not None:
            res = f"재개 {r['resumed_at']:+.1f}s, #{r['resumed_chunk']} {'읽음' if r['resumed_read'] else '안 읽음'}"
            lines.append(f'     "{r["resume_text"]}" → 위임 {d}, {res} | 넣기 전: {r["before_resume"] or "-"}')
        else:
            lines.append(f'     "{r["resume_text"]}" → 위임 {d}, 재개 없음: {r["resume_reply"] or "-"}')
    return lines


def _median(values: list[float], signed: bool = False) -> str:
    return (f"{float(np.median(values)):+.1f}s" if signed else f"{float(np.median(values)):.1f}s") if values else "-"


def report(tag: str) -> None:
    root = find_result_dir("radio_flow", tag)
    if root is None:
        raise SystemExit(f"결과 폴더 없음: {tag}")
    lines: list[str] = []
    allrows: list[tuple[str, dict[str, Any]]] = []
    for key, (kind, text, expect) in UTTERANCES.items():
        path = root / f"{key}.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text())["results"]
        valid = [r for r in rows if r["result"] == "done"]
        allrows += [(expect, r) for r in valid]
        n = len(valid)
        got = " ".join(f"{s} {sum(r['decision'] == s for r in valid)}" for s in ("on", "paused", "off", "none"))
        at = _median([r["decision_at"] for r in valid if r["decision_at"] is not None], True)
        lines.append(
            f"\n[{kind}] {text} ({key}) 기대 {expect} — 맞음 {sum(r['decision'] == expect for r in valid)}/{n} "
            f"(유효 {n}/{len(rows)}; {got}), 판정 시점 중앙 {at}"
        )
        for r in rows:
            lines.extend(format_row(r))
    rs = [r for _, r in allrows]
    lat = [x for r in rs for x in r["judge_latency"]]
    follow = [r for r in rs if "follow_text" in r]
    resume = [r for r in rs if "resume_text" in r]
    lines.append("\n전체")
    lines.append(f"  판정 맞음 {sum(r['decision'] == e for e, r in allrows)}/{len(rs)}")
    for e in ("on", "paused", "off"):
        sub = [r for x, r in allrows if x == e]
        if sub:
            got = " ".join(f"{s} {sum(r['decision'] == s for r in sub)}" for s in ("on", "paused", "off", "none"))
            lines.append(f"    기대 {e}: {got}")
    lines.append(
        f"  판정기 응답 중앙 {_median(lat)} (최대 {max(lat, default=0):.1f}s), "
        f"낡음 {sum(r['judge_stale'] for r in rs)}, 시간초과 {sum(r['judge_timeout'] for r in rs)}, "
        f"판정 전 들어간 분량 {sum(bool(r['late_chunks']) for r in rs)}/{len(rs)}, "
        f"그 분량을 대답 자리에서 읽음 {sum(bool(r.get('next_read')) for r in rs)}/{len(rs)}, "
        f"대사 중간 멈춤 대기 {sum(r.get('midline_waits', 0) for r in rs)}회"
    )
    lines.append(f"  이어 말한 뒤 위임(없어야 함) {sum(bool(r['follow_delegations']) for r in follow)}/{len(follow)}")
    for ask in RESUME_ASKS:
        sub = [r for r in resume if r["resume_text"] == ask]
        if sub:
            ok = [r for r in sub if r.get("resumed_at") is not None]
            lines.append(
                f'  "{ask}" 재개 {len(ok)}/{len(sub)} (중앙 {_median([r["resumed_at"] for r in ok], True)}), '
                f"넣은 분량 읽음 {sum(bool(r.get('resumed_read')) for r in ok)}/{len(ok)}"
            )
    normal = [g for r in rs for g in r.get("normal_gaps", [])]
    on_gaps = [r["on_gap"] for r in rs if r.get("on_gap") is not None]
    lines.append(
        f"  분량 사이 무음(rms 30 기준): 평소 중앙 {_median(normal)} (n={len(normal)}), "
        f"on 판정 뒤 중앙 {_median(on_gaps)} ({' '.join(f'{g:.1f}' for g in sorted(on_gaps))})"
    )
    lines.append(f"  끼어들기 전 덧붙인 분량 {sum(r['embellished'] for r in rs)}개 / {len(rs) * 3}")
    out = "\n".join(lines)
    print(out)
    (root / "report.txt").write_text(out + "\n")


def reanalyze(tag: str) -> None:
    root = find_result_dir("radio_flow", tag)
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
    out_dir = result_dir("radio_flow", args.tag)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-28s %(levelname)-5s %(message)s",
        handlers=[logging.FileHandler(out_dir / f"pipeline_{os.getpid()}.log")],
    )
    (out_dir / "instructions.txt").write_text(LIVE_RADIO_INSTRUCTIONS)
    config = {
        "chunk": args.chunk,
        "interrupt": args.interrupt,
        "gap_at": args.gap_at,
        "midline_wait": args.midline_wait,
        "at": args.at,
        "gap": args.gap,
        "watch": args.watch,
        "voice": LIVE_VOICE,
        "preamble": PREAMBLE,
        "radio_tools_line": RADIO_TOOLS_LINE,
        "radio_delegation": RADIO_DELEGATION,
        "notes": {"paused": PAUSE_NOTE, "on": RESUME_NOTE, "off": END_NOTE},
        "backend": {"model": BACKEND_MODEL, "reasoning": BACKEND_REASONING, "instructions": BACKEND_INSTRUCTIONS},
        "tools": TOOLS,
        "classifier": {"model": CLASSIFIER_MODEL, "instructions": CLASSIFIER_INSTRUCTIONS, "format": CLASSIFIER_FORMAT},
        "follow_up": FOLLOW_UP,
        "resume_asks": RESUME_ASKS,
        "chunks": list(CHUNKS),
        "utterances": UTTERANCES,
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1))
    for text in (FOLLOW_UP, *RESUME_ASKS, *(UTTERANCES[k][1] for k in args.utterances)):
        tts24(text)  # 캐시를 미리 채워 세션 중 합성 지연이 없게
    llm = OpenAILLM(model=CLASSIFIER_MODEL, temperature=0.0, reasoning_effort="none", max_tokens=120, tools=[])
    pool = ThreadPoolExecutor(max_workers=4)
    rows: dict[str, list[dict[str, Any]]] = {key: [] for key in args.utterances}
    for n in range(args.runs):
        for key in args.utterances:
            session = Session(key, n, args, llm, pool)
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
    pool.shutdown()
    print(f"results: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--utterances", nargs="*", choices=list(UTTERANCES), default=list(UTTERANCES))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=4, help="이 분량(1부터)을 말하는 도중에 끼어든다")
    parser.add_argument("--at", type=float, default=2.0, help="그 분량 첫 음성 이 초 뒤, 말소리가 날 때 끼어든다")
    parser.add_argument("--gap", type=float, default=1.0, help="첫 음성 뒤 이만큼 무음이면 다음 분량")
    parser.add_argument(
        "--interrupt", choices=("mid", "gap"), default="mid", help="말하는 도중 / 대사 뒤 틈에 끼어든다"
    )
    parser.add_argument("--gap-at", type=float, default=0.5, help="gap: 대사를 마치고 모델이 이만큼 조용할 때 끼어든다")
    parser.add_argument(
        "--midline-wait", type=float, default=1.5, help="대사 도중 멈췄으면 다음 분량 전에 더 기다릴 초"
    )
    parser.add_argument("--watch", type=float, default=20.0, help="진행(on)이면 사용자 목소리 끝 뒤 지켜볼 초")
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
