"""라디오 TTS 실험 — 대본은 앱이 TTS 로 틀고, 사용자가 끼어들면 TTS 를 멈추고 대화 모델이 대화한다.

대화 모델은 대본을 읽지 않는다. 앱이 문장마다 TTS(``TTS_MODEL``, 목소리는 라이브와 같은 ``LIVE_VOICE``)를 실시간
속도로 내보낸다. 라디오 동안 모델에게는 아무것도 알리지 않고, 끼어든 순간 한 번 thinking 으로 주제와 마지막 문장을
네가 한 말로 알린다(``CONTEXT_NOTE``) — 문장마다 알리면 모델이 그것을 받아 TTS 와 나란히 따라 읽었다.
라디오 동안 모델 출력은 내보내지 않는다(버린 말소리는 기록). 로봇 출력은 실시간 20 ms 프레임마다 앱이 고른다:
TTS / 모델 음성 / 무음.

끼어들기: 사용자 입력 전사 첫 조각이 오면 TTS 를 멈춘다(로컬 음성 감지 대신 — AEC 시험 전). ``--release now`` 면
그때부터 모델 대답을 그대로 내보내고, ``decision`` 이면 판정이 날 때까지 버퍼에 담는다. 판정기
(``exp_radio_flow.classify``, on / paused / off)는 사용자 전사가 ``USER_QUIET_SEC`` 멈추면 부르고, 모델 말이
``RECHECK_LETTERS`` 글자 넘게 늘면 다시 부른다. paused·off 는 오는 즉시 적용(버퍼면 앞 무음을 잘라 내보낸다). on 은
그 판정 뒤 새 사용자 말이 없을 때 — now 면 모델이 말하다 처음 쉬는 곳(``CUT_QUIET_SEC``)에서 모델 출력을 끊고,
decision 이면 새 모델 말(``RECHECK_LETTERS`` 이상)도 없을 때 버퍼를 버리고 — 쉼(``RESUME_PAUSE_SEC``) 뒤 끊긴 문장
처음부터 TTS 를 다시 튼다. ``CLASSIFY_TIMEOUT_SEC`` 안에 결과가
없으면 on.

멈추면 thinking(``PAUSE_NOTE``)과 백엔드 상태를 바꾸고, 사용자가 차례로 말한다 — 일반 대화 ``FOLLOW_UP``(위임이 없어야
한다), 재개 요청(``RESUME_ASKS`` 중 하나). 위임 → 백엔드 ``resume_radio`` → 모델의 말이 끝나면(무음
``RESUME_QUIET_SEC``, 길어야 ``RESUME_MAX_WAIT_SEC``) 끊긴 문장 처음부터 TTS 를 다시 튼다.

``--interrupt mid`` 는 ``--chunk`` 번째 대사를 틀기 시작하고 ``--at`` 초 뒤(문장 소리가 나는 중) 끼어들고, ``gap`` 은 그
대사 뒤 대사 사이 쉼 ``--gap-at`` 초에 끼어든다 — 다음 대사는 쉼(``--line-gap``)이 끝나면 그대로 나간다.

    uv run python scripts/gpt_live/exp_radio_tts.py --tag t1 --utterances content_why content_key
    uv run python scripts/gpt_live/exp_radio_tts.py --report t1

결과: var/experiments/radio_tts/<번호>_<태그>/ — <발화>.json, <발화>_<회차>.wav(왼쪽 사용자, 오른쪽 로봇 출력) +
.labels.txt, config.json, instructions.txt, pipeline_<pid>.log, report.txt(--report).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from exp_audio import write_conversation_wav
from exp_barge_in import user_clip, voiced_offsets
from exp_dirs import find_result_dir, result_dir
from exp_dj_commentary_chunks import CHUNKS, QUIET_RMS, TTS_CACHE, _rms, tts24
from exp_radio_delegation import FILLER
from exp_radio_flow import (
    BACKEND_INSTRUCTIONS,
    BACKEND_MODEL,
    BACKEND_REASONING,
    CLASSIFIER_INSTRUCTIONS,
    CLASSIFIER_MODEL,
    CLASSIFY_TIMEOUT_SEC,
    END_NOTE,
    FOLLOW_UP,
    LIVE_RADIO_INSTRUCTIONS,
    PAUSE_NOTE,
    RECHECK_LETTERS,
    RESUME_ASKS,
    TOOLS,
    USER_QUIET_SEC,
    UTTERANCES,
    classify,
    group_turns,
)

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

TTS_MODEL = "gpt-4o-mini-tts"  # 운영 인사 WAV 와 같은 모델·목소리
TTS_STYLE = "Speak Korean calmly and warmly at an unhurried pace, like a late-night radio host."
# 끼어든 순간 한 번 알리는 맥락. 멈춤을 알리면 맞장구에도 "라디오는 여기서 멈출게요", 앱이 틀었다고 하면 "재생되고
# 있었어요", 문장들을 다 주면 맞장구 뒤 그 문장들을 다시 읊었다 — 주제와 마지막 문장만 네가 한 말로 알린다.
TOPIC = "책상 서랍 속에 남아 있는 것들"  # assets/dj_segments/drawer.md 의 topic
CONTEXT_NOTE = "You are hosting a radio talk about «{topic}». Just before the user spoke, you were saying: «{last}»"
RESUME_NOTE = "Radio talk: resumed. The application plays the rest of your prepared talk in your voice again."

BYTES_PER_SEC = BRIDGE_SAMPLE_RATE * 2
FRAME_SEC = 0.02
FRAME_BYTES = int(BYTES_PER_SEC * FRAME_SEC)
START_DELAY_SEC = 1.5  # 세션 시작 뒤 라디오까지
RESUME_PAUSE_SEC = 0.4  # on 판정 뒤 TTS 를 다시 틀기 전 쉼
# now: on 판정 뒤 모델이 말하다 CUT_QUIET_SEC 쉬면(문장 끝) 모델 출력을 끊고 TTS — 맞장구 뒤 이어 말하기를 자른다.
# 쉼은 말소리 기준(rms CUT_RMS)으로 잰다 — rms 30 기준으로는 문장 사이 작은 소리 때문에 0.3 s 쉼이 잡히지 않았다.
CUT_QUIET_SEC = 0.3
CUT_RMS = 200
RESUME_QUIET_SEC = 0.6  # 재개 위임 뒤 모델 말이 이만큼 조용하면 TTS
RESUME_MAX_WAIT_SEC = 1.5  # 재개 위임 뒤 길어도 이만큼 기다리고 TTS (모델이 이야기를 지어내기 시작해도 끊는다)
HOLD_MAX_SEC = 15.0  # 끼어든 뒤 판정이 끝내 없으면 on
NEXT_TURN_QUIET_SEC = 1.5
NEXT_TURN_MAX_WAIT_SEC = 10.0
TURN_WATCH_SEC = 15.0
TRAILING_SEC = 3.0  # 대본을 다 튼 뒤 지켜볼 초
MAX_SEC = 220.0

logger = logging.getLogger("exp_radio_tts")


# ---------------------------------------------------------------------------
# 대본 TTS
# ---------------------------------------------------------------------------


def sentences(line: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.?!])\s+", line.strip()) if s]


def _trim(pcm: bytes) -> bytes:
    """앞뒤 무음 제거 (10 ms 단위 rms 100 이상인 구간 + 앞 50 ms, 뒤 100 ms)."""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    hop = BRIDGE_SAMPLE_RATE // 100
    n = len(x) // hop
    idx = np.flatnonzero(np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1)) >= 100)
    if not len(idx):
        return pcm
    a, b = max(0, int(idx[0]) - 5) * hop, min(n, int(idx[-1]) + 11) * hop
    return pcm[a * 2 : b * 2]


def tts_sentence(text: str) -> bytes:
    """라디오 문장 TTS — 24 kHz mono PCM16, 앞뒤 무음 제거. 파일 캐시."""
    TTS_CACHE.mkdir(exist_ok=True)
    key = hashlib.sha1(f"{TTS_MODEL}|{LIVE_VOICE}|{TTS_STYLE}|{text}".encode()).hexdigest()
    cache = TTS_CACHE / f"radio_{key}.pcm24k"
    if not cache.exists():
        import openai

        resp = openai.OpenAI().audio.speech.create(
            model=TTS_MODEL, voice=LIVE_VOICE, input=text, instructions=TTS_STYLE, response_format="pcm"
        )
        cache.write_bytes(_trim(resp.content))
    return cache.read_bytes()


def build_playlist(line_gap: float, sentence_gap: float) -> list[dict[str, Any]]:
    """대본 → 재생 목록. 항목은 문장(line, text, pcm) 또는 쉼(sec, after_line)."""
    items: list[dict[str, Any]] = []
    for i, line in enumerate(CHUNKS):
        for j, s in enumerate(sentences(line)):
            if j:
                items.append({"gap": sentence_gap, "after_line": None})
            items.append({"line": i, "text": s, "pcm": tts_sentence(s)})
        items.append({"gap": line_gap, "after_line": i})
    return items


# ---------------------------------------------------------------------------
# 세션
# ---------------------------------------------------------------------------


class Session:
    """세션 하나. 시간축은 로봇 출력(실시간 20 ms 프레임) 기준 초."""

    def __init__(
        self,
        key: str,
        run: int,
        args: argparse.Namespace,
        llm: OpenAILLM,
        pool: ThreadPoolExecutor,
        playlist: list[dict[str, Any]],
    ) -> None:
        self.key, self.args, self.llm, self.pool, self.playlist = key, args, llm, pool, playlist
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
        self.out = bytearray()  # 로봇 출력
        self.src: list[tuple[str, bool]] = []  # 프레임마다 (출처 tts/model/silence, 말소리 여부)
        self.model_q = bytearray()  # 받은 모델 음성 (talk·buffer 때만 쌓는다)
        self.user_pcm = bytearray()
        self.user_turns: list[dict[str, Any]] = []
        self.deltas: list[tuple[float, str, str]] = []
        self.events: list[dict[str, Any]] = []
        # 라디오 재생
        self.mode = "idle"  # idle / radio / buffer / talk / resuming / done
        self.item = 0  # 재생 목록 위치
        self.offset = 0  # 그 항목 안 바이트 위치
        self.bookmark: int | None = None  # 다시 틀 문장 항목
        self.line_start: dict[int, float] = {}  # 대사 번호 → 첫 문장을 틀기 시작한 위치
        self.said: list[str] = []  # 틀기 시작한 문장 (차례대로)
        self.gated_voice = 0.0  # 끼어들기 전 라디오 동안 버린 모델 말소리 길이 (혼자 말하려 함)
        self.gated_after = 0.0  # 끼어든 뒤 라디오로 돌아가 버린 모델 말소리 (맞장구 마저 등)
        # 끼어들기 판정
        self.status = "on"
        self.detect_pos: float | None = None
        self.pending = False  # 끼어든 뒤 판정을 기다리는 중
        self.decision_pos: float | None = None
        self.last_user_text: float | None = None
        self.judged_user = -1.0
        self.judged_reply = 0
        self.latest_on: dict[str, Any] | None = None
        self.inflight: list[tuple[Future[dict[str, Any]], dict[str, Any]]] = []
        self.calls: list[dict[str, Any]] = []
        # 대화
        self.plan: list[str] = []
        self.model_quiet = 0.0  # 출력한 모델 음성 뒤 무음 길이 (rms 30 기준)
        self.model_pause = 0.0  # 출력한 모델 말소리 뒤 쉼 (rms CUT_RMS 기준)
        self.last_model_voice: float | None = None
        self.resume_call_pos: float | None = None
        self.resume_pos: float | None = None
        self.pending_calls: list[LiveFunctionCall] = []
        self.outputs: dict[str, str] = {}
        self.result = "running"

    def pos(self) -> float:
        return len(self.out) / BYTES_PER_SEC

    def _event(self, name: str, **fields: Any) -> None:
        self.events.append({"pos": round(self.pos(), 2), "name": name, **fields})

    # -- 루프 ----------------------------------------------------------------

    def run(self) -> None:
        self.live.start()
        seg_text: dict[str, str] = {}
        t0 = time.monotonic()
        next_t = t0
        try:
            while self.result == "running":
                if self.user_pcm:
                    frame = bytes(self.user_pcm[:FRAME_BYTES]).ljust(FRAME_BYTES, b"\0")
                    del self.user_pcm[:FRAME_BYTES]
                else:
                    frame = bytes(FRAME_BYTES)
                self.live.send_audio(frame)
                while (ev := self.live.poll_event()) is not None:
                    if isinstance(ev, LiveAudio):
                        self._on_model_audio(ev.pcm)
                    elif isinstance(ev, LiveTranscript):
                        old = seg_text.get(ev.segment_id, "")
                        delta = ev.text[len(old) :] if ev.text.startswith(old) else ev.text
                        seg_text[ev.segment_id] = ev.text
                        if delta.strip():
                            self.deltas.append((self.pos(), ev.speaker, delta))
                            if ev.speaker == "user":
                                self._on_user_text()
                    elif isinstance(ev, LiveDelegationCreated):
                        self._event("delegation")
                    elif isinstance(ev, LiveFunctionCall):
                        self._on_call(ev)
                    elif isinstance(ev, LiveResponseDone):
                        self._on_response_done(ev)
                    elif isinstance(ev, LiveError):
                        self._event("error", message=ev.message)
                    elif isinstance(ev, LiveClosed):
                        self.result = "closed"
                self._poll_classifier()
                self._output_frame()
                if self.result == "running":
                    self._step(time.monotonic() - t0)
                if time.monotonic() - t0 > MAX_SEC and self.result == "running":
                    self.result = "timeout"
                next_t += FRAME_SEC
                time.sleep(max(0.0, next_t - time.monotonic()))
        finally:
            self.live.close(graceful=True)

    def _on_model_audio(self, pcm: bytes) -> None:
        if self.mode in ("talk", "buffer", "resuming"):
            self.model_q.extend(pcm)
        elif any(pcm) and _rms(pcm) >= QUIET_RMS:  # 라디오 중 모델 말소리 — 내보내지 않는다
            if self.user_turns:
                self.gated_after += len(pcm) / BYTES_PER_SEC
            else:
                self.gated_voice += len(pcm) / BYTES_PER_SEC

    def _output_frame(self) -> None:
        """이번 20 ms 프레임에 로봇이 낼 소리를 고른다."""
        if self.mode == "radio":
            frame, src = self._radio_frame(), "tts"
        elif self.mode in ("talk", "resuming") and self.model_q:
            frame = bytes(self.model_q[:FRAME_BYTES]).ljust(FRAME_BYTES, b"\0")
            del self.model_q[:FRAME_BYTES]
            src = "model"
        else:
            frame, src = bytes(FRAME_BYTES), "silence"
        voiced = any(frame) and _rms(frame) >= QUIET_RMS
        if src == "model" or self.mode in ("talk", "resuming"):
            if voiced:
                self.model_quiet, self.last_model_voice = 0.0, self.pos() + FRAME_SEC
            else:
                self.model_quiet += FRAME_SEC
            self.model_pause = 0.0 if voiced and _rms(frame) >= CUT_RMS else self.model_pause + FRAME_SEC
        self.out.extend(frame)
        self.src.append((src, bool(voiced)))

    def _radio_frame(self) -> bytes:
        while self.item < len(self.playlist):
            it = self.playlist[self.item]
            if "pcm" in it:
                if self.offset == 0:
                    self.line_start.setdefault(it["line"], self.pos())
                    self.said.append(it["text"])
                    self._event("say", line=it["line"] + 1, text=it["text"])
                if self.offset < len(it["pcm"]):
                    frame = bytes(it["pcm"][self.offset : self.offset + FRAME_BYTES]).ljust(FRAME_BYTES, b"\0")
                    self.offset += FRAME_BYTES
                    return frame
            elif self.offset < int(it["gap"] * BYTES_PER_SEC):
                self.offset += FRAME_BYTES
                return bytes(FRAME_BYTES)
            self.item, self.offset = self.item + 1, 0
        self.mode = "done"
        self._event("radio_end")
        return bytes(FRAME_BYTES)

    def _start_radio(self, from_item: int) -> None:
        self.item, self.offset, self.mode = from_item, 0, "radio"
        self.model_q.clear()

    # -- 끼어들기 ------------------------------------------------------------

    def _current_sentence(self) -> int:
        """멈출 때 다시 틀 문장 — 문장 중이면 그 문장, 쉼이면 다음 문장."""
        i = self.item
        while i < len(self.playlist) and "pcm" not in self.playlist[i]:
            i += 1
        return i

    def _on_user_text(self) -> None:
        self.last_user_text = self.pos()
        if self.mode == "radio" and self.user_turns and self.status == "on" and self.detect_pos is None:
            self.detect_pos = self.pos()
            self.bookmark = self._current_sentence()
            self.pending = True
            self.mode = "buffer" if self.args.release == "decision" else "talk"
            self.model_q.clear()
            self.live.append_thinking(CONTEXT_NOTE.format(topic=TOPIC, last=self.said[-1] if self.said else "-"))
            self._event("detect", bookmark=self.bookmark)

    def _in_sentence(self) -> bool:
        it = self.playlist[min(self.item, len(self.playlist) - 1)]
        return "pcm" in it and 0 < self.offset < len(it["pcm"])

    def _reply_letters(self) -> int:
        since = self.detect_pos or self.pos()
        text = "".join(d for p, sp, d in self.deltas if sp == "assistant" and p >= since)
        return len(re.sub(r"[\W_]+", "", text))

    def _stale(self, record: dict[str, Any]) -> bool:
        new_user = self.last_user_text is not None and self.last_user_text > record["upto_user"]
        return new_user or self._reply_letters() - record["upto_reply"] >= RECHECK_LETTERS

    def _waiting(self) -> bool:
        return any(not r.get("timeout") for _, r in self.inflight)

    def _maybe_classify(self, p: float) -> None:
        if self.last_user_text is None or self._waiting():
            return
        new_user = self.last_user_text > self.judged_user
        if new_user and p - self.last_user_text < USER_QUIET_SEC:
            return
        reply = self._reply_letters()
        if not new_user and reply - self.judged_reply < RECHECK_LETTERS:
            return
        line = CHUNKS[self.playlist[min(self.bookmark or 0, len(self.playlist) - 1)].get("line", 0)]
        exchange = group_turns(self.deltas, self.user_turns[0]["start"])
        record = {"pos": round(p, 2), "upto_user": self.last_user_text, "upto_reply": reply, "line": line}
        record["by"], record["exchange"] = ("user" if new_user else "reply"), exchange
        self.judged_user, self.judged_reply = self.last_user_text, reply
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
                record["stale"] = self._stale(record)
                if record["state"] != "on":
                    record["applied"] = self._decide(record["state"], "classifier")
                elif not record.get("timeout"):
                    self.latest_on = record
                self.calls.append(record)
            elif p - record["pos"] >= CLASSIFY_TIMEOUT_SEC and not record.get("timeout"):
                record["timeout"] = True
                self.latest_on = record

    def _decide(self, state: str, by: str) -> bool:
        """paused·off — 판정을 기다리던 중이면 적용한다 (버퍼면 앞 무음을 잘라 모델 대답을 내보낸다)."""
        if not self.pending:
            return False
        if self.mode == "buffer":
            self._trim_queue()
        self.pending = False
        self.mode, self.status, self.decision_pos = "talk", state, self.pos()
        self.plan = [FOLLOW_UP] + ([self.resume_ask] if state == "paused" else [])
        self.live.append_thinking(PAUSE_NOTE if state == "paused" else END_NOTE)
        self.live.update_backend(BACKEND_INSTRUCTIONS.format(status=state))
        self._event("decision", state=state, by=by)
        return True

    def _trim_queue(self) -> None:
        """모델 출력 버퍼 앞 무음 제거 — 대답이 시작된 곳부터 내보낸다."""
        q = self.model_q
        for i in range(0, len(q) - FRAME_BYTES + 1, FRAME_BYTES):
            frame = bytes(q[i : i + FRAME_BYTES])
            if any(frame) and _rms(frame) >= QUIET_RMS:
                del q[:i]
                return
        q.clear()

    def _hold_step(self, p: float) -> None:
        self._maybe_classify(p)
        on = self.latest_on
        if on is None:
            ok = False
        elif self.args.release == "decision":
            ok = not self._waiting() and not self._stale(on)
        else:  # 모델 말이 늘거나 다시 판정 중이어도 기다리지 않는다 — 새 사용자 말만 없으면 쉬는 곳에서 끊는다
            new_user = self.last_user_text is not None and self.last_user_text > on["upto_user"]
            ok = not new_user and self.model_pause >= CUT_QUIET_SEC
        if ok:
            assert on is not None
            on["applied"] = True
            self._resume_on("timeout" if on.get("timeout") else "classifier")
        elif self.detect_pos is not None and p - self.detect_pos >= HOLD_MAX_SEC and not self._waiting():
            self._resume_on("max")

    def _resume_on(self, by: str) -> None:
        """on — 모델 출력(버퍼 또는 남은 이어 말하기)을 버리고 쉼을 둔 뒤 끊긴 문장부터 TTS."""
        self.pending = False
        self.decision_pos = self.pos()
        self._event("decision", state="on", by=by)
        self.playlist.insert(self.bookmark or 0, {"gap": RESUME_PAUSE_SEC, "after_line": None})
        self._start_radio(self.bookmark or 0)

    # -- 백엔드 --------------------------------------------------------------

    def _on_call(self, call: LiveFunctionCall) -> None:
        self.pending_calls.append(call)
        self._event("call", tool=call.name)
        if call.name != "resume_radio":
            self.outputs[call.call_id] = json.dumps({"error": f"unknown function {call.name}"})
        elif self.status != "paused" or self.mode != "talk":
            self.outputs[call.call_id] = json.dumps({"error": "the radio talk is not paused"})
        else:
            self.outputs[call.call_id] = json.dumps({"status": "on"})
            self.status, self.mode, self.resume_call_pos = "on", "resuming", self.pos()
            self.live.append_thinking(RESUME_NOTE)
            self.live.update_backend(BACKEND_INSTRUCTIONS.format(status="on"))
            self._event("status", status="on")

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

    def _say(self, i: int) -> None:
        text, clip, _ = self.turns[i]
        self.user_pcm.extend(clip)
        self.user_turns.append({"text": text, "start": round(self.pos(), 3)})

    def _turn_end(self, i: int) -> float:
        return self.user_turns[i]["start"] + self.turns[i][2][1]

    def _step(self, elapsed: float) -> None:
        p = self.pos()
        if self.mode == "idle":
            if elapsed >= START_DELAY_SEC:
                self._start_radio(0)
            return
        if not self.user_turns:  # 끼어들기 전
            it = self.playlist[min(self.item, len(self.playlist) - 1)]
            target = self.args.chunk - 1
            if self.args.interrupt == "mid":
                started = self.line_start.get(target)
                if "pcm" in it and it["line"] == target and started is not None and p - started >= self.args.at:
                    self._say(0)
            elif it.get("after_line") == target and self.offset >= int(self.args.gap_at * BYTES_PER_SEC):
                self._say(0)
            if self.mode == "done":
                self.result = "missed"
            return
        last = len(self.user_turns) - 1
        since_last = p - self._turn_end(last)
        if self.mode in ("radio", "done") and self.decision_pos is None and self.detect_pos is None:
            if since_last >= self.args.watch:
                self.result = "undetected"  # 사용자 전사가 끝내 오지 않았다
            return
        if self.pending:
            self._hold_step(p)
            return
        if self.mode == "resuming":
            waited = p - (self.resume_call_pos or p)
            if (self.model_quiet >= RESUME_QUIET_SEC and waited >= 0.2) or waited >= RESUME_MAX_WAIT_SEC:
                self.resume_pos = p
                self._event("resume_tts", item=self.bookmark)
                self._start_radio(self.bookmark or 0)
            return
        if self.mode == "talk" and not self.user_pcm:
            nxt = len(self.user_turns)
            if nxt - 1 < len(self.plan):
                spoke = self.last_model_voice is not None and self.last_model_voice > self._turn_end(last)
                if (spoke and self.model_quiet >= NEXT_TURN_QUIET_SEC) or since_last >= NEXT_TURN_MAX_WAIT_SEC:
                    text = self.plan[nxt - 1]
                    self.turns.append((text, user_clip(text, trim=True), voiced_offsets(text, trim=True)))
                    self._say(nxt)
                    return
            if since_last >= TURN_WATCH_SEC:
                self.result = "done"
            return
        # 라디오가 다시 나가는 중 (on 판정 뒤 또는 재개 뒤)
        if self.mode == "done" or since_last >= max(self.args.watch, TURN_WATCH_SEC) + 10:
            self.result = "done"

    # -- 기록 ----------------------------------------------------------------

    def raw(self) -> dict[str, Any]:
        return {
            "utterance": self.key,
            "result": self.result,
            "resume_ask": self.resume_ask,
            "interrupt": self.args.interrupt,
            "events": self.events,
            "user_turns": [{**t, "voice": list(self.turns[i][2])} for i, t in enumerate(self.user_turns)],
            "classifier": self.calls,
            "detect_pos": self.detect_pos,
            "decision_pos": self.decision_pos,
            "resume_call_pos": self.resume_call_pos,
            "resume_pos": self.resume_pos,
            "gated_voice": round(self.gated_voice, 2),
            "gated_after": round(self.gated_after, 2),
            "deltas": [[round(p, 2), sp, d] for p, sp, d in self.deltas],
            "src": "".join({"tts": "t", "model": "m", "silence": "s"}[s] + ("1" if v else "0") for s, v in self.src),
        }

    def save(self, path: Path) -> None:
        clips = [(t["start"], self.turns[i][1]) for i, t in enumerate(self.user_turns)]
        write_conversation_wav(path, bytes(self.out), clips, BRIDGE_SAMPLE_RATE)
        labels = []
        for i, t in enumerate(self.user_turns):
            a, b = self.turns[i][2]
            labels.append((t["start"] + a, t["start"] + b, f"USER: {t['text']}"))
        for e in self.events:
            if e["name"] == "say":
                labels.append((e["pos"], e["pos"], f"TTS: {e['text']}"))
            elif e["name"] != "response_done":
                extra = " ".join(str(v) for k, v in e.items() if k not in ("pos", "name"))
                labels.append((e["pos"], e["pos"], f"{e['name']} {extra}".strip()))
        for c in self.calls:
            labels.append((c["pos"] + c["latency"], c["pos"] + c["latency"], f"judge {c['state']} ({c['reason']})"))
        # 모델 음성으로 내보낸 구간과 그때 도착한 모델 전사
        segs, cur = [], None
        for k, (s, v) in enumerate(self.src):
            on = s == "model" and v
            if on and cur is None:
                cur = k
            if not on and cur is not None:
                if k - cur > 2:
                    segs.append((cur * FRAME_SEC, k * FRAME_SEC))
                cur = None
        for a, b in segs:
            text = "".join(d for p, sp, d in self.deltas if sp == "assistant" and a - 2.0 <= p < b)
            labels.append((a, b, f"RAY: {text.strip()[-80:]}"))
        labels.sort()
        path.with_suffix(".labels.txt").write_text("".join(f"{a:.3f}\t{b:.3f}\t{t}\n" for a, b, t in labels))


# ---------------------------------------------------------------------------
# 분석
# ---------------------------------------------------------------------------


def _frames(raw: dict[str, Any]) -> list[tuple[str, bool]]:
    s = raw["src"]
    return [({"t": "tts", "m": "model", "s": "silence"}[s[i]], s[i + 1] == "1") for i in range(0, len(s), 2)]


def _first(frames: list[tuple[str, bool]], src: str, after: float) -> float | None:
    for k in range(int(after / FRAME_SEC), len(frames)):
        if frames[k] == (src, True):
            return k * FRAME_SEC
    return None


def analyze(raw: dict[str, Any]) -> dict[str, Any]:
    if raw["result"] != "done":
        return {}
    frames = _frames(raw)
    turns = raw["user_turns"]
    v0, v1 = turns[0]["start"] + turns[0]["voice"][0], turns[0]["start"] + turns[0]["voice"][1]
    starts = [t["start"] + t["voice"][0] for t in turns] + [float("inf")]
    ends = [t["start"] + t["voice"][1] for t in turns]
    a, b = int(v0 / FRAME_SEC), int(v1 / FRAME_SEC)
    decisions = [e for e in raw["events"] if e["name"] == "decision"]
    first = decisions[0] if decisions else None
    detect = raw["detect_pos"]
    reply_text = "".join(d for p, sp, d in raw["deltas"] if sp == "assistant" and (detect or 0) <= p < starts[1])
    row: dict[str, Any] = {
        "heard": "".join(d for p, sp, d in raw["deltas"] if sp == "user" and turns[0]["start"] <= p < starts[1]),
        "detect_after": round(detect - v0, 2) if detect is not None else None,  # 사용자 목소리 시작 → TTS 멈춤
        "tts_overlap": round(sum(f == ("tts", True) for f in frames[a:b]) * FRAME_SEC, 2),
        "decision": first["state"] if first else "none",
        "decision_by": first["by"] if first else "",
        "decision_at": round(first["pos"] - v1, 2) if first else None,
        "decision_reason": next((c["reason"] for c in raw["classifier"] if c.get("applied")), ""),
        "judge_states": [c["state"] for c in raw["classifier"]],
        "judge_latency": [c["latency"] for c in raw["classifier"]],
        "reply": reply_text.strip(),
        "filler": bool(FILLER.search(reply_text)),
        "gated_voice": raw["gated_voice"],
    }
    if first and first["state"] != "on" and detect is not None:
        heard = _first(frames, "model", detect)
        row["reply_heard_at"] = round(heard - v1, 2) if heard is not None else None  # 사용자 목소리 끝 → 대답 소리
    if first and first["state"] == "on":
        cut_at = first["pos"]  # on 이면 여기서 모델 출력을 끊었다 — 그 전까지 도착한 전사가 들린 말 (어림)
        row["reply_heard"] = "".join(
            d for p, sp, d in raw["deltas"] if sp == "assistant" and (detect or 0) <= p < cut_at
        ).strip()
        back = _first(frames, "tts", first["pos"])
        row["tts_back_at"] = round(back - v1, 2) if back is not None else None  # 사용자 목소리 끝 → TTS 다시
    delegations = [e["pos"] for e in raw["events"] if e["name"] == "delegation"]
    for i, name in ((1, "follow"), (2, "resume")):
        if i < len(turns):
            row[f"{name}_text"] = turns[i]["text"]
            row[f"{name}_delegations"] = [
                round(p - ends[i], 2) for p in delegations if starts[i] - 0.5 <= p < starts[i + 1]
            ]
            row[f"{name}_reply"] = "".join(
                d for p, sp, d in raw["deltas"] if sp == "assistant" and starts[i] <= p < starts[i + 1]
            ).strip()
    if len(turns) > 2:
        row["resumed_at"] = round(raw["resume_pos"] - ends[2], 2) if raw["resume_pos"] is not None else None
        if raw["resume_pos"] is not None:
            row["before_resume"] = "".join(
                d for p, sp, d in raw["deltas"] if sp == "assistant" and starts[2] <= p < raw["resume_pos"] + 0.3
            ).strip()
    return row


def format_row(r: dict[str, Any]) -> list[str]:
    if r["result"] != "done":
        return [f"  #{r['run']} {r['result']}"]
    states = "/".join(r["judge_states"]) or "-"
    lat = " ".join(f"{x:.1f}" for x in r["judge_latency"]) or "-"
    at = f"{r['decision_at']:+.1f}s" if r["decision_at"] is not None else "-"
    head = f"  #{r['run']} 판정 {r['decision']} @{at} (호출: {states}, 응답 {lat}s)"
    head += f" | TTS 멈춤 +{r['detect_after']}s, 겹침 {r['tts_overlap']}s"
    if r.get("reply_heard_at") is not None:
        head += f" | 대답 들림 {r['reply_heard_at']:+.1f}s"
    if r.get("tts_back_at") is not None:
        head += f" | TTS 다시 {r['tts_back_at']:+.1f}s"
    if r["gated_voice"]:
        head += f" | 라디오 중 버린 모델 말 {r['gated_voice']}s"
    lines = [
        head,
        f"     들림: {r['heard'].strip() or '-'} | 이유: {r['decision_reason']}",
        f"     모델 말: {r['reply'] or '-'}{' [빈말]' if r['filler'] else ''}",
    ]
    if r.get("reply_heard") is not None:
        lines.append(f"     그중 들린 말(끊기 전): {r['reply_heard'] or '-'}")
    if "follow_text" in r:
        d = " ".join(f"{t:+.1f}" for t in r["follow_delegations"]) or "없음"
        lines.append(f"     이어 말함 → 위임 {d}: {r['follow_reply'] or '-'}")
    if "resume_text" in r:
        d = " ".join(f"{t:+.1f}" for t in r["resume_delegations"]) or "없음"
        if r.get("resumed_at") is not None:
            lines.append(
                f'     "{r["resume_text"]}" → 위임 {d}, TTS 재개 {r["resumed_at"]:+.1f}s | 그 전 모델 말: '
                f"{r.get('before_resume') or '-'}"
            )
        else:
            lines.append(f'     "{r["resume_text"]}" → 위임 {d}, 재개 없음: {r["resume_reply"] or "-"}')
    return lines


def _median(values: list[float], signed: bool = False) -> str:
    if not values:
        return "-"
    v = float(np.median(values))
    return f"{v:+.1f}s" if signed else f"{v:.1f}s"


def report(tag: str) -> None:
    root = find_result_dir("radio_tts", tag)
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
        lines.append(
            f"\n[{kind}] {text} ({key}) 기대 {expect} — 맞음 {sum(r['decision'] == expect for r in valid)}/"
            f"{len(valid)} (유효 {len(valid)}/{len(rows)})"
        )
        for r in rows:
            lines.extend(format_row(r))
    rs = [r for _, r in allrows]
    lines.append("\n전체")
    lines.append(f"  판정 맞음 {sum(r['decision'] == e for e, r in allrows)}/{len(rs)}")
    for e in ("on", "paused", "off"):
        sub = [r for x, r in allrows if x == e]
        if sub:
            got = " ".join(f"{s} {sum(r['decision'] == s for r in sub)}" for s in ("on", "paused", "off", "none"))
            lines.append(f"    기대 {e}: {got}")
    lines.append(
        f"  TTS 멈춤(사용자 목소리 시작 기준) 중앙 {_median([r['detect_after'] for r in rs if r['detect_after']])}, "
        f"사용자 말과 TTS 겹침 중앙 {_median([r['tts_overlap'] for r in rs])}"
    )
    lines.append(
        f"  paused·off 대답 들림(사용자 목소리 끝 기준) 중앙 "
        f"{_median([r['reply_heard_at'] for r in rs if r.get('reply_heard_at') is not None], True)}, "
        f"on 판정 뒤 TTS 다시 중앙 {_median([r['tts_back_at'] for r in rs if r.get('tts_back_at') is not None], True)}"
    )
    lines.append(
        f"  라디오 중 모델이 말하려 함 {sum(bool(r['gated_voice']) for r in rs)}/{len(rs)}, "
        f"빈말 {sum(r['filler'] for r in rs)}/{len(rs)}"
    )
    follow = [r for r in rs if "follow_text" in r]
    lines.append(f"  이어 말한 뒤 위임(없어야 함) {sum(bool(r['follow_delegations']) for r in follow)}/{len(follow)}")
    for ask in RESUME_ASKS:
        sub = [r for r in rs if r.get("resume_text") == ask]
        if sub:
            ok = [r for r in sub if r.get("resumed_at") is not None]
            lines.append(f'  "{ask}" 재개 {len(ok)}/{len(sub)} (중앙 {_median([r["resumed_at"] for r in ok], True)})')
    out = "\n".join(lines)
    print(out)
    (root / "report.txt").write_text(out + "\n")


def run(args: argparse.Namespace) -> None:
    out_dir = result_dir("radio_tts", args.tag)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-28s %(levelname)-5s %(message)s",
        handlers=[logging.FileHandler(out_dir / f"pipeline_{os.getpid()}.log")],
    )
    (out_dir / "instructions.txt").write_text(LIVE_RADIO_INSTRUCTIONS)
    playlist = build_playlist(args.line_gap, args.sentence_gap)
    config = {
        "interrupt": args.interrupt,
        "release": args.release,
        "chunk": args.chunk,
        "at": args.at,
        "gap_at": args.gap_at,
        "line_gap": args.line_gap,
        "sentence_gap": args.sentence_gap,
        "watch": args.watch,
        "tts": {"model": TTS_MODEL, "voice": LIVE_VOICE, "style": TTS_STYLE},
        "notes": {
            "context": CONTEXT_NOTE,
            "topic": TOPIC,
            "paused": PAUSE_NOTE,
            "resume": RESUME_NOTE,
            "off": END_NOTE,
        },
        "backend": {"model": BACKEND_MODEL, "reasoning": BACKEND_REASONING, "instructions": BACKEND_INSTRUCTIONS},
        "classifier": {"model": CLASSIFIER_MODEL, "instructions": CLASSIFIER_INSTRUCTIONS},
        "follow_up": FOLLOW_UP,
        "resume_asks": RESUME_ASKS,
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
            session = Session(key, n, args, llm, pool, [dict(it) for it in playlist])
            try:
                session.run()
            except RuntimeError as exc:
                row = {"run": n, "utterance": key, "result": f"failed: {exc}", "events": []}
            else:
                raw = session.raw()
                row = {"run": n, **raw, **analyze(raw)}
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
    parser.add_argument("--release", choices=("now", "decision"), default="now", help="모델 대답을 바로 / 판정 뒤")
    parser.add_argument("--chunk", type=int, default=4, help="이 대사(1부터)에서 끼어든다")
    parser.add_argument("--at", type=float, default=2.0, help="mid: 그 대사를 틀기 시작하고 이 초 뒤")
    parser.add_argument("--gap-at", type=float, default=0.3, help="gap: 그 대사 뒤 쉼에서 이 초 뒤")
    parser.add_argument("--line-gap", type=float, default=0.8, help="대사 사이 쉼")
    parser.add_argument("--sentence-gap", type=float, default=0.35, help="한 대사 안 문장 사이 쉼")
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
