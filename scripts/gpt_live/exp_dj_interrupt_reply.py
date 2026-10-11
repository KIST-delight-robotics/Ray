"""DJ 분량 주입 중 끼어들기 대답 — 레이가 분량을 말하는 도중 여러 종류의 사용자 말로 끼어들고, 무엇이라고 답하는지 본다.

흐름은 분량 주입 설계 그대로다: 라이브 지시문 정리본(``live_instructions_simple.txt``, 위임 백엔드 없음), 첫 분량 전에
``read`` 안내를 instructions.append 로 한 번, 대사(``CHUNKS``)를 하나씩 commentary.append 로 넣고 모델 첫 음성 뒤 받은
무음 조각이 ``--gap`` 초 이어지면 다음 분량. ``--chunk`` 번째 분량의 첫 음성 ``--at`` 초 뒤, 모델이 말소리를 내고
있을 때 사용자 발화(TTS, 앞뒤 무음 제거)를 넣는다. 앱은 사용자 말소리를 마이크 레벨로 안다고 보고 그때부터 분량을
넣지 않는다(복귀 없음). 사용자 목소리가 끝난 뒤 ``--watch`` 초 지켜보고 끝낸다.

무효: 끼어들 시점 전에 그 분량이 끝남(missed), 주입 뒤 음성 없음(no_voice).

기록(회차마다): 들린 말, 끼어든 분량의 낭독(끼어든 곳 ‖ 표시, 대사 끝까지 읽었는지)과 그 뒤 대답 — 경계는 전사를 대사와
맞춰 찾는다(``split_reading``). 대답 첫 전사 도착(사용자 목소리 끝 기준, 전사 지연 포함), 사용자 목소리 동안·뒤 모델
말소리 길이, 위임 생성. 대답에서 대본 문구 재사용(넣은 대사와 공백·부호 빼고 ``REUSE_MIN_LETTERS`` 글자 이상 겹침 —
대답에 섞었거나 스스로 이어 읽음), 낭독 어휘("읽"), 물음표. 진행 기록(전사 조각·주입 위치)도 남겨 ``--reanalyze`` 로
다시 계산할 수 있다.

    # 종류마다 프로세스 하나, 병렬로 (같은 태그 폴더에 발화별 파일로 모인다)
    uv run python scripts/gpt_live/exp_dj_interrupt_reply.py --tag r1 --utterances content_why content_key
    uv run python scripts/gpt_live/exp_dj_interrupt_reply.py --report r1   # 발화별로 대답을 모아 보기 (report.txt)
    uv run python scripts/gpt_live/exp_dj_interrupt_reply.py --reanalyze r1   # 지표를 바꿨을 때

결과: var/experiments/dj_interrupt_reply/<번호>_<태그>/ — <발화>.json, <발화>_<회차>.wav(왼쪽 사용자, 오른쪽 모델) +
.labels.txt(Audacity 레이블), config.json, instructions.txt, pipeline_<pid>.log.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from exp_audio import read_model_channel, write_conversation_wav
from exp_barge_in import BIN_SEC, audio_metrics, speech_bins, speech_segments, user_clip, voiced_offsets
from exp_dirs import find_result_dir, result_dir
from exp_dj_commentary_chunks import CHUNKS, LIVE_INSTRUCTIONS, PREAMBLES, QUIET_RMS, _rms, tts24

from voice_pipeline.adapters.gpt_live import (
    GPTLiveSession,
    LiveAudio,
    LiveClosed,
    LiveDelegationCreated,
    LiveError,
    LiveSessionConfig,
    LiveTranscript,
)
from voice_pipeline.engines.gpt_live.instructions import LIVE_VOICE
from voice_pipeline.settings import BRIDGE_SAMPLE_RATE

# 발화 키 → (종류, 문장). 기본으로 끼어드는 4번 분량("가끔은 쓰다 만 편지도 나와요. …") 전에 3번 분량에서 볼펜·열쇠·
# 영수증이 나온다. 운영에서 위임으로 가는 발화(멈춤·곡 재생·날씨)는 넣지 않는다 — 이 세션엔 백엔드가 없다.
UTTERANCES = {
    "content_why": ("이야기 내용 질문", "왜 꼭 그런 게 남아 있을까?"),
    "content_key": ("이야기 내용 질문", "어디 건지 모를 열쇠는 왜 안 버리는 거야?"),
    "ray_drawer": ("레이에게 묻기", "너는 서랍에 뭐 넣어 둬?"),
    "ray_tidy": ("레이에게 묻기", "너도 정리 못 하는 편이야?"),
    "own_letter": ("자기 이야기", "나도 서랍에 옛날 편지 있는데."),
    "own_mess": ("자기 이야기", "우리 집 서랍은 완전 엉망이야."),
    "react_really": ("짧은 반응", "아, 진짜?"),
    "react_agree": ("짧은 반응", "맞아 맞아."),
    "song": ("진행·노래 질문", "오늘 무슨 노래 틀어 줄 거야?"),
    "repeat": ("진행·노래 질문", "방금 뭐라고 했어?"),
    "bored": ("시큰둥", "좀 지루하다."),
    "what": ("시큰둥", "이거 무슨 얘기야?"),
    "hungry": ("주제 밖 잡담", "배고프다."),
    "mood": ("주제 밖 잡담", "너 오늘 기분 어때?"),
}
PREAMBLE = PREAMBLES["read"]

BYTES_PER_SEC = BRIDGE_SAMPLE_RATE * 2
FRAME_SEC = 0.02
VOICED_SEC = 0.2  # 끼어들 때 받은 무음 조각이 이보다 짧아야 한다 — 모델이 말소리를 내는 중
START_DELAY_SEC = 1.0
PREAMBLE_SETTLE_SEC = 3.0  # 안내 뒤 첫 분량까지 최소 대기
PREAMBLE_QUIET_SEC = 1.5  # 안내에 대꾸했으면 이만큼 조용해진 뒤 첫 분량
NO_VOICE_SEC = 8.0
TRANSCRIPT_LAG_SEC = 1.0  # 전사는 소리보다 늦게 온다 — 소리 끝 뒤 이만큼까지 도착한 조각은 그 소리의 것으로 본다
REUSE_MIN_LETTERS = 8
MAX_SKIP_LETTERS = 4  # 낭독 전사와 대사의 일치 구간 사이가 이보다 벌어지면 낭독이 끝난 것
MIN_ANCHOR_LETTERS = 3
MIN_BLOCK_LETTERS = 2  # 한 글자 일치는 대답 속 우연("그럴 때 있죠" ↔ "그대로 남아 있는 거죠")이 사슬로 이어진다
TALKING_WINDOW_SEC = (0.3, 0.1)  # 사용자 목소리 시작 전·후 이 범위에 모델 말소리가 있으면 "말하는 중에 끼어듦"
MAX_SEC = 150.0

logger = logging.getLogger("exp_dj_interrupt_reply")


def _letter_str(text: str) -> str:
    """공백·문장부호를 뺀 글자열 — 전사 표기 차이를 덜 타게."""
    return re.sub(r"[\W_]+", "", text)


def split_reading(text: str, chunk: str) -> tuple[int, bool]:
    """분량부터의 모델 전사 ``text`` 에서 대사 낭독이 끝나는 문자 위치와, 대사 끝까지 읽었는지.

    대사와 일치하는 구간을 앞에서부터 따라가다 전사나 대사 쪽으로 ``MAX_SKIP_LETTERS`` 넘게 벌어지거나 일치가
    ``MIN_BLOCK_LETTERS`` 보다 짧으면 거기서 낭독이 끝난 것으로 본다 — ``read`` 는 하던 문장을 마저 읽고 쉼 없이
    대답으로 넘어가기도 해서 말소리로는 경계가 안 보인다. 첫 구간은 ``MIN_ANCHOR_LETTERS`` 이상이어야 한다(앞 분량
    전사 꼬리의 우연한 한두 글자 일치를 건너뛰게).
    """
    idx = [i for i, ch in enumerate(text) if ch.isalnum()]
    t = "".join(text[i] for i in idx)
    c = _letter_str(chunk)
    end_t = end_c = 0
    for b in difflib.SequenceMatcher(None, t, c, autojunk=False).get_matching_blocks():
        if b.size == 0:
            break
        if not end_t:
            if b.size >= MIN_ANCHOR_LETTERS:
                end_t, end_c = b.a + b.size, b.b + b.size
            continue
        if b.size < MIN_BLOCK_LETTERS or b.a - end_t > MAX_SKIP_LETTERS or b.b - end_c > MAX_SKIP_LETTERS:
            break
        end_t, end_c = b.a + b.size, b.b + b.size
    while end_t and end_t < len(t) and end_c < len(c) and t[end_t] == c[end_c]:
        end_t, end_c = end_t + 1, end_c + 1  # 더 긴 일치가 뒤에서 가져간 글자 ("나와요" 의 "요")
    cut = idx[end_t - 1] + 1 if end_t else 0
    while cut < len(text) and not text[cut].isalnum() and not text[cut].isspace():
        cut += 1  # 낭독 끝의 문장부호
    return cut, end_c >= len(c) - MAX_SKIP_LETTERS


def analyze(raw: dict[str, Any], pcm: bytes, chunk: int, watch: float) -> dict[str, Any]:
    """진행 기록 + 받은 모델 오디오 → 지표. 시각은 사용자 목소리 기준."""
    if raw["result"] != "done":
        return {}
    a, b = voiced_offsets(UTTERANCES[raw["utterance"]][1], trim=True)
    v0, v1 = raw["user_start"] + a, raw["user_start"] + b
    m = audio_metrics(pcm, v0, v1, watch)
    bins = speech_bins(pcm)
    lo, hi = int((v0 - TALKING_WINDOW_SEC[0]) / BIN_SEC), int((v0 + TALKING_WINDOW_SEC[1]) / BIN_SEC) + 1
    # 끼어든 분량부터의 모델 전사를 대사 낭독과 그 뒤(대답)로 나눈다
    pieces = [(p, d) for p, sp, d in raw["deltas"] if sp == "assistant" and p >= raw["appends"][chunk - 1]]
    text = "".join(d for _, d in pieces)
    cut, to_end = split_reading(text, CHUNKS[chunk - 1])
    heard_upto = len("".join(d for p, d in pieces if p < v0 + TRANSCRIPT_LAG_SEC))  # 끼어든 곳 (전사 도착으로 어림)
    reply_at, offset = None, 0
    for p, d in pieces:
        if offset + len(d) > cut and text[max(cut, offset) : offset + len(d)].strip():
            reply_at = round(p - v1, 1)
            break
        offset += len(d)
    reply = text[cut:].strip()
    return {
        "talking_at_interrupt": bool(bins[max(0, lo) : hi].any()),
        "overlap_sec": m["overlap_sec"],
        "after_sec": m["after_sec"],
        "heard": "".join(d for p, sp, d in raw["deltas"] if sp == "user" and p >= raw["user_start"]).strip(),
        "reading": text[: min(heard_upto, cut)].strip() + " ‖ " + text[min(heard_upto, cut) : cut].strip(),
        "read_to_end": to_end,
        "reply": reply,
        "reply_at": reply_at,  # 대답 첫 전사 조각 도착 (목소리 끝 기준, 전사 지연 포함)
        "reuse": script_reuse(reply, list(CHUNKS[: len(raw["appends"])])),
        "reading_word": "읽" in reply,
        "question": "?" in reply,
        "delegations": sum(e["name"] == "delegation" and e["pos"] >= v0 for e in raw["events"]),
        "sources": transcript_sources(raw.get("transcript_events")),
    }


def transcript_sources(events: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """원본 전사 조각을 받은 순서대로, 화자와 ``client_event_id`` 가 같은 것끼리 이어 붙인다."""
    if events is None:
        return None
    out: list[dict[str, Any]] = []
    for e in events:
        if not e["type"].endswith("transcript.delta") or not (e["delta"] or "").strip():
            continue
        who = "user" if e["type"].startswith("input") else "ray"
        if out and out[-1]["who"] == who and out[-1]["cid"] == e["cid"]:
            out[-1]["text"] += e["delta"]
        else:
            out.append({"who": who, "cid": e["cid"], "start_ms": e["start_ms"], "text": e["delta"]})
    return out


def script_reuse(text: str, chunks: list[str]) -> list[dict[str, Any]]:
    """``text`` 와 각 대사의 가장 긴 공통 글자열이 ``REUSE_MIN_LETTERS`` 이상인 대사들."""
    t = _letter_str(text)
    found = []
    for i, chunk in enumerate(chunks):
        c = _letter_str(chunk)
        m = difflib.SequenceMatcher(None, t, c, autojunk=False).find_longest_match(0, len(t), 0, len(c))
        if m.size >= REUSE_MIN_LETTERS:
            found.append({"chunk": i + 1, "text": t[m.a : m.a + m.size]})
    return found


class TaggedSession(GPTLiveSession):
    """commentary·안내를 event_id 와 함께 보내고, 원본 전사·수락 이벤트를 ``client_event_id`` 와 세션 시각째
    남기는 세션. 어댑터는 전사를 묶음 도구(TranscriptGrouper)에 넘기며 이 둘을 버린다."""

    _RECORDED = (
        "session.output_transcript.delta",
        "session.input_transcript.delta",
        "session.commentary.appended",
        "session.instructions.appended",
    )

    def __init__(self, config: LiveSessionConfig) -> None:
        super().__init__(config)
        self.raw_events: list[dict[str, Any]] = []  # 수신 스레드가 쌓는다

    def append_tagged(self, kind: str, content: str, event_id: str) -> None:
        self._guard()
        self._send(lambda c: getattr(c.session, kind).append(content=content, delegation_id=None, event_id=event_id))

    def _dispatch(self, event: Any) -> None:
        t = getattr(event, "type", None)
        if t in self._RECORDED:
            self.raw_events.append(
                {
                    "type": t.removeprefix("session."),
                    "cid": getattr(event, "client_event_id", None),
                    "delta": getattr(event, "delta", None),
                    "start_ms": getattr(event, "start_ms", None),
                    "end_ms": getattr(event, "end_ms", None),
                }
            )
        super()._dispatch(event)


class Session:
    """세션 하나의 진행과 기록. 위치는 받은 오디오 기준 초."""

    def __init__(self, key: str, args: argparse.Namespace) -> None:
        self.key, self.args = key, args
        self.text = UTTERANCES[key][1]
        self.clip = user_clip(self.text, trim=True)
        self.voice_off = voiced_offsets(self.text, trim=True)
        self.live = TaggedSession(
            LiveSessionConfig(instructions=LIVE_INSTRUCTIONS, voice=LIVE_VOICE, sample_rate=BRIDGE_SAMPLE_RATE)
        )
        self.pcm = bytearray()
        self.user_pcm = bytearray()
        self.user_start: float | None = None  # 사용자 클립을 넣기 시작한 위치
        self.chunk_voice: float | None = None  # 끼어든 분량의 첫 음성 위치
        self.appends: list[float] = []  # 분량을 넣은 위치 (순서 = 분량 번호 - 1)
        self.first_voice: float | None = None  # 마지막으로 넣은 분량의 첫 음성 위치
        self.last_voice: float | None = None
        self.quiet = 0.0  # 마지막 음성 뒤 받은 무음 조각 길이
        self.deltas: list[tuple[float, str, str, str]] = []  # (도착 위치, 화자, 세그먼트, 전사 조각)
        self.events: list[dict[str, Any]] = []
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
                            self.deltas.append((self.pos(), ev.speaker, ev.segment_id, delta))
                    elif isinstance(ev, LiveDelegationCreated):
                        self.events.append({"pos": round(self.pos(), 2), "name": "delegation"})
                    elif isinstance(ev, LiveError):
                        self.events.append({"pos": round(self.pos(), 2), "name": "error", "message": ev.message})
                    elif isinstance(ev, LiveClosed):
                        self.result = "closed"
                elapsed = now - t0
                if preamble_t is None:
                    if elapsed >= START_DELAY_SEC:
                        self.live.append_tagged("instructions", PREAMBLE, "ray_pre")
                        preamble_t = elapsed
                elif self.result == "running":
                    self._step(elapsed - preamble_t)
                if elapsed > MAX_SEC and self.result == "running":
                    self.result = "timeout"
                next_t += FRAME_SEC
                time.sleep(max(0.0, next_t - time.monotonic()))
        finally:
            self.live.close(graceful=True)

    def _append(self) -> None:
        self.appends.append(self.pos())
        self.first_voice = None
        n = len(self.appends)
        self.live.append_tagged("commentary", CHUNKS[n - 1], f"ray_c{n}")

    def _step(self, since_preamble: float) -> None:
        p = self.pos()
        if not self.appends:
            if since_preamble >= PREAMBLE_SETTLE_SEC and (self.last_voice is None or self.quiet >= PREAMBLE_QUIET_SEC):
                self._append()
        elif self.user_start is not None:
            if p - (self.user_start + self.voice_off[1]) >= self.args.watch:
                self.result = "done"
        elif self.first_voice is None:
            if p - self.appends[-1] >= NO_VOICE_SEC:
                self.result = "no_voice"
        elif len(self.appends) == self.args.chunk:
            if p - self.first_voice >= self.args.at and self.quiet < VOICED_SEC:
                self.user_pcm.extend(self.clip)
                self.user_start, self.chunk_voice = p, self.first_voice
            elif self.quiet >= self.args.gap:
                self.result = "missed"  # 끼어들 시점 전에 분량이 끝났다
        elif self.quiet >= self.args.gap:
            self._append()

    def raw(self) -> dict[str, Any]:
        """다시 분석할 수 있게 남기는 진행 기록."""
        return {
            "utterance": self.key,
            "result": self.result,
            "events": self.events,
            "appends": [round(p, 2) for p in self.appends],
            "chunk_voice": round(self.chunk_voice, 2) if self.chunk_voice is not None else None,
            "user_start": round(self.user_start, 3) if self.user_start is not None else None,
            "deltas": [[round(p, 2), sp, d] for p, sp, _, d in self.deltas],
            "transcript_events": list(self.live.raw_events),
        }

    def save(self, path: Path) -> None:
        clips = [(self.user_start, self.clip)] if self.user_start is not None else []
        write_conversation_wav(path, bytes(self.pcm), clips, BRIDGE_SAMPLE_RATE)
        labels = [(p, p, f"append #{i + 1}") for i, p in enumerate(self.appends)]
        if self.user_start is not None:
            a, b = self.voice_off
            labels.append((self.user_start + a, self.user_start + b, f"USER: {self.text}"))
        labels.extend((e["pos"], e["pos"], e["name"]) for e in self.events)
        for a, b in speech_segments(speech_bins(bytes(self.pcm))):
            text = "".join(d for p, sp, _, d in self.deltas if sp == "assistant" and a <= p < b + TRANSCRIPT_LAG_SEC)
            labels.append((a, b, f"RAY: {text.strip()}"))
        labels.sort()
        path.with_suffix(".labels.txt").write_text("".join(f"{a:.3f}\t{b:.3f}\t{t}\n" for a, b, t in labels))


# ---------------------------------------------------------------------------
# 출력·실행
# ---------------------------------------------------------------------------


def format_row(r: dict[str, Any]) -> list[str]:
    if r["result"] != "done":
        return [f"  #{r['run']} {r['result']}"]
    reply_at = f"+{r['reply_at']}s" if r["reply_at"] is not None else "-"
    head = f"  #{r['run']} 대답 전사 {reply_at}, 겹친 말 {r['overlap_sec']}s, 이후 말 {r['after_sec']}s"
    head += f" | 들림: {r['heard'] or '-'}"
    if not r["talking_at_interrupt"]:
        head += " (끼어들 때 말소리 없음)"
    spoken, _, rest = r["reading"].partition(" ‖ ")
    reading = f"…{spoken[-30:]} ‖ {rest} ({'끝까지' if r['read_to_end'] else '중간에 멈춤'})"
    marks = [f'대본#{u["chunk"]} "{u["text"]}"' for u in r["reuse"]]
    marks += ["읽"] * r["reading_word"] + ["?"] * r["question"] + [f"위임 {r['delegations']}"] * bool(r["delegations"])
    lines = [head, f"     낭독: {reading}", f"     대답: {r['reply'] or '-'}"]
    if marks:
        lines.append(f"     표시: {' · '.join(marks)}")
    return lines


def _median(values: list[float]) -> str:
    return f"{float(np.median(values)):.1f}s" if values else "-"


def report(tag: str) -> None:
    root = find_result_dir("dj_interrupt_reply", tag)
    if root is None:
        raise SystemExit(f"결과 폴더 없음: {tag}")
    lines: list[str] = []
    for key, (kind, text) in UTTERANCES.items():
        path = root / f"{key}.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text())["results"]
        valid = [r for r in rows if r["result"] == "done"]
        n = len(valid)
        lines.append(
            f"\n[{kind}] {text} ({key}) — 유효 {n}/{len(rows)}, "
            f"분량 끝까지 읽음 {sum(r['read_to_end'] for r in valid)}/{n}, "
            f"대답 전사 중앙 {_median([r['reply_at'] for r in valid if r['reply_at'] is not None])}, "
            f"대답 없음 {sum(not r['reply'] for r in valid)}/{n}, "
            f"대본 재사용 {sum(bool(r['reuse']) for r in valid)}/{n}, "
            f"물음표 {sum(r['question'] for r in valid)}/{n}, 위임 {sum(bool(r['delegations']) for r in valid)}/{n}"
        )
        for r in rows:
            lines.extend(format_row(r))
    out = "\n".join(lines)
    print(out)
    (root / "report.txt").write_text(out + "\n")


def reanalyze(tag: str) -> None:
    """저장된 진행 기록과 WAV 로 지표를 다시 계산해 발화별 json 을 고쳐 쓴다."""
    root = find_result_dir("dj_interrupt_reply", tag)
    if root is None:
        raise SystemExit(f"결과 폴더 없음: {tag}")
    for key in UTTERANCES:
        path = root / f"{key}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        chunk, watch = data["config"]["chunk"], data["config"]["watch"]
        for r in data["results"]:
            if r["result"] == "done":
                r.update(analyze(r, read_model_channel(root / f"{key}_{r['run']}.wav"), chunk, watch))
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    report(tag)


def run(args: argparse.Namespace) -> None:
    out_dir = result_dir("dj_interrupt_reply", args.tag)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-28s %(levelname)-5s %(message)s",
        handlers=[logging.FileHandler(out_dir / f"pipeline_{os.getpid()}.log")],
    )
    (out_dir / "instructions.txt").write_text(LIVE_INSTRUCTIONS)
    config = {
        "chunk": args.chunk,
        "at": args.at,
        "gap": args.gap,
        "watch": args.watch,
        "voice": LIVE_VOICE,
        "preamble": PREAMBLE,
        "chunks": list(CHUNKS),
        "utterances": UTTERANCES,
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1))
    for key in args.utterances:
        tts24(UTTERANCES[key][1])  # 캐시를 미리 채워 세션 중 합성 지연이 없게
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
                row = {"run": n, **raw, **analyze(raw, bytes(session.pcm), args.chunk, args.watch)}
                session.save(out_dir / f"{key}_{n}.wav")
            rows[key].append(row)
            print(f"[{key} #{n}] {UTTERANCES[key][1]}", flush=True)
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
    parser.add_argument("--watch", type=float, default=20.0, help="사용자 목소리가 끝난 뒤 지켜볼 초")
    parser.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--report", metavar="TAG", help="발화별로 대답을 모아 보여주고 끝낸다")
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
