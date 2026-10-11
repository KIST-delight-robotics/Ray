"""DJ 분량 주입 실험 — 짧은 대사를 commentary.append 로 하나씩 넣고, 모델이 말을 멈추면 다음 것을 넣는다.

라이브 지시문은 정리한 지시문(``live_instructions_simple.txt``, DJ·분량에 대한 안내 없음), 위임 없음, 입력은 무음뿐.
다음 분량은 모델의 첫 음성 뒤 받은 무음 조각이 ``--gap`` 초 이어지면 넣는다. 관찰용이라 넉넉히 잡는다 — 분량 안의
쉼을 끊지 않고, 모델이 쉬었다가 스스로 이어 말하는지까지 보려고.

기록(분량마다): 주입 → 첫 음성, 음성 길이, 첫 음성 뒤 음성 사이 쉼(0.3 s 이상), 그동안 나온 출력 전사.
요약: 쉼 분포와, 무음 기준이 T 였다면 분량 도중에 다음 것을 넣었을 분량 수. 덧붙임 정도(말한 글자 / 대사 글자).

    uv run python scripts/gpt_live/exp_dj_commentary_chunks.py --runs 3
    # 앞서 넣기: 분량의 첫 음성 SEC 초 뒤에 다음 분량을 미리 넣는다 (한 칸만 — 미리 넣은 분량 다음은 무음으로)
    uv run python scripts/gpt_live/exp_dj_commentary_chunks.py --runs 2 --ahead 2.0
    # 안내: 첫 분량 전에 instructions.append 로 한 번 (--preamble [read|light]), 또는 분량마다 앞머리로 (--prefix)
    uv run python scripts/gpt_live/exp_dj_commentary_chunks.py --runs 3 --preamble
    uv run python scripts/gpt_live/exp_dj_commentary_chunks.py --runs 3 --prefix
    # 분량 키우기(대사 2개씩) + 끼어들기: 2번째 분량 첫 음성 3 s 뒤 사용자 질문, 대화가 3 s 조용하면 다음 분량으로
    uv run python scripts/gpt_live/exp_dj_commentary_chunks.py --preamble --merge 2 --gap 1.0 \\
        --interrupt-chunk 2 --resume next

끼어들기: 앱은 사용자 전사가 오면 끼어든 것으로 보고 분량 넣기를 멈춘다. 사용자 음성을 넣는 동안은 마이크 레벨로
말소리를 안다고 보고 다음 분량을 넣지 않는다. 사용자 말이 끝나고 모델이 대답한 뒤 둘 다 ``--resume-after`` 초
조용하면 끊긴 분량을 다시(restart) 또는 다음 분량을(next) 넣는다.

결과: var/experiments/dj_commentary_chunks/<번호>_<태그>/ — results.json, instructions.txt, pipeline.log,
run_<n>.wav(왼쪽 사용자, 오른쪽 모델) + run_<n>.labels.txt(Audacity: 파일 → 가져오기 → 레이블).
위치·길이는 받은 오디오 기준 초(WAV 위치와 같다). 서버가 빠뜨린 조각(주로 무음 구간, 2~3%)은 WAV 에 없다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from exp_audio import write_conversation_wav
from exp_dirs import result_dir

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

BYTES_PER_SEC = BRIDGE_SAMPLE_RATE * 2
FRAME_SEC = 0.02
QUIET_RMS = 30  # LiveSessionLoop._QUIET_RMS 와 같게
MIN_PAUSE_SEC = 0.3  # 이보다 짧은 무음은 쉼으로 기록하지 않는다
NO_VOICE_SEC = 8.0  # 주입 뒤 이만큼 음성이 없으면 다음 분량으로
START_DELAY_SEC = 1.0  # 세션 시작 뒤 첫 분량까지
T_CANDIDATES = (0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5)
TTS_CACHE = Path(__file__).resolve().parent / "tts_cache"  # mic_live.py·하네스와 같은 캐시 폴더
INTERRUPT_SAY = "너는 서랍에 뭐 넣어 둬?"
NO_TRANSCRIPT_SEC = 3.0  # 사용자 음성이 끝나고 이만큼 전사가 없어도 끼어들기로 본다
RESUME_MAX_WAIT_SEC = 15.0  # 끼어들기 판정 뒤 이만큼 지나면 대화가 안 끝나도 복귀

# drawer.md 메모를 대사로 옮긴 것. 질문은 넣지 않는다 (대답을 기다리는 쉼이 섞이지 않게).
CHUNKS = (
    "늦은 시간이네요. 오늘은 책상 서랍 이야기를 잠깐 해 볼게요.",
    "서랍은 집에서 가장 느리게 바뀌는 자리예요. "
    "책상 위는 매일 달라지는데, 서랍 안은 몇 년이 지나도 그대로일 때가 많아요.",
    "열어 보면 꼭 나오는 것들이 있죠. 다 쓴 볼펜, 어디 건지 모를 열쇠, 접힌 영수증 같은 것들요.",
    "가끔은 쓰다 만 편지도 나와요. 끝맺지 못한 문장이 그대로 남아 있는 거죠.",
    "설레지 않으면 버리라는 정리 격언이 있잖아요. 서랍에 그대로 적용하면 반은 비워야 해서, 대개는 그냥 다시 닫게 돼요.",
    "그런데 생각해 보면, 서랍은 정리가 안 된 곳이라기보다 우리 대신 기억을 맡아 두는 자리인지도 몰라요.",
    "그래서 오늘은 서랍에 넣어 둔 마음에 관한 노래를 골랐어요. 십센치의 서랍, 들려드릴게요.",
)

# 정리한 라이브 지시문 — 공식 템플릿 + 근거가 기록된 줄만. 분량 주입 DJ 를 구현할 때 운영 지시문을 이것으로 바꾼다.
LIVE_INSTRUCTIONS = (Path(__file__).resolve().parent / "live_instructions_simple.txt").read_text(encoding="utf-8")

# commentary 는 "전달할 내용"으로 설계돼 있어(레퍼런스 "communicate", 가이드 "paraphrase") 낭독할 대사라고 알려 준다.
# --preamble NAME: 첫 분량 전에 instructions.append 로 한 번. 대꾸하면 그 말이 끝난 뒤 첫 분량을 넣는다.
# read 는 낭독 틀(덧붙임은 줄지만 끼어들기에 문장을 마저 끝내는 경향), light 는 덧붙이지 말라는 말만(덧붙임 억제 실패),
# read2 는 낭독 틀에서 slowly·then stop 을 빼고 문장 중간이라도 바로 멈추라고 명시.
PREAMBLES = {
    "read": (
        "From now on, the application sends you lines to read aloud as commentary, one at a time. "
        "Read each line as written, slowly and calmly, then stop. Do not add your own words or questions. "
        "If the user speaks to you, stop reading and talk with them."
    ),
    # read 에서 낭독 어휘만 "이야기"로 — 끼어들 때 "읽는 건 멈출게요" 처럼 새어 나와도 자연스러운 말이 되게
    "read_talk": (
        "From now on, the application gives you your talk as commentary, one part at a time. "
        "Say each part as written, slowly and calmly, then stop. Do not add your own words or questions. "
        "If the user speaks to you, pause the talk and talk with them."
    ),
    # read + 새어 나옴 대책 — 금지(noread) / 사용자에게는 자기 이야기라고 긍정형으로(own)
    "read_noread": (
        "From now on, the application sends you lines to read aloud as commentary, one at a time. "
        "Read each line as written, slowly and calmly, then stop. Do not add your own words or questions. "
        "If the user speaks to you, stop reading and talk with them. Never tell the user that you are reading."
    ),
    "read_own": (
        "From now on, the application sends you lines to read aloud as commentary, one at a time. "
        "Read each line as written, slowly and calmly, then stop. Do not add your own words or questions. "
        "If the user speaks to you, stop reading and talk with them. "
        "To the user, these lines are your own talk; if you mention them, call them your story."
    ),
    "light": "When you say commentary from the application, do not add your own words or questions.",
    "read2": (
        "From now on, the application sends you lines to read aloud as commentary, one at a time. "
        "Read each line as written. Do not add your own words or questions. "
        "If the user starts speaking, stop right away, even mid-sentence, and talk with them."
    ),
    # 낭독 틀 없이 차례의 끝만 분명히 — say_wait 는 다음 줄을, listen 은 사용자를 기다리게 (둘의 차이는 그 한 곳)
    "say_wait": (
        "The application sends you what to say next as commentary, one line at a time. "
        "Say each line naturally in your own voice, keeping its wording, then wait quietly for the next one. "
        "Do not add anything of your own."
    ),
    # say_wait 에 read 의 "then stop" 을 더한 것 — 덧붙임을 막은 열쇠로 의심되는 부분만 자연스러운 표현에 얹는다
    "say_stop": (
        "The application sends you what to say next as commentary, one line at a time. "
        "Say each line naturally in your own voice, keeping its wording, then stop and wait quietly for the next one. "
        "Do not add anything of your own."
    ),
    # read 에서 낭독 표현만 자연스러운 말하기로 바꾼 것 (끼어들기 문장은 그대로)
    "say_stop_talk": (
        "From now on, the application sends you lines to say as commentary, one at a time. "
        "Say each line naturally in your own voice, keeping its wording, then stop. "
        "Do not add your own words or questions. "
        "If the user speaks to you, stop and talk with them."
    ),
    "next_line": (
        "Each commentary from the application is one line of a longer talk, and the next line will follow. "
        "Say only that line, then wait."
    ),
    "listen": (
        "The application sends you what to say next as commentary, one line at a time. "
        "Say each line naturally in your own voice, keeping its wording, then pause and listen to the user. "
        "Do not add anything of your own."
    ),
}
PREAMBLE_SETTLE_SEC = 3.0  # 안내 뒤 첫 분량까지 최소 대기
PREAMBLE_QUIET_SEC = 1.5  # 안내에 대꾸했으면 이만큼 조용해진 뒤 첫 분량
# --prefix: 분량마다 commentary 앞머리로 (레퍼런스 예시처럼 내용과 지시를 한 commentary 에)
PREFIX = "Read this line aloud as written, then stop: "

logger = logging.getLogger("exp_dj_commentary_chunks")


class AckSession(GPTLiveSession):
    """commentary.appended 수락 시각을 남기는 세션 (어댑터는 ack 를 로그만 한다)."""

    def __init__(self, config: LiveSessionConfig) -> None:
        super().__init__(config)
        self.acks: list[float] = []

    def _dispatch(self, event: Any) -> None:
        if getattr(event, "type", None) == "session.commentary.appended":
            self.acks.append(time.monotonic())
        super()._dispatch(event)


@dataclass
class Chunk:
    index: int
    text: str
    early: bool  # 앞 분량을 말하는 중에 넣었다 (--ahead)
    append_t: float  # monotonic
    append_pos: float  # 받은 오디오 위치
    voiced_at_append: bool  # 넣을 때 모델이 말하고 있었다 (0.3 s 안에 음성)
    resumed: bool = False  # 끼어들기 뒤 복귀로 넣었다
    ack_sec: float | None = None
    first_voice_pos: float | None = None
    last_voice_pos: float | None = None
    voiced_sec: float = 0.0
    pauses: list[tuple[float, float]] = field(default_factory=list)  # (시작 위치, 길이)
    said: str = ""
    end_pos: float | None = None  # 다음을 넣었거나 끝낸 위치
    end_reason: str = ""  # quiet / early_next / no_voice / interrupted / session_end


def _rms(pcm: bytes) -> float:
    s = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(s * s))) if len(s) else 0.0


def _letters(text: str) -> int:
    """공백·문장부호를 뺀 글자 수 — 전사 표기(띄어쓰기, 숫자 읽기) 차이를 덜 타게."""
    return len(re.sub(r"[\W_]+", "", text))


def tts24(text: str) -> bytes:
    """OpenAI TTS 로 합성한 24 kHz mono PCM16 (세션 포맷 그대로). 같은 문장은 파일 캐시."""
    TTS_CACHE.mkdir(exist_ok=True)
    cache = TTS_CACHE / (hashlib.sha1(text.encode()).hexdigest() + ".pcm24k")
    if cache.exists():
        return cache.read_bytes()
    import openai

    resp = openai.OpenAI().audio.speech.create(
        model="gpt-4o-mini-tts", voice="alloy", input=text, response_format="pcm"
    )
    cache.write_bytes(resp.content)
    return resp.content


@dataclass
class Interrupt:
    """끼어들기 한 번의 기록. 위치는 받은 오디오 기준 초."""

    chunk: int  # 끼어든 분량 (0부터)
    user_start_pos: float
    user_sec: float
    user_end_pos: float | None = None
    detect_pos: float | None = None  # 사용자 전사가 처음 온 위치 — 앱이 끼어들기를 아는 시점
    model_stop_pos: float | None = None  # 사용자 말 시작 뒤 모델 음성이 처음 0.5 s 이상 끊긴 위치
    heard: str = ""
    reply: str = ""  # 끼어들기 판정부터 복귀 전까지 모델이 한 말
    resume_pos: float | None = None
    resume_mode: str = ""


def run_once(n: int, out_dir: Path, args: argparse.Namespace, instructions: str, texts: list[str]) -> dict[str, Any]:
    live = AckSession(LiveSessionConfig(instructions=instructions, voice=LIVE_VOICE, sample_rate=BRIDGE_SAMPLE_RATE))
    live.start()
    frame_bytes = int(BYTES_PER_SEC * FRAME_SEC)
    silence = bytes(frame_bytes)
    speech = tts24(args.interrupt_say) if args.interrupt_chunk else b""
    pcm = bytearray()
    chunks: list[Chunk] = []
    quiet = 0.0  # 마지막 음성 뒤 받은 무음 조각 길이
    last_voice_end: float | None = None
    seg_text: dict[str, str] = {}
    transcripts: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    finished_pos: float | None = None
    preamble_t: float | None = None
    pre_said = ""  # 첫 분량 전에 모델이 한 말 (안내에 대한 대꾸)
    user_pcm = bytearray()  # 아직 보내지 않은 사용자 음성
    intr: Interrupt | None = None
    conversing = False  # 끼어들기 판정 뒤 복귀 전 — 분량을 넣지 않는다
    user_end_t: float | None = None
    last_user_text_t: float | None = None
    detect_t: float | None = None
    t_start = time.monotonic()
    next_t = t_start

    def pos() -> float:
        return len(pcm) / BYTES_PER_SEC

    def append(i: int, *, early: bool = False, resumed: bool = False) -> None:
        p = pos()
        voiced = last_voice_end is not None and p - last_voice_end <= 0.3
        chunks.append(Chunk(i, texts[i], early, time.monotonic(), p, voiced, resumed=resumed))
        live.append_commentary((PREFIX if args.prefix else "") + texts[i])
        mark = " (early)" if early else " (resume)" if resumed else ""
        print(f"  [{p:6.1f}] append #{i + 1}{mark}{' — speaking' if voiced else ''}", flush=True)

    def advance(cur: Chunk, reason: str) -> None:
        nonlocal finished_pos
        cur.end_pos, cur.end_reason = pos(), reason
        if cur.index + 1 < len(texts):
            append(cur.index + 1)
        else:
            finished_pos = pos()

    def start_conversing(detected: bool) -> None:
        nonlocal conversing, detect_t
        assert intr is not None
        conversing, detect_t = True, time.monotonic()
        if detected:
            intr.detect_pos = pos()
        chunks[-1].end_pos, chunks[-1].end_reason = pos(), "interrupted"
        print(f"  [{pos():6.1f}] interrupted{'' if detected else ' (no user transcript)'}", flush=True)

    try:
        while True:
            now = time.monotonic()
            if user_pcm:
                frame = bytes(user_pcm[:frame_bytes]).ljust(frame_bytes, b"\0")
                del user_pcm[:frame_bytes]
                if not user_pcm and intr is not None:
                    user_end_t, intr.user_end_pos = now, pos()
            else:
                frame = silence
            live.send_audio(frame)
            closed = False
            while (ev := live.poll_event()) is not None:
                cur = chunks[-1] if chunks else None
                if isinstance(ev, LiveAudio):
                    start = pos()
                    sec = len(ev.pcm) / BYTES_PER_SEC
                    pcm.extend(ev.pcm)
                    if any(ev.pcm) and _rms(ev.pcm) >= QUIET_RMS:
                        if cur is None:
                            if not any(e["name"] == "voice_before_first_chunk" for e in events):
                                events.append({"pos": round(start, 2), "name": "voice_before_first_chunk"})
                        elif not conversing:
                            if cur.first_voice_pos is None:
                                cur.first_voice_pos = start
                            elif quiet >= MIN_PAUSE_SEC:
                                cur.pauses.append((start - quiet, quiet))
                            cur.voiced_sec += sec
                            cur.last_voice_pos = start + sec
                        last_voice_end = start + sec
                        quiet = 0.0
                    else:
                        quiet += sec
                    if (
                        intr is not None
                        and intr.model_stop_pos is None
                        and quiet >= 0.5
                        and pos() >= intr.user_start_pos + 0.5
                    ):
                        intr.model_stop_pos = max(pos() - quiet, intr.user_start_pos)
                elif isinstance(ev, LiveTranscript):
                    prev = transcripts.get(ev.segment_id)
                    transcripts[ev.segment_id] = {
                        "t": prev["t"] if prev else now - t_start,
                        "speaker": ev.speaker,
                        "text": ev.text.strip(),
                    }
                    if ev.speaker == "user":
                        if ev.text.strip() and intr is not None:
                            last_user_text_t = now
                            intr.heard = ev.text.strip()
                            if not conversing and intr.detect_pos is None and intr.resume_pos is None:
                                start_conversing(detected=True)
                        continue
                    old = seg_text.get(ev.segment_id, "")
                    delta = ev.text[len(old) :] if ev.text.startswith(old) else ev.text
                    seg_text[ev.segment_id] = ev.text
                    # 최신 분량에 붙인다 — 다음 분량 첫머리 전사가 주입 직후에 오기도 한다. 분량 사이 무음(--gap)이
                    # 전사 지연보다 길어야 맞고, 앞서 넣기에서는 경계가 흐리다 (WAV 로 확인)
                    if conversing and intr is not None:
                        intr.reply += delta
                    elif chunks:
                        chunks[-1].said += delta
                    else:
                        pre_said += delta
                elif isinstance(ev, LiveDelegationCreated):
                    events.append({"pos": round(pos(), 2), "name": "delegation_created", "id": ev.delegation_id})
                elif isinstance(ev, LiveError):
                    events.append({"pos": round(pos(), 2), "name": "error", "code": ev.code, "message": ev.message})
                elif isinstance(ev, LiveClosed):
                    events.append({"pos": round(pos(), 2), "name": "closed", "reason": ev.reason})
                    closed = True
            if closed or now - t_start > args.max:
                break

            cur = chunks[-1] if chunks else None
            if cur is None:
                if args.preamble and preamble_t is None:
                    if now - t_start >= START_DELAY_SEC:
                        live.append_instructions(PREAMBLES[args.preamble])
                        preamble_t = now
                        events.append({"pos": round(pos(), 2), "name": "preamble"})
                        print(f"  [{pos():6.1f}] preamble", flush=True)
                elif preamble_t is None:
                    if now - t_start >= START_DELAY_SEC:
                        append(0)
                elif now - preamble_t >= PREAMBLE_SETTLE_SEC and (
                    last_voice_end is None or quiet >= PREAMBLE_QUIET_SEC
                ):
                    append(0)
            elif finished_pos is not None:
                if pos() - finished_pos >= args.tail:
                    break
            elif conversing:
                assert intr is not None and detect_t is not None
                last_user = max(user_end_t or 0.0, last_user_text_t or 0.0)
                answered = (
                    last_voice_end is not None and intr.user_end_pos is not None and last_voice_end > intr.user_end_pos
                )
                calm = user_end_t is not None and now - last_user >= args.resume_after and quiet >= args.resume_after
                if (answered and calm) or now - detect_t >= RESUME_MAX_WAIT_SEC:
                    conversing = False
                    intr.resume_pos, intr.resume_mode = pos(), args.resume
                    nxt = cur.index if args.resume == "restart" else cur.index + 1
                    if nxt < len(texts):
                        append(nxt, resumed=True)
                    else:
                        finished_pos = pos()
            elif intr is not None and intr.resume_pos is None and intr.detect_pos is None:
                # 사용자 음성을 넣는 중 — 앱은 마이크 레벨로 말소리를 알므로 다음 분량을 넣지 않고 전사를 기다린다
                if user_end_t is not None and now - user_end_t >= NO_TRANSCRIPT_SEC:
                    start_conversing(detected=False)
            elif cur.first_voice_pos is None:
                if now - cur.append_t >= NO_VOICE_SEC:
                    advance(cur, "no_voice")
            elif (
                args.interrupt_chunk
                and intr is None
                and cur.index + 1 == args.interrupt_chunk
                and pos() - cur.first_voice_pos >= args.interrupt_after
            ):
                user_pcm.extend(speech)
                intr = Interrupt(cur.index, pos(), len(speech) / BYTES_PER_SEC)
                print(f"  [{pos():6.1f}] user: {args.interrupt_say}", flush=True)
            elif quiet >= args.gap:
                advance(cur, "quiet")
            elif (
                args.ahead is not None
                and not cur.early
                and cur.index + 1 < len(texts)
                and pos() - cur.first_voice_pos >= args.ahead
            ):
                cur.end_pos, cur.end_reason = pos(), "early_next"
                append(cur.index + 1, early=True)

            next_t += FRAME_SEC
            time.sleep(max(0.0, next_t - time.monotonic()))
    finally:
        live.close(graceful=True)

    for c in chunks:
        if c.end_pos is None:
            c.end_pos, c.end_reason = pos(), "session_end"
    for c, ack in zip(chunks, live.acks, strict=False):
        c.ack_sec = ack - c.append_t

    clips = [(intr.user_start_pos, speech)] if intr is not None else []
    write_conversation_wav(out_dir / f"run_{n}.wav", bytes(pcm), clips, BRIDGE_SAMPLE_RATE)
    labels = [(e["pos"], e["pos"], "preamble") for e in events if e["name"] == "preamble"]
    for c in chunks:
        mark = " (early)" if c.early else " (resume)" if c.resumed else ""
        labels.append((c.append_pos, c.append_pos, f"append #{c.index + 1}{mark}"))
        if c.first_voice_pos is not None and c.last_voice_pos is not None:
            labels.append((c.first_voice_pos, c.last_voice_pos, f"#{c.index + 1} {c.said.strip()}"))
        labels.extend((s, s + d, f"pause {d:.1f}s") for s, d in c.pauses)
    if intr is not None:
        labels.append((intr.user_start_pos, intr.user_start_pos + intr.user_sec, f"USER: {args.interrupt_say}"))
        for p, text in ((intr.model_stop_pos, "model stopped"), (intr.detect_pos, "user transcript")):
            if p is not None:
                labels.append((p, p, text))
        if intr.resume_pos is not None:
            labels.append(
                (intr.resume_pos, intr.resume_pos, f"resume ({intr.resume_mode}) | reply: {intr.reply.strip()}")
            )
    labels.sort()
    (out_dir / f"run_{n}.labels.txt").write_text("".join(f"{a:.3f}\t{b:.3f}\t{text}\n" for a, b, text in labels))

    return {
        "run": n,
        "pre_said": pre_said.strip(),
        "chunks": [_chunk_row(c) for c in chunks],
        "interrupt": _interrupt_row(intr) if intr is not None else None,
        "events": events,
        "transcripts": sorted(transcripts.values(), key=lambda s: s["t"]),
        "audio_sec": round(pos(), 2),
    }


def _interrupt_row(intr: Interrupt) -> dict[str, Any]:
    row = asdict(intr)
    row["reply"] = intr.reply.strip()
    for k, v in row.items():
        if isinstance(v, float):
            row[k] = round(v, 2)
    return row


def _chunk_row(c: Chunk) -> dict[str, Any]:
    row = asdict(c)
    row.pop("append_t")
    row["said"] = c.said.strip()
    row["latency"] = round(c.first_voice_pos - c.append_pos, 2) if c.first_voice_pos is not None else None
    row["said_ratio"] = round(_letters(c.said) / _letters(c.text), 2)  # 말한 글자 / 대사 글자
    row["pauses"] = [[round(s, 2), round(d, 2)] for s, d in c.pauses]
    for k in ("append_pos", "first_voice_pos", "last_voice_pos", "voiced_sec", "end_pos", "ack_sec"):
        if row[k] is not None:
            row[k] = round(row[k], 2)
    return row


def print_run(row: dict[str, Any]) -> None:
    print(f"[run {row['run']}] audio {row['audio_sec']}s, events: {[e['name'] for e in row['events']] or '-'}")
    if row["pre_said"]:
        print(f"  before #1: {row['pre_said']}")
    if (it := row.get("interrupt")) is not None:
        start = it["user_start_pos"]

        def rel(key: str) -> str:
            return f"+{it[key] - start:.1f}s" if it[key] is not None else "-"

        print(
            f"  interrupt #{it['chunk'] + 1} at {start}: model stop {rel('model_stop_pos')}, "
            f"transcript {rel('detect_pos')}, resume {rel('resume_pos')} ({it['resume_mode']}) "
            f"| heard: {it['heard']} | reply: {it['reply']}"
        )
    for c in row["chunks"]:
        pauses = " ".join(f"{d:.1f}" for _, d in c["pauses"]) or "-"
        lat = f"+{c['latency']:.1f}s" if c["latency"] is not None else "no voice"
        print(
            f"  #{c['index'] + 1}{'e' if c['early'] else ' '} {lat:>8}  voice {c['voiced_sec']:4.1f}s  "
            f"x{c['said_ratio']:.2f}  pauses [{pauses}]  end={c['end_reason']}  said: {c['said']}"
        )


def print_summary(rows: list[dict[str, Any]]) -> None:
    normal = [
        c
        for r in rows
        for c in r["chunks"]
        if not c["early"] and c["latency"] is not None and c["end_reason"] != "interrupted"
    ]
    if not normal:
        print("음성이 나온 분량이 없다")
        return
    pauses = sorted(d for c in normal for _, d in c["pauses"])
    lat = [c["latency"] for c in normal]
    ratios = [c["said_ratio"] for c in normal]
    print(f"\n분량 {len(normal)}개 (앞서 넣은 것·끊긴 것 제외)")
    print(f"주입 → 첫 음성 {min(lat):.1f}~{max(lat):.1f}s (중앙 {np.median(lat):.1f})")
    print(
        f"말한 글자 / 대사 글자: 중앙 {np.median(ratios):.2f} ({min(ratios):.2f}~{max(ratios):.2f}), "
        f"1.3배 초과 {sum(r > 1.3 for r in ratios)}/{len(normal)}, "
        f"물음표 든 분량 {sum('?' in c['said'] for c in normal)}/{len(normal)}"
    )
    talked = [r["pre_said"] for r in rows if r.get("pre_said")]
    if any(e["name"] == "preamble" for r in rows for e in r["events"]):
        print(f"안내에 대꾸: {len(talked)}/{len(rows)}")
    print(f"분량 안 쉼 {len(pauses)}회: {' '.join(f'{d:.1f}' for d in pauses) or '-'}")
    print("T(s)  도중에 넣었을 분량  분량 사이 정적(주입→음성 중앙 + T)")
    for t in T_CANDIDATES:
        early = sum(1 for c in normal if any(d >= t for _, d in c["pauses"]))
        print(f"{t:4.1f}  {early:3d}/{len(normal)}            {np.median(lat) + t:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--gap", type=float, default=4.0, help="첫 음성 뒤 이만큼 무음이면 다음 분량")
    parser.add_argument("--ahead", type=float, default=None, help="첫 음성 SEC 초 뒤에 다음 분량을 미리 넣는다")
    parser.add_argument("--tail", type=float, default=8.0, help="마지막 분량이 끝난 뒤 더 지켜볼 초")
    parser.add_argument(
        "--preamble",
        nargs="?",
        const="read",
        choices=list(PREAMBLES),
        help="첫 분량 전에 이름의 안내를 instructions.append 로 (이름 없으면 read)",
    )
    parser.add_argument("--prefix", action="store_true", help="분량마다 commentary 앞에 낭독 안내를 붙인다")
    parser.add_argument("--merge", type=int, default=1, help="대사를 이만큼씩 묶어 한 분량으로")
    parser.add_argument("--interrupt-chunk", type=int, default=None, help="이 분량(1부터)에서 사용자가 끼어든다")
    parser.add_argument("--interrupt-after", type=float, default=3.0, help="그 분량의 첫 음성 뒤 이 초에 끼어든다")
    parser.add_argument("--interrupt-say", default=INTERRUPT_SAY, help="끼어드는 사용자 발화 (TTS)")
    parser.add_argument("--resume", choices=("restart", "next"), default="next", help="끊긴 분량을 다시 / 다음 분량")
    parser.add_argument("--resume-after", type=float, default=3.0, help="사용자·모델이 이만큼 조용하면 복귀")
    parser.add_argument("--max", type=float, default=240.0, help="세션당 상한 초")
    parser.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    out_dir = result_dir("dj_commentary_chunks", args.tag)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-28s %(levelname)-5s %(message)s",
        handlers=[logging.FileHandler(out_dir / "pipeline.log")],
    )
    instructions = LIVE_INSTRUCTIONS
    texts = [" ".join(CHUNKS[i : i + args.merge]) for i in range(0, len(CHUNKS), args.merge)]
    (out_dir / "instructions.txt").write_text(instructions)
    config = {
        "gap": args.gap,
        "ahead": args.ahead,
        "tail": args.tail,
        "voice": LIVE_VOICE,
        "chunks": texts,
        "preamble": PREAMBLES[args.preamble] if args.preamble else None,
        "prefix": PREFIX if args.prefix else None,
        "interrupt": (
            {
                "chunk": args.interrupt_chunk,
                "after": args.interrupt_after,
                "say": args.interrupt_say,
                "resume": args.resume,
                "resume_after": args.resume_after,
            }
            if args.interrupt_chunk
            else None
        ),
    }

    rows: list[dict[str, Any]] = []
    for n in range(args.runs):
        print(f"[run {n}] start", flush=True)
        try:
            rows.append(run_once(n, out_dir, args, instructions, texts))
        except RuntimeError as exc:
            print(f"[run {n}] failed: {exc}", flush=True)
            continue
        print_run(rows[-1])
        (out_dir / "results.json").write_text(
            json.dumps({"config": config, "runs": rows}, ensure_ascii=False, indent=1)
        )
        time.sleep(1.5)
    print_summary(rows)
    print(f"results: {out_dir}")


if __name__ == "__main__":
    main()
