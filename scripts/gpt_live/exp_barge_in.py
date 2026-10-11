"""끼어들기 반응 비교 — 모델이 길게 말하는 도중 사용자가 끼어들 때, 발화의 출처에 따라 반응이 다른지 본다.

조건(길게 말하게 만드는 방법):
    normal            사용자 음성으로 "서랍에 대한 이야기를 좀 길게 들려줘" — 모델이 지어 말하는 일반 대화 턴
    commentary        대사 4개를 묶은 긴 분량 하나를 commentary.append 로 (안내 없음)
    commentary_pre    commentary + 첫머리 낭독 안내(``PREAMBLES["read"]``, instructions.append)
    commentary_light  commentary + 첫머리에 덧붙이지 말라는 안내만(``PREAMBLES["light"]``)
    commentary_read2  commentary + 낭독 안내 변형(``PREAMBLES["read2"]`` — 문장 중간이라도 바로 멈추라고 명시)
    commentary_<이름> commentary + ``PREAMBLES[<이름>]`` (say_wait / next_line / listen — 낭독 틀 없이 차례의 끝만)

끼어들기(종류마다 회차를 나눈다): question "너는 서랍에 뭐 넣어 둬?", reaction "아, 진짜?", stop "잠깐만".
모델의 첫 말소리 ``--at`` 초 뒤에 TTS 로 넣고, 앱은 그 뒤 아무것도 하지 않는다 — 사용자 말이 끝나고 ``--watch`` 초 동안
지켜본다. 라이브 지시문은 ``live_instructions_simple.txt``, 위임 없음. 모델 말이 끼어들 시점 전에 끝나면 무효.

지표(받은 오디오의 0.1 s 단위 rms, 말소리는 ``SPEECH_RMS`` 이상 — 말이 끝난 뒤 작은 소리를 말로 치지 않게).
시각은 모두 끼어드는 클립 안의 **실제 목소리 구간** 기준이다(TTS 클립 앞뒤 무음 제외 — "잠깐만" 은 앞 1.6 s 가 무음).
``--trim`` 이면 클립 앞뒤 무음을 잘라 넣어, 끼어드는 위치가 ``--at`` 에 맞게 한다(이전 결과와 비교할 때는 빼고 돌린다).
    yield_sec    말을 내려놓은 시점 — 사용자 목소리가 끝나기 전에 시작한 모델 마지막 말소리 구간의 끝 (목소리 시작 기준)
    overlap_sec  사용자 목소리 동안 모델 말소리 길이
    reply_sec    내려놓은 뒤 다시 낸 첫 말소리 (사용자 목소리 끝 기준)
    after_sec    사용자 목소리 끝부터 ``--watch`` 초 안의 모델 말소리 길이 (길면 스스로 이어 말했을 가능성)
위치는 받은 오디오 기준 초(WAV 위치와 같다). 입력·출력 시간축의 어긋남은 조건마다 같다.

    uv run python scripts/gpt_live/exp_barge_in.py --cond normal --tag b1          # 조건마다 프로세스 하나, 병렬로
    uv run python scripts/gpt_live/exp_barge_in.py --cond commentary_light --tag b1
    uv run python scripts/gpt_live/exp_barge_in.py --reanalyze b1   # 저장된 WAV 로 지표를 다시 계산 (지표를 바꿨을 때)
    uv run python scripts/gpt_live/exp_barge_in.py --compare b1

결과: var/experiments/barge_in/<번호>_<태그>/<조건>/ — results.json, pipeline.log, instructions.txt,
<종류>_<회차>.wav(왼쪽 사용자, 오른쪽 모델) + .labels.txt(Audacity 레이블: 사용자 발화, 모델 말소리 구간과 그때의 전사).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from exp_audio import read_model_channel, write_conversation_wav
from exp_dirs import find_result_dir, result_dir
from exp_dj_commentary_chunks import CHUNKS, LIVE_INSTRUCTIONS, PREAMBLES, tts24

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

COND_PREAMBLE = {
    "commentary_pre": "read",
    "commentary_read_talk": "read_talk",
    "commentary_read_noread": "read_noread",
    "commentary_read_own": "read_own",
    "commentary_light": "light",
    "commentary_read2": "read2",
    "commentary_say_wait": "say_wait",
    "commentary_say_stop": "say_stop",
    "commentary_say_stop_talk": "say_stop_talk",
    "commentary_next_line": "next_line",
    "commentary_listen": "listen",
}
CONDS = ("normal", "commentary", *COND_PREAMBLE)
KINDS = {"question": "너는 서랍에 뭐 넣어 둬?", "reaction": "아, 진짜?", "stop": "잠깐만"}
REQUEST = "서랍에 대한 이야기를 좀 길게 들려줘"
LONG_TEXT = " ".join(CHUNKS[:4])  # 약 25 s

BYTES_PER_SEC = BRIDGE_SAMPLE_RATE * 2
FRAME_SEC = 0.02
BIN_SEC = 0.1  # 판정 단위
SPEECH_RMS = 200
GAP_SEC = 0.5  # 이보다 짧은 말소리 사이 끊김은 한 구간으로 잇는다
STOP_WITHIN_SEC = 3.0  # 사용자 목소리 시작 뒤 이 안에 말을 내려놓지 않으면 "안 멈춤"
TRIM_PAD_SEC = 0.1  # --trim 때 목소리 앞뒤로 남기는 길이
END_GAP_SEC = 2.0  # 끼어들기 전 모델 말이 이만큼 끊기면 발화가 끝난 것 (무효)
START_DELAY_SEC = 1.0
PREAMBLE_SETTLE_SEC = 3.0
NO_TALK_SEC = 12.0  # 요청·주입 뒤 이만큼 말이 없으면 무효
MAX_SEC = 120.0

logger = logging.getLogger("exp_barge_in")


# ---------------------------------------------------------------------------
# 말소리 판정
# ---------------------------------------------------------------------------


def speech_bins(pcm: bytes) -> np.ndarray:
    """0.1 s 단위 말소리 여부."""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    hop = int(BIN_SEC * BRIDGE_SAMPLE_RATE)
    n = len(x) // hop
    if n == 0:
        return np.zeros(0, dtype=bool)
    return np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1)) >= SPEECH_RMS


def speech_segments(bins: np.ndarray) -> list[tuple[float, float]]:
    """말소리 구간 [(시작, 끝)] — ``GAP_SEC`` 미만의 끊김은 잇는다."""
    segs: list[list[float]] = []
    for i, on in enumerate(bins):
        t = i * BIN_SEC
        if not on:
            continue
        if segs and t - segs[-1][1] < GAP_SEC:
            segs[-1][1] = t + BIN_SEC
        else:
            segs.append([t, t + BIN_SEC])
    return [(a, b) for a, b in segs]


def user_clip(text: str, trim: bool) -> bytes:
    """사용자 TTS 클립. ``trim`` 이면 앞뒤 무음을 잘라 목소리 앞뒤 ``TRIM_PAD_SEC`` 만 남긴다
    (TTS 는 "잠깐만" 앞에 1.6 s 무음을 붙이기도 해 끼어드는 위치가 그만큼 밀린다)."""
    pcm = tts24(text)
    idx = np.flatnonzero(speech_bins(pcm))
    if not trim or not len(idx):
        return pcm
    bin_bytes = int(BIN_SEC * BYTES_PER_SEC)
    pad = int(TRIM_PAD_SEC * BYTES_PER_SEC)
    a = max(0, int(idx[0]) * bin_bytes - pad)
    b = min(len(pcm), (int(idx[-1]) + 1) * bin_bytes + pad)
    return pcm[a - a % 2 : b - b % 2]


def voiced_offsets(text: str, trim: bool = False) -> tuple[float, float]:
    """사용자 TTS 클립 안의 실제 목소리 구간 (클립 시작 기준 초)."""
    idx = np.flatnonzero(speech_bins(user_clip(text, trim)))
    return (float(idx[0] * BIN_SEC), float((idx[-1] + 1) * BIN_SEC)) if len(idx) else (0.0, 0.0)


def audio_metrics(pcm: bytes, v0: float, v1: float, watch: float) -> dict[str, Any]:
    """사용자 목소리 구간 [v0, v1) 에 대한 모델 반응 지표."""
    bins = speech_bins(pcm)
    times = np.arange(len(bins)) * BIN_SEC
    segs = speech_segments(bins)

    def speech_between(a: float, b: float) -> float:
        return round(float(bins[(times >= a) & (times < b)].sum() * BIN_SEC), 1)

    held = [s for s in segs if s[0] < v1 and s[1] > v0]
    yield_end = held[-1][1] if held else v0
    later = [s for s in segs if s[0] >= max(v1, yield_end)]
    reply = round(later[0][0] - v1, 1) if later and later[0][0] < v1 + watch else None
    return {
        "voice_start": round(v0, 2),
        "voice_end": round(v1, 2),
        "yield_sec": round(yield_end - v0, 1),
        "stopped": yield_end - v0 <= STOP_WITHIN_SEC,
        "overlap_sec": speech_between(v0, v1),
        "reply_sec": reply,
        "after_sec": speech_between(v1, v1 + watch),
    }


# ---------------------------------------------------------------------------
# 세션
# ---------------------------------------------------------------------------


class Session:
    """세션 하나의 진행과 기록. 위치는 받은 오디오 기준 초."""

    def __init__(self, cond: str, kind: str, at: float, watch: float, trim: bool = False) -> None:
        self.cond, self.kind, self.at, self.watch, self.trim = cond, kind, at, watch, trim
        self.live = GPTLiveSession(
            LiveSessionConfig(instructions=LIVE_INSTRUCTIONS, voice=LIVE_VOICE, sample_rate=BRIDGE_SAMPLE_RATE)
        )
        self.pcm = bytearray()
        self.user_pcm = bytearray()
        self.user_spans: list[tuple[float, float, str]] = []  # (시작, 끝, 문장) — 넣은 시각의 받은 오디오 위치
        self.deltas: list[tuple[float, str, str]] = []  # (도착 위치, 화자, 전사 조각)
        self.events: list[dict[str, Any]] = []
        self.trigger_pos: float | None = None  # 요청을 다 넣었거나 commentary 를 넣은 위치
        self.talk_start: float | None = None
        self.result = "running"

    def pos(self) -> float:
        return len(self.pcm) / BYTES_PER_SEC

    def say(self, text: str) -> None:
        speech = user_clip(text, self.trim)
        self.user_pcm.extend(speech)
        self.user_spans.append((self.pos(), self.pos() + len(speech) / BYTES_PER_SEC, text))

    def run(self) -> None:
        self.live.start()
        frame_bytes = int(BYTES_PER_SEC * FRAME_SEC)
        silence = bytes(frame_bytes)
        seg_text: dict[str, str] = {}
        t0 = time.monotonic()
        next_t = t0
        preamble_t: float | None = None
        last_speech: float | None = None  # 마지막 말소리 조각의 끝 위치
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
                        s = np.frombuffer(ev.pcm, dtype=np.int16).astype(np.float32)
                        if len(s) and np.sqrt(np.mean(s * s)) >= SPEECH_RMS:
                            last_speech = self.pos()
                            if self.trigger_pos is not None and self.talk_start is None and start >= self.trigger_pos:
                                self.talk_start = start
                    elif isinstance(ev, LiveTranscript):
                        old = seg_text.get(ev.segment_id, "")
                        delta = ev.text[len(old) :] if ev.text.startswith(old) else ev.text
                        seg_text[ev.segment_id] = ev.text
                        if delta.strip():
                            self.deltas.append((self.pos(), ev.speaker, delta))
                    elif isinstance(ev, LiveDelegationCreated):
                        self.events.append({"pos": round(self.pos(), 2), "name": "delegation"})
                    elif isinstance(ev, LiveError):
                        self.events.append({"pos": round(self.pos(), 2), "name": "error", "message": ev.message})
                    elif isinstance(ev, LiveClosed):
                        self.result = "closed"
                self._step(now - t0, last_speech, preamble_t)
                if self.cond in COND_PREAMBLE and preamble_t is None and now - t0 >= START_DELAY_SEC:
                    self.live.append_instructions(PREAMBLES[COND_PREAMBLE[self.cond]])
                    preamble_t = now - t0
                if now - t0 > MAX_SEC and self.result == "running":
                    self.result = "timeout"
                next_t += FRAME_SEC
                time.sleep(max(0.0, next_t - time.monotonic()))
        finally:
            self.live.close(graceful=True)

    def _step(self, elapsed: float, last_speech: float | None, preamble_t: float | None) -> None:
        p = self.pos()
        if self.trigger_pos is None:
            if self.cond == "normal":
                if not self.user_spans and elapsed >= START_DELAY_SEC:
                    self.say(REQUEST)
                elif self.user_spans and not self.user_pcm:
                    self.trigger_pos = p
            elif self.cond == "commentary" and elapsed >= START_DELAY_SEC:
                self.live.append_commentary(LONG_TEXT)
                self.trigger_pos = p
            elif preamble_t is not None and elapsed - preamble_t >= PREAMBLE_SETTLE_SEC:
                self.live.append_commentary(LONG_TEXT)
                self.trigger_pos = p
            return
        if self.talk_start is None:
            if p - self.trigger_pos >= NO_TALK_SEC:
                self.result = "no_talk"
            return
        if len(self.user_spans) < (2 if self.cond == "normal" else 1):
            if last_speech is not None and p - last_speech >= END_GAP_SEC:
                self.result = "too_short"  # 끼어들 시점 전에 말이 끝났다
            elif p - self.talk_start >= self.at:
                self.say(KINDS[self.kind])
            return
        a, b = voiced_offsets(KINDS[self.kind], self.trim)
        if p - (self.user_spans[-1][0] + b) >= self.watch:
            self.result = "done"

    def analyze(self) -> dict[str, Any]:
        row: dict[str, Any] = {"cond": self.cond, "kind": self.kind, "result": self.result, "events": self.events}
        if self.result != "done":
            return row
        u0, u1, _ = self.user_spans[-1]
        a, b = voiced_offsets(KINDS[self.kind], self.trim)
        row.update(
            {
                "talk_start": round(self.talk_start or 0.0, 2),
                "user_start": round(u0, 2),
                "user_end": round(u1, 2),
                **audio_metrics(bytes(self.pcm), u0 + a, u0 + b, self.watch),
                "said_before": "".join(d for p, sp, d in self.deltas if sp == "assistant" and p < u0 + a + 1.0).strip(),
                "said_after": "".join(d for p, sp, d in self.deltas if sp == "assistant" and p >= u0 + a + 1.0).strip(),
                "heard": "".join(d for p, sp, d in self.deltas if sp == "user" and p >= u0).strip(),
            }
        )
        return row

    def save(self, path: Path) -> None:
        clips = [(start, user_clip(text, self.trim)) for start, _, text in self.user_spans]
        write_conversation_wav(path, bytes(self.pcm), clips, BRIDGE_SAMPLE_RATE)
        labels = []
        for start, _, text in self.user_spans:
            a, b = voiced_offsets(text, self.trim)
            labels.append((start + a, start + b, f"USER: {text}"))
        for a, b in speech_segments(speech_bins(bytes(self.pcm))):  # 전사는 소리보다 늦게 오므로 끝 + 1 s 까지 붙인다
            text = "".join(d for p, sp, d in self.deltas if sp == "assistant" and a <= p < b + 1.0)
            labels.append((a, b, f"RAY: {text.strip()}"))
        labels.sort()
        path.with_suffix(".labels.txt").write_text("".join(f"{a:.3f}\t{b:.3f}\t{t}\n" for a, b, t in labels))


# ---------------------------------------------------------------------------
# 요약·실행
# ---------------------------------------------------------------------------


def _median(values: list[float]) -> float | None:
    return round(float(np.median(values)), 1) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for kind in KINDS:
        valid = [r for r in rows if r["kind"] == kind and r["result"] == "done"]
        replies = [r["reply_sec"] for r in valid if r["reply_sec"] is not None]
        out[kind] = {
            "valid": f"{len(valid)}/{sum(r['kind'] == kind for r in rows)}",
            "stopped": f"{sum(r['stopped'] for r in valid)}/{len(valid)}",
            "yield_median": _median([r["yield_sec"] for r in valid]),
            "overlap_median": _median([r["overlap_sec"] for r in valid]),
            "reply_median": _median(replies),
            "after_median": _median([r["after_sec"] for r in valid]),
        }
    return out


def print_row(r: dict[str, Any], n: int) -> None:
    if r["result"] != "done":
        print(f"[{r['cond']}/{r['kind']}#{n}] {r['result']}", flush=True)
        return
    print(
        f"[{r['cond']}/{r['kind']}#{n}] yield +{r['yield_sec']}s, overlap {r['overlap_sec']}s, "
        f"reply {r['reply_sec']}, after {r['after_sec']}s\n"
        f"    before: …{r['said_before'][-50:]}\n    after: {r['said_after']}",
        flush=True,
    )


def write_results(out_dir: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize(rows)
    (out_dir / "results.json").write_text(
        json.dumps({"config": config, "summary": summary, "results": rows}, ensure_ascii=False, indent=1)
    )
    return summary


def run(args: argparse.Namespace) -> None:
    out_dir = result_dir("barge_in", args.tag) / args.cond
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-28s %(levelname)-5s %(message)s",
        handlers=[logging.FileHandler(out_dir / "pipeline.log")],
    )
    (out_dir / "instructions.txt").write_text(LIVE_INSTRUCTIONS)
    for text in (REQUEST, *KINDS.values()):
        tts24(text)  # 캐시를 미리 채워 세션 중 합성 지연이 없게
    preamble = PREAMBLES[COND_PREAMBLE[args.cond]] if args.cond in COND_PREAMBLE else None
    config = {
        "cond": args.cond,
        "at": args.at,
        "watch": args.watch,
        "trim": args.trim,
        "kinds": KINDS,
        "request": REQUEST,
        "long_text": LONG_TEXT,
        "preamble": preamble,
    }
    rows: list[dict[str, Any]] = []
    for n in range(args.runs):
        for kind in args.kinds:
            session = Session(args.cond, kind, args.at, args.watch, args.trim)
            try:
                session.run()
            except RuntimeError as exc:
                rows.append({"run": n, "cond": args.cond, "kind": kind, "result": f"failed: {exc}", "events": []})
                print(f"[{args.cond}/{kind}#{n}] failed: {exc}", flush=True)
                continue
            row = {"run": n, **session.analyze()}
            rows.append(row)
            session.save(out_dir / f"{kind}_{n}.wav")
            print_row(row, n)
            write_results(out_dir, config, rows)
            time.sleep(1.5)
    print(json.dumps(write_results(out_dir, config, rows), ensure_ascii=False, indent=1))
    print(f"results: {out_dir}")


def reanalyze(tag: str) -> None:
    """저장된 WAV 로 지표를 다시 계산해 results.json 을 고쳐 쓴다 (전사 필드는 그대로)."""
    root = find_result_dir("barge_in", tag)
    if root is None:
        raise SystemExit(f"결과 폴더 없음: {tag}")
    for cond in CONDS:
        path = root / cond / "results.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        watch = data["config"]["watch"]
        for r in data["results"]:
            for stale in ("stop_sec", "reply_start_sec", "speech_after_sec"):
                r.pop(stale, None)
            if r["result"] != "done":
                continue
            pcm = read_model_channel(root / cond / f"{r['kind']}_{r['run']}.wav")
            a, b = voiced_offsets(KINDS[r["kind"]], data["config"].get("trim", False))
            r.update(audio_metrics(pcm, r["user_start"] + a, r["user_start"] + b, watch))
        write_results(root / cond, data["config"], data["results"])
        print(f"reanalyzed {cond}")


def compare(tag: str) -> None:
    root = find_result_dir("barge_in", tag)
    if root is None:
        raise SystemExit(f"결과 폴더 없음: {tag}")
    keys = ("valid", "stopped", "yield_median", "overlap_median", "reply_median", "after_median")
    conds = [c for c in CONDS if (root / c / "results.json").exists()]
    sums = {c: json.loads((root / c / "results.json").read_text()).get("summary", {}) for c in conds}
    for kind in KINDS:
        a, b = voiced_offsets(KINDS[kind])
        print(f"\n[{kind}] {KINDS[kind]} (목소리 {b - a:.1f}s)")
        print(f"  {'':16s}" + "".join(f"{c.removeprefix('commentary_'):>12s}" for c in conds))
        for k in keys:
            print(f"  {k:16s}" + "".join(f"{str(sums[c].get(kind, {}).get(k, '-')):>12s}" for c in conds))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cond", choices=CONDS)
    parser.add_argument("--kinds", nargs="*", choices=list(KINDS), default=list(KINDS))
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--at", type=float, default=4.0, help="모델 첫 말소리 뒤 이 초에 끼어든다")
    parser.add_argument("--watch", type=float, default=25.0, help="사용자 목소리가 끝난 뒤 지켜볼 초")
    parser.add_argument("--trim", action="store_true", help="사용자 TTS 클립 앞뒤 무음을 잘라 넣는다")
    parser.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--compare", metavar="TAG", help="조건별 요약을 나란히 보여주고 끝낸다")
    parser.add_argument("--reanalyze", metavar="TAG", help="저장된 WAV 로 지표를 다시 계산하고 끝낸다")
    args = parser.parse_args()
    if args.compare:
        compare(args.compare)
    elif args.reanalyze:
        reanalyze(args.reanalyze)
    elif args.cond is None:
        parser.error("--cond, --compare, --reanalyze 중 하나가 필요하다")
    else:
        run(args)


if __name__ == "__main__":
    main()
