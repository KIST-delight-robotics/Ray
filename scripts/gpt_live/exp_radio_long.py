"""긴 대본 라디오 실험 — 소재 여럿을 끼어들기 없이 끝까지 읽고, 곡을 지나 다음 소재로 넘어간다.

``exp_radio_ahead`` 의 B′ 규칙(넣기 시점·진행 추적·남은 문장 다시 넣기·운영 순서 시작)을 그대로 쓰고, 대본만
``radio_scripts/<소재>.txt``(한 줄에 대사 하나)로 바꾼다. 소재의 마지막 대사(곡 소개)가 전사로 끝나고 모델이
``--closing-quiet`` 조용하면 곡을 시작한다 — 운영은 곡을 틀 때 스트림을 닫으니 곡 소개 뒤 덧붙임은 잘린다. 곡 소개가
끊기면 ``--stall sentence`` 가 끊긴 문장을 다시 넣고, 그래도 조용하면 곡을 시작한다. 곡 동안(``--song-sec``)은
대사를 넣지 않고 입력은 무음이다(운영에선 마이크 원본으로 곡 소리가 들어간다). 그동안 모델이 한 말은 운영이면 스트림이
닫혀 버려지는 말이라 기록만 한다. 곡이 끝나면 다음 소재의 첫 대사를 넣는다. ``--next-start signal`` 이면 운영처럼
instructions.append(``SONG_ENDED_SIGNAL``)로도 깨운다. 끼어들기는 없다.

    uv run python scripts/gpt_live/exp_radio_long.py --tag g1 --index 0
    uv run python scripts/gpt_live/exp_radio_long.py --report g1

결과: var/experiments/radio_long/<번호>_<태그>/ — run_<번호>.json, run_<번호>.wav(왼쪽 사용자, 오른쪽 모델)
+ .labels.txt, config.json, code/(이 결과를 낸 스크립트).
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from exp_dirs import find_result_dir, result_dir
from exp_dj_commentary_chunks import tts24
from exp_radio_ahead import (
    ACK_MATCH,
    DEFINED_INSTRUCTIONS,
    DONE_SLACK_LETTERS,
    FRAME_SEC,
    LINE_TEMPLATE,
    MATCH_LETTERS,
    PLACES,
    REPUSH_WAIT_SEC,
    STALL_SEC,
    START_ASK,
    START_BACKEND_INSTRUCTIONS,
    AheadSession,
    _letters,
    _timing_rows,
    script_mask,
)
from exp_radio_flow import UTTERANCES, _voiced
from exp_radio_tts import sentences

from voice_pipeline.engines.gpt_live.instructions import LIVE_VOICE

SCRIPTS_DIR = Path(__file__).resolve().parent / "radio_scripts"
CLOSING_FALLBACK_SEC = 4.0  # 마지막 대사까지 넣었는데 전사로 끝을 못 찾으면, 말한 뒤 이만큼 조용할 때 곡
CLOSING_CAP_SEC = 60.0  # 마지막 대사를 넣고 이만큼 지나면 그냥 곡 (침묵·추적 실패 안전장치)
SONG_ENDED_SIGNAL = "Radio talk: the song has ended. Go on with your talk."
MAX_RUN_SEC = 900.0
PAUSE_LONG_SEC = 1.5  # 낭독 중 이보다 긴 쉼을 센다
PLACE_MARK = {"early": "전", "in_last": "·", "late": "뒤", "unknown": "?"}  # 회차 줄의 넣기 위치 (· = 마지막 문장 중)
DUMMY_UTTERANCE = "react_ok"  # Session 이 요구하는 끼어들기 발화 — 이 실험에선 쓰지 않는다

logger = logging.getLogger("exp_radio_long")


def load_script(name: str) -> list[str]:
    return [line.strip() for line in (SCRIPTS_DIR / f"{name}.txt").read_text().splitlines() if line.strip()]


class LongSession(AheadSession):
    max_sec = MAX_RUN_SEC

    def __init__(self, segments: list[str], index: int, args: argparse.Namespace) -> None:
        super().__init__(DUMMY_UTTERANCE, index, args, None, None)
        self.segments = segments
        units: list[tuple[int, str]] = []
        self.seg_first: list[int] = []  # 소재마다 첫 단위 번호 (끝에 전체 단위 수)
        for s, name in enumerate(segments):
            self.seg_first.append(len(units))
            units += [(s, line) for line in load_script(name)]
        self.seg_first.append(len(units))
        self.set_units(units)
        self.seg = 0
        self.phase = "start"  # start → reading → song → reading … → done
        self.songs: list[dict[str, Any]] = []  # 소재마다 {"seg", "start", "end", "by"}

    def _ready_for_next(self) -> bool:
        if self.phase != "reading":
            return False
        if len(self.appends) >= self.seg_first[self.seg + 1]:
            # 소재 경계 — 다음 소재는 곡이 끝난 뒤. 곡 소개가 끊겼으면(--stall sentence) 다시 넣기만 한다
            if self.args.stall == "sentence" and self.first_voice is not None and self.quiet >= STALL_SEC:
                last = len(self.appends) - 1
                unread = self._unread(last, self._progress()["reach"])
                if unread and self.repushed != last:
                    self._repush_text(last, unread)
            return False
        return super()._ready_for_next()

    def _start_song(self, by: str) -> None:
        self.songs.append({"seg": self.seg, "start": round(self.pos(), 2), "by": by})
        self._event("song_start", seg=self.seg, by=by)
        self.phase = "song"

    def _end_song(self) -> None:
        self.songs[-1]["end"] = round(self.pos(), 2)
        self._event("song_end", seg=self.seg)
        self.seg += 1
        if self.seg >= len(self.segments):
            self.result = "done"
            return
        first = self.seg_first[self.seg]
        self.done = max(self.done, first)  # 앞 소재는 끝난 것으로
        self.track_from, self.track_unit = self.pos(), first  # 새 소재부터 따라간다
        self.push_by = "segment"
        self._append()
        if self.args.next_start == "signal":
            self.live.append_instructions(SONG_ENDED_SIGNAL)
            self._event("signal")
        self.phase = "reading"

    def _step(self, since_preamble: float) -> None:
        p = self.pos()
        self._update_done()
        if self.phase == "start":
            if not self.appends:
                super()._step(since_preamble)  # 시작 요청과 진입 실패 판정
                return
            self.phase = "reading"
        if self.phase == "reading":
            if self._ready_for_next():
                self._append()
            end = self.seg_first[self.seg + 1]
            if self.done >= end and self.quiet >= self.args.closing_quiet:
                self._start_song("closing")
            elif len(self.appends) >= end:
                waited = self.repushed != end - 1 or p - self.repushed_at >= REPUSH_WAIT_SEC
                spoke = self.first_voice is not None and self.quiet >= CLOSING_FALLBACK_SEC and waited
                if spoke or p - self.appends[-1] >= CLOSING_CAP_SEC:
                    self._start_song("quiet" if spoke else "cap")
        elif self.phase == "song" and p - self.songs[-1]["start"] >= self.args.song_sec:
            self._end_song()

    def raw(self) -> dict[str, Any]:
        return {
            **super().raw(),
            "segments": self.segments,
            "seg_first": self.seg_first,
            "songs": self.songs,
            "next_start": self.args.next_start,
        }


# ---------------------------------------------------------------------------
# 분석
# ---------------------------------------------------------------------------


def _assistant(raw: dict[str, Any], a: float, b: float) -> list[tuple[float, str]]:
    return [(p, d) for p, sp, d in raw["deltas"] if sp == "assistant" and a <= p < b]


def _first_match(text: str, script: str) -> int:
    """``text`` 에서 대본과 ``ACK_MATCH`` 넘게 겹치는 첫 글자 자리 (없으면 끝)."""
    mask = script_mask(text, script, ACK_MATCH)
    return next((i for i, m in enumerate(mask) if m), len(text))


def analyze(raw: dict[str, Any], pcm: bytes) -> dict[str, Any]:
    units = [t for _, t in raw["units"]]
    first, songs, appends = raw["seg_first"], raw["songs"], raw["appends"]
    voiced = _voiced(pcm)
    segs = []
    for s in range(len(raw["segments"])):
        u0, u1 = first[s], first[s + 1]
        if len(appends) <= u0:
            break
        t0 = appends[u0]
        t1 = songs[s]["start"] if s < len(songs) else float("inf")
        script_lines = units[u0:u1]
        script = " ".join(script_lines)
        spoken = "".join(d for _, d in _assistant(raw, t0, t1))
        k = _first_match(spoken, script)
        lead, reading = spoken[:k].strip(), spoken[k:]
        ask = raw.get("start_ask")
        if s == 0 and ask:  # 첫 소재의 첫머리는 시작 요청이 끝난 뒤부터 — 한마디가 첫 대사를 넣기 전에 시작된다
            before = "".join(d for _, d in _assistant(raw, ask["start"] + ask["voice"][1], t0))
            lead = (before + spoken[:k]).strip()
        letters = _letters(reading)
        sents = [x for line in script_lines for x in sentences(line)]
        hits = [i for i, x in enumerate(sents) if sum(script_mask(_letters(x), letters)) >= 0.6 * len(_letters(x))]
        upto = max(hits) if hits else -1
        row: dict[str, Any] = {
            "seg": s,
            "name": raw["segments"][s],
            "lead": lead,
            "sentences": len(sents),
            "read_upto": upto + 1,
            "skipped": [i + 1 for i in range(upto + 1) if i not in hits],
            "off_script": round(sum(not m for m in script_mask(letters, _letters(script))) / max(1, len(letters)), 3),
            "timing": _timing_rows(raw, t0, t1, u0, list(range(u0 + 1, min(u1, len(appends)))), f"seg{s}"),
            "repush": [e["unit"] for e in raw["events"] if e["name"] == "repush" and t0 <= e["pos"] < t1],
            "stall": [
                e["unit"]
                for e in raw["events"]
                if e["name"] == "push" and e.get("by") == "stall" and t0 <= e["pos"] < t1
            ],
        }
        # 첫 대사: 넣은 뒤 15 s 안에 거의 그대로 (2자 이내 차이)
        line1 = _letters(units[u0])
        said1 = _letters("".join(d for _, d in _assistant(raw, t0, t0 + 15.0)))
        got = sum(
            b.size
            for b in difflib.SequenceMatcher(None, said1, line1, autojunk=False).get_matching_blocks()
            if b.size >= MATCH_LETTERS
        )
        row["first_line_ok"] = got >= len(line1) - 2
        # 곡 소개(마지막 대사) 끝 → 곡 시작, 그 사이 덧붙인 말
        last = _letters(units[u1 - 1])
        deltas = _assistant(raw, t0, t1)
        text, at = "", []
        for p, d in deltas:
            for ch in d:
                if _letters(ch):
                    text += ch
                    at.append(p)
        blocks = [
            b for b in difflib.SequenceMatcher(None, text, last, autojunk=False).get_matching_blocks() if b.size >= 6
        ]
        if blocks and blocks[-1].b + blocks[-1].size >= len(last) - DONE_SLACK_LETTERS:
            end_i = blocks[-1].a + blocks[-1].size
            row["closing_end"] = round(at[end_i - 1], 2)
            row["after_closing"] = text[end_i:]
            if s < len(songs):
                row["closing_to_song"] = round(songs[s]["start"] - at[end_i - 1], 2)
        # 낭독 중 쉼 (대본과 처음 겹친 글자 ~ 곡 소개 끝, rms 30) — 시작 한마디와 그 뒤 틈은 뺀다
        lead_letters = len(_letters(spoken[:k]))
        if lead_letters < len(at):
            a, b = at[lead_letters], row.get("closing_end", at[-1])
            seg_v = voiced[int(a / FRAME_SEC) : int(b / FRAME_SEC)]
            gaps, cur = [], 0
            for v in seg_v:
                if v:
                    if cur:
                        gaps.append(cur * FRAME_SEC)
                    cur = 0
                else:
                    cur += 1
            row["pause_max"] = round(max(gaps, default=0.0), 2)
            row["pauses_long"] = sum(g > PAUSE_LONG_SEC for g in gaps)
            row["reading_sec"] = round(b - a, 1)
        if s < len(songs):
            row["song_by"] = songs[s]["by"]
            end = songs[s].get("end", float("inf"))
            row["song_speech"] = "".join(d for _, d in _assistant(raw, songs[s]["start"], end)).strip()
            if s + 1 < len(raw["segments"]) and len(appends) > first[s + 1]:
                nxt = _assistant(raw, songs[s]["end"], float("inf"))
                row["next_voice"] = round(nxt[0][0] - songs[s]["end"], 2) if nxt else None
        segs.append(row)
    return {"segs": segs}


def format_run(r: dict[str, Any]) -> list[str]:
    head = f"#{r['index']} {r['result']}"
    if not r.get("segs"):
        return [head]
    lines = [head]
    for g in r["segs"]:
        places = "".join(PLACE_MARK[t["place"]] for t in g["timing"])
        parts = [
            f"  [{g['name']}] {g['read_upto']}/{g['sentences']}문장",
            f"건너뜀 {g['skipped'] or '-'}",
            f"대사 밖 {g['off_script']:.0%}",
            "첫 대사 " + ("그대로" if g["first_line_ok"] else "다름"),
            f"쉼 최대 {g.get('pause_max', '-')}s(1.5s↑ {g.get('pauses_long', '-')})",
            f"낭독 {g.get('reading_sec', '-')}s",
        ]
        if g["repush"] or g["stall"]:
            parts.append(f"다시 넣음 {g['repush']} 무음 넣기 {g['stall']}")
        lines.append(", ".join(parts))
        lines.append(f"     넣기 {places or '-'} | 첫머리「{g['lead'][:40]}」")
        if "song_by" in g:
            extra = f"곡 소개 끝 → 곡 {g.get('closing_to_song', '-')}s, 덧붙임「{g.get('after_closing', '')[:30]}」"
            lines.append(f"     곡({g['song_by']}): {extra}, 곡 중 말「{g['song_speech'][:50]}」")
            if "next_voice" in g:
                lines.append(f"     곡 끝 → 다음 소재 첫 음성 {g['next_voice']}s")
    return lines


def report(tag: str) -> None:
    root = find_result_dir("radio_long", tag)
    if root is None:
        raise SystemExit(f"결과 폴더 없음: {tag}")
    rows = [json.loads(p.read_text())["result"] for p in sorted(root.glob("run_*.json"))]
    lines: list[str] = []
    for r in rows:
        lines.extend(format_run(r))
    done = [r for r in rows if r.get("segs")]
    lines.append(
        f"\n전체 {len(rows)}회: "
        + ", ".join(f"{k} {sum(r['result'] == k for r in rows)}" for k in sorted({r["result"] for r in rows}))
    )
    nseg = max((len(r["segs"]) for r in done), default=0)
    for s in range(nseg):
        gs = [r["segs"][s] for r in done if len(r["segs"]) > s]
        timing = [t for g in gs for t in g["timing"]]
        leads = [t["lead"] for t in timing if "lead" in t]
        lines.append(f"  [{gs[0]['name']}] {len(gs)}회")
        full = sum(g["read_upto"] == g["sentences"] for g in gs)
        skipped = sum(bool(g["skipped"]) for g in gs)
        off = sum(g["off_script"] > 0.1 for g in gs)
        first_ok = sum(g["first_line_ok"] for g in gs)
        lines.append(
            f"    끝까지 읽음 {full}, 건너뛴 회차 {skipped}, 대사 밖 10% 넘음 {off}, 첫 대사 그대로 {first_ok}"
        )
        place = ", ".join(f"{PLACES[k]} {sum(t['place'] == k for t in timing)}" for k in PLACES)
        if leads:
            place += f" | 여유 중앙 {float(np.median(leads)):.1f}s (1 s 미만 {sum(x < 1 for x in leads)})"
        lines.append("    넣기: " + place)
        pm = [g["pause_max"] for g in gs if "pause_max" in g]
        rd = [g["reading_sec"] for g in gs if "reading_sec" in g]
        long_pause = sum(g.get("pauses_long", 0) > 0 for g in gs)
        repush = sum(len(g["repush"]) for g in gs)
        stall = sum(len(g["stall"]) for g in gs)
        pm_med = float(np.median(pm)) if pm else float("nan")
        rd_med = float(np.median(rd)) if rd else float("nan")
        lines.append(
            f"    쉼 최대 중앙 {pm_med:.2f}s, 1.5 s 넘는 쉼이 있는 회차 {long_pause}, "
            f"다시 넣음 {repush}, 무음 넣기 {stall}, 낭독 중앙 {rd_med:.0f}s"
        )
        lines.append("    첫머리: " + " / ".join(f"「{g['lead'][:30]}」" for g in gs))
        sung = [g for g in gs if "song_by" in g]
        if sung:
            c2s = [g["closing_to_song"] for g in sung if "closing_to_song" in g]
            by_closing = sum(g["song_by"] == "closing" for g in sung)
            added = sum(len(g.get("after_closing", "")) > 2 for g in sung)
            talked = sum(bool(g["song_speech"]) for g in sung)
            c2s_med = float(np.median(c2s)) if c2s else float("nan")
            lines.append(
                f"    곡: 곡 소개로 시작 {by_closing}/{len(sung)}, 곡 소개 끝 → 곡 중앙 {c2s_med:.1f}s, "
                f"덧붙임 {added}, 곡 중 말한 회차 {talked}"
            )
            said = [g["song_speech"][:40] for g in sung if g["song_speech"]]
            if said:
                lines.append("    곡 중 말: " + " / ".join(f"「{x}」" for x in said))
            nv = [g["next_voice"] for g in sung if g.get("next_voice") is not None]
            if nv:
                lines.append(f"    곡 끝 → 다음 소재 첫 음성 중앙 {float(np.median(nv)):.1f}s ({min(nv)}~{max(nv)})")
    out = "\n".join(lines)
    print(out)
    (root / "report.txt").write_text(out + "\n")


def build_args(ns: argparse.Namespace) -> argparse.Namespace:
    """AheadSession 이 읽는 옵션 — l11 후보 설정, 끼어들기 없음."""
    return argparse.Namespace(
        **vars(ns),
        unit="line",
        definition=True,
        start="request",
        pause_note="explicit",
        watch_end=True,
        resume_ask="alt",
        interrupt="mid",
        chunk=99,
        at=2.0,
        gap_at=0.2,
        gap=1.0,
        midline_wait=1.5,
        watch=20.0,
    )


def run(args: argparse.Namespace) -> None:
    out_dir = result_dir("radio_long", args.tag)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-28s %(levelname)-5s %(message)s",
        handlers=[logging.FileHandler(out_dir / f"pipeline_{os.getpid()}.log")],
    )
    code_dir = out_dir / "code"
    code_dir.mkdir(exist_ok=True)
    for name in ("exp_radio_long.py", "exp_radio_ahead.py", "exp_radio_flow.py"):
        shutil.copy(Path(__file__).with_name(name), code_dir / name)
    sargs = build_args(args)
    config = {
        "segments": args.segments,
        "scripts": {name: load_script(name) for name in args.segments},
        "song_sec": args.song_sec,
        "next_start": args.next_start,
        "song_ended_signal": SONG_ENDED_SIGNAL if args.next_start == "signal" else None,
        "closing_quiet": args.closing_quiet,
        "stall": args.stall,
        "session_args": {k: v for k, v in vars(sargs).items() if k not in ("tag", "report")},
        "line_template": LINE_TEMPLATE,
        "start_ask": START_ASK,
        "voice": LIVE_VOICE,
        "instructions": DEFINED_INSTRUCTIONS,
        "backend_instructions": START_BACKEND_INSTRUCTIONS,
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1))
    tts24(START_ASK)
    tts24(UTTERANCES[DUMMY_UTTERANCE][1])
    session = LongSession(args.segments, args.index, sargs)
    try:
        session.run()
    except RuntimeError as exc:
        row = {"index": args.index, "result": f"failed: {exc}"}
    else:
        raw = session.raw()
        row = {"index": args.index, **raw, **analyze(raw, bytes(session.pcm))}
        session.save(out_dir / f"run_{args.index}.wav")
    (out_dir / f"run_{args.index}.json").write_text(
        json.dumps({"config": config, "result": row}, ensure_ascii=False, indent=1)
    )
    print("\n".join(format_run(row)), flush=True)
    print(f"results: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segments", nargs="+", default=["drawer", "end_of_day"])
    parser.add_argument("--song-sec", type=float, default=15.0)
    parser.add_argument("--stall", choices=("remainder", "sentence"), default="sentence", help="말하고 조용해졌을 때")
    # 곡 소개를 끝까지 읽고 이만큼 조용하면 곡. 운영은 곡을 틀 때 스트림을 닫으니 덧붙임은 잘린다(long_s 의 10 s 덧붙임)
    parser.add_argument("--closing-quiet", type=float, default=0.5)
    parser.add_argument("--next-start", choices=("thinking", "signal"), default="thinking", help="곡 뒤 다음 소재 시작")
    parser.add_argument("--index", type=int, default=0, help="회차 번호 (병렬 실행 때 프로세스마다 다르게)")
    parser.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--report", metavar="TAG", help="모아 보여주고 끝낸다")
    args = parser.parse_args()
    if args.report:
        report(args.report)
    else:
        run(args)


if __name__ == "__main__":
    main()
