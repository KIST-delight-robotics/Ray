#!/usr/bin/env python3
"""Synthesize phonetic stress-test + natural WAVs for mouth-trajectory evaluation.

Each utterance targets a specific behavior of an amplitude-driven single-DOF jaw,
so the trajectory can be judged against a known phonetic expectation.

Reuses scripts/tts_to_file.py (no pipeline changes). Outputs to motion_eval/mouth/wavs/.

Usage:
    uv run python motion_eval/mouth/gen_wavs.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WAVDIR = Path(__file__).resolve().parent / "wavs"

# name -> (text, phonetic expectation for the jaw)
UTTERANCES: dict[str, tuple[str, str]] = {
    "bilabial_closure": (
        "Mama, papa, baby, bubble. Maybe my mommy will be home by Monday.",
        "양순음 /m,b,p/ 다수 → 입이 자주 닫혀야 함 (진폭기반의 약점 정조준)",
    ),
    "vowel_alternation": (
        "Ah, ee, ah, ee, ah, ee. Oo, ah, oo, ah, oo, ah. Ee, oo, ee, oo, ee, oo.",
        "개모음↔폐모음 교대 → 개구량이 규칙적으로 진동해야 함",
    ),
    "sustained_vowel": (
        "Aaaaaaah. Ooooooooh. Eeeeeeeee. Aaaaaaaah.",
        "지속 모음 → 입을 연 채 유지(떨림 없이), 닫혔다 열렸다 하면 안 됨",
    ),
    "plosive_bursts": (
        "Pa! Ta! Ka! Pa! Ta! Ka! Ba! Da! Ga! Ba! Da! Ga!",
        "파열음 연발 → 또렷한 개폐 버스트, 빠른 추종",
    ),
    "silence_gaps": (
        "One. Two. Three. Four. Five. Six. Seven. Eight.",
        "단어 사이 묵음 → 묵음에서 닫힘 (과검출/파닥임 점검)",
    ),
    "soft_to_loud": (
        "Shhh, very quietly now, whisper softly. SUDDENLY, LOUDLY, STOP RIGHT NOW!",
        "약→강 다이내믹 → 작을 땐 작게, 클 땐 크게 (포화 점검)",
    ),
    "fast_counting": (
        "One two three four five six seven eight nine ten eleven twelve.",
        "빠른 음절률 → 변조 스펙트럼 고역, 음절 병합 없이 따라가는가",
    ),
    "natural_sentence": (
        "Hello, my name is Ray, and it's really nice to meet you today.",
        "자연문 기준선 (baseline)",
    ),
}


def main() -> None:
    WAVDIR.mkdir(parents=True, exist_ok=True)
    tts = ROOT / "scripts" / "tts_to_file.py"
    out_tmp = ROOT / "output"

    for name, (text, expect) in UTTERANCES.items():
        tmp_name = f"evalmouth_{name}"
        print(f"\n=== {name} ===\n  기대: {expect}")
        subprocess.run(
            ["uv", "run", "python", str(tts), text, "--vendor", "elevenlabs", "-n", tmp_name],
            cwd=ROOT,
            check=True,
        )
        src = out_tmp / f"{tmp_name}.wav"
        dst = WAVDIR / f"{name}.wav"
        shutil.move(str(src), str(dst))
        print(f"  -> {dst.relative_to(ROOT)}")

    # 기준 비교용으로 대표 발화체(조용/큰/다이내믹) 몇 개 복사
    for ref in ("tts_bladerunner_tears", "tts_300_leonidas", "tts_dynamic_mouth"):
        src = out_tmp / f"{ref}.wav"
        if src.exists():
            shutil.copy(str(src), str(WAVDIR / f"ref_{ref}.wav"))
            print(f"copied ref: ref_{ref}.wav")

    print(f"\n완료: {WAVDIR}")
    for w in sorted(WAVDIR.glob("*.wav")):
        print(" ", w.name)


if __name__ == "__main__":
    sys.exit(main())
