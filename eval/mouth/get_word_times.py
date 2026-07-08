#!/usr/bin/env python3
"""Fetch ElevenLabs word-level timestamps for each eval utterance.

Saves eval/mouth/text/<name>.json = {"text":..., "words":[{word,start,end}]}.
analyze.py reads these and places each word as a subtitle snapped to the
nearest detected syllable nucleus.

Note: re-synthesizes with the same voice/model only to read timestamps; the
audio is discarded (the robot already played the captured WAV). Word timing
is near-deterministic; subtitles are snapped to nuclei so minor drift self-corrects.

Usage:
    uv run python eval/mouth/get_word_times.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
TXTDIR = Path(__file__).resolve().parent / "text"

from eval.mouth.gen_wavs import UTTERANCES  # noqa: E402
from voice_pipeline.tts.elevenlabs_tts import ElevenLabsTTS  # noqa: E402

# reference (movie/speech) utterances used in the eval set
REFS = {
    "ref_tts_bladerunner_tears": (
        "I've seen things you people wouldn't believe. Attack ships on fire off the "
        "shoulder of Orion. I watched C-beams glitter in the dark near the Tannhauser "
        "Gate. All those moments will be lost in time, like tears in rain. Time to die."
    ),
    "ref_tts_300_leonidas": (
        "Spartans! Ready your breakfast and eat hearty, for tonight, we dine in Hell! "
        "This is where we hold them. This is where we fight, and this is where they die! "
        "Remember this day, men, for it will be yours for all time. No retreat. No "
        "surrender. That is Spartan law. Give them nothing, but take from them everything! "
        "A new age has begun, an age of freedom, and all will know that three hundred "
        "Spartans gave their last breath to defend it!"
    ),
    "ref_tts_dynamic_mouth": (
        "Hello! Open wide — say AAAH, OHH, WOW! Now soft... shhh, whisper quietly. WAIT! "
        "STOP! Hear that BOOMING ROAR? Mmm, mouth opens, ahh, closes — pop, pop, bop! "
        "Loud proud crowd. Slowly... quiet. Goodbye!"
    ),
}


def main() -> None:
    TXTDIR.mkdir(parents=True, exist_ok=True)
    texts = {name: text for name, (text, _exp) in UTTERANCES.items()}
    texts.update(REFS)

    tts = ElevenLabsTTS()
    for name, text in texts.items():
        stream = tts.synthesize(text)
        for _ in stream:  # consume to populate timestamps
            pass
        words = [
            {"word": w.word, "start": round(w.start_sec, 3), "end": round(w.end_sec, 3)}
            for w in stream.timestamps
        ]
        out = TXTDIR / f"{name}.json"
        out.write_text(json.dumps({"text": text, "words": words}, ensure_ascii=False), encoding="utf-8")
        print(f"{name:28} {len(words):3d} words -> {out.relative_to(ROOT)}")

    print(f"\n완료: {TXTDIR}")


if __name__ == "__main__":
    main()
