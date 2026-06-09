"""Generate WAV files for eval questions using OpenAI TTS.

Assigns voices via round-robin from VOICES. Multi-turn scenarios use
one voice per scenario (all questions in the same scenario share a voice).

Usage:
    uv run python scripts/eval/prepare_audio.py data/eval/questions.json
    uv run python scripts/eval/prepare_audio.py data/eval/questions.json --output-dir data/eval/wav
    uv run python scripts/eval/prepare_audio.py data/eval/questions.json --speed 1.5 --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from voice_pipeline.tts.greeting_audio import synthesize_to_wav
from voice_pipeline.tts.tts import OpenAITTS

VOICES = [
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "onyx",
    "nova",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
]


def _get_tts(voice: str, cache: dict[str, OpenAITTS]) -> OpenAITTS:
    if voice not in cache:
        tts = OpenAITTS()
        tts._VOICE = voice
        cache[voice] = tts
    return cache[voice]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate eval question WAV files")
    parser.add_argument("questions", help="Path to questions JSON")
    parser.add_argument("--output-dir", default="data/eval/wav", help="Output directory")
    parser.add_argument("--model", default="gpt-4o-mini-tts")
    parser.add_argument("--speed", type=float, default=1.2)
    parser.add_argument("--force", action="store_true", help="Regenerate existing files")
    args = parser.parse_args()

    OpenAITTS._MODEL = args.model
    OpenAITTS._SPEED = args.speed

    data = json.loads(Path(args.questions).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tts_cache: dict[str, OpenAITTS] = {}
    manifest: dict[str, dict[str, str]] = {}
    generated = 0
    total = 0
    voice_idx = 0

    print(f"Preparing question WAVs → {output_dir}")
    print(f"  voices: {len(VOICES)} | speed: {args.speed}")

    for suite in data["suites"]:
        if suite.get("multi_turn"):
            for scenario in suite.get("scenarios", []):
                voice = VOICES[voice_idx % len(VOICES)]
                voice_idx += 1
                for q in scenario["questions"]:
                    total += 1
                    wav_path = output_dir / f"{q['id']}_{voice}.wav"
                    manifest[q["id"]] = {"path": str(wav_path), "voice": voice}

                    if wav_path.exists() and not args.force:
                        print(f"  skip (exists): {q['id']} [{voice}]")
                        continue

                    print(f"  generating: {q['id']} [{voice}] — {q['text'][:60]}")
                    synthesize_to_wav(_get_tts(voice, tts_cache), q["text"], wav_path)
                    generated += 1
                    print(f"    saved: {wav_path} ({wav_path.stat().st_size:,} bytes)")
        else:
            for q in suite.get("questions", []):
                total += 1
                voice = VOICES[voice_idx % len(VOICES)]
                voice_idx += 1
                wav_path = output_dir / f"{q['id']}_{voice}.wav"
                manifest[q["id"]] = {"path": str(wav_path), "voice": voice}

                if wav_path.exists() and not args.force:
                    print(f"  skip (exists): {q['id']} [{voice}]")
                    continue

                print(f"  generating: {q['id']} [{voice}] — {q['text'][:60]}")
                synthesize_to_wav(_get_tts(voice, tts_cache), q["text"], wav_path)
                generated += 1
                print(f"    saved: {wav_path} ({wav_path.stat().st_size:,} bytes)")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"\nDone: {generated} generated, {total - generated} skipped")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
