"""Generate WAV files for eval questions using OpenAI TTS.

Usage:
    uv run python scripts/eval/prepare_audio.py data/eval/questions.json
    uv run python scripts/eval/prepare_audio.py data/eval/questions.json --output-dir data/eval/wav
    uv run python scripts/eval/prepare_audio.py data/eval/questions.json --voice coral --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from voice_pipeline.tts.greeting_audio import synthesize_to_wav
from voice_pipeline.tts.tts import OpenAITTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate eval question WAV files")
    parser.add_argument("questions", help="Path to questions JSON")
    parser.add_argument("--output-dir", default="data/eval/wav", help="Output directory")
    parser.add_argument("--voice", default="ash")
    parser.add_argument("--model", default="gpt-4o-mini-tts")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--force", action="store_true", help="Regenerate existing files")
    args = parser.parse_args()

    OpenAITTS._VOICE = args.voice
    OpenAITTS._MODEL = args.model
    OpenAITTS._SPEED = args.speed
    tts = OpenAITTS()

    data = json.loads(Path(args.questions).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _iter_questions(suite: dict):
        if suite.get("multi_turn"):
            for scenario in suite.get("scenarios", []):
                yield from scenario["questions"]
        else:
            yield from suite.get("questions", [])

    manifest: dict[str, str] = {}
    total = sum(1 for s in data["suites"] for _ in _iter_questions(s))
    generated = 0

    print(f"Preparing {total} question WAVs → {output_dir}")

    for suite in data["suites"]:
        for q in _iter_questions(suite):
            wav_path = output_dir / f"{q['id']}.wav"
            manifest[q["id"]] = str(wav_path)

            if wav_path.exists() and not args.force:
                print(f"  skip (exists): {q['id']}")
                continue

            print(f"  generating: {q['id']} — {q['text'][:60]}")
            synthesize_to_wav(tts, q["text"], wav_path)
            generated += 1
            print(f"    saved: {wav_path} ({wav_path.stat().st_size:,} bytes)")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"\nDone: {generated} generated, {total - generated} skipped")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
