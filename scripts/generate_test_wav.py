"""Generate WAV files with known text using OpenAI TTS for cross-module testing.

Requires OPENAI_API_KEY env var.

Usage:
    uv run python scripts/tools/generate_test_wav.py [--output-dir DIR]

Generates WAV files in the output directory (default: test_fixtures/).
Each file is named with a sanitised version of its transcript.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from voice_pipeline.core.config import TTSConfig
from voice_pipeline.tts.greeting_audio import synthesize_to_wav
from voice_pipeline.tts.tts import OpenAITTS

# Known phrases paired with their expected ASR transcription.
# Keep them short and phonetically clear for reliable round-trip testing.
PHRASES: list[dict[str, str]] = [
    {
        "text": "Hello, my name is Ray. I am a voice assistant.",
        "filename": "hello_ray",
    },
    {
        "text": "The quick brown fox jumps over the lazy dog.",
        "filename": "pangram",
    },
    {
        "text": "Please turn on the lights in the living room.",
        "filename": "command_lights",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate test WAV files via OpenAI TTS")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_fixtures"),
        help="Directory to write WAV files (default: test_fixtures/)",
    )
    parser.add_argument(
        "--voice",
        default="alloy",
        help="OpenAI TTS voice (default: alloy)",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = TTSConfig(voice=args.voice, model="tts-1")
    tts = OpenAITTS(config)

    for phrase in PHRASES:
        out_path = output_dir / f"{phrase['filename']}.wav"
        print(f'Generating: {out_path}  ←  "{phrase["text"]}"')
        synthesize_to_wav(tts, phrase["text"], out_path, config.output_sample_rate)
        print(f"  ✓ saved ({out_path.stat().st_size:,} bytes)")

    # Write a manifest so tests can read phrase<->file mapping
    manifest_path = output_dir / "manifest.txt"
    with manifest_path.open("w") as f:
        for phrase in PHRASES:
            f.write(f"{phrase['filename']}.wav\t{phrase['text']}\n")
    print(f"\nManifest written to {manifest_path}")
    print("Done.")


if __name__ == "__main__":
    main()
