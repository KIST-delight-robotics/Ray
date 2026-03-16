"""Synthesize text to a WAV file using OpenAI TTS.

Requires OPENAI_API_KEY env var.

Usage:
    uv run python scripts/tts_to_file.py "안녕하세요, 반갑습니다."
    uv run python scripts/tts_to_file.py "Hello world" -n hello
    uv run python scripts/tts_to_file.py "Hello world" --voice coral --model gpt-4o-mini-tts
    echo "Hello" | uv run python scripts/tts_to_file.py -  # read from stdin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_pipeline.core.config import TTSConfig
from voice_pipeline.tts.greeting_audio import synthesize_to_wav
from voice_pipeline.tts.tts import OpenAITTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize text to WAV via OpenAI TTS")
    parser.add_argument(
        "text",
        help='Text to synthesize (use "-" to read from stdin)',
    )
    parser.add_argument(
        "-n",
        "--name",
        default="output",
        help="Output filename without extension (default: output)",
    )
    parser.add_argument("--voice", default="ash", help="TTS voice (default: ash)")
    parser.add_argument("--model", default="gpt-4o-mini-tts", help="TTS model (default: gpt-4o-mini-tts)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed 0.25-4.0 (default: 1.0)")
    parser.add_argument(
        "--instructions",
        default=None,
        help="Voice instructions (gpt-4o-mini-tts only)",
    )
    args = parser.parse_args()

    text: str = args.text
    if text == "-":
        text = sys.stdin.read().strip()
        if not text:
            print("Error: no text provided via stdin", file=sys.stderr)
            sys.exit(1)

    output = Path("output") / f"{args.name}.wav"

    config = TTSConfig(
        voice=args.voice,
        model=args.model,
        speed=args.speed,
        instructions=args.instructions,
    )
    tts = OpenAITTS(config)

    print(f"Synthesizing ({config.model}, {config.voice}, speed={config.speed})...")
    print(f'  Text: "{text[:80]}{"..." if len(text) > 80 else ""}"')

    synthesize_to_wav(tts, text, output, config.output_sample_rate)

    size = output.stat().st_size
    print(f"  Saved: {output} ({size:,} bytes)")


if __name__ == "__main__":
    main()
