"""Synthesize text to a WAV file with OpenAI TTS.

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

from voice_pipeline.adapters.tts_openai import OpenAITTS
from voice_pipeline.greeting_audio import synthesize_to_wav


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize text to WAV")
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
    parser.add_argument("--voice", default=None, help="Voice preset")
    parser.add_argument("--model", default=None, help="TTS model")
    parser.add_argument("--speed", type=float, default=None, help="Speed 0.25-4.0")
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

    # 클래스 변수 설정은 인스턴스 생성 전에 적용해야 함. 미지정 옵션은 클래스 기본값 유지.
    if args.voice is not None:
        OpenAITTS._VOICE = args.voice
    if args.model is not None:
        OpenAITTS._MODEL = args.model
    if args.speed is not None:
        OpenAITTS._SPEED = args.speed
    if args.instructions is not None:
        OpenAITTS._INSTRUCTIONS = args.instructions

    tts = OpenAITTS()
    output = Path("var/output") / f"{args.name}.wav"

    print(f"Synthesizing ({tts.voice_id})...")
    print(f'  Text: "{text[:80]}{"..." if len(text) > 80 else ""}"')

    synthesize_to_wav(tts, text, output)

    size = output.stat().st_size
    print(f"  Saved: {output} ({size:,} bytes)")


if __name__ == "__main__":
    main()
