"""Synthesize text to a WAV file.

Requires OPENAI_API_KEY (--vendor openai) or ELEVENLABS_API_KEY (--vendor elevenlabs) env var.

Usage:
    uv run python scripts/tts_to_file.py "안녕하세요, 반갑습니다."
    uv run python scripts/tts_to_file.py "Hello world" -n hello
    uv run python scripts/tts_to_file.py "Hello world" --vendor openai --voice coral --model gpt-4o-mini-tts
    uv run python scripts/tts_to_file.py "Hello world" --vendor elevenlabs --voice EXAVITQu4vr4xnSDxMaL
    echo "Hello" | uv run python scripts/tts_to_file.py -  # read from stdin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_pipeline.tts.elevenlabs_tts import ElevenLabsTTS
from voice_pipeline.tts.factory import _DEFAULT_VENDOR, create_tts
from voice_pipeline.tts.greeting_audio import synthesize_to_wav
from voice_pipeline.tts.openai_tts import OpenAITTS


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
    parser.add_argument(
        "--vendor",
        choices=["openai", "elevenlabs"],
        default=_DEFAULT_VENDOR,
        help=f"TTS vendor (default: {_DEFAULT_VENDOR})",
    )
    parser.add_argument("--voice", default=None, help="Voice preset (openai) or voice ID (elevenlabs)")
    parser.add_argument("--model", default=None, help="TTS model (vendor-specific)")
    parser.add_argument("--speed", type=float, default=None, help="Speed 0.25-4.0 (openai only)")
    parser.add_argument(
        "--instructions",
        default=None,
        help="Voice instructions (openai gpt-4o-mini-tts only)",
    )
    args = parser.parse_args()

    text: str = args.text
    if text == "-":
        text = sys.stdin.read().strip()
        if not text:
            print("Error: no text provided via stdin", file=sys.stderr)
            sys.exit(1)

    if args.vendor == "elevenlabs" and (args.speed is not None or args.instructions is not None):
        print("Error: --speed/--instructions are openai-only options", file=sys.stderr)
        sys.exit(1)

    # 클래스 변수 설정은 인스턴스 생성 전에 적용해야 함. 미지정 옵션은 클래스 기본값 유지.
    if args.vendor == "openai":
        if args.voice is not None:
            OpenAITTS._VOICE = args.voice
        if args.model is not None:
            OpenAITTS._MODEL = args.model
        if args.speed is not None:
            OpenAITTS._SPEED = args.speed
        if args.instructions is not None:
            OpenAITTS._INSTRUCTIONS = args.instructions
    else:
        if args.voice is not None:
            ElevenLabsTTS._VOICE_ID = args.voice
        if args.model is not None:
            ElevenLabsTTS._MODEL = args.model

    tts = create_tts(args.vendor)
    output = Path("output") / f"{args.name}.wav"

    print(f"Synthesizing ({tts.voice_id})...")
    print(f'  Text: "{text[:80]}{"..." if len(text) > 80 else ""}"')

    synthesize_to_wav(tts, text, output)

    size = output.stat().st_size
    print(f"  Saved: {output} ({size:,} bytes)")


if __name__ == "__main__":
    main()
