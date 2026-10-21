"""카탈로그 곡 하나를 C++ 프로세스로 바로 재생한다 (대화 파이프라인 없이).

머리·입 모션과 후면 LED 는 C++ 가 곡과 함께 구동한다. ``build/Ray`` 가 먼저 떠 있어야 한다.

Usage:
    uv run python scripts/play_song.py              # 곡 목록 출력
    uv run python scripts/play_song.py Butter_BTS   # 재생 (Ctrl-C 로 중단)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_pipeline.adapters.cpp_bridge import CppBridge, CppEventType
from voice_pipeline.engines.gpt_live.songs import load_song_catalog


def main() -> int:
    parser = argparse.ArgumentParser(description="카탈로그 곡 하나를 C++ 프로세스로 재생")
    parser.add_argument("song", nargs="?", help="곡 키 (assets/songs.json). 생략하면 목록 출력")
    args = parser.parse_args()

    catalog = load_song_catalog()
    if args.song not in catalog:
        if args.song:
            print(f"카탈로그에 없는 곡: {args.song}\n")
        for song in catalog.values():
            print(song.describe())
        return 0 if args.song is None else 1

    bridge = CppBridge()
    bridge.connect()
    try:
        bridge.send_play_audio_csv(args.song)
        print(f"재생: {catalog[args.song].describe()} (Ctrl-C 로 중단)")
        while (event := bridge.poll_event()) is None or event.event_type is not CppEventType.PLAYBACK_COMPLETE:
            time.sleep(0.1)
    except KeyboardInterrupt:
        bridge.send_stop()
        print("\n중단")
    finally:
        bridge.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
