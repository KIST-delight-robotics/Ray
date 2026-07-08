#!/usr/bin/env python3
"""Play a WAV file through the C++ Ray process and watch online motor motion.

C++ opens the WAV directly (sf_open) and generates the mouth/head motor
trajectory online from the audio while playing. This sends a single play_file
command for an arbitrary WAV and waits until playback completes.

Usage:
    # C++ already running (./build/Ray in another terminal):
    uv run python scripts/hardware/play_wav_motion.py output/tts_articulation_dynamic.wav

    # Start/stop the C++ process automatically:
    uv run python scripts/hardware/play_wav_motion.py output/tts_articulation_dynamic.wav --start-cpp
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from voice_pipeline.bridge.cpp_bridge import CppBridge
from voice_pipeline.core.types import CppEventType


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a WAV through C++ and watch motor motion")
    parser.add_argument("wav", help="Path to the WAV file to play")
    parser.add_argument(
        "--start-cpp",
        action="store_true",
        help="Automatically start/stop the C++ Ray process (./build/Ray)",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=CppBridge._PORT)
    parser.add_argument("--timeout", type=float, default=120.0, help="Max seconds to wait for completion")
    args = parser.parse_args()

    wav_path = Path(args.wav).resolve()
    if not wav_path.exists():
        print(f"Error: WAV not found: {wav_path}", file=sys.stderr)
        sys.exit(1)

    cpp_proc = None
    if args.start_cpp:
        print("Starting C++ process (./build/Ray)...")
        cpp_proc = subprocess.Popen(["./build/Ray"])
        time.sleep(3)
        if cpp_proc.poll() is not None:
            print(f"C++ process exited early (code {cpp_proc.returncode})", file=sys.stderr)
            sys.exit(1)
        print(f"C++ process started (PID {cpp_proc.pid})")

    CppBridge._HOST = args.host
    CppBridge._PORT = args.port
    bridge = CppBridge()

    try:
        bridge.connect()
        print(f"Connected to C++ at ws://{args.host}:{args.port}")

        bridge.send_play_file(str(wav_path))
        print(f"→ play_file: {wav_path}")

        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            event = bridge.poll_event()
            if event is not None:
                print(f"  ← C++: {event.event_type.value}")
                if event.event_type == CppEventType.PLAYBACK_COMPLETE:
                    print("✓ Playback complete")
                    break
            else:
                time.sleep(0.05)
        else:
            print("✗ Timeout waiting for playback_complete", file=sys.stderr)
    finally:
        try:
            bridge.disconnect()
            print("Disconnected from C++")
        except Exception:
            pass
        if cpp_proc is not None:
            print("Stopping C++ process...")
            cpp_proc.send_signal(signal.SIGINT)
            try:
                cpp_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cpp_proc.kill()


if __name__ == "__main__":
    main()
