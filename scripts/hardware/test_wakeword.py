"""Microphone + Wakeword integration test.

Usage:
    PYTHONPATH=. uv run python scripts/hardware/test_wakeword.py [--device INDEX] [--keyword WORD]

Listens on mic and prints when the wakeword is detected.
Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import logging
import queue
import time

from voice_pipeline.audio.audio_input import AudioInput
from voice_pipeline.audio.wakeword import WakewordDetector
from voice_pipeline.core.config import AudioConfig, AudioInputConfig, WakewordConfig
from voice_pipeline.core.types import AudioFrame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Show wakeword debug info
logging.getLogger("voice_pipeline.audio").setLevel(logging.DEBUG)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test wakeword detection with mic")
    parser.add_argument("--device", type=int, default=None, help="PyAudio device index")
    parser.add_argument("--keyword", type=str, default="ray", help="Wakeword to detect")
    parser.add_argument("--language", type=str, default="en-US", help="STT language code")
    args = parser.parse_args()

    audio_config = AudioConfig()
    input_config = AudioInputConfig(device_index=args.device)
    wakeword_config = WakewordConfig(
        keywords=(args.keyword,),
        language_code=args.language,
    )

    audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=300)

    print(f"Initializing wakeword detector (keyword='{args.keyword}', lang={args.language})...")
    detector = WakewordDetector(wakeword_config, audio_config)

    print(f"Starting mic capture (device={args.device or 'default'})...")
    audio_input = AudioInput(audio_queue, audio_config, input_config)
    audio_input.start()

    print(f"\nListening for wakeword '{args.keyword}'... (Ctrl+C to stop)\n")

    frame_count = 0
    detections = 0
    start = time.monotonic()

    try:
        while True:
            try:
                frame = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            frame_count += 1
            detected = detector.feed_audio(frame)

            if detected:
                detections += 1
                elapsed = time.monotonic() - start
                print(f"\n>>> WAKEWORD DETECTED! (#{detections} at {elapsed:.1f}s) <<<\n")

            # Periodic status
            if frame_count % 333 == 0:  # ~every 10s
                elapsed = time.monotonic() - start
                print(f"  [{elapsed:.0f}s] {frame_count} frames processed, {detections} detections")

    except KeyboardInterrupt:
        elapsed = time.monotonic() - start
        print(f"\n\nStopped after {elapsed:.1f}s — {frame_count} frames, {detections} detections")
    finally:
        audio_input.stop()
        detector.close()


if __name__ == "__main__":
    main()
