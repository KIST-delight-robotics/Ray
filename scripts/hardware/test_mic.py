"""Quick microphone test using AudioInput.

Usage:
    uv run python scripts/hardware/test_mic.py [--device INDEX] [--seconds N]

Records from mic, prints frame stats, and saves to test_recording.wav.
"""

from __future__ import annotations

import argparse
import queue
import struct
import time
import wave

from voice_pipeline.audio.audio_input import AudioInput
from voice_pipeline.core.config import AudioConfig, AudioInputConfig
from voice_pipeline.core.types import AudioFrame


def list_devices() -> None:
    """Print available audio input devices."""
    import pyaudio

    pa = pyaudio.PyAudio()
    print("\n=== Available Audio Input Devices ===")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(
                f"  [{i}] {info['name']}"
                f"  (channels={info['maxInputChannels']},"
                f" rate={info['defaultSampleRate']})"
            )
    print()
    pa.terminate()


def compute_rms(frame: AudioFrame, sample_width: int = 2) -> float:
    """Compute RMS amplitude of audio frame."""
    if sample_width != 2:
        return 0.0
    n_samples = len(frame) // 2
    if n_samples == 0:
        return 0.0
    samples = struct.unpack(f"<{n_samples}h", frame)
    ms = sum(s * s for s in samples) / n_samples
    return ms**0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="Test microphone capture")
    parser.add_argument("--device", type=int, default=None, help="PyAudio device index")
    parser.add_argument("--seconds", type=int, default=5, help="Recording duration")
    parser.add_argument("--list", action="store_true", help="List devices and exit")
    args = parser.parse_args()

    if args.list:
        list_devices()
        return

    list_devices()

    audio_config = AudioConfig()  # 16kHz, mono, 30ms frames, 16-bit
    input_config = AudioInputConfig(device_index=args.device)
    audio_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=300)

    print(
        f"Recording for {args.seconds}s  (rate={audio_config.sample_rate}, "
        f"frame={audio_config.frame_duration_ms}ms, device={args.device or 'default'})"
    )
    print("Speak into the microphone...\n")

    audio_input = AudioInput(audio_queue, audio_config, input_config)
    audio_input.start()

    frames: list[AudioFrame] = []
    start = time.monotonic()
    frame_count = 0
    dropped = 0

    try:
        while time.monotonic() - start < args.seconds:
            try:
                frame = audio_queue.get(timeout=0.1)
                frames.append(frame)
                frame_count += 1
                rms = compute_rms(frame)
                bar = "#" * min(int(rms / 200), 50)
                elapsed = time.monotonic() - start
                print(
                    f"\r  [{elapsed:5.1f}s] frames={frame_count:4d}  rms={rms:6.0f}  {bar:<50s}",
                    end="",
                    flush=True,
                )
            except queue.Empty:
                dropped += 1
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    finally:
        audio_input.stop()

    duration_s = len(frames) * audio_config.frame_duration_ms / 1000
    print(f"\n\nDone. Captured {frame_count} frames ({duration_s:.1f}s)")

    if audio_input._error:
        print(f"ERROR: {audio_input._error}")
        return

    if not frames:
        print("No frames captured!")
        return

    # Save to WAV
    out_path = "test_recording.wav"
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(audio_config.channels)
        wf.setsampwidth(audio_config.sample_width)
        wf.setframerate(audio_config.sample_rate)
        wf.writeframes(b"".join(frames))

    print(f"Saved to {out_path}")
    print(f"Play with: aplay {out_path}")


if __name__ == "__main__":
    main()
