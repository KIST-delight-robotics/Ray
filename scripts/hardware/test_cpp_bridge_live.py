#!/usr/bin/env python3
"""Live integration test: Python CppBridge ↔ C++ Ray process.

Usage:
    # Terminal 1: Start C++ (from project root)
    ./build/Ray

    # Terminal 2: Run this script (from project root)
    uv run python scripts/hardware/test_cpp_bridge_live.py

    # Or run both automatically:
    uv run python scripts/hardware/test_cpp_bridge_live.py --start-cpp
"""

from __future__ import annotations

import argparse
import signal
import struct
import subprocess
import sys
import time

# Project imports
sys.path.insert(0, ".")
from voice_pipeline.bridge.cpp_bridge import CppBridge
from voice_pipeline.core.config import CppBridgeConfig
from voice_pipeline.core.types import CppEventType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_sine_pcm(
    freq: float = 440.0,
    duration_sec: float = 0.5,
    sample_rate: int = 24000,
) -> bytes:
    """Generate a sine wave as 16-bit mono PCM bytes."""
    import math

    samples = []
    for i in range(int(sample_rate * duration_sec)):
        t = i / sample_rate
        value = int(32767 * 0.3 * math.sin(2 * math.pi * freq * t))
        samples.append(struct.pack("<h", value))
    return b"".join(samples)


def _poll_until(
    bridge: CppBridge,
    event_type: CppEventType,
    timeout: float = 15.0,
    label: str = "",
) -> bool:
    """Poll bridge until a specific event type is received or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        event = bridge.poll_event()
        if event is not None:
            print(f"    ← C++: {event.event_type.value}")
            if event.event_type == event_type:
                return True
        else:
            time.sleep(0.05)
    print(f"    ✗ Timeout waiting for {event_type.value} ({label})")
    return False


def _drain_events(bridge: CppBridge, wait: float = 0.5) -> None:
    """Drain any stale events."""
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        ev = bridge.poll_event()
        if ev:
            print(f"    (drained: {ev.event_type.value})")
        else:
            time.sleep(0.05)


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------


def test_connection(bridge: CppBridge) -> bool:
    print("\n[Test 1] Connection")
    try:
        bridge.connect()
        print("    ✓ Connected to C++ WebSocket server")
        return True
    except Exception as e:
        print(f"    ✗ Connection failed: {e}")
        return False


def test_play_file(bridge: CppBridge) -> bool:
    print("\n[Test 2] play_file (awake.wav)")
    _drain_events(bridge, wait=0.3)

    bridge.send_play_file("assets/audio/awake.wav")
    print("    → Sent play_file")

    ok_started = _poll_until(bridge, CppEventType.PLAYBACK_STARTED, label="play_file started")
    ok_complete = _poll_until(bridge, CppEventType.PLAYBACK_COMPLETE, label="play_file complete")

    if ok_started and ok_complete:
        print("    ✓ play_file round-trip OK")
        return True
    return False


def test_streaming(bridge: CppBridge) -> bool:
    print("\n[Test 3] Streaming (sine wave 0.5s)")
    _drain_events(bridge, wait=0.3)

    # Generate test audio
    pcm = _generate_sine_pcm(freq=440.0, duration_sec=0.5, sample_rate=24000)
    chunk_size = 4800  # 100ms chunks at 24kHz 16-bit mono

    bridge.send_stream_start()
    print("    → Sent stream_start")

    # Send audio in chunks
    sent = 0
    for i in range(0, len(pcm), chunk_size):
        chunk = pcm[i : i + chunk_size]
        bridge.send_audio(chunk)
        sent += len(chunk)
    print(f"    → Sent {sent} bytes in {(len(pcm) + chunk_size - 1) // chunk_size} chunks")

    bridge.send_audio_end()
    print("    → Sent audio_end")

    ok_started = _poll_until(bridge, CppEventType.PLAYBACK_STARTED, label="stream started")
    ok_complete = _poll_until(bridge, CppEventType.PLAYBACK_COMPLETE, label="stream complete")

    if ok_started and ok_complete:
        print("    ✓ Streaming round-trip OK")
        return True
    return False


def test_stop_interrupt(bridge: CppBridge) -> bool:
    print("\n[Test 4] Stop (interrupt during streaming)")
    _drain_events(bridge, wait=0.3)

    # Generate longer audio (2 seconds)
    pcm = _generate_sine_pcm(freq=330.0, duration_sec=2.0, sample_rate=24000)
    chunk_size = 4800

    bridge.send_stream_start()
    print("    → Sent stream_start")

    # Send enough chunks to start playback, then interrupt
    chunks_sent = 0
    for i in range(0, len(pcm), chunk_size):
        chunk = pcm[i : i + chunk_size]
        bridge.send_audio(chunk)
        chunks_sent += 1
        if chunks_sent >= 5:  # ~500ms worth
            break

    print(f"    → Sent {chunks_sent} chunks, now interrupting")

    # Wait briefly for playback to start
    ok_started = _poll_until(
        bridge,
        CppEventType.PLAYBACK_STARTED,
        timeout=5.0,
        label="stream started",
    )

    # Send stop
    bridge.send_stop()
    print("    → Sent stop")

    # Should get playback_complete (even on interrupt)
    ok_complete = _poll_until(
        bridge,
        CppEventType.PLAYBACK_COMPLETE,
        timeout=10.0,
        label="stop complete",
    )

    if ok_started and ok_complete:
        print("    ✓ Stop/interrupt round-trip OK")
        return True
    return False


def test_play_file_farewell(bridge: CppBridge) -> bool:
    print("\n[Test 5] play_file (sleep.wav)")
    _drain_events(bridge, wait=0.3)

    bridge.send_play_file("assets/audio/sleep.wav")
    print("    → Sent play_file (farewell)")

    ok_started = _poll_until(bridge, CppEventType.PLAYBACK_STARTED, label="farewell started")
    ok_complete = _poll_until(bridge, CppEventType.PLAYBACK_COMPLETE, label="farewell complete")

    if ok_started and ok_complete:
        print("    ✓ Farewell play_file round-trip OK")
        return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Live C++↔Python bridge test")
    parser.add_argument(
        "--start-cpp",
        action="store_true",
        help="Automatically start/stop the C++ Ray process",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", type=str, default="localhost")
    args = parser.parse_args()

    cpp_proc = None

    if args.start_cpp:
        print("Starting C++ process (./build/Ray)...")
        cpp_proc = subprocess.Popen(
            ["./build/Ray"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # Give C++ time to start WebSocket server
        time.sleep(3)
        if cpp_proc.poll() is not None:
            output = cpp_proc.stdout.read().decode() if cpp_proc.stdout else ""
            print(f"C++ process exited early (code {cpp_proc.returncode}):\n{output}")
            sys.exit(1)
        print(f"C++ process started (PID {cpp_proc.pid})")

    config = CppBridgeConfig(host=args.host, port=args.port)
    bridge = CppBridge(config)

    results: list[tuple[str, bool]] = []

    try:
        ok = test_connection(bridge)
        results.append(("Connection", ok))
        if not ok:
            print("\nConnection failed — aborting remaining tests.")
            return

        for name, test_fn in [
            ("play_file (greeting)", test_play_file),
            ("Streaming", test_streaming),
            ("Stop/interrupt", test_stop_interrupt),
            ("play_file (farewell)", test_play_file_farewell),
        ]:
            try:
                ok = test_fn(bridge)
            except Exception as e:
                print(f"    ✗ Exception: {e}")
                ok = False
            results.append((name, ok))
            # Brief pause between tests
            time.sleep(1)

    finally:
        try:
            bridge.disconnect()
            print("\nDisconnected from C++")
        except Exception:
            pass

        if cpp_proc is not None:
            print("Stopping C++ process...")
            cpp_proc.send_signal(signal.SIGINT)
            try:
                cpp_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cpp_proc.kill()

    # Summary
    print("\n" + "=" * 50)
    print("Results:")
    all_ok = True
    for name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_ok = False
    print("=" * 50)

    if all_ok:
        print("All tests passed!")
    else:
        print("Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
