#!/usr/bin/env python3
"""Simple WebSocket connection test between PC (Python) and Raspberry Pi (C++).

Usage:
    # 1) Test connection only (ping-pong)
    python scripts/test_ws_connection.py --host 192.168.x.x

    # 2) Send a short beep tone to verify audio playback
    python scripts/test_ws_connection.py --host 192.168.x.x --send-tone

    # 3) Custom port
    python scripts/test_ws_connection.py --host 192.168.x.x --port 8765
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import struct
import time

from websockets.sync.client import connect as ws_connect


def generate_sine_pcm(
    freq: float = 440.0,
    duration_sec: float = 1.0,
    sample_rate: int = 24000,
    amplitude: float = 0.3,
) -> bytes:
    """Generate a sine wave as 16-bit mono PCM bytes."""
    samples = []
    for i in range(int(sample_rate * duration_sec)):
        t = i / sample_rate
        value = int(amplitude * 32767 * math.sin(2 * math.pi * freq * t))
        samples.append(struct.pack("<h", value))
    return b"".join(samples)


def test_connection(host: str, port: int, send_tone: bool) -> None:
    uri = f"ws://{host}:{port}"
    print(f"[1/4] Connecting to {uri} ...")

    try:
        ws = ws_connect(uri, open_timeout=5)
    except Exception as e:
        print(f"  FAIL: {e}")
        print()
        print("Troubleshooting:")
        print(f"  - Is the C++ process running on {host}?")
        print(f"  - Is port {port} open? (ssh into RPi and run: ss -tlnp | grep {port})")
        print(f"  - Can you reach it? (ping {host})")
        return

    print("  OK: WebSocket connected!")

    # --- Test: send stream_start and see if C++ accepts it ---
    print("[2/4] Sending stream_start ...")
    ws.send(json.dumps({"type": "stream_start"}))
    print("  OK: stream_start sent")

    if send_tone:
        # --- Send a 1-second 440Hz tone ---
        print("[3/4] Sending 1s 440Hz tone ...")
        pcm = generate_sine_pcm(freq=440, duration_sec=1.0)
        chunk_size = 4800  # 100ms chunks at 24kHz 16-bit mono
        chunks_sent = 0
        for i in range(0, len(pcm), chunk_size):
            chunk = pcm[i : i + chunk_size]
            msg = json.dumps({"type": "audio", "data": base64.b64encode(chunk).decode()})
            ws.send(msg)
            chunks_sent += 1
        print(f"  OK: sent {chunks_sent} audio chunks ({len(pcm)} bytes total)")

        ws.send(json.dumps({"type": "audio_end"}))
        print("  OK: audio_end sent")

        # Wait for playback events
        print("[4/4] Waiting for C++ events (5s timeout) ...")
        events = []
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                raw = ws.recv(timeout=1.0)
                data = json.loads(raw)
                events.append(data.get("type", "unknown"))
                print(f"  <- received: {data}")
            except TimeoutError:
                continue
            except Exception:
                break

        if events:
            print(f"  OK: got {len(events)} event(s): {events}")
        else:
            print("  INFO: no events received (C++ may not send events for this flow)")
    else:
        # Just send audio_end to cleanly close the stream
        ws.send(json.dumps({"type": "audio_end"}))
        print("[3/4] Skipping tone (use --send-tone to test audio)")
        print("[4/4] Done")

    ws.close()
    print()
    print("Connection test PASSED!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test WebSocket connection to C++ on RPi")
    parser.add_argument("--host", required=True, help="Raspberry Pi IP address")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port (default: 8765)")
    parser.add_argument("--send-tone", action="store_true", help="Send a 1s sine tone to test audio playback")
    args = parser.parse_args()

    test_connection(args.host, args.port, args.send_tone)


if __name__ == "__main__":
    main()
