#!/usr/bin/env python3
"""
os_led_display.py — Pi-side OS_LED display + SPI ownership arbiter.

On startup:
  1. Drive READY (BCM GPIO17) HIGH so the ATtiny releases PB1 (WS2812 DIN)
     to high-Z. ATtiny's last frame is full white (PULSE_MAX = 255) — the
     strip latches that while we set up SPI.
  2. Sleep briefly to let ATtiny release the line. ATtiny needs only a few
     dozen ms after seeing READY HIGH, so keep this short.
  3. Drive WS2812 with a smooth handoff: start at full-brightness pure
     white (matches ATtiny's frozen frame, no visible jump), then over
     TRANSITION_FRAMES (~1.5 s) fade saturation 0→1 and brightness 1.0→
     runtime BRIGHTNESS. The strip "blooms" from white into the rainbow.
  4. Continue rainbow rotation at ~60 fps thereafter.

Ownership arbiter (so RAY can borrow the strip):
  This daemon is the single Pi-side owner of /dev/spidev0.0. The RAY voice
  pipeline must NOT open SPI while this daemon is running — instead it
  connects to the control socket (CONTROL_SOCK) and borrows the strip:

    RAY → "ACQUIRE\n"   : daemon fades the rainbow out to black, stops
                          writing SPI, then replies "GRANTED\n". RAY may
                          now drive the strip itself.
    RAY → "RELEASE\n"   : (or simply closing the socket / crashing) the
    or socket close       daemon fades black → rainbow back in.

  Only one side writes SPI at a time, so frames never interleave. A RAY
  crash drops the socket, which the daemon treats as RELEASE → the rainbow
  always comes back (the strip stays 1:1 with the real Pi state).

On SIGTERM (e.g. `systemctl stop`):
  - Black out the ring and drop READY LOW for a clean handoff back to the
    ATtiny.

WS2812 over SPI: encoding matched to rpi5_ws2812 (the library RAY uses), SPI at
6.5 MHz, each WS2812 bit encoded as one SPI byte:
  0 → 0b11000000 (T0H ≈ 308 ns)
  1 → 0b11111100 (T1H ≈ 923 ns)
A PREAMBLE_BYTES run of leading zeros (~52 µs LOW) provides the reset/latch.

SPI must be enabled in /boot/firmware/config.txt (`dtparam=spi=on`).
"""
import colorsys
import os
import signal
import socket
import sys
import threading
import time

import spidev
from gpiozero import OutputDevice

NUM_LEDS          = 24          # total LEDs in the series chain
RAINBOW_LEDS      = 8           # idle rainbow shows only on LEDs 1–8; 9–24 stay off
# WS2812 encoding matched byte-for-byte to rpi5_ws2812 (the library RAY drives the
# strip with), which is proven stable on this hardware. 6.5 MHz SPI, each WS2812 bit
# encoded as one SPI byte (8 SPI bits). The hand-rolled 3.2 MHz / 4-bit encoding was
# marginal on this WS2812B/SK6812 strip and flickered.
SPI_HZ            = 6_500_000
LED_ZERO          = 0b1100_0000  # WS2812 "0": ~308 ns high
LED_ONE           = 0b1111_1100  # WS2812 "1": ~923 ns high
PREAMBLE_BYTES    = 42           # leading zeros ≈ 52 µs reset/latch (rpi5_ws2812 PREAMBLE)
HANDOFF_WAIT_S    = 0.3         # ATtiny releases PB1 within ~80 ms of READY HIGH
FRAME_DT_S        = 1 / 60
ROT_PER_FRAME     = 0.005
BRIGHTNESS        = 0.25        # runtime brightness — caps current ~360 mA
TRANSITION_FRAMES = 90          # 1.5 s at 60 fps for the white → rainbow bloom

# Ownership arbiter
CONTROL_SOCK      = "/run/os-led.sock"  # RAY connects here to borrow the strip
FADE_OUT_FRAMES   = 18          # ~0.3 s rainbow → black when RAY acquires
FADE_IN_FRAMES    = 90          # ~1.5 s black → rainbow when RAY releases (서서히)
GRANT_WAIT_S      = 2.0         # max wait for the main loop to confirm the pause
RESUME_SETTLE_S   = 0.25        # wait after release before resuming, so the client
                                # has fully closed its SPI fd (no overlap on the bus)


# Per byte value → 8 SPI bytes (MSB first), each WS2812 bit = LED_ONE/LED_ZERO.
_LUT = [bytes(LED_ONE if (val >> (7 - i)) & 1 else LED_ZERO for i in range(8)) for val in range(256)]
_PREAMBLE = b"\x00" * PREAMBLE_BYTES


def encode_frame(pixels):
    buf = bytearray(_PREAMBLE)   # leading reset/latch, then GRB-encoded pixels
    for r, g, b in pixels:
        buf += _LUT[g]        # WS2812 wire order: G, R, B
        buf += _LUT[r]
        buf += _LUT[b]
    return bytes(buf)


def rainbow(phase, brightness=BRIGHTNESS, saturation=1.0, tail_brightness=0.0):
    # Only LEDs 1–RAINBOW_LEDS show the idle rainbow. The rest of the chain
    # (LEDs RAINBOW_LEDS+1 .. end) are off in steady state, but tail_brightness
    # lets them render as white at a given level so the boot hand-off can fade
    # them out smoothly (ATtiny's frozen white → off) instead of snapping off.
    px = []
    tail = int(max(0.0, min(1.0, tail_brightness)) * 255)
    for i in range(NUM_LEDS):
        if i >= RAINBOW_LEDS:
            px.append((tail, tail, tail))
            continue
        h = (i / RAINBOW_LEDS + phase) % 1.0   # full spectrum spread across the lit LEDs
        r, g, b = colorsys.hsv_to_rgb(h, saturation, brightness)
        px.append((int(r * 255), int(g * 255), int(b * 255)))
    return px


# ---------------------------------------------------------------------------
# Ownership arbiter — control socket server (runs on a daemon thread)
# ---------------------------------------------------------------------------
#
# Three threading primitives coordinate the borrow protocol with the main
# render loop:
#   pause_req  — set while a RAY client holds the token (main loop must yield)
#   paused     — set by the main loop once it has faded out and stopped writing
#   stop_evt   — process is shutting down
pause_req = threading.Event()
paused    = threading.Event()
stop_evt  = threading.Event()


def _control_server():
    """Accept one RAY client at a time and translate it to pause_req."""
    try:
        if os.path.exists(CONTROL_SOCK):
            os.unlink(CONTROL_SOCK)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(CONTROL_SOCK)
        os.chmod(CONTROL_SOCK, 0o666)   # RAY runs as a non-root user
        srv.listen(1)
        srv.settimeout(1.0)
    except Exception as exc:
        print(f"control socket unavailable ({exc}) — arbiter disabled", flush=True)
        return

    print(f"arbiter listening on {CONTROL_SOCK}", flush=True)
    while not stop_evt.is_set():
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        _serve_client(conn)

    try:
        srv.close()
        os.unlink(CONTROL_SOCK)
    except OSError:
        pass


def _serve_client(conn):
    """Hold the token for the lifetime of one client connection."""
    owns = False
    try:
        conn.settimeout(1.0)
        buf = b""
        while not stop_evt.is_set():
            try:
                data = conn.recv(64)
            except socket.timeout:
                continue
            if not data:
                break                       # socket closed / RAY crashed
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                cmd = line.strip().upper()
                if cmd == b"ACQUIRE":
                    pause_req.set()
                    # block until the render loop has actually stopped writing
                    paused.wait(timeout=GRANT_WAIT_S)
                    owns = True
                    try:
                        conn.sendall(b"GRANTED\n")
                    except OSError:
                        return
                elif cmd == b"RELEASE":
                    pause_req.clear()
                    owns = False
                    try:
                        conn.sendall(b"OK\n")
                    except OSError:
                        return
    finally:
        # Any disconnect (graceful or crash) releases the token.
        pause_req.clear()
        try:
            conn.close()
        except OSError:
            pass


def main():
    ready = OutputDevice(17, initial_value=False)
    ready.on()
    print("READY (GPIO17) HIGH — ATtiny may now release WS2812 DIN", flush=True)

    time.sleep(HANDOFF_WAIT_S)

    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = SPI_HZ
    spi.mode = 0
    spi.lsbfirst = False
    print(f"SPI0 @ {SPI_HZ/1e6:.1f} MHz — driving {NUM_LEDS} LEDs", flush=True)

    black = encode_frame([(0, 0, 0)] * NUM_LEDS)

    def cleanup(*_):
        stop_evt.set()
        spi.writebytes2(black)
        spi.close()
        ready.off()
        try:
            os.unlink(CONTROL_SOCK)
        except OSError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)

    # Start the ownership arbiter (RAY borrows the strip through this).
    threading.Thread(target=_control_server, name="os-led-arbiter", daemon=True).start()

    phase = 0.0
    frame_count = 0
    try:
        while True:
            # --- yield to RAY: fade rainbow → black, then stop writing SPI ---
            if pause_req.is_set() and not paused.is_set():
                print("arbiter: client acquired — fading out, releasing SPI", flush=True)
                for i in range(FADE_OUT_FRAMES):
                    scale = 1.0 - (i + 1) / FADE_OUT_FRAMES
                    spi.writebytes2(encode_frame(rainbow(phase, BRIGHTNESS * scale)))
                    time.sleep(FRAME_DT_S)
                spi.writebytes2(black)
                paused.set()                 # tell the arbiter the pause is committed

            # --- RAY owns the strip: do not touch SPI at all ---
            if paused.is_set():
                if not pause_req.is_set():
                    # RAY released. Wait for its SPI fd to fully close before we
                    # touch the bus, then bloom black → rainbow back in (서서히).
                    print("arbiter: client released — settling before resume", flush=True)
                    time.sleep(RESUME_SETTLE_S)
                    if pause_req.is_set():
                        continue             # RAY re-acquired during settle
                    spi.writebytes2(black)   # clean baseline regardless of last frame
                    for i in range(FADE_IN_FRAMES):
                        if pause_req.is_set():
                            break            # RAY re-acquired mid-fade
                        t = (i + 1) / FADE_IN_FRAMES
                        spi.writebytes2(encode_frame(rainbow(phase, BRIGHTNESS * t)))
                        phase = (phase + ROT_PER_FRAME) % 1.0
                        time.sleep(FRAME_DT_S)
                    paused.clear()
                    frame_count = TRANSITION_FRAMES  # skip the white bloom on resume
                    print("arbiter: rainbow resumed", flush=True)
                    continue
                time.sleep(FRAME_DT_S)
                continue

            # --- normal: rainbow (with the one-time white bloom at startup) ---
            if frame_count < TRANSITION_FRAMES:
                t = frame_count / TRANSITION_FRAMES
                cur_brightness = 1.0 - (1.0 - BRIGHTNESS) * t
                cur_saturation = t
                tail_brightness = 1.0 - t   # LEDs 9–24: white → off over the bloom
            else:
                cur_brightness = BRIGHTNESS
                cur_saturation = 1.0
                tail_brightness = 0.0

            spi.writebytes2(encode_frame(rainbow(phase, cur_brightness, cur_saturation, tail_brightness)))
            phase = (phase + ROT_PER_FRAME) % 1.0
            frame_count += 1
            time.sleep(FRAME_DT_S)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
