#!/usr/bin/env python3
"""
os_led_display.py — Pi-side OS_LED display + SPI ownership arbiter.

On startup:
  1. Drive READY (BCM GPIO17) HIGH so the ATtiny releases PB1 (WS2812 DIN)
     to high-Z. ATtiny's last frame is full white (PULSE_MAX = 255) — the
     strip latches that while we set up SPI.
  2. Sleep briefly to let ATtiny release the line. ATtiny needs only a few
     dozen ms after seeing READY HIGH, so keep this short.
  3. Drive WS2812 continuing the ATtiny's white breathing (same parabolic
     curve / 2.0 s period / PULSE_MIN..PULSE_MAX levels), starting at the
     peak so it picks up exactly where ATtiny's frozen full-white frame
     left off. The hand-off is invisible: the boot animation just keeps
     going, now driven by the Pi.
  4. Normally RAY acquires the strip during this hold and the boot
     breathing hands straight over to the RAY pattern — no rainbow at all.
     If RAY never shows up within BOOT_WHITE_HOLD_S, bloom white → rainbow
     (TRANSITION_FRAMES ≈ 1.5 s) and rotate at ~60 fps thereafter, so
     "Pi is up but RAY is dead" still looks different from a normal boot.

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

# Boot hold — continue ATtiny's white breathing instead of blooming into the
# rainbow, so 부팅 애니메이션 → RAY LED 로 곧장 넘어간다. Values mirror the
# firmware (OS_LED/README.md §3.1 "모든 ATtiny 애니메이션은 순백, 포물선 idx*(N-idx) 근사,
# PULSE_MIN=16~PULSE_MAX=255, 한 호흡 주기 ≈ 2.0 s") — they must stay in sync or
# the hand-off becomes visible as a brightness/rate jump.
BREATH_PERIOD_S   = 2.0
BREATH_FRAMES     = round(BREATH_PERIOD_S / FRAME_DT_S)   # 120 frames @60 fps
BREATH_MIN        = 16 / 255    # PULSE_MIN
BREATH_MAX        = 1.0         # PULSE_MAX (ATtiny's frozen hand-off frame)
# How long to keep breathing before deciding RAY isn't coming. RAY normally
# acquires ~25–50 s after this daemon starts (model loading dominates), so this
# is generous. Rounded to whole breaths so the hold always ends at a peak —
# the bloom below starts at full white, so ending anywhere else would jump.
BOOT_WHITE_HOLD_S = 150.0
BOOT_HOLD_FRAMES  = BREATH_FRAMES * round(BOOT_WHITE_HOLD_S / BREATH_PERIOD_S)

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


def breath_level(frame_i):
    """ATtiny's breathing curve: parabolic idx*(N-idx), starting at the peak.

    frame_i 0 → BREATH_MAX, so the first Pi-driven frame matches the full-white
    frame ATtiny latched at hand-off (no visible jump).
    """
    n = BREATH_FRAMES
    half = n // 2
    idx = (frame_i + half) % n          # phase-shift so frame 0 lands on the peak
    tri = idx * (n - idx) / (half * half)   # 0..1, peaks at idx == half
    return BREATH_MIN + (BREATH_MAX - BREATH_MIN) * tri


def white_frame(level):
    """Pure white on the whole chain (G=R=B), like every ATtiny animation."""
    v = int(max(0.0, min(1.0, level)) * 255)
    return [(v, v, v)] * NUM_LEDS


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
                # Fade out whatever is actually on screen: during the boot hold
                # that's the white breathing, afterwards the rainbow. Fading the
                # wrong one snaps brightness/colour at the exact moment RAY takes
                # over — the one transition this whole design tries to hide.
                boot_level = breath_level(frame_count) if frame_count < BOOT_HOLD_FRAMES else None
                for i in range(FADE_OUT_FRAMES):
                    scale = 1.0 - (i + 1) / FADE_OUT_FRAMES
                    if boot_level is None:
                        frame = rainbow(phase, BRIGHTNESS * scale)
                    else:
                        frame = white_frame(boot_level * scale)
                    spi.writebytes2(encode_frame(frame))
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
                    # Skip both the boot breathing hold and the bloom: RAY was
                    # already up, so the rainbow here means "RAY went away".
                    frame_count = BOOT_HOLD_FRAMES + TRANSITION_FRAMES
                    print("arbiter: rainbow resumed", flush=True)
                    continue
                time.sleep(FRAME_DT_S)
                continue

            # --- boot hold: keep breathing ATtiny's white, waiting for RAY ---
            if frame_count < BOOT_HOLD_FRAMES:
                spi.writebytes2(encode_frame(white_frame(breath_level(frame_count))))
                frame_count += 1
                time.sleep(FRAME_DT_S)
                continue

            if frame_count == BOOT_HOLD_FRAMES:
                print(
                    f"RAY did not acquire within {BOOT_WHITE_HOLD_S:.0f}s "
                    "— blooming into the rainbow", flush=True
                )

            # --- fallback: bloom white → rainbow, then rotate ---
            bloom_i = frame_count - BOOT_HOLD_FRAMES
            if bloom_i < TRANSITION_FRAMES:
                t = bloom_i / TRANSITION_FRAMES
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
