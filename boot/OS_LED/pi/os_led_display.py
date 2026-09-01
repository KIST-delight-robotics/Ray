#!/usr/bin/env python3
"""
os_led_display.py — Pi-side OS_LED display + SPI ownership arbiter.

동작 규칙은 하나다: **이 데몬이 스트립을 잡고 있는 동안엔 항상 노란 호흡**
(233,233,50 · 4.0 s 사인 · 밝기 0.15~1.0 — ATtiny 부팅 호흡, RAY 대기 호흡과 동일).

On startup:
  1. Drive READY (BCM GPIO17) HIGH so the ATtiny releases PB1 (WS2812 DIN)
     to high-Z. ATtiny's last frame is the breath peak — the strip latches
     that while we set up SPI.
  2. Sleep briefly to let ATtiny release the line.
  3. 노란 호흡을 피크 위상부터 이어서 구동 — ATtiny가 정지시킨 프레임과 첫
     Pi 프레임이 일치해 인수가 보이지 않는다.

Ownership arbiter (so RAY can borrow the strip):
  This daemon is the single Pi-side owner of /dev/spidev0.0. The RAY voice
  pipeline must NOT open SPI while this daemon is running — instead it
  connects to the control socket (CONTROL_SOCK) and borrows the strip:

    RAY → "ACQUIRE\n"   : 호흡을 RAY 첫 프레임(bar off, ring 0.15)으로 보간
                          페이드한 뒤 SPI 쓰기를 멈추고 "GRANTED\n" 응답.
    RAY → "RELEASE\n"   : (or simply closing the socket / crashing) 소등 유지.
    or socket close       재획득 시 검정 → RAY 첫 프레임으로 페이드.

  Only one side writes SPI at a time, so frames never interleave.
  반납 후에는 소등을 유지한다(개발 중 RAY를 끄면 LED도 꺼짐). RAY 재시작이면
  재획득 페이드로 돌아오고, 시스템 종료면 곧 SIGTERM으로 이 데몬이 내려가며
  ATtiny 종료 호흡(0부터 램프업)이 이어받는다 — 무지개 같은 별도 상태 표시는
  없다 (초기 인수 테스트용이었고 실운용에선 불필요해 제거, 2026-09).

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
import math
import os
import signal
import socket
import sys
import threading
import time

import spidev
from gpiozero import OutputDevice

NUM_LEDS          = 24          # total LEDs in the series chain
# WS2812 encoding matched byte-for-byte to rpi5_ws2812 (the library RAY drives the
# strip with), which is proven stable on this hardware. 6.5 MHz SPI, each WS2812 bit
# encoded as one SPI byte (8 SPI bits). The hand-rolled 3.2 MHz / 4-bit encoding was
# marginal on this WS2812B/SK6812 strip and flickered.
SPI_HZ            = 6_500_000
LED_ZERO          = 0b1100_0000  # WS2812 "0": ~308 ns high
LED_ONE           = 0b1111_1100  # WS2812 "1": ~923 ns high
PREAMBLE_BYTES    = 42           # leading zeros ≈ 52 µs reset/latch (rpi5_ws2812 PREAMBLE)
HANDOFF_WAIT_S    = 1.2         # safety margin before first SPI write. Nominal
                                # firmware releases PB1 within ~80 ms of READY HIGH,
                                # but waiting longer only prolongs the ATtiny's frozen
                                # full-white frame — never a bus conflict.
FRAME_DT_S        = 1 / 60

# 호흡 스펙 — 펌웨어와 반드시 동기 유지 (틀어지면 인수 순간 밝기/속도 점프가 보인다).
# RAY 대기(SLEEPING) 디밍과 동일 스펙: 4.0 s 사인 곡선, 밝기 0.15~1.0.
# ATtiny 펌웨어도 같은 색/속도의 노란 호흡으로 변경됨 — 세 구간(ATtiny → Pi → RAY)이
# 하나의 호흡으로 이어져 보이도록 유지할 것.
BREATH_PERIOD_S   = 4.0
BREATH_FRAMES     = round(BREATH_PERIOD_S / FRAME_DT_S)   # 240 frames @60 fps
BREATH_MIN        = 0.15        # RAY BreathingAnimation._MIN_BRIGHTNESS와 동일
BREATH_MAX        = 1.0         # ATtiny's frozen hand-off frame
BREATH_COLOR      = (233, 233, 50)   # RAY 대기 색과 동일 (ATtiny도 동일 색)
# How long to keep breathing before deciding RAY isn't coming. RAY normally

# Ownership arbiter
CONTROL_SOCK      = "/run/os-led.sock"  # RAY connects here to borrow the strip
FADE_OUT_FRAMES   = 42          # ~0.7 s 호흡 → RAY 첫 프레임 보간 (인수 시).
                                # 0.3 s에서는 bar 8구가 "툭 꺼짐"으로 보였다.
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
    """RAY 대기 디밍과 동일한 사인 호흡, 피크에서 시작.

    frame_i 0 → BREATH_MAX: 첫 Pi 프레임이 ATtiny가 핸드오프 때 정지시킨
    최대 밝기 프레임과 일치해 눈에 띄는 점프가 없다.
    """
    t = frame_i * FRAME_DT_S
    phase = (math.sin(2 * math.pi * t / BREATH_PERIOD_S + math.pi / 2) + 1) / 2
    return BREATH_MIN + (BREATH_MAX - BREATH_MIN) * phase


def breath_frame(level):
    """부팅 호흡 프레임: 체인 전체를 BREATH_COLOR(노랑)로, ATtiny 애니메이션과 동일."""
    lv = max(0.0, min(1.0, level))
    r, g, b = (int(c * lv) for c in BREATH_COLOR)
    return [(r, g, b)] * NUM_LEDS


RAY_BAR_LEDS = 8   # RAY 프레임 레이아웃: bar 8개(앞) + ring 16개(뒤).
                   # voice_pipeline/adapters/led.py 의 _BAR_COUNT와 반드시 일치.


def handoff_frame(level, t):
    """부팅 호흡(전체 노랑, 밝기 level) → RAY SLEEPING 첫 프레임으로 t(0→1) 보간.

    RAY의 첫 프레임은 bar 꺼짐 + ring을 BREATH_MIN(0.15) 노랑으로 그리므로,
    같은 프레임까지 페이드해 파킹하면 인수 순간에 검정 공백 없이 이어진다."""
    tt = max(0.0, min(1.0, t))
    bar_lv = level * (1.0 - tt)
    ring_lv = level + (BREATH_MIN - level) * tt
    bar = tuple(int(c * bar_lv) for c in BREATH_COLOR)
    ring = tuple(int(c * ring_lv) for c in BREATH_COLOR)
    return [bar] * RAY_BAR_LEDS + [ring] * (NUM_LEDS - RAY_BAR_LEDS)


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

    frame_count = 0
    had_client = False   # RAY가 한 번이라도 인수했으면 True — 이후 반납 시 소등 유지
    try:
        while True:
            # --- yield to RAY: 현재 화면 → RAY 첫 프레임으로 보간 후 SPI 정지 ---
            if pause_req.is_set() and not paused.is_set():
                print("arbiter: client acquired — fading out, releasing SPI", flush=True)
                # 호흡(부팅) 또는 검정(재획득)을 RAY 첫 프레임(bar off, ring 0.15)으로
                # 보간. 검정으로 껐다 켜면 인수 순간이 깜박여 보인다.
                level = 0.0 if had_client else breath_level(frame_count)
                for i in range(FADE_OUT_FRAMES):
                    t = (i + 1) / FADE_OUT_FRAMES
                    spi.writebytes2(encode_frame(handoff_frame(level, t)))
                    time.sleep(FRAME_DT_S)
                paused.set()                 # tell the arbiter the pause is committed
                had_client = True

            # --- RAY owns the strip: do not touch SPI at all ---
            if paused.is_set():
                if not pause_req.is_set():
                    # RAY released (수동 정지·크래시·시스템 종료). 호흡을 재개하지 않고
                    # 소등 유지 — 개발 중 RAY를 꺼두면 LED도 꺼진다. RAY가 재획득하면
                    # 검정 → ring 0.15 페이드로 다시 인수하고, 시스템 종료면 곧 SIGTERM으로
                    # 이 데몬이 내려가며 ATtiny 종료 호흡(0부터 램프업)이 이어받는다.
                    print("arbiter: client released — strip dark until next acquire", flush=True)
                    time.sleep(RESUME_SETTLE_S)
                    # `paused` set must always mean "this loop is not writing SPI":
                    # the arbiter grants ACQUIRE the moment it sees `paused`, so it
                    # has to be cleared BEFORE the first write. Clear first, then
                    # re-check pause_req — an ACQUIRE that slipped in during the
                    # settle re-parks untouched.
                    paused.clear()
                    if pause_req.is_set():
                        paused.set()         # RAY re-acquired during settle
                        continue
                    spi.writebytes2(black)
                    continue
                time.sleep(FRAME_DT_S)
                continue

            # --- 소등 대기: RAY가 한 번 잡았다 놓은 뒤에는 재획득까지 그리지 않음 ---
            if had_client:
                time.sleep(FRAME_DT_S)
                continue

            # --- 부팅 상태: 노란 호흡 (첫 인수까지) ---
            spi.writebytes2(encode_frame(breath_frame(breath_level(frame_count))))
            frame_count += 1
            time.sleep(FRAME_DT_S)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
