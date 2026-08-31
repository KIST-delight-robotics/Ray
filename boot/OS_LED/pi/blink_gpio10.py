#!/usr/bin/env python3
"""
blink_gpio10.py — Verify that the Pi has taken over WS2812 DIN.

Run AFTER the ATtiny has called led_release_ownership(). The script drives
BCM GPIO10 (phys pin 19 — SPI0 MOSI, shared with WS2812 DIN) HIGH/LOW at
2 Hz. Verification options:
  - Scope/logic-analyzer the line → expect a clean 2 Hz square wave.
  - Watch the LED strip → it will flicker with random colors (WS2812 sees
    the toggling as garbage protocol). Chaotic flicker = handoff worked.
    No reaction = ATtiny still owns the line (DDRB bit not cleared).

Pi 5 / Bookworm uses libgpiod via gpiozero's lgpio backend. RPi.GPIO is
broken on Pi 5 — do not use it.
"""
from gpiozero import LED
from time import sleep
from signal import signal, SIGTERM

led = LED(10)


def cleanup(*_):
    led.off()
    raise SystemExit(0)


signal(SIGTERM, cleanup)

try:
    while True:
        led.on()
        sleep(0.25)
        led.off()
        sleep(0.25)
except KeyboardInterrupt:
    cleanup()
