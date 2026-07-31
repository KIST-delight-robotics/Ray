#!/usr/bin/env python3
"""
os_led_poweroff.py — graceful poweroff when ATtiny asserts SHUTDOWN_REQ.

The ATtiny drives PB4 HIGH on the second touch. PB4 reaches Pi GPIO27
through a 10 kΩ + 20 kΩ voltage divider (5 V → ~3.3 V, Pi-safe). The
external 20 kΩ pulls GPIO27 LOW when ATtiny is idle.

We poll GPIO27 every 20 ms instead of using gpiozero edge callbacks —
`Button.when_pressed` on Pi 5's lgpio backend has been observed to miss
events. Polling at 50 Hz is plenty for human-timed signals and is much
more reliable.

On rising edge:
  - sync filesystems
  - `systemctl poweroff -f` — single-force does a graceful shutdown
    (services SIGTERM and clean up) but ignores inhibitor locks held by
    GUI sessions. Pair with system-fast-shutdown.conf for a 10-s timeout
    cap (otherwise hung services can stretch shutdown to 90 s).
  - we do NOT exit. If poweroff actually succeeds the kernel kills us;
    if it failed silently (polkit, stuck inhibitor, etc.), we stay alive
    so the next rising edge can retry instead of leaving a dead daemon.
"""
import os
import subprocess
import sys
import time

from gpiozero import Button

SHUTDOWN_REQ_PIN = 27   # BCM GPIO27 = Pi header pin 13
POLL_S = 0.02


def trigger(reason):
    print(f"shutdown signal: {reason} → graceful poweroff", flush=True)
    os.sync()
    rc = subprocess.run(["systemctl", "poweroff", "-f"], check=False).returncode
    print(f"systemctl returned {rc}", flush=True)


def main():
    if os.geteuid() != 0:
        print("must run as root (poweroff requires it)", file=sys.stderr)
        sys.exit(1)

    btn = Button(SHUTDOWN_REQ_PIN, pull_up=False, bounce_time=0.05)
    print(f"watching GPIO{SHUTDOWN_REQ_PIN} (polling) for shutdown request "
          f"from ATtiny", flush=True)

    if btn.is_pressed:
        trigger("line already HIGH at startup")

    prev = btn.is_pressed
    while True:
        cur = btn.is_pressed
        if cur and not prev:
            trigger("rising edge")
            # After triggering, wait until line returns LOW before arming
            # the next edge detect. ATtiny holds PB4 HIGH for ~30 s during
            # its wait_pi_off, so without this we'd retrigger every cycle.
            while btn.is_pressed:
                time.sleep(POLL_S)
        prev = cur
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
