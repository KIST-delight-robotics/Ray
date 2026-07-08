#!/usr/bin/env python3
import subprocess
import time
from pathlib import Path

REQ_LINE = "23"
DEBOUNCE_SECONDS = 0.2
POLL_SECONDS = 0.05


def detect_gpiochip() -> str:
    for candidate in ("/dev/gpiochip4", "/dev/gpiochip0"):
        if Path(candidate).exists():
            return candidate
    return "/dev/gpiochip0"


GPIOCHIP = detect_gpiochip()


def read_gpio() -> bool:
    result = subprocess.run(
        ["/usr/bin/gpioget", GPIOCHIP, REQ_LINE],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "1"


def main() -> None:
    print(f"attiny shutdown monitor started on {GPIOCHIP} line {REQ_LINE}", flush=True)

    while True:
        if read_gpio():
            print("shutdown request detected", flush=True)
            time.sleep(DEBOUNCE_SECONDS)
            if read_gpio():
                print("calling systemctl poweroff", flush=True)
                subprocess.run(["/usr/bin/systemctl", "poweroff"], check=False)
                time.sleep(5)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
