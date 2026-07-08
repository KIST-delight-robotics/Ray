#!/usr/bin/env bash
set -euo pipefail

GPIOCHIP="/dev/gpiochip4"
if [[ ! -e "$GPIOCHIP" ]]; then
  GPIOCHIP="/dev/gpiochip0"
fi

# Hold Pi physical pin 18 (GPIO24) LOW while the Pi is running.
# This makes the shutdown ACK protocol explicit:
# - LOW  = normal running / shutdown not complete yet
# - HIGH = shutdown complete, ATtiny85 may cut power
exec /usr/bin/gpioset -m signal "$GPIOCHIP" 24=0
