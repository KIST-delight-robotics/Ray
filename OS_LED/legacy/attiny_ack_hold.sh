#!/usr/bin/env bash
set -euo pipefail

# Legacy helper: the default install path now uses
# /usr/lib/systemd/system-shutdown/attiny_shutdown_ack
# so the ACK is asserted at the very end of poweroff/halt.

GPIOCHIP="/dev/gpiochip4"
if [[ ! -e "$GPIOCHIP" ]]; then
  GPIOCHIP="/dev/gpiochip0"
fi

ACK_HOLD_USEC=500000

# Hold Pi physical pin 18 (GPIO24) HIGH for a fixed window.
# ATtiny85 physical pin 7 treats this HIGH level as "Pi shutdown complete".
exec /usr/bin/gpioset -m time -u "$ACK_HOLD_USEC" "$GPIOCHIP" 24=1
