#!/usr/bin/env bash
# Flash the ATtiny85 firmware. Run as: sudo bash firmware/flash.sh
#
# Prerequisites (handled by user):
#   - ISP cables wired: Pi 3V3, GND, GPIO22/10/9/11 → ATtiny VCC, GND, RESET, PB0/PB1/PB2
#   - Runtime cables disconnected from ATtiny (TTP223 OUT, NPN base, WS2812 DIN, 5V, J2)
#   - firmware.hex built (`make` in firmware/)
#
# What this script does:
#   1. Stops os-led-display.service so it releases /dev/spidev0.0
#   2. Reads ATtiny signature (sanity check)
#   3. Erases the chip
#   4. Writes firmware.hex and verifies
#
# After this finishes:
#   - Disconnect ISP cables, restore runtime wiring
#   - `sudo systemctl start os-led-display` to bring the display service back

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
HEX="$HERE/firmware.hex"
AVRDUDE="/usr/bin/avrdude"
ARGS=(-c linuxspi -P /dev/spidev0.0:/dev/gpiochip4:22 -B 100 -i 100 -x disable_no_cs -p t85)

if [[ $EUID -ne 0 ]]; then
    echo "must run as root: sudo bash $0" >&2
    exit 1
fi

if [[ ! -f "$HEX" ]]; then
    echo "no firmware.hex at $HEX — run 'make' in firmware/ first" >&2
    exit 1
fi

echo "==> stopping os-led-display (releasing /dev/spidev0.0)"
systemctl stop os-led-display.service 2>/dev/null || true

echo
echo "==> signature read"
"$AVRDUDE" "${ARGS[@]}"

echo
echo "==> erase"
"$AVRDUDE" "${ARGS[@]}" -e

echo
echo "==> write + verify"
"$AVRDUDE" "${ARGS[@]}" -D -U "flash:w:$HEX"

echo
echo "==> flash complete."
echo "    Now: disconnect ISP cables, restore runtime wiring (5V, TTP223, NPN, J2, WS2812)."
echo "    Then: sudo systemctl start os-led-display"
