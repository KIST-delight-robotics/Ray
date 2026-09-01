#!/usr/bin/env bash
# Install OS_LED Pi-side daemons. Run as: sudo bash pi/install.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

if [[ $EUID -ne 0 ]]; then
    echo "must run as root: sudo bash $0" >&2
    exit 1
fi

echo "==> apt packages"
apt-get install -y python3-spidev python3-gpiozero

echo "==> scripts → /usr/local/bin"
install -m 755 "$HERE/os_led_display.py"  /usr/local/bin/os_led_display.py
install -m 755 "$HERE/os_led_poweroff.py" /usr/local/bin/os_led_poweroff.py

echo "==> systemd units"
install -m 644 "$HERE/os-led-display.service"  /etc/systemd/system/os-led-display.service
install -m 644 "$HERE/os-led-poweroff.service" /etc/systemd/system/os-led-poweroff.service

echo "==> logind override"
mkdir -p /etc/systemd/logind.conf.d
install -m 644 "$HERE/logind-override.conf" /etc/systemd/logind.conf.d/10-os-led.conf

echo "==> system shutdown timeout cap"
mkdir -p /etc/systemd/system.conf.d
install -m 644 "$HERE/system-fast-shutdown.conf" /etc/systemd/system.conf.d/10-os-led.conf

echo "==> poweroff ACK shutdown hook (tells ATtiny 'real poweroff, not reboot')"
install -m 755 "$HERE/os-led-poweroff-ack" /usr/lib/systemd/system-shutdown/os-led-poweroff-ack

echo "==> reload + enable + restart"
systemctl daemon-reexec
systemctl daemon-reload
systemctl enable --now os-led-display.service os-led-poweroff.service
systemctl restart os-led-display.service os-led-poweroff.service

echo "==> done. status:"
systemctl status os-led-display os-led-poweroff --no-pager
