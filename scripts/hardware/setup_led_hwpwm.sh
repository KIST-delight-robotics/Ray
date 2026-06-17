#!/usr/bin/env bash
# LED 밝기를 RP1 하드웨어 PWM(GPIO13 = PWM1)으로 전환하기 위한 1회 시스템 설정.
# softPwm(~100Hz, 플리커)을 대체 → 캐리어 20kHz 무플리커.
#
# 하는 일:
#   1) /boot/firmware/config.txt 에 pwm-2chan 오버레이 추가 (GPIO12=PWM0, GPIO13=PWM1)
#   2) gpio 그룹 생성 + 현재 사용자 추가 (Ray가 비root로 PWM 제어 가능하게)
#   3) 부팅 시 pwmchip0 채널1 export+period+enable+권한을 잡는 systemd 서비스 설치/활성화
#   4) 재부팅 안내
#
# 사용: sudo bash scripts/hardware/setup_led_hwpwm.sh   (그 후 sudo reboot)
set -euo pipefail

PWM_HZ=20000                       # LED PWM 캐리어 주파수 (20kHz → 무플리커·무버즈)
PERIOD_NS=$(( 1000000000 / PWM_HZ ))   # = 50000 ns
CONFIG=/boot/firmware/config.txt
OVERLAY='dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4'
TARGET_USER="${SUDO_USER:-$(id -un)}"

if [ "$(id -u)" -ne 0 ]; then echo "sudo로 실행하세요: sudo bash $0"; exit 1; fi

echo "[1/4] config.txt 오버레이 추가"
if grep -qF "$OVERLAY" "$CONFIG"; then
  echo "  - 이미 있음, 건너뜀"
else
  printf '\n# LED 밝기 하드웨어 PWM (GPIO13 = PWM1)\n%s\n' "$OVERLAY" >> "$CONFIG"
  echo "  - 추가함: $OVERLAY"
fi

echo "[1b/4] PWM0 클럭 부모 수정 오버레이 설치 (clk_pwm0 rate=0 → xosc 50MHz)"
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SELF_DIR/pwm0-clk-fix.dtbo" ]; then
  cp "$SELF_DIR/pwm0-clk-fix.dtbo" /boot/firmware/overlays/
  if grep -q '^dtoverlay=pwm0-clk-fix' "$CONFIG"; then echo "  - dtoverlay 이미 있음"; else
    echo 'dtoverlay=pwm0-clk-fix' >> "$CONFIG"; echo "  - dtoverlay=pwm0-clk-fix 추가"; fi
else
  echo "  - 경고: pwm0-clk-fix.dtbo 없음. dtc -@ -I dts -O dtb -o 로 먼저 컴파일 필요"
fi

echo "[2/4] gpio 그룹 + 사용자($TARGET_USER) 추가"
groupadd -f gpio
usermod -aG gpio "$TARGET_USER"

echo "[3/4] 부팅 PWM 준비 스크립트 + systemd 서비스 설치 (${PWM_HZ}Hz)"
cat > /usr/local/bin/led-pwm-setup.sh <<EOF
#!/bin/sh
# 부팅 시 pwmchip0 채널1(GPIO13)을 export하고 period/enable/권한 설정.
CHIP=/sys/class/pwm/pwmchip0
[ -e "\$CHIP/pwm1" ] || echo 1 > "\$CHIP/export"
sleep 0.3
echo $PERIOD_NS > "\$CHIP/pwm1/period"
echo 0          > "\$CHIP/pwm1/duty_cycle"
echo 1          > "\$CHIP/pwm1/enable"
chgrp gpio "\$CHIP/pwm1/duty_cycle" "\$CHIP/pwm1/period" "\$CHIP/pwm1/enable"
chmod g+rw "\$CHIP/pwm1/duty_cycle" "\$CHIP/pwm1/period" "\$CHIP/pwm1/enable"
EOF
chmod +x /usr/local/bin/led-pwm-setup.sh

cat > /etc/systemd/system/led-pwm.service <<'EOF'
[Unit]
Description=LED hardware PWM (GPIO13/PWM1) export+enable
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/led-pwm-setup.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable led-pwm.service

echo "[4/4] 완료. 적용하려면 재부팅하세요:  sudo reboot"
echo "      재부팅 후 확인:  cat /sys/class/pwm/pwmchip0/pwm1/period   (=$PERIOD_NS 면 정상)"
