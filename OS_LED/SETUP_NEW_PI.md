# OS_LED — 새 라즈베리파이 이식 가이드 (단일 정리본)

> 이 문서 하나 + 같은 폴더의 파일들로 다른 Raspberry Pi에 OS_LED 시스템 전체를 재현한다.
> 설계/회로/프로토콜 원리는 `ARCHITECTURE.md`, 펌웨어 플래시 트러블슈팅 상세는 §6 참고.
> 이 zip은 **현재 동작 중인 시스템의 실제 설치본과 일치함이 확인된** 스냅샷이다.

---

## 0. 현재 기준 환경

| 항목 | 값 |
|---|---|
| 보드 | Raspberry Pi 5 |
| OS | Ubuntu 24.04 LTS (aarch64) |
| Python | 3.12 |
| 커널 GPIO 칩 | `/dev/gpiochip4` (Pi 5 RP1). **Pi 4 이하면 `/dev/gpiochip0`** — 플래시 명령의 칩 번호 주의 |
| avrdude | `/usr/bin/avrdude` **7.1** (linuxspi 전용 — §6 참고) |

---

## 1. 옮겨야 할 것 한눈에

1. **프로젝트 디렉토리** `~/OS_LED/` 전체 (이 안에 데몬·유닛·펌웨어 소스·hex·설치 스크립트가 다 있음)
2. **apt 패키지** (§3)
3. **부팅 설정** `config.txt`의 `dtparam=spi=on` (§4)
4. **Pi-side 데몬 설치** — `pi/install.sh` 한 방 (§5)
5. **ATtiny85 펌웨어 플래시** — 새 칩을 쓸 때만 (§6)

`legacy/` 폴더는 **현재 Pi에 깔려있는 구설계 잔재**다 — 새 Pi엔 설치하지 않는다(§8).

---

## 2. 디렉토리 구성

```
OS_LED/
├── SETUP_NEW_PI.md          # 이 문서 (이식 가이드)
├── ARCHITECTURE.md          # 설계/회로/프로토콜/상태머신 전체 문서
├── firmware/                # ATtiny85 펌웨어
│   ├── main/main.c          # 상태머신 본체 (12V 자동부팅·reboot 처리 포함 — §7)
│   ├── common/pins.h        # 핀 정의
│   ├── common/ws2812.h      # bit-bang WS2812 드라이버
│   ├── Makefile             # build / fuses / flash 타깃
│   ├── flash.sh             # 안전 플래시 스크립트 (서비스 정지 포함)
│   └── firmware.hex         # ★ 빌드 산출물 — 그대로 플래시 가능 (최신 동작 반영)
├── pi/                      # Pi-side 데몬 일체 (= 실제 설치본과 동일)
│   ├── install.sh           # ★ 설치 원클릭
│   ├── os_led_display.py    # READY HIGH + SPI 무지개 (60fps)
│   ├── os_led_poweroff.py   # GPIO27 폴링 → poweroff
│   ├── os-led-display.service / os-led-poweroff.service
│   ├── logind-override.conf         # HandlePowerKey=ignore / LongPress=poweroff
│   ├── system-fast-shutdown.conf    # DefaultTimeoutStopSec=10s
│   └── blink_gpio10.py      # DIN 소유권 검증 툴
└── legacy/                  # 현재 Pi의 구설계(GPIO23/24) 잔재 — 새 Pi 미설치 (§8)
    ├── attiny_ack_idle.sh / attiny_ack_hold.sh / attiny_shutdown_monitor.py
    ├── attiny_shutdown_ack             # 종료 훅
    └── attiny_ack_idle.service / attiny_shutdown_monitor.service
```

복사:
```bash
# 기존 Pi에서
tar czf os_led.tar.gz -C ~ OS_LED
scp os_led.tar.gz <newpi>:~
# 새 Pi에서
tar xzf os_led.tar.gz -C ~
```

---

## 3. apt 패키지

```bash
# 런타임 (데몬 필수)
sudo apt install -y python3-spidev python3-gpiozero python3-lgpio gpiod
# AVR 툴체인 (펌웨어 빌드/플래시할 때만)
sudo apt install -y gcc-avr avr-libc binutils-avr avrdude
```
- 런타임: `python3-spidev`(WS2812 SPI), `python3-gpiozero`+`python3-lgpio`(GPIO), `gpiod`(gpioset/gpioget)
- `avrdude` **7.1**(`/usr/bin/avrdude`) — 6.x는 Pi 5 RESET 핀 번호 제한으로 실패. 반드시 7.1.

---

## 4. 부팅 설정 (`/boot/firmware/config.txt`)

필수는 하나:
```
dtparam=spi=on
```
확인:
```bash
grep spi /boot/firmware/config.txt    # dtparam=spi=on
ls /dev/spidev0.0                     # 부팅 후 존재해야 함
```

---

## 5. Pi-side 데몬 설치

```bash
sudo bash ~/OS_LED/pi/install.sh
```

`install.sh`가 하는 일:

| 대상 | 내용 |
|---|---|
| `/usr/local/bin/os_led_display.py` (755) | READY(GPIO17) HIGH + SPI 무지개 |
| `/usr/local/bin/os_led_poweroff.py` (755) | GPIO27 20ms 폴링 → `systemctl poweroff -f` |
| `/etc/systemd/system/os-led-display.service` | Restart=always |
| `/etc/systemd/system/os-led-poweroff.service` | Restart=always (root) |
| `/etc/systemd/logind.conf.d/10-os-led.conf` | `HandlePowerKey=ignore`, `HandlePowerKeyLongPress=poweroff` |
| `/etc/systemd/system.conf.d/10-os-led.conf` | `DefaultTimeoutStopSec=10s` (종료 90초 늘어짐 방지) |
| systemd | daemon-reload + enable --now + restart |

확인:
```bash
systemctl status os-led-display os-led-poweroff --no-pager
```

---

## 6. ATtiny85 펌웨어 플래시 — ⚠️ 실전 트러블슈팅 포함

> 기존 칩을 그대로 새 Pi에 옮기면 이 절은 불필요. 새 칩을 굽거나 펌웨어 수정 시에만.

### 6.1 ISP 배선 (Pi 5 헤더 ↔ ATtiny85 DIP-8)

ATtiny 노치/점을 위로 두고, 그 왼쪽 위가 핀1.

| Pi 헤더 핀 | Pi 신호 | ATtiny 핀 | ATtiny 신호 |
|---|---|---|---|
| 1 | **3V3** | 8 | VCC (전원 — 5V 아님!) |
| 6 | GND | 4 | GND |
| 15 | GPIO22 | 1 | RESET |
| 19 | GPIO10 (MOSI) | 5 | PB0 |
| 21 | GPIO9 (MISO) | 6 | PB1 |
| 23 | GPIO11 (SCK) | 7 | PB2 |

- **전원은 Pi 3V3(핀1)** — ISP 중 5V로 돌리면 Pi GPIO에 5V 역류. 별도 외부전원 없음(VCC선 꽂는 순간이 전원 인가).
- **배선 순서**: GND 먼저 → 신호 4선 → 3V3 마지막 (글리치 방지).
- **런타임 케이블은 모두 분리**(연결돼 있으면 ISP 깨짐): TTP223 OUT(PB0), NPN base(PB2), WS2812 DIN(PB1), J2.

### 6.2 플래시 명령 (이 환경에서 검증된 방식)

**중요 — 이 Pi의 avrdude 7.1엔 `linuxgpio`가 없다** (유효 프로그래머에 `linuxspi`만 있음).
`-c linuxgpio`는 `cannot find programmer id linuxgpio`로 실패하니 **linuxspi만 사용**한다.

```bash
sudo systemctl stop os-led-display   # /dev/spidev0.0 점유 해제 (필수)

# 1) 시그니처 확인 — 0x1e930b 나와야 진행
sudo /usr/bin/avrdude -c linuxspi -P /dev/spidev0.0:/dev/gpiochip4:22 -B 250 -i 100 -x disable_no_cs -p t85

# 2) fuse (신품 칩 1회 — 16MHz PLL, 안 하면 LED 색 깨짐)
sudo /usr/bin/avrdude -c linuxspi -P /dev/spidev0.0:/dev/gpiochip4:22 -B 250 -i 100 -x disable_no_cs -p t85 \
    -U lfuse:w:0xE1:m -U hfuse:w:0xDF:m -U efuse:w:0xFF:m

# 3) erase
sudo /usr/bin/avrdude -c linuxspi -P /dev/spidev0.0:/dev/gpiochip4:22 -B 250 -i 100 -x disable_no_cs -p t85 -e

# 4) write + verify
sudo /usr/bin/avrdude -c linuxspi -P /dev/spidev0.0:/dev/gpiochip4:22 -B 250 -i 100 -x disable_no_cs -p t85 \
    -D -U flash:w:/home/limdaemin/OS_LED/firmware/firmware.hex
```

> `flash.sh`는 `-B 100`을 쓴다. 안정적인 배선이면 `make && sudo bash flash.sh`로 충분하지만,
> **신품 칩 + 브레드보드 배선에선 `-B 100`이 자주 실패**한다(아래 참고). 그럴 땐 위의 `-B 250` 수동 명령을 쓴다.

### 6.3 시그니처가 안 읽힐 때 (이번 이식에서 실제 겪은 것)

증상별 원인:
- **`0x000102` 또는 시그니처 깨짐** → MISO(핀21↔PB1 핀6) 접촉 불량 / 칩 방향 뒤집힘 / 전원 부족
- **`AVR device not responding`** → RESET·SCK·전원 경로 끊김, 또는 직전 Ctrl-C로 RP1 SPI 드라이버 락
- **증상이 시도마다 바뀜** → 브레드보드 점퍼 접촉 불량(가장 흔함)

해결 순서:
1. **칩 방향** 확인(노치 위, 핀1=RESET). 거꾸로면 매번 실패.
2. **전원 측정**(멀티미터): ATtiny 핀8(VCC)–핀4(GND) 사이 **3.3V** 떠야 함.
3. **도통 측정**(Pi 끄고, 칩 다리에 직접 프로브): 6선 각각 삑소리. 특히 MISO·SCK.
4. **브레드보드 함정**: 점퍼와 칩 다리가 같은 행(5홀 그룹)에 꽂혔는지, 칩이 중앙 홈을 걸쳐 앉았는지.
5. **클럭을 더 낮춰서**: `-B 100` 실패 시 `-B 250`(≈4kHz)으로. 신품 칩은 1MHz라 더 느린 ISP가 안전.
6. **Ctrl-C 금지**: 쓰기/읽기 중 중단하면 RP1 SPI 드라이버가 잠겨 `not responding`이 계속됨 → `sudo reboot`로 해제.

### 6.4 플래시 후

ISP 케이블 6선 제거 → 런타임 배선 복구 → 12V 인가:
- ATtiny VCC(핀8) ← **상시 5V** (Pi 꺼져도 유지되는 레일. 3V3 아님!)
- PB0←TTP223 OUT / PB1↔WS2812 DIN / PB2→NPN base→J2 / PB3←GPIO17(외부 10k 풀다운) / PB4→GPIO27(10k+20k 분압) / GND 공통
- `sudo systemctl start os-led-display`

⚠️ ISP의 3V3선을 안 뽑고 런타임 5V를 연결하면 VCC 충돌. ISP 6선 모두 제거 후 런타임 배선.

---

## 7. 펌웨어 동작 요약 (터치/부팅/종료)

| 상황 | 트리거 | J2 | 결과 |
|---|---|---|---|
| **켜기** | IDLE에서 터치 **0.5초** | 펄스 | fade-up + 호흡 → Pi 부팅 → 무지개 |
| **끄기** | RUNNING에서 터치 **2.0초 연속** | — | 종료 요청 → Pi off → fade-out |
| **12V 자동부팅** | 전원 인가(터치 없음) | 안 함 | fade-up + 호흡 → Pi 자동부팅 → 무지개 |
| **sudo reboot** | 터치 없이 Pi 다운 | 안 함 | 종료 호흡 → 끔 → 부팅 fade-up → 무지개 |
| **비상 리셋** | 어느 상태든 **5초 홀드** | — | ATtiny 소프트 리셋(재캘) |

- **끄기는 2초 연속**이어야 한다. 2초 미만에 손 떼면 종료 요청을 안 보낸다(실수 종료 방지, 의도된 설계). 짧게 끊어 누른 누적은 무효 — 한 번에 2초 유지해야 함.
- 끄는 시간을 바꾸려면 `firmware/main/main.c`의 `SHUTDOWN_TOUCH_HOLD_MS`(기본 2000ms) 수정 후 재빌드.
- 12V 자동부팅·reboot 시 J2를 안 누르는 이유, 90초 grace 등 상세 동작과 한계는 `ARCHITECTURE.md` §5 "부팅 트리거 3종과 J2 정책" 참고.

---

## 8. ⚠️ `legacy/` — 새 Pi에 설치하지 말 것

현재 Pi엔 **구버전 설계(GPIO23/24 핸드셰이크)의 서비스가 아직 active 상태로 남아 있다.**
현행 설계는 GPIO17(READY)/GPIO27(SHUTDOWN_REQ)만 쓰므로, `legacy/`의 파일들은 새 Pi에 **설치하지 않는다.**

| 레거시 항목 | 원래 위치 | 구설계에서 하던 일 |
|---|---|---|
| `attiny_ack_idle.service` + `attiny_ack_idle.sh` | `/etc/systemd/system/`, `/usr/local/bin/` | GPIO24를 LOW 유지 (구 ACK) |
| `attiny_shutdown_monitor.service` + `.py` | 〃 | GPIO23 폴링 → poweroff (현행 `os-led-poweroff`가 GPIO27로 대체) |
| `attiny_shutdown_ack` | `/usr/lib/systemd/system-shutdown/` | 종료 끝에 GPIO24 HIGH 0.5s (구 ACK) |
| `attiny_ack_hold.sh` | `/usr/local/bin/` | 위의 수동 버전 |

새 Pi엔 `pi/install.sh`만 돌리면 되고 `legacy/`는 무시한다.

기존 Pi에서 정리하려면(선택):
```bash
sudo systemctl disable --now attiny_ack_idle attiny_shutdown_monitor
sudo rm /etc/systemd/system/attiny_ack_idle.service /etc/systemd/system/attiny_shutdown_monitor.service
sudo rm /usr/local/bin/attiny_ack_idle.sh /usr/local/bin/attiny_ack_hold.sh /usr/local/bin/attiny_shutdown_monitor.py
sudo rm /usr/lib/systemd/system-shutdown/attiny_shutdown_ack
# system.conf.d 중복 dropin(같은 내용)도 하나로:
sudo rm /etc/systemd/system.conf.d/10-os-led-fast-shutdown.conf
sudo systemctl daemon-reload
```
또 `/usr/local/bin/avrdude`(수동 빌드 6.1, Pi 5에서 동작 안 함)와 `/usr/local/bin/gpio`(WiringPi)도 현행 미사용 — 새 Pi에 복사 금지.

---

## 9. 새 Pi 전체 절차 (체크리스트)

```bash
# 1. 디렉토리 복사 (§2)
# 2. 패키지
sudo apt install -y python3-spidev python3-gpiozero python3-lgpio gpiod
sudo apt install -y gcc-avr avr-libc binutils-avr avrdude    # 펌웨어 작업 시만
# 3. SPI 확인
grep spi /boot/firmware/config.txt
# 4. 데몬 설치
sudo bash ~/OS_LED/pi/install.sh
# 5. (새 ATtiny일 때만) 플래시 — §6, 런타임 케이블 분리 + linuxspi -B 250
# 6. 동작 검증
systemctl status os-led-display os-led-poweroff --no-pager
```

### 동작 검증 시나리오
1. **12V 인가** → 1.5초 후 fade-up + 호흡 → Pi 자동부팅 완료 시 무지개
2. **터치 0.5초** (Pi off 상태) → J2로 부팅 → 무지개
3. **터치 2초** (Pi 켜짐) → 종료 요청 → Pi off → fade-out
4. **sudo reboot** → 종료 호흡 → 끔 → 부팅 fade-up → 무지개
5. DIN 소유권 검증: `sudo systemctl stop os-led-display && python3 ~/OS_LED/pi/blink_gpio10.py` (2Hz 토글) → 끝나면 서비스 재시작

### 하드웨어 체크 (회로 새로 구성 시 — 상세는 ARCHITECTURE.md §7)
- ATtiny·WS2812는 **상시 5V** (Pi 꺼져도 유지)
- READY(GPIO17↔PB3) **외부 10k 풀다운 필수**
- SHUTDOWN_REQ(PB4→GPIO27) **10k+20k 분압**
- ATtiny VCC 100nF, WS2812 5V 1000µF, DIN 직렬 330Ω 권장
- 5V 공급기는 부팅/종료 순백 피크 ~1.4A 감당
