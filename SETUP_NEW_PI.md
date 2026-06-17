# OS_LED — 새 라즈베리파이 이식 가이드 (단일 파일 정리본)

> 이 문서 하나로 다른 Raspberry Pi에 OS_LED 시스템 전체를 재현할 수 있도록,
> **현재 Pi에 설치된 모든 것 + 설치/플래시 파이프라인**을 정리했다.
> 시스템 설계/회로/펌웨어 동작 원리는 `ARCHITECTURE.md` 참고 — 여기는 "세팅 재현"에만 집중한다.

---

## 0. 현재 기준 환경

| 항목 | 값 |
|---|---|
| 보드 | Raspberry Pi 5 |
| OS | Ubuntu 24.04 LTS (aarch64) |
| Python | 3.12 (시스템 기본) |
| 커널 GPIO 칩 | `/dev/gpiochip4` (Pi 5 RP1. **Pi 4 이하면 `/dev/gpiochip0`** — 스크립트/명령의 칩 번호 주의) |

⚠️ Pi 5가 아닌 보드로 옮기면 ISP 플래시 명령(`/dev/gpiochip4`, RP1 SPI 함정 우회 옵션)이 달라진다. §6의 주석 참고.

---

## 1. 옮겨야 할 것 한눈에

1. **프로젝트 디렉토리** `~/OS_LED/` 전체 (이 안에 데몬·유닛·펌웨어 소스·설치 스크립트가 다 있음)
2. **apt 패키지** (§3)
3. **부팅 설정** `config.txt`의 `dtparam=spi=on` (§4)
4. **Pi-side 데몬 설치** — `pi/install.sh` 한 방 (§5)
5. **AVR 툴체인 + 펌웨어 플래시** — 새 ATtiny85를 쓸 경우에만 (§6)

**옮기지 말아야 할 것(레거시)** 도 있다 — §7 필독.

---

## 2. 프로젝트 디렉토리 복사

```bash
# 기존 Pi에서
tar czf os_led.tar.gz -C ~ OS_LED
scp os_led.tar.gz <newpi>:~

# 새 Pi에서
tar xzf os_led.tar.gz -C ~
```

디렉토리 구성:

```
OS_LED/
├── ARCHITECTURE.md          # 설계/회로/프로토콜 전체 문서 (필독)
├── SETUP_NEW_PI.md          # 이 문서
├── firmware/                # ATtiny85 펌웨어
│   ├── Makefile             # build / fuses / flash 타깃
│   ├── flash.sh             # 안전 플래시 스크립트 (서비스 정지 포함)
│   ├── main/main.c          # 상태머신 본체
│   ├── common/pins.h        # 핀 정의
│   ├── common/ws2812.h      # bit-bang WS2812 드라이버
│   └── firmware.hex         # 빌드 산출물 (그대로 플래시 가능)
└── pi/                      # Pi-side 데몬 일체
    ├── install.sh           # ★ 설치 원클릭 스크립트
    ├── os_led_display.py    # READY HIGH + SPI 무지개 (60fps)
    ├── os_led_poweroff.py   # GPIO27 폴링 → poweroff
    ├── os-led-display.service
    ├── os-led-poweroff.service
    ├── logind-override.conf         # HandlePowerKey=ignore / LongPress=poweroff
    ├── system-fast-shutdown.conf    # DefaultTimeoutStopSec=10s
    └── blink_gpio10.py      # DIN 소유권 검증 툴
```

---

## 3. apt 패키지

### 런타임 (데몬 구동에 필수)
```bash
sudo apt install -y python3-spidev python3-gpiozero python3-lgpio gpiod
```
- `python3-spidev` 3.6 — WS2812 SPI 인코딩 출력
- `python3-gpiozero` 2.0.1 + `python3-lgpio` 0.2 — READY/SHUTDOWN_REQ GPIO 제어 (Pi 5에서 gpiozero의 lgpio 백엔드)
- `gpiod` — `gpioset`/`gpioget` CLI (디버깅용. 레거시 스크립트도 사용)

> `install.sh`가 `python3-spidev python3-gpiozero`는 자동 설치한다. lgpio는 보통 의존성으로 따라오지만 명시해두는 게 안전.

### AVR 툴체인 (펌웨어 빌드/플래시할 때만)
```bash
sudo apt install -y gcc-avr avr-libc binutils-avr avrdude
```
- `avrdude` **7.1** (`/usr/bin/avrdude`) — **반드시 이 버전**. 6.x는 Pi 5에서 RESET 핀 번호 제한(0–31)으로 실패.

---

## 4. 부팅 설정 (`/boot/firmware/config.txt`)

필수 항목은 하나뿐:
```
dtparam=spi=on
```
`os_led_display.py`가 `/dev/spidev0.0`을 쓰기 위함. Ubuntu 24.04 Pi 이미지에는 기본 활성화돼 있으니 확인만:
```bash
grep spi /boot/firmware/config.txt   # dtparam=spi=on 있어야 함
ls /dev/spidev0.0                    # 부팅 후 존재 확인
```
그 외 현재 Pi의 config.txt는 전부 배포판 기본값 (i2c/audio/kms 등 — OS_LED와 무관).

---

## 5. Pi-side 데몬 설치 — 파이프라인

원클릭:
```bash
sudo bash ~/OS_LED/pi/install.sh
```

이 스크립트가 하는 일 (수동으로 할 때의 전체 목록):

| 단계 | 대상 | 내용 |
|---|---|---|
| apt | `python3-spidev`, `python3-gpiozero` | 설치 |
| 스크립트 | `/usr/local/bin/os_led_display.py` (755) | READY(GPIO17) HIGH + SPI 무지개 |
| 스크립트 | `/usr/local/bin/os_led_poweroff.py` (755) | GPIO27 20ms 폴링 → `systemctl poweroff -f` |
| 유닛 | `/etc/systemd/system/os-led-display.service` | Restart=always |
| 유닛 | `/etc/systemd/system/os-led-poweroff.service` | Restart=always (root) |
| drop-in | `/etc/systemd/logind.conf.d/10-os-led.conf` | `HandlePowerKey=ignore`, `HandlePowerKeyLongPress=poweroff` — J2 펄스가 logind 기본 동작과 충돌하지 않게 |
| drop-in | `/etc/systemd/system.conf.d/10-os-led.conf` | `DefaultTimeoutStopSec=10s` — 종료가 90초 늘어지는 것 방지 (ATtiny 종료 타임아웃 30s 안에 꺼져야 함) |
| systemd | `daemon-reexec` + `daemon-reload` + `enable --now` + `restart` | 두 서비스 활성화 |

설치 후 확인:
```bash
systemctl status os-led-display os-led-poweroff --no-pager
```

---

## 6. ATtiny85 펌웨어 빌드 & 플래시 — 파이프라인

> **기존에 쓰던 ATtiny85 칩을 그대로 새 Pi에 연결한다면 이 절 전체가 불필요.**
> 새 칩을 굽거나 펌웨어를 수정할 때만 해당.

### 6.1 ISP 배선 (Pi 5 헤더 ↔ ATtiny85 DIP-8)

| Pi 헤더 핀 | Pi 신호 | ATtiny 핀 | ATtiny 신호 |
|---|---|---|---|
| 1 | 3V3 | 8 | VCC |
| 6 | GND | 4 | GND |
| 15 | GPIO22 | 1 | RESET |
| 19 | GPIO10 (MOSI) | 5 | PB0 |
| 21 | GPIO9 (MISO) | 6 | PB1 |
| 23 | GPIO11 (SCK) | 7 | PB2 |

**플래시 전 ATtiny에서 분리해야 할 런타임 케이블** (안 빼면 SPI 깨지거나 Pi가 도중에 꺼짐):
- TTP223 OUT (pin 5 — MOSI와 충돌)
- NPN base (pin 7 — SCK 토글이 J2를 펄스시켜 **Pi가 플래시 중 종료됨**)
- WS2812 DIN (pin 6 — MISO 부하)
- +5V (pin 8 — 대신 Pi 3V3 사용. 5V 상태로 ISP 하면 Pi GPIO에 5V 역류)
- J2 케이블 (NPN 안전벨트)

PB3(←GPIO17), PB4 분압(→GPIO27)은 그대로 둬도 됨 — ISP가 안 쓰는 핀.

### 6.2 신품 칩 1회: fuse
신품은 8MHz/CKDIV8 상태라 fuse를 안 맞추면 WS2812 타이밍이 전부 깨진다:
```bash
cd ~/OS_LED/firmware && make fuses
# lfuse=0xE1 (16MHz PLL, CKDIV8 off) / hfuse=0xDF / efuse=0xFF
```

### 6.3 빌드 + 플래시 (표준 경로)
```bash
cd ~/OS_LED/firmware && make && sudo bash flash.sh
# 완료 후: ISP 케이블 제거 → 런타임 배선 복구 →
sudo systemctl start os-led-display
```

`flash.sh`가 하는 일과 **그렇게 하는 이유** (수동 avrdude 시에도 동일하게 지켜야 함):

1. **`systemctl stop os-led-display` 먼저** — 이 서비스가 `/dev/spidev0.0`에 60fps로 쓰고 있어서, 안 멈추면 ISP 트래픽과 섞여 시그니처 깨짐(`0x000102`)·쓰기 중 행·RP1 드라이버 락(재부팅 필요)이 난다. **수동 플래시 때 가장 잊기 쉬운 단계.**
2. 시그니처 확인 — `0x1e930b`(ATtiny85) 나와야 진행.
3. erase와 write를 **별도 avrdude 호출로 분리** + `-B 100 -i 100`(저속) — Pi 5 RP1 SPI가 paged-write 버스트 첫 바이트들을 흘리는 하드웨어 함정 우회. 단일 호출이나 고속이면 verify 실패.
4. 쓰기 도중 **Ctrl-C 절대 금지** — RP1 SPI 드라이버가 락돼서 재부팅 전까지 SPI 전체 불능.

수동 명령 (flash.sh 내용과 동일):
```bash
sudo systemctl stop os-led-display
sudo /usr/bin/avrdude -c linuxspi -P /dev/spidev0.0:/dev/gpiochip4:22 -B 100 -i 100 -x disable_no_cs -p t85          # 시그니처
sudo /usr/bin/avrdude -c linuxspi -P /dev/spidev0.0:/dev/gpiochip4:22 -B 100 -i 100 -x disable_no_cs -p t85 -e       # erase
sudo /usr/bin/avrdude -c linuxspi -P /dev/spidev0.0:/dev/gpiochip4:22 -B 100 -i 100 -x disable_no_cs -p t85 -D -U flash:w:firmware.hex
```
> Pi 4 이하에서는 `-P`의 `/dev/gpiochip4`를 `/dev/gpiochip0`으로 변경. RP1 함정(-B 100, erase/write 분리)은 Pi 5 전용 우회이므로 Pi 4에선 `-B 10` 단일 호출도 동작할 수 있으나, 위 명령 그대로 써도 무방(느릴 뿐).

### 6.4 SPI 드라이버가 락됐을 때: linuxgpio 폴백
linuxspi 트랜잭션이 중간에 끊겨 터미널이 행되면(재부팅 필요 상태), 커널 SPI를 우회하는 bit-bang으로 전환. **배선 변경 불필요**, 같은 핀:
```bash
sudo /usr/bin/avrdude -c linuxgpio -x gpiochip=4 -x reset=22 -x sck=11 -x mosi=10 -x miso=9 -i 10 -p t85             # 시그니처
sudo /usr/bin/avrdude -c linuxgpio -x gpiochip=4 -x reset=22 -x sck=11 -x mosi=10 -x miso=9 -i 10 -p t85 -e
sudo /usr/bin/avrdude -c linuxgpio -x gpiochip=4 -x reset=22 -x sck=11 -x mosi=10 -x miso=9 -i 10 -p t85 -D -U flash:w:firmware.hex
```
느리지만(쓰기 ~1분) Ctrl-C 해도 커널이 안 락된다. Ubuntu의 avrdude 7.1에 linuxgpio가 포함돼 있는지는 새 Pi에서 `avrdude -c linuxgpio` 시도로 확인 — 미포함이면 linuxspi만 사용.

---

## 7. ⚠️ 옮기지 말 것 — 현재 Pi에만 있는 레거시

현재 Pi에는 **구버전 설계(GPIO23/24 핸드셰이크)의 잔재가 아직 활성 상태로 남아 있다.**
현행 설계(ARCHITECTURE.md)는 GPIO17(READY)/GPIO27(SHUTDOWN_REQ)만 쓰므로, 아래는 새 Pi에 **설치하지 않는다**:

| 레거시 항목 | 위치 | 하던 일 (구설계) |
|---|---|---|
| `attiny_ack_idle.service` | `/etc/systemd/system/` + `/usr/local/bin/attiny_ack_idle.sh` | GPIO24를 LOW로 유지 (구 ACK 프로토콜) |
| `attiny_shutdown_monitor.service` | `/etc/systemd/system/` + `/usr/local/bin/attiny_shutdown_monitor.py` | GPIO23 폴링 → poweroff (현행 `os-led-poweroff`가 GPIO27로 대체) |
| `attiny_shutdown_ack` | `/usr/lib/systemd/system-shutdown/` | 종료 마지막에 GPIO24 HIGH 0.5s (구 ACK) |
| `attiny_ack_hold.sh` | `/usr/local/bin/` | 위의 수동 버전 |
| `/usr/local/bin/avrdude` | 수동 빌드된 **6.1** | Pi 5에서 동작 안 함 — `/usr/bin/avrdude`(7.1)에 가려져 있을 뿐. 복사 금지 |
| `/usr/local/bin/gpio` | WiringPi 3.18 수동 설치 | 디버깅용이었음. 현행 시스템 미사용 (필요하면 `gpiod`의 `gpioset/gpioget`으로 충분) |
| `~/ATtiny85/blinky` | 홈 디렉토리 | 초기 테스트용 blink 프로젝트 |

(참고: 기존 Pi에서도 정리하려면 `sudo systemctl disable --now attiny_ack_idle attiny_shutdown_monitor` 후 위 파일들 삭제.)

---

## 8. 새 Pi 전체 절차 요약 (체크리스트)

```bash
# 1. OS_LED 디렉토리 복사 (§2)
# 2. 패키지
sudo apt install -y python3-spidev python3-gpiozero python3-lgpio gpiod
sudo apt install -y gcc-avr avr-libc binutils-avr avrdude   # 펌웨어 작업 시에만

# 3. SPI 확인
grep spi /boot/firmware/config.txt    # dtparam=spi=on

# 4. 데몬 설치
sudo bash ~/OS_LED/pi/install.sh

# 5. (새 ATtiny일 때만) fuse + 플래시  — §6, 런타임 케이블 분리 필수
cd ~/OS_LED/firmware && make fuses && make && sudo bash flash.sh

# 6. 배선 복구 후 동작 검증
systemctl status os-led-display os-led-poweroff --no-pager
sudo systemctl stop os-led-display && python3 ~/OS_LED/pi/blink_gpio10.py   # DIN 소유권 확인 (2Hz 토글)
sudo systemctl start os-led-display
```

### 동작 검증 시나리오
1. **부팅 핸드오프**: 전원 인가 → 터치 0.5s → 흰색 fade-up + 호흡 → 부팅 완료되면 무지개로 전환 (READY HIGH 핸드오프 성공)
2. **종료 핸드오프**: 무지개 중 터치 2s → ATtiny 호흡으로 전환 → Pi 꺼짐 → fade-out (GPIO27 → poweroff 성공)
3. **재부팅**: `sudo reboot` 시 LED가 ATtiny 호흡으로 갔다가 부팅 후 무지개 복귀

### 하드웨어 측 잊지 말 것 (회로 새로 꾸밀 때 — 상세는 ARCHITECTURE.md §7)
- ATtiny·WS2812는 **상시 5V** (Pi가 꺼져도 유지되는 레일)
- READY 라인(GPIO17↔PB3)에 **외부 10k 풀다운 필수**
- SHUTDOWN_REQ(PB4→GPIO27)는 10k+20k 분압 (5V→3.3V)
- ATtiny VCC에 100nF, LED 5V에 1000µF, DIN에 직렬 330Ω 권장
- 5V 공급기는 피크 ~1.4A 감당 (부팅/종료 순백 255 구간)
