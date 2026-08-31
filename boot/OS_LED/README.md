# OS_LED — 전원/부팅 LED 서브시스템 (ATtiny85 + Pi 데몬)

**ATtiny85 = 항상 켜져 있는 전원 컨트롤러 + LED 애니메이터.** 터치 센서(TTP223)로 전원
ON/OFF 의도를 받아 Pi의 PWR_BTN(J2 헤더)을 눌러 부팅시키거나 graceful shutdown을 요청하고,
Pi가 살아 있는 동안엔 WS2812 LED의 데이터 라인을 Pi에게 넘겨준다. 그 외 모든 구간
(대기/부팅 중/종료 중)에는 ATtiny가 직접 LED를 그린다.

> **폴더 상태**: `pi/`의 모든 파일은 설치본(`/usr/local/bin/*`, `/etc/systemd/**`)과 동일한
> **소스 원본**이다 (2026-08-31 diff 검증). 수정 시 `install.sh`로 재설치해야 반영된다.
> 펌웨어 소스(`firmware/`)와 구설계 보존본(`legacy/`)은 아직 이 저장소에 없다 — 추후 추가 예정.
> 그때까지 펌웨어 지식(§3, §7)은 이 문서가 유일한 기록이다.

## 1. 설계 의도 — 왜 ATtiny인가

Pi는 자기 자신이 꺼져 있을 때 아무것도 할 수 없다. 이 제품(음성 대화 로봇 RAY)은 물리 버튼
없이 **터치 한 번으로 켜고 끄는 가전 같은 UX**가 목표라서, Pi off 상태에서도 다음을 수행할
상시 전원 MCU가 필요했다:

1. **터치 감지** — Pi off 상태에서 0.5초 터치로 부팅 트리거
2. **부팅 트리거** — Pi 5의 J2(PWR_BTN)를 전기적으로 눌러줌 (GPIO로는 불가능)
3. **전 구간 LED 피드백** — 부팅 중·종료 중에도 애니메이션이 이어져 "죽은 시간"이 없어 보임
4. **graceful shutdown** — 터치 2초 → Pi에게 신호 → 완전히 꺼진 것을 확인 후 LED off

핀 5개(터치 IN, LED OUT, J2 OUT, READY IN, SHUTDOWN_REQ OUT)로 충분해 ATtiny85(DIP-8)를
채택했다. 핀이 정확히 꽉 차서(PB0~PB4, PB5=RESET 유지) "부팅 시작 신호가 따로 없다"는
트레이드오프가 생겼다(§3.4).

| 구성 요소 | 역할 |
|---|---|
| Raspberry Pi 5 | 메인 컴퓨터. 켜져 있는 동안 LED를 그리고, GPIO 2선으로 ATtiny와 핸드셰이크 |
| ATtiny85 (내부 PLL 16 MHz) | 전원/상태 컨트롤러. **상시 5V로 항상 켜져 있음** |
| WS2812B LED 24개 (직렬 체인) | 상태 표시등. ATtiny·Pi 데몬은 순백, RAY는 대화 연출 |
| TTP223 터치 센서 | 사용자 입력. **반드시 MOMENTARY 모드(TOG=GND), active-HIGH** |
| NPN 트랜지스터 | PB2 → 베이스. 오픈컬렉터로 J2를 GND로 당겨 부팅 트리거 |
| 상시 5V 전원 | **Pi가 poweroff 돼도 유지되는** 5V — 가장 흔한 실수 지점(§2.4) |

## 2. 핀맵과 배선

### 2.1 ATtiny85 핀맵

```
           ┌─────────────┐
   RESET 1 │● PB5    VCC │ 8  ──── +5V (상시)
     PB3 2 │             │ 7  PB2 ──▶ NPN base → Pi J2 (PWR_BTN 펄스)
     PB4 3 │             │ 6  PB1 ──▶ WS2812 DIN (Pi GPIO10과 물리적 공유)
     GND 4 │             │ 5  PB0 ◀── TTP223 OUT
           └─────────────┘
```

### 2.2 Pi ↔ ATtiny 4선 인터페이스

| 신호 | 방향 | Pi (BCM) | 헤더 핀 | ATtiny | 의미 |
|---|---|---|---|---|---|
| READY | Pi→ATtiny | GPIO17 | 11 | PB3 | Pi 유저스페이스 살아있음 = HIGH |
| SHUTDOWN_REQ | ATtiny→Pi | GPIO27 | 13 | PB4 | "graceful하게 꺼져라" = HIGH |
| WS2812 DIN | 공유 | GPIO10 (MOSI) | 19 | PB1 | LED 데이터. 소유권 프로토콜로 충돌 회피(§4) |
| PWR_BTN | ATtiny→Pi | — | J2 헤더 | PB2→NPN | 부팅 트리거. active-low 100 ms 펄스 |

### 2.3 필수 수동 소자 (없으면 오동작)

- **READY: 외부 10 kΩ 풀다운 필수.** Pi GPIO는 유저스페이스 전까지 floating — 풀다운이
  없으면 부팅 중 채터링으로 ATtiny가 "Pi 살아있음"을 오판한다.
- **SHUTDOWN_REQ: 10 kΩ(상단) + 20 kΩ(하단) 분압.** ATtiny 출력 5 V를 Pi(3.3 V)에 직결
  금지. 하단 20 kΩ이 idle 시 GPIO27 풀다운 겸용.
- **WS2812 DIN: 직렬 330~470 Ω.** 보호 + 소유권 전환 순간 양쪽 동시 드라이브 시 전류 제한.
- **NPN 베이스에 직렬 1 kΩ.**
- **디커플링: ATtiny VCC-GND 100 nF(칩 옆), WS2812 5V 레일 1000 µF.** 없으면 LED
  돌입전류로 brown-out/리셋 발생 이력 있음.

### 2.4 전원 ⚠️

- ATtiny와 WS2812는 **Pi가 꺼져도 살아 있는 5V 레일**(USB-C 입력의 업스트림 등)에서 받아야
  한다. Pi가 게이팅하는 레일이면 "터치로 켜기"와 "종료 애니메이션"이 모두 죽는다.
- 전류 예산: 순백(255) 24구 풀화이트 = **피크 ~1.4 A** (부팅/종료 중 ATtiny, 부팅 홀드 중
  Pi 데몬). 5V 공급기가 견뎌야 한다.
- **공통 GND 필수** (Pi ↔ ATtiny ↔ LED ↔ 터치센서).

## 3. 펌웨어 (ATtiny85) — 상태 머신

소스: `firmware/main/main.c` + `common/pins.h` + `common/ws2812.h` (**저장소 추가 예정**,
빌드 산출물 `firmware.hex`가 현재 칩에 구워진 최신본).

### 3.1 시각 상태

```
OFF      — 소등
BOOTING  — 이징 fade-up(0→255, ~1.3 s) + 오버슈트 dip → READY 기다리며 깊은 호흡
RUNNING  — Pi가 LED 구동 (ATtiny는 라인을 high-Z로 놓고 관전)
SHUTTING — ATtiny가 호흡 구동
FADE_OFF — anticipation dip 후 fade-down(255→0) → OFF
```

모든 ATtiny 애니메이션은 **순백**(G=R=B), 호흡은 sine 대신 포물선 `idx*(N-idx)` 근사.
레벨 `PULSE_MIN=16 ~ PULSE_MAX=255`, 한 주기 ≈ 2.0 s. **Pi 데몬의 부팅 홀드(§5.1)가 이
파라미터를 그대로 복제**하므로 펌웨어를 바꾸면 데몬 상수도 함께 바꿔야 한다.

### 3.2 터치 규칙

| 상태 | 입력 | 동작 | 상수 |
|---|---|---|---|
| IDLE | 0.5 s 지속 터치 | 부팅 (J2 펄스) | `BOOT_TOUCH_HOLD_MS=500` |
| RUNNING | **2.0 s 연속** 터치 | 종료 요청 (PB4 HIGH) | `SHUTDOWN_TOUCH_HOLD_MS=2000` |
| BOOTING | 45 s 이후 새 터치 | J2 재펄스 | `BOOT_RETRY_GRACE_MS=45000` |
| SHUTTING | 새 터치 | PB4 재assert | — |
| 모든 상태 | 5 s 홀드 | 비상 소프트 리셋 (터치 재캘리브레이션) | `EMERGENCY_HOLD_MS=5000` |

종료가 "2초 연속"인 것은 실수 종료 방지 — 짧게 끊어 누른 누적은 무효.

### 3.3 자동 재시도와 안전망

- **부팅 자동 재시도**: 90 s 후부터 30 s마다 J2 재펄스. 단 **45 s 이전엔 절대 금지** —
  부팅 중 PWR_BTN은 오히려 종료시킬 수 있다.
- **종료 자동 재시도**: PB4를 10 s마다 재assert — Pi가 rising edge를 놓쳐도 복구.
- **타임아웃**: 부팅 5분, 종료 30 s.
- **하드웨어 워치독 8 s** (`WDTO_8S`): 모든 루프가 `wdt_reset()` 호출.
- **워치독 리셋 후 Pi 보호**: 리셋 직후 `pins_init` 이전에 READY 샘플(`pi_was_up`) —
  Pi가 LED 구동 중이면 라인을 뺏지 않고 RUNNING으로 합류.
- **stuck-touch drain**: IDLE 진입 시 터치 HIGH면 풀릴 때까지 대기, 10 s 초과 시 재캘리브레이션.
- **READY 디바운스 5샘플(≈80 ms)**: 노이즈 스파이크로 "Pi 죽음" 오판 방지.

### 3.4 부팅 트리거 3종과 J2 정책

READY는 Pi 데몬이 `multi-user.target` 이후에 올리므로 **"부팅 완료" 신호이지 "부팅 시작"
신호가 아니다** (부팅 중 ~20 s는 풀다운 때문에 LOW). ATtiny는 부팅 시작 감지 핀이 없어
자기 상태 + 터치 유무로 추론한다:

| 트리거 | ATtiny 관찰 | J2 펄스 | 근거 |
|---|---|---|---|
| 터치로 켜기 | IDLE 중 0.5 s 터치, READY LOW | **한다** | Pi가 꺼져 있으므로 깨워야 함 |
| 전원 인가 자동부팅 | 콜드부팅, READY LOW | 안 한다 | Pi도 스스로 부팅 중이라고 가정 |
| `sudo reboot` / SW poweroff | 무터치 READY HIGH→LOW | 안 한다 | 스스로 다시 올라올 것으로 가정 |

- 추론이 틀려도 무해: grace 90 s 안에 READY가 안 오면 fade-down 후 IDLE 터치 대기로 폴백.
- **알려진 한계**: reboot과 터미널 poweroff를 구분 못 함 (둘 다 "무터치 READY LOW") —
  터미널 poweroff 후에도 90 s간 부팅 호흡을 하다 꺼진다. 4선 인터페이스의 의도된 트레이드오프.

### 3.5 WS2812 비트뱅 타이밍

`ws2812.h`는 16 MHz 기준 사이클 카운팅 인라인 어셈블리 (비트당 20사이클 = 1.25 µs).
**fuse가 16 MHz PLL이 아니면 색이 전부 깨진다.** 전송 중 인터럽트 disable, 24구 한 프레임
≈ 720 µs, 프레임 간 리셋 LOW 60 µs, 색 순서 **GRB**.

### 3.6 Fuse (신품 칩 1회 필수)

신품은 8 MHz + CKDIV8(=1 MHz)라 그대로면 타이밍이 깨진다:

```
lfuse = 0xE1   (16 MHz 내부 PLL, CKDIV8 off)
hfuse = 0xDF   (RSTDISBL 유지 → ISP 계속 가능)
efuse = 0xFF   (BOD disabled)
```

## 4. WS2812 DIN 소유권 프로토콜

ATtiny PB1과 Pi GPIO10(MOSI)이 **물리적으로 같은 선**이다:

```
대기/부팅 중/종료 중 → ATtiny가 PB1을 OUTPUT으로 잡고 직접 그림
Pi 실행 중           → ATtiny가 PB1을 INPUT(high-Z)으로 풀고, Pi가 SPI로 그림
```

전환은 READY 하나로 핸드셰이크한다 (ACK 없음 — Pi는 ATtiny가 실제로 놓았는지 관측 불가):

- **부팅 핸드오프**: 데몬이 READY HIGH → ATtiny 디바운스(≈80 ms) 후 PB1 INPUT 전환.
  Pi는 READY HIGH 후 **0.3 s 대기**(`HANDOFF_WAIT_S`) 후 SPI 시작 → 충돌 윈도우 없음.
  ⚠️ 이 0.3 s는 ATtiny 디바운스 80 ms와 커플링 — 한쪽만 줄이지 말 것.
- **종료 핸드오프**: 데몬 cleanup이 검은 프레임 + READY LOW → ATtiny가 PB1 회수, 어두운
  흰색부터 호흡 재개 (밝기 점프 없음).
- **시각적 연속성**: ATtiny의 마지막 프레임은 풀화이트(255)로 얼어 있고, 데몬은 동일한
  풀화이트에서 시작해 **같은 파라미터의 흰 호흡을 이어 그린다**(§5.1) — 핸드오프 순간이
  보이지 않는다.

검증 도구: `pi/blink_gpio10.py` (Pi가 GPIO10을 잡을 수 있는지 2 Hz 토글).

## 5. Pi-side 데몬

설치·실행 유닛 (`install.sh`가 일괄 설치):

| 유닛/파일 | 역할 |
|---|---|
| `os-led-display.service` → `os_led_display.py` | READY assert + 부팅 흰 호흡 + 소유권 arbiter (§5.1) |
| `os-led-poweroff.service` → `os_led_poweroff.py` | GPIO27 감시 → graceful poweroff (§5.2) |
| `logind.conf.d/10-os-led.conf` | J2 펄스의 power-key 이벤트 이중 트리거 방지 (짧은 누름 무시, 5 s 롱프레스만 비상용) |
| `system.conf.d/10-os-led.conf` | `DefaultTimeoutStopSec=10s` — hung 서비스가 ATtiny 종료 타임아웃(30 s)을 넘기지 않게 캡 |

**혼동 금지**: `led-pwm.service`는 OS_LED가 아니라 로봇 몸체 LED(C++ Ray, GPIO13 하드웨어
PWM)용 별개 시스템이다 (`scripts/hardware/setup_led_hwpwm.sh` 참고).

### 5.1 `os_led_display.py` — 디스플레이 + 소유권 arbiter

시작 시 READY HIGH → 0.3 s 대기 → SPI 시작. SIGTERM 시 검은 프레임 → READY LOW → 소켓
정리 (이 순서 덕에 ATtiny가 자연스럽게 라인을 되찾는다). `Restart=always`.

**렌더링 (현행 동작)**
- **부팅 홀드**: ATtiny의 흰 호흡을 동일 파라미터(포물선, 2.0 s 주기, 16~255, 정점에서
  시작)로 이어 그리며 RAY를 기다린다 — 정상 부팅에서는 `흰 호흡 → RAY LED`로 무지개 없이
  직행한다. 홀드는 호흡 정수배(`BOOT_WHITE_HOLD_S=150`)라 항상 정점에서 끝난다.
- **무지개 = 폴백/이상 신호**: RAY가 150 s 안에 안 오면 흰색→무지개 블룸(1.5 s). RAY가
  잡았다 놓은 뒤에도 무지개 복귀 — "Pi는 살아있는데 RAY가 없음"의 표시다. 무지개는 앞
  8구만, 밝기 0.25 캡(~360 mA), 60 fps.
- **SPI 인코딩**: 6.5 MHz, WS2812 1비트 = SPI 1바이트(`0→0b1100_0000`, `1→0b1111_1100`),
  프레임 앞 0x00×42 리셋. RAY가 쓰는 `rpi5_ws2812`와 바이트 단위 동일 — 과거 3.2 MHz/4-bit
  인코딩은 이 스트립에서 마진 부족으로 깜빡여 교체됐다.

**소유권 arbiter** — Pi 안에서 SPI를 쓰려는 주체가 둘(이 데몬, RAY 파이프라인)이라, 데몬이
`/dev/spidev0.0`의 유일한 상시 소유자이고 RAY는 유닉스 소켓 `/run/os-led.sock`(0666)으로
빌려간다:

```
RAY → "ACQUIRE\n" : 현재 화면(흰 호흡/무지개)을 0.3 s fade-out → SPI 쓰기 완전 중단
                    → "GRANTED\n". 이후 RAY가 직접 그림.
RAY → "RELEASE\n" : (또는 소켓 close = RAY 크래시 포함) 0.25 s settle 후 무지개 fade-in.
```

- **연결 유지 = 토큰 보유.** RAY가 죽으면 소켓이 끊겨 자동 복귀 — 스트립이 방치되지 않는다.
- 내부 동기화 이벤트 3개: `pause_req` / `paused` / `stop_evt`. **레이스 주의**: `paused`는
  "루프가 SPI를 안 쓰는 중"을 뜻하므로 fade-in **시작 전에** clear한다 — 순서를 바꾸면
  GRANTED가 나갔는데 데몬이 아직 SPI를 쓰는 겹침이 생긴다. 수정하지 말 것.
- RAY 쪽 클라이언트: `voice_pipeline/led/arbiter_client.py`. 부팅 순서 경합 대비 소켓 연결
  재시도(이 유닛 파일이 설치된 기기는 30 s, 미설치 개발기는 5 s) 후 standalone 폴백.
  RAY는 파이프라인 준비 완료 시점(첫 `set_state`)에 ACQUIRE한다.

### 5.2 `os_led_poweroff.py` — 종료 요청 수신

- GPIO27을 **20 ms 폴링**. gpiozero edge 콜백은 Pi 5 lgpio에서 이벤트를 놓치는 것이 관측돼
  **의도적으로 폴링** — 콜백으로 되돌리지 말 것.
- rising edge → `os.sync()` → `systemctl poweroff -f` (single force: graceful하되 GUI
  inhibitor만 무시). `DefaultTimeoutStopSec=10s`와 페어로 ATtiny 30 s 타임아웃 안에 종료.
- 트리거 후 프로세스를 종료하지 않는다 — poweroff 실패 시 다음 edge에서 재시도 (ATtiny도
  10 s마다 재assert → 양쪽 재시도가 만난다). 라인이 LOW로 돌아올 때까지 재무장하지 않고,
  시작 시점에 이미 HIGH면 즉시 트리거한다.

## 6. 새 Pi 이식

전제: Raspberry Pi 5 (Pi 4 이하는 gpiochip 번호 다름 — §7.2), Ubuntu 24.04, 시스템 python3
(venv 아님), avrdude 7.x (6.x는 Pi 5에서 실패).

```bash
# 1. 이 폴더 복사 후 패키지
sudo apt install -y python3-spidev python3-gpiozero python3-lgpio gpiod
# 2. SPI 활성화: /boot/firmware/config.txt 에 dtparam=spi=on → /dev/spidev0.0 존재 확인
# 3. 데몬 설치 (스크립트+유닛+드롭인+enable 일괄)
sudo bash OS_LED/pi/install.sh
# 4. (새 칩일 때만) 펌웨어 플래시 — §7
# 5. 배선 — §2 (풀다운/분압/캡/상시5V 체크리스트)
```

**`legacy/`(구설계 GPIO23/24)는 어떤 Pi에도 설치 금지** — 보존용 사본일 뿐이다.

동작 검증 순서:
1. 전원 인가 → fade-up + 호흡 → 부팅 완료 시 Pi가 흰 호흡 인계 → RAY 준비 시 RAY LED 직행
   (RAY 미설치/미기동이면 150 s 후 무지개)
2. Pi off에서 터치 0.5 s → J2 펄스 부팅
3. Pi on에서 터치 2 s 연속 → 종료 호흡 → 소등
4. `sudo reboot` → 종료 호흡 → 부팅 호흡 (터치·J2 없이)
5. arbiter: 소켓에 ACQUIRE를 보내 데몬이 화면을 끄는지, 소켓을 끊으면 무지개로 복귀하는지:
   ```python
   import socket, time
   s = socket.socket(socket.AF_UNIX); s.connect("/run/os-led.sock")
   s.sendall(b"ACQUIRE\n"); print(s.recv(32)); time.sleep(3); s.close()
   ```
6. DIN 소유권: `sudo systemctl stop os-led-display && python3 OS_LED/pi/blink_gpio10.py`
   → 2 Hz 토글 → `sudo systemctl start os-led-display`

## 7. 펌웨어 빌드/플래시 (새 칩 또는 펌웨어 수정 시에만)

기존 칩을 회로째 옮기면 불필요. `firmware.hex`를 그대로 플래시해도 된다.

### 7.1 ISP 배선 (Pi 5 헤더 ↔ ATtiny85)

| Pi 헤더 | Pi 신호 | ATtiny 핀 | 신호 |
|---|---|---|---|
| 1 | **3V3** | 8 | VCC (**ISP 중엔 5V 아님!**) |
| 6 | GND | 4 | GND |
| 15 | GPIO22 | 1 | RESET |
| 19 | GPIO10 (MOSI) | 5 | PB0 |
| 21 | GPIO9 (MISO) | 6 | PB1 |
| 23 | GPIO11 (SCK) | 7 | PB2 |

배선 순서: GND → 신호 4선 → 3V3 마지막. **런타임 케이블 전부 분리 필수** — 특히 NPN base:
SCK(PB2) 토글이 J2를 펄스시켜 플래시 도중 Pi가 꺼진 사고 이력 있음.

### 7.2 플래시 명령 (검증된 방식)

avrdude 7.1의 **`linuxspi`만 사용** (`linuxgpio` 없음). Pi 5는 `/dev/gpiochip4`(RP1),
**Pi 4 이하는 `/dev/gpiochip0`**.

```bash
sudo systemctl stop os-led-display   # /dev/spidev0.0 점유 해제 — 반드시 먼저

# 1) 시그니처 확인 — 0x1e930b가 나와야 진행
sudo avrdude -c linuxspi -P /dev/spidev0.0:/dev/gpiochip4:22 -B 250 -i 100 -x disable_no_cs -p t85
# 2) fuse (신품 1회)
sudo avrdude -c linuxspi -P /dev/spidev0.0:/dev/gpiochip4:22 -B 250 -i 100 -x disable_no_cs -p t85 \
    -U lfuse:w:0xE1:m -U hfuse:w:0xDF:m -U efuse:w:0xFF:m
# 3) erase → 4) write+verify (분리는 Pi5 RP1 SPI 첫바이트 유실 대책)
sudo avrdude -c linuxspi -P /dev/spidev0.0:/dev/gpiochip4:22 -B 250 -i 100 -x disable_no_cs -p t85 -e
sudo avrdude -c linuxspi -P /dev/spidev0.0:/dev/gpiochip4:22 -B 250 -i 100 -x disable_no_cs -p t85 \
    -D -U flash:w:OS_LED/firmware/firmware.hex
```

신품 칩(1 MHz) + 브레드보드에선 `-B 100`이 자주 실패 → `-B 250`(≈4 kHz)이 안전.
**쓰기/읽기 도중 Ctrl-C 절대 금지** — RP1 SPI 드라이버가 잠겨 재부팅으로만 풀린다.

| 증상 | 원인 → 조치 |
|---|---|
| 시그니처 깨짐 (`0x000102` 등) | MISO 접촉 불량 / 칩 방향 / 전원 → 노치·3.3V 실측·도통 체크 |
| `AVR device not responding` | RESET·SCK·전원 단선, 또는 직전 Ctrl-C로 SPI 락 → `sudo reboot` |
| 시도마다 증상이 바뀜 | 브레드보드 점퍼 접촉 불량(가장 흔함) |

플래시 후: ISP 6선 **모두** 제거(3V3 남긴 채 5V 연결 시 VCC 충돌) → 런타임 배선 복구 →
`sudo systemctl start os-led-display`.

## 8. 알려진 약점 / 관측 기록

1. **순백 피크 ~1.4 A** — 약한 5V 어댑터에선 brown-out 가능. 펌웨어 최대 레벨 캡 고려.
2. **BOD off (efuse=0xFF)** — VCC가 출렁이면 미정의 동작 가능. 디커플링 필수, 필요시
   BOD 2.7V fuse 고려. (2026-08-20 일회성 흰색/RAY LED 교대 깜박임의 유력 원인 후보 —
   재현 시 BOD fuse + 디커플링 실장 확인이 1순위)
3. **poweroff/reboot 구분 불가** — §3.4 한계.
4. **핸드오프 타이밍 커플링** — Pi `HANDOFF_WAIT_S=0.3` ↔ ATtiny 디바운스 80 ms ↔ 데몬
   호흡 파라미터(§5.1) ↔ arbiter `paused` clear 순서. 어느 한쪽만 고치면 충돌.
5. **warm reboot 핸드오프 순간 짧은 깜박임** — GUI/`sudo reboot` 재시작에서만 관측, 전원버튼
   콜드부팅은 무증상. §2.3의 소유권 전환 동시 드라이브 창과 정합. 기본 운용이 전원버튼이라
   추적하지 않기로 결정 (2026-08-27).
6. **sleep/PCINT0 데드코드** — 펌웨어에 정의만 있고 미사용. IDLE에서 5 ms busy-poll(수 mA).

## 9. 파일 맵

```
OS_LED/
├── README.md                # 이 문서
├── pi/                      # ★ 설치본과 동일한 소스 원본 (수정 → install.sh 재실행)
│   ├── install.sh           # 설치 원클릭 (스크립트+유닛+드롭인+enable)
│   ├── os_led_display.py    # READY + 부팅 흰 호흡 + arbiter (§5.1)
│   ├── os_led_poweroff.py   # GPIO27 폴링 → poweroff (§5.2)
│   ├── os-led-display.service / os-led-poweroff.service
│   ├── logind-override.conf / system-fast-shutdown.conf
│   └── blink_gpio10.py      # DIN 소유권 검증 툴
├── firmware/                # (추가 예정) main.c, pins.h, ws2812.h, Makefile, firmware.hex
└── legacy/                  # (추가 예정, 보존용) 구설계 GPIO23/24 — 어떤 Pi에도 설치 금지
```

RAY 저장소 연동 지점: `voice_pipeline/adapters/led.py`(RAY 쪽 소켓 클라이언트 + LED 연출 — rpi5_ws2812, 데몬과 동일 인코딩).

