# OS_LED — 시스템 전체 정리본 (핸드오프 문서)

> Raspberry Pi 5 + ATtiny85 + WS2812 24-LED 링 기반의 "터치 전원 + 상태 표시등" 시스템.
> 이 문서는 (1) 펌웨어/데몬 로직을 처음 보는 사람(또는 다른 Claude Code)이 이해하도록,
> (2) 회로를 실제로 구성/디버깅하는 데 필요한 모든 전기적 정보를, (3) 알려진 약점/보완사항을 담는다.

---

## 0. 한 줄 요약

ATtiny85가 **항상 켜져 있는 전원 컨트롤러 + LED 애니메이터**다. 터치 센서(TTP223)로 사용자의
전원 ON/OFF 의도를 감지하고, Pi의 PWR_BTN을 눌러 부팅시키거나 GPIO로 graceful shutdown을 요청한다.
Pi가 살아 있는 동안에는 WS2812 데이터 라인 소유권을 Pi에게 넘겨주고, 그 외 모든 구간(부팅 중/종료 중/대기)에는
ATtiny가 LED를 직접 그린다.

---

## 1. 하드웨어 구성요소

| 요소 | 역할 |
|---|---|
| Raspberry Pi 5 | 메인 컴퓨터. 평소엔 LED 무지개를 그리고, GPIO로 ATtiny와 핸드셰이크 |
| ATtiny85 (DIP-8, 16 MHz 내부 PLL) | 전원/상태 컨트롤러. 항상 켜져 있음 |
| WS2812B 링 24개 | 상태 표시등 (흰색 호흡/무지개) |
| TTP223 터치 센서 | 사용자 입력. **반드시 MOMENTARY 모드(TOG=GND)**, active-HIGH |
| NPN 트랜지스터 | Pi의 PWR_BTN(J2 헤더)을 오픈컬렉터로 당겨 부팅 트리거 |
| 5V 전원 (상시) | **Pi가 꺼져도 유지되는** 5V 입력. 아래 §6 전원 주의 참고 |

---

## 2. ATtiny85 핀맵 (firmware/common/pins.h 기준)

```
           ┌─────────────┐
   RESET 1 │● PB5    VCC│ 8  ──── +5V (상시)
     PB3 2 │           │ 7  PB2 ──▶ NPN base  (J2 PWR_BTN 펄스)
     PB4 3 │           │ 6  PB1 ──▶ WS2812 DIN (Pi GPIO10과 공유)
     GND 4 │       PB0│ 5  ◀──── TTP223 OUT  (PCINT0)
           └─────────────┘
```

| ATtiny 핀 | 신호 | 방향 | 연결 | 비고 |
|---|---|---|---|---|
| PB0 (5) | TTP223 OUT | IN | 터치센서 출력 | active-HIGH, momentary |
| PB1 (6) | WS2812 DIN | OUT/IN(공유) | LED 링 DIN **+ Pi GPIO10(MOSI)** | 소유권 전환됨 (§4) |
| PB2 (7) | NPN base | OUT | NPN 베이스 → Pi J2(PWR_BTN) | active-HIGH로 NPN ON → J2 LOW |
| PB3 (2) | READY | IN | ← Pi GPIO17 | Pi가 살아있으면 HIGH. **외부 10k 풀다운 필수** |
| PB4 (3) | SHUTDOWN_REQ | OUT | → Pi GPIO27 (10k+20k 분압) | 종료 요청 시 HIGH |
| VCC (8) | +5V 상시 | — | 5V 레일 | |
| GND (4) | GND | — | 공통 GND | Pi와 반드시 공통 접지 |

### Fuse (필수 — 신품 칩은 8MHz/CKDIV8라 안 맞춰주면 타이밍 전부 깨짐)
```
lfuse = 0xE1   (16 MHz 내부 PLL, CKDIV8 off, SUT=10)
hfuse = 0xDF   (RSTDISBL=1 유지 → ISP 계속 가능)
efuse = 0xFF   (BOD disabled — §7 보완사항 참고)
```
한 번만: `cd firmware && make fuses`

---

## 3. Pi ↔ ATtiny 신호 인터페이스 (4선)

| 신호 | 방향 | Pi 핀(BCM) | Pi 헤더 | ATtiny | 의미 |
|---|---|---|---|---|---|
| READY | Pi→ATtiny | GPIO17 | 11 | PB3 | Pi 유저스페이스 살아있음 = HIGH |
| SHUTDOWN_REQ | ATtiny→Pi | GPIO27 | 13 | PB4 | 종료해달라 = HIGH (분압 5V→3.3V) |
| WS2812 DIN | 공유 | GPIO10(MOSI) | 19 | PB1 | LED 데이터. 소유권 전환 |
| PWR_BTN(J2) | ATtiny→Pi | — | J2 헤더 | PB2→NPN | 부팅 트리거 (active-low 펄스) |

- **READY (GPIO17→PB3)**: Pi의 `os_led_display.py`가 시작하자마자 HIGH로 올림. 종료(SIGTERM) 시 LOW.
  Pi GPIO는 유저스페이스가 잡기 전까지 floating이라 **외부 10k 풀다운**으로 LOW 고정해야 함(안 그러면 임계전압에서 채터링).
- **SHUTDOWN_REQ (PB4→GPIO27)**: ATtiny 5V 출력 → 10k+20k 분압으로 3.33V. 20k가 idle 시 GPIO27을 LOW로 당김.
- **WS2812 DIN (PB1↔GPIO10)**: 둘 다 드라이브 가능 → 소유권 프로토콜로 충돌 회피(§4).
- **PWR_BTN(J2)**: NPN 오픈컬렉터로 PWR_BTN을 GND로 당김. 100ms 펄스(`J2_PULSE_MS`)는
  PMIC 디바운스(~50ms)는 넘기고 long-press(5s=강제오프)에는 안 걸리는 안전 구간.

---

## 4. WS2812 DIN 라인 소유권 프로토콜 (핵심)

PB1과 Pi GPIO10이 **물리적으로 같은 선**에 묶여 있어 동시에 드라이브하면 충돌난다. 해소 규칙:

```
대기/부팅중/종료중  → ATtiny가 PB1을 OUTPUT으로 잡고 LED를 직접 그림
Pi 실행중           → ATtiny가 PB1을 INPUT(high-Z)로 풀고, Pi가 SPI로 그림
```

전환은 READY 신호로 핸드셰이크:
- **부팅 핸드오프**: ATtiny가 호흡 애니메이션 돌리며 READY를 폴링. Pi가 부팅 완료 →
  `os_led_display.py`가 READY HIGH로 올림 → ATtiny가 (디바운스 5샘플≈80ms 후) 감지 →
  `led_release_ownership()` (PB1을 INPUT으로). Pi는 READY HIGH 후 **0.3s 기다렸다가** SPI 시작
  (`HANDOFF_WAIT_S`) → 충돌 윈도우 없음(80ms ≪ 300ms 마진).
- **종료 핸드오프**: Pi의 `cleanup()`이 검은 프레임 쓰고 READY LOW → ATtiny가 감지 →
  `led_take_ownership()` → 어두운 흰색에서 시작하는 호흡(점프 없이 부드럽게).

소유권 검증 도구: `pi/blink_gpio10.py` (Pi가 GPIO10 잡았는지 2Hz 토글로 확인).

---

## 5. 펌웨어 상태 머신 (firmware/main/main.c)

### 시각 상태
```
OFF       — 꺼짐 (strip dark)
BOOTING   — 0→255 이징 fade-up(+오버슈트) 후, READY 기다리며 깊은 호흡
RUNNING   — Pi가 무지개 구동 (ATtiny는 LED 라인 high-Z)
SHUTTING  — ATtiny가 호흡 구동
FADE_OFF  — 255→0 이징 fade-down(앞에 anticipation dip) 후 OFF
```

### 터치 규칙
| 상태 | 입력 | 동작 |
|---|---|---|
| IDLE | 0.5s 지속 터치 | BOOT 시작 (`BOOT_TOUCH_HOLD_MS`) |
| RUNNING | 2.0s 지속 터치 | SHUTDOWN 요청 (`SHUTDOWN_TOUCH_HOLD_MS`) |
| BOOTING | 45s 후 새 터치 | J2 재펄스 (Pi가 첫 PWR_BTN 놓쳤을 때) |
| SHUTTING | 새 터치 | PB4 재펄스 (데몬 재트리거) |
| 모든 상태 | 5s 지속 홀드 | 비상 소프트 리셋(`force_reset`, 최후수단) |

### 자동 재시도 (사용자 개입 불필요)
- **부팅**: 90s 후부터 30s마다 ATtiny가 J2 자동 재펄스(`BOOT_AUTO_RETRY_*`).
  45s 전엔 절대 재펄스 안 함(부팅 중 재펄스 = 의도치 않은 종료 위험).
- **종료**: PB4를 10s마다 재assert(`SHUTDOWN_RETRY_INTERVAL_MS`) — 데몬이 rising edge 놓쳐도 복구.

### 안전망 (defense in depth)
- 명시적 타임아웃: 부팅 5분(`BOOT_TIMEOUT_MS`), 종료 30s(`SHUTDOWN_TIMEOUT_MS`).
- **워치독 8s** (`WDTO_8S`): 어느 코드 경로든 멈추면 칩 하드리셋. 모든 루프가 `wdt_reset()` 호출.
- 리셋 후 `pi_was_up`을 pins_init **전에** 샘플 → Pi가 LED 라인 구동 중이면 안 뺏음.
- IDLE 진입 시 stuck-HIGH 터치 먼저 drain (10s 넘게 stuck이면 force_reset → 재캘리브레이션).
- **pi_ready 디바운스**(5샘플): GPIO17 노이즈 스파이크로 "Pi 죽음" 오판 → 무지개 중 흰 플래시 방지.

### main() 흐름
```
wdt_enable(8s); pi_ready 샘플
 ├ pi_was_up(리셋시 Pi 살아있음)? → LED 안 잡고 running_after_recovery로 goto
 └ 정상 콜드부팅(12V 인가) → pins_init; LED 잡고 OFF; TTP223 캘 1.5s
   [COLD-BOOT] fade_up(0) → pulse_until_ready_or_timeout(90s, J2 없음)
        보드가 Pi를 자동부팅하면 READY HIGH → release 후 running으로 goto
        타임아웃(자동부팅 안 하는 보드)이면 fade_down 후 아래 IDLE로 폴백
   for(;;):
     [IDLE] 0.5s 터치 대기 (도중 Pi가 뜨면 auto-adopt)
     [BOOT] fade_up(1) → pulse_until_ready_or_timeout(5min, J2 허용)   ← 터치로 켤 때만 J2 펄스
            └ 타임아웃이면 fade_down 후 IDLE 복귀
     led_release_ownership()
   running_after_recovery:
     [RUNNING→SHUTDOWN] pi_self_booting=0; for(;;){ 2s터치 or Pi off 대기;
        터치 없이 Pi off → pi_self_booting=1; break (reboot/sw-poweroff);
        터치 → shutdown_req_assert; wait_pi_off; release; 성공시 break }
     [OFF] led_take_ownership; pulse_for_ms; fade_down
     [REBOOT?] pi_self_booting이면 fade_up(0) → pulse_until_ready_or_timeout(90s, J2 없음)
        Pi 복귀(READY HIGH) → release 후 running으로 goto
        grace 타임아웃(진짜 종료였음) → fade_down 후 IDLE 복귀
```

### 부팅 트리거 3종과 J2 정책 (reboot / 12V 자동부팅 처리)

READY(GPIO17)는 `os_led_display` 데몬이 `multi-user.target` 이후 뜰 때 올라오므로
**"부팅 완료" 신호이지 "부팅 시작" 신호가 아니다.** 부팅 중(20~40s)에는 외부 풀다운으로 LOW.
ATtiny85는 핀이 꽉 차 있어(PB0~4 사용, PB5는 RESET) 부팅 시작용 입력을 따로 둘 수 없다.
그래서 "Pi가 ATtiny 도움 없이 스스로 부팅 중"인 상황을 **자기 상태 + 터치 유무로 추론**하고,
J2 펄스 없이 부팅 애니메이션을 켜둔 채 READY가 오는지 기다려 **사후 검증**한다.

| 트리거 | ATtiny가 관찰 | J2 | 비고 |
|---|---|---|---|
| 터치로 켜기 | IDLE 중 0.5s 터치, READY LOW | **펄스** | Pi 꺼져 있음 → PWR_BTN 눌러 깨워야 |
| 12V 자동부팅 | 콜드부팅(main 시작), READY LOW | **안 함** | 전원 막 들어옴 → Pi도 자동부팅 중이라 가정 |
| reboot / sw-poweroff | RUNNING 중 터치 없이 READY HIGH→LOW | **안 함** | Pi가 스스로 내려감 → 다시 올라올 것으로 가정 |

- J2를 안 누르는 두 경우는 `pulse_until_ready_or_timeout(..., allow_j2=0)`로 호흡+READY 폴링만 함.
  Pi가 스스로 부팅 중인데 PWR_BTN을 누르면 부팅을 방해/종료시킬 수 있어서다.
- 추론이 틀려도 손해 없음: grace(`COLD_BOOT_TIMEOUT_MS`/`REBOOT_GRACE_MS`, 각 90s) 안에 READY가
  안 오면 fade_down 후 평소처럼 IDLE 터치 대기로 폴백한다.
- **한계**: ATtiny는 reboot과 (터미널)poweroff를 구분 못함 — 둘 다 "무터치 READY LOW"로 동일.
  그래서 poweroff도 90s 부팅 호흡을 했다가 READY가 안 오면 꺼진다. 4선 인터페이스의 트레이드오프.
  확실히 구분하려면 Pi가 부팅 초기에 READY를 올리고 핸드오프를 별도 엣지로 분리해야 하는데,
  핀 부족 + 핸드오프 재설계가 필요해 현재는 추론+grace 방식을 채택.

### LED 애니메이션 디테일
- `fade_up_with_j2_pulse`: 이징 포물선 0→255(~1.3s) + 오버슈트 dip(255→215) + settle.
  J2는 step5(~100ms)에서 release. "프리미엄 가전" 느낌 목적.
- `fade_down`: anticipation dip(255→230) + kick(→255) + 이징 255→0. fade-up과 비대칭.
- `breath_level`: 포물선(idx*(N-idx))로 sine 호흡 근사 (trig/LUT 없이). `[PULSE_MIN=16, PULSE_MAX=255]`.
- 모든 색은 **순백(G=R=B 동일 레벨)**. `fill_white(level)`.

---

## 6. WS2812 신호 타이밍

### ATtiny측 (bit-bang, ws2812.h) — 16MHz, 비트당 20사이클=1.25µs
```
'0': T0H 5cyc(312ns) / T0L 15cyc(937ns)
'1': T1H 13cyc(812ns) / T1L 7cyc(437ns)
RES: 프레임 사이 LOW >50µs (코드는 60µs)
전송 중 인터럽트 disable. 24LED ≈ 720µs.
```
사이클 카운팅된 인라인 어셈블리이므로 **F_CPU/fuse가 16MHz가 아니면 색이 깨진다.**

### Pi측 (SPI 인코딩, os_led_display.py) — SPI 3.2MHz, WS2812 1비트=SPI 4비트
```
0 → 0b1000  (T0H 312ns / T0L 938ns)
1 → 0b1110  (T1H 938ns / T1L 312ns)
WS2812 2비트 = SPI 1바이트(바이트정렬). LATCH 30바이트 0 (~75µs LOW).
전제: /boot/firmware/config.txt 에 dtparam=spi=on
```
색 순서는 둘 다 **GRB**.

---

## 7. 전원 — 회로 구성 시 가장 중요한 부분 ⚠️

### (A) 상시 5V 레일 필수
ATtiny와 WS2812는 **Pi가 poweroff 돼도 유지되는 5V**(USB-C 입력/업스트림 측)에서 전원을 받아야 한다.
- Pi가 꺼져도 ATtiny가 살아 있어야 → 터치로 다시 부팅 가능(J2 펄스).
- 종료 애니메이션이 Pi 꺼진 뒤에도 보이려면 LED도 상시 5V여야 함.
- **Pi가 끊는 레일(예: 특정 GPIO로 게이팅된 전원)에서 따오면 안 됨.**

### (B) 전류 예산
- **런타임 무지개**: Pi가 `BRIGHTNESS=0.25`로 제한 → 24LED 약 360mA.
- **부팅/종료 순백(255)**: 24 × 3채널 풀 = **최대 ~1.4A 피크**. 5V 공급기는 이 피크를 견뎌야 함.
  → 보완 후보: ATtiny 흰색 레벨을 128로 캡하면 피크 절반 (§8).

### (C) 디커플링/벌크 캐패시터 (현재 회로에 없으면 추가 권장)
- ATtiny VCC-GND 사이 **100nF** 세라믹 (칩 바로 옆).
- WS2812 5V 레일에 **1000µF 전해** (LED 돌입전류 흡수, 표준 권장).
- 캡 없으면 brown-out/EMI로 ATtiny 리셋·SPI 마진 불안정 (flash 메모리에도 기록됨).

### (D) 분압/풀 저항
- READY(GPIO17↔PB3): **외부 10k 풀다운** (필수).
- SHUTDOWN_REQ(PB4→GPIO27): **10k(상단)+20k(하단)** 분압 → 3.33V, 20k가 idle pull-down 겸함.
- WS2812 DIN: 직렬 **330~470Ω** 권장 (표준 보호 + PB1/GPIO10 충돌 시 전류 제한).

### (E) NPN / J2
- NPN 오픈컬렉터로 PWR_BTN을 GND로 당김. 베이스에 직렬저항(예: 1k) 권장.
- **펌웨어 플래시(ISP) 중에는 NPN base를 ATtiny에서 분리**해야 함 — SCK(PB2) 토글이 J2를 펄스시켜
  Pi가 플래시 도중 꺼지는 문제 있었음 (memory: attiny-flash-procedure).

---

## 8. 알려진 약점 / 보완사항

> 우선순위 순. "현재 동작은 하지만 개선 여지" 위주.

1. **부팅/종료 순백 전류 (~1.4A 피크)** — 中
   `fill_white(255)`를 그대로 쓰면 약한 5V 어댑터에서 brown-out 가능.
   → `show()`에 최대 레벨 캡(예: 160) 두거나, 부팅 fade의 peak를 낮추면 안전.

2. **sleep/PCINT0 데드코드** — 低 (정리용)
   `pins.h`의 `enter_sleep_idle()`, `pcint0_enable()` 정의돼 있으나 **어디서도 호출 안 함**.
   IDLE에서 5ms `_delay_ms` busy-poll만 함 → 항상 수 mA 소모. 상시전원이라 기능엔 문제없지만,
   (a) 데드코드 삭제하거나 (b) PCINT0 wake + SLEEP_MODE_IDLE로 IDLE 소비전류 절감 가능.

3. **BOD(브라운아웃) 비활성** — 中
   `efuse=0xFF` → BOD off. LED 돌입전류로 VCC가 출렁이면 ATtiny가 미정의 상태로 갈 수 있음.
   → 디커플링 캡(§7C) 추가 + 필요시 BOD를 2.7V 정도로 켜는 fuse 고려.

4. **디커플링/벌크 캡 미명시** — 中
   회로도/실물에 §7C 캡이 없으면 추가. memory에 "no decoupling cap"이 marginal wiring 원인으로 기록됨.

5. **WS2812 DIN 직렬저항 미명시** — 低
   PB1↔GPIO10 공유 라인이라 소유권 전환 순간의 짧은 충돌 가능성. 330Ω 직렬이면 충돌전류 제한 + 표준 보호.

6. **첫 비트 손실 (ISP, 펌웨어 자체 아님)** — 참고
   Pi5 RP1 SPI가 paged-write 앞 바이트를 흘림 → flash 시 `-B 100 -i 100` + erase/write 분리 필수
   (memory: pi5-attiny-isp). 이미 Makefile/flash.sh에 반영됨.

7. **READY 디바운스 80ms 핸드오프 마진** — 양호 (변경 금지 주의)
   Pi `HANDOFF_WAIT_S=0.3`에 의존. Pi측 0.3을 줄이면 충돌 위험 — 함께 봐야 함.

8. **`os_led_poweroff`가 edge 폴링** — 양호 (의도된 설계)
   Pi5 lgpio의 `when_pressed`가 이벤트 놓침 → 20ms 폴링으로 우회. 콜백으로 되돌리지 말 것.

---

## 9. Pi-side 데몬 (pi/)

| 파일 | 역할 |
|---|---|
| `os_led_display.py` | 시작 시 READY HIGH → 0.3s 후 SPI로 흰색→무지개 bloom(1.5s) → 60fps 무지개. SIGTERM시 검은프레임+READY LOW |
| `os_led_poweroff.py` | GPIO27 20ms 폴링. rising edge → `os.sync()` + `systemctl poweroff -f`. 성공시 커널이 죽임, 실패시 살아남아 재시도 |
| `os-led-display.service` | display 데몬 (Restart=always) |
| `os-led-poweroff.service` | poweroff 데몬 (Restart=always, root 필요) |
| `logind-override.conf` | HandlePowerKey=ignore / LongPress=poweroff |
| `system-fast-shutdown.conf` | DefaultTimeoutStopSec=10s (종료 90s 늘어짐 방지) |
| `install.sh` | 위 전부 설치 + enable + restart |
| `blink_gpio10.py` | DIN 소유권 검증 툴 |

`poweroff -f`(single force): graceful(서비스 SIGTERM)이지만 inhibitor lock 무시. 10s 타임아웃 캡과 페어.

---

## 10. 빌드 / 플래시 / 설치 치트시트

```bash
# 펌웨어 빌드 + 플래시 (ISP 배선 + 런타임 케이블 분리 상태에서)
cd /home/limdaemin/OS_LED/firmware && make && sudo bash flash.sh
# 플래시 후: ISP 케이블 제거 → 런타임 배선 복구 →
sudo systemctl start os-led-display

# fuse는 신품 칩에 1회만
cd firmware && make fuses

# Pi 데몬/유닛 설치·갱신
sudo bash /home/limdaemin/OS_LED/pi/install.sh
```

ISP 배선 / 플래시 중 분리해야 할 케이블 / Pi5 RP1 SPI 함정 등 상세는
auto-memory의 `attiny-flash-procedure`, `pi5-attiny-isp`, `pi5-attiny-isp-gpio-fallback` 참고.

### 플래시 시 절대 잊지 말 것
- **`os-led-display` 먼저 정지** (`/dev/spidev0.0` 점유 해제). flash.sh가 해주지만 수동 avrdude 땐 직접.
- 쓰기 도중 Ctrl-C 금지 (RP1 SPI 드라이버 락 → 재부팅 필요).
- ISP 중 ATtiny에서 분리: TTP223 OUT, NPN base, WS2812 DIN, +5V(대신 Pi 3V3), J2 케이블.
