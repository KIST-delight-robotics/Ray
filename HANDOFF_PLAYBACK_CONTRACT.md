# [Pi → Windows] 궤적 재생 계약서 (LED 중심)

> Windows에서 만든 궤적/오디오를 Pi 로봇이 재생하는 방법·파일계약·하드웨어 정리.
> 전부 `cpp/main.cpp`(함수 `csv_control_motor`, 1904~2145줄)·`cpp/config.toml`·`CMakeLists.txt`에서 확인한 사실.
> 검증일 2026-06-09, 환경 Raspberry Pi 5 (aarch64), 레포 루트 `/home/ray_mk3/KIST_RAY/Ray`.

---

## A. 궤적 재생

### 재생 실행 명령 (테스트용 CSV 재생 모드)
```bash
cd /home/ray_mk3/KIST_RAY/Ray         # ★작업 디렉토리 = 레포 루트 (경로가 'assets/...' 상대경로라 필수)
./build/Ray --csv <곡이름>            # 예: ./build/Ray --csv V_ZionT
```
- 진입점: `main.cpp:2898` `if (argv[1]=="--csv") csv_control_motor(argv[2])`.
- 별도 env 불필요(LD_LIBRARY_PATH 등 없음). 단 모터/시리얼 권한(`/dev/ttyUSB0`)과 PWM 권한 필요(아래 B 참고, 이미 셋업됨).
- 음성 파이프라인 정식 경로(웹소켓 9200)로도 같은 함수가 호출됨(`main.cpp:2782`). 단독 재생 확인은 `--csv` 모드가 제일 간단.

### 빌드 (바이너리 `build/Ray`)
```bash
cd /home/ray_mk3/KIST_RAY/Ray
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # MOTOR_ENABLED 기본 ON (모터+WiringPi+PWM)
cmake --build build -j
```
- `MOTOR_ENABLED=ON`(기본)이라야 LED/모터가 실제 구동된다. OFF면 더미(모션 미구동).

### 재생기가 읽는 입력 파일 (절대경로, `<곡>` = 동일 이름으로 통일)
| 채널 | 절대경로 | 열 구성 | 비고 |
|---|---|---|---|
| 헤드 | `/home/ray_mk3/KIST_RAY/Ray/assets/headMotion/<곡>.csv` | `roll,pitch,yaw` (3열) | 40ms/행 |
| 입 | `/home/ray_mk3/KIST_RAY/Ray/assets/mouthMotion/<곡>-delta-big.csv` | `mouth_delta, ratio, …` (col0,col1 사용) | 40ms/행. ratio는 코드에서 ×1.4 |
| **LED** | `/home/ray_mk3/KIST_RAY/Ray/assets/ledMotion/<곡>-led.csv` | `tick, brightness` (2열) | 40ms/행. 아래 B |
| 오디오 | `/home/ray_mk3/KIST_RAY/Ray/assets/audio/music/<곡>.wav` | — | SFML로 재생 |

- 헤드·입 파일은 **필수**(없으면 함수가 즉시 return). LED는 없으면 graceful 비활성(`main.cpp:1949-1954`). 오디오 없으면 return.
- "비브라토" 같은 별도 파일은 이 재생기엔 **없다**. 입력은 위 4종뿐 — vibrato는 head/mouth 중 하나에 합쳐서 넣어야 함.

### 파일명 규칙 (정확히)
- head: `<곡>.csv`  ·  mouth: `<곡>-delta-big.csv`  ·  led: `<곡>-led.csv`  ·  audio: `<곡>.wav`
- 네 파일의 `<곡>` 부분이 **완전히 동일**해야 한 곡으로 묶여 재생됨.

---

## B. LED 데이터 계약

### LED CSV 열 구성 (`main.cpp:1955-1963`)
2열, **헤더 없음**, 한 행 = 한 40ms 프레임:
```
col0 = tick        # ID6 Dynamixel 막대 '각도' (절대 tick, led_csv_home=1550 기준)
col1 = brightness  # 밝기, 0.0 ~ 1.0 (float)
```
- col0 → `led_csv_tick_to_goal()`로 우리 home에 재앵커링 후 ID6 모터 위치로 전송(`main.cpp:2112`).
- col1 → `set_led_brightness()`로 하드웨어 PWM duty로 전송(`main.cpp:2114`).

### "모터 안 움직임" — 열 개수
- **여전히 2열을 기대한다.** col1(밝기)이 없으면 밝기가 0으로 떨어져 LED가 꺼진다(`main.cpp:1962`, 기본값 brightness=0).
- 모터를 안 움직이려면 **col0를 매 행 `1550`(=led_csv_home)으로 고정**하면 됨 → 재앵커링 결과가 home이라 ID6 정지. col1만 밝기로 변조.
  즉 권장 행 형식: `1550,<brightness>`.

### ⚠️ 지금은 LED CSV가 무시됨 (반드시 확인)
`main.cpp:2073-2108` 에 **[임시 테스트] 블록**이 있어, CSV에서 읽은 tick·brightness를 **덮어쓴다**.
현재 설정: `BLINK_ON=true`(0.3s 주기 점멸), `MOVE_ON=false`(모터 정지). → **CSV 내용이 반영되지 않음.**
- Windows가 만든 LED CSV로 구동하려면 **이 블록(2073-2108)을 삭제/주석** 처리해야 한다(주석에 "되돌릴 때 이 블록 삭제"라고 명시됨).
- 블록을 지우면 위 계약(col0=tick, col1=brightness)대로 CSV가 구동된다.

### 프레임 간격 / 헤더
- **40ms/행 = 25fps** (`constexpr FRAME_INTERVAL = 40ms`, `main.cpp:1933`). Windows의 40ms 격자와 동일.
- **헤더 없음.** 첫 행부터 데이터. (현 샘플 `Alone_coogie-led.csv`: 4477행 × 40ms ≈ 179s.)
- 행 수는 head/mouth와 같은 길이로, 곡 길이에 맞춰야 함(부족하면 그 채널만 먼저 끝남).

### 밝기 값 범위
- **0.0 ~ 1.0 (float).** `set_led_brightness`가 0~1로 clamp 후 `duty = brightness × period`로 환산(`main.cpp:539-545`).
- **0~255 / 0~100 아님.** PWM raw 직접 X. 1.0 = 완전 켜짐, 0.0 = 꺼짐.

### LED 채널 / 하드웨어 (현재 상태: 준비됨 ✅)
- **밝기** = RP1 **하드웨어 PWM**, `pwmchip0/pwm1` = **GPIO13(PWM1, 물리 33번)**. 캐리어 20kHz(period=50000ns, 측정값).
  - `led-pwm.service`(systemd, enabled+active 확인됨)가 부팅 시 export/period/enable/권한 설정. 셋업: `scripts/hardware/setup_led_hwpwm.sh`(sudo 1회 + 재부팅).
  - config: `led_pwm_pin=13`, `led_pwm_range=100`(softPwm 잔재, 현재는 하드웨어 PWM 사용).
- **각도(막대)** = ID6 Dynamixel(`ids=[1..6]`의 6번, "LED"), 위치제어. `default_led=1550`, `led_csv_home=1550`, `led_dir=1`.
  - (메모리의 "ID6 LED 미설정"은 옛 정보 — config·서비스 모두 설정 완료 상태.)

---

## C. 전송 위치 (scp 목적지)
Windows에서 만든 파일을 Pi의 해당 assets 폴더로 그대로 보낸다(`<곡>` 이름 일치):
```bash
scp <곡>-led.csv         ray_mk3@161.122.114.128:/home/ray_mk3/KIST_RAY/Ray/assets/ledMotion/
scp <곡>-delta-big.csv   ray_mk3@161.122.114.128:/home/ray_mk3/KIST_RAY/Ray/assets/mouthMotion/
scp <곡>.csv             ray_mk3@161.122.114.128:/home/ray_mk3/KIST_RAY/Ray/assets/headMotion/
scp <곡>.wav             ray_mk3@161.122.114.128:/home/ray_mk3/KIST_RAY/Ray/assets/audio/music/
```

---

## D. 동기화 (오디오 ↔ 궤적)
- 로봇이 **노래(wav)를 직접 튼다**: `csv_control_motor`가 `assets/audio/music/<곡>.wav`를 SFML `sf::Music`으로 재생(`main.cpp:1919-1929, 2018-2020`). → **wav도 같이 보내야 함**(C 참고).
- **싱크 방식**: 모션 프레임은 벽시계로 페이싱 — `sleep_until(csv_start_time + 40ms*step)`(`main.cpp:2129`). 음악은 첫 프레임에서 `music.play()` 한 번 호출. 즉 **궤적 0행 = 노래 0초**에 함께 출발, 이후 40ms 간격 유지.
- 따라서 **궤적은 25fps(40ms/행) 고정**이어야 노래와 안 어긋난다. fps가 다르면 시간이 흐를수록 드리프트.
- 시작점 맞추기: 별도 오프셋 없음. CSV 첫 행을 노래 0초에 대응시켜 생성하면 됨.

---

## 요약 체크리스트 (Windows가 한 곡 보낼 때)
1. `<곡>` 이름 통일 → head/mouth/led/wav 4파일 생성(전부 25fps, 헤더 없음).
2. LED CSV = `1550,<brightness 0~1>` 행들(모터 정지 + 밝기만). 곡 길이만큼.
3. 4파일을 C의 경로로 scp.
4. (1회) Pi에서 `main.cpp:2073-2108` 임시 블록 삭제 후 재빌드 — 안 하면 CSV 무시됨.
5. `cd 레포루트 && ./build/Ray --csv <곡>` 로 재생 확인.
