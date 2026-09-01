# boot/ — 부팅 시퀀스 구성요소 모음

라즈베리파이 전원 인가부터 "말 걸어도 되는 상태"까지의 모든 부팅 관련 파일을 모아둔다.
런타임 코드(voice_pipeline, cpp)를 제외하면, 새 기기 부팅 세팅에 필요한 것은 전부 여기에 있다.

## 부팅 타임라인 (unit4 실측, 커널 기준 monotonic)

| 시각 | 이벤트 | 주체 |
|---|---|---|
| 전원 인가 | 노란(233,233,50) 호흡 시작 | ATtiny 펌웨어 (Pi와 무관하게 항상) |
| ~+13s | LED 하드웨어 PWM 준비 | `led-pwm.service` (system) |
| ~+13s | Pi가 스트립 인수, **같은 노란 호흡을 이어서** 표시 | `os-led-display` 데몬 |
| ~+14s | C++ 기동 → 모터 토크 온 → 자이로 캘리브레이션 (이완→수평→장력) | `ray-cpp.service` |
| ~+27s+ | 캘리브레이션 완료, WebSocket 서버 오픈 | ray-cpp |
| ~+52s | 파이썬 파이프라인 준비 → LED를 RAY가 인수(노란 호흡 유지) → **준비 완료 차임** → 웨이크워드 대기 | `ray-python.service` |

호흡은 세 구간(ATtiny → Pi 데몬 → RAY)이 같은 색(233,233,50)·같은 속도(4.0s 사인, 밝기 0.15~1.0)로
이어지도록 맞춰져 있다. 소리는 "준비 완료" 시점에 한 번만 난다 — 값을 바꾸면 세 곳을 함께 바꿀 것
(ATtiny 펌웨어, `OS_LED/pi/os_led_display.py`, `voice_pipeline/adapters/led.py`의 BreathingAnimation).

데몬은 첫 인수 전까지 노란 호흡을 그리고, RAY가 잡았다 놓은 뒤에는 **소등을 유지**한다
(개발 중 RAY를 꺼두면 LED도 꺼짐; RAY 재시작 시 페이드로 재인수). 시스템 종료 시에는
데몬이 내려가며 ATtiny 종료 호흡이 이어받는다.
(초기 인수 테스트용이던 무지개 표시는 실운용에 불필요해 제거, 2026-09)

## 구성요소

| 경로 | 역할 | 설치 위치 |
|---|---|---|
| `OS_LED/` | 전원/부팅 LED 서브시스템 (ATtiny 펌웨어 지식 + Pi 데몬). 상세: [OS_LED/README.md](OS_LED/README.md) | `sudo bash OS_LED/pi/install.sh` |
| `systemd/ray-cpp.service` | C++(모터·오디오) 부팅 자동실행. 시작 시 자이로 캘리브레이션 수행 | `~/.config/systemd/user/` (경로 수정 후 복사) |
| `systemd/ray-python.service` | 파이썬 파이프라인 부팅 자동실행 | `~/.config/systemd/user/` (경로 수정 후 복사) |
| `sounds/ready.oga` | 준비 완료 차임. `voice_pipeline/__main__.py`의 `READY_CHIME_PATH`가 참조 | 저장소 그대로 사용 (WorkingDirectory 기준 상대경로) |

## 새 기기 설치 절차

기본 환경(uv, 모델, API 키 등)은 [docs/SETUP.md](../docs/SETUP.md) 를 먼저 따른다. 그 다음:

```bash
# 1. OS_LED 데몬 + 전원 훅
sudo bash boot/OS_LED/pi/install.sh

# 2. systemd user 유닛 — 파일 내 경로(%h/KIST_RAY/Ray)를 이 기기의 저장소 경로로 바꿔 복사
mkdir -p ~/.config/systemd/user
sed 's|%h/KIST_RAY/Ray|%h/<저장소경로>|g' boot/systemd/ray-cpp.service    > ~/.config/systemd/user/ray-cpp.service
sed 's|%h/KIST_RAY/Ray|%h/<저장소경로>|g' boot/systemd/ray-python.service > ~/.config/systemd/user/ray-python.service

# 3. 환경 파일 (API 키, RAY_UNIT, tiktoken 캐시 경로)
mkdir -p ~/.config/ray
cat > ~/.config/ray/ray.env <<'EOF'
RAY_UNIT=unitN
OPENAI_API_KEY=...
ELEVENLABS_API_KEY=...
GOOGLE_APPLICATION_CREDENTIALS=/home/<user>/...json
TIKTOKEN_CACHE_DIR=/home/<user>/.cache/tiktoken
EOF
chmod 600 ~/.config/ray/ray.env

# 4. 부팅 자동실행 (GUI 없는 구성에서도 user 매니저가 뜨도록 linger 필수)
loginctl enable-linger $USER
systemctl --user daemon-reload
systemctl --user enable ray-cpp.service ray-python.service
```

## 소리 (준비 완료 차임)

- 재생 시점: ray-python이 웨이크워드 대기에 진입하는 순간 1회 (`_play_ready_chime()`).
- 교체: `boot/sounds/ready.oga`를 바꾸면 됨 (pw-play가 읽을 수 있는 포맷 — wav/ogg/oga 등).
- 과거의 `boot-chime.service`(PipeWire 기동 직후 재생, 기기 로컬 유닛)는 폐기됨 —
  캘리브레이션 도중에 뜬금없이 울리는 문제로 재생 주체를 파이프라인으로 이동.
