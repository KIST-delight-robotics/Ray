# music_dance — 음악 동기 LED + 모터

노래 한 곡을 받아 **리듬에 맞춰 LED가 디밍**되고 **모터(Dynamixel ID6)가 움직이는** 데모.
기존 음성 파이프라인(`voice_pipeline/`)·로봇 제어(`cpp/`)와 **완전히 분리**된 독립 모듈이다.
(지금은 따로 실행. 나중에 메인 로봇의 한 모드로 통합 예정 — 분석 코어를 재사용한다.)

## 구조

```
music_dance/
├── analysis/analyze.py   # Python: librosa HPSS 분석 → timeline.csv
├── motion/main.cpp       # C++: timeline + WAV → 재생 + 모터 + LED 동기 구동
├── motion/CMakeLists.txt
└── run.sh                # 분석 → 빌드 → 실행 한 방에
```

**한 프로세스 · 한 클럭**: C++가 WAV 를 재생하며 재생 위치(steady_clock)를 마스터 클럭으로
삼아 매 20ms 타임라인을 샘플링해 LED 밝기와 모터 위치를 동시에 쓴다. → 동기화 자연 보장.

## 신호 설계 (LED 밝기)

단색 LED 1채널. HPSS 로 하모닉/퍼커시브를 분리해 블렌드:

- **하모닉**(지속음) → 느린 글로우 바닥
- **퍼커시브**(드럼) → fast-attack/slow-release 펀치
- 처리: dB 스케일 → 퍼센타일 정규화 → 비대칭 스무딩 → 감마(2.2) → 바닥값(~12%)

모터는 퍼커시브 엔벨로프를 서보용으로 더 평활화해 비트에 맞춰 끄덕인다.

## 실행

```bash
cd music_dance
./run.sh                      # 루트의 V_ZionT_MR.wav 사용
./run.sh /path/to/song.wav    # 다른 곡
```

`run.sh` 가 (1) 분석 (2) 빌드 (3) PWM 채널 권한 준비(sudo 1회) (4) 실행까지 한다.

### 부분 실행 / 직접 실행

```bash
# 분석만
uv run --with librosa --with soundfile python analysis/analyze.py ../V_ZionT_MR.wav -o timeline.csv

# 빌드만
cmake -S motion -B motion/build && cmake --build motion/build -j

# 실행 (옵션)
motion/build/dance --timeline timeline.csv --wav ../V_ZionT_MR.wav \
  --port /dev/ttyUSB0 --baud 2000000 --id 6 \
  --motor-home 100 --motor-amp 300 --pwmchip 0 --pwmchan 1 \
  --no-motor     # 모터 없이 LED+오디오만
  --no-led       # LED 없이 모터+오디오만
  --sync-ms 80   # 오디오 버퍼 지연 보정(시각효과를 80ms 늦춤)
```

## 하드웨어 메모

- **LED**: 라즈베리파이 하드웨어 PWM `pwmchip0` 채널 1(GPIO13 계열). sysfs(`/sys/class/pwm`)로 제어.
  - 핀이 PWM 기능으로 먹싱돼야 출력됨. 안 나오면 `/boot/firmware/config.txt` 에
    `dtoverlay=pwm-2chan` 추가 후 재부팅. (PWM0/1 핀 매핑은 보드/오버레이에 따라 다름 — "일단 1로" 시도.)
  - export 는 root 권한 필요 → `run.sh` 가 sudo 로 한 번 열고 권한을 넘긴다.
- **모터**: Dynamixel ID6, `/dev/ttyUSB0`, 2 Mbaud(기존 config.toml 과 동일). 시리얼 접근은 `dialout` 그룹.
  - 안전을 위해 **Position 모드(0~4095 단일 회전)**, 기본 진폭 300틱(약 26°). `--motor-amp` 로 조정.
  - 핑 실패(배선/baud 불일치) 시 모터는 자동 비활성화되고 오디오+LED 만 구동된다.
- 종료(Ctrl-C) 시 모터는 홈 복귀 후 토크 해제, LED 는 소등한다.
