# 온라인 입 모양 궤적 평가 (Online Mouth Trajectory Evaluation)

기존 파이프라인(`cpp/`, `voice_pipeline/`)을 **건드리지 않는** 자급식 평가 도구.
TTS WAV를 로봇으로 구동하면서 온라인 생성된 입 모터 궤적 로그를 찍고, **음성학적으로
객관적인 지표**로 "부드럽고 사람다운가"를 평가한다.

> 오프라인(MATLAB) 궤적은 이 장비에 생성기가 없으므로 현재 범위에서 제외.
> 오프라인 레퍼런스를 받으면 `analyze.py`에 비교 축을 추가한다.

## 구성

```
motion_eval/mouth/
├── gen_wavs.py     # 음성학 스트레스 예문 + 자연문 WAV 합성 (scripts/tts_to_file.py 재사용)
├── capture.sh      # ./build/Ray 구동 + 각 WAV 재생 + pos4_audio 로그를 예문명으로 캡처
├── analyze.py      # 객관 지표 계산 + HTML 리포트 생성
├── wavs/           # 평가용 WAV
├── logs/           # 캡처된 예문별 온라인 로그 (target/actual/audio, 40ms tick)
└── reports/        # report.html
```

## 사용법

```bash
# 1) 평가 WAV 합성 (한 번)
uv run python motion_eval/mouth/gen_wavs.py

# 2) 로봇 구동하며 로그 캡처 (하드웨어)
bash motion_eval/mouth/capture.sh

# 3) 객관 분석 + 리포트
uv run python motion_eval/mouth/analyze.py
# -> motion_eval/mouth/reports/report.html
```

## 평가 지표 (음성학·공학 근거)

### 하드웨어 (평가 해석에 필수)

3D 프린트 스켈레톤 얼굴, 모터 5개, **텐던(실)-풀리 구동**. 모터가 실을 감으면 당겨지고
풀면 풀린다 (당기기만 가능, 푸시 불가).

| ID | 모터 | 역할 |
|----|------|------|
| 1 | Pitch | 목젖 라인에서 올라오는 고개 끄덕임 |
| 2 | Roll_R | (정면 기준) 왼쪽 뒤 |
| 3 | Roll_L | 오른쪽 |
| 4 | Yaw | 고개 좌우 회전 |
| 5 | **Mouth** | 턱에 실 연결 — **실을 당겨 입이 열림** |

- **닫힘은 능동 구동이 아니라 수동 복원**(중력/탄성) → 열림/닫힘 동특성 비대칭 가능 → `asym` 지표로 확인.
- 입 실을 당기면 고개가 딸려오므로 **보상계수 2개**: `mouth_pitch_compensation`(0.6, pitch 빼줌),
  `mouth_back_compensation`(0.8, roll R/L 빼줌). 보상 부족/과다 시 입 움직일 때 고개 들썩.
- ⚠ **로그의 `actual_pos4`는 ID5 모터(실 스풀) 위치**이지 실제 턱 벌어짐(mm)이 아니다.
  실 슬랙/늘어짐으로 모터 추종이 좋아도(r≈0.99) 물리적 개구는 다를 수 있음 →
  **진짜 개구량 검증은 영상 필요**.

### 평가 전제

이 로봇은 **턱 1자유도(개폐)**. 기준은 "사람이 같은 말을 할 때의 턱 운동".
진폭 엔벨로프는 그 대리값이며 구조적 한계(양순음 /m,b,p/는 에너지 있어도 닫혀야 함)가 있다.

| 지표 | 의미 | 좋은 값 |
|---|---|---|
| 음절 이벤트 정합 (precision/recall) | 음절 핵마다 입을 한 번씩 여는가 | recall·precision ↑ |
| 변조 스펙트럼 피크 | 입 궤적의 주기성이 음절률(3~8Hz)에 있는가 | 3~8Hz, 오디오와 일치 |
| 동기 지연 | 오디오↔입 시간차 | ±지각한계(영상 ≤~125ms 뒤) |
| 추종오차/지연 (target↔actual) | 서보가 명령을 따르는가 | r↑, RMS↓ |
| 부드러움 (jerk/chatter) | 떨림 없이 매끈한가 | chatter ≤ 음절률 |
| 포화율 | 만개/완전닫힘에 고정되는가 | 낮음 |
| 무음 폐쇄율 | 묵음에서 닫히는가 (과검출) | 묵음 구간 닫힘 |
