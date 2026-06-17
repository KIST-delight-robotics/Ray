# Decision Log — Work in Progress

진행 중인 작업의 결정 기록. 작업 완료 후 정리하여 `decisions.md`에 통합.


## 턴 종료(cancel/interrupt) 재설계 — Phase 상태기계

- **cancel/interrupt 경계 = begin_streaming**: "레이 음성이 잠깐이라도 났으면 무조건 interrupt(cancel 아님)"라는 사용자 관점 기준을 만족시키려면 재생이 *가능해지는* 마지막 Python 지점을 경계로 잡아야 함. `playback_started`는 C++가 `play()` 호출 시점(가청 시작보다 SFML/ALSA 버퍼만큼 앞)에 보내고 Python은 폴링 지연 후 처리 → 가청 시작과의 선후가 수십 ms 내에서 모호. `begin_streaming`(=`send_stream_start`)은 그 이전엔 C++에 재생 명령이 없어 *물리적 무음 보장*이 되는 유일 지점. 그래서 cancel은 항상 begin_streaming 이전(브리지 무접촉)이라 `send_stop` 불필요·STOPPING 미경유, interrupt는 항상 이후 — STOPPING이 항상 interrupt 의미가 되어 reason 태그 불필요.

- **STREAMING은 종료-감지 공백 구간(단일채널 interrupt 안 함)**: `robot_audio`(VAP 참조 채널)는 `playback_started`가 클럭을 세팅해야 생김. 그 전 STREAMING에선 VAP가 interrupt vs backchannel을 구분할 robot 채널이 없음. 단일채널 fallback은 (a) backchannel 오인터럽트, (b) stop_pos≈0 빈 기록 문제가 있어 채택하지 않고 interrupt 감지를 PLAYING으로 미룸. 비용: STREAMING 내에서 시작·종료하는 짧은 barge-in은 그 턴엔 무시됨(begin_streaming에서 ASR이 reset되고 이후 누적되므로 다음 턴 입력으로 이연 — 유실은 아님). bridge_ms 실측 중앙값 ~97ms라 공백이 보통 짧고, 긴 STREAMING(느린 TTS)이 최악. 근본 대응은 C++ prebuffer(INTERVAL_MS=360ms 분량)·TTS throughput이지 단일채널 감지가 아님.

- **cancel 신호 = user_is_speaking 전제 + p_now/p_fut(즉시), grace는 similarity**: cancel은 interrupt와 동일 구조 — `user_is_speaking`을 전제로 하고(실제로 말해야 함) 그 위에서 turn-taking 확률 `p_now/p_fut`로 플로어 회수를 확인. user_is_speaking *단독*은 backchannel·노이즈에 약하고, p *단독*은 무음 중 확률 변동에 오발화하므로 둘 다 필요. VAP가 네이티브 10Hz(각 결과가 이미 ~100ms 적분)라 100ms 미만 프레임 sustain은 같은 캐시 추론 재독이라 무의미 → 즉시 발화. ASR finalization noise는 시간 grace 대신 **마지막 prepare 텍스트(=응답이 생성된 기준)** 와의 유사도로 거름 — prepare-skip 게이트(`sim≥0.8`→재생성 생략)와 **같은 비교의 양면**이라 별도 기준선(T0) 추가 없이 `_last_prepare_text` 재사용. 이 user_is_speaking 전제가 "침묵 timeout(turn_shift Path2)으로 shift했는데 p만 높은" 모순 입력에서의 turn_shift↔cancel thrash도 막음.

- **detector 상태 wipe를 turn_shift→commit으로 지연(PENDING 도입)**: turn_shift는 로봇이 실제 커밋(begin_streaming)하기 전까진 잠정적. 기존엔 turn_shift 직후 per-frame 상태를 전부 지워 "cancel=같은 턴 연속"이 불가능했음. PENDING은 상태를 보존해 cancel 시 매끄럽게 rewind하고, commit에서 비로소 wipe+dialog append. 부수 효과로 detector가 interrupt 모드(ROBOT_TURN)에 진입하는 시점이 robot_audio가 생기는 시점과 일치 → 기존의 "ROBOT_TURN인데 robot_audio 없음" 사각이 소멸.

- **stale 응답 방지 = turn_shift의 prepare 선점 (detector 내부)**: turn_shift 조건이 충족돼도, 마지막 prepare 이후 ASR이 *유효하게(비유사)* 바뀐 게 남아 있으면(=`_check_prepare`가 발화하면) turn_shift 대신 **prepare를 먼저** 내보내 새 텍스트로 재생성하고 다음 프레임에 shift. "늦은 finalization으로 준비된 응답이 stale" 케이스를 detector의 **기존 유사도 게이트**(`_last_prepare_text` 대비)로 그대로 처리 — SessionLoop에 임베더/`similarity_fn` 주입 불필요(유사도 검사 중복 회피). prepare 선점은 `_check_prepare`의 `_asr_has_changed` 게이트 덕에 *미처리 변화가 있을 때만* 일어나, 텍스트가 안정된 흔한 경우엔 turn_shift가 바로 fire(speculation 이득 유지). (SessionLoop에 별도 `similarity_fn` staleness 가드를 두는 안은 always-on detector 검사와 불일치·중복이라 기각.)
- **Python↔C++ 순수 전송은 무시 가능(~0.04ms 편도, Pi 루프백 IXWebSocket 실측)**: turn-taking 타이밍에서 전송은 0으로 취급. bridge_ms(~57-97ms)는 통신이 아니라 C++ prebuffer + Python 프레임 폴링(~30ms)이 지배.

## [설계 검토 중] 음악 댄스 모드 — 리듬 동기 LED 디밍 + 모터

> 상태: 설계 단계. 구현 전 검토 필요.
> 결정: 비트 검출 Python(librosa). **기존 코드와 완전 분리된 새 최상위 폴더에 순수 Python 단독**으로 구현.
> 지금은 독립 실행(메인 음성 로봇과 동시 구동 X — 시리얼 포트 단독 점유), **나중에 메인 로봇의 한 모드로 통합 예정**.

### 목표

곡 오디오를 입력받아 **리듬/비트(+에너지·피치)** 를 분석하고, 그 비트에 맞춰
- **LED를 디밍(번쩍임)** 하고
- **모터가 동기되어 움직이는** 댄스 상태를 만든다.

LED 깜빡임과 모터 움직임이 **시각적으로 딱 맞아 보여야** 한다.

### 아키텍처 결정 — 순수 Python 단독 (한 프로세스·한 클럭)

`voice_pipeline/`·`cpp/`를 전혀 건드리지 않는 **새 최상위 폴더**(가칭 `music_dance/`)에 모든 것을 Python으로:
- 비트/리듬 분석: **librosa**
- 모터·LED 제어: **dynamixel_sdk (Python)** — `/dev/ttyUSB0` 직접 점유
- 오디오 재생: **sounddevice/pyaudio**

→ 한 프로세스 안에서 **재생 위치(playback position)라는 단일 클럭**이 LED 디밍과 모터 골을 함께 구동.
브릿지/IPC가 없으므로 지연·지터가 사라지고 동기화가 자연히 해결됨. (이전 검토의 C++실행/브릿지 후보들은 폐기.)

> ⚠️ 기존 `cpp/`는 곡을 **보컬 스템**으로 재생·모션 생성하는 경로(`play_music`)와 40ms 모터 루프, RPY2DXL
> 케이블 기구학을 갖고 있음. 단독 Python 버전은 이를 재사용하지 않고 처음부터 가볍게 만들되, 모터 틱 매핑
> (home 위치, mouth 범위 등 config.toml 값)과 기구학 개념은 참고만 한다. **통합 시점**에 메인 로봇으로
> 분석 코어(analyzer/timeline)를 옮길 수 있도록 제어부와 분리해 둔다.

### 핵심 미결 질문 — LED 제어 능력 (하드웨어 확인 필요)

현재 ID6 "LED"는 **서보 위치**(`writeSingleGoalPosition` → `ADDR_GOAL_POSITION`)로 제어 중.
드라이버에 PWM write 없음. 사용자 견해: "PWM 1 아니면 0" (온/오프 가능성).

- (a) **연속 밝기 가능**(PWM 듀티 범위 or 위치→밝기 매핑) → 진짜 디밍.
- (b) **온/오프만 가능** → 비트에 맞춘 **온/오프 스트로브**, 또는 소프트웨어 PWM(고속 듀티 변조)로 의사 디밍.
- 단독 Python에서는 dynamixel_sdk로 ID6에 Goal Position(또는 PWM 모드 시 Goal PWM, addr 100) 기록.
  → 디밍하려면 PWM 모드(operating_mode=16) + Goal PWM 범위 확인 필요.

### LED 구동 신호 — 확정 (단색 LED, 밝기 1채널, HPSS 기반)

LED는 **단색** → 색조 매핑 없음. **밝기만** 제어. 곡을 **오프라인 통째 분석**하므로
HPSS(median filtering)를 자유롭게 사용. 음향학적으로 타당하고 자연스러운 배경 워시를 위해
밝기를 **두 성분의 블렌드**로 구동:

```
brightness(t) = clamp( floor + w_h·H̃(t) + w_p·P̃(t) , 0, 1 )
```

- **H̃ = 하모닉 에너지 (무드/글로우 바닥)** — `librosa.effects.hpss`의 하모닉 성분.
  지속음(코드·멜로디·보컬). 곡의 무드대로 **천천히 차오르는 베이스 밝기**. 느린 attack/release.
- **P̃ = 퍼커시브 에너지 (비트 번쩍)** — HPSS 퍼커시브 성분. 킥·스네어·하이햇 트랜지언트.
  **fast attack / slow release** 로 톡톡 튀는 강조. (체감 펄스 강화하려면 저역 가중 옵션.)
- 이전의 "저음 에너지 + RMS"보다 격상: 멜로디와 드럼이 섞이지 않아 **박은 선명, 무드는 매끄럽게**.

**처리 체인 (순서가 곧 자연스러움):**
1. HPSS로 H/P 분리 (오프라인, 곡 전체).
2. 각 성분 프레임별 에너지(RMS, hop~512). **dB(로그) 스케일** + 지각 음량 가중(최소 dB, 가능하면 A-weighting).
3. **퍼센타일 정규화**(예: 5~95%ile → 0~1) — 조용한 곡/시끄러운 곡 모두 다이내믹 레인지 활용, 이상치 강건.
4. **비대칭 스무딩** — 하모닉: 느린 attack·release(부드러운 바닥). 퍼커시브: fast attack / slow release(펀치).
5. 블렌드(`w_h`,`w_p`) 후 **감마 보정 ~2.2** (지각-선형 밝기, 저밝기 뭉갬 방지).
6. **밝기 바닥값(floor ~10~20%)** — 완전 소등 반복(스트로브) 방지. 배경 워시는 늘 은은히 켜져 있어야.

**음향학적 근거:** 사람은 음량을 로그(dB)로, 밝기를 비선형(감마)으로 지각 → 두 보정이 "귀로 듣는 크기 ≈
눈으로 보는 밝기" 정합을 만든다. 퍼커시브의 fast-attack/slow-release는 실제 타악기 엔벨로프(빠른 어택,
지수 감쇠)를 모사 → "반응한다"는 느낌의 핵심.

- 연속 디밍 불가(온/오프뿐)일 경우: 퍼커시브 임계값 통과 시 펄스(스트로브화), 하모닉 글로우는 표현 불가.
- **실시간 주의:** HPSS는 lookahead 필요 → 오프라인 전제에서만 유효. 통합 후 실시간 마이크 입력 시 재설계 필요.

### 컴포넌트 설계 (잠정, `music_dance/`)

```
music_dance/
├── analyzer.py    # librosa: beat_track/onset_strength/(chroma) → DanceTimeline  ← 통합 시 재사용 코어
├── timeline.py    # DanceTimeline: 시각→(밝기[0..1], 모터 포즈) 샘플링
├── motor.py       # dynamixel_sdk 래퍼: LED 밝기 + 서보 골 쓰기
├── player.py      # 오디오 재생 + 재생 위치(마스터 클럭) 제공
├── dance.py       # 오케스트레이션: 곡 로드→분석→재생+동기 구동 루프
└── __main__.py    # CLI: python -m music_dance <song.wav>
```

구동 루프(단일 스레드): 재생 위치 t를 읽음 → `timeline.sample(t)` → LED 밝기·모터 골 기록 →
짧은 주기(예: 20~40ms) 반복. 한 클럭이라 LED·모터·오디오가 정합.

### 열린 항목

- LED 제어 능력 (연속/온오프) — **하드웨어 확인 필요**.
- 모터 안무: 비트에 맞춘 단순 동작(끄덕임/흔들기)부터. 기구학 전체(RPY2DXL) 필요 여부는 동작 보고 결정.
- 곡 입력: 파일 경로 CLI 인자로 시작. (트리거/음성명령은 통합 단계에서.)
- 폴더명 확정(`music_dance/` 가칭).
- LED 연속 디밍 가능 여부 (PWM 듀티/위치→밝기) — 하드웨어 확인 필요. 온/오프뿐이면 스트로브화.


## [구현] LED — cpp 파이프라인 통합 (오프라인 CSV 구동), 밝기 경로 확정

> 위 "순수 Python 단독" 검토와는 별개 트랙. PC에서 오프라인 분석한 `<곡>-led.csv`를
> **기존 cpp `csv_control_motor`가 헤드·입과 같은 40ms 프레임으로 함께 구동**하는 핸드오프
> (`LED_handoff_bundle`)를 채택해 구현. 단독 Python 버전이 아니라 현 메인 cpp 경로에 얹었다.

### 밝기 구동 경로 — WiringPi 소프트웨어 PWM (GPIO), Dynamixel과 분리

미결이던 "LED 밝기 제어 능력"을 **별도 GPIO 핀 PWM**으로 확정. ID6 Dynamixel은 LED 막대
**각도(위치 제어)** 만 담당하고, **밝기는 라파 GPIO 핀의 PWM**으로 따로 구동한다(두 채널 분리).

- **softPwm 선택 이유:** WiringPi가 이미 자이로 I2C용으로 링크돼 있어 추가 의존성 0.
  softPwm은 **핀 제약이 없어** 배선 핀을 자유롭게 고를 수 있다. 단점은 주파수 한계 —
  기본 단위 100µs이라 `range=100`이면 ~100Hz. 핸드오프 권장(>200Hz)엔 못 미쳐 영상/움직임에서
  플리커가 보일 수 있다. **더 높은 주파수가 필요하면 하드웨어 PWM 핀 + `pwmWrite`로 교체**(단,
  HW PWM은 특정 핀만 가능). 첫 통합은 범용·무제약 우선으로 softPwm.
- **핀 = BCM GPIO13** (물리 33핀). `wiringPiSetupGpio()`로 BCM 번호를 그대로 config에 받는다
  (wPi 번호 환산 불필요). 기존 자이로 코드는 `wiringPiSetup()`(wPi 번호)를 쓰지만 현재 비활성
  (`initialize_robot_posture`/`gyro_test` 주석)이라 한 바이너리에 두 넘버링이 충돌하지 않음.
  자이로 재활성 시 numbering 모드 통일 필요.
- `led_pwm_pin < 0` 이면 비활성: 배선 전이나 다른 보드에서 임의 GPIO를 건드리지 않게 방어.

### 각도 좌표 — CSV 절대 tick을 우리 home에 재앵커링 (bundle 3 스펙 변경)

bundle 1~2: LED CSV col0 = ID6 홈 기준 **상대 deg**, cpp가 `default_led + led_dir*deg*ticks_per_deg`로
변환. bundle 3부터 **col0 = PC가 만든 절대 tick(1550~2233, PC home_tick=1550 기준)** 으로 바뀜.
핸드오프는 "cpp 변환 없이 그대로 write"를 요구하지만, 그건 **하드웨어 ID6도 1550=벽정면**일 때만 맞다.

문제: 이 로봇의 ID6 실제 home(`default_led`)은 1550이 아닐 수 있다(사용자가 직접 튜닝). 절대 tick을
그대로 쓰면 LED가 엉뚱한 각으로 가거나 기구 하드스톱에 처박혀 모터가 버틴다.

결정: **CSV 절대 tick을 우리 home에 재앵커링**한다.
```
goal = default_led + led_dir * (csv_tick - led_csv_home)   // led_csv_home = PC home = 1550
```
- `(csv_tick - led_csv_home)` = PC가 만든 0~60° 스윕의 상대 오프셋(모양·크기 보존).
- `default_led`로 전체 궤적을 물리 home에 맞춰 올리고내림(사용자가 원하던 knob). `default_led==led_csv_home`이면
  절대치 직접 write와 동일.
- **PC 재생성 불필요** — 절대치든 상대치든 수학적으로 동일, cpp에서 빼는 쪽이 깔끔.
- `led_dir`로 우리 모터 회전 방향 보정(필요 시 -1). 더 이상 `ticks_per_deg`는 안 씀(CSV가 tick 단위).
- 한 zip에 여러 곡(곡별 `trajectory/<곡>/`) → `assets/{head,mouth,led}Motion/`에 곡명 그대로 배치.

### 프레임 정렬

LED CSV는 헤드/입과 **행 수·시간축이 같다**는 핸드오프 전제에 의존. `csv_control_motor`의
SKIP_FRAMES 보간 구간에서도 LED 행을 같은 박자로 소비해 정렬을 유지(LED는 보간 없이 원본값).
LED 파일이 없으면 graceful 비활성(각도·밝기 미구동), 모터 1~5는 정상 재생.

### 정리한 것

- 전류 측정용 임시 ID6 사인파 스레드(`led_motor_sine_loop`) 제거 — 측정 완료.

### 시그널 핸들러 데드락 (Ctrl+C가 음악 안 멈춤) — 워처 스레드로 전환

csv 재생 중 Ctrl+C를 눌러도 프로세스가 안 죽고 노래가 계속 나오던 버그. 원인: `signal_handler`가
**인터럽트당한 스레드 컨텍스트에서** `cleanup_dynamixel()` → `dxl_mutex_` 잠금을 시도했는데,
csv 루프(메인 스레드)가 매 프레임 모터 write로 그 뮤텍스를 쥐고 있어 **자기 자신을 기다리는 데드락**.
`std::_Exit`까지 못 가고, SIGTERM도 같은 데드락이라 SIGKILL만 들었다. (애초에 시그널 핸들러에서
mutex/`cout`/`new`·`delete`는 async-signal-safe가 아님.)

해결: 핸들러는 **원자 플래그(`g_shutdown_requested`)만** 세우고 반환. 별도 `shutdown_watcher`
스레드가 폴링하다 **정상 컨텍스트에서** LED 소등 + 토크 해제 후 `_Exit`. 별도 스레드라 메인이
뮤텍스를 놓는 순간 정상 획득 → 데드락 없음. dxl_driver를 `delete`하진 않음(메인 스레드와
use-after-free 경합 회피) — 토크만 끄고 즉시 종료, 나머지는 OS가 회수.


## 차후 고려

- **SimilarityConfig/MemoryConfig 임베딩 필드 중복**: 양쪽 config에 model, use_onnx 등이 중복 존재. 공유 EmbeddingConfig 추출 여부는 실제 사용 패턴 보고 판단.
- **similarity.compare() 임베딩 캐싱**: TurnDetector 호출 패턴에서 `a`(이전 텍스트)가 반복됨. 한쪽 임베딩을 캐싱하면 추론 비용 절반 가능. 기존 코드도 동일 패턴이라 regression은 아님.
- **similarity 유닛 테스트 부재**: EmbeddingSimilarity, DiffLibSimilarity, create_similarity 팩토리에 대한 유닛 테스트가 없음. 현재는 TurnDetector 테스트에서 ISimilarity를 mock하여 간접 검증.
