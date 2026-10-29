# 부팅 자동실행

> 2026-08 부팅 자동실행 작업의 기록. 같은 작업의 후속만 덧붙이고, 이후 다른 작업의 변경은 반영하지 않는다.

## 개요

전원을 켜면 C++ 프로세스(`build/Ray`)와 Python 파이프라인이 systemd 로 자동 실행되게 한 작업이다(2026-08-27).
다룬 것은 세 가지다.
- 유닛 사이의 기동 순서
- 부팅 때 네트워크가 없으면 죽는 의존 제거
- 기동 시간 측정

부팅 구성요소의 현재 구성과 설치 방법은 [boot/README.md](../../boot/README.md)에 있다.

## 기동 순서

- **user 유닛 + linger (system 유닛 기각)**:
  - `build/Ray` 는 SFML→OpenAL→PipeWire 로 소리를 내는데, PipeWire 는 유저 세션(`/run/user/1000`)에만 있다.
  - 대가: user 매니저는 system 유닛(`os-led-display`, `led-pwm`)에 `After=` 를 걸 수 없다. 그래서 경계를 넘는
    순서는 파일/소켓 존재 게이트(`ExecStartPre` 폴링)로 대신한다.
  - GUI 를 끄면 로그인 세션이 없으므로, `loginctl enable-linger` 없이는 RAY 도 PipeWire 도 뜨지 않는다.
- **python 은 cpp 게이트에 묶지 않는다**:
  - `After=ray-cpp` 는 cpp 의 `ExecStartPre`(PWM 게이트) 완료까지 python 을 잡아둔다. 실측 9.4 s.
  - python 은 모델 로딩(~44 s) 뒤에야 cpp 에 연결하고 `CppBridge` 에 재시도가 있으므로, 먼저 뜨는 편이 항상
    이득이다. 그래서 `Wants=` 만 유지한다.
- **cpp 의 PWM 게이트 상한은 60 s**: `led-pwm.service`(After=multi-user.target)가 부팅 +44 s 에 돌기 때문에, 10 s
  상한은 매번 타임아웃해 "밝기 비활성"으로 떴다(3회 부팅 재현).

## 부팅 시 네트워크 의존 제거

- **tiktoken 인코딩 사전**:
  - `/tmp/data-gym-cache` 에 캐시되는데 `/tmp` 는 부팅마다 비워진다. 그래서 매 부팅 3.6 MB 를 다운로드했다
    (8.6 s, 로컬은 0.65 s).
  - 다운로드 실패 시 `KeyError` 만 잡혀 있어 python 이 죽고, 네트워크가 올 때까지 재시작 루프를 돈다.
  - `TIKTOKEN_CACHE_DIR` 을 홈으로 옮겼다. (→ 이후 [저장소 폴더 정리](2026-08-repo-restructure.md)에서 `var/cache/` 로 이동)
- **`SentenceTransformer` 로드**:
  - 캐시가 있어도 허브 메타데이터 왕복(~3.4 s)을 한다. `local_files_only=True` 로 막았다.
  - 새 기기 첫 실행은 캐시가 없어 실패하므로, [SETUP.md](../SETUP.md)에 1회 부트스트랩을 명시했다.
- **`HF_HUB_OFFLINE=1` 은 대안이 아니다**:
  - `file_name` 을 명시해도 sentence-transformers 가 `/api/models/.../tree/main` 조회를 시도해
    `OfflineModeIsEnabled` 로 죽는다(2회 재현).
  - 환경변수 한 줄로 끝날 것 같지만 부팅 시 크래시 루프를 만든다. 생성자 인자 `local_files_only` 만 동작한다.

## 기동 시간 — 임포트 13 s 의 구성 (측정)

- **구성**:
  - CPU 5.7 s: torch 2.2 / openai 0.65 / google.cloud.speech 0.63
  - SD 콜드 I/O 3.3 s: 터치되는 페이지 `.so` 126 MB + `.pyc` 28 MB (전체 556 MB 중)
  - 경합 4.2 s: gdm 자동로그인 → GNOME Shell 기동이 +23.6~35.7 s 로 임포트 구간과 정확히 겹친다.
- **네트워크 사용은 0건**: `connect()` 를 가로채서 확인했다.
- **torch 는 제거 불가**: silero_vad(onnx 모드여도 최상위 임포트), 텐서 변환, sentence_transformers 세 갈래로 묶여 있다.
- **google/openai lazy import 는 기각**: 기동 시 클라이언트를 생성하므로 임포트만 미룰 수 없다. 생성까지 미루면 첫
  웨이크워드 응답이 1~2 s 느려진다 — 부팅 1회와 매 대화의 교환이다.

## 함정 — 부팅 시 시계를 믿을 수 없다

RTC 배터리가 없어, 부팅 후 ~30–40 s 동안 시계가 며칠 틀리다가 NTP 로 점프한다. 로그 파일명·타임스탬프가 파일 중간에서
점프하므로, 부팅 타임라인 진단은 `journalctl -o short-monotonic` 으로만 한다.
