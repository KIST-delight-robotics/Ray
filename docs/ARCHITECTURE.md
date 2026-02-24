# 음성 대화 로봇 — Python 파이프라인 아키텍처 설계 v2

임시 - 언제든 바뀔 수 있음

## 1. 시스템 모드

```
SLEEP ──(wakeword)──▶ GREETING ──(재생 완료)──▶ ACTIVE ──(종료 키워드/타임아웃)──▶ FAREWELL ──(재생 완료)──▶ SLEEP
```

| 상태 | 동작 | 차단 |
|------|------|------|
| **SLEEP** | WakewordDetector만 동작 | 대화 파이프라인 전체 |
| **GREETING** | C++에 인사 재생 신호 전송, 재생 완료 대기 | wakeword, barge-in, ASR, 턴 판정 |
| **ACTIVE** | 전체 대화 파이프라인 동작 | wakeword |
| **FAREWELL** | C++에 종료 인사 재생 신호 전송, 재생 완료 대기 | wakeword, barge-in, ASR, 턴 판정 |

- GREETING/FAREWELL: 녹음된 음성을 C++에서 재생. Python은 오디오 스트리밍 없이 신호만 전송.
- 종료 키워드 체크: 턴 확정 후, LLM 요청 전에 수행.
- 타임아웃: 마지막 ASR 갱신 이후 N초 경과 기준.


## 2. 모듈 목록 및 입출력

### 2.1 AudioInput

| 항목 | 내용 |
|------|------|
| 역할 | 마이크에서 오디오 스트림을 캡처하여 큐에 넣음 |
| 실행 | 별도 스레드에서 항상 동작 (모드 무관) |
| 소유 | SessionManager |
| 출력 | audio_queue (오디오 프레임 큐) |
| 소비자 | SLEEP: SessionManager → WakewordDetector, ACTIVE: Orchestrator → ASR + TurnDetector |
| 모드 전환 시 | 큐 비우기 (stale 프레임 제거) 후 새 소비자 루프 시작 |


### 2.2 WakewordDetector

| 항목 | 내용 |
|------|------|
| 역할 | Sleep 모드에서 wakeword 감지 |
| 입력 | 오디오 프레임 |
| 출력 | wakeword 감지 이벤트 |
| 비고 | Sleep 모드에서만 활성 |


### 2.3 ASR

| 항목 | 내용 |
|------|------|
| 역할 | 실시간 스트리밍 음성→텍스트 변환 |
| 입력 | 오디오 프레임 (Orchestrator가 매 프레임 전달) |
| 출력 | 현재까지의 텍스트 (Orchestrator가 폴링으로 조회) |
| 내부 | 스트리밍 API 세션 관리, partial/final result 처리 |
| 생명주기 | Orchestrator가 ACTIVE 진입 시 시작, 종료 시 정리 |
| 인터페이스 | 벤더 교체 가능하도록 추상화 |


### 2.4 VAP (TurnDetector 내부)

| 항목 | 내용 |
|------|------|
| 역할 | 턴 테이킹 판단을 위한 음성 활동 예측 |
| 입력 | 사용자 오디오 (AudioInput), 로봇 오디오 (TTS 오디오 + 재생 타이밍 동기화) |
| 출력 | `p_now`, `p_fut`, `user_is_speaking` |
| 실행 | 오디오 프레임 단위로 주기적 (TurnDetector 내부에서 호출) |
| 비고 | 로봇 오디오는 Python이 보유한 TTS 오디오를 C++ 재생 타이밍에 맞춰 제공 |


### 2.5 TurnGPT (TurnDetector 내부)

| 항목 | 내용 |
|------|------|
| 역할 | 텍스트 기반 턴 종료 확률 예측 |
| 입력 | 대화 기록 텍스트 |
| 출력 | 확률값 (0~1) |
| 실행 | ASR 텍스트 변경 감지 시 (TurnDetector 내부에서 호출) |


### 2.6 TurnDetector

| 항목 | 내용 |
|------|------|
| 역할 | 순수 턴 판정기. 오디오와 ASR 텍스트를 받아 판정 결과만 반환. 외부 모듈을 직접 호출하지 않음. |
| 입력 | 오디오 프레임 (매 프레임), 현재 ASR 텍스트 (매 프레임), 로봇 오디오 (재생 중일 때, Orchestrator가 재생 위치 기반으로 제공) |
| 출력 | TurnDecision (아래 참조) |
| 내부 소유 | VAP 인스턴스, TurnGPT 인스턴스, 모든 타이밍 상태, ASR 변경 감지 (유사도 비교 포함), threshold/timeout 설정 |
| 외부 의존 | 없음. SpeechGenerator, ASR 등의 존재를 모름. |

**TurnDecision 출력:**

| 신호 | 의미 | Orchestrator 동작 |
|------|------|-------------------|
| **turn_shift** | 사용자 턴 종료 감지. 로봇이 발화권을 가져도 되는 시점. | 응답 재생 시작 (또는 준비 안 됐으면 생성 후 시작) |
| **interrupt** | 사용자 끼어들기 감지. 로봇이 발화를 멈춰야 하는 시점. | 로봇 발화 중단 |
| **prepare** | 응답 사전 준비 신호. 내부에서 유사도 비교까지 완료한 확정 신호. | 기존 준비 취소 + 새 응답 준비 시작 |


### 2.7 LLM

| 항목 | 내용 |
|------|------|
| 역할 | 대화 응답 생성 |
| 입력 | 조립된 컨텍스트 (메시지 목록) |
| 출력 | 스트리밍 텍스트 청크 |
| 관리 범위 | 프롬프트 템플릿, tool 정의/실행, 모델 파라미터, API 호출 |
| 인터페이스 | 벤더 교체 가능하도록 추상화 |


### 2.8 TTS

| 항목 | 내용 |
|------|------|
| 역할 | 텍스트→오디오 변환 |
| 입력 | 텍스트 |
| 출력 | 오디오 데이터 + (선택적) 단어별 타임스탬프 |
| 인터페이스 | 벤더 교체 가능. 타임스탬프 지원 여부는 구현체마다 다름. |


### 2.9 UtteranceTruncator

| 항목 | 내용 |
|------|------|
| 역할 | barge-in 시 재생된 부분까지의 텍스트를 산출 |
| 입력 | 원본 텍스트, 재생 중단 시점, (선택적) 단어별 타임스탬프 |
| 출력 | 잘린 텍스트 |
| 전략 | **TimestampTruncator**: 단어별 타임스탬프 기반 정밀 자르기 |
|       | **DurationRatioTruncator**: 타임스탬프 없을 때 재생 비율로 추정 |
| 비고 | 전략 인터페이스로 분리. TTS 구현체에 의존하지 않음. |


### 2.10 ContextBuilder

| 항목 | 내용 |
|------|------|
| 역할 | LLM 호출 전에 컨텍스트를 조립 |
| 입력 소스 | ConversationHistory (과거 대화), 현재 ASR 텍스트 (파라미터), 시스템 프롬프트, tool 정의, (향후) RAG 결과, 장기기억 |
| 출력 | LLM에 전달할 메시지 목록 |
| 확장 | 새로운 컨텍스트 소스가 추가되면 여기만 확장 |


### 2.11 ConversationHistory

| 항목 | 내용 |
|------|------|
| 역할 | 대화 기록 저장/조회. 순수 데이터 저장소. |
| 단위 | **세션 단위** 관리. 세션 생성/종료 시 자동 초기화/저장. |
| 입력 | 메시지 (`list[dict]`). dict 스키마는 LLM 벤더에 의존하며, LLM 구현 시 확정. |
| 출력 | 메시지 목록 (전체 또는 최근 N턴) |
| 확장 포인트 | **StorageBackend**: 영속화 방식 (메모리 / 파일 / DB 등) |
| 비고 | assistant 메시지 저장 시점: 재생 완료(전체 텍스트) 또는 barge-in 중단(UtteranceTruncator 결과) |


### 2.12 SpeechGenerator

| 항목 | 내용 |
|------|------|
| 역할 | ContextBuilder + LLM + TTS를 연결하여 응답 오디오 생성. speculative prepare 관리. |
| 상태 | idle → preparing → ready → idle |
| 호출 흐름 | 1. ContextBuilder로 컨텍스트 조립 |
|           | 2. LLM으로 응답 텍스트 생성 (스트리밍) |
|           | 3. TTS로 오디오 변환 |
| 출력 | ResponseData (응답 전체 텍스트, 오디오 데이터, 단어별 타임스탬프) |
| 핵심 동작 | **사전 준비**: 턴 확정 전에 LLM+TTS를 미리 실행 → ready 상태로 전환 |
|           | **준비 취소**: ASR 업데이트로 컨텍스트 변경 시 이전 준비를 취소 → idle |
|           | **결과 인출**: ready 상태에서 ResponseData를 꺼내감 → idle |
| 외부 의존 | ContextBuilder, LLM, TTS. |


### 2.13 CppBridge

| 항목 | 내용 |
|------|------|
| 역할 | Python ↔ C++ 통신 |
| 통신 방식 | **WebSocket** (기존 방식 유지) |
| Python → C++ | TTS 오디오 전송, 제어 명령 (barge-in 중단, 재생 시작 등), 인사/종료 인사 재생 신호 |
| C++ → Python | 재생 상태 이벤트 (재생 시작, 재생 중 위치, 재생 완료, 중단 완료 + 중단 시점) |
| 메시지 구분 | 메시지 타입 태그로 오디오/명령/이벤트 구분 |
| 비고 | 인터페이스는 추상화 유지 (향후 ZeroMQ 등으로 교체 가능) |


### 2.14 LEDController

| 항목 | 내용 |
|------|------|
| 역할 | WS2812 LED 색상/애니메이션 제어 |
| 입력 | LED 명령 (색상 변경, 애니메이션 패턴 등) |
| 트리거 | 미정 (턴 관련 타이밍 등) |
| 인터페이스 | 추상화하여 구현체 교체 가능 |
| 구현체 (안) | **DirectLEDController**: Python에서 직접 하드웨어 제어 (현재) |
|              | **BridgeLEDController**: CppBridge 경유 C++ 제어 (향후 가능하면) |


### 2.15 SessionManager

| 항목 | 내용 |
|------|------|
| 역할 | 최상위 상태 머신. 모드 전환 + 세션 수명 관리. |
| 소유 | AudioInput (스레드 + 큐), WakewordDetector, Orchestrator, ConversationHistory |
| 참조 | CppBridge (인사/종료 인사 신호) |
| SLEEP | audio_queue에서 프레임 → WakewordDetector. wakeword 감지 시 GREETING 전환. |
| GREETING | CppBridge에 인사 재생 신호 → 재생 완료 대기 → ACTIVE 전환. |
| ACTIVE | audio_queue 비우기 → session_id 발급, ConversationHistory 초기화 → Orchestrator.run(audio_queue) 호출. return 시 FAREWELL 전환. |
| FAREWELL | CppBridge에 종료 인사 재생 신호 → 재생 완료 대기 → ConversationHistory 저장 → SLEEP 전환. |
| 비고 | GREETING/FAREWELL 시 C++에 오디오 스트리밍 없이 신호만 전송. 큐 소비 안 함. |


### 2.16 Orchestrator

| 항목 | 내용 |
|------|------|
| 역할 | ACTIVE 모드의 대화 루프. 프레임 구동. TurnDecision에 따라 모듈 간 실행 흐름 제어. |
| 입력 | audio_queue (SessionManager로부터 전달받음) |
| 내부 상태 | 현재 ASR 텍스트, 재생 상태 (idle / playing / stop_pending), 현재 ResponseData |
| 매 프레임 | 1. audio_queue에서 프레임 꺼냄 → 2. ASR에 오디오 전달 → 3. ASR에서 현재 텍스트 조회 → 4. TurnDetector에 프레임 + 텍스트 + 로봇 오디오 전달 → TurnDecision → 5. CppBridge 이벤트 확인 |
| prepare 시 | SpeechGenerator: 기존 준비 취소 + 새 준비 시작 (현재 ASR 텍스트 전달, 백그라운드 실행) |
| turn_shift 시 | 종료 키워드 체크 → 종료면 return |
|               | → ConversationHistory에 user 메시지 저장 (현재 ASR 텍스트) |
|               | → SpeechGenerator에서 ResponseData 인출 (ready → 즉시 / preparing → 완료 대기 / idle → 생성부터) |
|               | → CppBridge로 오디오 전송 → 재생 상태를 playing으로 전환 |
|               | → ASR 리셋 |
| interrupt 시 | CppBridge에 중단 명령 전송 → 재생 상태를 stop_pending으로 전환 |
| CppBridge 이벤트 | **재생 완료**: ResponseData의 전체 텍스트 → ConversationHistory에 assistant 저장 → 재생 상태 idle |
| (매 프레임 확인) | **중단 완료** (stop_pending 중): C++가 보낸 재생 위치 + ResponseData → UtteranceTruncator → 잘린 텍스트를 ConversationHistory에 assistant 저장 → 재생 상태 idle |
|                  | **재생 위치**: VAP 로봇 오디오 동기화에 사용 |
| 종료 조건 | 종료 키워드 감지 또는 ASR 갱신 타임아웃 → return |
| ACTIVE 진입 시 | ASR 시작, audio_queue → ASR + TurnDetector 라우팅 |
| ACTIVE 종료 시 | ASR 중지, 리소스 정리 |
| 외부 의존 | ASR, TurnDetector, SpeechGenerator, CppBridge, UtteranceTruncator, ConversationHistory |
| 실행 모델 | 프레임 구동 동기 루프. I/O 작업 (LLM, TTS 등)은 각 모듈 내부에서 백그라운드 처리. |


## 3. 전체 호출 구조

```
SessionManager (최상위 상태 머신)
│
├─ SLEEP:  audio_queue → WakewordDetector
│
└─ ACTIVE: Orchestrator
             ├── audio_queue → ASR (오디오 전달 / 텍스트 조회)
             ├── audio_queue + ASR 텍스트 → TurnDetector → TurnDecision
             ├── SpeechGenerator (응답 생성만)
             │     └── ContextBuilder → LLM → TTS → ResponseData
             ├── CppBridge (오디오 전송 + 재생 이벤트 수신)
             ├── UtteranceTruncator (barge-in 시 텍스트 자르기)
             └── ConversationHistory
```


## 4. VAP 로봇 오디오 동기화

```
Python 보유: TTS 오디오 전체 + 단어별 타임스탬프(있을 경우)
C++ 전달:   재생 상태 이벤트 (시작, 현재 위치, 완료)

동기화 방식:
  1. Orchestrator가 ResponseData의 오디오를 CppBridge로 전송
  2. CppBridge가 재생 시작/위치 이벤트를 전달
  3. Orchestrator가 재생 위치를 TurnDetector에 전달
  4. VAP 모듈은 ResponseData의 TTS 오디오를 재생 위치에 맞춰 소비

확장 가능성:
  - 나중에 OS 캡처나 C++ 오디오 수신이 필요하면
    VAP의 로봇 오디오 입력 인터페이스만 교체
```


## 5. 디렉토리 구조

```
voice_pipeline/
├── core/
│   ├── interfaces.py              # 모든 모듈의 인터페이스 정의
│   ├── events.py                  # 이벤트 타입 + Event Bus
│   └── config.py                  # 설정/하이퍼파라미터
│
├── audio/
│   ├── audio_input.py             # 마이크 캡처, 오디오 분배
│   └── wakeword.py                # Wakeword 감지
│
├── asr/
│   └── asr.py                     # ASR 인터페이스 + 구현체
│
├── turn_taking/
│   ├── vap.py                     # VAP 래퍼
│   ├── turngpt.py                 # TurnGPT 래퍼
│   └── turn_detector.py              # 종합 턴 판정
│
├── llm/
│   ├── llm.py                     # LLM 인터페이스 + 구현체
│   ├── prompts.py                 # 프롬프트 템플릿 관리
│   └── tools.py                   # Tool 정의 및 실행
│
├── tts/
│   ├── tts.py                     # TTS 인터페이스 + 구현체
│   └── utterance_truncator.py     # 발화 텍스트 자르기 전략
│
├── context/
│   └── context_builder.py         # LLM 컨텍스트 조립
│                                  #   - History에서 최근 턴
│                                  #   - 시스템 프롬프트
│                                  #   - Tool 정의
│                                  #   - (향후) RAG, 장기기억
│
├── history/
│   ├── conversation_history.py    # 세션 단위 대화 기록 저장/조회
│   └── storage_backend.py         # 영속화 (메모리 / 파일 / DB)
│
├── generation/
│   └── speech_generator.py        # LLM+TTS 오케스트레이션
│                                  # speculative prepare 관리
│
├── bridge/
│   └── cpp_bridge.py              # C++ 통신
│
├── led/
│   └── led_controller.py          # LED 인터페이스 + 구현체 (Direct / Bridge)
│
├── orchestrator/
│   └── orchestrator.py            # ACTIVE 모드 대화 루프
│
├── session/
│   └── session_manager.py         # 최상위 상태 머신, 세션 수명 관리
│
└── tests/
    ├── test_asr.py
    ├── test_vap.py
    ├── test_turngpt.py
    ├── test_turn_detector.py
    ├── test_llm.py
    ├── test_tts.py
    ├── test_utterance_truncator.py
    ├── test_context_builder.py
    ├── test_speech_generator.py
    ├── test_conversation_history.py
    ├── test_cpp_bridge.py
    ├── test_led_controller.py
    ├── test_orchestrator.py
    ├── test_session_manager.py
    └── test_pipeline_integration.py
```


## 6. 테스트 전략

| 테스트 대상 | 방법 |
|-------------|------|
| ASR / LLM / TTS | 인터페이스 Mock으로 단위 테스트 |
| VAP / TurnGPT | 녹음된 오디오/텍스트로 단위 테스트 |
| TurnDetector | 오디오 샘플 + ASR 텍스트 시퀀스 입력 → TurnDecision 검증 (내부 VAP/TurnGPT는 Mock 주입) |
| ContextBuilder | History/RAG/Memory Mock 주입, 조립 결과 검증 |
| SpeechGenerator | LLM/TTS Mock 주입, 상태 전이 검증 |
| UtteranceTruncator | 타임스탬프 있는/없는 케이스 각각 |
| ConversationHistory | 세션 생성/저장/복원 검증 |
| LEDController | 인터페이스 Mock으로 명령 발행 검증 |
| Orchestrator | TurnDecision별 시나리오 검증, 재생 상태 전이 검증, 프레임 루프 동작 검증 |
| SessionManager | 모드 전환 + 세션 수명 주기 검증 |
| 통합 테스트 | 전체 파이프라인을 Mock으로 조립하여 시나리오 테스트 |


## 7. 미결정 사항

- [x] ~~메인 루프 오케스트레이터 구조~~ → SessionManager(최상위 상태 머신) → Orchestrator(ACTIVE 전용, 프레임 구동 루프)
- [x] ~~메인 루프 실행 방식~~ → 프레임 구동 동기 루프 + 모듈 내부 백그라운드 처리
- [x] ~~오디오 분배 방식~~ → AudioInput 별도 스레드 → 큐 → 소비자 루프에서 꺼내 씀 (SessionManager 소유)
- [x] ~~ASR 텍스트 전달 방식~~ → 폴링 (Orchestrator가 매 프레임 조회). ConversationHistory에는 턴 확정 시에만 저장.
- [ ] VAP 로봇 오디오 소스 (TTS 오디오 + 타이밍 동기화 vs OS 캡처 vs C++ 전달)
- [ ] 대화 기록 StorageBackend 선정
- [ ] Wakeword 엔진 선정
- [ ] ASR / LLM / TTS 벤더 확정
- [ ] RAG / 장기기억 설계 (향후)
- [ ] LED 동작 정의 (어떤 타이밍에 어떤 색상/애니메이션)
- [ ] LED 제어 위치 확정 (Python 직접 vs C++ 경유)
