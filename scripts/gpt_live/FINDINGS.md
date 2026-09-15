# GPT-Live 실험 기록

레이의 턴테이킹·ASR·TTS를 GPT-Live 하나로 대체할 수 있는지 보기 위한 실측 기록.
실험 도구는 `mic_live.py`(마이크 또는 TTS 스크립트 입력 → GPT-Live → 스피커, 전 이벤트 로그).
실행 로그는 `logs/`, 공식 문서 원문은 `ref/`, 재현용 스크립트는 `script_*.txt`.

기간: 2026-09-14 ~ 09-15. 환경: Raspberry Pi 5, Wi-Fi(5 GHz), reSpeaker XVF3800(마이크·스피커, 온칩 AEC), 모델 `gpt-live-1`, 백엔드 `gpt-5.4-mini`.

---

## 1. API 기본 사실 (문서 + 확인)

- 엔드포인트 `wss://api.openai.com/v1/live/sessions`. Realtime API와 별개. 클라이언트 이벤트 11종, 서버 이벤트 21종 (`ref/api-reference-live-primary-websocket.md`).
- **수동 턴 제어 이벤트가 없다.** 턴 시작/끝, 발화 시작/끝, 응답 생성/취소 같은 이벤트가 존재하지 않는다. 모델이 언제 말하고 멈출지를 전부 결정한다.
- 세션 시작 시 고정되는 것: 모델, `instructions`(≤16,384 토큰), 오디오 포맷, 목소리, 위임 모드. 이후에는 append(≤500 토큰)로 추가만 가능.
- 오디오: WebSocket은 raw PCM16 mono, 16 kHz 또는 24 kHz(입·출력 공유). 우리는 16 kHz 사용(reSpeaker 캡처와 동일).
- 세션 만료: `session.started`의 `expires_at`로 계산하면 **120분**. 문서에는 수치 없음.
- 컨텍스트 128k 토큰(instructions + 대화 텍스트 + 오디오 토큰). `session.usage.updated`(15초 간격)의 `context_window.usage_ratio`로 사용량 확인 가능. 90% 초과 시 원본 instructions + 최근 8,192 토큰 요약을 들고 엔진 교체(문서).
- **채워지는 속도(실측, 3분)**: 모델이 말하는 중 약 48 tok/s, 무음(마이크만 흐름) 약 20 tok/s. → 90% 도달까지 계속 말하면 약 40분, 보통 대화 약 50분, 거의 무음 약 95분. 2시간 세션이면 교체 2회 정도. 교체 자체는 미관찰.
- openai-python 3.13.0에 `client.live.connect()`와 전체 타입 포함. 프로젝트는 `openai<2.0`이라 도입 시 SDK 업그레이드 필요.
- 요금 $0.05/분(무음 포함, 초 단위). 동시 세션 수 제한(Tier1 25).

## 2. 오디오 출력 특성

- **실시간 페이싱.** 100 ms(3200 B) 조각이 초당 정확히 10개 온다. TTS처럼 앞서 보내지 않는다. 말하지 않을 때도 무음 조각이 같은 속도로 계속 온다(전화선과 같음).
- 따라서 클라이언트에 지터 버퍼가 필수이고, 그 크기가 곧 추가 지연이다. 한 번 밀린 지연은 서버가 따라잡아 주지 않으므로 무음 조각을 버리는 등으로 스스로 줄여야 한다.
- 조각 간격 분포(5분, 2,910개): 95%가 110 ms 이내, 200 ms 초과 31회(약 10초에 1회), 300~400 ms 17회, 최대 510 ms. 이후 여러 실행에서도 최대 500 ms 안팎. **400 ms 여유면 5분에 1회 끊김, 500 ms 넘게 잡아야 전부 흡수.**
- 지연은 모델이 말하는 중이든 무음이든 차이 없음(전송량이 같음). Wi-Fi 절전 끄기는 효과 없었음.
- 레이 C++ 재생부(`CustomSoundStream::onGetData`)는 이미 콜백 방식 + 큐 비면 무음 삽입 구조. 재생 시작은 첫 360 ms 덩이가 모여 분석되는 즉시(초반 지연 ≈ 360 ms). 정상 상태에서는 분석이 모터 제어보다 두 사이클 앞서 돌아 soundStream 큐에 약 720 ms가 앞서 있고, 이것이 지터 흡수 여유(관측 최대 정지 510 ms를 덮음). 720 ms를 넘는 정지가 나면 그때만 무음이 끼고 그만큼 지연이 남는다.
- `stream_start`가 재생 중 들어오면 수신 스레드가 즉시 버퍼를 비우고 플래그를 바꾸므로(stale 청크 방어), 발화를 겹쳐 보내면 상태가 뒤엉킨다. 기존 SessionLoop는 `playback_complete` 전에 다음 `stream_start`를 보내지 않아 문제 없음. Live 적용은 "세션 = 스트림 하나"(`generate_head_motion=false`, 무음 조각 포함 그대로 전송)로 하기로 결정(2026-09-15).
- 첫 소리까지: `session.started`까지 1.5~3.5초, 그 뒤 모델이 스스로 인사(지시 없어도).

## 3. 전사 (transcript)

- `session.input_transcript.delta` / `session.output_transcript.delta`가 200 ms 오디오 단위로 온다. 턴 완료 이벤트 없음, 조각이 글자 중간에서 끊김, 사용자·모델 조각이 인터리브. 턴 묶기는 클라이언트 몫(SDK `TranscriptGrouper` 참고).
- `start_ms/end_ms`는 세션 타임라인상 **소리가 있었던 구간**이고, 전사가 도착하는 시각은 그보다 0.6~0.8초 늦다.
- 전사가 별도 인식기인지 모델 자체 출력인지 **문서·레퍼런스 어디에도 없음.** 관측: (a) 사용자 말을 잘못 들었을 때 전사와 모델 반응이 두 번 연속 같은 방식으로 틀림("날짜가" → "날 차로 친"), (b) 출력 전사에서 같은 한국어 숫자 발화가 "1. 2. 3." / "일, 이, 삼" / "하나, 둘, 셋"으로 실행마다 다르게 표기됨. → 전사는 모델의 이해/생성 텍스트로 보는 것이 자연스럽다. **모델의 오인식을 전사로 검출할 수 없다.**
- 문서 경고: "transcripts can contain mistakes, unfinished phrases, and later corrections". 정정이 어떤 형태로 오는지는 미확인.

## 4. 위임 (delegation)

- 위임 여부는 **지시 없이도 모델이 스스로 판단**한다(기본 instructions만으로 날씨 질문을 위임). 위임 정책 지시문은 "무엇을 넘기고 무엇을 직접 답할지"의 경계를 좁히는 용도.
- 모드는 세션당 하나(`client` | `responses`). 변경은 새 세션.

### client 모드
- `session.delegation.created`에는 `id`, `target`, `offset_ms`만 있고 **요청 텍스트가 없다**(API 설계). 우리가 전사에서 요청을 재구성해야 한다.
- 결과는 `session.commentary.append(delegation_id=…)`로 돌려주면 모델이 패러프레이즈해 말한다(고정 답 "서울은 맑고 24도야" → "서울은 지금 맑고 24도야. 가볍게 나가도 괜찮을 듯!").
- 답을 안 주면 모델은 "잠깐만 기다려줘", "아직 로딩 중이야"를 반복하며 무한 대기하고, 재촉하면 위임을 다시 생성한다.
- 결과가 너무 빨리(1.8초) 오면 "확인해볼게" 직후 결과를 말해 어색해진다.

### responses 모드
- OpenAI가 백엔드 모델을 호출한다. 백엔드 설정(`delegation.responses`: model, instructions, tools, reasoning 등)은 `session.update`로 세션 중 변경 가능한 유일한 영역.
- 백엔드 스트리밍 이벤트가 `response.event` 봉투로 그대로 중계된다(`response.created`, `output_text.delta`, `web_search_call.*`, `output_item.done`, `completed`). 라이프사이클 스냅샷은 input/tools/output이 비워져 온다.
- **백엔드가 받는 입력은 대화 이력 전체**(user/assistant 메시지, 전사 텍스트 그대로, 모델의 직전 발화 포함). 시드 히스토리 5개 전부, 음성 대화 3턴 전부, 60초 침묵 뒤에도 앞 턴 참조 가능. (백엔드에 "받은 입력을 인용하라"는 지시로 확인. 단 백엔드가 이 지시를 매번 따르지는 않아 "이것만 받았다"의 증거로는 못 쓴다.)
- 함수 툴: `tools`에 function 정의 → 백엔드가 호출하면 `output_item.done(type=function_call)` → 우리가 실행 → `response.item.create(function_call_output)` → `response.create` → 백엔드 2차 응답 → 모델 발화. 배터리 조회 예제로 **위임부터 발화까지 약 2.2초.**
- 백엔드 지시문은 답 형식(한국어, 짧게)은 잘 따르지만 "web search 쓰지 마라"는 무시됨. `tool_choice: auto`가 우선하는 듯.
- 모델은 백엔드 답에서 필요한 부분만 골라 자기 말투로 말한다("요청 확인: …" 인용부는 읽지 않음).
- 위임부터 web_search 완료까지 1~2초.
- **종료 툴(`end_conversation`)**: 대화 모델은 툴을 직접 갖지 못하므로 "작별 인사 → 위임 → 백엔드가 툴 호출" 경로만 가능. 대화 모델은 위임과 동시에 자기 작별 인사를 하므로 "기다려 달라"는 대기가 끼지 않았다(11/11). 위임 발생률은 **프롬프트 형식에 크게 좌우**: 규칙을 덧붙여 쓴 프롬프트 3/5, 그걸 "MUST"로 강화 0/3, 공식 템플릿 구조(Backend tools / Delegate when / Do not delegate when)로 정리한 프롬프트 **6/6**. 툴 결과 뒤 `response.create`로 이어가면 백엔드가 작별 인사를 또 만들고 대화 모델이 그걸 한 번 더 말할 수 있어(1/3), 종료 툴일 때는 `response.create`를 생략(인사 1회). 단 위임을 미완으로 두면 `session.close` 뒤 `session.closed`가 **약 9.4초** 뒤에 온다(3/3 일관, 드레인 대기로 추정). 과금(`usage.seconds`)은 close 요청 시점에서 멈추고 오류 이벤트도 없으므로, 레이는 `session.closed`를 기다리지 않고 C++ `playback_complete`만 기다려 SLEEP으로 간다. 이어가는 변형은 close 0.75초지만 인사가 두 번 나옴(2/4).
- **종료 시퀀스 실측**: 툴 호출 → `session.input_audio.mute`(ack 0.24초, 세션 정지 없음) → 출력 0 조각 1초 지속으로 인사 끝 판정(툴 호출 후 4~5초) → `session.close`. 상한 8초. 템플릿 프롬프트에서도 작별 위임 누락 1회 있음(누적 7/9) → 전사 키워드 매칭 폴백 유지.

## 5. 세션 중 텍스트 주입 (append)

세기(1~40) 도중 주입해 발화가 유지되는지 확인. 대조군(주입 없음)은 끝까지 셈.

| 주입 | 표본 | 발화 유지 |
|---|---|---|
| `instructions.append` (영어 전환) | 3 | 0/3 — 즉시 침묵 |
| `instructions.append` (중립: "한국어로 계속") | 1 | 0/1 — 즉시 침묵 |
| `instructions.append` ("끊지 말고 계속" 명시) | 2 | 1/2 — 한 번은 이어짐, 한 번은 세기를 버리고 다른 주제로 |
| `instructions.append` ("계속하되 영어로") | 1 | 0/1 — "I understand—switching to English now." 후 중단 |
| `thinking.append` (사실 주입) | 3 | 2/3 |
| `thinking.append` ("끊지 말고 계속") | 1 | 1/1 |
| `thinking.append` (영어 전환) | 4 | 4/4 — 세기를 이어가며 언어만 전환 |

- **instructions.append는 내용과 무관하게 진행 중 발화를 끊는다.** 지시 자체는 반영된다(이후 영어로 답). 모델이 말하지 않는 틈에 넣어야 한다. 말하는 중인지 알려주는 이벤트가 없으므로 출력 오디오 레벨/전사 유입으로 판단해야 한다.
- **thinking.append는 발화를 유지하며 반영된다.** 언어 전환 같은 발화 방식 변화도 문장 중간에서 자연스럽게 바뀜. 단 강도가 약해, 영어 전환 thinking 후 한국어 질문에는 한국어로 답함(instructions는 규칙, thinking은 참고 정보).
- **주입 사실을 소리 내어 말할 수 있다.** "I'm switching to English now." 등. "조용히 적용하고 언급하지 마라"를 명시해도 2회 중 1회 그대로 말함. 문구로 막을 수 없으니 새어 나가도 되는 표현으로 넣어야 한다.
- ack(`*.appended`)는 0.6~0.7초 뒤에 오고 "주입 수락"만 뜻한다.
- instructions.append는 역으로 **모델의 말을 멈추게 하는 유일한 수단**이다(문서의 가드레일 예시가 이 원리).

## 6. 대화 동작 관찰

- Full-duplex: 사용자 말이 끝나기 전에 "안녕!", "야, 재헌아!" 하고 겹쳐 말함. 사용자가 말을 시작하면 세기를 스스로 멈춤(클라이언트 신호 없음).
- **방 안의 다른 대화에 끼어든다.** "말 걸 때만 답하라"는 지시를 넣어도 주변 대화에 반응. 침묵 강제는 지시로 안 되며 마이크 mute나 세션 종료로 해야 한다. → SLEEP 상태를 GPT-Live 세션으로 대신하기 어렵다.
- 한국어 인식·발화는 대체로 자연스럽다. 이름 "재헌"을 "재혼"으로 발음한 사례 있음. 오인식 사례("날짜가" → "날 차로 친") 있음.
- 시계·날짜 지식이 없고 스스로도 그렇게 말한다. 날짜·뉴스는 백엔드 web_search로 정확히 답함.
- 세션 중 XVF3800 AEC는 정상 동작(스피커 소리에 모델이 반응하지 않음).

## 7. 레이 적용 시 남는 문제 (요약)

- 턴테이킹 모듈(VAP/TurnGPT/추측 생성) 전부 불필요 → SessionLoop 재작성 수준.
- 바지인: 모델이 스스로 멈춤. 클라이언트는 재생만 끊을 수 있고, 재생을 끊어도 모델은 자기가 전부 말한 것으로 안다.
- "끝없는 스트림" vs C++ `stream_start/audio_end` 발화 단위 프로토콜.
- 프롬프트: 매 턴 재조립 불가. 프로필/이월은 시작 시 `input`(≤8,192 토큰)·instructions, 검색 기억은 thinking.append(≤500 토큰).
- 기억 쓰기: 전사 기반 유지 가능하나 턴 경계를 우리가 만들어야 함. 오인식이 그대로 저장됨.
- 웨이크워드/SLEEP은 로컬 유지(비용·끼어들기).
- 세션 종료(결정 2026-09-15): 기본은 사용자 전사 키워드 매칭 + 유휴 타임아웃, `end_conversation` 툴은 빠른 경로로 병행. 종료 판정 후 모델 발화가 끝나기를 기다려 `session.close`. 작별 WAV는 생략(모델이 인사함), 인사 WAV는 연결 지연 마스킹용으로 유지.
- 평가·트레이싱: ASR 단계 격리 불가, LLM/TTS 단계 지표 소멸.
- 미실험: 긴 세션의 엔진 교체 순간의 동작(40~50분 세션 필요), 전사 "later corrections" 형태, 인사말 유도(`instructions.append`)와 사전 WAV 병행, 한국어 이름 발음 제어(프롬프팅 가이드의 IPA 표기).

## 8. 실험 도구 사용법

```
uv run python scripts/gpt_live/mic_live.py [옵션]   # 프로젝트 venv (openai 3.x) 로 실행
  --seconds N            최대 실행 시간 (스크립트 모드는 스크립트 끝나면 조기 종료)
  --no-play              스피커 재생 안 함
  --prebuffer-ms N       재생 프리버퍼 (실제 효과는 장치 버퍼 ~45 ms에 한정, §2 참고)
  --save out.wav         받은 오디오 저장
  --delegation none|client-demo|responses
  --backend-model M      responses 백엔드 모델 (기본 gpt-5.4-mini)
  --probe-backend-input  백엔드에 "받은 입력을 인용하라" 지시
  --seed "role:text"     시작 히스토리 (반복 가능)
  --script FILE          마이크 대신 TTS 스크립트 입력 (say/wait/instructions/thinking/commentary 줄)
  --instructions "..."   대화 모델 지시문 덮어쓰기
```

로그는 실행마다 `logs/<시각>.log`. ALSA 경고는 `2>&1 | grep -v "^ALSA lib\|^Cannot connect\|^jack server\|^JackShm"`로 걸러서 본다.
