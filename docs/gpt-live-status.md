# GPT-Live 엔진 적용 — 진행 상황 핸드오프 (2026-09-16)

브랜치 `feat/gpt-live`. 커밋 2개(SDK 3.x 업그레이드, 조사 도구·기록) 뒤의 작업은 **모두 미커밋 상태**로 워킹 트리에 있다.
실측 기록은 `scripts/gpt_live/FINDINGS.md`, 설계 판단은 `docs/decisions-wip.md`의 "GPT-Live 엔진" 절.

## 1. 목표와 현재 위치

레이의 ASR + 턴테이킹(VAP/TurnGPT) + LLM + TTS 체인을 OpenAI GPT-Live(`gpt-live-1`, full-duplex 음성 모델) 하나로
대체하는 `live` 엔진을 만든다. cascade 엔진과 `settings.ENGINE`으로 공존.

- 1단계(SDK 업그레이드, 브랜치) **완료·커밋**.
- 2단계(엔진 구현) **코드 완료, 실기기 시험 중**. 대화는 된다. 남은 문제는 모션 끊김과 소리–입 싱크(§4).
- 3단계(장기기억 읽기, 프로필 시드, 레이 함수 툴 확장) 미착수.

## 2. 구현된 것 (미커밋)

| 파일 | 내용 |
|---|---|
| `voice_pipeline/adapters/gpt_live.py` | SDK `client.live.connect()` 래퍼. 수신 스레드 → 프로젝트 이벤트(`LiveAudio`, `LiveTranscript`, `LiveFunctionCall`, …) 큐. 전사는 SDK `TranscriptGrouper`로 화자별 세그먼트로 묶어 전달. |
| `voice_pipeline/live_session.py` | `LiveSessionLoop`. 마이크 16k→24k 리샘플 → 세션, 출력 오디오 → 브리지(세션 = 스트림 하나, `live=True`), 전사 → 히스토리·utterances, 종료(키워드·유휴 60 s·`end_conversation` 툴·closed·브리지 오류·기아·stop), 종료 시퀀스(mute → 출력 무음 1 s → close → `audio_end`). **파이썬 무음 채우기/버리기**(§4.2). 대화 모델·백엔드 지시문, `LIVE_VOICE="cedar"`, 인사 문구. |
| `voice_pipeline/wiring.py` | `engine` 분기. live면 ASR·VAP·TurnGPT 미로드. responses 위임 + 함수 툴(`end_conversation`, `search_memory`, `adjust_volume`, `set_brightness`, `get_device_settings`). |
| `voice_pipeline/device_settings.py` | 볼륨(10단계, wpctl)·LED 밝기(off/low/medium/high) 툴 정의·핸들러 + 현재 상태 조회. `var/device_settings.json` 에 저장, 시작 시 재적용. |
| `voice_pipeline/__main__.py` | live면 작별 WAV 생략, 인사 WAV를 `OpenAITTS(voice=cedar, model=gpt-4o-mini-tts)`로 "네, 부르셨어요?" 생성. 콘솔에 live 로거 표시. 종료 시 `vap/asr` None 가드. |
| `voice_pipeline/settings.py` | `ENGINE = "live"`, `BRIDGE_SAMPLE_RATE = 24000`. |
| `voice_pipeline/adapters/cpp_bridge.py` | `send_stream_start(live=True)` → `{"type":"stream_start","live":true}`. |
| `voice_pipeline/adapters/wakeword.py` | 주 언어 ko-KR + 대안 en-US, 키워드 `("ray","레이")`, 한글은 부분 문자열 매칭. |
| `voice_pipeline/adapters/tts_openai.py` | `voice`, `model` 생성자 인자. |
| `voice_pipeline/greeting_audio.py` | `greeting_text/farewell_text` 인자. |
| `cpp/main.cpp` | `stream_live` 플래그: live면 헤드모션 온라인 생성 대신 대기 모션, 시작 전 `kLivePrebufferCycles=2`(720 ms) 모음. **싱크 진단 로그** `audio_sync_<mode>.csv`(모터 틱마다 모션이 가정하는 오디오 위치·실제 재생 위치·재생기가 끼운 무음 누적). 빌드 통과. |
| 테스트 | `tests/test_live_session.py`, `tests/adapters/test_gpt_live.py` 등 추가. 전체 996 통과. |
| 진단 도구 | `scripts/hardware/wakeword_diag.py`(마이크 레벨·VAD·STT 결과), `scripts/hardware/audio_sync_report.py`(싱크 로그 요약), `scripts/gpt_live/mic_live.py`(GPT-Live 단독 실험, TTS 스크립트 입력 가능). |
| 문서 | `docs/decisions-wip.md`(설계 판단), `docs/modules/bridge.md`(live 필드), `docs/modules/wakeword.md`. |

## 3. 확정된 사실 (실측)

- GPT-Live 출력은 **실시간 속도로만** 온다(100 ms 조각 초당 10개, 무음도 0 조각으로 계속). 앞서 보내지 않는다. **관측치**이며 문서 보증은 없다 — 공식 문서는 조각 크기·속도·버퍼링에 대해 아무 말도 하지 않는다("no timing fields, no output-audio-done event"). 코드는 `len(pcm)`으로 길이를 계산하므로 조각 크기가 바뀌어도 동작한다.
- 총량은 실시간보다 **짧다** — 2026-09-16 `store: true` 세션의 서버 녹음(`GET /v1/live/sessions/{id}/content`, 오른쪽 채널 = 출력)과 받은 오디오를 상호상관으로 대조해 **확정**. 받은 오디오는 서버가 생성한 것과 샘플 단위로 같고(오프셋 −2 ms 고정, 상관 1.00), 빠진 구간만 녹음에 남아 있다. 지연이 아니라 유실이고 늦게라도 오지 않는다.
  - 유실은 **무음 구간에 집중**: 유선·전부 무음 세션 97.3%(단일 프레임 누락 약 36회, 도착 간격 200 ms로 드러남), 유선·90% 말소리 세션 99.8%(167초 지점에서 400 ms 한 번, 도착 간격엔 흔적 없음 — 말소리 유실은 클라이언트가 감지·복구할 수 없고 문장 중간이면 들린다).
  - Wi-Fi 세션(9/14~15, 7회)은 96~98.5%. 유선 대비 추가 손실은 정지와 상관.
  - 사이드밴드 WS의 반사 오디오 타임스탬프로 확인하는 방법은 **불가** — 서버 제어 가이드에 사이드밴드는 WebRTC·SIP 세션 전용("If your backend already owns the primary WebSocket connection, it already receives the session's events"), attach 시 404. 같은 가이드에 "Reflected output ranges can have gaps for dropped frames"로 프레임 드롭이 명시돼 있다.
- **모델은 재촉 없으면 사용자가 말할 때까지 침묵한다** (run1: 180 s 무음, 전사 0). `commentary.append` 재촉 시 0.9 s 만에 인사(run2). 프로덕션 코드에는 재촉이 없어 현재 흐름은 WAV 인사 → 연결 → 사용자 발화 대기. 세션 시작 직후 0.2 s의 미세 잡음(진폭 ≤50)이 있어 말소리 판정은 rms ≥ 30 기준(2026-09-16 수정 전에는 "Model started speaking"이 이 잡음에 찍혔다).
- 호출어 감지 → 마이크 전송 시작까지 **4.6~6.2 s**였다: 인사 WAV 2.07 s + GPT-Live 연결 2.0~4.1 s가 직렬. → **겹치기 구현(2026-09-16, 실기기 검증 전)**: `__main__`이 WAV 재생을 보낸 뒤 live 엔진은 기다리지 않고 바로 세션을 만들고, `LiveSessionLoop(wait_for_playback_complete=True)`가 연결 즉시 마이크를 보내며 WAV의 `playback_complete`를 받은 뒤 `stream_start`(그 전 출력 조각은 버림, 10 s 폴백). 예상 2.0~4.1 s.
- **LED 매칭**: 바(IDLE)는 "마이크가 세션으로 흐르는 구간"과 일치시킨다 — 켜기·끄기는 세션 루프 소유. live: 연결 직후 IDLE, 종료 시퀀스(입력 mute) 진입 시 SLEEPING. cascade: `asr.start()` 뒤 IDLE, 세션 루프 종료 시 SLEEPING(작별 WAV는 링 호흡 상태). `__main__`의 GREETING IDLE 점등은 제거(WAV·연결 중 4~6 s 먼저 켜지던 문제).
- 네트워크 정지(조각 간격 > 250 ms)는 **Wi-Fi에서만** 관측(약 10초에 1회, 300~500 ms 흔함, 1.1 s도). 유선 3세션(각 3분)은 최대 214~363 ms, 정지 뒤 몰림 없음. 정지 뒤 몰려와도 총량은 완전히 회복되지 않는다(정지 창 도착률 90~94%). wlan0 링크는 −61 dBm, 19.5 Mbit/s로 낮은 속도에 머물러 있었다.
- C++ 재생부는 큐가 비면 `onGetData`가 100 ms 무음을 스스로 낸다(데이터 소비 없이). 모션부(`control_motor`)는 절대 시계라 데이터가 없으면 기다렸다 몰아친다.
- 위임·종료·전사·주입 동작은 FINDINGS.md 참고. 웨이크워드는 정상(끊김의 원인 아님). GNOME 소리 설정을 열어 두면 PipeWire가 마이크를 점유해 `No input device matching 'respeaker'`가 난다.

## 4. 현재 문제와 지금까지의 조치

### 4.1 모션 끊김 — 원인 A: 시계 시작 시 여유 0 → **해결(C++ 프리버퍼)**
C++는 첫 360 ms가 차면 시계를 시작하고 매 사이클 "그 시각의 덩이"를 요구. 실시간 유입에서는 덩이가 매번 막 도착 중이라 매 사이클 멈춤(모터 틱 4 ms 몰아치기, motion 로그 16:30). → `live` 스트림은 두 덩이(720 ms)가 찰 때까지 시계 시작 안 함. 시작 직후 끊김 사라짐.

### 4.2 모션 끊김 — 원인 B: 총량 부족으로 여유가 소진 → **해결(파이썬 무음 채우기)**
2~3% 부족이 720 ms 여유를 ~20 s에 소진, 이후 세션 끝까지 매 사이클 멈춤(무음 구간에서도 회복 안 됨). → `LiveSessionLoop._on_audio`:
- lead = 보낸 오디오 길이 − 첫 조각 이후 경과 시간.
- 무음(정확히 0) 조각 뒤: lead < −0.1 이면 0까지 0 조각을 채움 / lead > +0.3 이면 0 조각을 버림.
- 어절 사이 쉼(rms < 30) 뒤: lead < −0.3 이면 0 조각 **하나만** 채움(60 s 넘는 연속 발화 대비).
- 말소리 조각은 절대 버리거나 미루지 않음. 소리·모션이 같은 스트림을 쓰므로 싱크 무영향.
결과: 107 s / 68 s / 181 s 세션에서 lead가 ±0.15 안, 멈춤은 네트워크 정지 시점 1~2회로 감소.

### 4.3 소리–입 싱크 어긋남 — **미해결, 원인 확정**
`audio_sync` 로그(17:56 세션, 181 s): 어긋남이 0 → **306 ms**(20~30 s, 재생기 무음 300 ms) → **711 ms**(80~90 s, +400 ms) 계단식으로 늘고 되돌아오지 않음. 두 시점은 조각 간격 390 ms·1,110 ms의 네트워크 정지와 일치.

메커니즘: 정지가 실효 여유를 넘으면 재생기가 무음을 끼워 **소리만** 뒤로 밀리고, 모션은 절대 시계로 따라잡아 제자리로 돌아옴 → 차이가 영구 잔류. TTS 시절엔 발화마다 스트림을 새로 시작해 초기화됐지만 live는 스트림이 하나라 누적.
실효 여유는 720 ms가 아니라 ~360 ms: `stream_and_split`이 360 ms 단위로만 재생 큐에 옮기므로 재생 큐엔 한 덩이만 있고 나머지는 원시 버퍼에 남는다. 390 ms 정지로도 언더런.

### 4.4 소리–입 싱크 방어 — **1차 구현 (2026-09-16, 실기기 검증 전)**
증상(모션 시계)이 아니라 **두 소비자가 공유하는 지점** `stream_and_split`에서 부족을 처리한다.
1. **구현**: 사이클 기한 + 유예 `kSplitGraceMs`(450 ms)까지 360 ms가 차길 기다리고, 그래도 모자라면 있는 만큼 + 0으로 채워 **소리 큐와 분석 큐에 같은 덩이를** 넣는다. 소리·입이 같은 자리에서 같이 멈추고 같이 이어진다. 모션 몰아치기도 없어진다. 시계 시작 전(cycle < 0)과 스트림 종료 뒤 꼬리는 채우지 않는다. 채움마다 `[split] cycle N: padded X ms` 로그, 스트림 끝에 합계.
   - 유예를 두는 이유: 기한에 바로 채우면 지터(10~100 ms)마다 말소리에 구멍이 난다. 소비자는 두 사이클(720 ms) 뒤에 이 덩이를 쓰므로 재생기 선취·분석 시간을 빼면 ~500 ms까지 늦어도 굶지 않는다.
   - 프리버퍼 2덩이는 시작 즉시 큐로 넘어가 정상 상태에서도 덩이가 기한보다 ~360 ms 늦게 완성된다. 여유 없이 시작하면 G=450은 "평소보다 90 ms 더 늦으면 채움"이 되어 지터마다 조금씩 채운다 — **1차 실기기 시험(13:05, Wi-Fi 166 s)에서 실측**: 20~60 ms × 9회 = 240 ms를 23~33 s 구간에 채운 뒤 여유가 쌓여 나머지 130 s는 0회, 재생기 무음 0 ms, lag 최대 14 ms, 틱 간격 최대 44 ms(끊김 0회). 싱크는 완벽했으나 말소리 중간의 미세 구멍과 +240 ms 지연이 남았다.
   - 재생기 자체 무음(`onGetData`)은 폴백으로 남기고, 불리면 `[Sound] underrun` 로그. 정상이면 0회.
2. **시작 여유 + 되돌리기 (2차, 같은 날)**: `kLiveStartSlackMs` — live 프리버퍼를 2덩이 + S로 잡아 S를 원시 버퍼에 남긴 채 시작(TTS 스트림은 무영향). **S=200으로 실기기 검증(13:41, Wi-Fi 368 s)**: 540 ms 정지 1회에 채움 540 ms → 소리·입 같이 멈춤, lag 0~11 ms, 재생기 무음 0, 틱 끊김 0회, 이후 무음에서 320 ms 되돌림(나머지는 유실분이라 되돌릴 잉여 없음, 버퍼는 S 수준으로 복귀). 330~360 ms 간격 4회는 채움 없이 흡수. 82 s의 60 ms 채움 1회는 파이썬 채우기 밴드(lead −0.1~0)가 C++ 여유를 100 ms 깎은 상태에서 240 ms 지터가 걸린 것.
   - **S 선정(실측 기준)**: 채우기 없이 흡수되는 간격 ≈ S + 190(문답) / S − 10(긴 연속 발화, 파이썬 말 중 채우기 문턱 −0.3 때문). Wi-Fi 25.8분 분포: >300 ms 1분 1회, >400 2.9분, >450 5.2분, >500 8.6분, 최대 1,110. → **S=400 채택**(흡수 490, 채움 5~9분 1회, 지연 720+400=1,120 ms). 유선(최대 214~363)은 100~200으로 충분. 긴 발화가 잦으면 파이썬 밴드 축소(채우기 목표 0→+0.1, 말 중 문턱 −0.3→−0.15)가 지연 비용 없이 S+200 효과.
   - 오프셋 = 360·P + S 가 live의 정상 상태 지연이다(시작 오프셋이 세션 내내 유지). 720 중 360은 채워지는 중인 덩이(불가피), ~240은 재생기 선취(OpenAL 3버퍼×80 ms, 줄이면 G 상한 480→600으로 올라가 지연 없이 여유 +110~150), 90은 G가 사용. 선취 축소는 미착수. `kSplitTrimFloorMs`(200) — 채운 누계가 남아 있고 원시 버퍼가 200 ms보다 많이 남았으면 앞쪽 0 샘플을 "남은 양 − 200"과 채운 누계 중 작은 만큼만 버려 응답 지연을 원위치. 실제 남은 양만 버리니 진동 없음, 말소리 샘플 무영향, 파이썬 회계도 무영향(채움·되돌림 상쇄). 로그 `[split] cycle N: trimmed X ms`. 부작용: 문장 사이 자연 쉼이 채운 만큼(수백 ms) 짧아질 수 있음.
   - 기각한 대안: 파이썬 타이머 채우기(C++가 흡수할 수 있는 정지에도 구멍을 냄 — 채우기는 기한을 아는 C++가 마지막 순간에), C++→파이썬 `padded` 보고(회계를 한곳에 모으려는 시도였지만 C++ 안에서 상쇄되므로 불필요).
3. 파이썬 무음 채우기 유지(서버 생략분은 파이썬이 조용한 자리에, 정지는 C++가 최후 방어).
검증(Wi-Fi 3분 세션, 콘솔은 `build/Ray 2>&1 | tee var/log/ray_console.log`): `[Sound] underrun` 0회, `[split] padded`는 정지(≈300 ms 이상) 시점에만, 그 뒤 무음에서 `trimmed`로 원위치, `audio_sync_RESPONSES.csv`의 lag 0 근처, motion 로그 40 ms 간격 유지.

### 4.4a 유선 전환 뒤 판단 (2026-09-16)
- 4.2 무음 채우기는 유선에서도 필요 — 무음 유실 1~3%면 720 ms 여유가 27~72초에 소진된다. 유실이 무음에 몰리므로 "무음 조각 뒤에서만 채운다"가 맞는 자리. 유선에서는 100 ms 단위의 작고 잦은 보정, Wi-Fi에서는 정지 뒤 몰아 채우기 + 버리기가 번갈아 동작(17:56 세션 3분에 5초).
- 정지 중에는 조각이 안 오므로 채우기 트리거가 없다 → 정지에 의한 끊김·소리-입 어긋남(4.3)은 채우기로 못 막고 C++(4.4)의 몫. 유선이면 정지가 없어 4.4는 보류 가능. 로봇을 Wi-Fi로 운용할지에 따라 결정.

### 4.5 기타 관찰
- 시작음이 늦게 느껴지는 것: 웨이크워드 감지 자체에 VAD 후행 300 ms + Google STT 왕복(0.5~1 s)이 있어 원래 구조. 로그상 감지 → 인사 재생은 즉시.
- 웨이크워드 인식 불량 보고는 GNOME 소리 설정의 마이크 점유가 원인이었음(§3).
- 정지가 말할 때/무음일 때 어느 쪽에 몰리는지는 미확인(단독 측정에선 무관). `_on_audio`에서 300 ms 넘는 간격을 상태와 함께 INFO로 찍으면 다음 실행에서 갈림 — 아직 안 넣음.

## 5. 실행·확인 방법

```
# 터미널 1 (모터·재생). 기기별 모터 값은 config/robot.toml
build/Ray
# 터미널 2
uv run ray
```
"레이"/"Ray"로 깨움 → 인사 "네, 부르셨어요?"(cedar) → Live 세션. 종료는 작별 인사(`end_conversation` 툴, 모델 재량) 또는 키워드("잘 가", "이제 갈게", "여기까지", "bye", …) 또는 유휴 60 s. GNOME 소리 설정 창은 닫아 둘 것.

로그:
- `var/log/pipeline/<시각>.log`: `GPT-Live connected in …`, `LiveSessionLoop started`, `Model started speaking`, `user:`/`assistant:` 전사, 조각 간격 300 ms 이상이면 `Audio chunk gap …ms (speaking|silent, 단계)`, `Ending session (…)`, 종료 시 `Audio summary: sent, padded, dropped, max chunk gap`. 15 s 주기 `Audio lead …` 줄은 DEBUG(`voice_pipeline.live_session=DEBUG`로 켬).
- `var/log/motion/<시각>/console.log`: C++ 콘솔(cout/cerr) 복제. `[split] padded/trimmed`(시각·버퍼 잔량 포함), `[Sound] underrun`, `[MainLoop]`, `[CALIB]`.
- `var/log/motion/<시각>/Standard_Log.csv`: 모터 틱(정상 40 ms 간격, 대기 모드는 약 1 Hz로 저속 기록). `audio_sync_RESPONSES.csv`: 싱크 진단(`kEnableAudioSyncLog`).
- 요약: `uv run python scripts/hardware/audio_sync_report.py var/log/motion/<시각>/audio_sync_RESPONSES.csv`

단위 테스트: `uv run --all-groups python -m pytest -q` (`uv run pytest`가 안 되면 `.venv/bin/pytest` 셔뱅이 옛 경로일 수 있음 → `uv sync --all-groups --reinstall-package pytest`).

## 6. 남은 항목

- [x] §4.4 2차 실기기 검증(S=200, Wi-Fi 368 s) → S=400 채택. 유선 프리셋(100~200) 전환 방법은 미정
- [ ] 호출어 → 청취 시작 겹치기 + LED 매칭 실기기 검증: 감지 → `LiveSessionLoop started` 2.0~4.1 s, 바 LED 가 연결 시점에 켜지고 종료 시퀀스에서 꺼지는지, WAV 끝 → `stream_start` 전환이 매끄러운지
- [ ] 지연 없이 여유 늘리기: `onGetData` 선취 80→40 ms + G 450→560 실험 (`[Sound] underrun` 로그로 확인)
- [ ] 파이썬 채우기 밴드 축소(목표 +0.1, 말 중 문턱 −0.15) — 긴 발화 사용이 있으면
- [ ] 종료 시 `audio_end`를 "출력 무음 1 s" 판정 직후 보내기(지금은 close 뒤라 C++가 끝에 1.3 s 채움, SLEEP 전환 지연)
- [ ] 프리버퍼 한 덩이로 줄이는 실험(지연 −360 ms) — 유선이면 우선
- [ ] 검증 스크립트(store 녹음 대조 `verify_live_audio.py` / `analyze_recording.py`, 현재 세션 스크래치에만 있음) `scripts/gpt_live/`로 이관 여부. `store: true`는 OpenAI에 녹음이 30일 남으므로 실험 전용 표시 필요.
- [ ] 2단계 커밋(위 표의 파일 일괄) — 커밋 메시지 초안: `feat(live): GPT-Live 엔진 추가 — 세션 루프, 어댑터, live 스트림 프리버퍼·헤드모션 대기, 무음 채우기, 한·영 웨이크워드, 한국어 인사`
- [ ] 3단계: 기억 읽기(`thinking.append`), 프로필/최근 세션 시드(`input`), 레이 함수 툴(배터리·Matter 등), 언어 전환(`instructions.append`는 모델이 조용할 때만)
- [ ] 긴 세션 컨텍스트 교체(~50 min) 동작 관찰, 전사 "later corrections" 형태 확인
- [ ] `docs/decisions-wip.md` 4.2~4.4 내용 반영(현재는 프리버퍼·위임·종료까지만 기록)
