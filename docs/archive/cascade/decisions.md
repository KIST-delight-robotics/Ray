# cascade 엔진 (ASR → 턴 감지 → LLM → TTS)

> 사용하지 않는 cascade 엔진의 기록. 고정 — 이후 변경은 반영하지 않는다.

## 개요

GPT-Live 이전의 대화 엔진. Google STT 로 받아 적고, VAP(음향)·TurnGPT(텍스트) 두 모델로 발화 끝을 판정하고, OpenAI
LLM 응답을 문장 단위로 TTS 에 흘려 C++ 로 재생했다. 2026-02 부터 만들었고, 2026-09 GPT-Live 엔진이 들어온 뒤로는
`settings.ENGINE` 으로 고를 수 있는 대안으로 남았다.

- 설계 초안: [ARCHITECTURE.md](ARCHITECTURE.md)
- RPi 5 모델 벤치마크 원문: [benchmarks/](benchmarks/)
- 두 엔진이 같이 쓰던 결정(C++ 프로토콜, 세션 수명, Silero VAD 상태 오염, 대화 기록, 임베더):
  [history/initial-build.md](../../history/initial-build.md)
- 장기기억 1차(블록 주입·retained buffer 포함): [history/memory-v1.md](../../history/memory-v1.md)
- 평가 시스템: [archive/eval/decisions.md](../eval/decisions.md)


## TTS (ElevenLabs)

- **기본 vendor = ElevenLabs**: OpenAI tts-1은 TTFB 0.9~8.2s로 편차가 크고 짧은 문장에서 realtime factor 0.54×(재생보다
  느림)까지 떨어짐. ElevenLabs flash_v2_5는 Pi 실측 TTFB 중앙값 220~305ms — vendor 전환 자체가 큰 지연 개선
  (`scripts/bench/bench_tts.py`).
- **ElevenLabs `stream/with-timestamps` endpoint**: base64+JSON 스트림이라 raw PCM 대비 네트워크 ~33% 오버헤드가
  있지만, character alignment → word timestamp 집계로 `truncate_by_timestamps` 정밀 barge-in 절단이 처음으로
  활성화됨 (OpenAI는 timestamp 미지원이라 항상 ratio 추정 fallback이었음). Pi 실측 오버헤드는 plain stream 대비 TTFB
  +15~35ms 수준이라 수용. 문제가 되면 `_iter_chunks`만 교체해 기본 stream endpoint로 전환 가능한 구조.
- **`pcm_24000` 고정**: OpenAITTS와 샘플레이트를 맞춰 VAP 참조 채널·C++ 재생·greeting WAV 등 downstream 전부 무변경.
  tier 제한도 없음 (`pcm_44100`은 Pro 전용).
- **vendor 선택 = `create_tts()` 팩토리 + 모듈 내 파라미터 기본값**: 팩토리는 추상화 수단이 아니라(추상화는 `ITTS`가
  담당) "이름→클래스" 매핑의 단일 위치 — 스크립트 `--vendor` 인자 같은 런타임 선택용. 기본 vendor는 파라미터 기본값으로
  (`create_embedder(backend="local")` 선례). env var는 설정 표면 증가, mutable 모듈 전역은 wiring 결정 분산이라 기각.
  (팩토리 위치는 이후 구조 정리로 `wiring.py`)
- **ElevenLabs SDK gotcha — lazy generator**: `stream_with_timestamps()`는 generator function이라 HTTP 요청·에러가 첫
  `next()`에서 발생 (OpenAI의 eager CM enter와 다름). 인증/voice 에러도 `synthesize()`가 아닌 iteration 중 TTSError로
  표면화. SDK 기본 timeout 240s는 실시간 대화에 부적합해 override 필수 (기록 당시 10s).
- **timestamps 집계는 절대 raise 금지**: sentence 모드 consumer에서 `stream.timestamps` 읽기 예외는 턴 전체를
  실패시킴 (consumer_error 경로). alignment 길이 불일치는 절단+warning, 시간 역전/음수는 clamp. timestamps는
  best-effort 부가 기능 — 실패해도 오디오 재생은 정상이어야 함.
- **단어 집계 = 공백 run 분리, 스트림 종료 시 1회**: `truncate_by_timestamps`가 `text.split()` 토큰을 전제하므로
  character alignment를 `ch.isspace()` run 기준으로 묶음. 단어가 chunk 경계에 걸칠 수 있어 chunk별이 아니라 스트림
  종료 시점 1회 집계 (alignment 시간은 오디오 시작 기준 절대값).
- **ElevenLabs Free tier voice 제약**: Rachel 등 구형 premade voice는 라이브러리로 이관돼 API 호출 시 402
  `paid_plan_required`. 현행 default voice(Sarah/George/Daniel/Jessica 등)는 사용 가능. scoped API 키는 `voices_read`
  권한이 없어 voice 목록 조회도 불가 — voice 교체 시 TTS 실호출로 검증해야 함.


## TurnGPT

- **KV cache 활용**: ASR이 incremental update(같은 prefix, 늘어나는 suffix)를 보내므로 매번 전체 dialog를 재처리하면
  낭비. 토큰 prefix 비교 후 새 토큰만 forward. 동일 입력이면 캐시된 확률 반환.
- **Context window = 최근 턴 유지**: 완료된 최근 턴 `_KEEP_TURNS=2`개 + 진행 중 턴만 유지(`<ts>` 경계 기준).
  `_MAX_CONTEXT_TOKENS=1024`(GPT-2 한계)는 그 뒤 hard truncation 안전망. Eviction은 KV cache를 무효화하지 않음 —
  forward의 prefix 매칭이 재사용/재계산을 알아서 결정.
- **ONNX backend**: RPi 추론 성능용. ONNX 경로는 torch 미사용 — 토큰화도 numpy(`return_tensors="np"`), torch import는
  PyTorch 백엔드 전용. KV cache 지원은 ONNX input name에 `past_key_0` 존재 여부로 감지.
- **ONNX threads = 2**: RPi 5 (4코어) 벤치마크 1–4스레드. 2스레드가 fp32/int8 모두 최적 (int8 42ms, fp32 111ms).
  4스레드는 contention으로 오히려 느림.
- **int8 양자화**: 4x 작고 2.6x 빠르며 TRP 차이 무시 가능 (~0.04). 배포 권장.

상세 수치: [benchmarks/turngpt_benchmark_v2.md](benchmarks/turngpt_benchmark_v2.md)


## MaAI VAP

- **INT8 양자화 부적합**: CPC 인코더(Conv1D+LSTM, 2.5M params)에 `quantize_dynamic` 적용 시 1.9x 느려지고 정확도
  대폭 하락. TurnGPT(MatMul 위주, 163M params)와 달리 작은 Conv 커널에서는 양자화 오버헤드가 연산 절감을 초과하고,
  모델이 이미 L2 캐시에 들어가 메모리 대역폭 이점 없음.
- **배치 처리 무의미**: No-cache 배치(N frames)는 KV cache + 1 frame보다 느림. KV cache가 이미 반복 연산을 제거하므로
  텐서 크기를 키워도 추가 이점 없음.
- **ONNX transformer export**: Transformer를 ONNX export하여 전체 파이프라인(encoder+transformer)을 ORT로 실행.
  - PyTorch dispatch overhead 완전 제거. Mean 24ms (vs PyTorch 106ms, 3.9x). `torch.compile` warmup 100프레임 문제도
    해소.
  - 변환 시 dict KV cache → 12개 flat stacked tensor. Cross-attention source 순서 주의 (원본 입력을 src로 전달,
    업데이트된 값 아님). 수치 차이 max 6.8e-6.
- **ORT 싱글스레드 최적**: Transformer ONNX 추가 후에도 `ort_threads=1`이 최적. `ort_threads=4`는 동기화 비용으로 2x
  느려짐 (48ms vs 24ms).
- **PyTorch `p_now` 리스트 반환 버그**: `VapGPT.forward()`가 `p_now`을 `[speaker1, speaker2]` 리스트로 반환.
  `float(out["p_now"])`은 항상 실패하지만 integration test 없어 미발견. `p_now[0]` 추출로 수정.
- **torch.compile 한계**: 작은 텐서([1,1,256]) 연산에서 커널 launch + 메모리 할당 오버헤드가 남아 이론치(0.4ms) 대비
  32ms. 10ms 이하를 원하면 ONNX export 필요 — 그래서 ONNX가 기본.

상세 수치: [benchmarks/vap_rpi_benchmark.md](benchmarks/vap_rpi_benchmark.md)


## Threaded Model Wrappers (VAP, TurnGPT)

- **별도 스레드 필요 이유**: RPi 5 worst case VAP 24ms + TurnGPT 30ms = 54ms, 30ms frame budget 초과. ONNX Runtime이
  GIL 해제하므로 별도 스레드에서 진정한 병렬성 확보. 원본 VoiceActivityProjection 래퍼는 RPi에서 비현실적이라 제거하고
  MaAI로 단일화.
- **VAP 스레드는 프로세스 수명, TurnGPT 래퍼는 세션 수명**: ThreadedVAP는 모델 로드+워밍업이 비싸 세션마다 재생성하지
  않고 `session_id` 리바인드 + `reset()`으로 재사용, 프로세스 종료 시 `stop()`. ThreadedTurnGPT는 세션마다 새로 생성.
- **1-frame TurnGPT delay 허용**: `process_frame()`에서 poll 후 submit. Frame N의 submit 결과는 Frame N+1에서 poll.
  30ms 지연은 500ms+ 단위의 turn-taking 판단에 무시 가능.


## TurnDetector

- **Paper-based two-path algorithm** (Skantze & Irfan, 2025): Path 1 (VAP sustained robot-favor) OR Path 2 (TurnGPT
  graduated silence timeout). 단일 모델 의존 회피 — VAP가 빠른 응답(~500ms), TurnGPT가 VAP 실패 시 eventual
  turn-taking 보장.
- **Backchannel vs interrupt 구분**: ROBOT_TURN에서 robot_audio 있을 때 `p_now > threshold`만으로는 interrupt 판단
  불충분 — `p_fut`도 user를 favor해야 함. 백채널("응", "네")은 p_now만 spike하고 p_fut는 robot-favoring 유지.
- **`robot_audio=None` 시 unconditional interrupt 제거**: robot audio 없이 `user_is_speaking=True`만으로 interrupt 하던
  것을 제거. 원인: 1–3s generation gap 동안 주변 소음이 false interrupt 유발 (실측: 7회 LLM 호출, 4.7s 지연). VAP는
  robot 채널 없이 interrupt/backchannel 구분 불가 → `TurnDecision.none()` 반환. (이후 Phase 상태기계 도입으로 generation
  gap은 ROBOT_TURN이 아닌 PENDING 구간이 되었고, 이 구간의 사용자 재발화는 아래 cancel 경로가 처리.)
- **`user_is_speaking` 전제조건**: 논문 pseudocode (Appendix A lines 56-61) 요구. 없으면 turn-shift 직후 VAP context가
  아직 user-biased일 때 transient spike로 false interrupt 발생.
- **VAP error default `VAPResult(0, 0, False)`**: "robot favored"처럼 보이지만, transient error는 500ms 지속 안 되므로
  Path 1 false trigger 없음. Persistent failure 시 Path 2(TurnGPT + silence)가 독립 작동.


## 턴 종료(cancel/interrupt) — Phase 상태기계

- **cancel/interrupt 경계 = begin_streaming**: "레이 음성이 잠깐이라도 났으면 무조건 interrupt(cancel 아님)"라는 사용자
  관점 기준을 만족시키려면 재생이 *가능해지는* 마지막 Python 지점을 경계로 잡아야 함.
  - `playback_started`는 C++가 `play()` 호출 시점(가청 시작보다 SFML/ALSA 버퍼만큼 앞)에 보내고 Python은 폴링 지연 후
    처리 → 가청 시작과의 선후가 수십 ms 내에서 모호.
  - `begin_streaming`(=`send_stream_start`)은 그 이전엔 C++에 재생 명령이 없어 *물리적 무음 보장*이 되는 유일 지점.
  - 그래서 cancel은 항상 begin_streaming 이전(브리지 무접촉)이라 `send_stop` 불필요·STOPPING 미경유, interrupt는 항상
    이후 — STOPPING이 항상 interrupt 의미가 되어 reason 태그 불필요.
- **STREAMING은 종료-감지 공백 구간(단일채널 interrupt 안 함)**: `robot_audio`(VAP 참조 채널)는 `playback_started`가
  클럭을 세팅해야 생김. 그 전 STREAMING에선 VAP가 interrupt vs backchannel을 구분할 robot 채널이 없음.
  - 단일채널 fallback은 (a) backchannel 오인터럽트, (b) stop_pos≈0 빈 기록 문제로 기각, interrupt 감지를 PLAYING으로
    미룸.
  - 비용: STREAMING 내에서 시작·종료하는 짧은 barge-in은 그 턴엔 무시됨(begin_streaming에서 ASR이 reset되고 이후
    누적되므로 다음 턴 입력으로 이연 — 유실 아님). bridge_ms 실측 중앙값 ~97ms라 공백은 보통 짧고, 긴 STREAMING(느린
    TTS)이 최악. 근본 대응은 C++ prebuffer·TTS throughput.
- **cancel 신호 = user_is_speaking 전제 + p_now/p_fut(즉시), grace는 유사도**: cancel은 interrupt와 동일 구조 —
  `user_is_speaking`을 전제로 하고(실제로 말해야 함) 그 위에서 `p_now/p_fut`로 플로어 회수를 확인. (→ 이후 ASR 유사도
  경로에서는 `user_is_speaking` 전제를 뺐다 — 아래 "PENDING cancel")
  - user_is_speaking *단독*은 backchannel·노이즈에 약하고, p *단독*은 무음 중 확률 변동에 오발화하므로 둘 다 필요.
  - VAP가 네이티브 10Hz(각 결과가 이미 ~100ms 적분)라 100ms 미만 sustain은 같은 캐시 추론 재독이라 무의미 → 즉시 발화.
  - ASR finalization noise는 시간 grace 대신 **마지막 prepare 텍스트(=응답이 생성된 기준)** 와의 유사도로 거름 —
    prepare-skip 게이트와 **같은 비교의 양면**이라 별도 기준선 추가 없이 `_last_prepare_text` 재사용.
  - 이 user_is_speaking 전제가 "침묵 timeout으로 shift했는데 p만 높은" 모순 입력에서의 turn_shift↔cancel thrash도 막음.
- **detector 상태 wipe를 turn_shift→commit으로 지연 (PENDING 도입)**: turn_shift는 로봇이 실제 커밋(begin_streaming)하기
  전까진 잠정적. 기존엔 turn_shift 직후 per-frame 상태를 전부 지워 "cancel=같은 턴 연속"이 불가능했음. PENDING은 상태를
  보존해 cancel 시 매끄럽게 rewind하고, commit에서 비로소 wipe+dialog append. 부수 효과로 detector가 interrupt
  모드(ROBOT_TURN)에 진입하는 시점이 robot_audio가 생기는 시점과 일치 → "ROBOT_TURN인데 robot_audio 없음" 사각 소멸.
- **stale 응답 방지 = turn_shift의 prepare 선점 (detector 내부)**: turn_shift 조건이 충족돼도, 마지막 prepare 이후 ASR이
  *유효하게(비유사)* 바뀐 게 남아 있으면 turn_shift 대신 **prepare를 먼저** 내보내 새 텍스트로 재생성하고 다음 프레임에
  shift.
  - "늦은 finalization으로 준비된 응답이 stale" 케이스를 detector의 기존 유사도 게이트로 그대로 처리 — SessionLoop에
    임베더/similarity 주입 불필요(검사 중복 회피).
  - `at_turn_shift`일 때만 발화 스로틀(turngpt/0.2s)을 우회하고 `_asr_has_changed` 게이트는 유지 — 텍스트가 안정된
    흔한 경우엔 turn_shift가 바로 fire(speculation 이득 유지).
- **Python↔C++ 순수 전송은 무시 가능(~0.04ms 편도, Pi 루프백 IXWebSocket 실측)**: turn-taking 타이밍에서 전송은 0으로
  취급. bridge_ms(~57–97ms)는 통신이 아니라 C++ prebuffer + Python 프레임 폴링(~30ms)이 지배.

### PENDING cancel — ASR 유사도 경로의 user_is_speaking 전제 제거 (2026-07)

- **실측 계기**: "goodbye" 발화를 ASR이 'you'로 중간 인식 → TurnGPT가 그 시점에 turn_shift → 326ms 뒤 최종 텍스트
  'goodbye' 도착. 커밋까지 1.6초 이상 남아 취소 창이 열려 있었지만, `_process_pending` 진입부의
  `user_is_speaking=False` 조기 반환이 유사도 검사 실행 자체를 차단 → 'you'에 대한 응답 재생 + exit 키워드('goodbye')
  미감지.
- **두 cancel 경로는 신호의 시제가 다름 — 가드를 경로별로 분리**:
  - VAP user-favor는 "지금 발화권을 잡고 있다"는 현재형 신호라 `user_is_speaking` 전제가 정합적 (VAD 침묵 + VAP
    예측만으로 취소는 과격) → 가드 유지.
  - ASR 텍스트 변경은 이미 일어난 발화가 늦게 처리된 과거형 증거 → 전제 제거. turn_shift 자체가 VAD 침묵을 조건으로
    발화되므로, 수백 ms 늦는 ASR final은 이 게이트에 구조적으로 항상 걸렸다 (증거 도착 시각과 사건 시각의 혼동).
- **유예시간 없이 PENDING 전체에서 평가**: 자연 경계는 commit(begin_streaming). VAD 침묵 중 ASR 텍스트가 새로 생기는
  경우는 사실상 늦은 final뿐이고, 주변 발화로 인한 오취소는 기존에도 VAD-speaking 경로로 가능했던 리스크라 새로
  생기는 게 아님. 오취소 비용도 되감기 후 turn_shift 재발화(~1초 지연)로 낮음. 소음 환경에서 오취소가 관찰되면 그때
  유예창 도입 검토.


## Silero VAD — 세션 내부 오염

LSTM 상태가 세션 간 오염되는 실측과 세션 시작·SLEEP 진입 리셋은
[history/initial-build.md](../../history/initial-build.md#silero-vad) 에 있다.

- **세션 내부 오염은 commit() 시점 리셋으로**: 멀티턴 세션에서 시작 1회 리셋만으론 턴2부터 감지율 5~12배
  붕괴(67.9%→5.7%) — 오염원은 로봇 응답(마이크에 거의 안 잡힘)이 아니라 **사용자 자신의 직전 발화**.
  - 리셋 시점은 begin_streaming(=TurnDetector.commit, `vad_reset_fn` 주입) 채택: STREAMING 구간은 인터럽트 판정에
    robot_audio가 필요해 VAD 결과가 어디에도 안 쓰이는 유일한 공백이고, 갭에서 말을 시작해도 리셋 *후* onset이라 즉시
    감지 → PLAYING 진입 시 user_is_speaking 준비됨.
  - playback_started 리셋(갭 발화를 한가운데서 절단)·playback_complete 리셋(빠른 재발화와 겹침)은 기각.


## 응답 생성 (SpeechGenerator)

- **Streaming API**: `poll_audio() → bytes | None` + `stream_done`으로 TTS 청크를 즉시 CppBridge에 전달. 전체 합성 완료
  대기 불필요.
- **`max_workers=2`**: 1이면 새 `prepare()`가 cancel된 run이 blocking API call(1–3s)에서 빠져나올 때까지 대기. 2면 즉시
  시작. Turn detector 시그널 빈도상 3+ 동시 취소는 비현실적.
- **영어 기준 문장 경계 감지**: `.` `!` `?` + 뒤따르는 공백으로 판단. 약어(`Mr.`, `Dr.`, `etc.`)와 단일 대문자
  이니셜(`U.S.`)은 frozenset 기반 휴리스틱으로 제외. NLP 토크나이저 대비 의존성 없고 지연 없음. 한국어 등 다른 언어
  지원 시 별도 전략 필요.
- **`min_flush_words = 4`**: 문장 경계가 감지되어도 누적 단어 수가 임계값 미만이면 다음 문장과 합쳐서 TTS.
  `"Sure!"` 같은 짧은 감탄사가 단독 TTS 호출되면 HTTP 오버헤드가 실제 합성 시간을 초과. 대부분의 LLM 첫 문장은
  5~15단어이므로 즉시 발송됨.
- **TTS 파이프라이닝 (producer-consumer)**: LLM 스트리밍 + 문장 감지(producer)와 TTS 결과 drain(consumer)을 별도
  스레드로 분리. 문장 N의 TTS 진행 중 문장 N+1의 TTS를 동시 시작하여 문장 간 gap 최소화. 로컬
  `ThreadPoolExecutor(max_workers=2)`를 파이프라인 실행마다 생성/소멸하여 기존 `self._executor`(prepare 재시작용)와 독립.
- **인용 태그 처리**: `[MEMORIES: M1, M3]`는 문장 종결 부호가 없으므로 SentenceDetector 버퍼에 자연스럽게 잔류. LLM
  스트림 종료 시 `flush()` → `parse_citation_tag()` 순으로 처리. 중간 문장에 태그가 섞일 가능성 없음.
- **`_text` 누적 갱신**: consumer가 각 문장의 TTS 합성 완료 직후(해당 오디오 큐 적재 직전) `self._text`를 갱신.
  `get_text()`가 TTS가 끝난 문장 범위만 반환하므로 barge-in truncation 정확도가 full 모드(전체 텍스트 즉시 설정) 대비
  개선됨.
- **Sentence streaming에도 SessionLoop/CppBridge 변경 없음**: 기존 `send_stream_start → send_audio(chunk)* →
  send_audio_end` 프로토콜이 chunk 출처와 무관하게 동작. C++는 버퍼에 쌓이는 PCM을 360ms 단위로 drain하므로 TTS 호출
  횟수를 알 필요 없음.
- **Sentence streaming barge-in 호환성**: PLAYING 중 interrupt 시 generator를 cancel하지 않는 기존 동작 그대로 유지.
  consumer가 계속 실행되어 `stream_done = True` 설정 → deferred truncation 패턴이 동일하게 적용. 다음 `prepare()` 호출
  시 `cancel_event`가 문장 간 체크에서 잡혀 불필요한 TTS는 최대 진행 중 1회로 제한.


## SessionLoop

- **만성 batch-drained는 루프 비용이 아니라 시스템 부하 신호**: "Batch-drained 2 frames"가 ~170ms 간격으로 1110회
  발생한 런 분석.
  - 발생 간격이 케이던스를 알려줌 — 루프가 프레임 예산 30ms를 δ만큼 초과하면 30/δ 프레임마다 한 번 2프레임 drain,
    170ms 간격 = δ≈5ms.
  - 단계별 누적 타이머 실측 결과 루프 고유 비용은 2~3ms/it(Silero VAD가 ~75%, ASR gRPC feed는 0.03ms/it로 무혐의)에
    불과했고, 동일 코드·오디오의 다음 런에서는 17회로 소멸 → 원인은 머신 전체 부하(외부 CPU 경쟁·발열 스로틀링).
  - 판별 기준: 프레임 루프와 무관한 백그라운드 스레드인 **VAP cycle overrun이 동반 증가**하면 시스템 부하, 특정
    스테이지 스파이크면 코드 문제.
  - 가끔의 5~10프레임 drain은 LLM HTTP 응답 처리+VAP 슬로우가 겹치는 일시 스파이크로 `_MAX_BATCH_FRAMES=10` 설계 범위
    내.
  - 계측 코드는 진단 후 제거 — 재발 시 단계별 마킹 + 세션 종료 debug 요약을 재삽입하면 한 줄로 구분 가능.


## 임베딩 유사도 (TurnDetector)

- **Sentence embedding > SequenceMatcher**: SequenceMatcher는 문자 겹침 측정 — `"what is your"` vs `"what is your
  name"`이 0.87 (threshold에 차단됨). Sentence embedding (all-MiniLM-L6-v2)은 0.66 (정상 통과). 논문이 semantic
  similarity 사용.
- **ISimilarity 모듈 제거 → IEmbedder 직접 주입**: similarity 래퍼 모듈을 없애고 TurnDetector가 `IEmbedder.embed_batch`로
  코사인 유사도를 직접 계산 (`_text_similarity`). 임베더는 process-level 단일 인스턴스를 memory와 공유.


## 컨텍스트 조립과 토큰 예산 (ContextBuilder)

- **`token_count` 이중 소스**: assistant 메시지는 API `output_tokens` (정확값), user/truncated 메시지는 tiktoken
  fallback. ContextBuilder는 저장된 값을 읽어 re-tokenization 없이 예산 계산.
- **Tool definition token cost 실측**: tiktoken으로 definition structure를 추정하면 부정확. API `input_tokens`를 tool
  유무로 비교하여 실측 (`web_search` = 294 tokens). ContextBuilder가 예산에서 차감. (→ 이후 아래 "히스토리 롤링 요약"에서
  전역 예산과 함께 `tools_token_cost` 회계 제거)
- **Turn-level atomic budgeting**: ContextBuilder가 `get_turns()`로 turn_id 기준 그룹 단위로 포함/제외. Tool call +
  result + assistant text는 분리 불가 — 한 turn이 예산 초과하면 통째로 제외.

### 세션 내 히스토리 롤링 요약 + 고정 블록 예산 (2026-07)

- **전역 4096 예산 폐기 근거는 실측**: ray.db 55세션 실측 결과 대화 자체는 극소(교환당 평균 ~21토큰, 역대 최장 세션
  전체가 507토큰)인 반면, 실제 input_tokens(평균 4.4k)의 지배 요인은 web_search 호스티드 툴이 서버측에서 주입하는 히든
  프롬프트(~4.2k, 예산으로 제어 불가·대부분 캐시됨)였다. 전역 예산은 통제할 수 없는 것을 통제하는 척했고, remainder
  방식은 고정 블록이 커지면 히스토리가 조용히 전멸하는 구조라 블록별 독립 예산으로 전환. `tools_token_cost` 회계도 같은
  이유로 제거.
- **히스토리 예산 8192는 "요약이 거의 안 발동"하도록 넉넉하게**: 테스트 대화가 짧아 실측치는 하한 — 발화가 실측의
  5배여도 트리거(0.75)까지 ~40교환. 요약은 정상 경로가 아니라 아주 긴 세션의 안전장치. 히든 프롬프트가 이미 4.2k인
  환경이라 캐시되는 히스토리를 키우는 비용은 미미.
- **요약은 컨텍스트 뷰 레벨 대체 — storage 원본 불변**: 세션 종료 후 MemoryWriter가 storage에서 전체 트랜스크립트를
  읽어 에피소드를 추출하므로, 요약이 원본 메시지를 대체하면 메모리 추출 입력이 손실된다. 요약 상태는 summarizer
  메모리에만 존재(세션 스코프, 파생 데이터라 영속화 불필요). 메모리 검색 쿼리도 history를 직접 읽어 무영향. (→ 이후
  아래 "이전 세션 컨텍스트 개편"에서 롤링 요약만 영속화)
- **트리거는 ContextBuilder.build() 내부**: 토큰 회계가 이미 있는 유일한 곳이라 SessionLoop/인터페이스 변경 없이 후킹.
  백그라운드 스왑이라 이번 호출엔 미반영, 다음 build부터 반영 — 트리거 비율 0.75가 in-flight 구간의 여유. 스왑 전
  초과분은 기존 drop-oldest가 hard cap으로 잔존(요약 실패 시 fallback 겸용).
- **하드캡 도달 요약은 폐기**: 요약 LLM `max_tokens=512`(오작동 가드)에 걸린 결과는 문장 중간 절단이라, 손상된 요약으로
  교체하는 것보다 이전 상태 유지 + 다음 트리거 재시도가 낫다. 판정은 `output_tokens >= _HARD_CAP_TOKENS`
  휴리스틱(LLMResult가 finish_reason을 노출하지 않아). 평소 크기는 프롬프트 soft limit(250 words)으로 제어하고, 회계는
  목표치가 아닌 생성물 실측 토큰으로 한다.
- **커트라인은 교환 경계 정렬**: ConversationHistory가 user/assistant 메시지에 각각 turn_id를 부여하므로(1턴 = 1메시지)
  최근 n턴 유지가 교환 중간을 자를 수 있다 — 질문만 요약에 들어가고 답변이 밖에 남는 것을 막기 위해 첫 유지 턴이 user로
  시작할 때까지 커트라인을 뒤로 민다. barge-in의 사후 `update_message`는 항상 최신 턴 대상이라 n(20턴=10교환) 유지가
  요약된 턴의 사후 수정도 방어.
- **요약 스레드는 daemon 단발 Thread**: single-flight(동시 1개)라 executor가 불필요하고, non-daemon executor 스레드는
  프로세스 종료를 in-flight LLM 호출(최대 30초)만큼 블로킹한다. 세션 정리는 ThreadedTurnGPT와 같은 패턴(다음 세션 생성
  시 close).

### 이전 세션 컨텍스트 개편 — 최근 세션 블록 분리 + 직전 세션 원문 이월 (2026-07)

- **세션 단위 선택·절단 채택 (에피소드 단위 기각)**: "블록에 있음 ⟺ 검색에서 제외" 불변식이 세션 ID만으로 성립해
  retriever 인터페이스가 무변경. 에피소드 단위(최근 n개)는 세션 경계와 어긋나는 부분 포함 세션이 생겨 에피소드 ID 제외
  메커니즘이 필요해진다.
  - 기존 사각지대도 해소 — 구 코드는 가져온 최근 3세션을 무조건 `exclude_session_ids`에 넣었는데 build()의 예산 drop으로
    블록에 안 실린 세션도 검색에서 빠질 수 있었다. 이제 제외셋은 실제 포함된 세션 기준으로 로드 시점에 확정.
- **soft cap (최신 1세션 보장)**: 세션 통째 포함이 전제라 hard cap이면 긴 최신 세션에서 블록이 통째로 비는 역전이 생긴다
  — 캡 초과가 빈 블록보다 낫다. 이후 세션은 통째로 들어갈 때만 추가하고, 첫 미달 세션에서 중단(건너뛰고 더 작은 옛
  세션을 계속 담으면 시간 연속성이 깨짐).
- **원문 트랜스크립트 폴백 제거 → 직전 세션 이월로 대체**: 추출 미완 레이스가 걸리는 세션은 사실상 직전(N-1) 하나뿐 —
  N-2 이전은 다음 세션 시작 시점이면 처리 완료다.
  - 직전 세션을 항상 원문으로 히스토리에 이월하면 레이스 자체가 사라지고, 짧은 타임아웃 후 재-wake 시 대화가 끊김 없이
    이어지는 UX 이점, 표현이 원문(직전)→에피소드(그 이전)로 자연 강등되는 수명 주기를 얻는다.
  - 간격이 길어도 항상 이월(결정론적 동작 우선), 간격 정보는 타임스탬프 마커로 전달.
- **퇴거 우선 규칙 — 이전/현재 세션 요약 혼합 금지**: 트리거(예산 75%) 도달 시 이월분을 통째로 제거하고 에피소드를 최근
  세션 블록에 승격 → 그래도 초과면 그때 현재 세션만 롤링 요약.
  - 이전 세션 전용 요약 LLM 호출은 없다 — 에피소드가 이미 그 세션의 요약("별도 요약 호출 없음" 원칙 유지)이고, 곧 버릴
    중간 요약을 만들 이유가 없다.
  - 이월 길이별 분기도 불필요: 길면 퇴거만으로 여유가 생기고 짧으면 같은 판정에서 요약이 바로 이어진다. 요약기는 이월
    존재 시 `maybe_schedule` 호출 자체가 게이트되므로 summarizer 로직 무변경.
- **이월 재료는 저장소에서 재조립 — 롤링 요약만 신규 영속화**: 원문은 write-through로 이미 messages에 있고, 세션 종료 시
  휘발되는 유일한 조각이 롤링 요약이라 스왑 성공 시마다 저장(최신 1세션분, 단일 행 upsert). `through_turn_id`를 함께
  저장해 요약이 커버한 턴을 원문 이월에서 제외(중복 주입 방지).
  - 블록 전체 스냅샷 저장 대안은 기각 — 이중 저장(진실 원천 분열), 종료 시점 기록이라 크래시 창 확대(현 방식은 크래시된
    세션도 이월됨), 통짜 텍스트라 턴 단위 drop 불가, 마커의 새 세션 시각이 저장 시점엔 미정.
- **퇴거 시 캡 재적용 안 함**: 이월 세션 에피소드는 캡 초과여도 블록 3에 추가하고 기존 세션을 drop하지 않는다 — 제외셋이
  세션 내내 불변(검색 일관성, 동시성 단순화). 캡은 다음 세션 시작 시 재적용된다. 캐시 무효화는 세션당 최대 1회.
- **이월분 예산 처리는 라이브 턴 우선**: 히스토리 예산(8192) 내에서 현재 세션 턴을 먼저 채우고 잔여 예산에 이월을 렌더 —
  이월 턴이 전역에서 가장 오래된 턴이므로 oldest-drop 순서가 보존된다. 에피소드 미도착(write 지연/실패) 엣지에서 별도
  처리 없이 이월 턴부터 자연 소거되는 퇴화 동작.
- **후보 수 상한(10) 제거 → 캡 도달까지 lazy 페이지 워크**: 최초 구현의 후보 조회 상한은 토큰 캡과 별개로 바인딩될 수
  있는 의미 섞인 상수였다 (에피소드 0건 세션도 슬롯을 소모 → 캡이 남았는데 블록이 짧아지는 케이스).
  - 세션 하나 확인 비용(인덱스 조회+포맷+토큰 카운트)이 밀리초 미만이고 워크는 캡 바인딩에서 멈추므로 상한의 실익이
    없음 — 세션 목록은 전량 조회(GROUP BY라 어차피 전체 스캔), 에피소드 로딩만 페이지 단위 lazy로 전환해 페이지
    크기(20)를 동작에 영향 없는 순수 배치 상수로 강등.
  - 시간 기준 컷오프 대안은 기각 — "항상 이월 + 간격은 타임스탬프로 전달" 철학과 어긋나고, 세션 헤더가 이미 날짜를
    나른다.
  - 에피소드 토큰 수 사전 저장(컬럼 추가) 대안도 기각 — 계산이 병목이 아니고, 토크나이저 교체 시 낡아지며, 캡 판정
    대상은 헤더 포함 포맷 텍스트라 합산만으로는 부정확.
- **SQLite 히스토리 백엔드에 내부 락 추가**: 기존에는 ConversationHistory의 락이 유일한 직렬화 지점이었는데, 요약
  영속화로 summarizer 워커 스레드가 같은 커넥션에 쓰는 두 번째 writer가 됐다. SQLiteMemoryStorage와 같은 패턴으로 백엔드
  자체를 스레드 안전하게 전환.


## 기록 당시 남은 문제

- **TurnDetector 유사도 임베딩 캐싱**: `_text_similarity` 호출 패턴에서 한쪽 텍스트(`_last_prepare_text`)가 반복됨. 한쪽
  임베딩을 캐싱하면 추론 비용 절반 가능.
