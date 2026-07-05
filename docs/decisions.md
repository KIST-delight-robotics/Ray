# Decision Log

비자명한 설계 결정과 시행착오 기록. 코드에서 바로 읽을 수 있는 구현 상세는 제외.


## ASR

- **Google Cloud STT V1**: V2는 batch/adaptation 등 불필요한 기능만 추가. 실시간 스트리밍 용도에는 V1으로 충분.


## LLM

- **OpenAI Responses API**: System message는 `instructions` param으로 전달, `input`에 넣지 않음. `previous_response_id`는 사용하지 않음 — ContextBuilder에서 토큰 예산 기반으로 직접 컨텍스트를 관리.


## TTS

- **기본 vendor = ElevenLabs**: OpenAI tts-1은 TTFB 0.9~8.2s로 편차가 크고 짧은 문장에서 realtime factor 0.54×(재생보다 느림)까지 떨어짐. ElevenLabs flash_v2_5는 Pi 실측 TTFB 중앙값 220~305ms — vendor 전환 자체가 큰 지연 개선 (`scripts/bench/bench_tts.py`).
- **ElevenLabs `stream/with-timestamps` endpoint**: base64+JSON 스트림이라 raw PCM 대비 네트워크 ~33% 오버헤드가 있지만, character alignment → word timestamp 집계로 `truncate_by_timestamps` 정밀 barge-in 절단이 처음으로 활성화됨 (OpenAI는 timestamp 미지원이라 항상 ratio 추정 fallback이었음). Pi 실측 오버헤드는 plain stream 대비 TTFB +15~35ms 수준이라 수용. 문제가 되면 `_iter_chunks`만 교체해 기본 stream endpoint로 전환 가능한 구조.
- **`pcm_24000` 고정**: OpenAITTS와 샘플레이트를 맞춰 VAP 참조 채널·C++ 재생·greeting WAV 등 downstream 전부 무변경. tier 제한도 없음 (`pcm_44100`은 Pro 전용).
- **vendor 선택 = `create_tts()` 팩토리 + 모듈 내 파라미터 기본값**: 팩토리는 추상화 수단이 아니라(추상화는 `ITTS`가 담당) "이름→클래스" 매핑의 단일 위치 — 스크립트 `--vendor` 인자 같은 런타임 선택용. 기본 vendor는 `factory.py`의 `_DEFAULT_VENDOR` 파라미터 기본값으로 (`create_embedder(backend="local")` 선례). env var는 설정 표면 증가, mutable 모듈 전역은 wiring 결정 분산이라 기각.
- **ElevenLabs SDK gotcha — lazy generator**: `stream_with_timestamps()`는 generator function이라 HTTP 요청·에러가 첫 `next()`에서 발생 (OpenAI의 eager CM enter와 다름). 인증/voice 에러도 `synthesize()`가 아닌 iteration 중 TTSError로 표면화. SDK 기본 timeout 240s는 실시간 대화에 부적합해 override 필수 (현재 10s).
- **timestamps 집계는 절대 raise 금지**: sentence 모드 consumer에서 `stream.timestamps` 읽기 예외는 턴 전체를 실패시킴 (consumer_error 경로). alignment 길이 불일치는 절단+warning, 시간 역전/음수는 clamp. timestamps는 best-effort 부가 기능 — 실패해도 오디오 재생은 정상이어야 함.
- **단어 집계 = 공백 run 분리, 스트림 종료 시 1회**: `truncate_by_timestamps`가 `text.split()` 토큰을 전제하므로 character alignment를 `ch.isspace()` run 기준으로 묶음. 단어가 chunk 경계에 걸칠 수 있어 chunk별이 아니라 스트림 종료 시점 1회 집계 (alignment 시간은 오디오 시작 기준 절대값).
- **ElevenLabs Free tier voice 제약**: Rachel 등 구형 premade voice는 라이브러리로 이관돼 API 호출 시 402 `paid_plan_required`. 현행 default voice(Sarah/George/Daniel/Jessica 등)는 사용 가능. scoped API 키는 `voices_read` 권한이 없어 voice 목록 조회도 불가 — voice 교체 시 TTS 실호출로 검증해야 함.
- **OpenAI WAV header quirk (역사적)**: OpenAI TTS가 WAV 포맷 요청 시 `n_frames = INT_MAX`인 malformed 헤더를 반환하는 경우 있음. 현재는 `response_format="pcm"` raw 스트리밍이라 해당 없음 — WAV가 필요한 소비자에 넘길 때만 주의.


## TurnGPT

- **KV cache 활용**: ASR이 incremental update(같은 prefix, 늘어나는 suffix)를 보내므로 매번 전체 dialog를 재처리하면 낭비. 토큰 prefix 비교 후 새 토큰만 forward. 동일 입력이면 캐시된 확률 반환.
- **Context window = 최근 턴 유지**: 완료된 최근 턴 `_KEEP_TURNS=2`개 + 진행 중 턴만 유지(`<ts>` 경계 기준). `_MAX_CONTEXT_TOKENS=1024`(GPT-2 한계)는 그 뒤 hard truncation 안전망. Eviction은 KV cache를 무효화하지 않음 — forward의 prefix 매칭이 재사용/재계산을 알아서 결정.
- **ONNX backend**: RPi 추론 성능용. backend는 `_BACKEND` 클래스 리터럴로 선택 (기본 "onnx"). ONNX 경로는 torch 미사용 — 토큰화도 numpy(`return_tensors="np"`), torch import는 PyTorch 백엔드 전용. KV cache 지원은 ONNX input name에 `past_key_0` 존재 여부로 감지.
- **ONNX threads = 2**: RPi 5 (4코어) 벤치마크 1–4스레드. 2스레드가 fp32/int8 모두 최적 (int8 42ms, fp32 111ms). 4스레드는 contention으로 오히려 느림.
- **int8 양자화**: 4x 작고 2.6x 빠르며 TRP 차이 무시 가능 (~0.04). 배포 권장.


## MaAI VAP

- **INT8 양자화 부적합**: CPC 인코더(Conv1D+LSTM, 2.5M params)에 `quantize_dynamic` 적용 시 1.9x 느려지고 정확도 대폭 하락. TurnGPT(MatMul 위주, 163M params)와 달리 작은 Conv 커널에서는 양자화 오버헤드가 연산 절감을 초과하고, 모델이 이미 L2 캐시에 들어가 메모리 대역폭 이점 없음.
- **배치 처리 무의미**: No-cache 배치(N frames)는 KV cache + 1 frame보다 느림. KV cache가 이미 반복 연산을 제거하므로 텐서 크기를 키워도 추가 이점 없음.
- **ONNX transformer export**: Transformer를 ONNX export하여 전체 파이프라인(encoder+transformer)을 ORT로 실행. PyTorch dispatch overhead 완전 제거. Mean 24ms (vs PyTorch 106ms, 3.9x). `torch.compile` warmup 100프레임 문제도 해소. 변환 시 dict KV cache → 12개 flat stacked tensor. Cross-attention source 순서 주의 (원본 입력을 src로 전달, 업데이트된 값 아님). 수치 차이 max 6.8e-6.
- **ORT 싱글스레드 최적**: Transformer ONNX 추가 후에도 `ort_threads=1`이 최적. `ort_threads=4`는 동기화 비용으로 2x 느려짐 (48ms vs 24ms).
- **PyTorch `p_now` 리스트 반환 버그**: `VapGPT.forward()`가 `p_now`을 `[speaker1, speaker2]` 리스트로 반환. `float(out["p_now"])`은 항상 실패하지만 integration test 없어 미발견. `p_now[0]` 추출로 수정.
- **torch.compile 한계**: 작은 텐서([1,1,256]) 연산에서 커널 launch + 메모리 할당 오버헤드가 남아 이론치(0.4ms) 대비 32ms. 10ms 이하를 원하면 ONNX export 필요. 현재 ONNX가 기본.


## TurnDetector

- **Paper-based two-path algorithm** (Skantze & Irfan, 2025): Path 1 (VAP sustained robot-favor) OR Path 2 (TurnGPT graduated silence timeout). 단일 모델 의존 회피 — VAP가 빠른 응답(~500ms), TurnGPT가 VAP 실패 시 eventual turn-taking 보장.
- **Backchannel vs interrupt 구분**: ROBOT_TURN에서 robot_audio 있을 때 `p_now > threshold`만으로는 interrupt 판단 불충분 — `p_fut`도 user를 favor해야 함. 백채널("응", "네")은 p_now만 spike하고 p_fut는 robot-favoring 유지.
- **`robot_audio=None` 시 unconditional interrupt 제거**: robot audio 없이 `user_is_speaking=True`만으로 interrupt 하던 것을 제거. 원인: 1–3s generation gap 동안 주변 소음이 false interrupt 유발 (실측: 7회 LLM 호출, 4.7s 지연). VAP는 robot 채널 없이 interrupt/backchannel 구분 불가 → `TurnDecision.none()` 반환. (이후 Phase 상태기계 도입으로 generation gap은 ROBOT_TURN이 아닌 PENDING 구간이 되었고, 이 구간의 사용자 재발화는 아래 cancel 경로가 처리 — 다음 섹션 참조.)
- **`user_is_speaking` 전제조건**: 논문 pseudocode (Appendix A lines 56-61) 요구. 없으면 turn-shift 직후 VAP context가 아직 user-biased일 때 transient spike로 false interrupt 발생.
- **VAP error default `VAPResult(0, 0, False)`**: "robot favored"처럼 보이지만, transient error는 500ms 지속 안 되므로 Path 1 false trigger 없음. Persistent failure 시 Path 2(TurnGPT + silence)가 독립 작동.


## 턴 종료(cancel/interrupt) — Phase 상태기계

- **cancel/interrupt 경계 = begin_streaming**: "레이 음성이 잠깐이라도 났으면 무조건 interrupt(cancel 아님)"라는 사용자 관점 기준을 만족시키려면 재생이 *가능해지는* 마지막 Python 지점을 경계로 잡아야 함. `playback_started`는 C++가 `play()` 호출 시점(가청 시작보다 SFML/ALSA 버퍼만큼 앞)에 보내고 Python은 폴링 지연 후 처리 → 가청 시작과의 선후가 수십 ms 내에서 모호. `begin_streaming`(=`send_stream_start`)은 그 이전엔 C++에 재생 명령이 없어 *물리적 무음 보장*이 되는 유일 지점. 그래서 cancel은 항상 begin_streaming 이전(브리지 무접촉)이라 `send_stop` 불필요·STOPPING 미경유, interrupt는 항상 이후 — STOPPING이 항상 interrupt 의미가 되어 reason 태그 불필요.
- **STREAMING은 종료-감지 공백 구간(단일채널 interrupt 안 함)**: `robot_audio`(VAP 참조 채널)는 `playback_started`가 클럭을 세팅해야 생김. 그 전 STREAMING에선 VAP가 interrupt vs backchannel을 구분할 robot 채널이 없음. 단일채널 fallback은 (a) backchannel 오인터럽트, (b) stop_pos≈0 빈 기록 문제로 기각, interrupt 감지를 PLAYING으로 미룸. 비용: STREAMING 내에서 시작·종료하는 짧은 barge-in은 그 턴엔 무시됨(begin_streaming에서 ASR이 reset되고 이후 누적되므로 다음 턴 입력으로 이연 — 유실 아님). bridge_ms 실측 중앙값 ~97ms라 공백은 보통 짧고, 긴 STREAMING(느린 TTS)이 최악. 근본 대응은 C++ prebuffer·TTS throughput.
- **cancel 신호 = user_is_speaking 전제 + p_now/p_fut(즉시), grace는 유사도**: cancel은 interrupt와 동일 구조 — `user_is_speaking`을 전제로 하고(실제로 말해야 함) 그 위에서 `p_now/p_fut`로 플로어 회수를 확인. user_is_speaking *단독*은 backchannel·노이즈에 약하고, p *단독*은 무음 중 확률 변동에 오발화하므로 둘 다 필요. VAP가 네이티브 10Hz(각 결과가 이미 ~100ms 적분)라 100ms 미만 sustain은 같은 캐시 추론 재독이라 무의미 → 즉시 발화. ASR finalization noise는 시간 grace 대신 **마지막 prepare 텍스트(=응답이 생성된 기준)** 와의 유사도로 거름 — prepare-skip 게이트와 **같은 비교의 양면**이라 별도 기준선 추가 없이 `_last_prepare_text` 재사용. 이 user_is_speaking 전제가 "침묵 timeout으로 shift했는데 p만 높은" 모순 입력에서의 turn_shift↔cancel thrash도 막음.
- **detector 상태 wipe를 turn_shift→commit으로 지연 (PENDING 도입)**: turn_shift는 로봇이 실제 커밋(begin_streaming)하기 전까진 잠정적. 기존엔 turn_shift 직후 per-frame 상태를 전부 지워 "cancel=같은 턴 연속"이 불가능했음. PENDING은 상태를 보존해 cancel 시 매끄럽게 rewind하고, commit에서 비로소 wipe+dialog append. 부수 효과로 detector가 interrupt 모드(ROBOT_TURN)에 진입하는 시점이 robot_audio가 생기는 시점과 일치 → "ROBOT_TURN인데 robot_audio 없음" 사각 소멸.
- **stale 응답 방지 = turn_shift의 prepare 선점 (detector 내부)**: turn_shift 조건이 충족돼도, 마지막 prepare 이후 ASR이 *유효하게(비유사)* 바뀐 게 남아 있으면 turn_shift 대신 **prepare를 먼저** 내보내 새 텍스트로 재생성하고 다음 프레임에 shift. "늦은 finalization으로 준비된 응답이 stale" 케이스를 detector의 기존 유사도 게이트로 그대로 처리 — SessionLoop에 임베더/similarity 주입 불필요(검사 중복 회피). `at_turn_shift`일 때만 발화 스로틀(turngpt/0.2s)을 우회하고 `_asr_has_changed` 게이트는 유지 — 텍스트가 안정된 흔한 경우엔 turn_shift가 바로 fire(speculation 이득 유지).
- **Python↔C++ 순수 전송은 무시 가능(~0.04ms 편도, Pi 루프백 IXWebSocket 실측)**: turn-taking 타이밍에서 전송은 0으로 취급. bridge_ms(~57–97ms)는 통신이 아니라 C++ prebuffer + Python 프레임 폴링(~30ms)이 지배.


## Silero VAD

- **세션 시작마다 `reset_states()` 필수**: LSTM 상태가 세션 간 유지되어, 직전 세션 하나(23s)만 선행해도 조용한 음성(peak 0.107)의 감지율이 38.9%→0%로 붕괴 — 같은 녹음을 단독으로 돌리면 max 1.000. 손상된 상태는 이후 오디오(무음 포함)로 자연 회복되지 않으며, 음량이 낮을수록 취약. ONNX/PyTorch 무관. 리셋 비용 ~11µs(청크 추론의 1/60)로 무시 가능. 워밍업 공백(~250ms)은 발화 *도중* 리셋에만 발생 — 리셋 후 발화가 *시작*되는 경우는 즉시 감지.
- **세션 내부 오염은 commit() 시점 리셋으로**: 멀티턴 세션에서 시작 1회 리셋만으론 턴2부터 감지율 5~12배 붕괴(67.9%→5.7%) — 오염원은 로봇 응답(마이크에 거의 안 잡힘)이 아니라 **사용자 자신의 직전 발화**. 리셋 시점은 begin_streaming(=TurnDetector.commit, `vad_reset_fn` 주입) 채택: STREAMING 구간은 인터럽트 판정에 robot_audio가 필요해 VAD 결과가 어디에도 안 쓰이는 유일한 공백이고, 갭에서 말을 시작해도 리셋 *후* onset이라 즉시 감지 → PLAYING 진입 시 user_is_speaking 준비됨. playback_started 리셋(갭 발화를 한가운데서 절단)·playback_complete 리셋(빠른 재발화와 겹침)은 기각.
- **SLEEP 진입 리셋은 wakeword 모듈 책임**: wakeword는 인식 사이클 종료마다 자체 리셋해 SLEEP *중* 오염은 자가 회복되지만, ACTIVE→SLEEP 잔류 오염은 "발화가 threshold를 넘어야 리셋 발동" 구조라 조용한 호출에 회복 기회가 없음(미감지→리셋 불발 순환). `WakewordDetector.reset()` 공개 메서드를 `__main__`의 SLEEP 전환 3곳에서 호출 — wakeword의 `_pre_buffer`/`_vad_buffer` 잔류 오디오(STT에 묵은 청크 전송 문제)까지 모듈 내부에서 함께 정리.


## Async Thread Separation (VAP, TurnGPT)

- **별도 스레드 필요 이유**: RPi 5 worst case VAP 24ms + TurnGPT 30ms = 54ms, 30ms frame budget 초과. ONNX Runtime이 GIL 해제하므로 별도 스레드에서 진정한 병렬성 확보.
- **1-frame TurnGPT delay 허용**: `process_frame()`에서 poll 후 submit. Frame N의 submit 결과는 Frame N+1에서 poll. 30ms 지연은 500ms+ 단위의 turn-taking 판단에 무시 가능.


## SpeechGenerator

- **Streaming API**: `poll_audio() → bytes | None` + `stream_done`으로 TTS 청크를 즉시 CppBridge에 전달. 전체 합성 완료 대기 불필요.
- **`max_workers=2`**: 1이면 새 `prepare()`가 cancel된 run이 blocking API call(1–3s)에서 빠져나올 때까지 대기. 2면 즉시 시작. Turn detector 시그널 빈도상 3+ 동시 취소는 비현실적.
- **영어 기준 문장 경계 감지**: `.` `!` `?` + 뒤따르는 공백으로 판단. 약어(`Mr.`, `Dr.`, `etc.`)와 단일 대문자 이니셜(`U.S.`)은 frozenset 기반 휴리스틱으로 제외. NLP 토크나이저 대비 의존성 없고 지연 없음. 한국어 등 다른 언어 지원 시 별도 전략 필요.
- **`min_flush_words = 4`**: 문장 경계가 감지되어도 누적 단어 수가 임계값 미만이면 다음 문장과 합쳐서 TTS. `"Sure!"` 같은 짧은 감탄사가 단독 TTS 호출되면 HTTP 오버헤드가 실제 합성 시간을 초과. 대부분의 LLM 첫 문장은 5~15단어이므로 즉시 발송됨.
- **TTS 파이프라이닝 (producer-consumer)**: LLM 스트리밍 + 문장 감지(producer)와 TTS 결과 drain(consumer)을 별도 스레드로 분리. 문장 N의 TTS 진행 중 문장 N+1의 TTS를 동시 시작하여 문장 간 gap 최소화. 로컬 `ThreadPoolExecutor(max_workers=2)`를 파이프라인 실행마다 생성/소멸하여 기존 `self._executor`(prepare 재시작용)와 독립.
- **인용 태그 처리**: `[MEMORIES: M1, M3]`는 문장 종결 부호가 없으므로 SentenceDetector 버퍼에 자연스럽게 잔류. LLM 스트림 종료 시 `flush()` → `parse_citation_tag()` 순으로 처리. 중간 문장에 태그가 섞일 가능성 없음.
- **`_text` 누적 갱신**: consumer가 각 문장의 TTS 합성 완료 직후(해당 오디오 큐 적재 직전) `self._text`를 갱신. `get_text()`가 TTS가 끝난 문장 범위만 반환하므로 barge-in truncation 정확도가 full 모드(전체 텍스트 즉시 설정) 대비 개선됨.
- **Sentence streaming SessionLoop/CppBridge 변경 없음**: 기존 `send_stream_start → send_audio(chunk)* → send_audio_end` 프로토콜이 chunk 출처와 무관하게 동작. C++는 버퍼에 쌓이는 PCM을 360ms 단위로 drain하므로 TTS 호출 횟수를 알 필요 없음.
- **Sentence streaming barge-in 호환성**: PLAYING 중 interrupt 시 generator를 cancel하지 않는 기존 동작 그대로 유지. consumer가 계속 실행되어 `stream_done = True` 설정 → deferred truncation 패턴이 동일하게 적용. 다음 `prepare()` 호출 시 `cancel_event`가 문장 간 체크에서 잡혀 불필요한 TTS는 최대 진행 중 1회로 제한.


## SessionLoop

- **만성 batch-drained는 루프 비용이 아니라 시스템 부하 신호**: "Batch-drained 2 frames"가 ~170ms 간격으로 1110회 발생한 런 분석 — 발생 간격이 케이던스를 알려줌(루프가 프레임 예산 30ms를 δ만큼 초과하면 30/δ 프레임마다 한 번 2프레임 drain, 170ms 간격 = δ≈5ms). 단계별 누적 타이머 실측 결과 루프 고유 비용은 2~3ms/it(Silero VAD가 ~75%, ASR gRPC feed는 0.03ms/it로 무혐의)에 불과했고, 동일 코드·오디오의 다음 런에서는 17회로 소멸 → 원인은 머신 전체 부하(외부 CPU 경쟁·발열 스로틀링). 판별 기준: 프레임 루프와 무관한 백그라운드 스레드인 **VAP cycle overrun이 동반 증가**하면 시스템 부하, 특정 스테이지 스파이크면 코드 문제. 가끔의 5~10프레임 drain은 LLM HTTP 응답 처리+VAP 슬로우가 겹치는 일시 스파이크로 `_MAX_BATCH_FRAMES=10` 설계 범위 내. 계측 코드는 진단 후 제거 — 재발 시 단계별 마킹 + 세션 종료 debug 요약을 재삽입하면 한 줄로 구분 가능.


## C++ ↔ Python Protocol

### WebSocket topology
- **C++ server, Python client**: C++가 `ix::WebSocketServer` port 9200. Python이 `websockets`로 접속. 단일 연결 — `g_client_ws`에 저장.

### Protocol 단순화
- **`turn_id` 제거**: WebSocket TCP 순서 보장 + Python 상태 머신(playback_complete 대기 후 다음 turn)으로 stale chunk 오염 방지. C++는 `stream_start` 시 버퍼 클리어.
- **`playback_stopped` 제거 → `playback_complete`로 통합**: Python이 자체 `Phase.STOPPING` 상태(+`_stop_pending_time`)로 정상 완료/barge-in 구분. C++는 항상 `playback_complete`만 전송.
- **`playback_position` 스트림 제거**: `stop_pos = stop_pending_time - playback_start_time`으로 시간 기반 추정. Localhost에서 ±100ms 정확도 허용.
- **메시지 통합**: `stream_start` ← `responses_only` + `responses_stream_start`. `audio_end` ← `responses_stream_end`. `play_file` ← `play_audio` + `send_greeting` + `send_farewell`. `stop` ← `user_interruption`.
- **`playback_started` (신규)**: `control_motor` cycle 0에서 `soundStream.play()` 후 전송. Barge-in 위치 추정 및 VAP robot audio 타이밍 기준점.
- **Interrupt 시 `playback_complete` 전송**: C++가 interrupt cleanup 후에도 `playback_complete` 전송. Python의 STOP_PENDING 상태가 정상 해소됨.


## LED / 음악 댄스

- **LED 밝기 = RPi 하드웨어 PWM(sysfs), Dynamixel과 분리**: 밝기는 RP1 하드웨어 PWM(`/sys/class/pwm/pwmchip0/pwm1`, GPIO13)에 duty(ns)를 써서 구동, 부팅 systemd 서비스(`setup_led_hwpwm.sh`)가 캐리어를 세팅. 초기엔 WiringPi softPwm(핀 무제약, 추가 의존성 0)이었으나 ~100Hz 주파수 한계(플리커)로 하드웨어 PWM으로 교체. `led_pwm_pin < 0`이면 비활성 — 배선 전이나 다른 보드에서 임의 GPIO를 건드리지 않게 방어. ID6 Dynamixel LED 각도 모터 구동은 제거됨 — 메인 파이프라인은 밝기 1채널만.
- **LED CSV 프레임 정렬**: LED CSV는 헤드/입 CSV와 행 수·시간축이 같다는 핸드오프 전제에 의존. `csv_control_motor`의 SKIP_FRAMES 보간 구간에서도 LED 행을 같은 박자로 소비해 정렬 유지(LED는 보간 없이 원본값). LED 파일이 없으면 graceful 비활성 — 모터 1~5는 정상 재생.
- **시그널 핸들러 데드락 → 워처 스레드**: csv 재생 중 Ctrl+C가 안 듣던 버그 — `signal_handler`가 인터럽트당한 스레드 컨텍스트에서 `dxl_mutex_`를 잡으려다, 매 프레임 모터 write로 뮤텍스를 쥔 메인 스레드(=자기 자신)를 기다리는 데드락. SIGTERM도 동일, SIGKILL만 들었음. 해결: 핸들러는 원자 플래그(`g_shutdown_requested`)만 세우고 반환, 별도 `shutdown_watcher` 스레드가 정상 컨텍스트에서 LED 소등+토크 해제 후 `_Exit`. dxl_driver는 `delete` 안 함(use-after-free 경합 회피) — 토크만 끄고 OS 회수에 맡김. (교훈: 시그널 핸들러에서 mutex/`cout`/`new`·`delete`는 async-signal-safe가 아님.)
- **음악 댄스 데모 = 한 프로세스·한 클럭**: `music_dance/`는 Python(librosa) 오프라인 분석 → `timeline.csv` → C++ 플레이어가 WAV 재생 위치(steady_clock)를 마스터 클럭으로 20ms마다 LED 밝기·모터 골을 샘플링 구동. 브릿지/IPC가 없어 지연·지터 없이 동기화가 자연 해결. (당초 구동부까지 순수 Python 계획이었으나 C++로 구현됨. 메인 로봇 모드 통합은 미래 작업 — 분석 코어 재사용 전제로 분리 유지.)
- **LED 밝기 신호 = HPSS 블렌드**: 단색 LED 1채널이므로 밝기만 제어. `librosa.effects.hpss`로 하모닉(지속음 → 느린 attack/release 글로우 바닥)/퍼커시브(트랜지언트 → fast attack/slow release 펀치)를 분리 블렌드 — 멜로디와 드럼이 섞이지 않아 박은 선명, 무드는 매끄럽게. 처리 체인: dB 스케일 → 퍼센타일 정규화(곡별 다이내믹 레인지 활용, 이상치 강건) → 비대칭 스무딩 → 감마 ~2.2(지각-선형 밝기) → 바닥값 ~12%(스트로브 방지). 사람은 음량을 로그로, 밝기를 비선형으로 지각 → 두 보정이 "귀로 듣는 크기 ≈ 눈으로 보는 밝기" 정합을 만듦. HPSS는 lookahead 필요 → 오프라인 전제, 실시간(마이크) 입력 시 재설계 필요.


## Session Lifecycle

- **Session factory 도입**: 이전 설계는 단일 SessionLoop/ConversationHistory를 재사용하며 `reset()`으로 상태 정리. 그러나 TurnDetector의 `_dialog_parts`가 `reset()`으로 안 지워져서 세션 간 상태 누수 발생. 수정: `session_factory`가 매 세션마다 SessionLoop, TurnDetector, SpeechGenerator, ConversationHistory를 새로 생성 (ContextBuilder는 SpeechGenerator 내부에서 생성되어 함께 재생성됨).
- **Three-tier lifecycle**: (1) Process-level — 모델, API 클라이언트, 하드웨어, executor (비싼 초기화, 1회). (2) Session-level — 상태 있는 orchestration 객체 (factory 재생성). (3) Turn-level — 경량 `reset()` (ASR 버퍼, TurnDetector 프레임 카운터).


## Embedding / Similarity

- **Sentence embedding > SequenceMatcher**: SequenceMatcher는 문자 겹침 측정 — `"what is your"` vs `"what is your name"`이 0.87 (threshold에 차단됨). Sentence embedding (all-MiniLM-L6-v2)은 0.66 (정상 통과). 논문이 semantic similarity 사용.
- **ISimilarity 모듈 제거 → IEmbedder 직접 주입**: similarity 래퍼 모듈을 없애고 TurnDetector가 `IEmbedder.embed_batch`로 코사인 유사도를 직접 계산 (`_text_similarity`). 임베더는 process-level 단일 인스턴스를 memory와 공유.
- **기본 임베더 = ONNX qint8 ARM64**: `create_embedder` 기본값이 `onnx/model_qint8_arm64.onnx`. qint8 배치 임베딩 기준으로 TurnDetector `_SIMILARITY_THRESHOLD`는 0.85로 상향.
- **sentence-transformers 3.x 핀**: 5.x는 `transformers>=5.0` 필요, `optimum`의 ONNX runtime과 충돌. 추론 전용이므로 기능 차이 없음.


## Conversation History

- **Write-through SQLite**: 매 메시지 즉시 INSERT. 세션 중 crash 시 최대 진행 중 turn 1개만 유실 (batch-at-end였으면 전체 세션 유실). `save()`는 `ended_at` 설정 + WAL checkpoint만 수행.
- **Graduated DB corruption recovery**: 정상 open → WAL 파일 삭제 재시도 → corrupt 파일 백업 + 신규 DB 생성. RPi 전원 차단 시 대부분 WAL만 손상되므로 WAL 삭제만으로 복구되는 경우가 많음.
- **Responses API format 직접 저장**: 중간 canonical format 없이 vendor-specific dict 그대로 `item_json`에 저장. Vendor 교체 시 migration script 필요 (의도적 트레이드오프).
- **`token_count` 이중 소스**: assistant 메시지는 API `output_tokens` (정확값), user/truncated 메시지는 tiktoken fallback. ContextBuilder는 저장된 값을 읽어 re-tokenization 없이 예산 계산.
- **Tool definition token cost 실측**: tiktoken으로 definition structure를 추정하면 부정확. API `input_tokens`를 tool 유무로 비교하여 실측 (`web_search` = 294 tokens). ContextBuilder가 예산에서 차감.
- **Turn-level atomic budgeting**: ContextBuilder가 `get_turns()`로 turn_id 기준 그룹 단위로 포함/제외. Tool call + result + assistant text는 분리 불가 — 한 turn이 예산 초과하면 통째로 제외.


## Memory Storage

- **단일 DB**: 기존 conversation history와 같은 `data/ray.db` 사용. 별도 `sqlite3.Connection`으로 접속 (WAL 모드에서 안전). DB 분리는 세션 참조 cross-join 불가, 관리 이중화 문제로 불리.
- **utterances 별도 테이블**: 기존 `messages`(Responses API JSON 형식)와 별도로 `utterances`(role, text, timestamp 평문)를 둠. messages는 LLM 컨텍스트 재구성용, utterances는 메모리 Write 시 에피소드 추출용. messages에서 파생 가능하지만, tool_call 등 비텍스트 항목 필터링이 필요하고 용도가 다르므로 분리.
- **FTS5 쿼리 sanitize**: ASR 텍스트가 검색 쿼리로 들어오므로 `-`, `"`, `NOT` 등 FTS5 특수문자가 연산자로 해석되면 silent fail. 각 토큰을 `""`로 감싸서 리터럴 처리. Prefix 검색(`*`)도 차단되지만, 검색 전략에서 필요 시 별도 쿼리 모드로 대응.
- **IEmbedder를 공유 모듈로 분리**: similarity와 memory가 동일 모델을 독립 로드하고 있어, `IEmbedder`를 `core/interfaces.py`로 승격하고 구현체를 `embedding/` 모듈로 추출. `IVectorIndex`는 memory 내부 전용이므로 그대로 유지.
- **실패 시 `None` 반환**: `add_episode`/`upsert_profile`이 DB 에러 시 `-1` 대신 `None`을 반환. `-1`은 다운스트림에서 유효한 ID로 오인될 위험 있음.


## Memory Retriever

- **`exclude_session_ids`를 `retrieve()` 파라미터로**: 생성자가 아닌 매 턴 호출 시 전달. 세션이 길어지면 히스토리 블록 토큰 제한으로 이전 세션 요약이 빠질 수 있어, 컨텍스트 상태를 호출자가 반영해야 함. retriever가 session_id를 직접 관리할 필요 없어져서 더 단순해짐.
- **Retained overflow eviction 기준: TTL 우선**: salience(현재 턴 쿼리 기반)가 아닌 TTL 기준으로 evict. retained buffer는 토픽 전환 시에도 인용된 기억을 보호하는 장치인데, 현재 쿼리 salience로 evict하면 토픽 전환 시 보호가 무력화됨. 동일 TTL 내에서는 저장된 salience(진입/갱신 시점 값)로 tiebreak.
- **프로필은 retriever 범위 밖**: 프로필은 세션 내 불변이므로 매 턴 검색할 필요 없음. 세션 시작 시 `storage.get_all_profiles()` 1회 호출로 처리.


## Memory Writer

- **세션 종료 시 즉시 추출**: 기존 설계(히스토리에서 밀려날 때 처리)에서 변경. 에피소드를 세션 종료 시 바로 추출하고, 추출된 에피소드가 세션 요약 역할도 겸함. 별도 요약 LLM 호출 절약.
- **3단계 순차 LLM 호출**: 에피소드 추출 → 프로필 fact 추출 → 프로필 Merge. 에피소드 추출과 프로필 추출은 작업 성격이 다르고(서사 생성 vs fact 식별), 프로필 추출과 merge도 다름(fact 식별 vs 기존 슬롯 대비 판정). 합치면 프롬프트 복잡도가 올라가고 실패 시 영향 범위가 커짐.
- **별도 LLMConfig(`write_llm`)**: 대화용 LLM과 모델·설정을 독립. 추출은 더 저렴한 모델(gpt-4o-mini), 낮은 temperature(0.0), 높은 max_tokens(4096), 도구 없음. 모델별 파라미터(reasoning_effort 등)가 다를 수 있어 필드 몇 개 추가보다 LLMConfig 통째 분리가 유연.
- **`ILLM.generate()`에 `response_format` 추가**: structured output(JSON schema)을 호출 시점에 지정. config가 아닌 파라미터인 이유: 에피소드/프로필/merge 각각 스키마가 다름. 기존 대화 호출은 None(기본값)으로 영향 없음.
- **프로필 Merge — `(topic, sub_topic)` 키 매칭**: 기존 슬롯에 P1/P2 인덱스를 매기는 대신, fact별로 기존 내용을 인라인 표시. LLM이 APPEND/UPDATE/ABORT만 판정하고, 코드에서 `(topic, sub_topic)` 키로 기존 슬롯을 매칭. 슬롯 순서 무관, `get_all_profiles()` 반환 순서에 비의존.
- **토큰 제한 — 조건부 경고**: 프로필 슬롯의 토큰 수를 `token_counter`로 측정, `profile_max_content_tokens`의 70% 초과 시 해당 fact에만 제한 지시 추가. 모든 슬롯에 무조건 제한을 명시하면 불필요한 압축을 유발.
- **importance 고정값(1.0)**: LLM이 0~1 연속값을 일관되게 매기기 어렵고, 기준 확정에 실사용 데이터가 필요. 데이터 쌓인 후 범주형(high/medium/low) 또는 reinforcement(인용 횟수) 기반으로 전환 검토. `citation_count` 필드는 미래용으로 추가해둠.
- **윈도우 처리 — 에피소드 추출만 분할**: 프로필 추출은 에피소드(이미 압축된 입력)를 받으므로 원문보다 훨씬 짧아 윈도우 불필요. 에피소드 추출도 기본은 세션 전체 1회 처리, `write_max_input_tokens` 초과 시에만 턴 단위 분할 + 겹침.
- **utterance에 `token_count` 저장**: 윈도우 분할 시 재토큰화 없이 저장된 값을 합산. orchestrator가 utterance 저장 시 이미 계산된 값을 전달.


## Memory Integration

- **메모리 쓰기는 `__main__` 주도**: SessionLoop이 MemoryWriter를 직접 의존하지 않도록, 세션 종료 시 `__main__.py`(FAREWELL 전환·종료 finally)에서 `write_executor.submit(memory_writer.process_session, ...)`를 직접 호출. SessionLoop은 메모리 시스템 존재 여부를 모름.
- **session_id를 factory에서 생성**: 기존엔 세션 루프 쪽에서 uuid를 생성했으나, factory에서 생성하도록 변경. factory가 profiles/previous sessions 로딩 시 현재 session_id를 exclude 조건에 사용해야 하므로, factory가 생성하고 `SessionComponents.session_id`로 반환.
- **Utterance 저장 위치: SessionLoop**: history 저장과 동일 시점에 `memory_storage.add_utterance()` 호출. SpeechGenerator나 ContextBuilder가 아닌 SessionLoop에서 수행하는 이유: 최종 확정 텍스트만 저장을 보장 (barge-in truncation 반영).
- **NumpyVectorIndex / SQLiteMemoryStorage에 threading.Lock 추가**: process-level 싱글턴이 3개 쓰레드(메인, SpeechGenerator 백그라운드, write_executor)에서 접근. NumpyVectorIndex는 `_ids`/`_matrix` 동시 읽기/쓰기 보호, SQLiteMemoryStorage는 단일 커넥션 직렬화.
- **임베딩 인스턴스 공유**: process-level 임베더 1개를 `TrackedEmbedder`로 감싸 memory(retriever/writer)와 TurnDetector 유사도 계산이 공유. (초기엔 config 경로 분리 문제로 별도 인스턴스였으나, ISimilarity 모듈 제거와 함께 단일 인스턴스로 통합됨.)
- **이전 세션 요약 = 에피소드 그대로 사용**: `get_episodes_by_session_ids()`로 이전 세션 에피소드를 로드하여 `format_session_summary_block()`으로 포맷. 별도 요약 LLM 호출 없음.
- **MemoryRetriever 매 세션 새로 생성**: retained buffer를 세션 간 격리하기 위해 factory에서 매번 생성. process-level 싱글턴인 memory_storage/vector_index/embedder를 주입받으므로 생성 비용 낮음.


## Eval

- **Prepare 유사도 게이트 평가 — judge 판정은 이진(meaning_changed / response_appropriate)**: 목적이 "threshold가 적절한가"라는 예/아니오 질문이라 5점 척도보다 harmful skip *비율*이 바로 해석됨. 응답까지 judge에 보여주는 이유는 의미가 달라져도 응답이 우연히 여전히 적절한 경우와 실제 피해(harmful)를 구분하기 위함 — threshold 문제의 심각도를 과대평가하지 않게 함.
- **similarity 값을 CallRecord로 기록**: judge 판정만으로는 "현 threshold에서 bad skip N건"까지만 알고 *얼마로 올려야 하는지*를 모름. 게이트의 4종 결정(skip/keep/regenerate/cancel)마다 similarity를 기록하면 harmful skip의 유사도 구간이 보여 조정 폭이 정량화되고, regenerate 기록으로 "threshold를 낮추면 skip으로 바뀌었을 건수"도 역산 가능. 단, counterfactual 시뮬레이션이 아니라 현재 threshold에서 실제 발생한 결정만 판정.
- **턴 매칭 = `call_records.turn_index` 전용 컬럼** (초기엔 metadata JSON): 질문 단위 e2e 드릴다운이 vap/turngpt/tts 등 *모든* 모듈 호출을 턴별로 귀속해야 해서 게이트에만 있는 metadata로는 부족. `ICallStore`에 공유 턴 카운터(`set_turn_index`/`current_turn_index`)를 두고 TurnDetector(턴 권위)가 전환마다 갱신 — 백그라운드 스레드의 vap/turngpt까지 같은 인덱스를 스탬프. 생성(LLM/TTS)은 `commit()`으로 끝난 *직전* 교환에 귀속, 카운터는 `commit()`에서만 증가 — cancel rewind는 같은 턴의 연속이라 reset/cancel에서 올리면 안 됨. 타임스탬프는 µs 정밀도(`utc_now_str`)로 올려 cross-stage 기록 정렬 가능. report.py는 구 DB의 컬럼 부재를 감지해 metadata 폴백.
- **asr_text ≠ system_text가 곧 게이트 skip의 흔적**: `system_text`(messages의 user 메시지)는 `_begin_streaming`이 기록한 generator의 prepare 입력이고, `asr_text`는 turn shift 시점의 최종 ASR 텍스트. 두 값이 (정규화 후) 다른 경로는 유사도 게이트의 skip/keep뿐이라, 별도 계측 없이 이 차이만으로 judge 후보를 선별 가능.
- **질문 단위 e2e 드릴다운 = 기존 대시보드 인라인 펼침**: 카테고리 통계만으론 한 질문의 전 단계(ASR→턴감지→게이트→생성→TTS)가 어떻게 맞물렸는지 볼 수 없음. 질문은 이미 각 탭에 나열되므로 그 자리에서 펼치는 게 탭 이동 0회로 맥락 유지에 최적 — 전용 탭(질문 목록 중복)·CLI 승격(비시각·단건)은 기각. 기존 `.collapsible-toggle/.collapsible-body` JS 재사용.
- **드릴다운 데이터 경로 = report.py 사전 집계, dashboard는 scored.json만**: dashboard는 `f(scored.json)` 순수 함수 유지(자족적·공유 용이). report.py가 call_records를 turn_index로 묶어 턴별 `stage_calls`를 첨부, eval.db 직접 조회는 계약을 깨고 두 소스를 요구해 기각. vap 원시 프레임은 요약만(count/avg/max) — scored.json 폭증 방지. `call_issues`는 turn_index 도입으로 세션→턴 단위로 정확화.
- **배경 잡음 주입 = 질문 오디오와 분리된 별도 재생, 사전 믹싱 폐기**: MUSAN 디지털 사전 믹싱(질문 WAV에 SNR별 잡음을 미리 합성)을 구현까지 했다가 제거. 이유: 턴테이킹 감지에서 SNR이 의미 있으려면 질문 발화가 *끝난 뒤*(침묵 기반 턴 판정이 일어나는 구간)에도 잡음이 지속되어야 하는데, 질문 WAV에 믹싱하면 질문이 끝나는 순간 잡음도 끊겨 정작 판정 구간이 무잡음이 됨. 현재는 잡음을 질문 오디오와 분리해 같은 스피커로 별도 재생 (eval 코드 밖 운영 — 재생 자동화 코드 없음). 정밀한 디지털 SNR 통제는 포기하는 트레이드오프.
