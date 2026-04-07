# Decision Log

비자명한 설계 결정과 시행착오 기록. 코드에서 바로 읽을 수 있는 구현 상세는 제외.


## ASR

- **Google Cloud STT V1**: V2는 batch/adaptation 등 불필요한 기능만 추가. 실시간 스트리밍 용도에는 V1으로 충분.


## LLM

- **OpenAI Responses API**: System message는 `instructions` param으로 전달, `input`에 넣지 않음. `previous_response_id`는 사용하지 않음 — ContextBuilder에서 토큰 예산 기반으로 직접 컨텍스트를 관리.


## TTS

- **WAV header quirk**: OpenAI TTS가 `n_frames = INT_MAX`인 malformed WAV를 반환하는 경우 있음. ASR 등 헤더 검증하는 소비자에 넘길 때는 ffmpeg re-encode 필요.


## TurnGPT

- **KV cache 활용**: ASR이 incremental update(같은 prefix, 늘어나는 suffix)를 보내므로 매번 전체 dialog를 재처리하면 낭비. 토큰 prefix 비교 후 새 토큰만 forward. 동일 입력이면 캐시된 확률 반환.
- **Context window eviction**: max 1024 tokens(GPT-2 한계). 초과 시 oldest turn을 `<ts>` 기준으로 삭제, 80% 이하까지. Eviction은 KV cache 전체 무효화 (full rebuild). 80% headroom은 limit 근처에서 thrashing 방지.
- **ONNX backend**: RPi 추론 성능용. PyTorch는 여전히 필요 (토큰화에 torch tensor 사용) — 목표는 속도 개선이지 PyTorch 제거가 아님. `onnx_model_path` 설정 여부로 backend 선택. KV cache 지원은 ONNX input name에 `past_key_0` 존재 여부로 감지.
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
- **`robot_audio=None` 시 unconditional interrupt 제거**: ROBOT_TURN에서 robot audio 없을 때(generation gap, PLAYBACK_STARTED 전) `user_is_speaking=True`만으로 interrupt 하던 것을 제거. 원인: 1–3s generation gap 동안 주변 소음이 false interrupt 유발 (실측: 7회 LLM 호출, 4.7s 지연). VAP는 robot 채널 없이 interrupt/backchannel 구분 불가. 수정: `TurnDecision.none()` 반환. 사용자 추가 발화는 orchestrator의 ASR text change cancel (0.5s grace) 로 처리.
- **`user_is_speaking` 전제조건**: 논문 pseudocode (Appendix A lines 56-61) 요구. 없으면 turn-shift 직후 VAP context가 아직 user-biased일 때 transient spike로 false interrupt 발생.
- **VAP error default `VAPResult(0, 0, False)`**: "robot favored"처럼 보이지만, transient error는 500ms 지속 안 되므로 Path 1 false trigger 없음. Persistent failure 시 Path 2(TurnGPT + silence)가 독립 작동.


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
- **`_text` 누적 갱신**: consumer가 각 문장의 TTS 오디오를 큐에 적재 완료한 후 `self._text`를 갱신. `get_text()`가 반환하는 텍스트는 항상 오디오가 생산된 범위와 일치하여 barge-in truncation 정확도가 full 모드 대비 개선됨.
- **Sentence streaming Orchestrator/CppBridge 변경 없음**: 기존 `send_stream_start → send_audio(chunk)* → send_audio_end` 프로토콜이 chunk 출처와 무관하게 동작. C++는 버퍼에 쌓이는 PCM을 360ms 단위로 drain하므로 TTS 호출 횟수를 알 필요 없음.
- **Sentence streaming barge-in 호환성**: PLAYING 중 interrupt 시 generator를 cancel하지 않는 기존 동작 그대로 유지. consumer가 계속 실행되어 `stream_done = True` 설정 → deferred truncation 패턴이 동일하게 적용. 다음 `prepare()` 호출 시 `cancel_event`가 문장 간 체크에서 잡혀 불필요한 TTS는 최대 진행 중 1회로 제한.


## C++ ↔ Python Protocol

### WebSocket topology
- **C++ server, Python client**: C++가 `ix::WebSocketServer` port 8765. Python이 `websockets`로 접속. 단일 연결 — `g_client_ws`에 저장.

### Protocol 단순화
- **`turn_id` 제거**: WebSocket TCP 순서 보장 + Python 상태 머신(playback_complete 대기 후 다음 turn)으로 stale chunk 오염 방지. C++는 `stream_start` 시 버퍼 클리어.
- **`playback_stopped` 제거 → `playback_complete`로 통합**: Python이 자체 `STOP_PENDING` 상태로 정상 완료/barge-in 구분. C++는 항상 `playback_complete`만 전송.
- **`playback_position` 스트림 제거**: `stop_pos = stop_pending_time - playback_start_time`으로 시간 기반 추정. Localhost에서 ±100ms 정확도 허용.
- **메시지 통합**: `stream_start` ← `responses_only` + `responses_stream_start`. `audio_end` ← `responses_stream_end`. `play_file` ← `play_audio` + `send_greeting` + `send_farewell`. `stop` ← `user_interruption`.
- **`playback_started` (신규)**: `control_motor` cycle 0에서 `soundStream.play()` 후 전송. Barge-in 위치 추정 및 VAP robot audio 타이밍 기준점.
- **Interrupt 시 `playback_complete` 전송**: C++가 interrupt cleanup 후에도 `playback_complete` 전송. Python의 STOP_PENDING 상태가 정상 해소됨.


## Session Lifecycle

- **Session factory 도입**: 이전 설계는 단일 Orchestrator/ConversationHistory를 재사용하며 `reset()`으로 상태 정리. 그러나 TurnDetector의 `_dialog_parts`가 `reset()`으로 안 지워져서 세션 간 상태 누수 발생. 수정: `session_factory`가 매 세션마다 Orchestrator, TurnDetector, SpeechGenerator, ContextBuilder, ConversationHistory를 새로 생성.
- **Three-tier lifecycle**: (1) Process-level — 모델, API 클라이언트, 하드웨어, executor (비싼 초기화, 1회). (2) Session-level — 상태 있는 orchestration 객체 (factory 재생성). (3) Turn-level — 경량 `reset()` (ASR 버퍼, TurnDetector 프레임 카운터).


## Similarity

- **Sentence embedding > SequenceMatcher**: SequenceMatcher는 문자 겹침 측정 — `"what is your"` vs `"what is your name"`이 0.87 (0.8 threshold에 차단됨). Sentence embedding (all-MiniLM-L6-v2)은 0.66 (정상 통과). 논문이 semantic similarity 사용.
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

- **`on_session_end` 콜백 패턴**: SessionManager가 MemoryWriter를 직접 의존하지 않도록 `Callable[[str, str], None]` 콜백으로 분리. `__main__.py`에서 `write_executor.submit(memory_writer.process_session, ...)` 클로저를 주입. SessionManager는 메모리 시스템 존재 여부를 모름.
- **session_id를 factory에서 생성**: 기존 SessionManager에서 uuid를 생성했으나, factory에서 생성하도록 변경. factory가 profiles/previous sessions 로딩 시 현재 session_id를 exclude 조건에 사용해야 하므로, factory가 생성하고 `SessionComponents.session_id`로 반환.
- **Utterance 저장 위치: Orchestrator**: history 저장과 동일 시점에 `memory_storage.add_utterance()` 호출. SpeechGenerator나 ContextBuilder가 아닌 Orchestrator에서 수행하는 이유: 최종 확정 텍스트만 저장을 보장 (barge-in truncation 반영).
- **NumpyVectorIndex / SQLiteMemoryStorage에 threading.Lock 추가**: process-level 싱글턴이 3개 쓰레드(메인, SpeechGenerator 백그라운드, write_executor)에서 접근. NumpyVectorIndex는 `_ids`/`_matrix` 동시 읽기/쓰기 보호, SQLiteMemoryStorage는 단일 커넥션 직렬화.
- **임베딩 인스턴스 공유 결정**: similarity와 memory가 같은 모델을 사용하지만, 현재는 별도 인스턴스로 유지. similarity는 `SimilarityConfig`로 생성, memory embedder는 `MemoryConfig`로 생성. 모델이 동일해도 config 경로가 달라 강제 공유 시 config 의존성이 복잡해짐. 메모리 사용량이 문제되면 추후 통합.
- **이전 세션 요약 = 에피소드 그대로 사용**: `get_episodes_by_session_ids()`로 이전 세션 에피소드를 로드하여 `format_session_summary_block()`으로 포맷. 별도 요약 LLM 호출 없음.
- **MemoryRetriever 매 세션 새로 생성**: retained buffer를 세션 간 격리하기 위해 factory에서 매번 생성. process-level 싱글턴인 memory_storage/vector_index/embedder를 주입받으므로 생성 비용 낮음.
