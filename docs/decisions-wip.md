# Decision Log — Work in Progress

진행 중인 작업의 결정 기록. 작업 완료 후 정리하여 `decisions.md`에 통합.


## Conversation History Redesign

- **Write-through SQLite**: 매 메시지 즉시 INSERT. 세션 중 crash 시 최대 진행 중 turn 1개만 유실 (batch-at-end였으면 전체 세션 유실). `save()`는 `ended_at` 설정 + WAL checkpoint만 수행.
- **Graduated DB corruption recovery**: 정상 open → WAL 파일 삭제 재시도 → corrupt 파일 백업 + 신규 DB 생성. RPi 전원 차단 시 대부분 WAL만 손상되므로 WAL 삭제만으로 복구되는 경우가 많음.
- **Responses API format 직접 저장**: 중간 canonical format 없이 vendor-specific dict 그대로 `item_json`에 저장. Vendor 교체 시 migration script 필요 (의도적 트레이드오프).
- **`token_count` 이중 소스**: assistant 메시지는 API `output_tokens` (정확값), user/truncated 메시지는 tiktoken fallback. ContextBuilder는 저장된 값을 읽어 re-tokenization 없이 예산 계산.
- **Tool definition token cost 실측**: tiktoken으로 definition structure를 추정하면 부정확. API `input_tokens`를 tool 유무로 비교하여 실측 (`web_search` = 294 tokens). ContextBuilder가 예산에서 차감.
- **Turn-level atomic budgeting**: ContextBuilder가 `get_turns()`로 turn_id 기준 그룹 단위로 포함/제외. Tool call + result + assistant text는 분리 불가 — 한 turn이 예산 초과하면 통째로 제외.


## Memory Storage Layer (Phase 1)

- **단일 DB**: 기존 conversation history와 같은 `data/ray.db` 사용. 별도 `sqlite3.Connection`으로 접속 (WAL 모드에서 안전). DB 분리는 세션 참조 cross-join 불가, 관리 이중화 문제로 불리.
- **utterances 별도 테이블**: 기존 `messages`(Responses API JSON 형식)와 별도로 `utterances`(role, text, timestamp 평문)를 둠. messages는 LLM 컨텍스트 재구성용, utterances는 메모리 Write 시 에피소드 추출용. messages에서 파생 가능하지만, tool_call 등 비텍스트 항목 필터링이 필요하고 용도가 다르므로 분리.
- **FTS5 쿼리 sanitize**: ASR 텍스트가 검색 쿼리로 들어오므로 `-`, `"`, `NOT` 등 FTS5 특수문자가 연산자로 해석되면 silent fail. 각 토큰을 `""`로 감싸서 리터럴 처리. Prefix 검색(`*`)도 차단되지만, Phase 2 검색 전략에서 필요 시 별도 쿼리 모드로 대응.
- **IEmbedder를 공유 모듈로 분리**: similarity와 memory가 동일 모델을 독립 로드하고 있어, `IEmbedder`를 `core/interfaces.py`로 승격하고 구현체를 `embedding/` 모듈로 추출. `IVectorIndex`는 memory 내부 전용이므로 그대로 유지.
- **실패 시 `None` 반환**: `add_episode`/`upsert_profile`이 DB 에러 시 `-1` 대신 `None`을 반환. `-1`은 다운스트림에서 유효한 ID로 오인될 위험 있음.


## Memory Retriever (Phase 2)

- **`exclude_session_ids`를 `retrieve()` 파라미터로**: 생성자가 아닌 매 턴 호출 시 전달. 세션이 길어지면 히스토리 블록 토큰 제한으로 이전 세션 요약이 빠질 수 있어, 컨텍스트 상태를 호출자가 반영해야 함. retriever가 session_id를 직접 관리할 필요 없어져서 더 단순해짐.
- **Retained overflow eviction 기준: TTL 우선**: salience(현재 턴 쿼리 기반)가 아닌 TTL 기준으로 evict. retained buffer는 토픽 전환 시에도 인용된 기억을 보호하는 장치인데, 현재 쿼리 salience로 evict하면 토픽 전환 시 보호가 무력화됨. 동일 TTL 내에서는 저장된 salience(진입/갱신 시점 값)로 tiebreak.
- **프로필은 retriever 범위 밖**: 프로필은 세션 내 불변이므로 매 턴 검색할 필요 없음. Phase 4 통합에서 세션 시작 시 `storage.get_all_profiles()` 1회 호출로 처리.


## Memory Writer (Phase 3)

- **세션 종료 시 즉시 추출**: 기존 설계(히스토리에서 밀려날 때 처리)에서 변경. 에피소드를 세션 종료 시 바로 추출하고, 추출된 에피소드가 세션 요약 역할도 겸함. 별도 요약 LLM 호출 절약.
- **3단계 순차 LLM 호출**: 에피소드 추출 → 프로필 fact 추출 → 프로필 Merge. 에피소드 추출과 프로필 추출은 작업 성격이 다르고(서사 생성 vs fact 식별), 프로필 추출과 merge도 다름(fact 식별 vs 기존 슬롯 대비 판정). 합치면 프롬프트 복잡도가 올라가고 실패 시 영향 범위가 커짐.
- **별도 LLMConfig(`write_llm`)**: 대화용 LLM과 모델·설정을 독립. 추출은 더 저렴한 모델(gpt-4o-mini), 낮은 temperature(0.0), 높은 max_tokens(4096), 도구 없음. 모델별 파라미터(reasoning_effort 등)가 다를 수 있어 필드 몇 개 추가보다 LLMConfig 통째 분리가 유연.
- **`ILLM.generate()`에 `response_format` 추가**: structured output(JSON schema)을 호출 시점에 지정. config가 아닌 파라미터인 이유: 에피소드/프로필/merge 각각 스키마가 다름. 기존 대화 호출은 None(기본값)으로 영향 없음.
- **프로필 Merge — `(topic, sub_topic)` 키 매칭**: 기존 슬롯에 P1/P2 인덱스를 매기는 대신, fact별로 기존 내용을 인라인 표시. LLM이 APPEND/UPDATE/ABORT만 판정하고, 코드에서 `(topic, sub_topic)` 키로 기존 슬롯을 매칭. 슬롯 순서 무관, `get_all_profiles()` 반환 순서에 비의존.
- **토큰 제한 — 조건부 경고**: 프로필 슬롯의 토큰 수를 `token_counter`로 측정, `profile_max_content_tokens`의 70% 초과 시 해당 fact에만 제한 지시 추가. 모든 슬롯에 무조건 제한을 명시하면 불필요한 압축을 유발.
- **importance 고정값(1.0)**: LLM이 0~1 연속값을 일관되게 매기기 어렵고, 기준 확정에 실사용 데이터가 필요. 데이터 쌓인 후 범주형(high/medium/low) 또는 reinforcement(인용 횟수) 기반으로 전환 검토. `citation_count` 필드는 미래용으로 추가해둠.
- **윈도우 처리 — 에피소드 추출만 분할**: 프로필 추출은 에피소드(이미 압축된 입력)를 받으므로 원문보다 훨씬 짧아 윈도우 불필요. 에피소드 추출도 기본은 세션 전체 1회 처리, `write_max_input_tokens` 초과 시에만 턴 단위 분할 + 겹침.
- **utterance에 `token_count` 저장**: 윈도우 분할 시 재토큰화 없이 저장된 값을 합산. Phase 4에서 orchestrator가 utterance 저장 시 이미 계산된 값을 전달.


## Memory Integration (Phase 4)

- **`on_session_end` 콜백 패턴**: SessionManager가 MemoryWriter를 직접 의존하지 않도록 `Callable[[str, str], None]` 콜백으로 분리. `__main__.py`에서 `write_executor.submit(memory_writer.process_session, ...)` 클로저를 주입. SessionManager는 메모리 시스템 존재 여부를 모름.
- **session_id를 factory에서 생성**: 기존 SessionManager에서 uuid를 생성했으나, factory에서 생성하도록 변경. factory가 profiles/previous sessions 로딩 시 현재 session_id를 exclude 조건에 사용해야 하므로, factory가 생성하고 `SessionComponents.session_id`로 반환.
- **Utterance 저장 위치: Orchestrator**: history 저장과 동일 시점에 `memory_storage.add_utterance()` 호출. SpeechGenerator나 ContextBuilder가 아닌 Orchestrator에서 수행하는 이유: 최종 확정 텍스트만 저장을 보장 (barge-in truncation 반영).
- **NumpyVectorIndex / SQLiteMemoryStorage에 threading.Lock 추가**: process-level 싱글턴이 3개 쓰레드(메인, SpeechGenerator 백그라운드, write_executor)에서 접근. NumpyVectorIndex는 `_ids`/`_matrix` 동시 읽기/쓰기 보호, SQLiteMemoryStorage는 단일 커넥션 직렬화.
- **임베딩 인스턴스 공유 결정**: similarity와 memory가 같은 모델을 사용하지만, 현재는 별도 인스턴스로 유지. similarity는 `SimilarityConfig`로 생성, memory embedder는 `MemoryConfig`로 생성. 모델이 동일해도 config 경로가 달라 강제 공유 시 config 의존성이 복잡해짐. 메모리 사용량이 문제되면 추후 통합.
- **이전 세션 요약 = 에피소드 그대로 사용**: `get_episodes_by_session_ids()`로 이전 세션 에피소드를 로드하여 `format_session_summary_block()`으로 포맷. 별도 요약 LLM 호출 없음 (Phase 3 결정 확정).
- **MemoryRetriever 매 세션 새로 생성**: retained buffer를 세션 간 격리하기 위해 factory에서 매번 생성. process-level 싱글턴인 memory_storage/vector_index/embedder를 주입받으므로 생성 비용 낮음.


### 차후 고려

- **SimilarityConfig/MemoryConfig 임베딩 필드 중복**: 양쪽 config에 model, use_onnx 등이 중복 존재. 공유 EmbeddingConfig 추출 여부는 실제 사용 패턴 보고 판단.
- **similarity.compare() 임베딩 캐싱**: TurnDetector 호출 패턴에서 `a`(이전 텍스트)가 반복됨. 한쪽 임베딩을 캐싱하면 추론 비용 절반 가능. 기존 코드도 동일 패턴이라 regression은 아님.
- **similarity 유닛 테스트 부재**: EmbeddingSimilarity, DiffLibSimilarity, create_similarity 팩토리에 대한 유닛 테스트가 없음. 현재는 TurnDetector 테스트에서 ISimilarity를 mock하여 간접 검증.


## Sentence Streaming (Phase 4-5)

- **영어 기준 문장 경계 감지**: `.` `!` `?` + 뒤따르는 공백으로 판단. 약어(`Mr.`, `Dr.`, `etc.`)와 단일 대문자 이니셜(`U.S.`)은 frozenset 기반 휴리스틱으로 제외. NLP 토크나이저 대비 의존성 없고 지연 없음. 한국어 등 다른 언어 지원 시 별도 전략 필요.
- **`min_flush_words = 4`**: 문장 경계가 감지되어도 누적 단어 수가 임계값 미만이면 다음 문장과 합쳐서 TTS. `"Sure!"` 같은 짧은 감탄사가 단독 TTS 호출되면 HTTP 오버헤드가 실제 합성 시간을 초과. 대부분의 LLM 첫 문장은 5~15단어이므로 즉시 발송됨.
- **TTS 파이프라이닝 (producer-consumer)**: LLM 스트리밍 + 문장 감지(producer)와 TTS 결과 drain(consumer)을 별도 스레드로 분리. 문장 N의 TTS 진행 중 문장 N+1의 TTS를 동시 시작하여 문장 간 gap 최소화. 로컬 `ThreadPoolExecutor(max_workers=2)`를 파이프라인 실행마다 생성/소멸하여 기존 `self._executor`(prepare 재시작용)와 독립.
- **인용 태그 처리**: `[MEMORIES: M1, M3]`는 문장 종결 부호가 없으므로 SentenceDetector 버퍼에 자연스럽게 잔류. LLM 스트림 종료 시 `flush()` → `parse_citation_tag()` 순으로 처리. 중간 문장에 태그가 섞일 가능성 없음.
- **`_text` 누적 갱신**: consumer가 각 문장의 TTS 오디오를 큐에 적재 완료한 후 `self._text`를 갱신. `get_text()`가 반환하는 텍스트는 항상 오디오가 생산된 범위와 일치하여 barge-in truncation 정확도가 full 모드 대비 개선됨.
- **Orchestrator/CppBridge 변경 없음**: 기존 `send_stream_start → send_audio(chunk)* → send_audio_end` 프로토콜이 chunk 출처와 무관하게 동작. C++ 측은 버퍼에 쌓이는 PCM을 360ms 단위로 drain하므로 TTS 호출 횟수를 알 필요 없음.
- **Barge-in 호환성**: PLAYING 중 interrupt 시 generator를 cancel하지 않는 기존 동작 그대로 유지. consumer가 계속 실행되어 `stream_done = True` 설정 → deferred truncation 패턴이 동일하게 적용. 다음 `prepare()` 호출 시 `cancel_event`가 문장 간 체크에서 잡혀 불필요한 TTS는 최대 진행 중 1회로 제한.
