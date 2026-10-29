# 장기기억 1차

> 2026-03 ~ 2026-04 장기기억 1차 작업의 기록. 완료 후 고정 — 이후 변경은 반영하지 않는다.

## 개요

대화에서 에피소드와 사용자 프로필을 뽑아 SQLite(벡터·BM25 검색)에 저장하고, cascade 엔진이 관련 기억을 검색해 LLM
입력 블록으로 넣는 첫 장기기억 시스템. 저장소(Phase 1) → 검색(Phase 2, 하이브리드 검색 + retained buffer) →
추출(Phase 3, 에피소드·프로필) 순으로 구현하고 세션 루프·`__main__` 에 붙였다.

초기 설계 원문은 [archive/memory-v1-design/](../archive/memory-v1-design/) 에 있다. GPT-Live 엔진에 맞춘 이후의
변경은 그다음 장기기억 작업의 기록에 있다(이 기록을 고정한 시점에는 `decisions-wip.md` 에서 진행 중).


## 저장소 (Memory Storage)

- **단일 DB**: 기존 conversation history와 같은 `data/ray.db` 사용. 별도 `sqlite3.Connection`으로 접속 (WAL 모드에서
  안전). DB 분리는 세션 참조 cross-join 불가, 관리 이중화 문제로 불리. (경로는 이후 저장소 폴더 정리로 `var/ray.db` —
  [repo-structure.md](repo-structure.md))
- **utterances 별도 테이블**: 기존 `messages`(Responses API JSON 형식)와 별도로 `utterances`(role, text, timestamp
  평문)를 둠. messages는 LLM 컨텍스트 재구성용, utterances는 메모리 Write 시 에피소드 추출용. messages에서 파생
  가능하지만, tool_call 등 비텍스트 항목 필터링이 필요하고 용도가 다르므로 분리.
- **FTS5 쿼리 sanitize**: ASR 텍스트가 검색 쿼리로 들어오므로 `-`, `"`, `NOT` 등 FTS5 특수문자가 연산자로 해석되면
  silent fail. 각 토큰을 `""`로 감싸서 리터럴 처리. Prefix 검색(`*`)도 차단되지만, 검색 전략에서 필요 시 별도 쿼리
  모드로 대응.
- **IEmbedder를 공유 모듈로 분리**: similarity와 memory가 동일 모델을 독립 로드하고 있어, `IEmbedder`를 공유
  인터페이스로 승격하고 구현체를 별도 모듈로 추출. `IVectorIndex`는 memory 내부 전용이므로 그대로 유지. (위치는 이후
  구조 정리로 `types.py` · `adapters/embedder.py` — [repo-structure.md](repo-structure.md))
- **실패 시 `None` 반환**: `add_episode`/`upsert_profile`이 DB 에러 시 `-1` 대신 `None`을 반환. `-1`은 다운스트림에서
  유효한 ID로 오인될 위험 있음.


## 검색 (Memory Retriever)

- **`exclude_session_ids`를 `retrieve()` 파라미터로**: 생성자가 아닌 매 턴 호출 시 전달. 세션이 길어지면 히스토리
  블록 토큰 제한으로 이전 세션 요약이 빠질 수 있어, 컨텍스트 상태를 호출자가 반영해야 함. retriever가 session_id를
  직접 관리할 필요 없어져서 더 단순해짐. (→ gpt_live 엔진은 이후 검색 제외를 쓰지 않게 됨 — 이후 장기기억 작업
  기록의 "장기기억 상시 컨텍스트 재설계")
- **Retained overflow eviction 기준: TTL 우선**: salience(현재 턴 쿼리 기반)가 아닌 TTL 기준으로 evict. retained
  buffer는 토픽 전환 시에도 인용된 기억을 보호하는 장치인데, 현재 쿼리 salience로 evict하면 토픽 전환 시 보호가
  무력화됨. 동일 TTL 내에서는 저장된 salience(진입/갱신 시점 값)로 tiebreak.
- **프로필은 retriever 범위 밖**: 프로필은 세션 내 불변이므로 매 턴 검색할 필요 없음. 세션 시작 시
  `storage.get_all_profiles()` 1회 호출로 처리. (→ 이후 프로필 슬롯은 사용자 노트로 대체 — 이후 장기기억 작업
  기록의 "장기기억 상시 컨텍스트 재설계")


## 추출 (Memory Writer)

- **세션 종료 시 즉시 추출**: 기존 설계(히스토리에서 밀려날 때 처리)에서 변경. 에피소드를 세션 종료 시 바로
  추출하고, 추출된 에피소드가 세션 요약 역할도 겸함. 별도 요약 LLM 호출 절약.
- **3단계 순차 LLM 호출**: 에피소드 추출 → 프로필 fact 추출 → 프로필 Merge. 에피소드 추출과 프로필 추출은 작업
  성격이 다르고(서사 생성 vs fact 식별), 프로필 추출과 merge도 다름(fact 식별 vs 기존 슬롯 대비 판정). 합치면
  프롬프트 복잡도가 올라가고 실패 시 영향 범위가 커짐. (→ 이후 프로필 두 단계는 사용자 노트 갱신 1회 호출로 대체 —
  이후 장기기억 작업 기록의 "장기기억 상시 컨텍스트 재설계")
- **별도 LLMConfig(`write_llm`)**: 대화용 LLM과 모델·설정을 독립. 추출은 더 저렴한 모델(gpt-4o-mini), 낮은
  temperature(0.0), 높은 max_tokens(4096), 도구 없음. 모델별 파라미터(reasoning_effort 등)가 다를 수 있어 필드 몇 개
  추가보다 LLMConfig 통째 분리가 유연.
- **`ILLM.generate()`에 `response_format` 추가**: structured output(JSON schema)을 호출 시점에 지정. config가 아닌
  파라미터인 이유: 에피소드/프로필/merge 각각 스키마가 다름. 기존 대화 호출은 None(기본값)으로 영향 없음.
- **프로필 Merge — `(topic, sub_topic)` 키 매칭**: 기존 슬롯에 P1/P2 인덱스를 매기는 대신, fact별로 기존 내용을
  인라인 표시. LLM이 APPEND/UPDATE/ABORT만 판정하고, 코드에서 `(topic, sub_topic)` 키로 기존 슬롯을 매칭. 슬롯 순서
  무관, `get_all_profiles()` 반환 순서에 비의존. (→ 이후 슬롯 대신 항목 목록 + 추가·교체·삭제 연산으로 대체 —
  이후 장기기억 작업 기록의 "장기기억 상시 컨텍스트 재설계")
- **토큰 제한 — 조건부 경고**: 프로필 슬롯의 토큰 수를 `token_counter`로 측정, `profile_max_content_tokens`의 70%
  초과 시 해당 fact에만 제한 지시 추가. 모든 슬롯에 무조건 제한을 명시하면 불필요한 압축을 유발.
- **importance 고정값(1.0)**: LLM이 0~1 연속값을 일관되게 매기기 어렵고, 기준 확정에 실사용 데이터가 필요. 데이터
  쌓인 후 범주형(high/medium/low) 또는 reinforcement(인용 횟수) 기반으로 전환 검토. `citation_count` 필드는
  미래용으로 추가해둠.
- **윈도우 처리 — 에피소드 추출만 분할**: 프로필 추출은 에피소드(이미 압축된 입력)를 받으므로 원문보다 훨씬 짧아
  윈도우 불필요. 에피소드 추출도 기본은 세션 전체 1회 처리, `write_max_input_tokens` 초과 시에만 턴 단위 분할 + 겹침.
- **utterance에 `token_count` 저장**: 윈도우 분할 시 재토큰화 없이 저장된 값을 합산. 세션 루프가 utterance 저장 시
  이미 계산된 값을 전달.


## 파이프라인 연결 (Memory Integration)

- **메모리 쓰기는 `__main__` 주도**: SessionLoop이 MemoryWriter를 직접 의존하지 않도록, 세션 종료 시
  `__main__.py`(FAREWELL 전환·종료 finally)에서 `write_executor.submit(memory_writer.process_session, ...)`를 직접
  호출. SessionLoop은 메모리 시스템 존재 여부를 모름.
- **session_id를 factory에서 생성**: factory가 profiles/previous sessions 로딩 시 현재 session_id를 exclude 조건에
  사용해야 하므로, 세션 루프가 아니라 factory가 생성하고 `SessionComponents.session_id`로 반환.
- **Utterance 저장 위치: SessionLoop**: history 저장과 동일 시점에 `memory_storage.add_utterance()` 호출.
  SpeechGenerator나 ContextBuilder가 아닌 SessionLoop에서 수행하는 이유: 최종 확정 텍스트만 저장을 보장 (barge-in
  truncation 반영).
- **NumpyVectorIndex / SQLiteMemoryStorage에 threading.Lock 추가**: process-level 싱글턴이 3개 쓰레드(메인,
  SpeechGenerator 백그라운드, write_executor)에서 접근. NumpyVectorIndex는 `_ids`/`_matrix` 동시 읽기/쓰기 보호,
  SQLiteMemoryStorage는 단일 커넥션 직렬화.
- **임베딩 인스턴스 공유**: process-level 임베더 1개를 `TrackedEmbedder`로 감싸 memory(retriever/writer)와
  TurnDetector 유사도 계산이 공유. 초기엔 config 경로 분리 문제로 별도 인스턴스였으나, ISimilarity 모듈 제거와 함께
  단일 인스턴스로 통합.
- **이전 세션 요약 = 에피소드 그대로 사용**: `get_episodes_by_session_ids()`로 이전 세션 에피소드를 로드하여
  `format_session_summary_block()`으로 포맷. 별도 요약 LLM 호출 없음. (→ 이후 cascade 는 직전 세션 원문 이월 —
  [archive/cascade/decisions.md](../archive/cascade/decisions.md), gpt_live 는 최근 대화 원문 — 이후 장기기억 작업
  기록의 "장기기억 상시 컨텍스트 재설계")
- **MemoryRetriever 매 세션 새로 생성**: retained buffer를 세션 간 격리하기 위해 factory에서 매번 생성.
  process-level 싱글턴인 memory_storage/vector_index/embedder를 주입받으므로 생성 비용 낮음.
