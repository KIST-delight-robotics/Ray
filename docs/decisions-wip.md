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


### 차후 고려

- **임베딩 인스턴스 공유**: 현재 similarity와 memory가 각자 embedder를 생성. 같은 모델이면 `__main__.py`에서 하나 만들어 양쪽에 주입하여 메모리 절약 가능. Phase 4 와이어링 시 결정.
- **SimilarityConfig/MemoryConfig 임베딩 필드 중복**: 양쪽 config에 model, use_onnx 등이 중복 존재. 공유 EmbeddingConfig 추출 여부는 와이어링 시 실제 사용 패턴 보고 판단.
- **similarity.compare() 임베딩 캐싱**: TurnDetector 호출 패턴에서 `a`(이전 텍스트)가 반복됨. 한쪽 임베딩을 캐싱하면 추론 비용 절반 가능. 기존 코드도 동일 패턴이라 regression은 아님.
- **similarity 유닛 테스트 부재**: EmbeddingSimilarity, DiffLibSimilarity, create_similarity 팩토리에 대한 유닛 테스트가 없음. 현재는 TurnDetector 테스트에서 ISimilarity를 mock하여 간접 검증.
