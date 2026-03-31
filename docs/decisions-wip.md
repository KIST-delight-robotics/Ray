# Decision Log — Work in Progress

진행 중인 작업의 결정 기록. 작업 완료 후 정리하여 `decisions.md`에 통합.


## Memory Storage Layer (Phase 1)

- **단일 DB**: 기존 conversation history와 같은 `data/ray.db` 사용. 별도 `sqlite3.Connection`으로 접속 (WAL 모드에서 안전). DB 분리는 세션 참조 cross-join 불가, 관리 이중화 문제로 불리.
- **utterances 별도 테이블**: 기존 `messages`(Responses API JSON 형식)와 별도로 `utterances`(role, text, timestamp 평문)를 둠. messages는 LLM 컨텍스트 재구성용, utterances는 메모리 Write 시 에피소드 추출용. messages에서 파생 가능하지만, tool_call 등 비텍스트 항목 필터링이 필요하고 용도가 다르므로 분리.
- **FTS5 쿼리 sanitize**: ASR 텍스트가 검색 쿼리로 들어오므로 `-`, `"`, `NOT` 등 FTS5 특수문자가 연산자로 해석되면 silent fail. 각 토큰을 `""`로 감싸서 리터럴 처리. Prefix 검색(`*`)도 차단되지만, Phase 2 검색 전략에서 필요 시 별도 쿼리 모드로 대응.
- **임베딩/벡터 인덱스 인터페이스 분리**: `IEmbedder`, `IVectorIndex`를 인터페이스로 정의하여 numpy→hnswlib, 모델 교체 등에 대비. cross-module 인터페이스가 아니므로 `core/interfaces.py`가 아닌 memory 모듈 내부에 배치.
- **실패 시 `None` 반환**: `add_episode`/`upsert_profile`이 DB 에러 시 `-1` 대신 `None`을 반환. `-1`은 다운스트림에서 유효한 ID로 오인될 위험 있음.


## Conversation History Redesign

- **Write-through SQLite**: 매 메시지 즉시 INSERT. 세션 중 crash 시 최대 진행 중 turn 1개만 유실 (batch-at-end였으면 전체 세션 유실). `save()`는 `ended_at` 설정 + WAL checkpoint만 수행.
- **Graduated DB corruption recovery**: 정상 open → WAL 파일 삭제 재시도 → corrupt 파일 백업 + 신규 DB 생성. RPi 전원 차단 시 대부분 WAL만 손상되므로 WAL 삭제만으로 복구되는 경우가 많음.
- **Responses API format 직접 저장**: 중간 canonical format 없이 vendor-specific dict 그대로 `item_json`에 저장. Vendor 교체 시 migration script 필요 (의도적 트레이드오프).
- **`token_count` 이중 소스**: assistant 메시지는 API `output_tokens` (정확값), user/truncated 메시지는 tiktoken fallback. ContextBuilder는 저장된 값을 읽어 re-tokenization 없이 예산 계산.
- **Tool definition token cost 실측**: tiktoken으로 definition structure를 추정하면 부정확. API `input_tokens`를 tool 유무로 비교하여 실측 (`web_search` = 294 tokens). ContextBuilder가 예산에서 차감.
- **Turn-level atomic budgeting**: ContextBuilder가 `get_turns()`로 turn_id 기준 그룹 단위로 포함/제외. Tool call + result + assistant text는 분리 불가 — 한 turn이 예산 초과하면 통째로 제외.
