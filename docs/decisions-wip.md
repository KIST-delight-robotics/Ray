# Decision Log — Work in Progress

진행 중인 작업의 결정 기록. 작업 완료 후 정리하여 `decisions.md`에 통합.


## Conversation History Redesign

- **Write-through SQLite**: 매 메시지 즉시 INSERT. 세션 중 crash 시 최대 진행 중 turn 1개만 유실 (batch-at-end였으면 전체 세션 유실). `save()`는 `ended_at` 설정 + WAL checkpoint만 수행.
- **Graduated DB corruption recovery**: 정상 open → WAL 파일 삭제 재시도 → corrupt 파일 백업 + 신규 DB 생성. RPi 전원 차단 시 대부분 WAL만 손상되므로 WAL 삭제만으로 복구되는 경우가 많음.
- **Responses API format 직접 저장**: 중간 canonical format 없이 vendor-specific dict 그대로 `item_json`에 저장. Vendor 교체 시 migration script 필요 (의도적 트레이드오프).
- **`token_count` 이중 소스**: assistant 메시지는 API `output_tokens` (정확값), user/truncated 메시지는 tiktoken fallback. ContextBuilder는 저장된 값을 읽어 re-tokenization 없이 예산 계산.
- **Tool definition token cost 실측**: tiktoken으로 definition structure를 추정하면 부정확. API `input_tokens`를 tool 유무로 비교하여 실측 (`web_search` = 294 tokens). ContextBuilder가 예산에서 차감.
- **Turn-level atomic budgeting**: ContextBuilder가 `get_turns()`로 turn_id 기준 그룹 단위로 포함/제외. Tool call + result + assistant text는 분리 불가 — 한 turn이 예산 초과하면 통째로 제외.
