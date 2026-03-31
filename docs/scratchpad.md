# Phase 2 (Memory Read) 사전 논의 결과

논의일: 2026-03-31. 구현 계획 작성 전 결정된 사항.

---

## 모듈 구조

- 파일: `memory/retriever.py`, 클래스: `MemoryRetriever`
- 인터페이스: `IMemoryRetriever` → `core/interfaces.py`
- 역할: 쿼리를 받아 관련 에피소드를 검색·랭킹하고, 턴 간 retained buffer를 관리하여 블록 4에 주입할 기억 목록을 반환
- Retained Buffer는 retriever 내부 상태로 관리

## 시그니처

- `retrieve(query: str) → MemoryReadResult`
- 쿼리 구성(STT + N턴 concat)은 Phase 4 통합에서 처리. retriever는 검색/랭킹/필터링 전담
- 쿼리 구성과 검색은 별개 책임 — retriever가 IConversationHistory를 알 필요 없음

## 파라미터 (MemoryConfig 확장)

| 파라미터 | 초기값 | 비고 |
|----------|--------|------|
| max_memories | 10 | 블록 4 주입 총 에피소드 수 |
| min_new_slots | 4 | 신규 검색 결과 최소 보장, retained 상한 = 6 |
| retained_ttl | 3 | 인용된 기억 보호 턴 수 |
| vector_top_k | 20 | 벡터 검색 후보 수 |
| bm25_top_k | 20 | BM25 검색 후보 수 |
| rrf_k | 60 | RRF 상수 (원 논문 기본값) |
| recency_half_life_days | 30 | 지수 감쇠 반감기 |
| salience_threshold | 0.0 | 비활성 (importance 확정 후 도입) |

## Salience 공식

```
salience = similarity(RRF) × recency_decay × importance
recency_decay = exp(-0.693 × days / half_life)
```

- reinforcement 제외
- 기억 인용 시 last_cited_at 리프레시로 동적 피드백
- importance는 Write(Phase 3) 시 LLM 판정, 스케일 미확정

## 슬롯 배분

- retained 우선 배치 (상한 6), 나머지 신규로 채움
- 신규 최소 4슬롯 상시 보장
- retained가 상한 초과 시 salience 낮은 것부터 evict

## 필터링

- salience 임계값 필터: 초기 비활성 (0.0)
- 메타데이터 필터: 현재 세션 에피소드 제외
- top-K로만 제한

## 참고 프로젝트 (설계 문서 03-read.md에 기재)

- Hindsight: 벡터+BM25 → RRF(k=60), cross-encoder(확장 옵션)
- MemU: salience = similarity × recency_decay × reinforcement → importance로 대체
- EverMemOS: Foresight(확장 옵션)
