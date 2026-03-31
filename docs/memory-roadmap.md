# 장기기억 구현 계획

설계 문서(`ray-memory/01~05`)를 구현으로 옮기기 위한 단계별 작업 분배.
각 Phase는 별도 세션에서 진행하며, Phase 내 세부 결정은 구현 시점에 한다.

---

## Phase 의존 관계

```
Phase 1 (Storage) ← Phase 2 (Read)  ← Phase 4 (Integration)
Phase 1 (Storage) ← Phase 3 (Write) ← Phase 4 (Integration)
```

Phase 2와 3은 서로 독립. Phase 4는 2, 3 완료 후 진행.

---

## Phase 1: Storage Layer

메모리 시스템의 영속 저장소와 벡터 인덱스.

**범위**:
- 메모리 DB 스키마 (episodes, profiles, FTS5)
- 벡터 인덱스 (numpy in-memory, SQLite BLOB 영속)
- 임베딩 생성 래퍼
- 스토리지 인터페이스 + 구현

**산출물**: 에피소드/프로필 CRUD, 벡터+BM25 검색이 동작하는 저장소 모듈 + 테스트

---

## Phase 2: Memory Read

대화 중 매 턴 실행되는 검색 파이프라인.

**범위**:
- 쿼리 구성 (STT + 직전 N턴)
- 벡터 + BM25 병렬 검색 → RRF → Salience 랭킹 → 필터링
- Retained Buffer (TTL 기반 인용 기억 보호)
- 프로필 Read (전체 로드)

**산출물**: 쿼리를 받아 ranked 에피소드 리스트 + 프로필을 반환하는 모듈 + 테스트

---

## Phase 3: Memory Write

세션 종료 후 비동기 배치 처리.

**범위**:
- 에피소드 추출 (LLM 기반, 윈도우 처리, importance 판정)
- 프로필 Merge (APPEND/UPDATE/ABORT)
- 임베딩 생성 + 인덱스 갱신

**산출물**: 세션 원문을 받아 에피소드/프로필을 저장하는 모듈 + 테스트

---

## Phase 4: Integration

기존 파이프라인과 연결.

**범위**:
- ContextBuilder 확장 (블록 2: 프로필, 블록 4: 기억, 토큰 예산)
- 기억 인용 파싱 (`[MEMORIES: ...]` → recency 갱신, TTS 전 제거)
- 세션 라이프사이클 훅 (시작 시 로드, 종료 시 Write 트리거)
- 프롬프트 업데이트
