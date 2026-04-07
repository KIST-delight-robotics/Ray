# 저장소

---

## 1. SQLite

모든 영속 데이터의 단일 저장소.

### 에피소드 테이블

```
episodes:
  id: 정수 PK
  text: 내러티브 텍스트
  timestamp: 일시
  session_id: 세션 참조
  importance: 질적 중요도 (LLM 판정)
  last_cited_at: 마지막 기억 인용 일시 (recency_decay 기준, 초기값 = timestamp)
  citation_count: 인용 횟수 (향후 reinforcement 신호용, 초기값 = 0)
  embedding: 벡터 (BLOB)
```

### 프로필 테이블

```
profiles:
  id: 정수 PK
  topic: 상위 분류 (basic_info, interest, personality, interaction_style 등)
  sub_topic: 하위 분류
  content: 슬롯 내용
  updated_at: 마지막 갱신 일시
```

### 원문 테이블

```
utterances:
  id: 정수 PK
  session_id: 세션 참조
  role: speaker (user/assistant)
  text: 발화 텍스트
  timestamp: 일시
  token_count: 토큰 수 (윈도우 분할 시 재토큰화 없이 사용)
```

세션 메타데이터(시작/종료 일시 등)는 conversation history 모듈에서 관리한다.

### 처리 상태 테이블

```
processed_sessions:
  session_id: 텍스트 PK
  processed_at: 처리 완료 일시
```

메모리 추출(process_session)이 시도된 세션을 기록한다. 에피소드가 0건이어도 마킹. 세션 시작 시 이전 세션 표현 방식을 결정하는 데 사용 (02-session.md §3: 에피소드 없음 + 처리 완료 → 생략, 미처리 → 원문 포함).

### FTS5 인덱스

- 에피소드 text에 대해 FTS5 가상 테이블 구성 → BM25 검색에 사용
- 기본 언어는 영어이므로 기본 토크나이저로 동작. 한국어 등 다른 언어 지원 시 형태소 분석기(mecab 등) 또는 n-gram 토크나이저 필요

---

## 2. 벡터 인덱스

에피소드 임베딩을 RAM에 전체 로드하여 유사도 검색.

### 후보

| | numpy (전수 검색) | hnswlib (ANN) |
|---|---|---|
| 정확도 | 정확 | 근사 |
| 속도 | 건수 증가 시 선형 저하 | 건수 증가에 강함 |
| 영속성 | 별도 구현 필요 | ID 관리/영속성 내장 |

- 초기 건수가 적은 동안은 numpy로 충분, 건수 증가 시 hnswlib 전환 검토
- 서비스 시작 시 SQLite에서 임베딩을 읽어 RAM에 로드
- 벡터 용량: 384차원 float32 기준, 1만 건 ~15MB

---

## 3. 용량 추정

라즈베리파이 5 환경 기준.

- 에피소드: 세션당 평균 2~5건, 하루 수 세션 → 일 10~20건 수준
- 1년 기준 수천 건, 벡터 포함 수 MB 수준
- SQLite + RAM 벡터 인덱스로 Pi5에서 충분히 운용 가능

---

## 4. 확장 옵션

### 4-1. 멀티유저

- 현재는 단일유저 전제
- 멀티유저 시: 테이블에 user_id 컬럼 추가 + 검색 시 필터링
- 또는 유저별 별도 SQLite 파일 (유저 삭제 = 파일 삭제로 격리 가능)
