# Memory Module

Long-term episodic memory and user profile management. Extracts memories from conversations, stores them in SQLite with vector/BM25 search, and retrieves relevant episodes for LLM context injection.

Design rationale: `docs/ray-memory/01-05`.


## Data Flow

```
[During session]
Orchestrator._save_utterance()
  -> storage.add_utterance(role, text, timestamp, token_count)

SpeechGenerator (per turn)
  -> retriever.retrieve(query, exclude_session_ids)
       vector search (cosine) + BM25 (FTS5) -> RRF -> salience ranking
       retained buffer management
  -> ContextBuilder.build(current_text, memory_result)  [Block 4]
  -> LLM generates response with [MEMORIES: M1, M2] tag
  -> parse citations -> retriever.update_citations([1, 2])

[Session end]
on_session_end callback -> write_executor.submit(...)
  -> MemoryWriter.process_session(session_id, started_at)
       1. Episode extraction      (LLM, per window)
       2. Cross-window dedup      (embedding + LLM, sequential)
       3. Store episodes          (DB + batch embedding + vector index)
       4. Profile fact extraction  (LLM)
       5. Profile merge           (LLM, APPEND/UPDATE/ABORT)
       6. Mark session processed
```


## Components

| File | Role |
|------|------|
| `types.py` | `Episode`, `Profile`, `MemoryReadResult` dataclasses |
| `storage.py` | `SQLiteMemoryStorage` (production), `InMemoryMemoryStorage` (test) |
| `vector_index.py` | `NumpyVectorIndex` -- exact cosine search, numpy matrix, < 10k vectors |
| `retriever.py` | `MemoryRetriever` -- hybrid search, RRF fusion, salience ranking, retained buffer |
| `writer.py` | `MemoryWriter` -- episode/profile extraction pipeline (3-4 LLM calls per session) |
| `prompts.py` | LLM prompts + JSON schemas for extraction, merge, dedup |
| `exceptions.py` | `MemoryStorageError`, `MemoryWriteError` |


## Storage Schema

Database: `data/ray.db` (shared with conversation history, separate connection).

```
episodes           -- episodic memories
  id, text, timestamp, session_id, importance, last_cited_at,
  citation_count, embedding (BLOB)

profiles           -- user profile slots (topic::sub_topic -> content)
  id, topic, sub_topic, content, updated_at

utterances         -- raw conversation text for extraction
  id, session_id, role, text, timestamp, token_count

processed_sessions -- tracks which sessions have been extracted
  session_id (PK), processed_at

episodes_fts       -- FTS5 virtual table on episodes.text (auto-synced via triggers)
```

WAL mode, `threading.Lock` for connection serialization.


## Retrieval Pipeline

```
query (current STT + recent turns)
  -> embed query
  -> vector search (top 20) + BM25 search (top 20)
  -> RRF fusion: score = 1/(k + rank + 1), k=60
  -> salience = rrf_score * recency_decay * importance
       recency_decay = exp(-ln(2) * days / 30)
  -> retained buffer: cited memories protected for N turns (TTL=3)
  -> slot allocation: max 10 total, min 4 new
  -> MemoryReadResult(episodes, scores, index_to_id)
```


## Write Pipeline

Triggered asynchronously via `write_executor` (single-threaded) after session ends.

- **Min gate**: sessions with < 2 utterances are skipped (not marked as processed)
- **Windowing**: sessions > 8000 tokens split with 25% overlap
- **Cross-window dedup**: embedding cosine similarity > 0.8 triggers LLM judgment (MERGE / KEEP_BOTH / DISCARD). Sequential processing -- each candidate compares against updated result embeddings
- **LLM model**: `gpt-4o-mini` (temperature=0.0, max_tokens=4096)
- **Importance**: fixed at 1.0 (calibration deferred to real usage data)

### Profile Schema

```
basic_info     :: name, age, location, occupation, language
interest       :: movie, music, book, game, food, sport, hobby
personality    :: traits, values, communication_style
interaction_style :: tone_preference, topic_preference, humor_style
```

LLM may create new sub_topics within existing topics.


## Lifecycle & Threading

**Process-level singletons** (survive across sessions):
- `SQLiteMemoryStorage`, `NumpyVectorIndex`, embedder, `MemoryWriter`, `write_executor`

**Session-level** (created per session in factory):
- `MemoryRetriever` (retained buffer is session-scoped)

**Thread access**:
- Main thread (Orchestrator): `add_utterance`
- Background thread (SpeechGenerator): `retrieve`, `update_citations`
- Write thread (write_executor): `process_session` -> all storage/index writes

All three access `SQLiteMemoryStorage` and `NumpyVectorIndex` concurrently, guarded by their internal locks.


## `SQLiteMemoryStorage.__init__` 인자

| 인자 | Default | 의미 |
|---|---|---|
| `db_path` | (필수) | SQLite 파일 경로. WAL recovery + FTS5 초기화 수행. |
| `dimension` | `384` | episode embedding 벡터 차원 (`load_all_embeddings` shape 검증). keyword-only. |

## `InMemoryMemoryStorage.__init__` 인자

| 인자 | Default | 의미 |
|---|---|---|
| `dimension` | `384` | 테스트용 embedding 차원. |

## `MemoryRetriever.__init__` 인자

| 인자 | Default | 의미 |
|---|---|---|
| `storage` | (필수) | IMemoryStorage 구현체. |
| `vector_index` | (필수) | IVectorIndex 구현체. |
| `embedder` | (필수) | IEmbedder 구현체 (query embedding). |

### `MemoryRetriever` 클래스 변수

| 변수 | 값 | 의미 |
|---|---|---|
| `_MAX_MEMORIES` | `10` | 턴당 Block 4 주입 최대 에피소드 수. |
| `_MIN_NEW_SLOTS` | `4` | 새 검색 결과에 확보할 최소 슬롯 수. |
| `_RETAINED_TTL` | `3` | 인용된 메모리가 retained 버퍼에 머무는 턴 수. |
| `_VECTOR_TOP_K` | `20` | 벡터 검색 후보 수. |
| `_BM25_TOP_K` | `20` | BM25 검색 후보 수. |
| `_RRF_K` | `60` | RRF 융합 상수 (원 논문 default). |
| `_RECENCY_HALF_LIFE_DAYS` | `30.0` | 시간 감쇠 반감기 (일). |
| `_SALIENCE_THRESHOLD` | `0.0` | salience 최소 기준 (`0.0` = 비활성화). |

## `MemoryWriter.__init__` 인자

| 인자 | Default | 의미 |
|---|---|---|
| `storage` | (필수) | IMemoryStorage. |
| `vector_index` | (필수) | IVectorIndex. |
| `embedder` | (필수) | IEmbedder. |
| `llm` | (필수) | ILLM (에피소드/프로필 추출용). |
| `token_counter` | (필수) | `Callable[[str], int]`. 프로필 슬롯 토큰 측정. |

### `MemoryWriter` 클래스 변수

| 변수 | 값 | 의미 |
|---|---|---|
| `_MIN_UTTERANCES` | `2` | 에피소드 추출에 필요한 최소 utterance 수. |
| `_WRITE_MAX_INPUT_TOKENS` | `8000` | 에피소드 추출 윈도우 최대 토큰 수 (초과 시 분할). |
| `_WRITE_WINDOW_OVERLAP_RATIO` | `0.25` | 인접 윈도우 overlap 비율. |
| `_WRITE_DEDUP_THRESHOLD` | `0.8` | 중복 판정 코사인 유사도 임계값. |
| `_PROFILE_MAX_CONTENT_TOKENS` | `128` | 프로필 슬롯 content 최대 토큰 수. |
| `_PROFILE_CONTENT_WARN_RATIO` | `0.7` | content 토큰이 예산의 몇 배를 넘으면 요약 경고 표시. |

## 모듈 상수

### `memory/storage.py`
| 변수 | 값 | 의미 |
|---|---|---|
| `_DEFAULT_DIMENSION` | `384` | 기본 embedding 차원 (all-MiniLM-L6-v2 기준). |
| `_DEFAULT_DB_PATH` | `"data/ray.db"` | 기본 SQLite 파일 경로 (History/Trace와 공유). |

### `memory/prompts.py`
| 변수 | 값 | 의미 |
|---|---|---|
| `PROFILE_SCHEMA` | dict | topic/subtopic enum (LLM prompt 계약). |

Context budget (class vars on `ContextBuilder`, see `voice_pipeline/context/README.md`):

| Variable | Default | Description |
|----------|---------|-------------|
| `_MAX_MEMORY_TOKENS` | `512` | Block 4 token budget |
| `_MAX_PROFILE_TOKENS` | `256` | Block 2 token budget |
| `_MAX_PREV_SESSION_TOKENS` | `512` | Block 3 token budget |
| `_MAX_CONTEXT_TOKENS` | `4096` | Total LLM input token budget |
