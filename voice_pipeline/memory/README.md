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


## Config

`MemoryConfig` in `core/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `db_path` | `data/ray.db` | SQLite database path |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `embedding_backend` | `local` | `local` or `api` |
| `embedding_dimension` | `384` | Vector dimension |
| `use_onnx` | `False` | ONNX Runtime for local model |
| `max_memories` | `10` | Max episodes in Block 4 per turn |
| `min_new_slots` | `4` | Reserved slots for new search results |
| `retained_ttl` | `3` | Turns to keep cited memories |
| `vector_top_k` | `20` | Vector search candidates |
| `bm25_top_k` | `20` | BM25 search candidates |
| `rrf_k` | `60` | RRF fusion constant |
| `recency_half_life_days` | `30.0` | Exponential decay half-life |
| `salience_threshold` | `0.0` | Min salience (0 = disabled) |
| `write_max_input_tokens` | `8000` | Max tokens per extraction window |
| `write_window_overlap_ratio` | `0.25` | Window overlap fraction |
| `write_dedup_threshold` | `0.8` | Cosine sim threshold for dedup |
| `profile_max_content_tokens` | `128` | Max tokens per profile slot |
| `profile_max_subtopics` | `20` | Max subtopics before reorg |

Context budget (in `ConversationHistoryConfig`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_memory_tokens` | `512` | Block 4 token budget |
| `max_profile_tokens` | `256` | Block 2 token budget |
| `max_prev_session_tokens` | `512` | Block 3 token budget |
