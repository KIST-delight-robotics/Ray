# History Module

Session-scoped conversation history with write-through SQLite persistence.

## Storage Model

- **In-memory**: authoritative for reads during a session (`get_messages()`, `get_turns()`)
- **SQLite**: write-through on every mutation for crash safety
- Each message = one DB row. `turn_id` groups multi-message turns (tool calls).

## Schema

```sql
sessions (session_id TEXT PK, started_at TEXT, ended_at TEXT, summary TEXT)
messages (session_id, msg_id, turn_id, item_json, token_count, metrics_json, created_at)
```

- `item_json`: single message dict in OpenAI Responses API input format
- `token_count`: pre-computed context budget cost (LLM output_tokens or tiktoken)
- `metrics_json`: LLM call metadata (NULL for non-LLM messages)

## Config

### `ConversationHistoryConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_context_tokens` | `int` | `4096` | Token budget for ContextBuilder |
| `storage_backend` | `"memory" \| "sqlite"` | `"sqlite"` | Backend type |
| `storage_path` | `str` | `"data/ray.db"` | SQLite DB file path |

## Usage

```python
from voice_pipeline.history import ConversationHistory, MemoryStorageBackend

history = ConversationHistory(MemoryStorageBackend(), token_counter)
history.new_session("session-1")

# Simple messages
history.add_user_message("hello")
history.add_assistant_message("hi there", metrics=llm_metrics)

# Tool call turn
turn_id = history.begin_turn()
history.add_message(function_call_item, turn_id=turn_id, metrics=metrics1)
history.add_message(function_call_output_item, turn_id=turn_id)
history.add_message(assistant_item, turn_id=turn_id, metrics=metrics2)

# Read
messages = history.get_messages()  # flat list for LLM
turns = history.get_turns()        # grouped for ContextBuilder

# Barge-in correction
history.update_message(msg_id, "truncated text")

# Session end
history.save()  # sets ended_at
```

## SQLite Safety

- **WAL mode** + `synchronous=NORMAL`: concurrent reads, crash-safe writes
- **Graduated corruption recovery**: normal open → WAL delete → new DB
- **INSERT failure non-fatal**: logs warning, continues in memory-only mode
- **WAL checkpoint** on session end (`PRAGMA wal_checkpoint(TRUNCATE)`)

## Module Structure

```
history/
├── __init__.py
├── conversation_history.py   # ConversationHistory(IConversationHistory)
├── storage_backend.py        # SQLiteStorageBackend, MemoryStorageBackend
├── exceptions.py             # HistoryError
└── README.md
```
