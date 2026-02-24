# Decision Log

## 2026-02-24 — Phase 0: Project Setup

- **Git strategy**: Orphan branch `revamped` on existing repo (`KIST-delight-robotics/Ray`). Clean history, no legacy baggage.
- **Package manager**: `uv` with `pyproject.toml`. No `requirements.txt`.
- **Dev tools**: `ruff` for linting + formatting, `pytest` for testing. All configured in `pyproject.toml`.
- **Python version**: `>=3.11` (required for modern typing features).
- **Directory structure**: Full skeleton created upfront per CLAUDE.md spec. All directories have `__init__.py` for proper package resolution.

## 2026-02-24 — Phase 1: Foundation (`core/`)

- **Incremental interfaces**: Only Phase 2 consumer interfaces defined (IConversationHistory, IStorageBackend, IUtteranceTruncator, IContextBuilder). Remaining interfaces added just before their consuming phase to avoid premature churn.
- **IContextBuilder.build(current_text)**: History injected via constructor (not method param). Matches "inject via constructor" rule and keeps the call site simple.
- **TTSResult in types.py**: Pure data structure placed alongside WordTimestamp, not in interfaces.py.
- **CppEvent.position_sec is Optional**: `None` for events where position is meaningless (PLAYBACK_STARTED, PLAYBACK_COMPLETE). Avoids ambiguous `0.0` default.
- **TurnDecision**: Frozen dataclass with `__post_init__` validation (at most one signal True). `none()` class method eliminates nullable returns.
- **ResponseData mutable**: Not frozen because audio bytes are large — frozen dataclasses hash fields, and hashing large bytes is expensive.
- **TTSResult.timestamps as tuple**: Immutable (frozen dataclass), unlike ResponseData.timestamps which is a mutable list.
- **Config**: No empty stubs for future phases. Only AudioConfig and ConversationHistoryConfig defined. PipelineConfig grows as modules are added.
- **IStorageBackend**: Included because Phase 2 ConversationHistory implementation depends on it.

## 2026-02-24 — Phase 2: Independent Modules (`history/`, `tts/utterance_truncator`, `context/`)

- **Token-based context management**: Replaced `max_turns_in_context` with `max_context_tokens` in config. `ConversationHistory.get_messages()` returns all messages (pure storage layer). Token budget management is `ContextBuilder`'s responsibility — fills context in reverse chronological order within the token budget.
- **TokenCounter type alias**: `Callable[[str], int]` in `core/types.py`. Simpler than a full ABC interface. Vendor-specific implementations (e.g., tiktoken) will be provided in Phase 3.
- **ContextBuilder system_prompt**: Plain string constructor parameter for now. Will be sourced from `llm/prompts.py` in Phase 3 and passed in at construction time.
- **UtteranceTruncator strategies**: `TimestampTruncator` for precision with word-level timestamps, `DurationRatioTruncator` for estimation without timestamps. `DurationRatioTruncator` always requires `total_duration_sec` and ignores timestamps entirely — no overlapping logic.
- **MemoryStorageBackend only**: File/DB backends deferred. Deep copies on load/save to prevent aliasing between backend and history.
- **HistoryError**: Raised on operations without an active session. Inherits from `PipelineError`.
- **`__init__.py` re-exports**: All Phase 2 modules re-export public classes via `__init__.py` for cleaner import paths. Applied consistently going forward.
