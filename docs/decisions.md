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

## 2026-03-02 — Phase 3 Step 2: ASR Module (`asr/`)

- **Vendor**: Google Cloud Speech-to-Text. Only vendor needed for the project; interface allows swapping later.
- **API version**: V1 (`google.cloud.speech`), not V2. V1 is stable, well-documented, and sufficient for streaming recognition. V2 adds features (e.g., batch, adaptation) not needed here.
- **Threading model**: Background daemon reader thread per stream. Orchestrator thread calls `feed_audio()`/`get_text()` synchronously. Audio flows through a bounded `queue.Queue(maxsize=300)` (~9s at 30ms frames). Reader thread calls `streaming_recognize()` and updates `_transcript` under a `threading.Lock`.
- **Encoding derivation**: `AudioConfig.sample_width` maps to gRPC encoding via `_ENCODING_MAP` (2 → LINEAR16). Unsupported widths raise `ASRError` at `_start_stream()` time. Added `sample_width: int = 2` to `AudioConfig`.
- **Sentinel shutdown**: `_stop_stream()` puts `b""` sentinel in the queue to unblock `_audio_generator()`. Reader thread exits cleanly when it receives the sentinel or `_running` event is cleared.
- **No auto-restart on stream limit**: Google's streaming API has a ~5 minute limit. The orchestrator handles this via `reset()` between turns — no automatic restart logic inside `GoogleCloudASR`.
- **Single exception class**: `ASRError(PipelineError)` for all ASR failures. gRPC errors (`GoogleAPICallError`) and unexpected exceptions are wrapped into `ASRError` and stored for the orchestrator thread to raise via `_check_error()`. Error is cleared after first raise to prevent stale re-raises.
- **Queue backpressure**: `feed_audio()` uses `put_nowait()` with `queue.Full` catch — drops the frame and logs a warning. Prevents unbounded memory growth if gRPC ingestion is slow.
- **Client cleanup**: `self._client.transport.close()` releases the gRPC channel on `stop()`.
- **Sample rate validation**: `_start_stream()` validates `sample_rate` is within Google STT's supported range (8000–48000 Hz). Raises `ASRError` if out of range. WAV files at any valid sample rate work without resampling.
- **Default language**: English (`en-US`) for ASR, `("ray",)` for wakeword keywords. LLM and TTS defaults (`gpt-4o`, `alloy`) already work with English without an explicit language field.
- **Transcript accumulation**: `is_final` results are concatenated into `_final_transcript`; interim results are stored in `_interim_transcript` and replaced on each update. `get_text()` returns `_final_transcript + _interim_transcript`. `reset()` clears both. Fixes a bug where previous final segments were lost when a new result arrived.
- **Client cleanup on start() failure**: `start()` wraps `_start_stream()` in try/except. If validation fails (bad sample rate, unsupported encoding), the already-created `SpeechClient` is closed and `_client` set to `None`. Prevents gRPC channel leak.
- **3-tier test structure**: Unit (`test_<module>.py`, no marker), integration (`test_<module>_integration.py`, `@requires_api`), stress (`test_<module>_stress.py`, `@requires_stress`). All three markers excluded from default pytest run. Integration tests cover error recovery (invalid config, stale stream, double stop). Stress tests cover rapid reset cycles, sustained streaming, back-to-back start/stop.
- **Test helpers in conftest.py**: WAV helpers (`WavInfo`, `read_wav_frames`, etc.) and fixtures (`speech_wav`, `asr_lang`) shared via `tests/asr/conftest.py` instead of duplicating across test files.
- **API constraints documented separately**: Google STT v1 constraints in `asr/google_stt_v1_constraints.md`. Module README focuses on usage, not API limits.
- **Documentation language**: All docs written in English.
