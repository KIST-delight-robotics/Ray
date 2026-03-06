# Decision Log

## Phase 1 — Foundation (`core/`)

- **Incremental interfaces**: Only next-phase consumer interfaces defined at each step. Remaining interfaces added just before their consuming phase.
- **ResponseData mutable**: Not frozen because hashing large audio bytes is expensive.
- **CppEvent.position_sec is Optional**: `None` for events where position is meaningless. Avoids ambiguous `0.0`.
- **TurnDecision**: `__post_init__` validates at most one signal True. `none()` class method eliminates nullable returns.

## Phase 2 — Independent Modules (`history/`, `utterance_truncator`, `context/`)

- **Token-based context management**: `ConversationHistory.get_messages()` returns all (pure storage). `ContextBuilder` fills context in reverse chronological order within `max_context_tokens` budget.
- **TokenCounter as `Callable[[str], int]`**: Type alias in `core/types.py`. Simpler than a full ABC.
- **UtteranceTruncator dual strategy**: `TimestampTruncator` (word-level timestamps) vs `DurationRatioTruncator` (no timestamps, uses `total_duration_sec`). No overlapping logic.
- **MemoryStorageBackend**: Deep copies on load/save to prevent aliasing.

## Phase 3 — External Modules

### ASR (`asr/`)

- **Google Cloud STT V1**: V2 adds batch/adaptation features not needed here.
- **Threading**: Daemon reader thread + bounded `queue.Queue(maxsize=300)`. `feed_audio()` drops frames on full queue (backpressure).
- **No auto-restart on 5-min stream limit**: Orchestrator handles via `reset()` between turns.
- **Transcript accumulation**: Final segments concatenated; interim replaced on each update. `get_text()` returns both combined.
- **Error pattern**: gRPC errors wrapped in `ASRError`, stored under lock, raised via `_check_error()`, cleared after first raise. Reused by CppBridge.

### LLM (`llm/`)

- **OpenAI Responses API**: System message via `instructions` param, not embedded in `input`.
- **No `previous_response_id`**: We manage context ourselves via ContextBuilder with token budgeting.
- **Explicit stream cleanup**: `try/finally` with `stream.close()` instead of context-manager-inside-generator antipattern.
- **SDK-delegated retry**: `max_retries` passed to SDK constructor. No custom retry to avoid double-retry.
- **Token counter**: `create_token_counter(model)` uses tiktoken, falls back to `o200k_base` for unknown models.

### TTS (`tts/`)

- **`TTSStream` (Iterator[bytes])**: Yields PCM chunks incrementally. `.audio`/`.timestamps`/`.result` only available after full iteration.
- **Eager CM entry**: `response_cm.__enter__()` called immediately in `synthesize()`, not inside generator. Ensures safe cleanup even if generator never started.
- **Single-exit guarantee**: Shared `exited` flag prevents double `__exit__()` between generator and `close_fn`.
- **No word timestamps from OpenAI**: `DurationRatioTruncator` handles barge-in estimation.
- **Model-specific instructions**: Explicit `_SUPPORTS_INSTRUCTIONS` set (not prefix matching).

### CppBridge (`bridge/`)

- **JSON + base64 protocol**: Single parsing path, simpler debugging. ~33% bandwidth overhead acceptable on localhost.
- **Connection retry for startup race**: Fixed 1s sleep, up to 3 attempts. No exponential backoff for localhost.
- **WebSocket params**: `proxy=None` (avoid v15+ auto-proxy), `ping_interval=None`, `compression=None`.
- **Fresh state on reconnect**: New queue, cleared error. No generation ID needed.

### Wakeword Detector

- **Silero VAD + Google STT `recognize()`**: VAD segments speech, non-streaming STT transcribes, `\b` regex matches keywords.
- **VAD rechunking**: Pipeline 480 samples (30ms@16kHz) → Silero needs 512 samples (32ms). Residual buffer across calls.
- **State machine**: IDLE → SPEECH → TRAILING → recognition. Safety cap at `max_speech_duration_sec`.
- **Fail-closed**: VAD/STT errors return `False`. Only init failures raise `WakewordError`.

### LED Controller (`led/`)

- **Optional hardware**: `try/except ImportError` on `rpi5_ws2812`. Import absence → noop mode, no runtime flag.
- **`LEDAnimation` Protocol**: `runtime_checkable`, pluggable per-state. `StaticAnimation` as default.
- **Animation thread**: `Event`-based sleeping; `set_state()` wakes thread immediately.
- **LEDConfig**: `bar_count=8`, `ring_count=16`, `brightness` 0–255 (converted to 0.0–1.0 for driver).

## Phase 4 — Composite Modules

### VAP Wrapper (`turn_taking/vap.py`)

- **Rolling stereo buffer**: `(1, 2, n_samples)` on CPU, copied to device only at inference. Channel 0 = user, channel 1 = robot.
- **Robot audio resampling**: `torchaudio.functional.resample` from TTS 24kHz to pipeline 16kHz.
- **Cached result**: Inference only when `samples_since_inference >= step_samples`.
- **Error resilience**: All errors caught, returns default result. Never propagates to orchestrator.

### TurnGPT Wrapper (`turn_taking/turngpt.py`)

- **Stateful (KV cache)**: Maintains `past_key_values` across `predict()` calls. Compares token-level prefix with previous input; only new tokens are forwarded. Identical input returns cached probability without model call. `reset()` clears all cache state. Rationale: ASR sends incremental updates (same prefix, growing suffix) — reprocessing the entire dialog each time is wasteful.
- **Direct model API**: Replaced `string_list_to_trp()` with separate `tokenizer()` → `model()` → `get_trp()` calls to control KV cache passing.
- **Context window eviction**: `max_context_tokens` (default 1024, GPT-2 limit). When exceeded, oldest turns evicted at text level (split by `<ts>`) until token count ≤ 80% of max. Eviction invalidates cache entirely (full rebuild). Headroom (0.8) prevents thrashing near the limit.
- **Lazy import**: `from turngpt import TurnGPT` inside constructor. Package absence raises `TurnGPTError` at construction.
- **Text formatting contract**: Wrapper passes input as-is. TurnDetector owns `<ts>` formatting.
- **Open**: Proactive cache warming (pre-forwarding robot turn tokens after turn completion) — deferred until latency measurement shows it's needed. No wrapper change required; TurnDetector adds one `predict()` call.
