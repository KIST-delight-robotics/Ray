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
- **WAV header quirk**: OpenAI TTS may return WAV with `n_frames = INT_MAX` (malformed header). Use ffmpeg to re-encode if feeding into ASR or other consumers that validate headers.
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

### TurnDetector (`turn_taking/turn_detector.py`)

- **Paper-based two-path algorithm** (Skantze & Irfan, 2025): Path 1 (VAP sustained robot-favor) OR Path 2 (TurnGPT graduated silence timeout). Either path alone triggers turn_shift. This avoids single-model dependency — VAP gives fast response (~500ms) while TurnGPT ensures eventual turn-taking even if VAP fails.
- **Internal turn state transition**: `_turn_state` switches to `ROBOT_TURN` immediately on turn_shift (not on external signal). Prevents race condition where user speech during the generation gap could produce a spurious `prepare` instead of `interrupt`. Per-frame state is reset at transition, not deferred to `reset()`.
- **VAP favor timer reset on user speech**: `_vap_favor_robot_elapsed_sec` resets when `user_is_speaking=True`, not only when VAP probabilities flip. Prevents stale accumulation across speech gaps.
- **Backchannel distinction via p_fut**: During ROBOT_TURN with robot_audio, `p_now > threshold` alone is insufficient for interrupt — `p_fut` must also favor user. This filters out backchannels ("yeah", "mhm") where p_now spikes but p_fut stays robot-favoring.
- **Prepare similarity gate**: `SequenceMatcher.ratio() >= 0.8` suppresses redundant prepare signals when ASR text changes minimally. Avoids wasteful LLM+TTS restarts on minor ASR corrections.
- **`notify_turn_complete` ignores `role`**: TurnGPT's `<ts>` format marks turn boundaries without speaker identity. The `role` parameter exists in the interface contract for potential future use but is not needed by the current TurnGPT model.
- **VAP error default behavior**: `VAPResult(0, 0, False)` on errors looks like "robot favored," which could accumulate toward false turn_shift via Path 1. Accepted because transient errors won't sustain for 500ms, and persistent VAP failure falls back to Path 2 (TurnGPT + silence timing) which operates independently.

### SpeechGenerator (`generation/speech_generator.py`)

- **Streaming API over batch**: Replaced `get_result() -> ResponseData` with `poll_audio() -> bytes | None` + `stream_done` + `get_text()` + `get_response_data()`. Allows Orchestrator to stream TTS chunks to CppBridge as they arrive instead of waiting for full synthesis. `GeneratorState.READY` renamed to `STREAMING`.
- **Per-run queue isolation**: Each `prepare()` creates a new `queue.Queue`. Background task captures the queue reference at submission time. Even if a stale producer puts to its old queue, Orchestrator only reads from the current queue. No cross-run contamination possible.
- **Run-ID guard**: Monotonic `_run_id` counter. Background task captures `run_id` at submission. All state writes check `run_id == self._run_id` under lock. Stale runs silently exit without writing state. This is the primary safety mechanism — `cancel_event` is cooperative and can miss blocking I/O windows, but run_id guard is absolute.
- **Cooperative cancellation via `threading.Event`**: Checked between pipeline steps and during LLM/TTS chunk iteration. Cannot interrupt blocking `next()` calls on LLM/TTS iterators — Python generators don't support cross-thread interruption during execution. Best-effort `.close()` on iterators/streams when cancel is detected at a check point.
- **`max_workers=2` default**: With `max_workers=1`, a new `prepare()` run queues behind the cancelled run until it exits. If the cancelled run is blocked on a first-token API call (1–3s), the new run is delayed. With 2 workers, the new run starts immediately on the other worker while the cancelled run drains cooperatively. Pileup concern: at most 2 runs active simultaneously; stale runs exit within one blocking call duration. Voice pipeline prepare() frequency is bounded by turn detector signals, so 3+ rapid cancellations within one API timeout is unrealistic.
- **Timestamp retrieval fallback**: `tts_stream.timestamps` access wrapped in `try/except`. If timestamps fail, empty list used instead of failing the entire run. Audio and text are already produced at that point.
- **`get_text()` accessible in FAILED state**: After STREAMING → FAILED (mid-stream TTS error), the LLM text is still valid and useful for logging/debugging. Blocking it would discard useful information.

## Phase 5 — Orchestrator

### ConversationHistory ID-based update

- **Message IDs**: `add_user_message`/`add_assistant_message` return sequential `int` IDs. Messages stored with internal `_id` field, stripped on `get_messages()` and `save()`. Enables `update_message(id, text)` for barge-in truncation correction (Case C: approximate → precise).
- **No deep copy in `get_messages()`**: Switched from `copy.deepcopy()` to dict comprehension stripping `_id`. New dicts are created per call so external mutation doesn't affect internal state.

### Orchestrator design

- **Decision before drain (step 5 before step 7)**: Turn detection runs before draining audio to bridge. An interrupt is processed before sending more audio, preventing unnecessary data transmission.
- **User message save at `_begin_streaming()` only**: Not saved at `turn_shift`. During `awaiting_response`, the user may continue speaking — saving early would create stale entries. If generation fails, user turn is not recorded (no orphan messages).
- **`_check_generator_completion` passes empty text**: When generator becomes STREAMING while `awaiting_response`, `_begin_streaming("")` is called. The method uses `_saved_user_text` (stable) internally, not the volatile `_last_asr_text` which may have been reset by ASR errors or prior operations.
- **Three-case barge-in truncation**: Case A (timestamps from ResponseData) and B (duration ratio from ResponseData) are immediate. Case C (stream not done) saves approximate truncation immediately, then defers correction via `_pending_truncation` — each frame checks if generator has finished and updates via `history.update_message()`.
- **Pending truncation cleanup**: Cleared in 5 situations — stream_done (with correction), generator FAILED (keep approximate), new `_begin_streaming()`, new `_handle_prepare()`, and `_end_session()`. Prevents stale pending state from leaking across turns.
- **STOP_PENDING watchdog (5s default)**: If C++ never responds to send_stop, force IDLE. Stale events arriving after watchdog timeout are naturally ignored (playback state is already IDLE, so the state guards on PLAYBACK_COMPLETE/PLAYBACK_STOPPED don't match).
- **Interrupt during awaiting**: TurnDetector switches to ROBOT_TURN on turn_shift (preventing spurious `prepare` from user speech during generation gap). If user speaks during awaiting, TurnDetector emits `interrupt`. Orchestrator cancels generation, calls `turn_detector.reset()` to restore USER_TURN, clears awaiting state.
- **`turn_detector.reset()` on generator FAILED**: Same as interrupt — must restore USER_TURN so TurnDetector doesn't stay stuck in ROBOT_TURN.
- **Bridge send errors not fatal**: `send_audio()`/`send_stop()` failures are logged. The subsequent `_poll_cpp_events()` on the same or next frame will detect the broken connection and terminate the session. Avoids duplicating termination logic.
- **DurationRatioTruncator direct import from tts module**: Orchestrator is the wiring layer per CLAUDE.md design. Creating new `DurationRatioTruncator` instances per barge-in with response-specific `total_duration_sec` is inherently a concrete operation — cannot be abstracted behind the `IUtteranceTruncator` interface without factory complexity.
- **Exit keyword matching**: Punctuation stripped, case-insensitive, word boundary via set membership. `"goodbye"` does not match keyword `"bye"` — each keyword is an exact word.

## Phase 6 — SessionManager + AudioInput

### AudioInput (`audio/audio_input.py`)

- **Lazy PyAudio import**: `import pyaudio` in constructor. `ImportError` → `AudioInputError` at construction time, not at runtime.
- **Daemon thread**: If main thread dies, audio thread dies too. Clean shutdown via `stop()` in normal flow.
- **Always drop on queue full**: `put_nowait()` with `queue.Full` caught and logged. No config flag — blocking the capture thread is never acceptable.
- **Error attribute**: Thread captures exception to `_error` attribute for external inspection, then exits. No re-raise — thread errors are silent to the main loop.

### Orchestrator stop signal

- **`request_stop()` + `threading.Event`**: External stop signal checked at the top of `_run_frame()`. Lets SessionManager cancel a running Orchestrator on shutdown.
- **Clear event at `run()` start**: `self._stop_event.clear()` prevents a stale stop from a previous session from immediately terminating the next one.

### SessionManager (`session/session_manager.py`)

- **Orchestrator as direct dependency (not factory)**: Orchestrator's `run()` already calls `_start_session()`/`_end_session()` — it resets itself between sessions. No need for re-creation.
- **Flush CppBridge events before greeting/farewell**: `_flush_bridge_events()` drains all pending events. Prevents acting on stale `PLAYBACK_COMPLETE` from a previous cycle.
- **`history.save()` guarded**: Called in FAREWELL and `shutdown()`, but only if `_session_started` is True. Prevents crash on cold-start shutdown before any session.
- **CppBridge connect on startup**: `run()` calls `cpp_bridge.connect()` before entering the main loop. Ensures connection is established before any greeting/farewell.
- **`poll_event()` exception handling**: Greeting/farewell polling loops catch `poll_event()` exceptions. Bridge errors break out of the poll loop but don't crash SessionManager.
- **Audio queue drain on ACTIVE entry**: `_drain_audio_queue()` clears stale frames before passing the queue to Orchestrator. Prevents the first ASR/TurnDetector frames from containing old audio.
- **Greeting/farewell timeout**: Timeout expiry is treated as playback done (log warning, proceed). No error raised.
- **`bridge.disconnect()` on exit**: `run()` finally 블록에서 `bridge.disconnect()` 호출. `connect()`와 대칭.
- **Orchestrator 내부 상태 초기화**: `_start_session()`에서 모든 내부 상태(`_playback_state`, `_awaiting_response` 등) 초기화. `request_stop()`으로 비정상 종료 후 재사용 시 이전 세션 상태가 남는 문제 방지.
- **`SessionManager(ISessionManager)` 상속**: `core/interfaces.py`의 `ISessionManager` 구현. 프로젝트의 인터페이스 규칙 준수.
