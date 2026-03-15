# Decision Log

## Phase 1 — Foundation (`core/`)

- **Incremental interfaces**: Only next-phase consumer interfaces defined at each step. Remaining interfaces added just before their consuming phase.
- **ResponseData mutable**: Not frozen because hashing large audio bytes is expensive.
- **CppEvent minimal**: Only `event_type` field. No `position_sec` — barge-in position estimated via time-based tracking in Orchestrator.
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
- **ONNX backend**: Added for inference performance on RPi. PyTorch still required (tokenization uses torch tensors) — goal is speed, not PyTorch elimination. Backend selected by config: `onnx_model_path` set → ONNX, otherwise PyTorch. KV cache support detected via `past_key_0` in ONNX input names. TRP extraction identical to PyTorch: softmax → EOS token probability (verified `get_trp` is just `x[..., eos_token_id]`).
- **ONNX threads default = 2**: Benchmarked 1–4 threads on RPi 5 (4-core). 2 threads optimal for both fp32 and int8: fastest latency (42ms int8, 111ms fp32) while leaving 2 cores for other modules. 4 threads causes contention and is slower.
- **int8 quantization**: 4x smaller (157MB vs 623MB), 2.6x faster, TRP difference negligible (~0.04). Recommended for deployment.
- **Open**: Proactive cache warming (pre-forwarding robot turn tokens after turn completion) — deferred until latency measurement shows it's needed. No wrapper change required; TurnDetector adds one `predict()` call.

### MaAI VAP optimization (`turn_taking/maai_vap.py`)

- **`use_torch_compile` default → True**: MaAI 트랜스포머(3.6M params, dim=256)의 병목은 연산량(7.2M FLOPs)이 아니라 258개 PyTorch 모듈의 dispatch overhead. `torch.compile(mode="default")`이 P50 기준 56ms→32ms (1.8x). 10Hz 100ms budget 대비 53% 여유 확보.
- **INT8 양자화 부적합**: CPC 인코더(Conv1D+LSTM, 2.5M params)에 `quantize_dynamic` 적용 시 1.9x 느려지고 정확도 대폭 하락. TurnGPT(MatMul 위주, 163M params)와 달리 작은 Conv 커널에서는 양자화 오버헤드가 연산 절감을 초과하고, 모델이 이미 L2 캐시에 들어가므로 메모리 대역폭 이점 없음.
- **배치 처리 무의미**: No-cache 배치(N frames)는 KV cache + 1 frame보다 느림. KV cache가 이미 반복 연산을 제거하므로 텐서 크기를 키워도 추가 이점 없음.
- **torch.compile 한계**: Inductor가 그래프를 fuse하지만, 작은 텐서([1,1,256]) 연산에서 커널 launch + 메모리 할당 오버헤드가 남아 이론치(0.4ms) 대비 32ms. PyTorch 안에서 추가 개선 어려움. 10ms 이하를 원하면 트랜스포머도 ONNX export 필요.
- **동시 실행 안정적**: VAP(10Hz, compile=ON) + TurnGPT(3Hz, ONNX int8) 동시 실행 시 CPU 34%, 두 모델 모두 budget 내 안정적. ASR/LLM/TTS에 ~60% 여유.
- **`use_onnx_transformer` default → True**: Transformer를 ONNX export하여 전체 파이프라인(encoder+transformer)을 ORT로 실행. PyTorch dispatch overhead 완전 제거. Mean 24ms (vs PyTorch 106ms, 3.9x speedup). Budget 초과 0% (PyTorch 35.5%). `torch.compile` warmup 100프레임 문제도 해소. ONNX 변환 시 dict KV cache → 12개 flat stacked tensor로 변환. ALiBi 마스크 pre-compute. Cross-attention source 순서 주의 필요 (원본 입력을 src로 전달, 업데이트된 값 아님). 수치 차이 max 6.8e-6 (1,200프레임 CANDOR 실제 음성).
- **ORT 싱글스레드 최적 유지**: Transformer ONNX 추가 후에도 `ort_threads=1`이 최적. `ort_threads=4`는 스레드 동기화 비용으로 2x 느려짐 (48ms vs 24ms). PyTorch threads는 전체 ONNX 파이프라인에서 영향 없음 (ORT가 자체 스레드풀 사용).
- **PyTorch `p_now` 리스트 반환 버그**: `VapGPT.forward()`가 `p_now`을 `[speaker1, speaker2]` 리스트로 반환. 기존 코드 `float(out["p_now"])`은 항상 실패하지만 integration test 없어 미발견. `_process_transformer_pytorch`에서 `p_now[0]` 추출로 수정.

### TurnDetector (`turn_taking/turn_detector.py`)

- **Paper-based two-path algorithm** (Skantze & Irfan, 2025): Path 1 (VAP sustained robot-favor) OR Path 2 (TurnGPT graduated silence timeout). Either path alone triggers turn_shift. This avoids single-model dependency — VAP gives fast response (~500ms) while TurnGPT ensures eventual turn-taking even if VAP fails.
- **Internal turn state transition**: `_turn_state` switches to `ROBOT_TURN` immediately on turn_shift (not on external signal). Prevents race condition where user speech during the generation gap could produce a spurious `prepare` instead of `interrupt`. Per-frame state is reset at transition, not deferred to `reset()`.
- **VAP favor timer reset on user speech**: `_vap_favor_robot_elapsed_sec` resets when `user_is_speaking=True`, not only when VAP probabilities flip. Prevents stale accumulation across speech gaps.
- **Backchannel distinction via p_fut**: During ROBOT_TURN with robot_audio, `p_now > threshold` alone is insufficient for interrupt — `p_fut` must also favor user. This filters out backchannels ("yeah", "mhm") where p_now spikes but p_fut stays robot-favoring.
- **Prepare similarity gate**: `SequenceMatcher.ratio() >= 0.8` suppresses redundant prepare signals when ASR text changes minimally. Avoids wasteful LLM+TTS restarts on minor ASR corrections.
- **`notify_turn_complete` ignores `role`**: TurnGPT's `<ts>` format marks turn boundaries without speaker identity. The `role` parameter exists in the interface contract for potential future use but is not needed by the current TurnGPT model.
- **VAP error default behavior**: `VAPResult(0, 0, False)` on errors looks like "robot favored," which could accumulate toward false turn_shift via Path 1. Accepted because transient errors won't sustain for 500ms, and persistent VAP failure falls back to Path 2 (TurnGPT + silence timing) which operates independently.

### Async thread separation (`turn_taking/async_vap.py`, `async_turngpt.py`)

- **Both VAP and TurnGPT on dedicated threads**: RPi 5 worst case VAP 24ms + TurnGPT 30ms = 54ms, exceeding 30ms frame budget by 80%. ONNX Runtime releases GIL during inference, so separate threads achieve true parallelism. Frame loop now fully decoupled from inference latency.
- **AsyncVAP implements IVAP**: Drop-in replacement. `feed_audio()` buffers audio pairs and returns latest cached result (non-blocking). Background thread runs at configurable rate (default 10Hz), drains buffer, concatenates frames, and runs inference. Works with any `IVAP` implementation (VAPWrapper, MaAIVAPWrapper).
- **AsyncTurnGPT uses submit/poll pattern (not IVAP-like)**: TurnGPT's usage pattern (text input, infrequent calls) differs from VAP's (audio input, every frame). submit/poll is more natural than pretending it's the same interface. `SyncTurnGPTAdapter` wraps sync `ITurnGPT` for unit test compatibility.
- **1-frame TurnGPT delay accepted**: `process_frame()` polls at the top, submits at the bottom. Result from frame N's submit arrives at frame N+1's poll. 30ms delay is negligible for turn-taking decisions that operate on 500ms+ timescales.
- **`_pending_text = None` guards stale results**: If `clear_pending()` is called (turn transition) before inference completes, the background thread checks `_pending_text is not None` before storing the result. Stale predictions are silently discarded.
- **Reset delegated to background thread**: `AsyncTurnGPT.reset()` sets `_pending_reset` flag. The background thread calls `turngpt.reset()` — KV cache access stays on the same thread as `predict()`, avoiding races.
- **Session-scoped threads**: Created per session in `session_factory()`, stopped on next session start and program exit. Model objects remain process-level singletons (warmup preserved).
- **`__main__.py` switched to MaAIVAPWrapper**: Full ONNX pipeline (encoder + transformer) as default. `PipelineConfig.maai_vap` field added. Old `VAPWrapper` (PyTorch) still available for testing.

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
- **History records `generator.input_text`**: `_begin_streaming()` reads `generator.input_text` (the text passed to the most recent `prepare()`) for history recording, not the current ASR text at turn_shift time. This ensures the recorded user message matches what the LLM actually saw. `input_text` is managed by SpeechGenerator's lifecycle: set on `prepare()`, cleared on `cancel()`/`reset()`/`get_response_data()`.
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

- **Session factory pattern (replaces direct Orchestrator injection)**: Previous design reused a single Orchestrator/ConversationHistory across sessions, relying on `reset()` for state cleanup. TurnDetector's `_dialog_parts` (dialog context) was not cleared by `reset()`, causing state leakage between sessions. New design: `session_factory: Callable[[], SessionComponents]` creates fresh Orchestrator, TurnDetector, SpeechGenerator, ContextBuilder, and ConversationHistory per session. Process-level singletons (ASR, LLM, TTS, VAP, TurnGPT, CppBridge, executor) are captured by the factory closure.
- **Three-tier lifecycle model**: (1) Process-level — model loads, API clients, hardware, shared executor (expensive init, once). (2) Session-level — stateful orchestration objects (factory-recreated per session). (3) Turn-level — lightweight `reset()` within session objects (ASR buffer, TurnDetector frame counters).
- **Shared ThreadPoolExecutor**: SpeechGenerator accepts optional `executor` param. When externally provided, `shutdown()` only cancels in-flight work (sets cancel_event) without closing the executor. Main entry point creates the executor, injects into factory, shuts down in `finally`. Thread count stays fixed across sessions (max_workers=2).
- **History save ownership**: Removed `_save_history()` from Orchestrator's `_end_session()`. SessionManager is the sole caller of `history.save()` — in `_run_farewell()` and `shutdown()`. Prevents double-save and clarifies lifecycle ownership.
- **`orchestrator.run()` exception guard**: Wrapped in `try/except` in `_run_active()` to ensure FAREWELL mode is always reached. Without this, an exception from `_end_session()` (e.g., `asr.stop()` failure) would skip history save entirely since Orchestrator no longer saves.
- **`_session_lock`**: Protects `_current_orchestrator` and `_current_history` against race between `shutdown()` (signal handler thread) and `_run_active()` (main thread).
- **External audio_queue injection**: `audio_queue` parameter allows main entry point to create the queue and pass it to both AudioInput and SessionManager, avoiding circular dependency.
- **Flush CppBridge events before greeting/farewell**: `_flush_bridge_events()` drains all pending events. Prevents acting on stale `PLAYBACK_COMPLETE` from a previous cycle.
- **`history.save()` guarded**: Called in FAREWELL and `shutdown()`, but only if `_session_started` and `_current_history is not None`. Prevents crash on cold-start shutdown before any session.
- **CppBridge connect on startup**: `run()` calls `cpp_bridge.connect()` before entering the main loop. Ensures connection is established before any greeting/farewell.
- **`poll_event()` exception handling**: Greeting/farewell polling loops catch `poll_event()` exceptions. Bridge errors break out of the poll loop but don't crash SessionManager.
- **Audio queue drain on ACTIVE entry**: `_drain_audio_queue()` clears stale frames before passing the queue to Orchestrator. Prevents the first ASR/TurnDetector frames from containing old audio.
- **Greeting/farewell timeout**: Timeout expiry is treated as playback done (log warning, proceed). No error raised.
- **`bridge.disconnect()` on exit**: `run()` finally block calls `bridge.disconnect()`, symmetric with `connect()` at startup.
- **`SessionManager(ISessionManager)` inheritance**: Implements `ISessionManager` from `core/interfaces.py`. Follows the project's interface convention.

### Main Entry Point (`voice_pipeline/__main__.py`)

- **Process-level singletons in `main()`**: All expensive objects (models, API clients, hardware) created once. Session factory closure captures them by reference.
- **`vap.reset()` / `turngpt.reset()` in factory**: Called at session start to clear wrapper-level caches/buffers. These are process-level singletons but carry session-scoped state (rolling audio buffer, dialog context).
- **Windows signal handling**: `SIGINT` (Ctrl+C) + `SIGBREAK` (Ctrl+Break). No `SIGTERM` on Windows. Signal handler calls `sm.shutdown()` which is thread-safe via `_session_lock`.
- **`finally` cleanup order**: `executor.shutdown(wait=True)` first (waits for in-flight tasks), then `wakeword.close()` and `led.close()` (hardware release).

## C++ ↔ Python Protocol Alignment

### WebSocket topology
- **C++ as server, Python as client**: C++ runs `ix::WebSocketServer` on port 8765. Python connects via `websockets`. Reversed from legacy (both were clients to separate servers). Single connection expected — `g_client_ws` stores the one connected Python client.

### Protocol simplification
- **No `turn_id`**: Removed from C++. WebSocket TCP ordering + Python's state machine (wait for `playback_complete` before next turn) prevent stale chunk contamination. C++ clears buffers on `stream_start`.
- **No `playback_stopped`**: Merged into `playback_complete`. Python tracks its own `STOP_PENDING` state to distinguish normal completion from barge-in interruption. Simpler C++ — always sends `playback_complete` regardless of how playback ended.
- **No `playback_position` stream**: Position estimated via time: `stop_pos = stop_pending_time - playback_start_time`. Acceptable ~±100ms accuracy on localhost. `playback_started` event marks the timing reference.
- **`stream_start` replaces `responses_only` + `responses_stream_start`**: Single message to signal streaming intent. C++ clears old buffers and sets streaming flag on receipt.
- **`audio_end` replaces `responses_stream_end`**: Sent once by Python when TTS stream is fully drained.
- **`play_file` replaces `play_audio` + `send_greeting` + `send_farewell`**: Generic file playback. SessionManager passes config paths (`greeting_audio_path`, `farewell_audio_path`).
- **`stop` replaces `user_interruption`**: Python sends on barge-in. C++ sets `user_interruption_flag` (internal name preserved), threads check it cooperatively.
- **`playback_started` (new)**: Sent from `control_motor` at cycle 0 after `soundStream.play()`. Provides timing reference for barge-in position estimation and future VAP robot audio feed.

### C++ changes kept minimal
- **`play_music` and `play_audio_csv` preserved**: Not used by Python pipeline but kept in C++ to avoid breaking existing functionality.
- **Thread model unchanged**: `stream_and_split`, `generate_motion`, `control_motor` structure untouched. Only message handling and WebSocket setup modified.
- **`send_to_python()` helper**: Thread-safe send via `g_client_ws_mutex`. All `webSocket.sendText()` calls replaced.
- **Interruption sends `playback_complete`**: After cleanup, C++ now also sends `playback_complete` on interrupt (previously only sent on normal completion). This lets Python's STOP_PENDING state resolve cleanly.

## Runtime — Interrupt, Similarity, Storage

### Interrupt detection (`turn_detector`)
- **`user_is_speaking` prerequisite**: Paper pseudocode (Skantze & Irfan 2025, Appendix A lines 56-61) requires `user_is_speaking=True` before checking p_now/p_fut for interrupts. Without this guard, transient VAP probability spikes (e.g. right after turn-shift when VAP context is still user-biased) cause false interrupts even when nobody is speaking.
- **STOP_PENDING watchdog reset**: Watchdog must call `turn_detector.reset()` alongside `_reset_playback_state()` to prevent orphaned ROBOT_TURN state when PLAYBACK_COMPLETE never arrives from bridge.

### Similarity scoring (`core/similarity`)
- **Sentence embedding over SequenceMatcher**: SequenceMatcher measures character overlap — `"what is your"` vs `"what is your name"` scores 0.87 (blocked by 0.8 threshold). Sentence embedding (all-MiniLM-L6-v2) scores 0.66 (correctly passes gate). The paper uses semantic similarity (all-MiniLM-L6-v2) for this comparison.
- **sentence-transformers 3.x**: Pinned to 3.x for `optimum`/`transformers` compatibility. 5.x requires `transformers>=5.0` which conflicts with `optimum`'s ONNX runtime integration. No functional difference for inference-only usage.
- **ONNX backend deferred**: At 22M params, torch inference (~4ms) matches or beats ONNX (~6ms) on desktop CPU. ONNX `use_onnx` config option preserved for RPi 5 testing.

### File storage (`history/storage_backend`)
- **Wrapped JSON format**: Sessions saved as `{"session_id": ..., "started_at": ..., "messages": [...]}` instead of bare list. `load()` handles both formats for backward compatibility.
- **Default backend changed to file**: `data/sessions/` directory, auto-created. Memory backend still available via config.
