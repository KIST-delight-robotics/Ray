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
- **Test structure**: Unit (`test_<module>.py`, no marker) + integration/stress (`test_<module>_integration.py`, `test_<module>_stress.py`, `@requires_api`). Integration tests cover error recovery (invalid config, stale stream, double stop). Stress tests cover rapid reset cycles, sustained streaming, back-to-back start/stop.
- **Test helpers in conftest.py**: WAV helpers (`WavInfo`, `read_wav_frames`, etc.) and fixtures (`speech_wav`, `asr_lang`) shared via `tests/asr/conftest.py` instead of duplicating across test files.
- **API constraints documented separately**: Google STT v1 constraints in `asr/google_stt_v1_constraints.md`. Module README focuses on usage, not API limits.
- **Documentation language**: All docs written in English.

## 2026-03-03 — Phase 3 Step 3: LLM Module (`llm/`)

- **Vendor**: OpenAI Responses API. Only vendor needed; interface allows swapping later.
- **System message → `instructions`**: System message extracted from ContextBuilder output and passed via the Responses API's `instructions` parameter (idiomatic pattern), not embedded in `input`.
- **No `previous_response_id`**: We manage context ourselves via ContextBuilder with token budgeting, not the Responses API's server-side conversation state.
- **API key via env var**: Standard `OPENAI_API_KEY` environment variable, not in config. The SDK reads it automatically.
- **Config field mapping**: `LLMConfig.max_tokens` maps to API's `max_output_tokens`.
- **Explicit stream cleanup**: `create(stream=True)` + `try/finally` with `stream.close()`. Avoids the context-manager-inside-generator antipattern where `__exit__` is deferred to GC. Caller must exhaust or `.close()` the iterator.
- **Broad error wrapping**: All SDK and streaming exceptions (including `APITimeoutError`) wrapped in `LLMError` for predictable orchestrator behavior. Non-OpenAI exceptions during streaming also wrapped.
- **SDK-delegated retry**: Transient errors (429, 500, 503, connection) handled by OpenAI SDK's built-in retry with exponential backoff. Exposed via `LLMConfig.max_retries` (default 2). No custom retry logic to avoid double-retry.
- **Request timeout**: `LLMConfig.timeout_sec` (default 30s) prevents stalled API calls from blocking the voice pipeline. Passed to SDK client constructor.
- **Token counter**: `create_token_counter(model)` factory uses tiktoken. Falls back to `o200k_base` encoding for unknown models. Returns a `TokenCounter` callable matching the type alias in `core/types.py`.
- **Default system prompt**: `DEFAULT_SYSTEM_PROMPT` in `prompts.py`. Kept short and conversational for the Ray voice assistant persona.

## 2026-03-03 — Phase 3 Step 4: TTS Module (`tts/`)

- **Vendor**: OpenAI Audio API. Only vendor needed; `ITTS` interface allows swapping later.
- **Streaming redesign**: `ITTS.synthesize()` returns `TTSStream` (Iterator[bytes]) instead of `TTSResult`. Audio chunks arrive incrementally so SpeechGenerator can buffer and send to C++ on demand. `TTSResult` is still used as a convenience via `TTSStream.result` property after full iteration.
- **`TTSStream` in `core/types.py`**: Concrete `Iterator[bytes]` wrapper. Yields PCM chunks, collects full audio internally via `bytearray`. Exposes `.audio`, `.timestamps`, `.result` only after complete iteration (raises `RuntimeError` otherwise). `_done` flag set only on natural `StopIteration`, not on `close()`.
- **`timestamps_fn` callback**: Deferred timestamp retrieval. OpenAI passes `None` (no timestamps). Future vendors (e.g., ElevenLabs) can pass a callable. Result cached on first access.
- **Eager CM entry**: `synthesize()` calls `response_cm.__enter__()` immediately, not inside the generator. This ensures `close_fn` can always safely exit the CM, even if the generator was never started. `__enter__` errors are wrapped as `TTSError`.
- **Single-exit guarantee**: Shared `exited` flag between `_iter_chunks` and `_safe_close` prevents double `__exit__()` calls. The generator marks exited on completion/error/GeneratorExit; `close_fn` checks the flag before attempting exit.
- **PCM output**: 24 kHz, 16-bit signed little-endian, mono (headerless). Fixed by OpenAI when `response_format="pcm"`. Matches `TTSConfig.output_sample_rate` default.
- **No word timestamps**: OpenAI TTS does not support word-level timestamps. `stream.timestamps` returns `()`. `DurationRatioTruncator` handles this case for barge-in.
- **Model-specific instructions**: `_SUPPORTS_INSTRUCTIONS = {"gpt-4o-mini-tts"}` explicit set. `instructions` kwarg only sent for supported models; unsupported models log a warning and omit the parameter. No prefix matching — easy to extend.
- **`save_to_file()`**: Non-streaming convenience method on `OpenAITTS` (not on `ITTS` interface). Uses `response_format="wav"` and `write_to_file()`. For testing/utility only.
- **Input validation**: Text must be non-empty, non-whitespace, ≤4096 chars. Speed must be 0.25–4.0. Validated before API call to fail fast.
- **SDK-delegated retry**: Same pattern as LLM — `max_retries` and `timeout_sec` passed to `openai.OpenAI()` constructor. No custom retry logic.
- **TTSConfig new fields**: `speed` (0.25–4.0, default 1.0), `timeout_sec` (default 30.0), `max_retries` (default 2), `instructions` (default empty, for gpt-4o-mini-tts only).
- **Thread safety**: Not TTSStream's concern. SpeechGenerator handles concurrent synthesize() calls. Each TTSStream instance is consumed by a single thread.

## 2026-03-03 — Phase 3 Step 5: CppBridge Module (`bridge/`)

- **Transport**: `websockets` sync client (v14–16). Stable, well-documented sync API. Async not needed since bridge runs from the orchestrator's sync frame loop.
- **Message protocol**: JSON text frames for all messages. Audio encoded as base64 in JSON (`{"type": "audio", "data": "<base64>"}`). Single parsing path, simpler debugging. ~33% bandwidth overhead acceptable for PCM at 24 kHz over localhost.
- **Threading model**: Single-threaded lifecycle (connect/disconnect/send_*/poll_event from orchestrator thread only). Daemon receiver thread calls `recv(timeout=1.0)` in a loop, checking `_running` event flag. Same pattern as ASR's `_audio_generator` timeout loop.
- **Event queue**: Unbounded `queue.Queue`. Events are tiny frozen dataclasses (4 types), orchestrator polls at ~33 Hz (30 ms frame duration). No backpressure risk.
- **Error propagation**: Receiver stores `BridgeError` under lock on connection loss. Orchestrator discovers via `_check_error()` in `poll_event()`/`send_*()`. Error cleared after first raise. Same pattern as ASR.
- **Connection retry**: `connect()` retries up to `reconnect_attempts` (default 3) with 1s fixed sleep between attempts. No exponential backoff — this is a localhost connection, not a distributed service. Retries exist for startup race (Python up before C++).
- **WebSocket connect params**: Explicit `proxy=None` (avoid v15+ auto-proxy), `ping_interval=None` (no WebSocket keepalive on localhost — TCP handles it), `compression=None` (avoid CPU overhead on base64 audio).
- **Fresh state on reconnect**: `connect()` creates a new `queue.Queue` and clears stale error. Old receiver thread exits when `_running` is cleared during `disconnect()`. No generation ID needed.
- **Config fields**: `reconnect_attempts` (3), `recv_timeout_sec` (1.0), `connect_timeout_sec` (5.0), `close_timeout_sec` (5.0) added to `CppBridgeConfig`.
- **Integration tests**: Self-contained — `websockets.sync.server.serve()` as in-process test server. No external C++ process required. Server-side connection tracking for explicit close testing.
- **Stress tests**: Rapid connect/disconnect cycles (10x), high-volume audio streaming (1000 chunks), event flood (500 events), concurrent send+poll (no deadlock).

## 2026-03-03 — Phase 3 Step 6: Wakeword Detector

- **Architecture**: Silero VAD (speech segmentation) + Google STT `recognize()` (keyword matching). VAD detects speech boundaries, STT transcribes the segment, regex checks for keywords.
- **Non-streaming STT**: `client.recognize()` instead of streaming. Wakeword utterances are <3s, SessionManager is in SLEEP mode so brief blocking (~200-500ms) is acceptable. Timeout via `stt_timeout_sec` (default 5s).
- **VAD rechunking**: Pipeline frames are 480 samples (30ms@16kHz), Silero VAD requires 512 samples (32ms). Residual buffer carries across `feed_audio()` calls. Duration calculations calibrated in ms based on 32ms per VAD chunk.
- **Silero VAD via `silero-vad` package**: `load_silero_vad(onnx=False)` returns JIT model (~2MB). Model states reset between detection cycles (not every call). `torch` pulled in as transitive dependency (needed for Phase 4 VAP/TurnGPT anyway).
- **State machine**: IDLE → SPEECH (prob > threshold) → TRAILING (prob < threshold) → recognition (silence exceeds `speech_pad_ms`). TRAILING can recover back to SPEECH if speech resumes. Safety cap forces recognition at `max_speech_duration_sec`.
- **Word-boundary keyword matching**: `\b` regex instead of substring `in` to avoid false positives (e.g., "array" matching "ray"). Case-insensitive. All STT result alternatives checked, not just top-1.
- **Fail-closed error handling**: VAD inference errors and STT/network errors log a warning and return `False`. Only initialization failures (model load, client creation) raise `WakewordError`. VAD `reset_states()` errors suppressed to avoid masking primary exceptions.
- **Resource cleanup**: `close()` method for STT client transport teardown. Idempotent, error-suppressing. SessionManager calls it when destroying the detector.
- **Threading**: Not thread-safe by design. SessionManager feeds frames from a single SLEEP loop thread. No locking needed.
- **Duration math**: Uses `_bytes_per_sec` (sample_rate * sample_width * channels) for all duration calculations, correctly handling multi-channel audio.
- **STT phrase hints**: `SpeechContext(phrases=keywords)` boosts recognition of wakeword terms. `max_alternatives=5` to check multiple transcript hypotheses.
- **Config additions**: `WakewordConfig` extended with `language_code`, `speech_pad_ms`, `min_speech_duration_ms`, `max_speech_duration_sec`, `stt_timeout_sec`.

## 2026-03-03 — Phase 3 Step 7: LED Controller (`led/`)

- **Optional hardware**: `rpi5_ws2812` imported at module level with `try/except ImportError`. When absent, `_strip` is `None` and frame writes are skipped (noop mode). No runtime flag needed — import presence drives the behavior.
- **Animation protocol**: `LEDAnimation` is a `runtime_checkable` Protocol with `reset()`, `render(tick, bar_count, ring_count)`, and `frame_interval_sec`. Pluggable per-state: custom animations just implement the protocol.
- **StaticAnimation**: Built-in fixed-color animation. All default state mappings use `StaticAnimation` with placeholder colors. Real colors/animations added later without architecture changes.
- **Animation thread**: Daemon thread with `Event`-based sleeping (`_state_changed.wait(timeout=interval)`). `set_state()` signals the event so the thread wakes immediately instead of waiting for the previous animation's interval to expire.
- **Thread safety**: `threading.Lock` protects `_state`, `_tick`, and `_animations` reads/writes. `_stop_event` and `_state_changed` are `threading.Event` (inherently thread-safe).
- **close() lifecycle**: Sets `_stop_event`, signals `_state_changed` to wake thread, `join(timeout=2.0)`, logs warning if thread didn't exit (matching CppBridge pattern), then applies OFF frame.
- **Missing animation fallback**: If no animation is registered for a state, the loop applies an all-off frame instead of leaving stale LEDs.
- **Render error resilience**: Exceptions from `animation.render()` are logged at DEBUG and suppressed. Visual feedback is non-critical — fail-open.
- **Hardware driver API**: `rpi5_ws2812` uses `WS2812SpiDriver(spi_bus, spi_device, led_count)` + `driver.get_strip()`. Brightness is 0.0-1.0 float on the strip. No explicit `close()` — cleanup is turn-off-and-show.
- **LEDConfig redesign**: Replaced generic `led_count`/`spi_device`/`noop` fields with `bar_count=8`, `ring_count=16`, `spi_pin=10`, `brightness=128` (0-255 integer, converted to 0.0-1.0 for the driver). `spi_pin` documents physical wiring; driver uses `spi_bus=0, spi_device=0`.
- **No integration tests**: Hardware-dependent module with no real device in CI. All unit tests run in noop mode or with mocked driver class.

## 2026-03-04 — Phase 4 Step 1: VAP Wrapper (`turn_taking/vap.py`)

- **Rolling stereo buffer**: `(1, 2, n_samples)` on CPU. User audio in channel 0, robot audio in channel 1. Copied to device only at inference time to avoid persistent GPU memory.
- **Model loading**: `VapGPT(VapConfig())` + `torch.load(state_dict, weights_only=True)` + `load_state_dict`. All errors wrapped in `VAPError`.
- **Timing validation**: Zero `n_samples`, `step_samples`, or `tt_frames` raises `VAPError` at construction. Prevents silent incorrect slicing.
- **Robot audio resampling**: `torchaudio.functional.resample` from TTS rate (24kHz) to pipeline rate (16kHz). Pad/trim to match user audio length.
- **Cached result**: Inference runs only when `samples_since_inference >= step_samples`. Between inferences, the cached `VAPResult` is returned.
- **Error resilience**: `feed_audio()` catches all exceptions and returns `_DEFAULT_RESULT`. `_run_inference()` also catches internally. Never propagates errors to orchestrator.
- **PCM conversion**: 16-bit signed LE to float32 via `struct.unpack` + normalize by 32768. No numpy dependency.
- **Oversized frame clamping**: Frames larger than the context buffer are silently truncated to buffer size.

## 2026-03-04 — Phase 4 Step 2: TurnGPT Wrapper (`turn_taking/turngpt.py`)

- **Stateless wrapper**: No internal buffer, cache, or counter. Each `predict()` call is independent. `reset()` is a no-op (satisfies interface contract).
- **Model loading**: `TurnGPT.load_from_checkpoint(checkpoint_path)` — single-step load (restores tokenizer + embeddings). All errors wrapped in `TurnGPTError`.
- **Inference**: `string_list_to_trp(dialog_text, add_post_eos_token=False)` → extract `trp_probs[0, -1].item()`. Last position gives the probability for the current (partial) turn.
- **Empty guard**: Empty or whitespace-only input returns `0.0` without calling the model. Avoids tokenizer errors on empty strings.
- **Error resilience**: All inference errors caught, logged as warning, return `0.0`. Never propagates errors to TurnDetector/Orchestrator. Matches VAP wrapper pattern.
- **Lazy import**: `from turngpt import TurnGPT` inside constructor `try/except`. Package absence raises `TurnGPTError` at construction, not import time.
- **Text formatting contract**: Wrapper does not modify input text. TurnDetector is responsible for correct `<ts>` formatting. Trailing separators or malformed input are passed through to the model.
