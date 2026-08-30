# Voice Conversation Robot — Python Pipeline Architecture


## 1. System Modes

```
SLEEP ──(wakeword)──▶ GREETING ──(playback done)──▶ ACTIVE ──(exit keyword / timeout)──▶ FAREWELL ──(playback done)──▶ SLEEP
```

| Mode | Behavior | Blocked |
|------|----------|---------|
| **SLEEP** | Only WakewordDetector runs | Entire conversation pipeline |
| **GREETING** | Send greeting playback signal to C++, wait for completion | Wakeword, barge-in, ASR, turn detection |
| **ACTIVE** | Full conversation pipeline active | Wakeword |
| **FAREWELL** | Send farewell playback signal to C++, wait for completion | Wakeword, barge-in, ASR, turn detection |

- GREETING/FAREWELL: Pre-recorded audio played by C++. Python sends signal only, no audio streaming.
- Exit keyword check: Performed after turn confirmation, before LLM request.
- Session timeout: Elapsed time since last ASR text change **while robot is not speaking or generating**. Does not count during playback or response generation.


## 2. Modules

### 2.1 AudioInput

| Item | Description |
|------|-------------|
| Role | Capture audio stream from microphone and push to queue |
| Execution | Separate thread, always running (mode-independent) |
| Owned by | SessionManager |
| Output | `audio_queue` (audio frame queue) |
| Consumers | SLEEP: SessionManager → WakewordDetector, ACTIVE: Orchestrator → ASR + TurnDetector |
| On ACTIVE entry | Drain queue (remove stale frames accumulated during GREETING) |
| Note | Queue is not consumed during GREETING/FAREWELL. Drained on ACTIVE entry. |


### 2.2 WakewordDetector

| Item | Description |
|------|-------------|
| Role | Detect wakeword during SLEEP mode |
| Input | Audio frames |
| Output | Wakeword detection event |
| Note | Active only in SLEEP mode |


### 2.3 ASR

| Item | Description |
|------|-------------|
| Role | Real-time streaming speech-to-text |
| Input | Audio frames (Orchestrator feeds per frame) |
| Output | Current text (Orchestrator polls per frame) |
| Internal | Streaming API session management, partial/final result handling |
| Lifecycle | Orchestrator starts on ACTIVE entry, stops on ACTIVE exit |
| Interface | Vendor-swappable via `IASR` abstraction |


### 2.4 VAP (internal to TurnDetector)

| Item | Description |
|------|-------------|
| Role | Voice activity prediction for turn-taking |
| Input | User audio (from AudioInput), robot audio (TTS audio + playback timing sync) |
| Output | `p_now`, `p_fut`, `user_is_speaking` |
| Execution | Periodic per audio frame (called internally by TurnDetector) |
| Note | Robot audio: Python-held TTS audio provided in sync with C++ playback timing |


### 2.5 TurnGPT (internal to TurnDetector)

| Item | Description |
|------|-------------|
| Role | Text-based turn-end probability prediction |
| Input | Conversation history text |
| Output | Probability (0–1) |
| Execution | On ASR text change (called internally by TurnDetector) |


### 2.6 TurnDetector

| Item | Description |
|------|-------------|
| Role | Pure turn decision maker. Receives audio and ASR text, returns decisions only. Does not call external modules directly. |
| Input | Audio frames (per frame), current ASR text (per frame), robot audio (during playback, provided by Orchestrator based on playback position) |
| Output | TurnDecision (see below) |
| Owns internally | VAP instance, TurnGPT instance, all timing state, ASR change detection (including similarity comparison), threshold/timeout settings |
| External dependencies | None. Unaware of SpeechGenerator, ASR, etc. |

**TurnDecision output:**

| Signal | Meaning | Orchestrator action |
|--------|---------|---------------------|
| **turn_shift** | User turn ended. Robot may take the floor. | Start response playback (or generate first if not ready) |
| **interrupt** | User barge-in detected. Robot should stop speaking. | Stop robot playback |
| **prepare** | Pre-generation signal. Internally confirmed after similarity comparison of stabilized text. | Cancel existing preparation + start new response preparation |


### 2.7 LLM

| Item | Description |
|------|-------------|
| Role | Generate conversation responses |
| Input | Assembled context (message list), optional tool definitions |
| Output | `LLMStream` — streaming text chunks, `.result` provides `LLMResult` (text, tool_calls, metrics) after consumption |
| Metrics | `LLMMetrics` captured per call: Usage (input/output/cached/reasoning tokens), model, latency_ms, ttft_ms |
| Tools | `tools` parameter: `None` = config defaults, `[]` = disabled. Tool definitions + token costs managed in `adapters/llm_openai.py` |
| Interface | Vendor-swappable via `ILLM` abstraction |


### 2.8 TTS

| Item | Description |
|------|-------------|
| Role | Text-to-speech synthesis |
| Input | Text |
| Output | Audio data + (optional) word-level timestamps |
| Interface | Vendor-swappable via `ITTS`. Timestamp support varies by implementation. |


### 2.9 UtteranceTruncator

| Item | Description |
|------|-------------|
| Role | Compute the spoken portion of text at barge-in |
| Input | Original text, playback stop position, (optional) word-level timestamps |
| Output | Truncated text |
| Strategies | `truncate_by_timestamps`: word-level timestamps로 정확히 절단 |
|            | `truncate_by_ratio`: 재생 시간 비율로 추정 |
| Strategy selection | SessionLoop이 `ResponseData`에 timestamps가 있는지로 선택 (section 2.16) |
| Note | `session_loop.py`의 순수 함수. TTS 구현과 독립. |


### 2.10 ContextBuilder

| Item | Description |
|------|-------------|
| Role | Assemble context before LLM calls |
| Input sources | ConversationHistory (past turns), current ASR text (parameter), system prompt |
| Output | Message list for LLM |
| Extension | New context sources added here: tool definitions (with LLM tools), RAG results, long-term memory (future) |


### 2.11 ConversationHistory

| Item | Description |
|------|-------------|
| Role | Store/retrieve conversation history with write-through persistence. |
| Scope | **Per-session** management. Auto-initialized on session start. |
| Storage | **Write-through**: in-memory list for reads, SQLite INSERT on every mutation. `save()` only sets `ended_at`. |
| Message format | OpenAI Responses API input format. Each message = one DB row. `turn_id` groups multi-message turns (tool calls). |
| Token tracking | `token_count` pre-computed at save time (LLM `output_tokens` or tiktoken fallback). `metrics_json` stores full LLM call metadata. |
| Read paths | `get_messages()` → flat list for LLM input (memory only). `get_turns()` → grouped by turn_id for ContextBuilder budgeting. |
| Backend | `SQLiteStorageBackend` (WAL mode). Tests use the `":memory:"` path |
| Threading | `threading.Lock` on all public methods. Writes from main thread, reads from background thread. |


### 2.12 SpeechGenerator

| Item | Description |
|------|-------------|
| Role | Chain ContextBuilder → LLM → TTS to produce response audio. Manage speculative preparation. |
| States | `idle` → `preparing` → `streaming` → `idle` (normal), `preparing`/`streaming` → `failed` (error) |
| Flow | 1. ContextBuilder assembles context |
|      | 2. LLM generates full response text (streaming collected) |
|      | 3. TTS streams audio chunks → `poll_audio()` |
| Output | `poll_audio()` for incremental PCM chunks during `streaming`, `get_response_data()` for full ResponseData after `stream_done` |
| Key behaviors | **Speculative prepare**: run LLM+TTS in background before turn confirmation. Audio chunks available via `poll_audio()` as soon as state reaches `streaming`. |
|               | **Cancel**: on context change (new ASR text), cancel previous preparation → `idle`. Run-ID guard ensures stale runs never write state. |
|               | **Streaming consumption**: Orchestrator polls `poll_audio()` per frame, checks `stream_done` to know when complete. `get_response_data()` returns full ResponseData and transitions to `idle`. |
| Execution | `ThreadPoolExecutor(max_workers=2)`. New `prepare()` starts immediately on fresh worker while cancelled run drains cooperatively. |
| Dependencies | ContextBuilder, LLM, TTS |


### 2.13 CppBridge

| Item | Description |
|------|-------------|
| Role | Python ↔ C++ communication |
| Transport | **WebSocket** (existing approach) |
| Python → C++ | TTS audio transmission, control commands (barge-in stop, playback start), greeting/farewell playback signals |
| C++ → Python | Playback state events (started, position, complete, stopped + stop position) |
| Message format | Type tags to distinguish audio/commands/events |
| Note | Interface abstracted for future transport replacement (e.g. ZeroMQ) |


### 2.14 LEDController

| Item | Description |
|------|-------------|
| Role | WS2812 LED color/animation control |
| Input | LED state commands |
| Trigger | TBD (turn-related timing, etc.) |
| Interface | Abstract, swappable implementations |
| Implementations | **DirectLEDController**: Python direct hardware control (current) |
|                 | **BridgeLEDController**: C++ control via CppBridge (future, if feasible) |


### 2.15 SessionManager

| Item | Description |
|------|-------------|
| Role | Top-level state machine. Mode transitions + session lifecycle management. |
| Owns | AudioInput (thread + queue), WakewordDetector, Orchestrator, ConversationHistory |
| References | CppBridge (greeting/farewell signals) |
| SLEEP | Feed frames from audio_queue → WakewordDetector. On wakeword detection → transition to GREETING. |
| GREETING | Ensure bridge connected (reconnect if needed, return to SLEEP on failure) → send greeting signal → wait for playback completion → transition to ACTIVE. |
| ACTIVE | Drain audio_queue → issue session_id, initialize ConversationHistory → call `Orchestrator.run(audio_queue)`. On return → transition to FAREWELL. |
| FAREWELL | Send farewell signal via CppBridge → wait for playback completion → save ConversationHistory → transition to SLEEP. |
| Startup | `run()` calls `bridge.connect()` then `audio_input.start()`. |
| Shutdown | `shutdown()` sets event + `orchestrator.request_stop()`. `run()` finally block calls `audio_input.stop()` + `bridge.disconnect()`. |
| Note | During GREETING/FAREWELL: signal only, no audio streaming to C++. audio_queue not consumed (drained on ACTIVE entry). Stale CppBridge events flushed before greeting/farewell. |


### 2.16 Orchestrator

| Item | Description |
|------|-------------|
| Role | ACTIVE mode conversation loop. Frame-driven. Controls module execution flow based on TurnDecision. |
| Input | audio_queue (received from SessionManager) |
| Internal state | PlaybackState (`idle`/`playing`/`stop_pending`), `awaiting_response` flag, current ResponseData, sent audio buffer, pending truncation |
| Dependencies | IASR, TurnDetector, SpeechGenerator, CppBridge, ConversationHistory, LEDController (벤더 교체 대상인 ASR만 인터페이스) |

**Per-frame loop (never blocks):**

1. Dequeue frame from audio_queue (with timeout)
2. Feed frame to ASR
3. Poll current text from ASR
4. Track text changes (for session timeout)
5. Turn detection → handle decision (**before** audio drain — interrupt processed before sending more audio)
6. Poll CppBridge events
7. If PLAYING: drain audio to bridge
8. If `awaiting_response`: check SpeechGenerator completion
9. If pending truncation: check deferred truncation
10. STOP_PENDING watchdog check
11. Audio starvation check (terminate if no frames for `audio_starvation_timeout_sec`)
12. Session timeout check

**On `prepare`:**

- Combine text (saved + current if awaiting), restart SpeechGenerator. Clear pending truncation.

**On `turn_shift`:**

1. Check exit keyword → if match, return (end session)
2. Check SpeechGenerator state:
   - `streaming` → `_begin_streaming()` (see below)
   - `preparing` → set `awaiting_response`, save user text, LED THINKING
   - `idle` → trigger generation, set `awaiting_response`, LED THINKING

**`_begin_streaming(text)`:**

1. Combine user text (saved + current, filtering empties)
2. Save user message to ConversationHistory (returns message ID)
3. Notify TurnDetector of user turn completion
4. Reset ASR
5. Drain available audio to CppBridge
6. Set playback state to PLAYING, LED SPEAKING

**User message save policy:** Saved **once** at `_begin_streaming()` with final combined text. Not saved at turn_shift to avoid stale entries when text is combined during awaiting. If generation fails, user turn is not recorded.

**While `awaiting_response` (checked per frame, step 8):**

- `STREAMING` → `_begin_streaming()` → clear awaiting
- `FAILED` → skip turn, reset TurnDetector, clear awaiting, LED LISTENING
- During awaiting, TurnDetector is in ROBOT_TURN. User speech triggers:
  - **`interrupt`**: cancel generation, reset TurnDetector, clear awaiting, LED LISTENING
  - **`prepare`**: combine saved + current text, restart generation

**On `interrupt`:**

- During PLAYING: send_stop → STOP_PENDING (start watchdog timer)
- During awaiting: cancel generation, reset TurnDetector, clear awaiting

**CppBridge events (checked per frame):**

| Event | Guard | Action |
|-------|-------|--------|
| **PLAYBACK_COMPLETE** | PLAYING only | Save full text to history, notify TurnDetector, reset to IDLE |
| **PLAYBACK_STOPPED** | STOP_PENDING only | Barge-in truncation (see below), reset to IDLE |
| **PLAYBACK_POSITION** | PLAYING only | Update tracked position |

Events arriving in wrong state (stale events) are silently ignored.

**Barge-in truncation (on PLAYBACK_STOPPED):**

| Case | Condition | Action |
|------|-----------|--------|
| **A** | ResponseData available + has_timestamps | TimestampTruncator (injected) → save to history |
| **B** | ResponseData available + no timestamps | DurationRatioTruncator (from audio length) → save to history |
| **C** | No ResponseData (stream not done) | DurationRatioTruncator (from sent buffer length) → save approximate → set `_pending_truncation` |

**Deferred truncation (Case C follow-up, checked per frame):**

- Generator stream completes → get ResponseData → re-truncate with precise data → `history.update_message()` to correct the approximate entry
- Generator FAILED → keep approximate truncation, clear pending
- Pending truncation cleared on: stream_done, FAILED, new `_begin_streaming()`, new `_handle_prepare()`, session end

**Robot audio for TurnDetector:** `get_robot_audio_chunk()` extracts a 30ms chunk from `_sent_audio_buffer` at the current playback position. Returns None when not PLAYING or buffer exhausted.

**STOP_PENDING watchdog:** If no PLAYBACK_STOPPED event arrives within `stop_pending_timeout_sec` (default 5s), force transition to IDLE. Stale events arriving after watchdog timeout are ignored (state is already IDLE).

**Exit keyword:** Case-insensitive word boundary match after stripping punctuation. Checked on turn_shift before generation.

**Session timeout:** Timer since last ASR text change. Paused (timer reset) during PLAYING or `awaiting_response`. Resets on any text change.

**Error handling:**

| Source | Policy |
|--------|--------|
| ASR / TurnDetector | Log warning, skip frame, continue |
| SpeechGenerator FAILED | Skip turn, reset TurnDetector, LED LISTENING |
| CppBridge | Terminate session |
| History / LED | Log warning, continue |

**ACTIVE lifecycle:** Start ASR + LED LISTENING on entry (all internal state reset for clean session). Stop ASR, shutdown generator, save history, LED OFF on exit. Supports `request_stop()` for external cancellation (SessionManager shutdown).

**Execution model:** Frame-driven synchronous loop — never blocks on I/O. Background operations (LLM, TTS) run within SpeechGenerator; Orchestrator polls for completion.


## 3. Call Structure

```
SessionManager (top-level state machine)
│
├─ SLEEP:  audio_queue → WakewordDetector
│
└─ ACTIVE: Orchestrator
             ├── audio_queue → ASR (feed audio / poll text)
             ├── audio_queue + ASR text → TurnDetector → TurnDecision
             ├── SpeechGenerator (response generation only)
             │     └── ContextBuilder → LLM → TTS → ResponseData
             ├── CppBridge (audio transmission + playback event reception)
             ├── UtteranceTruncator (text truncation on barge-in)
             └── ConversationHistory
```


## 4. VAP Robot Audio Synchronization

```
Python holds: Full TTS audio + word-level timestamps (if available)
C++ provides: Playback state events (started, current position, complete)

Synchronization:
  1. Orchestrator sends ResponseData audio to CppBridge
  2. CppBridge relays playback start/position events
  3. Orchestrator passes playback position to TurnDetector
  4. VAP module consumes TTS audio aligned to playback position

Future extension:
  - If OS audio capture or C++ audio relay becomes needed,
    only replace VAP's robot audio input interface
```


## 5. Directory Structure

기준: **밖의 것을 감싸면 `adapters/`에 파일 하나, 안의 로직은 top-level 파일**. 패키지는 선택 가능한
서브시스템(`memory/`)만. 읽는 순서는 `voice_pipeline/__init__.py` docstring 참조.

```
voice_pipeline/
├── __main__.py        # 모드 루프 (SLEEP → GREETING → ACTIVE → FAREWELL)
├── wiring.py          # 컴포넌트 조립 (프로세스/세션 수준), TTS 벤더 선택
├── session_loop.py    # ACTIVE 프레임 루프 — ASR, 턴 감지, 재생, barge-in
├── generator.py       # SpeechGenerator (ContextBuilder → LLM → TTS) + SentenceDetector
├── prompt.py          # DEFAULT_SYSTEM_PROMPT, 블록 포매터, ContextBuilder, HistorySummarizer
├── turn_detector.py   # VAP + TurnGPT + VAD 결합 판정
├── history.py         # ConversationHistory + SQLiteStorageBackend
├── text_session.py    # 텍스트 전용 세션 (eval --text)
├── greeting_audio.py  # 인사/작별 오디오 사전 생성
├── trace.py           # PipelineTrace/CallRecord + SQLite 스토어 + Tracked 래퍼 (관측용)
├── types.py           # IASR/ILLM/ITTS/IEmbedder + 계약 타입(스트림·결과), AudioFrame/TokenCounter
├── settings.py        # 오디오 형식, DB 경로, 토큰 예산
├── adapters/          # 외부 경계 — 벤더·하드웨어·외부 모델 래퍼
│   ├── audio_input.py   asr_google.py   wakeword.py
│   ├── llm_openai.py    tts_openai.py   tts_elevenlabs.py   token_counter.py
│   ├── vap.py           turngpt.py      embedder.py
│   └── cpp_bridge.py    led.py
├── memory/            # 장기 기억 서브시스템 (storage, retriever, writer, vector_index)
└── tests/             # adapters/ · memory/ · integration/ + top-level test_<file>.py
```

Test structure and development conventions are documented in CLAUDE.md.


## 6. Threading Model & Debugging

### Threads at runtime

| Thread | Owner | Lifetime | Purpose |
|--------|-------|----------|---------|
| Main | SessionManager | `run()` duration | State machine loop, Orchestrator frame loop |
| AudioInput | AudioInput | `start()` → `stop()` | PyAudio capture → `audio_queue` |
| CppBridge receiver | CppBridge | `connect()` → `disconnect()` | WebSocket recv → `event_queue` |
| SpeechGenerator workers (×2) | SpeechGenerator | per `prepare()` | LLM + TTS background generation |
| LED animation | LEDController | `set_state()` lifecycle | Animation render loop |

### Log namespaces

```
voice_pipeline                  # 모드 전환 (__main__)
voice_pipeline.session_loop     # 프레임 루프, 턴 처리, barge-in
voice_pipeline.generator        # SpeechGenerator
voice_pipeline.prompt           # ContextBuilder, HistorySummarizer
voice_pipeline.turn_detector    # TurnDetector
voice_pipeline.history / .memory / .trace / .wiring / .text_session / .types
voice_pipeline.audio / .wakeword / .asr / .llm / .tts / .bridge / .led / .embedding
voice_pipeline.adapters.vap / .adapters.turngpt
```

### Error propagation summary

| Source | Orchestrator policy | SessionManager policy |
|--------|--------------------|-----------------------|
| ASR | Log, skip frame, continue | N/A (Orchestrator handles) |
| TurnDetector | Log, skip frame, continue | N/A |
| SpeechGenerator | Skip turn, reset TurnDetector | N/A |
| CppBridge | **Terminate session** | Greeting: reconnect or SLEEP; farewell: log, break poll loop |
| History / LED | Log, continue | Log, continue |
| WakewordDetector | N/A | **Crash** (real bug — should be fixed) |
| AudioInput thread | Audio starvation timeout (5s) → **terminate session** | SLEEP: detect via `error` property → crash |


## 7. Known Limitations

- **`generator.shutdown()` reuse**: Orchestrator calls `generator.shutdown()` in `_end_session()`, terminating the ThreadPoolExecutor. If the same SpeechGenerator instance is reused for a second session, `prepare()` will fail unless the implementation re-creates the executor.
- **AudioInput thread death in SLEEP mode**: If the capture thread dies, SessionManager detects it via `audio_input.error` and crashes. In ACTIVE mode, Orchestrator's audio starvation timeout (`audio_starvation_timeout_sec`, default 5s) terminates the session. After returning to SLEEP, SessionManager detects the error and crashes. Recovery depends on external process supervision (e.g. systemd `Restart=always`).
- **CppBridge mid-session disconnect**: If the bridge disconnects during ACTIVE mode, Orchestrator terminates the session → FAREWELL → SLEEP. On the next wakeword, `_run_greeting()` calls `bridge.connect()` to reconnect. If reconnect fails (C++ process still down), SessionManager returns to SLEEP and retries on the next wakeword.
- **Signal handling**: No SIGINT/SIGTERM handler. `Ctrl+C` kills the process without calling `shutdown()`, so `history.save()` is skipped.
- **Session crash safety**: With write-through storage, messages are persisted on every turn boundary. Crash during session loses at most `ended_at` timestamp. Signal handler calls `shutdown()` → `history.save()` for clean termination.


## 8. Open Design Questions

- ~~ConversationHistory StorageBackend selection~~ → SQLite write-through (resolved)
- ~~ContextBuilder tool definition integration~~ → Tool token costs in `adapters/llm_openai.py`, deducted from budget (resolved)
- RAG / long-term memory design (see `docs/ray-memory-design.md`)
- LED behavior definition (which state → which color/animation)
- LED control location (Python direct vs C++ relay)
- Client-side tool execution loop in SpeechGenerator (structure ready, implementation deferred)
