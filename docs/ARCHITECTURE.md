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
| Input | Assembled context (message list) |
| Output | Streaming text chunks |
| Scope | Prompt templates, model parameters, API calls. Tool definitions/execution (future). |
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
| Strategies | **TimestampTruncator**: precise truncation using word-level timestamps (stateless) |
|            | **DurationRatioTruncator**: estimation from playback duration ratio. Requires `total_duration_sec` at construction time — a new instance per response. |
| Strategy selection | Orchestrator selects based on whether `ResponseData` has timestamps (see section 2.16) |
| Note | Strategy interface (`IUtteranceTruncator`). Independent of TTS implementation. |


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
| Role | Store/retrieve conversation history. Pure data store. |
| Scope | **Per-session** management. Auto-initialized on session start, saved on session end. |
| Input | Messages (`list[dict]`). Dict schema is LLM vendor-dependent, determined at LLM implementation time. |
| Output | Message list (all or recent N turns) |
| Extension point | **StorageBackend**: persistence strategy (memory / file / DB) |
| Note | Assistant message saved on: playback complete (full text) or barge-in interruption (UtteranceTruncator result) |


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
| GREETING | Send greeting signal via CppBridge → wait for playback completion → transition to ACTIVE. |
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
| Dependencies | IASR, ITurnDetector, ISpeechGenerator, ICppBridge, IConversationHistory, IUtteranceTruncator, ILEDController |

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
11. Session timeout check

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

```
voice_pipeline/
├── core/
│   ├── types.py                 # Shared data types (TurnDecision, ResponseData, etc.)
│   ├── interfaces.py            # All module interfaces
│   ├── exceptions.py            # PipelineError base
│   └── config.py                # Dataclass-based configuration
│
├── audio/
│   ├── audio_input.py           # Mic capture → audio_queue
│   ├── wakeword.py              # Wakeword detection
│   └── exceptions.py
│
├── asr/
│   ├── asr.py                   # ASR interface implementation
│   └── exceptions.py
│
├── turn_taking/
│   ├── vap.py                   # VAP wrapper
│   ├── turngpt.py               # TurnGPT wrapper
│   ├── turn_detector.py         # Combined turn decision
│   └── exceptions.py
│
├── llm/
│   ├── llm.py                   # LLM interface implementation
│   ├── prompts.py               # Prompt template management
│   ├── tools.py                 # Tool definitions & execution
│   └── exceptions.py
│
├── tts/
│   ├── tts.py                   # TTS interface implementation
│   ├── utterance_truncator.py   # Barge-in text truncation strategies
│   └── exceptions.py
│
├── context/
│   └── context_builder.py       # LLM context assembly
│
├── history/
│   ├── conversation_history.py  # Per-session conversation history
│   ├── storage_backend.py       # Persistence (memory / file / DB)
│   └── exceptions.py
│
├── generation/
│   ├── speech_generator.py      # ContextBuilder → LLM → TTS orchestration
│   └── exceptions.py
│
├── bridge/
│   ├── cpp_bridge.py            # C++ WebSocket communication
│   └── exceptions.py
│
├── led/
│   └── led_controller.py        # LED interface + implementations
│
├── orchestrator/
│   ├── orchestrator.py          # ACTIVE mode conversation loop
│   └── exceptions.py
│
└── session/
    └── session_manager.py       # Top-level state machine
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
voice_pipeline.audio        # AudioInput, WakewordDetector
voice_pipeline.asr          # ASR streaming
voice_pipeline.turn_taking  # TurnDetector, VAP, TurnGPT
voice_pipeline.generation   # SpeechGenerator
voice_pipeline.llm          # LLM API calls
voice_pipeline.tts          # TTS synthesis
voice_pipeline.bridge       # CppBridge WebSocket
voice_pipeline.led          # LED controller
voice_pipeline.orchestrator # Frame loop, turn handling, barge-in
voice_pipeline.session      # SessionManager state transitions
voice_pipeline.core         # TTSStream
```

### Error propagation summary

| Source | Orchestrator policy | SessionManager policy |
|--------|--------------------|-----------------------|
| ASR | Log, skip frame, continue | N/A (Orchestrator handles) |
| TurnDetector | Log, skip frame, continue | N/A |
| SpeechGenerator | Skip turn, reset TurnDetector | N/A |
| CppBridge | **Terminate session** | Greeting/farewell: log, break poll loop |
| History / LED | Log, continue | Log, continue |
| WakewordDetector | N/A | **Crash** (real bug — should be fixed) |
| AudioInput thread | Capture stops silently (`_error` set) | Not detected (queue starves) |


## 7. Known Limitations

- **`generator.shutdown()` reuse**: Orchestrator calls `generator.shutdown()` in `_end_session()`, terminating the ThreadPoolExecutor. If the same SpeechGenerator instance is reused for a second session, `prepare()` will fail unless the implementation re-creates the executor.
- **AudioInput thread death undetected**: If the capture thread dies (device error, etc.), the audio queue starves. SessionManager stays in SLEEP waiting for frames that never come. No health-check mechanism exists.
- **CppBridge reconnect**: `bridge.connect()` is only called once at `SessionManager.run()` start. If the bridge disconnects mid-session, Orchestrator terminates the session → FAREWELL → SLEEP. On the next wakeword, greeting is sent on a dead connection. Needs reconnect logic before greeting or in the SLEEP loop.
- **Signal handling**: No SIGINT/SIGTERM handler. `Ctrl+C` kills the process without calling `shutdown()`, so `history.save()` is skipped.


## 8. Open Design Questions

- ConversationHistory StorageBackend selection (memory / file / DB)
- ContextBuilder tool definition integration (deferred until LLM tools)
- RAG / long-term memory design
- LED behavior definition (which state → which color/animation)
- LED control location (Python direct vs C++ relay)
