# Voice Conversation Robot — Python Pipeline Architecture v2

Temporary — subject to change at any time.


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
| Note | During GREETING/FAREWELL: signal only, no audio streaming to C++. audio_queue not consumed (drained on ACTIVE entry). |


### 2.16 Orchestrator

| Item | Description |
|------|-------------|
| Role | ACTIVE mode conversation loop. Frame-driven. Controls module execution flow based on TurnDecision. |
| Input | audio_queue (received from SessionManager) |
| Internal state | Current ASR text, playback state (`idle` / `playing` / `stop_pending`), awaiting_response flag, current ResponseData |

**Per-frame loop (never blocks):**

1. Dequeue frame from audio_queue
2. Feed frame to ASR
3. Poll current text from ASR
4. Feed frame + text + robot audio to TurnDetector → receive TurnDecision
5. Handle TurnDecision (see below)
6. Check CppBridge events
7. If `awaiting_response`: check SpeechGenerator completion (see below)
8. Check session timeout

**On `prepare`:**

- SpeechGenerator: cancel existing preparation + start new preparation (pass current ASR text, runs in background)

**On `turn_shift`:**

1. Check exit keyword → if match, return (end session)
2. Save user message to ConversationHistory (current ASR text)
3. Check SpeechGenerator state:
   - `streaming` → begin streaming audio to CppBridge via `poll_audio()` → set playback state to `playing` → reset ASR
   - `preparing` or `idle` → if `idle`, trigger generation now → set `awaiting_response` flag → reset ASR
4. (Frame loop continues — no blocking)

**While `awaiting_response` (checked per frame, step 7):**

- Check SpeechGenerator state
- When `streaming` → begin streaming audio to CppBridge via `poll_audio()` → set playback state to `playing` → clear `awaiting_response`
- During `awaiting_response`, playback state is `idle` (C++ is not playing anything). Since the robot is not speaking, TurnDetector will not emit `interrupt` — instead, new user speech produces `prepare` or `turn_shift`:
  - **`prepare`**: new user text is stabilizing. Cancel current generation, start fresh generation for combined text (saved user message + current ASR text). Remain in `awaiting_response`.
  - **`turn_shift`**: user finished additional speech. Append new ASR text to the previously saved user message (combine into one turn), cancel current generation, start fresh generation for the combined text. Remain in `awaiting_response`.

**On `interrupt` (during `playing`):**

- Send stop command to CppBridge → set playback state to `stop_pending`

**CppBridge events (checked per frame):**

| Event | Action |
|-------|--------|
| **Playback complete** | Save full ResponseData text to ConversationHistory as assistant → set playback state to `idle` |
| **Playback stopped** (during `stop_pending`) | Use stop position + ResponseData → select UtteranceTruncator strategy → save truncated text to ConversationHistory as assistant → set playback state to `idle` |
| **Playback position** | Pass to TurnDetector for VAP robot audio synchronization |

**Truncation strategy selection (on barge-in):**

- If `ResponseData.has_timestamps` → use TimestampTruncator (stateless, pre-injected)
- Otherwise → create DurationRatioTruncator with `total_duration_sec` derived from audio length

**Exit conditions:** Exit keyword detected or session timeout → return

**Session timeout:** Elapsed time since last ASR text change while playback state is `idle` and not `awaiting_response`. Timer resets on ASR text change and on playback completion. Does not count during `playing`, `stop_pending`, or `awaiting_response`.

**ACTIVE lifecycle:** Start ASR on entry, stop ASR and clean up resources on exit.

**Dependencies:** ASR, TurnDetector, SpeechGenerator, CppBridge, UtteranceTruncator, ConversationHistory

**Execution model:** Frame-driven synchronous loop — never blocks on I/O. Background operations (LLM, TTS) run within respective modules; Orchestrator polls for completion.


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
│   └── orchestrator.py          # ACTIVE mode conversation loop
│
└── session/
    └── session_manager.py       # Top-level state machine
```

Test structure and development conventions are documented in CLAUDE.md.


## 6. Open Items

- [x] ~~Main loop orchestrator structure~~ → SessionManager (top-level state machine) → Orchestrator (ACTIVE-only, frame-driven loop)
- [x] ~~Main loop execution model~~ → Frame-driven sync loop + module-internal background processing
- [x] ~~Audio distribution~~ → AudioInput separate thread → queue → consumer loop (SessionManager-owned)
- [x] ~~ASR text delivery~~ → Polling (Orchestrator polls per frame). ConversationHistory saves only on turn confirmation.
- [x] ~~VAP robot audio source~~ → TTS audio + playback timing sync (Python-held audio, C++ provides position events)
- [ ] ConversationHistory StorageBackend selection
- [x] ~~Wakeword engine selection~~ → Silero VAD (speech segmentation) + Google STT `recognize()` (keyword matching)
- [x] ~~LLM / TTS vendor finalization~~ → OpenAI (LLM: Responses API, TTS: Audio API). ASR: Google Cloud STT.
- [ ] Exit keyword list and configuration location
- [ ] Session timeout value and configuration location
- [ ] ContextBuilder tool definition integration (deferred until LLM tools implementation)
- [ ] RAG / long-term memory design (future)
- [ ] LED behavior definition (which timing → which color/animation)
- [ ] LED control location (Python direct vs C++ relay)
