# Voice Conversation Robot — Project Guide

Python pipeline that handles real-time voice input, turn-taking detection, LLM response generation, and TTS synthesis.
Audio playback is handled by a C++ process, communicating over WebSocket.


## System Structure

```
SLEEP ──(wakeword)──▶ GREETING ──▶ ACTIVE ──(exit keyword/timeout)──▶ FAREWELL ──▶ SLEEP
```

```
SessionManager (top-level state machine)
├─ AudioInput (separate thread → audio_queue)
├─ SLEEP:  audio_queue → WakewordDetector
└─ ACTIVE: Orchestrator
             ├── ASR
             ├── TurnDetector (VAP + TurnGPT)
             ├── SpeechGenerator (ContextBuilder → LLM → TTS)
             ├── CppBridge
             ├── UtteranceTruncator
             └── ConversationHistory
```


## Directory Structure

```
voice_pipeline/
├── core/
│   ├── types.py               # Shared data types (TurnDecision, ResponseData, etc.)
│   ├── interfaces.py          # All module interfaces (IASR, ITTS, ILLM, etc.)
│   ├── exceptions.py          # PipelineError base only
│   └── config.py              # Dataclass-based configuration
│
├── audio/
│   ├── audio_input.py         # Mic capture → audio_queue
│   ├── wakeword.py            # Wakeword detection (Silero VAD + Google STT)
│   └── exceptions.py
│
├── asr/
│   ├── asr.py                 # ASR interface impl
│   └── exceptions.py
│
├── turn_taking/
│   ├── vap.py                 # VAPWrapper(IVAP) — VoiceActivityProjection
│   ├── maai_vap.py            # MaAIVAPWrapper(IVAP) — MaAI ONNX (default)
│   ├── async_vap.py           # AsyncVAP(IVAP) — background thread wrapper
│   ├── turngpt.py             # TurnGPTWrapper(ITurnGPT)
│   ├── async_turngpt.py       # AsyncTurnGPT — background thread wrapper
│   ├── turn_detector.py       # TurnDetector — combined turn decision
│   └── exceptions.py
│
├── llm/
│   ├── llm.py                 # LLM interface impl
│   ├── prompts.py             # Prompt templates
│   ├── tools.py               # Tool definitions & execution
│   ├── token_counter.py       # Token counting utilities
│   └── exceptions.py
│
├── tts/
│   ├── tts.py                 # TTS interface impl
│   ├── greeting_audio.py      # Pre-generated greeting/farewell audio
│   ├── utterance_truncator.py # Barge-in text truncation strategies
│   └── exceptions.py
│
├── context/
│   └── context_builder.py     # LLM context assembly
│
├── history/
│   ├── conversation_history.py
│   └── storage_backend.py     # Persistence (memory / sqlite)
│
├── generation/
│   ├── speech_generator.py    # ContextBuilder → LLM → TTS orchestration
│   └── exceptions.py
│
├── bridge/
│   ├── cpp_bridge.py          # C++ WebSocket communication
│   └── exceptions.py
│
├── led/
│   ├── led_controller.py      # LED interface + impls (Direct / Bridge)
│   ├── animations.py          # LED animation patterns
│   └── exceptions.py
│
├── similarity/
│   └── similarity.py          # Semantic similarity (embedding / difflib)
│
├── embedding/
│   └── embedder.py            # IEmbedder implementations + factory
│
├── memory/
│   ├── types.py               # Episode, Profile, MemoryReadResult data types
│   ├── vector_index.py        # IVectorIndex interface + NumpyVectorIndex
│   ├── storage.py             # SQLiteMemoryStorage + InMemoryMemoryStorage
│   ├── retriever.py           # MemoryRetriever — hybrid search + retained buffer
│   └── exceptions.py
│
├── orchestrator/
│   └── orchestrator.py        # ACTIVE mode conversation loop
│
├── session/
│   └── session_manager.py     # Top-level state machine
│
└── tests/
    ├── core/
    ├── audio/
    ├── asr/
    ├── turn_taking/
    ├── llm/
    ├── tts/
    ├── context/
    ├── history/
    ├── generation/
    ├── bridge/
    ├── led/
    ├── embedding/
    ├── orchestrator/
    ├── memory/
    ├── session/
    └── integration/
```


## Documentation

- **docs/SETUP.md**: Raspberry Pi 5 initial setup guide.
- **docs/decisions.md**: Finalized decision log for completed work.
- **docs/decisions-wip.md**: Decision log for work in progress. Merged into `decisions.md` after cleanup when the work is complete.
- **docs/ARCHITECTURE.md**: System architecture details.
- **docs/ray-memory/**: Long-term memory system design (overview, session, read, write, storage).
- **Module READMEs** (`turn_taking/README.md`, etc.): External repo setup, constraints, config params.


## External Model Dependencies

Some modules (VAP, TurnGPT, Wakeword etc.) wrap externally cloned model repositories.

- External repo APIs must not leak beyond the wrapper. The rest of the pipeline depends only on project interfaces.
- Wrappers accept model/repo path via config.
- Setup details (repo URL, version, install steps) are documented in each module's own README, not here.


## Environment & Commands

- Python 3.11+, uv (pyproject.toml), ruff (format + check), pytest
- `uv run pytest` — run all tests
- `uv run pytest voice_pipeline/tests/asr` — run module tests
- `ruff check --fix && ruff format` — lint + format


## Coding Rules

- **Interfaces**: `I` prefix (`IASR`, `ITTS`). All defined in `core/interfaces.py`. Inject via constructor using interface types.
- **Vendor abstraction**: ASR, LLM, TTS are interface-backed. Impl selection via config.
- **Dependency direction**: always `module → core`. Modules must not import each other directly. TurnDetector does not know about SpeechGenerator or ASR; Orchestrator wires them.
- **Type hints** required. **Docstrings** required on interface methods.
- **Configuration**: `core/config.py` dataclass-based. Add fields as modules are implemented.
- **Logging**: `voice_pipeline.*` namespace (`voice_pipeline.asr`, `voice_pipeline.orchestrator`, etc.)


## Concurrency Model

threading + `queue.Queue` based.

- AudioInput: separate thread → `audio_queue`
- Orchestrator: main thread, frame-driven sync loop via `audio_queue.get(timeout=...)`
- SpeechGenerator: `concurrent.futures.ThreadPoolExecutor` for background LLM+TTS
- CppBridge: WebSocket receiver on separate thread → `event_queue` → Orchestrator consumes via `poll_event()`
- Minimize shared state between threads. Use `threading.Lock` when necessary.


## Error Handling

- Inside modules: handle transient errors with retries. Raise module exception when retries exhausted.
- Orchestrator fallback policy:
  - ASR / LLM / TTS failure → skip the current turn, stay in ACTIVE.
  - CppBridge disconnect → terminate session (→ FAREWELL → SLEEP). Reconnect attempted in GREETING before next session.
  - Audio starvation (no frames for `audio_starvation_timeout_sec`) → terminate session (→ FAREWELL → SLEEP).


## Testing

Tests must pass after each Phase before proceeding to the next.

### Test tiers

| Tier | File pattern | Marker | Default run | Purpose |
|------|-------------|--------|-------------|---------|
| Unit | `test_<module>.py` | (none) | Yes | Logic in isolation, external deps mocked |
| Integration | `test_<module>_integration.py` | `@pytest.mark.requires_api` | No | Real API/service verification |
| Stress | `test_<module>_stress.py` | `@pytest.mark.requires_api` | No | Load, duration, rapid-cycle scenarios |
| Cross-module | `tests/integration/test_*.py` | varies | varies | End-to-end flows spanning modules |

### Running tests

```bash
uv run pytest                                    # unit tests only (default)
uv run pytest -m requires_api                    # all real-service tests (integration + stress)
uv run pytest -m ''                              # everything
```

### Mocking rules

- **External services** (ASR, LLM, TTS, CppBridge): must be mocked in unit tests.
- **External model wrappers** (VAP, TurnGPT, Wakeword): mock the wrapper interface. Tests requiring real models use `@pytest.mark.requires_model`.

### Integration test conventions

- **Module-local**: Place in `tests/<module>/`, not `tests/integration/` (which is reserved for cross-module tests).
- **Environment variables** for test inputs (file paths, language codes, etc.) — never hardcoded.
- **Mirror orchestrator usage**: test the same call patterns the orchestrator will use (e.g., frame-by-frame feed+get, reset between turns, mid-stream stop).
- **Error recovery**: test real failure scenarios — invalid credentials, errors during streaming, recovery via reset/restart.


## Decision Log

Record non-obvious design decisions and lessons learned in `docs/decisions.md` (finalized) or `docs/decisions-wip.md` (in progress). Entries in `decisions-wip.md` are ordered chronologically (oldest on top, newest at the bottom).

Each entry should focus on **why this choice was made** — not what was built. Include:
- Trade-offs where alternatives existed
- Gotchas and constraints discovered through trial and error
- Context too broad for a code comment

Exclude:
- API design, schema structure, type definitions (readable from code)
- Obvious engineering patterns (error fallback, locking, etc.)
- Refactoring history (removed X, replaced with Y)


## Commit Convention

```
<type>(<scope>): <subject>
```

- **type**: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`
- **scope**: module name (`core`, `asr`, `tts`, `orchestrator`, …) or `project` for cross-cutting changes
- **subject**: lowercase, imperative, no period (e.g. `add ASR interface`)

Commit at natural checkpoints — when a task is complete and before starting the next one. Don’t split a single task into multiple commits.


