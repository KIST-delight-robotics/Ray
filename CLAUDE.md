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
docs/
├── ARCHITECTURE.md
└── decisions.md

voice_pipeline/
├── core/
│   ├── types.py               # Shared data types (TurnDecision, ResponseData, etc.)
│   ├── interfaces.py          # All module interfaces (IASR, ITTS, ILLM, etc.)
│   ├── exceptions.py          # PipelineError base only
│   └── config.py              # Dataclass-based configuration
│
├── audio/
│   ├── audio_input.py         # Mic capture → audio_queue
│   ├── wakeword.py            # Wakeword detection (may wrap external engine)
│   └── exceptions.py
│
├── asr/
│   ├── asr.py                 # ASR interface impl
│   └── exceptions.py
│
├── turn_taking/
│   ├── vap.py                 # VAP wrapper (external model)
│   ├── turngpt.py             # TurnGPT wrapper (external model)
│   ├── turn_detector.py       # Combined turn decision
│   └── exceptions.py
│
├── llm/
│   ├── llm.py                 # LLM interface impl
│   ├── prompts.py             # Prompt templates
│   ├── tools.py               # Tool definitions & execution (planned)
│   ├── token_counter.py       # Token counting utilities
│   └── exceptions.py
│
├── tts/
│   ├── tts.py                 # TTS interface impl
│   ├── utterance_truncator.py # Barge-in text truncation strategies
│   └── exceptions.py
│
├── context/
│   └── context_builder.py     # LLM context assembly
│
├── history/
│   ├── conversation_history.py
│   └── storage_backend.py     # Persistence (memory / file / DB)
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
    ├── orchestrator/
    ├── session/
    └── integration/
```


## Documentation

- **SCRATCHPAD.md**: Claude's working memory. Read at session start, update freely.
- **decisions.md**: Append after each Phase. Record: key design choices and their rationale (especially when alternatives existed or it deviates from the original plan), gotchas/edge cases that affect integration or tuning, and constraints future phases should respect. Skip trivial implementation details obvious from reading the code.
- **Module READMEs** (`turn_taking/README.md`, etc.): Created when implementing the module. Covers external repo setup, constraints, config params.


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
  - CppBridge disconnect → terminate session (→ FAREWELL → SLEEP).


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


## Commit Convention

```
<type>(<scope>): <subject>
```

- **type**: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`
- **scope**: module name (`core`, `asr`, `tts`, `orchestrator`, …) or `project` for cross-cutting changes
- **subject**: lowercase, imperative, no period (e.g. `add ASR interface`)

Commit granularity upon completing a phase: make commits by module or by a meaningful unit of work. Don’t put an entire phase into a single commit.


## Implementation Order

```
Phase 1 — Foundation:    core/ (types, interfaces, exceptions, config)
Phase 2 — Independent:   history/, tts/utterance_truncator, context/
Phase 3 — External:      asr/, llm/, tts/, bridge/, audio/wakeword, led/
Phase 4 — Composite:     turn_taking/, generation/
Phase 5 — Orchestration: orchestrator/
Phase 6 — Top-level:     session/, audio/audio_input
Phase 7 — Integration tests
```


## Codex Review Workflow

### Plan review (before ExitPlanMode)
Before calling ExitPlanMode, always run a Codex review of the plan first:
1. Pass the full plan content to Codex via `bash ~/.claude/skills/codex/run_codex.sh`
2. Ask Codex to review the plan for risks, blind spots, and missing considerations
3. Present Codex's feedback alongside the plan to the user
4. Only then call ExitPlanMode for user confirmation

### Code review (after task completion)
After marking a task as completed (TaskUpdate status=completed), run a Codex code review:
1. Collect: brief task description + `git diff HEAD` output
2. Pass both to Codex via `bash ~/.claude/skills/codex/run_codex.sh`
3. Ask Codex to review for bugs, style issues, missing tests, and improvements
4. Present the review to the user

Add interfaces/types needed by the next Phase to `core/` before starting it. Phase 1 defines only confirmed shared types (TurnDecision, ResponseData, etc.).
