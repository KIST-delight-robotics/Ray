# Voice Conversation Robot — Project Guide

Python pipeline that handles real-time voice input, turn-taking detection, LLM response generation, and TTS synthesis.
Audio playback and motor control are handled by a C++ process, communicating over WebSocket.


## Collaboration Workflow

- **Discuss before implementing.** For non-trivial work, first resolve ambiguities and
  present a plan (or a proposed fix). Start code changes only after the user explicitly
  confirms. Answering the user's question is not approval to implement.
- **Questions get answers, not edits.** When the user asks a question or describes a
  problem, respond with findings and proposed solutions, then stop and wait.
- **Mid-work deviations**: if an important change from the agreed plan comes up, stop,
  explain it with alternatives, and continue only after the user decides.
- **Commits require explicit approval.** Never commit unless the user explicitly asks or
  approves that specific commit. Announcing an intent to commit is not approval.


## System Structure

```
SLEEP ──(wakeword)──▶ GREETING ──▶ ACTIVE ──(exit keyword/timeout)──▶ FAREWELL ──▶ SLEEP
```

```
__main__.py (mode loop + DI)
├─ AudioInput (separate thread → audio_queue)
├─ SLEEP:  audio_queue → WakewordDetector
└─ ACTIVE: SessionLoop
             ├── ASR
             ├── TurnDetector (VAP + TurnGPT)
             ├── SpeechGenerator (ContextBuilder → LLM → TTS)
             ├── CppBridge ⇄ WebSocket(:9200) ⇄ C++ Ray process (audio playback + motors)
             └── ConversationHistory
```


## Repository Layout

Top-level only — for module details, inspect the folder directly (every module has a docstring).

- `voice_pipeline/` — Python conversation pipeline. **Start with the `voice_pipeline/__init__.py`
  docstring** — it lists the files in reading order. Layout rule: external wrappers (vendors,
  hardware, external models) live in `adapters/` one file each; internal logic is top-level
  files (`session_loop.py`, `generator.py`, `prompt.py`, …); `memory/` is the only subpackage
  (an optional subsystem).
- `cpp/` — C++ audio playback + motor control process (see **C++ Process** below)
- `evaluation/` — E2E evaluation pipeline (audio prep, run, report, score, dashboard)
- `scripts/` — dev utilities, benchmarks (`bench/`), hardware checks (`hardware/`)
- `docs/` — project docs (see **Documentation** below)
- `data/` — datasets and runtime data. Gitignored as a whole, but a few required files are
  tracked (e.g. `data/segments/`, `data/eval/questions.json`). `git add` refuses paths under
  it; commit tracked files via pathspec: `git commit -- data/<path>`
- `logs/` — runtime logs (`pipeline/` Python; `motion/`, `pos4_audio/` C++)
- `models/`, `external/`, `third_party/` — model files and external repos
- `build/` — C++ build output


## C++ Process

Audio playback + motor control (`build/Ray`). The Python pipeline connects to it over
WebSocket (port 9200).

- **Build**: `cmake --build build --target Ray`
- **Run**: `RAY_UNIT` env var is **required** — selects the per-device motor home positions
  from `[robot.unitN]` in `cpp/config.toml`. Missing/typo → startup fails with a config error.
  Set it per device (e.g. `export RAY_UNIT=unit1` in `~/.bashrc`).
- **Config**: `cpp/config.toml` — shared params + per-unit sections. Do not keep per-device
  local edits; add or update a `[robot.unitN]` section instead.
- **Python-side dev without the robot**: run `scripts/mock_cpp_server.py` instead of `build/Ray`.


## Documentation

- **docs/SETUP.md**: Raspberry Pi 5 initial setup guide.
- **docs/decisions.md**: Finalized decision log for completed work.
- **docs/decisions-wip.md**: Decision log for work in progress. Merged into `decisions.md` after cleanup when the work is complete.
- **docs/ARCHITECTURE.md**: System architecture details.
- **docs/eval-system.md**: Evaluation system design.
- **docs/ray-memory/**: Long-term memory system design (overview, session, read, write, storage).
- **docs/troubleshooting/**: Hardware issue investigations (DAC I2S/DMA, ReSpeaker USB).
- **docs/benchmarks/**: Model benchmark reports (TurnGPT, VAP).
- **docs/modules/**: Setup guides for adapters with external repos/hardware/API constraints
  (asr, tts, turn_taking, wakeword, led, bridge protocol). Parameter tables belong in code, not here.


## External Model Dependencies

Some modules (VAP, TurnGPT, Wakeword etc.) wrap externally cloned model repositories.

- External repo APIs must not leak beyond the wrapper. The rest of the pipeline depends only on project interfaces.
- Wrappers accept model/repo path via config.
- Setup details (repo URL, version, install steps) are documented in `docs/modules/<name>.md`, not here.


## Environment & Commands

- Python 3.11+, uv (pyproject.toml), ruff (format + check), pytest
- `uv run pytest` — run all tests
- `uv run pytest voice_pipeline/tests/adapters/test_asr.py` — run one module's tests
- `ruff check --fix && ruff format` — lint + format


## Coding Rules

- **Interfaces only for vendor-swappable components**: `IASR`, `ILLM`, `ITTS`, `IEmbedder` (in
  `types.py`). Everything else is injected as its concrete class — do not add an ABC for a
  component with one implementation. Tests mock concrete classes with `Mock(spec=Class)`.
- **Where new code goes**: wrapping something external (vendor API, hardware, external model) →
  one file in `adapters/`. Internal logic → extend an existing top-level file, or add one new
  file. A new subpackage only for an optional subsystem like `memory/`. No per-module
  `exceptions.py` / `__init__.py` re-exports / README.
- **Dependency direction**: `adapters/` imports only `types`, `settings`, and `trace` (the
  recording API — used like `logging`, never injected). Top-level modules may import each other one-way (`session_loop → generator →
  prompt → history`); never the reverse, and never `adapters → top-level logic`. `wiring.py`
  is the only place that knows every component. `evaluation → voice_pipeline`, never the reverse.
- **Entry-point wiring**: production (`__main__.py`) and eval (`evaluation/run.py`) share the component graph via `voice_pipeline/wiring.py` (`build_components()` + `ProcessComponents.create_session()`). Production code exposes only neutral injection points (paths, toggles, callbacks) — never eval-specific behavior or branches. Dependency direction: `evaluation → voice_pipeline`, never the reverse.
- **Type hints** required. **Docstrings** required on interface methods.
- **Configuration**: vendor/module-specific knobs are class variables and constructor parameters.
  Values shared by several modules (audio format, DB path, token budgets) live in `settings.py`.
  Never reach into another module's private class variable.
- **Logging**: `voice_pipeline.*` namespace (`voice_pipeline.asr`, `voice_pipeline.session_loop`, etc.)
- **Tracing**: `trace.py` is a module-level API like `logging` — `record_call(...)` for an external
  call, `save_turn(...)` for a turn; `session_id`/`turn_index` come from `set_session`/`set_turn`
  context, never from constructor parameters. No-op when no sink is installed. Tests use the
  `call_log` / `turn_log` fixtures (`tests/conftest.py`).


## Concurrency Model

threading + `queue.Queue` based.

- AudioInput: separate thread → `audio_queue`
- SessionLoop: main thread, frame-driven sync loop via `audio_queue.get(timeout=...)`
- SpeechGenerator: `concurrent.futures.ThreadPoolExecutor` for background LLM+TTS
- CppBridge: WebSocket receiver on separate thread → `event_queue` → SessionLoop consumes via `poll_event()`
- Minimize shared state between threads. Use `threading.Lock` when necessary.


## Error Handling

- Inside modules: handle transient errors with retries. When exhausted, raise
  `RuntimeError("<what failed>: <cause>") from exc` — no custom exception classes (callers
  only ever catch `Exception`; add a class when a caller actually needs to distinguish).
- SessionLoop fallback policy:
  - ASR / LLM / TTS failure → skip the current turn, stay in ACTIVE.
  - CppBridge disconnect → terminate session (→ FAREWELL → SLEEP). Reconnect attempted in GREETING before next session.
  - Audio starvation (no frames for `audio_starvation_timeout_sec`) → terminate session (→ FAREWELL → SLEEP).


## Testing

### Test tiers

| Tier | File pattern | Marker | Default run | Purpose |
|------|-------------|--------|-------------|---------|
| Unit | `test_<file>.py` | (none) | Yes | Logic in isolation, external deps mocked |
| Integration | `test_<file>_integration.py` | `@pytest.mark.requires_api` | No | Real API/service verification |
| Stress | `test_<file>_stress.py` | `@pytest.mark.requires_api` | No | Load, duration, rapid-cycle scenarios |
| Cross-module | `tests/integration/test_*.py` | varies | varies | End-to-end flows spanning modules |

Tests mirror the source layout: `tests/adapters/` for adapters (shared fixtures in
`tests/adapters/conftest.py`), `tests/memory/`, and top-level `tests/test_<file>.py` for the rest.
Test doubles that tests need to inspect (e.g. recording call/trace stores) live in `tests/fakes.py`;
SQLite-backed stores use the `":memory:"` path in tests.

### Running tests

```bash
uv run pytest                                    # unit tests only (default)
uv run pytest -m requires_api                    # all real-service tests (integration + stress)
uv run pytest -m ''                              # everything
```

### Mocking rules

- **External services** (ASR, LLM, TTS, CppBridge): must be mocked in unit tests.
- **External model wrappers** (VAP, TurnGPT, Wakeword): mock the wrapper class (`Mock(spec=ThreadedVAP)`). Tests requiring real models use `@pytest.mark.requires_model`.

### Integration test conventions

- **Module-local**: Place next to the unit test (`tests/adapters/test_asr_integration.py`), not in `tests/integration/` (reserved for cross-module tests).
- **Environment variables** for test inputs (file paths, language codes, etc.) — never hardcoded.
- **Mirror SessionLoop usage**: test the same call patterns SessionLoop will use (e.g., frame-by-frame feed+get, reset between turns, mid-stream stop).
- **Error recovery**: test real failure scenarios — invalid credentials, errors during streaming, recovery via reset/restart.

### Bug reproduction

Do **not** add pytest test files to reproduce or verify bugs. Reproduce through the production code path instead: wire components via `voice_pipeline/wiring.py` (`build_components()` / `create_session()`) in a throwaway script, or use `evaluation/run.py --text` (LLM-only text mode, skips audio/turn-taking) for generation/context behavior.


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
- **scope**: module name (`core`, `asr`, `tts`, `session_loop`, …) or `project` for cross-cutting changes
- **subject**: 한글로 작성, 명사형 종결, 마침표 없음 (e.g. `ASR 인터페이스 추가`)

Propose a commit at natural checkpoints — when a task is complete and before starting the next one — and commit only after the user approves. Don’t split a single task into multiple commits.
