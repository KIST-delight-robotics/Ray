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
             ├── UtteranceTruncator
             └── ConversationHistory
```


## Repository Layout

Top-level only — for module details, inspect the folder directly (every module has
docstrings; modules wrapping external repos have their own README).

- `voice_pipeline/` — Python conversation pipeline (entry: `__main__.py`, wiring: `wiring.py`)
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
- **Dependency direction**: always `module → core`. Modules must not import each other directly. TurnDetector does not know about SpeechGenerator or ASR; `voice_pipeline/wiring.py` wires them.
- **Entry-point wiring**: production (`__main__.py`) and eval (`evaluation/run.py`) share the component graph via `voice_pipeline/wiring.py` (`build_components()` + `ProcessComponents.create_session()`). Production code exposes only neutral injection points (paths, toggles, callbacks) — never eval-specific behavior or branches. Dependency direction: `evaluation → voice_pipeline`, never the reverse.
- **Type hints** required. **Docstrings** required on interface methods.
- **Configuration**: per-module class variables and constructor parameters. No centralized config object.
- **Logging**: `voice_pipeline.*` namespace (`voice_pipeline.asr`, `voice_pipeline.session_loop`, etc.)


## Concurrency Model

threading + `queue.Queue` based.

- AudioInput: separate thread → `audio_queue`
- SessionLoop: main thread, frame-driven sync loop via `audio_queue.get(timeout=...)`
- SpeechGenerator: `concurrent.futures.ThreadPoolExecutor` for background LLM+TTS
- CppBridge: WebSocket receiver on separate thread → `event_queue` → SessionLoop consumes via `poll_event()`
- Minimize shared state between threads. Use `threading.Lock` when necessary.


## Error Handling

- Inside modules: handle transient errors with retries. Raise module exception when retries exhausted.
- SessionLoop fallback policy:
  - ASR / LLM / TTS failure → skip the current turn, stay in ACTIVE.
  - CppBridge disconnect → terminate session (→ FAREWELL → SLEEP). Reconnect attempted in GREETING before next session.
  - Audio starvation (no frames for `audio_starvation_timeout_sec`) → terminate session (→ FAREWELL → SLEEP).


## Testing

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
