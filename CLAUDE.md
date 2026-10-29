# Voice Conversation Robot — Project Guide

Python pipeline that runs real-time voice conversation on OpenAI GPT-Live — one full-duplex model listens,
takes turns and speaks — with wakeword detection, long-term memory and device tools around it.
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
SLEEP ──(wakeword)──▶ GREETING ──▶ ACTIVE ──(exit keyword / end_conversation / idle timeout)──▶ FAREWELL ──▶ SLEEP
```

```
__main__.py (mode loop)
├─ AudioInput (separate thread → audio_queue)
├─ SLEEP:  audio_queue → WakewordDetector (Silero VAD + Google STT)
├─ ACTIVE: LiveSessionLoop (engines/gpt_live/loop.py)
│            ├── GPTLiveSession ⇄ OpenAI GPT-Live (conversation model + responses backend with function tools)
│            ├── CppBridge ⇄ WebSocket(:9200) ⇄ C++ Ray process (audio playback + motors)
│            ├── ConversationHistory + memory utterances (transcripts)
│            └── Radio (radio mode)
└─ after each session: MemoryWriter extracts long-term memories in the background
```


## Repository Layout

Top-level only — for module details, inspect the folder directly (every module has a docstring).

- `voice_pipeline/` — Python conversation pipeline. **Start with the `voice_pipeline/__init__.py`
  docstring** — it lists the files in reading order. Layout rule: external wrappers (vendors,
  hardware, external models) live in `adapters/` one file each; the conversation engine lives in
  `engines/gpt_live/`; what isn't engine-specific is a top-level file (`history.py`,
  `session_context.py`, `device_settings.py`, …); `memory/` is the optional long-term-memory subsystem.
- `cpp/` — C++ audio playback + motor control process (see **C++ Process** below)
- `scripts/` — dev utilities, benchmarks (`bench/`), hardware checks (`hardware/`)
- `boot/` — 부팅 시퀀스 일체: OS_LED 서브시스템, systemd 자동실행 유닛, 준비 완료 차임 (see `boot/README.md`)
- `docs/` — project docs (see **Documentation** below)
- `assets/` — files shipped with the repo and read by the programs (audio, motion CSVs, head
  segments). All tracked, except large music/vocal WAVs (distributed separately).
- `var/` — everything the programs write at runtime: `ray.db`, `log/` (`pipeline/` Python;
  `motion/`, `pos4_audio/` C++), `audio/` (TTS-generated greeting cache), `cache/`, `experiments/`
  (GPT-Live experiment results). Gitignored as a whole; nothing here is ever tracked.
- `config/` — files the operator writes per device (gitignored; only `*.example` tracked):
  `ray.env` (API keys, paths; read by `~/.bashrc` and the systemd units) and `robot.toml`
  (this device's motor home positions + sensor offset, read by the C++ process).
- `third_party/` — external C++ library checkouts
- `build/` — C++ build output


## C++ Process

Audio playback + motor control (`build/Ray`). The Python pipeline connects to it over
WebSocket (port 9200).

- **Build**: `cmake --build build --target Ray`
- **Config**: `cpp/config.toml` — params shared by every device (tracked). `config/robot.toml` —
  this device's motor home positions + sensor offset (gitignored, copy from
  `config/robot.toml.example`). Missing file/key → startup fails with a config error. Never put
  per-device values in `config.toml`.
- **Python-side dev without the robot**: run `scripts/mock_cpp_server.py` instead of `build/Ray`.


## Documentation

Docs are either **living** (kept in sync with the code) or **records** (frozen at the time of the work).

Living — update them when your change affects them:
- **docs/SETUP.md**: Raspberry Pi 5 initial setup guide.
- **docs/motor-calibration.md**: How the C++ motor calibration works.
- **docs/modules/**: Setup guides for adapters with external repos/hardware/API constraints
  (tts, wakeword, led, bridge protocol, vendor API constraints). Parameter tables belong in code, not here.
- **docs/troubleshooting/**: Hardware issue investigations, one file per issue — add to it if the issue recurs.
- **scripts/gpt_live/FINDINGS.md**: Observed GPT-Live API behavior.
- **docs/decisions-wip.md**: Decision log for work in progress (see **Decision Log**).

Records — never rewrite them to reflect later changes (see **Decision Log** for what may be added):
- **docs/history/**: One record per completed work worth revisiting, named after the work — `YYYY-MM-<work>.md`
  (month the work started), or a folder when it has notes. Its decisions, reasoning and trial-and-error. Read it
  when touching the same area, to learn why things are the way they are.
- **docs/archive/**: Records of removed or superseded designs (cascade engine, eval, story mode, initial
  memory design). Excluded from search via `.ignore`; open by path only when explicitly needed.


## Environment & Commands

- Python 3.11+, uv (pyproject.toml), ruff (format + check), pytest
- `uv run pytest` — run all tests
- `uv run pytest voice_pipeline/tests/adapters/test_wakeword.py` — run one module's tests
- `ruff check --fix && ruff format` — lint + format


## Coding Rules

- **Interfaces only for vendor-swappable components**: `ILLM`, `ITTS`, `IEmbedder` (in
  `types.py`). Everything else is injected as its concrete class — do not add an ABC for a
  component with one implementation. Tests mock concrete classes with `Mock(spec=Class)`.
- **Where new code goes**: wrapping something external (vendor API, hardware, external model) →
  one file in `adapters/`. Conversation-engine logic → `engines/gpt_live/` (backend tools go in
  `engines/gpt_live/tools.py`). Logic outside the engine → a top-level file. A new subpackage only
  for an optional subsystem like `memory/`. No per-module `exceptions.py` / `__init__.py`
  re-exports / README.
- **Dependency direction**: `adapters/` imports only `types`, `settings`, and `trace` (the
  recording API — used like `logging`, never injected). The engine imports top-level modules
  (`engines/gpt_live → session_context → memory`, `engines/gpt_live → history`), never the reverse.
  Inside the engine, one-way (`loop → radio / tools → instructions → songs`). Never
  `adapters → logic`. `wiring.py` is the only place that knows every component.
  `scripts → voice_pipeline`, never the reverse.
- **Entry-point wiring**: production (`__main__.py`) and experiment harnesses (`scripts/gpt_live/pipeline_harness.py`) share the component graph via `voice_pipeline/wiring.py` (`build_components()` + `ProcessComponents.create_session()`). Production code exposes only neutral injection points (paths, toggles, callbacks) — never harness-specific behavior or branches.
- **Type hints** required. **Docstrings** required on interface methods.
- **Comments and docstrings** describe current behavior and why — never change history ("removed X",
  "no longer Y", "since the experiment…"). Keep them short; history belongs in commits and the decision log.
  Don't point them at docs (`see docs/…`, `FINDINGS §N`) — docs may point to code, not the other way around.
  If the reason doesn't fit in a line or two, it belongs in the decision log, not the comment. Exception:
  operator-facing log messages may name a setup doc.
- **Configuration**: vendor/module-specific knobs are class variables and constructor parameters.
  Values shared by several modules (audio format, DB and asset paths, playback levels) live in `settings.py`.
  Never reach into another module's private class variable.
- **Logging**: `voice_pipeline.*` namespace (`voice_pipeline.live_session`, `voice_pipeline.memory`, etc.)
- **Tracing**: `trace.py` is a module-level API like `logging` — `record_call(...)` for an external
  call; `session_id` comes from the `set_session` context, never from constructor parameters. No-op
  when no sink is installed. Tests use the `call_log` fixture (`tests/conftest.py`).


## Concurrency Model

threading + `queue.Queue` based.

- AudioInput: separate thread → `audio_queue`
- LiveSessionLoop: main thread, frame-driven sync loop via `audio_queue.get(timeout=...)`
- GPTLiveSession: daemon receiver thread → event queue → the loop consumes via `poll_event()`
- Backend tool handlers and radio judges: a shared `ThreadPoolExecutor` (from `wiring.py`); the loop polls their futures
- CppBridge: WebSocket receiver on separate thread → `event_queue` → the loop consumes via `poll_event()`
- MemoryWriter: single-worker executor in `__main__.py`, one job per finished session
- Minimize shared state between threads. Use `threading.Lock` when necessary.


## Error Handling

- Inside modules: handle transient errors with retries. When exhausted, raise
  `RuntimeError("<what failed>: <cause>") from exc` — no custom exception classes (callers
  only ever catch `Exception`; add a class when a caller actually needs to distinguish).
- LiveSessionLoop fallback policy:
  - Backend tool handler failure → the error goes back to the backend as the tool output; the session continues.
  - GPT-Live session closed by the server → terminate session (→ FAREWELL → SLEEP).
  - CppBridge disconnect → terminate session (→ FAREWELL → SLEEP). Reconnect attempted in GREETING before next session.
  - Audio starvation (no mic frames for `_AUDIO_STARVATION_TIMEOUT_SEC`) → terminate session (→ FAREWELL → SLEEP).


## Testing

### Test tiers

| Tier | File pattern | Marker | Default run | Purpose |
|------|-------------|--------|-------------|---------|
| Unit | `test_<file>.py` | (none) | Yes | Logic in isolation, external deps mocked |
| Integration | `test_<file>_integration.py` | `@pytest.mark.requires_api` | No | Real API/service verification |
| Stress | `test_<file>_stress.py` | `@pytest.mark.requires_api` | No | Load, duration, rapid-cycle scenarios |

Tests mirror the source layout: `tests/adapters/` for adapters (shared fixtures in
`tests/adapters/conftest.py`), `tests/engines/gpt_live/` for the engine, `tests/memory/`, and
top-level `tests/test_<file>.py` for the rest.
Test doubles that tests need to inspect (e.g. the recording call store) live in `tests/fakes.py`;
SQLite-backed stores use the `":memory:"` path in tests.

### Running tests

```bash
uv run pytest                                    # unit tests only (default)
uv run pytest -m requires_api                    # all real-service tests (integration + stress)
uv run pytest -m ''                              # everything
```

### Mocking rules

- **External services** (GPT-Live session, LLM, TTS, Google STT, CppBridge): must be mocked in unit tests.
- **External model wrappers** (Wakeword, embedder): mock the wrapper class (`Mock(spec=WakewordDetector)`). Tests requiring real models use `@pytest.mark.requires_model`.

### Integration test conventions

- **Module-local**: Place next to the unit test (`tests/adapters/test_wakeword_integration.py`).
- **Environment variables** for test inputs (file paths, language codes, etc.) — never hardcoded.
- **Mirror session-loop usage**: test the same call patterns the session loop uses (e.g., frame-by-frame feed, `poll_event` draining, mid-stream close).
- **Error recovery**: test real failure scenarios — invalid credentials, errors during streaming, recovery via reset/restart.

### Bug reproduction

Do **not** add pytest test files to reproduce or verify bugs. Reproduce through the production code path instead: wire components via `voice_pipeline/wiring.py` (`build_components()` / `create_session()`) in a throwaway script, or run a session through the GPT-Live experiment harness (`scripts/gpt_live/pipeline_harness.py` — synthetic voice input, stubbed C++).


## Decision Log

Record non-obvious design decisions and lessons learned in `docs/decisions-wip.md` while the work is in
progress — in detail, as it happens, ordered chronologically (oldest on top, newest at the bottom). Open
follow-ups go under "차후 고려" at its end.

When the user decides a work is complete, move its entries into a record in `docs/history/` and remove them
from `decisions-wip.md`. Clean them up on the way: drop plain work logs (which file got what — git has that),
split crammed paragraphs, group field-test notes, and mark conclusions that were later overturned instead of
deleting them. Keep design intent, reasoning, trial-and-error and measurements; an overview may be added, but
only from what the entries and git history show.

A record is curated, not a commit log: write one only for work with design decisions or trial-and-error worth
revisiting — small changes need no record. Name it after the work, not the module (`2026-09-motor-calibration.md`,
not `motor.md`). A follow-up of the same work, or a fact missing from it, is appended to that record; a different
work — even in the same area — gets a new file. Never edit an existing record to reflect later changes.

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
- **scope**: module name (`gpt_live`, `memory`, `radio`, `bridge`, …) or `project` for cross-cutting changes
- **subject**: 한글로 작성, 명사형 종결, 마침표 없음 (e.g. `기억 검색에 날짜 범위 인자 추가`)

Propose a commit at natural checkpoints — when a task is complete and before starting the next one — and commit only after the user approves. Don’t split a single task into multiple commits.
