# Scratchpad

Claude's working memory. Read at session start, update freely.

## Current State

Phase 6 complete (reviewed + fixes applied). All runtime modules implemented:
- `audio/audio_input.py` — mic capture daemon thread
- `session/session_manager.py` — top-level state machine (implements `ISessionManager`)
- Orchestrator `request_stop()` for external stop signal
- Orchestrator `_start_session()` resets all internal state for clean reuse

## Phase Status

| Phase | Status |
|-------|--------|
| 1 — Foundation (core/) | Done |
| 2 — Independent (history, truncator, context) | Done |
| 3 — External (asr, llm, tts, bridge, wakeword, led) | Done |
| 4 — Composite (turn_taking, generation) | Done |
| 5 — Orchestration (orchestrator) | Done |
| 6 — Top-level (session, audio_input) | Done |
| 7 — Integration tests | Not started |

## Known Limitations (deferred)

- **`generator.shutdown()` reuse**: Orchestrator calls `generator.shutdown()` in `_end_session()`, terminating the ThreadPoolExecutor. Generator implementations must handle re-initialization on next `prepare()` or this will break multi-session reuse.
- **AudioInput thread death undetected**: If capture thread dies, queue starves silently. No health-check mechanism yet.
- **CppBridge reconnect**: `bridge.connect()` only called at `run()` start. If bridge disconnects mid-session and Orchestrator terminates → FAREWELL → SLEEP → next wakeword, bridge is not reconnected before greeting.
- **Signal handling**: No SIGINT/SIGTERM handler. `Ctrl+C` skips `shutdown()`.

## Open Items

- Integration tests (Phase 7) — cross-module end-to-end flows
- Real hardware testing with PyAudio
