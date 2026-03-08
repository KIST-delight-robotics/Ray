# Scratchpad

Claude's working memory. Free-form notes, observations, and context carried across sessions.

## Current Status

Phase 4 complete. All composite modules implemented:
- VAP wrapper, TurnGPT wrapper, TurnDetector (turn_taking/)
- SpeechGenerator (generation/)

Next: Phase 5 — Orchestrator.

## Phase 4 Notes

### SpeechGenerator threading model
- `max_workers=2` so cancelled runs don't block new runs during API I/O.
- Python generators can't be interrupted cross-thread during `next()`. Only cooperative cancel via `threading.Event` + run-ID guard.
- Per-run `queue.Queue` isolation prevents stale audio contamination.
