# Scratchpad

Claude's working memory. Free-form notes, observations, and context carried across sessions.

## Current Status

Phase 5 complete. Orchestrator implemented:
- IConversationHistory updated with message IDs and update_message
- OrchestratorConfig added to config.py
- Orchestrator: frame-driven conversation loop with barge-in truncation (3 cases + deferred correction)
- 39 unit tests, 458 total tests passing

Next: Phase 6 — SessionManager + AudioInput.

## Phase 5 Notes

### Orchestrator frame loop ordering
- Decision before drain (step 5 before 7) — interrupt processed before sending more audio.
- Deferred truncation check (step 9) runs every frame to catch generator completion.

### Barge-in Case C (deferred truncation)
- Approximate truncation saved immediately (DurationRatioTruncator from sent buffer length).
- `_pending_truncation` holds msg_id + stop_position. Each frame checks generator.stream_done.
- On stream completion: re-truncate with full ResponseData, update_message to correct.
- Cleanup in 5 places: stream_done, FAILED, new streaming, new prepare, session end.

### IConversationHistory breaking change
- add_user_message/add_assistant_message now return int IDs.
- Updated StubHistory in context tests to match.
- get_messages() returns new dicts (no _id), not deepcopy.

## Phase 4 Notes

### SpeechGenerator threading model
- `max_workers=2` so cancelled runs don't block new runs during API I/O.
- Python generators can't be interrupted cross-thread during `next()`. Only cooperative cancel via `threading.Event` + run-ID guard.
- Per-run `queue.Queue` isolation prevents stale audio contamination.
