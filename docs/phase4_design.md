# Phase 4 — Remaining Implementation Design

## TurnDetector Decision Logic

### Internal state
- `_prev_asr_text`, `_text_stable_since_ms`, `_prepare_fired`
- `_silence_frame_count`, `_has_robot_audio`

### Per-frame priority order
1. **Interrupt**: robot_audio present AND VAP `user_is_speaking=True`
2. **Text change detection**: SequenceMatcher ratio < threshold -> reset stability, clear prepare flag, reset silence count
3. **Prepare**: text stable for `prepare_stable_ms` AND TurnGPT > threshold AND not already fired
4. **Turn-shift**: VAP `user_is_speaking=False` for `turn_shift_silence_frames` OR hard silence timeout
5. **No-action**: TurnDecision.none()

### TurnGPT text building
- TurnDetector internally manages `<ts>`-formatted dialog for TurnGPT
- Orchestrator pushes completed turns via `notify_turn_complete(role, text)`
- `reset()` clears per-frame state only, NOT dialog context
- Dialog context format: `"user text<ts>robot text<ts>current partial"` (no trailing `<ts>`)

## SpeechGenerator

### State machine
```
IDLE --prepare()--> PREPARING --(done)--> READY --get_result()--> IDLE
                       |                    |
                    cancel()-->IDLE       cancel()-->IDLE
                       |
                   (error)-->FAILED --prepare()--> PREPARING
                                |
                             cancel()-->IDLE
```

### Background generation flow
1. `context_builder.build(text)` -> check cancel
2. `llm.generate(messages)` -> iterate chunks, check cancel between chunks
3. `full_text = "".join(chunks)` -> check cancel
4. `tts.synthesize(full_text)` -> consume stream, check cancel between chunks
5. Build `ResponseData(text, audio, timestamps)`
6. Under lock: store result, transition to READY

### Cancellation
- `threading.Event` per preparation cycle
- Check between LLM chunks, before TTS, between TTS chunks
- Close LLM iterator / TTS stream on cancel

### Error handling
- LLM/TTS errors -> catch, log, transition to FAILED
- Orchestrator detects FAILED -> skips turn, stays ACTIVE
- Next prepare() clears FAILED and starts fresh
