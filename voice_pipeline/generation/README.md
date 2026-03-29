# Generation Module

Background speech generation pipeline: ContextBuilder → LLM → TTS.

## Overview

`SpeechGenerator` runs the generation pipeline in a background thread. Orchestrator calls `prepare()` speculatively (on TurnDetector's `prepare` signal) and streams TTS audio via `poll_audio()` on `turn_shift`.

## State Flow

```
IDLE → PREPARING → STREAMING → IDLE
                 ↘ FAILED
       STREAMING ↘ FAILED
```

- **IDLE**: Ready for `prepare()`.
- **PREPARING**: Background thread running (context build → LLM → TTS setup).
- **STREAMING**: First TTS chunk arrived. `poll_audio()` returns audio, `get_text()` available.
- **FAILED**: LLM/TTS error, empty text, or zero TTS chunks. Orchestrator skips the turn.

## Usage Pattern

```python
# On TurnDetector prepare signal:
speech_generator.prepare(asr_text)

# On TurnDetector turn_shift:
if speech_generator.state == GeneratorState.STREAMING:
    playback_active = True

# Each frame while playback_active:
while (chunk := speech_generator.poll_audio()) is not None:
    bridge.send_audio(chunk)

if speech_generator.stream_done:
    response_data = speech_generator.get_response_data()
    # response_data.metrics_list contains LLMMetrics from each LLM call
    metrics = response_data.metrics_list[-1] if response_data.metrics_list else None
    history.add_assistant_message(response_data.text, metrics)
    playback_active = False
```

## Cancellation

- `prepare()` while PREPARING/STREAMING: cancels current run, starts new one.
- `cancel()`: stops current run, returns to IDLE.
- Mechanism: `threading.Event` polled between pipeline steps + run-ID guard on all state writes.
- Limitation: cannot interrupt blocking API calls (`next()` on LLM/TTS iterators). Bounded by API timeout configs.
- `max_workers=2` (default) ensures new runs start immediately while cancelled runs drain.

## Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_workers` | `2` | Thread pool size. 2 prevents new-run delay when cancelled run is blocked on API I/O. |

## Module Structure

```
generation/
├── __init__.py
├── exceptions.py          # GenerationError, SpeechGeneratorError
├── speech_generator.py    # SpeechGenerator(ISpeechGenerator)
└── README.md
```
