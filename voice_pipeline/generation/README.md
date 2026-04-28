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

## `SpeechGenerator.__init__` 인자

| 인자 | Default | 의미 |
| --- | --- | --- |
| `context_builder` | — | LLM context 조립 모듈 |
| `llm` | — | LLM 인터페이스 |
| `tts` | — | TTS 인터페이스 |
| `executor` | `None` | 백그라운드 파이프라인 executor. `None`이면 `MAX_WORKERS` 기반 내부 생성. 외부 주입 시 shutdown()은 닫지 않음 |
| `retriever` | `None` | 메모리 retriever (optional) |
| `history` | `None` | retriever query 조립용 대화 이력 |
| `exclude_session_ids` | `None` | retriever가 제외할 세션 ID 집합 |

## 클래스 변수

| 변수 | 값 | 의미 |
| --- | --- | --- |
| `MAX_WORKERS` | `2` | 백그라운드 파이프라인 스레드 풀 크기 (외부 executor 공유용) |
| `_PIPELINE_MODE` | `"full"` | TTS 파이프라인 모드 (`"full"` / `"sentence"`) |
| `_QUERY_CONTEXT_TURNS` | `3` | 메모리 검색 query에 포함할 최근 history turn 수 |
| `_MIN_FLUSH_WORDS` | `4` | sentence 모드 TTS flush 최소 단어 수 |
| `_TTS_EXECUTOR_WORKERS` | `2` | sentence 모드 TTS 동시 합성 워커 수 |
| `_CONSUMER_JOIN_TIMEOUT_SEC` | `120.0` | sentence consumer · 문장 TTS future 대기 상한 (초) |
| `_CANCEL_POLL_INTERVAL_SEC` | `0.1` | consumer cancel_event 재확인 poll 주기 (초) |

## Module Structure

```
generation/
├── __init__.py
├── exceptions.py          # GenerationError, SpeechGeneratorError
├── speech_generator.py    # SpeechGenerator(ISpeechGenerator)
└── README.md
```
