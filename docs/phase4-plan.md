# Phase 4 — Step Breakdown (turn_taking/, generation/)

Phase 4 builds two composite modules: **turn_taking/** (VAP + TurnGPT → turn decisions) and **generation/** (ContextBuilder → LLM → TTS → ResponseData). Detailed implementation notes saved in `memory/phase4_notes.md`.

Research: `docs/research_vap.md`, `docs/research_turngpt.md`

---

## Step 1 — Core Additions

Add Phase 4 interfaces, config, and types before module implementation.

| What | File |
|------|------|
| Interfaces | `core/interfaces.py` — add `IVAP`, `ITurnGPT`, `ITurnDetector`, `ISpeechGenerator` |
| Types | `core/types.py` — add `GeneratorState` enum (IDLE, PREPARING, READY, FAILED) |
| Config | `core/config.py` — add `VAPConfig`, `TurnGPTConfig`, `TurnDetectorConfig`, `SpeechGeneratorConfig`; update `PipelineConfig` |
| Tests | `tests/core/test_config.py` — verify new config defaults |

---

## Step 2 — VAP Wrapper

Wrap the external VoiceActivityProjection repo. Rolling audio buffer + periodic inference.

| What | File |
|------|------|
| Implementation | `turn_taking/vap.py` — `VAPWrapper(IVAP)` |
| Exceptions | `turn_taking/exceptions.py` — `TurnTakingError`, `VAPError`, `TurnGPTError`, `TurnDetectorError` |
| Tests | `tests/turn_taking/test_vap.py` — mock torch model |
| Docs | `turn_taking/README.md` — external repo setup for VAP and TurnGPT |
| Re-exports | `turn_taking/__init__.py` |

Key: stereo buffer `(1,2,n_samples)` at 16kHz, inference every `step_sec`, robot audio resampled from 24kHz.

---

## Step 3 — TurnGPT Wrapper

Wrap the external TurnGPT repo. Text → turn-shift probability.

| What | File |
|------|------|
| Implementation | `turn_taking/turngpt.py` — `TurnGPTWrapper(ITurnGPT)` |
| Tests | `tests/turn_taking/test_turngpt.py` — mock TurnGPT model |
| Re-exports | `turn_taking/__init__.py` |

Key: `<ts>`-delimited dialog text, `string_list_to_trp`, no trailing `<ts>` for partial turns.

---

## Step 4 — TurnDetector

Combine VAP + TurnGPT + timing into TurnDecision (turn_shift / interrupt / prepare / none).

| What | File |
|------|------|
| Implementation | `turn_taking/turn_detector.py` — `TurnDetector(ITurnDetector)` |
| Tests | `tests/turn_taking/test_turn_detector.py` — mock IVAP + ITurnGPT |
| Re-exports | `turn_taking/__init__.py` |

Key: priority-ordered logic (interrupt → text change → prepare → turn_shift), text similarity via SequenceMatcher, configurable thresholds.

---

## Step 5 — SpeechGenerator

Chain ContextBuilder → LLM → TTS with background preparation and cancellation.

| What | File |
|------|------|
| Implementation | `generation/speech_generator.py` — `SpeechGenerator(ISpeechGenerator)` |
| Exceptions | `generation/exceptions.py` — `GenerationError(PipelineError)` |
| Tests | `tests/generation/test_speech_generator.py` — mock IContextBuilder + ILLM + ITTS |
| Re-exports | `generation/__init__.py` |

Key: ThreadPoolExecutor, state machine (IDLE→PREPARING→READY / FAILED), cancel via threading.Event.

---

## Dependency Order

```
Step 1 ──▶ Step 2 ──▶ Step 3 ──▶ Step 4 ──▶ Step 5
           └── independent ──┘               └── independent of 2-4
```

Steps 2-3 are independent. Step 5 is independent of 2-4. All depend on Step 1.