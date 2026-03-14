# Scratchpad

Claude's working memory. Read at session start, update freely.

## Current State

Phase 1–6 complete. All modules implemented and unit-tested (567 tests pass).
Async thread separation for VAP + TurnGPT complete.
`__main__.py` uses MaAIVAPWrapper (ONNX) as default VAP.

Remaining: Phase 7 (integration tests).

## What to do next

### Phase 7 — Integration tests (`tests/integration/`)

Cross-module end-to-end flows with mocked external services. Key scenarios:
1. **Full conversation cycle**: SessionManager SLEEP → wakeword → GREETING → ACTIVE (Orchestrator runs a turn) → exit keyword → FAREWELL → SLEEP
2. **Barge-in flow**: Mid-playback interrupt → STOP_PENDING → truncation → history updated
3. **Speculative prepare**: TurnDetector prepare → SpeechGenerator background run → turn_shift → immediate streaming
4. **Error recovery**: ASR failure mid-session → skip turn → continue. CppBridge disconnect → session terminates cleanly.
5. **Multi-session reuse**: Run two sessions on the same Orchestrator instance to verify state reset.

### Before real hardware testing

These known limitations should be addressed:
- **`generator.shutdown()` reuse** — Orchestrator calls `shutdown()` at session end, killing the ThreadPoolExecutor. Second session will fail. Fix: re-create executor in `prepare()` if shut down, or don't call `shutdown()` in `_end_session()`.
- **CppBridge reconnect** — Bridge only connected once at `run()` start. Needs reconnect before greeting or periodic health check.
- **Signal handling** — Add SIGINT/SIGTERM handler that calls `SessionManager.shutdown()`.

## Module quick reference

| Module | Key file | Interface | Config |
|--------|----------|-----------|--------|
| AudioInput | `audio/audio_input.py` | `IAudioInput` | `AudioInputConfig` |
| Wakeword | `audio/wakeword.py` | `IWakewordDetector` | `WakewordConfig` |
| ASR | `asr/asr.py` | `IASR` | `ASRConfig` |
| LLM | `llm/llm.py` | `ILLM` | `LLMConfig` |
| TTS | `tts/tts.py` | `ITTS` | `TTSConfig` |
| CppBridge | `bridge/cpp_bridge.py` | `ICppBridge` | `CppBridgeConfig` |
| LED | `led/led_controller.py` | `ILEDController` | `LEDConfig` |
| VAP | `turn_taking/vap.py` | `IVAP` | `VAPConfig` |
| MaAI VAP | `turn_taking/maai_vap.py` | `IVAP` | `MaAIVAPConfig` |
| AsyncVAP | `turn_taking/async_vap.py` | `IVAP` | — (wraps IVAP) |
| TurnGPT | `turn_taking/turngpt.py` | `ITurnGPT` | `TurnGPTConfig` |
| AsyncTurnGPT | `turn_taking/async_turngpt.py` | submit/poll | — (wraps ITurnGPT) |
| TurnDetector | `turn_taking/turn_detector.py` | `ITurnDetector` | `TurnDetectorConfig` |
| SpeechGenerator | `generation/speech_generator.py` | `ISpeechGenerator` | `SpeechGeneratorConfig` |
| ContextBuilder | `context/context_builder.py` | `IContextBuilder` | — |
| History | `history/conversation_history.py` | `IConversationHistory` | `ConversationHistoryConfig` |
| Truncator | `tts/utterance_truncator.py` | `IUtteranceTruncator` | — |
| Orchestrator | `orchestrator/orchestrator.py` | — (concrete) | `OrchestratorConfig` |
| SessionManager | `session/session_manager.py` | `ISessionManager` | `SessionConfig` |
