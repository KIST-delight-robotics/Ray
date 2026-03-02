# Phase 3 — External Modules

## Vendors

| Module | Vendor | Notes |
|---|---|---|
| asr/ | Google Cloud Speech-to-Text | Streaming API, auth via `GOOGLE_APPLICATION_CREDENTIALS` |
| llm/ | OpenAI | Auth via `OPENAI_API_KEY` |
| tts/ | OpenAI TTS | |
| tts/ | ElevenLabs | May support word-level timestamps |
| bridge/ | WebSocket | Python=server, C++=client, JSON protocol |
| audio/wakeword | Silero VAD + Google STT | VAD → STT → keyword match |
| led/ | rpi5-ws2812 | RPi5 SPI only, include NoOp impl |

## Steps

| Step | Module | Status |
|------|--------|--------|
| 1 | core/ — add Phase 3 interfaces and configs | Done |
| 2 | asr/ | Done |
| 3 | llm/ | |
| 4 | tts/ (OpenAI + ElevenLabs) | |
| 5 | bridge/ | |
| 6 | audio/wakeword | |
| 7 | led/ | |

Step 1 is prerequisite. Steps 2-7 are independent.

## Notes

- Refer to official API docs per vendor. Do not reference legacy codebase.
- Each implementation conforms to its interface contract (format conversion, auth, etc. are internal).
- Tests follow unit (mock) + integration/stress (`@requires_api`) structure.
- Test inputs (file paths, API keys, language codes) via environment variables, never hardcoded.
- Each module gets a README (English, usage-focused). Vendor API constraints in a separate doc within the module directory.
