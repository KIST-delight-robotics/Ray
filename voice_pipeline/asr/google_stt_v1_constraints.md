# Google STT v1 API Constraints

## 1. Audio Input

### Encoding & Format

| Item | Value |
|------|-------|
| Supported encodings | LINEAR16, FLAC, MULAW, AMR, AMR_WB, OGG_OPUS, SPEEX_WITH_HEADER_BYTE, MP3, WEBM_OPUS |
| Sample rate | 8,000–48,000 Hz (`sample_rate_hertz`) |
| Channel count | LINEAR16/OGG_OPUS/FLAC: 1–8 channels (`audio_channel_count`) |

### Size & Duration Limits

| Method | Max duration | Max size |
|--------|-------------|----------|
| Synchronous (`Recognize`) | ~**1 min** | **10 MB** (local file) |
| Asynchronous (`LongRunningRecognize`) | ~**480 min** | No limit (when using GCS URI) |
| Streaming (`StreamingRecognize`) | ~**5 min** | **10 MB** (per message) |

- Audio longer than 1 minute via synchronous method requires a GCS URI.
- No file size limit when using GCS.

---

## 2. Streaming

### Protocol

- **gRPC only** — streaming is not available via REST API.

### Request Structure

- **First message**: must contain only `streaming_config`, no `audio_content`.
- **Subsequent messages**: must contain only `audio_content`, no `streaming_config`.

### Session Constraints

| Item | Value |
|------|-------|
| Max streaming duration | **5 minutes** (error returned if exceeded) |
| 60-second warning | `WARNING: Speech recognition request exceeded limit of 60 seconds` |
| Audio send rate | Must maintain approximately **real-time rate** |
| Request count per session | Counted as **1 request** regardless of frame count |

### Streaming Response (`StreamingRecognizeResponse`)

- `is_final: true` — finalized result (will not change)
- `is_final: false` — interim result (received when `interim_results: true`)
- `stability` — range 0.0 (unstable) to 1.0 (stable)
- `total_billed_time` — included only in the final response

### Key Config Fields (`StreamingRecognitionConfig`)

| Field | Description |
|-------|-------------|
| `single_utterance` | Auto-close stream on end-of-speech detection (supported by some models only) |
| `interim_results` | Whether to receive interim results |
| `enable_voice_activity_events` | Enable VAD (voice activity detection) events |
| `voice_activity_timeout` | Auto-close stream after sustained silence |

---

## 3. Output

### RecognitionConfig Constraints

| Item | Value |
|------|-------|
| `max_alternatives` | 0–**30** |
| `language_code` | Required (BCP-47 tag) |

### Speech Adaptation (Hints)

| Item | Limit |
|------|-------|
| Max phrases per request | **5,000** |
| Max total characters per request | **100,000** |
| Max characters per phrase | **100** |
| Transcript normalization entries | **100** |

---

## 4. Quotas

| Item | Value |
|------|-------|
| Recognition requests | **900/min** |
| Adaptation resource requests | **10/min** |
| Daily processing | **480 hours** |
| Daily quota reset | Midnight PST |

---

## References

- [Quotas and limits (v1)](https://docs.cloud.google.com/speech-to-text/docs/v1/quotas)
- [Transcribe streaming audio (v1)](https://docs.cloud.google.com/speech-to-text/docs/v1/transcribe-streaming-audio)
- [RPC reference: google.cloud.speech.v1](https://docs.cloud.google.com/speech-to-text/docs/reference/rpc/google.cloud.speech.v1)
