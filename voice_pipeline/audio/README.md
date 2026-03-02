# Audio Module

Audio capture and wakeword detection for the voice pipeline.

## Wakeword Detection

`WakewordDetector` detects trigger words in audio using Silero VAD for speech
segmentation and Google Cloud STT for keyword matching.

### Architecture

```
feed_audio(frame) → bool
  │
  ├── 1. Rechunk: 480-sample frames → 512-sample VAD chunks
  ├── 2. Silero VAD: speech probability per chunk
  ├── 3. State machine: IDLE → SPEECH → TRAILING → recognition
  └── 4. Google STT recognize() + word-boundary keyword match
```

### Usage

```python
from voice_pipeline.audio import WakewordDetector
from voice_pipeline.core.config import AudioConfig, WakewordConfig

config = WakewordConfig(keywords=("ray",))
audio_config = AudioConfig()  # 16kHz, mono, 16-bit

detector = WakewordDetector(config, audio_config)

# In your audio loop:
for frame in audio_frames:
    if detector.feed_audio(frame):
        print("Wakeword detected!")
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `keywords` | `("ray",)` | Tuple of trigger words |
| `vad_threshold` | `0.5` | VAD speech probability threshold |
| `language_code` | `"en-US"` | Google STT language code |
| `speech_pad_ms` | `300` | Trailing silence (ms) before triggering STT |
| `min_speech_duration_ms` | `100` | Ignore speech segments shorter than this |
| `max_speech_duration_sec` | `3.0` | Force STT after this duration |
| `stt_timeout_sec` | `5.0` | Timeout for Google STT `recognize()` call |

### Dependencies

- **silero-vad**: Silero VAD model (~2MB JIT). Requires PyTorch.
- **google-cloud-speech**: Google Cloud STT for keyword recognition.

### Error Handling

- **Initialization failures** (model load, client creation): raise `WakewordError`.
- **Runtime STT errors** (network, timeout): log warning, return `False` (fail closed).

### Testing

```bash
# Unit tests (mocked VAD + STT)
uv run pytest voice_pipeline/tests/audio/test_wakeword.py -v

# Integration tests (real models + API)
WAKEWORD_TEST_WAV=path/to/wakeword.wav \
  uv run pytest voice_pipeline/tests/audio/test_wakeword_integration.py -v -m "requires_api and requires_model"
```
