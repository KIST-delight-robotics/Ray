# LED Module

Visual feedback controller for the voice pipeline. Drives 24 WS2812 LEDs (8 bar + 16 ring) via SPI on Raspberry Pi 5. Falls back to logging-only mode when the hardware driver is not installed.

## LED States

| State | Animation | Description |
|-------|-----------|-------------|
| `OFF` | All black | Controller closed / inactive |
| `SLEEPING` | Breathing (ring only) | Sleep mode, bar LEDs off, ring fades in/out |
| `IDLE` | Static base color | Active session default |

Base color: `(233, 233, 50)`.

## Usage

```python
from voice_pipeline.led import LEDController
from voice_pipeline.core.config import LEDConfig
from voice_pipeline.core.types import LEDState

controller = LEDController(LEDConfig())

controller.set_state(LEDState.SLEEPING)
controller.set_state(LEDState.IDLE)
controller.set_state(LEDState.OFF)

controller.close()
```

## Architecture

```
LEDController
├── _strip: WS2812 strip | None    # hardware (optional)
├── _thread: Thread                 # animation loop (daemon)
├── _state: LEDState                # current state
├── _animations: dict               # state → LEDAnimation
└── _lock: Lock                     # thread safety
```

- **Animation thread**: daemon thread runs a render loop. Each tick calls `animation.render()` to produce an RGB frame, applies it to the strip, then sleeps for `frame_interval_sec`.
- **`set_state()`**: swaps the active animation, resets the tick counter, and calls `animation.reset()`. Thread-safe via lock.
- **Noop mode**: when `rpi5_ws2812` is not installed, `_strip` is `None` and frame application is skipped. The animation loop still runs (useful for debugging state transitions via logs).

## Custom Animations

Implement the `LEDAnimation` protocol:

```python
from voice_pipeline.led import LEDAnimation

class PulseAnimation:
    @property
    def frame_interval_sec(self) -> float:
        return 0.03

    def reset(self) -> None:
        pass

    def render(self, tick, bar_count, ring_count):
        brightness = abs((tick % 60) - 30) / 30.0
        v = int(brightness * 255)
        return [(0, 0, v)] * (bar_count + ring_count)
```

Pass a custom map to the controller:

```python
from voice_pipeline.core.types import LEDState

controller = LEDController(config, animations={
    LEDState.IDLE: PulseAnimation(),
    # ... other states
})
```

## Configuration

| Field | Default | Description |
|-------|---------|-------------|
| `bar_count` | `8` | LEDs in bar segment (indices 0-7) |
| `ring_count` | `16` | LEDs in ring segment (indices 8-23) |
| `spi_pin` | `10` | SPI GPIO pin (Pi 5 SPI0 MOSI) |
| `brightness` | `128` | Global brightness 0-255 |

## Hardware Driver

See [rpi5_ws2812_reference.md](rpi5_ws2812_reference.md) for the library API.

## Error Handling

- **Driver not installed**: noop mode, no error raised.
- **Driver init failure** (SPI access, permissions): raises `LEDError`.
- **Animation render errors**: logged and suppressed (fail-open for visual feedback).
