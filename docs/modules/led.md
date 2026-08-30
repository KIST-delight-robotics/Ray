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
from voice_pipeline.adapters.led import LEDController
from voice_pipeline.types import LEDState

controller = LEDController()

controller.set_state(LEDState.SLEEPING)
controller.set_state(LEDState.IDLE)
controller.set_state(LEDState.OFF)

controller.close()
```

## 클래스 변수

`LEDController` 내부 상수. 외부 참조 없음 (`_` prefix).

| 변수 | 값 | 의미 |
|------|------|------|
| `_BAR_COUNT` | `8` | 바 세그먼트 LED 개수 |
| `_RING_COUNT` | `16` | 링 세그먼트 LED 개수 |
| `_LED_COUNT` | `24` | 전체 LED 개수 |
| `_BRIGHTNESS` | `1.0` | LED 전체 밝기 (0.0=꺼짐, 1.0=최대) |
| `_NOOP_SLEEP_SEC` | `0.1` | 애니메이션 없을 때 스레드 폴링 간격 (초) |
| `_CLOSE_JOIN_TIMEOUT_SEC` | `2.0` | close 시 애니메이션 스레드 종료 대기 (초) |
| `_ANIMATIONS` | dict | 상태별 애니메이션 맵 |

`StaticAnimation` / `BreathingAnimation` 상수:

| 변수 | 값 | 의미 |
|------|------|------|
| `StaticAnimation._FRAME_INTERVAL_SEC` | `0.1` | 렌더 틱 간격 (초) |
| `BreathingAnimation._CYCLE_SEC` | `4.0` | 페이드 한 주기 시간 (초) |
| `BreathingAnimation._MIN_BRIGHTNESS` | `0.15` | 페이드 최소 밝기 (0.0~1.0) |
| `BreathingAnimation._FRAME_INTERVAL_SEC` | `0.03` | 렌더 틱 간격 (초) |

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
from voice_pipeline.adapters.led import LEDAnimation

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

To install a custom animation map, subclass `LEDController` and override `_ANIMATIONS`:

```python
from voice_pipeline.types import LEDState
from voice_pipeline.adapters.led import LEDController

class CustomLEDController(LEDController):
    _ANIMATIONS = {
        LEDState.IDLE: PulseAnimation(),
        # ... other states
    }
```

## Hardware Driver

See [rpi5_ws2812_reference.md](rpi5_ws2812_reference.md) for the library API.

## Error Handling

- **Driver not installed**: noop mode, no error raised.
- **Driver init failure** (SPI access, permissions): raises `RuntimeError`.
- **Animation render errors**: logged and suppressed (fail-open for visual feedback).
