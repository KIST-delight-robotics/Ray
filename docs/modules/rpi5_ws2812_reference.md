# rpi5-ws2812 Library API Reference

Driver library for WS2812 (NeoPixel) LEDs on Raspberry Pi 5 via SPI.

- **Repository**: https://github.com/niklasr22/rpi5-ws2812
- **PyPI**: https://pypi.org/project/rpi5-ws2812/
- **Install**: `pip install rpi5-ws2812`

## Quick Start

```python
from rpi5_ws2812.ws2812 import Color, WS2812SpiDriver

driver = WS2812SpiDriver(spi_bus=0, spi_device=0, led_count=24)
strip = driver.get_strip()
strip.set_brightness(0.5)  # 0.0 - 1.0

strip.set_pixel_color(0, Color(255, 0, 0))  # red
strip.set_all_pixels(Color(0, 0, 255))       # all blue
strip.show()                                  # flush to hardware
```

## API

### `WS2812SpiDriver(spi_bus, spi_device, led_count)`

Constructor. Opens the SPI device.

| Parameter | Type | Description |
|-----------|------|-------------|
| `spi_bus` | `int` | SPI bus number (typically `0`) |
| `spi_device` | `int` | SPI chip-select (typically `0` for CE0) |
| `led_count` | `int` | Total number of LEDs in the chain |

### `driver.get_strip() -> strip`

Returns the strip control object.

### `strip.set_brightness(value: float)`

Set global brightness. Range `0.0` (off) to `1.0` (full).

### `strip.set_pixel_color(index: int, color: Color)`

Set a single LED to the given color.

### `strip.set_all_pixels(color: Color)`

Set all LEDs to the same color.

### `strip.show()`

Flush the pixel buffer to hardware (SPI write).

### `Color(r: int, g: int, b: int)`

RGB color. Each channel 0-255. Library handles GRB wire encoding internally.

## Cleanup

No explicit `close()` method. Turn off LEDs before exit:

```python
strip.set_all_pixels(Color(0, 0, 0))
strip.show()
```

## Pi 5 SPI Setup

1. Enable SPI via `raspi-config` (Interface Options > SPI).
2. Wire WS2812 DIN to GPIO 10 (SPI0 MOSI, physical pin 19).
3. SPI device path: `/dev/spidev0.0` (bus=0, device=0).
4. User must have permission to access `/dev/spidev*` (spi group or root).
