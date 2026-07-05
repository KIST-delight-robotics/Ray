#!/usr/bin/env python3
"""LED 직접 제어 스크립트. 원하는 대로 수정해서 사용.

Usage:
    uv run python scripts/hardware/led.py
"""

import time

from rpi5_ws2812.ws2812 import Color, WS2812SpiDriver

BAR_COUNT = 8
RING_COUNT = 16
LED_COUNT = BAR_COUNT + RING_COUNT
BRIGHTNESS = 1.0  # 0.0 ~ 1.0

driver = WS2812SpiDriver(spi_bus=0, spi_device=0, led_count=LED_COUNT)
strip = driver.get_strip()
strip.set_brightness(BRIGHTNESS)

# -- 여기서부터 원하는 대로 수정 --

# 전체 끄기
strip.set_all_pixels(Color(0, 0, 0))
strip.show()

# 전체 켜기
# strip.set_all_pixels(Color(233, 233, 50))
# strip.show()

# 특정 LED 하나만 (예: 0번 빨간색)
# strip.set_pixel_color(0, Color(255, 0, 0))
# strip.show()

# bar(0~7) 켜기
# for i in range(BAR_COUNT):
#     strip.set_pixel_color(i, Color(0, 0, 255))
# strip.show()

# ring(8~23) 켜기
# for i in range(BAR_COUNT, LED_COUNT):
#     strip.set_pixel_color(i, Color(0, 255, 0))
# strip.show()

# 3초 후 모두 끄기
time.sleep(3)
strip.set_all_pixels(Color(0, 0, 0))
strip.show()
