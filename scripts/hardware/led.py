#!/usr/bin/env python3
"""LED 직접 제어 스크립트. 원하는 대로 수정해서 사용.

OS_LED 무지개 데몬과 자동 전환된다: 실행하면 무지개를 멈추고(토큰 획득) LED를
제어하고, 끝나면(또는 Ctrl+C/에러) 무지개가 다시 켜진다. 데몬이 없으면 단독 동작.

Usage:
    uv run python scripts/hardware/led.py
"""

import contextlib

from rpi5_ws2812.ws2812 import Color, WS2812SpiDriver

from voice_pipeline.led.arbiter_client import OSLedArbiterClient

BAR_COUNT = 8
RING_COUNT = 16
LED_COUNT = BAR_COUNT + RING_COUNT
BRIGHTNESS = 1.0  # 0.0 ~ 1.0

# 무지개 데몬에서 스트립을 빌린다(SPI 동시 사용 충돌 방지). 데몬이 없으면 no-op.
arbiter = OSLedArbiterClient()
arbiter.acquire()

driver = WS2812SpiDriver(spi_bus=0, spi_device=0, led_count=LED_COUNT)
strip = driver.get_strip()
strip.set_brightness(BRIGHTNESS)

try:
    # -- 켜기: 원하는 대로 수정 (색/범위) --
    strip.set_all_pixels(Color(233, 233, 50))   # 전체 켜기
    strip.show()

    # 다른 예시:
    # strip.set_pixel_color(0, Color(255, 0, 0)); strip.show()                          # 0번만 빨강
    # for i in range(BAR_COUNT): strip.set_pixel_color(i, Color(0, 0, 255))             # bar(0~7) 파랑
    # for i in range(BAR_COUNT, LED_COUNT): strip.set_pixel_color(i, Color(0, 255, 0))  # ring(8~23) 초록
    # strip.show()

    # -- 끌 때까지 LED 유지 (이 동안 무지개는 멈춤) --
    print("LED ON — 끄려면 Enter (또는 Ctrl+C)", flush=True)
    with contextlib.suppress(EOFError, KeyboardInterrupt):
        input()
finally:
    # 스트립을 무지개 데몬에 돌려준다: 소등 → SPI 완전 종료 → 토큰 해제.
    try:
        strip.set_all_pixels(Color(0, 0, 0))
        strip.show()
        driver._device.close()
    except Exception:
        pass
    arbiter.release()
