"""비동기 LED 애니메이션 효과."""

import math
import time
import asyncio

from hardware.led import led_set_ring_pixels, led_set_bar_pixels, led_clear, RING_SIZE


async def run_scanning_led_bar(r: int, g: int, b: int, speed: float = 0.08):
    """
    바 LED(0~7)가 좌우로 왕복하는 스캔 애니메이션 (Knight Rider 효과).
    asyncio.create_task()로 실행하고 task.cancel()로 중단합니다.
    """
    pos = 0
    direction = 1

    try:
        while True:
            colors = []
            for i in range(8):
                if i == pos:
                    colors.append((r, g, b))
                elif i == pos - direction and 0 <= i <= 7:
                    colors.append((r // 5, g // 5, b // 5))
                else:
                    colors.append((0, 0, 0))

            led_set_bar_pixels(colors)

            pos += direction
            if pos >= 7:
                direction = -1
            elif pos <= 0:
                direction = 1

            await asyncio.sleep(speed)

    except asyncio.CancelledError:
        raise


async def run_thinking_led_spin(r: int, g: int, b: int, speed: float = 4.0, focus: float = 10.0):
    """
    LLM 생각 중 표시를 위한 비동기 LED 애니메이션 (원형 회전).
    asyncio.create_task()로 실행하고 task.cancel()로 중단합니다.
    """
    start_shift = 4  # 12번 위치에서 시작하기 위한 오프셋

    try:
        while True:
            t = time.time() * speed
            colors = []

            for i in range(RING_SIZE):
                angle = ((i - start_shift) / RING_SIZE) * 2 * math.pi
                wave = math.sin(t + angle)
                brightness = math.pow((wave + 1) / 2, focus)

                cr = int(r * brightness)
                cg = int(g * brightness)
                cb = int(b * brightness)
                colors.append((cr, cg, cb))

            led_set_ring_pixels(colors)
            await asyncio.sleep(0.02)

    except asyncio.CancelledError:
        led_clear()
        raise
