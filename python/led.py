from rpi5_ws2812.ws2812 import Color, WS2812SpiDriver
import logging

# ---------------------------------------------------------
# [전역 설정] 모듈이 import 될 때 딱 한 번 실행됨.
# ---------------------------------------------------------
LED_COUNT = 24
SPI_BUS = 0
SPI_DEVICE = 0

try:
    # 스트립 객체 생성 (전역 변수)
    _driver = WS2812SpiDriver(spi_bus=SPI_BUS, spi_device=SPI_DEVICE, led_count=LED_COUNT)
    strip = _driver.get_strip()
    strip.clear()  # 초기화 시 끄기
    strip.show()   # 적용
    logging.info(f"✅ LED Strip initialized with {LED_COUNT} LEDs.")
    
except Exception as e:
    logging.error(f"❌ LED 초기화 실패: {e}")
    strip = None  # 하드웨어가 없을 때 에러 방지용

# ---------------------------------------------------------
# [제어 함수] 외부에서 이 함수들을 호출하여 LED 제어.
# ---------------------------------------------------------

def led_set_pixel(index, r, g, b):
    """특정 LED 하나만 색상 변경"""
    if strip:
        strip.set_pixel_color(index, Color(r, g, b))
        strip.show()

def led_set_all(r, g, b):
    """전체 LED 색상 변경"""
    if strip:
        for i in range(LED_COUNT):
            strip.set_pixel_color(i, Color(r, g, b))
        strip.show()

def led_set_ring(r, g, b):
    """링 LED 색상 변경"""
    if strip:
        for i in range(8, 24):
            strip.set_pixel_color(i, Color(r, g, b))
        strip.show()

def led_set_bar(r, g, b):
    """바 LED 색상 변경"""
    if strip:
        for i in range(0, 8):
            strip.set_pixel_color(i, Color(r, g, b))
        strip.show()

def led_clear():
    """LED 끄기"""
    if strip:
        strip.clear()
        strip.show()

# 테스트용 코드 (직접 실행할 때만 작동)
if __name__ == "__main__":
    import time
    print("Testing LED...")
    led_set_all(255, 0, 0) # 빨강
    time.sleep(1)
    led_clear()