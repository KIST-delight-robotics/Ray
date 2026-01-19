from rpi5_ws2812.ws2812 import Color, WS2812SpiDriver
import logging
import time
import math

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

def led_smooth_wave(r, g, b, speed=4.0, focus=10.0):
    """
    사인파(Sine Wave)를 이용하여 부드러운 회전을 구현합니다.
    :param r, g, b: 색상
    :param speed: 회전 속도 (값이 클수록 빠름. 2.0 ~ 6.0 추천)
    :param focus: 빛의 집중도 (값이 클수록 점 하나만 밝고, 작으면 전체가 은은함)
    """
    if not strip:
        return
        
    ring_size = 8
    top_offset = 8
    bottom_offset = 16
    
    # 시작 위치를 12번으로 변경하기 위한 오프셋
    # 8번이 0번째이므로, 12번은 4번째 인덱스 (12 - 8 = 4)
    start_shift = 4 

    try:
        while True:
            # 현재 시간을 가져와서 끊김 없는 흐름을 만듦 (프레임 속도와 무관하게 부드러움)
            t = time.time() * speed
            
            for i in range(ring_size):
                # 1. 각 LED의 각도 계산 (0 ~ 2*PI)
                angle = ((i - start_shift) / ring_size) * 2 * math.pi
                
                # 2. 사인파 계산: sin(시간 + 각도)
                # 결과는 -1.0 ~ 1.0 사이를 오감
                wave = math.sin(t + angle)
                
                # 3. 값 보정: -1~1 범위를 0~1 범위로 변경
                brightness = (wave + 1) / 2
                
                # 4. 집중도(Focus) 적용 (지수 함수)
                # 단순히 0~1을 쓰는 게 아니라, 거듭제곱을 하면 
                # 밝은 부분은 유지되고 어두운 부분은 급격히 0에 가까워짐 -> 꼬리 효과 저절로 생성
                brightness = math.pow(brightness, focus)
                
                # 색상 적용
                cr = int(r * brightness)
                cg = int(g * brightness)
                cb = int(b * brightness)
                
                final_color = Color(cr, cg, cb)
                
                # 위/아래 동시 적용
                strip.set_pixel_color(top_offset + i, final_color)
                strip.set_pixel_color(bottom_offset + i, final_color)
            
            strip.show()
            # 연산 부하를 줄이기 위한 최소한의 대기 (너무 길면 끊겨 보이니 짧게)
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        led_clear()


def led_breathing(r, g, b, duration=0.01, repeat=3):
    """
    링 LED가 숨쉬듯이 부드럽게 밝아졌다 어두워지는 효과 (Breathing)
    :param r, g, b: 목표 색상 (최대 밝기일 때의 색)
    :param duration: 밝기 변화 단계별 대기 시간 (낮을수록 빠름, 기본 0.01)
    :param repeat: 반복 횟수
    """
    if not strip:
        return

    # 밝기 단계 (0% ~ 100% ~ 0% 로 변화시킬 단계 수)
    steps = 50 

    try:
        for _ in range(repeat):
            # 1. 서서히 밝아짐 (Fade In)
            for i in range(steps + 1):
                factor = i / steps  # 0.0 ~ 1.0 비율 계산
                
                # 비율에 따른 색상 계산 (int 변환 필수)
                cr = int(r * factor)
                cg = int(g * factor)
                cb = int(b * factor)
                
                # 링 LED 영역 (8번 ~ 23번)만 적용
                for pixel in range(8, 24):
                    strip.set_pixel_color(pixel, Color(cr, cg, cb))
                
                strip.show()
                time.sleep(duration)

            # (옵션) 최대 밝기에서 아주 잠깐 멈춤
            time.sleep(0.1)

            # 2. 서서히 어두워짐 (Fade Out)
            for i in range(steps, -1, -1):
                factor = i / steps  # 1.0 ~ 0.0 비율 계산
                
                cr = int(r * factor)
                cg = int(g * factor)
                cb = int(b * factor)
                
                for pixel in range(8, 24):
                    strip.set_pixel_color(pixel, Color(cr, cg, cb))
                
                strip.show()
                time.sleep(duration)
                
            # (옵션) 완전히 꺼진 상태에서 아주 잠깐 멈춤
            time.sleep(0.2)
            
    except KeyboardInterrupt:
        led_clear()

# 테스트용 코드 (직접 실행할 때만 작동)
if __name__ == "__main__":
    import time
    print("Testing LED...")
    led_set_ring(200, 200, 200)
    time.sleep(1)
    
    # for i in range(16, 24):
    #     strip.set_pixel_color(i, Color(255//2, 255//2, 50//2))
    # strip.show()
        
    # led_smooth_wave(255, 255, 50, speed=4.0, focus=15.0)
    # led_breathing(50, 50, 233, duration=0.01, repeat=5)
    time.sleep(10)
    # led_clear()