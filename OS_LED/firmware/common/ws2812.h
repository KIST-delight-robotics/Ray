#ifndef OS_LED_WS2812_H
#define OS_LED_WS2812_H

/*
 * WS2812B bit-bang driver for ATtiny85 @ 16 MHz (internal PLL).
 *
 * Timing per bit, 20 cycles @ 62.5 ns = 1.25 us:
 *   '0':  T0H = 5 cyc (312 ns)   T0L = 15 cyc (937 ns)
 *   '1':  T1H = 13 cyc (812 ns)  T1L = 7 cyc  (437 ns)
 *   RES:  LOW > 50 us between frames
 *
 * Driven on LED_DATA_PIN (PB1, see pins.h). Interrupts are disabled
 * during transmission; 24 LEDs is ~720 us total.
 */

#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>
#include <stdint.h>
#include "pins.h"

#ifndef F_CPU
#error "F_CPU must be defined as 16000000UL"
#endif

#define WS_NOP1  "nop      \n\t"
#define WS_NOP2  "rjmp .+0 \n\t"

/* Per-bit NOP padding (cycle-counted for 20 cyc total):
 *   w1=3 nops between OUT-HIGH and the bit test (T0H = 5)
 *   w2=6 nops between bit-test path and OUT-LOW   (T1H = 13)
 *   w3=3 nops as tail before dec/brne              (total = 20)
 */
static inline void ws2812_send_byte_(uint8_t byte, uint8_t maskhi, uint8_t masklo) {
    uint8_t ctr;
    asm volatile(
        "       ldi   %0, 8         \n\t"
        "ws_b%=:                     \n\t"
        "       out   %2, %3         \n\t"   /* cyc 1 : HIGH                      */
        WS_NOP1 WS_NOP2                       /* cyc 2-4: w1=3                    */
        "       sbrs  %1, 7          \n\t"   /* cyc 5 : skip OUT-LOW if bit=1    */
        "       out   %2, %4         \n\t"   /* cyc 6 : LOW (bit=0 only)         */
        "       lsl   %1             \n\t"   /* cyc 7 : shift                    */
        WS_NOP2 WS_NOP2 WS_NOP2               /* cyc 8-13: w2=6                   */
        "       out   %2, %4         \n\t"   /* cyc 14: LOW (bit=1 falling edge) */
        WS_NOP1 WS_NOP2                       /* cyc 15-17: w3=3                  */
        "       dec   %0             \n\t"   /* cyc 18                           */
        "       brne  ws_b%=         \n\t"   /* cyc 19-20 (taken)                */
        : "=&d" (ctr), "+r" (byte)
        : "I" (_SFR_IO_ADDR(PORTB)), "r" (maskhi), "r" (masklo)
    );
}

/* Send a GRB pixel array. led_count = number of LEDs (each is 3 bytes G,R,B). */
static inline void ws2812_send_grb(const uint8_t *grb, uint16_t led_count) {
    uint8_t sreg_prev = SREG;
    cli();

    uint8_t maskhi = PORTB |  _BV(LED_DATA_PIN);
    uint8_t masklo = PORTB & ~_BV(LED_DATA_PIN);
    DDRB |= _BV(LED_DATA_PIN);

    uint16_t bytes = led_count * 3;
    const uint8_t *p = grb;
    while (bytes--) {
        ws2812_send_byte_(*p++, maskhi, masklo);
    }

    SREG = sreg_prev;
    _delay_us(60);   /* >50 us LOW latches the frame */
}

#endif /* OS_LED_WS2812_H */
