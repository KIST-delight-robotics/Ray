#ifndef OS_LED_PINS_H
#define OS_LED_PINS_H

/*
 * ATtiny85 pin map (DIP-8, top view, notch up)
 *
 *           ┌─────────────┐
 *   RESET 1 │● PB5    VCC│ 8  ──── +5V
 *     PB3 2 │           │ 7  PB2 ──▶ NPN base (Pi J2 pulse)
 *     PB4 3 │           │ 6  PB1 ──▶ WS2812 DIN (shared with Pi GPIO10)
 *     GND 4 │       PB0│ 5  ◀──── TTP223 OUT (PCINT0 wake)
 *           └─────────────┘
 *     PB3 ◀── Pi GPIO17 (READY input)
 *     PB4 ──▶ Pi GPIO27 (shutdown request) via 10k+20k voltage divider
 *
 * Clock: 16 MHz internal PLL
 *   lfuse = 0xE1   (CKSEL=0001 PLL, CKDIV8=off, SUT=10)
 *   hfuse = 0xDF   (factory default — keep RSTDISBL=1 to preserve ISP)
 *   efuse = 0xFF
 */

#include <avr/io.h>
#include <avr/interrupt.h>
#include <avr/sleep.h>
#include <util/delay.h>
#include <stdint.h>

/* ─── pin assignments ─────────────────────────── */
#define TTP223_PIN         PB0    /* INPUT,  PCINT0 wake source             */
#define LED_DATA_PIN       PB1    /* OUTPUT/INPUT (ownership gated)         */
#define NPN_PIN            PB2    /* OUTPUT, drives NPN base (J2 power-on)  */
#define READY_PIN          PB3    /* INPUT,  Pi-driven HIGH when up         */
#define SHUTDOWN_REQ_PIN   PB4    /* OUTPUT, asserted HIGH to ask Pi to off */

#define TTP223_MASK        _BV(TTP223_PIN)
#define LED_MASK           _BV(LED_DATA_PIN)
#define NPN_MASK           _BV(NPN_PIN)
#define READY_MASK         _BV(READY_PIN)
#define SHUTDOWN_REQ_MASK  _BV(SHUTDOWN_REQ_PIN)

/* J2 active-low pulse width: Pi's PWR_BTN debounce is ~50 ms,
 * 100 ms gives margin without entering long-press (5 s = hard off). */
#define J2_PULSE_MS    100

/* ─── pin setup ───────────────────────────────── */
static inline void pins_init(void) {
    /* Outputs: LED (initial owner during BOOTING), NPN, DEBUG.
     * Inputs : TTP223, READY. */
    DDRB  = LED_MASK | NPN_MASK | SHUTDOWN_REQ_MASK;

    /* All outputs LOW, no internal pull-ups on inputs.
     * READY needs external 10K pull-down — Pi GPIO is floating until userspace
     * drives it, and a floating input would chatter at the threshold. */
    PORTB = 0;
}

/* ─── LED line ownership (PB1 shared with Pi GPIO10) ─── */
static inline void led_take_ownership(void) {
    PORTB &= ~LED_MASK;   /* drive LOW first to avoid glitch */
    DDRB  |=  LED_MASK;
}

static inline void led_release_ownership(void) {
    DDRB  &= ~LED_MASK;   /* INPUT → high-Z */
    PORTB &= ~LED_MASK;   /* pull-up off (high-Z really)     */
}

/* ─── input helpers ───────────────────────────── */
static inline uint8_t touch_active(void) {
    return (PINB & TTP223_MASK) ? 1 : 0;
}

static inline uint8_t pi_ready(void) {
    return (PINB & READY_MASK) ? 1 : 0;
}

/* ─── J2 pulse (active-low, NPN open-collector style) ─── */
static inline void pi_j2_pulse(void) {
    PORTB |=  NPN_MASK;
    _delay_ms(J2_PULSE_MS);
    PORTB &= ~NPN_MASK;
}

/* ─── shutdown request (PB4 → Pi GPIO27 via divider) ─── */
static inline void shutdown_req_assert(void)  { PORTB |=  SHUTDOWN_REQ_MASK; }
static inline void shutdown_req_release(void) { PORTB &= ~SHUTDOWN_REQ_MASK; }

/* ─── PCINT0 wake source on TTP223 line ───────── */
static inline void pcint0_enable(void) {
    GIMSK |= _BV(PCIE);
    PCMSK |= _BV(PCINT0);
    sei();
}

/* ─── sleep (IDLE — peripherals keep running) ─── */
static inline void enter_sleep_idle(void) {
    set_sleep_mode(SLEEP_MODE_IDLE);
    sleep_enable();
    sleep_cpu();
    sleep_disable();
}

#endif /* OS_LED_PINS_H */
