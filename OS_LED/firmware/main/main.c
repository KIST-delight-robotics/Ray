/*
 * OS_LED — ATtiny85 firmware (sustained-touch + WDT + auto-retry)
 *
 * Visual states:
 *   OFF        — strip dark
 *   BOOTING    — fade up 0→255, then deep slow breath 16↔255
 *   RUNNING    — Pi drives rainbow via os-led-display
 *   SHUTTING   — deep slow breath driven by ATtiny
 *   FADE_OFF   — 255→0 then OFF
 *
 * Touch behaviour:
 *   IDLE        : 0.5 s sustained → BOOT
 *   RUNNING     : 2.0 s sustained → SHUTDOWN
 *   BOOTING     : after 45 s wait, fresh touch → re-pulse J2 (Pi may have
 *                 missed the first PWR_BTN press due to PMIC debounce)
 *   SHUTTING    : any fresh touch → re-assert PB4 (kick daemon again)
 *   any state   : 5 s sustained hold → emergency soft reset (last resort)
 *
 * Auto-retry (no user input needed):
 *   - SHUTDOWN wait re-asserts PB4 every 10 s so a missed rising edge
 *     gets re-triggered without user knowing.
 *   - Slow Pi boot is given 45 s before any retry. Re-pulsing during
 *     active boot could trigger an unintended shutdown.
 *
 * Safety nets (defense in depth):
 *   - Explicit timeouts (5 min boot, 30 s shutdown) prevent unbounded waits.
 *   - Watchdog timer (8 s) hard-resets the chip if any code path hangs.
 *   - After reset, pi_ready is sampled before pins_init so we don't grab
 *     the LED line while Pi is driving it.
 *   - IDLE wait drains any active touch first (handles stuck-HIGH TTP223
 *     and finger held over from emergency hold).
 *
 * TTP223 must be in MOMENTARY mode (TOG grounded). 1.5 s post-power-on cal
 * delay lets baseline settle before we start reading touches.
 */

#define F_CPU 16000000UL

#include "../common/pins.h"
#include "../common/ws2812.h"
#include <avr/wdt.h>
#include <avr/interrupt.h>
#include <util/delay.h>
#include <stdint.h>

#define NUM_LEDS                 24
#define FADE_STEPS               64     /* main eased phase steps        */
#define FADE_STEP_MS             20     /* 64 * 20 = 1280 ms main phase  */
#define NPN_RELEASE_STEP          5     /* 5 * 20 = 100 ms = J2_PULSE_MS */
#define OVERSHOOT_DEPTH          40     /* dip from 255 down to 215      */
#define OVERSHOOT_DIP_STEP_MS     7     /* descent to dip                 */
#define OVERSHOOT_RETURN_STEP_MS 12     /* slower return — settle feel    */
#define ANTICIPATION_DEPTH       25     /* shutdown: dip 255 → 230        */
#define ANTICIPATION_DIP_STEP_MS  9
#define ANTICIPATION_RIS_STEP_MS  6     /* quick "kick" back to peak      */
#define TOUCH_POLL_MS      5
#define READY_POLL_MS     50
#define TOUCH_RELEASE_MS  50

#define PULSE_MIN         16
#define PULSE_MAX        255
#define PULSE_STEPS      128    /* one breath cycle */
#define PULSE_PEAK_IDX    64    /* parabola peaks at idx = N/2 */
#define PULSE_STEP_MS     16    /* 128 * 16 ≈ 2.0 s per breath cycle */
#define SHUTDOWN_PULSE_MS 1024  /* half cycle: 0 → peak; smooth from Pi's
                                   black/rainbow last frame to dim white,
                                   then ramp up to PULSE_MAX for fade_down */

#define BOOT_TOUCH_HOLD_MS       500
#define SHUTDOWN_TOUCH_HOLD_MS  2000
#define EMERGENCY_HOLD_MS       5000

#define BOOT_TIMEOUT_MS     300000UL
#define SHUTDOWN_TIMEOUT_MS  30000UL
#define COLD_BOOT_TIMEOUT_MS  90000UL  /* 12V auto-power-on: wait this long for READY before falling back to IDLE */
#define REBOOT_GRACE_MS       90000UL  /* sudo reboot: wait this long for Pi to return before staying off */

#define TTP223_CAL_MS       1500

/* Auto-retry tuning */
#define BOOT_RETRY_GRACE_MS         45000UL  /* don't re-pulse J2 before this — Pi may be slow booting */
#define BOOT_RETRY_COOLDOWN_MS       3000UL  /* min between J2 re-pulses (touch- or auto-triggered) */
#define BOOT_AUTO_RETRY_AFTER_MS    90000UL  /* start ATtiny-driven auto re-pulses after this */
#define BOOT_AUTO_RETRY_PERIOD_MS   30000UL  /* auto re-pulse cadence once auto-retry is active */
#define SHUTDOWN_RETRY_INTERVAL_MS  10000UL  /* auto re-pulse PB4 every 10 s */
#define DRAIN_TIMEOUT_MS            10000UL  /* if touch stuck HIGH this long, force_reset */

static uint8_t led_buf[NUM_LEDS * 3];

/* Debounced pi_ready — filters glitches/noise on the GPIO17 wire so a
 * brief EMI spike or jumper bounce doesn't fake "Pi died" and trigger an
 * unwanted SHUTDOWN sequence (the visible "white flash mid-rainbow"). */
#define PI_READY_DEBOUNCE_SAMPLES 5

static uint8_t pi_debounce_state = 0;
static uint8_t pi_debounce_flips = 0;

static void pi_ready_debounce_init(void) {
    pi_debounce_state = pi_ready();
    pi_debounce_flips = 0;
}

static uint8_t pi_ready_stable(void) {
    uint8_t raw = pi_ready();
    if (raw == pi_debounce_state) {
        pi_debounce_flips = 0;
    } else if (++pi_debounce_flips >= PI_READY_DEBOUNCE_SAMPLES) {
        pi_debounce_state = raw;
        pi_debounce_flips = 0;
    }
    return pi_debounce_state;
}

static void force_reset(void) {
    cli();
    wdt_enable(WDTO_15MS);
    for (;;) { }
}

static void fill_white(uint8_t level) {
    for (uint16_t i = 0; i < NUM_LEDS * 3; i++) {
        led_buf[i] = level;
    }
}

static void show(uint8_t level) {
    fill_white(level);
    ws2812_send_grb(led_buf, NUM_LEDS);
}

/* Boot fade-in with overshoot — premium appliance feel:
 *   Phase 1 ease-out parabola 0 → 255  (energetic power-on surge, ~1.3 s)
 *   Phase 2 brief dip 255 → 225          (overshoot recoil, ~280 ms)
 *   Phase 3 smooth return to 255         (settle, ~360 ms)
 * When pulse_j2 is set, NPN is released at NPN_RELEASE_STEP (~100 ms) during
 * Phase 1. pulse_j2=0 plays the same animation without touching J2 — used when
 * the Pi is already booting on its own (12V auto-power-on or sudo reboot). */
static void fade_up(uint8_t pulse_j2) {
    if (pulse_j2) PORTB |= NPN_MASK;

    for (uint16_t step = 0; step < FADE_STEPS; step++) {
        uint16_t prod = (uint16_t)step * (uint16_t)(2 * FADE_STEPS - step);
        uint8_t level = (uint8_t)(prod >> 4);
        show(level);
        _delay_ms(FADE_STEP_MS);
        wdt_reset();
        if (pulse_j2 && step == NPN_RELEASE_STEP) {
            PORTB &= ~NPN_MASK;
        }
    }
    show(255);

    for (uint8_t i = 1; i <= OVERSHOOT_DEPTH; i++) {
        show((uint8_t)(255 - i));
        _delay_ms(OVERSHOOT_DIP_STEP_MS);
        wdt_reset();
    }
    for (uint8_t i = OVERSHOOT_DEPTH; i > 0; i--) {
        show((uint8_t)(255 - i + 1));
        _delay_ms(OVERSHOOT_RETURN_STEP_MS);
        wdt_reset();
    }
    show(255);
}

/* Shutdown fade-out with anticipation — premium appliance feel:
 *   Phase 1 quick dip 255 → 230          (anticipation, ~225 ms)
 *   Phase 2 quick rise back 230 → 255    ("kick", ~150 ms)
 *   Phase 3 ease-in parabola 255 → 0     (lingering then snap off, ~1.3 s)
 * Total ≈ 1.7 s, asymmetric vs fade-in for a deliberate "winding-down" feel. */
static void fade_down(void) {
    for (uint8_t i = 1; i <= ANTICIPATION_DEPTH; i++) {
        show((uint8_t)(255 - i));
        _delay_ms(ANTICIPATION_DIP_STEP_MS);
        wdt_reset();
    }
    for (uint8_t i = ANTICIPATION_DEPTH; i > 0; i--) {
        show((uint8_t)(255 - i + 1));
        _delay_ms(ANTICIPATION_RIS_STEP_MS);
        wdt_reset();
    }
    show(255);

    for (uint16_t step = 0; step < FADE_STEPS; step++) {
        uint16_t sq = (uint16_t)step * (uint16_t)step;
        uint16_t shifted = sq >> 4;
        uint8_t darken = (shifted > 255) ? 255 : (uint8_t)shifted;
        show((uint8_t)(255 - darken));
        _delay_ms(FADE_STEP_MS);
        wdt_reset();
    }
    show(0);
}

/* Parabolic ease curve, peaks at idx = PULSE_PEAK_IDX. Returns brightness
 * in [PULSE_MIN, PULSE_MAX]. Approximates a sine breath — slow change at
 * peak/trough, faster in the middle — without trig or LUT. */
static uint8_t breath_level(uint8_t idx) {
    uint16_t p = (uint16_t)idx * (uint16_t)(PULSE_STEPS - idx);
    uint16_t scaled = (uint16_t)(((uint32_t)p * (PULSE_MAX - PULSE_MIN)) >> 12);
    return PULSE_MIN + (uint8_t)scaled;
}

/* Smooth breath while waiting for READY HIGH. Escape paths besides
 * 5-min timeout: 5 s emergency hold → force_reset; 45 s grace + fresh
 * touch → re-pulse J2; 90 s+ → auto re-pulse J2 every 30 s. */
static uint8_t pulse_until_ready_or_timeout(uint32_t timeout_ms, uint8_t allow_j2) {
    uint8_t idx = PULSE_PEAK_IDX;   /* start at peak — smooth handoff from fade_up */
    uint32_t elapsed = 0;
    uint16_t emerg_held = 0;
    uint8_t prev_touch = touch_active();
    uint32_t since_last_retry = 0;

    while (!pi_ready_stable()) {
        show(breath_level(idx));
        _delay_ms(PULSE_STEP_MS);
        wdt_reset();
        elapsed += PULSE_STEP_MS;
        since_last_retry += PULSE_STEP_MS;
        if (elapsed >= timeout_ms) return 0;

        uint8_t cur_touch = touch_active();
        if (cur_touch) {
            emerg_held += PULSE_STEP_MS;
            if (emerg_held >= EMERGENCY_HOLD_MS) force_reset();
        } else {
            emerg_held = 0;
        }

        if (allow_j2 && cur_touch && !prev_touch
            && elapsed >= BOOT_RETRY_GRACE_MS
            && since_last_retry >= BOOT_RETRY_COOLDOWN_MS) {
            pi_j2_pulse();
            since_last_retry = 0;
        }
        prev_touch = cur_touch;

        if (allow_j2 && elapsed >= BOOT_AUTO_RETRY_AFTER_MS
            && since_last_retry >= BOOT_AUTO_RETRY_PERIOD_MS) {
            pi_j2_pulse();
            since_last_retry = 0;
        }

        idx = (uint8_t)((idx + 1) & (PULSE_STEPS - 1));
    }
    show(PULSE_MAX);
    return 1;
}

/* Shutdown breath: start at idx=0 (PULSE_MIN, dim white) so the transition
 * from Pi's last frame (black if cleanup ran, rainbow otherwise) is a
 * brightness drop rather than a jarring jump to full white. Rises through
 * the parabola to PULSE_PEAK_IDX (PULSE_MAX) for clean fade_down handoff. */
static void pulse_for_ms(uint16_t duration_ms) {
    uint8_t idx = 0;
    uint16_t elapsed = 0;
    while (elapsed < duration_ms) {
        show(breath_level(idx));
        _delay_ms(PULSE_STEP_MS);
        wdt_reset();
        elapsed += PULSE_STEP_MS;
        idx = (uint8_t)((idx + 1) & (PULSE_STEPS - 1));
    }
    while (idx != PULSE_PEAK_IDX) {
        show(breath_level(idx));
        _delay_ms(PULSE_STEP_MS);
        wdt_reset();
        if (idx < PULSE_PEAK_IDX) idx++;
        else idx--;
    }
    show(PULSE_MAX);
}

/* Cycle PB4 LOW briefly then back HIGH so daemon sees a fresh rising edge. */
static void shutdown_req_retrigger(void) {
    PORTB &= ~SHUTDOWN_REQ_MASK;
    _delay_ms(100);
    PORTB |= SHUTDOWN_REQ_MASK;
}

/* Wait for Pi to drop READY LOW.
 * - 5 s emergency hold → force_reset
 * - Auto re-trigger PB4 every 10 s in case daemon missed first edge
 * - Fresh touch → immediate re-trigger (user kicking) */
static uint8_t wait_pi_off_or_timeout(uint32_t timeout_ms) {
    uint32_t elapsed = 0;
    uint16_t emerg_held = 0;
    uint32_t since_last_retry = 0;
    uint8_t prev_touch = touch_active();

    while (pi_ready_stable()) {
        _delay_ms(READY_POLL_MS);
        wdt_reset();
        elapsed += READY_POLL_MS;
        since_last_retry += READY_POLL_MS;
        if (elapsed >= timeout_ms) return 0;

        uint8_t cur_touch = touch_active();
        if (cur_touch) {
            emerg_held += READY_POLL_MS;
            if (emerg_held >= EMERGENCY_HOLD_MS) force_reset();
        } else {
            emerg_held = 0;
        }

        if (cur_touch && !prev_touch) {
            shutdown_req_retrigger();
            since_last_retry = 0;
        }
        prev_touch = cur_touch;

        if (since_last_retry >= SHUTDOWN_RETRY_INTERVAL_MS) {
            shutdown_req_retrigger();
            since_last_retry = 0;
        }
    }
    return 1;
}

/* IDLE wait: drain stuck/held-over touch first, then look for fresh
 * sustained press. Returns 0 if Pi unexpectedly comes up (auto-adopt).
 * If touch stays HIGH > DRAIN_TIMEOUT_MS (TTP223 likely glitched), force a
 * soft reset so the chip re-cals from scratch. */
static uint8_t wait_sustained_touch_or_pi_up(uint16_t hold_ms) {
    uint32_t drain_elapsed = 0;
    while (touch_active()) {
        if (pi_ready_stable()) return 0;
        _delay_ms(TOUCH_POLL_MS);
        wdt_reset();
        drain_elapsed += TOUCH_POLL_MS;
        if (drain_elapsed >= DRAIN_TIMEOUT_MS) force_reset();
    }
    _delay_ms(TOUCH_RELEASE_MS);
    wdt_reset();

    for (;;) {
        while (!touch_active()) {
            if (pi_ready_stable()) return 0;
            _delay_ms(TOUCH_POLL_MS);
            wdt_reset();
        }
        uint16_t held = 0;
        while (touch_active() && held < hold_ms) {
            if (pi_ready_stable()) return 0;
            _delay_ms(TOUCH_POLL_MS);
            wdt_reset();
            held += TOUCH_POLL_MS;
        }
        if (held >= hold_ms) return 1;
    }
}

static uint8_t wait_sustained_touch_or_pi_off(uint16_t hold_ms) {
    uint32_t drain_elapsed = 0;
    while (touch_active()) {
        if (!pi_ready_stable()) return 0;
        _delay_ms(TOUCH_POLL_MS);
        wdt_reset();
        drain_elapsed += TOUCH_POLL_MS;
        if (drain_elapsed >= DRAIN_TIMEOUT_MS) force_reset();
    }
    _delay_ms(TOUCH_RELEASE_MS);
    wdt_reset();
    for (;;) {
        while (!touch_active()) {
            if (!pi_ready_stable()) return 0;
            _delay_ms(TOUCH_POLL_MS);
            wdt_reset();
        }
        uint16_t held = 0;
        while (touch_active() && held < hold_ms) {
            if (!pi_ready_stable()) return 0;
            _delay_ms(TOUCH_POLL_MS);
            wdt_reset();
            held += TOUCH_POLL_MS;
        }
        if (held >= hold_ms) return 1;
    }
}

int main(void) {
    wdt_enable(WDTO_8S);
    _delay_ms(50);
    wdt_reset();
    pi_ready_debounce_init();
    uint8_t pi_was_up = (PINB & READY_MASK) ? 1 : 0;
    uint8_t pi_self_booting = 0;  /* set when Pi drops READY without a touch (reboot / sw-poweroff) */

    if (pi_was_up) {
        /* WDT or cold reset while Pi running — don't grab the LED line. */
        DDRB = NPN_MASK | SHUTDOWN_REQ_MASK;
        PORTB = 0;
        _delay_ms(TTP223_CAL_MS);
        wdt_reset();
        goto running_after_recovery;
    }

    pins_init();
    led_take_ownership();
    show(0);
    _delay_ms(TTP223_CAL_MS);
    wdt_reset();

    /* Cold power-on (12 V just applied). On boards that auto-power the Pi, it
     * is already booting on its own — play the boot animation and wait for
     * READY WITHOUT pulsing J2 (a J2 press could fight or abort the
     * in-progress boot). If READY never arrives within the window (auto-boot
     * disabled, or Pi held off), fade back down and fall through to the
     * touch-driven IDLE loop. */
    fade_up(0);
    if (pulse_until_ready_or_timeout(COLD_BOOT_TIMEOUT_MS, 0)) {
        led_release_ownership();
        goto running_after_recovery;
    }
    fade_down();

    for (;;) {
        /* ─── IDLE ─── */
        uint8_t want_boot = wait_sustained_touch_or_pi_up(BOOT_TOUCH_HOLD_MS);
        if (!want_boot) {
            led_release_ownership();
            goto running_after_recovery;
        }

        /* Final pi_ready check before pulsing J2 — Pi could have come up
         * during the 0.5 s hold. Pulsing J2 against a running Pi would
         * trigger an unwanted shutdown. */
        if (pi_ready_stable()) {
            led_release_ownership();
            goto running_after_recovery;
        }

        /* ─── BOOT (touch-driven; Pi is off, so pulse J2) ─── */
        fade_up(1);
        if (!pulse_until_ready_or_timeout(BOOT_TIMEOUT_MS, 1)) {
            fade_down();
            continue;
        }
        led_release_ownership();

running_after_recovery:
        /* ─── RUNNING → SHUTDOWN (retry on daemon failure) ─── */
        pi_self_booting = 0;
        for (;;) {
            uint8_t got_touch = wait_sustained_touch_or_pi_off(SHUTDOWN_TOUCH_HOLD_MS);
            if (!got_touch) { pi_self_booting = 1; break; }  /* Pi dropped READY without a touch = reboot / sw-poweroff */
            shutdown_req_assert();
            uint8_t off = wait_pi_off_or_timeout(SHUTDOWN_TIMEOUT_MS);
            shutdown_req_release();
            if (off) break;
        }

        /* ─── OFF ─── */
        led_take_ownership();
        pulse_for_ms(SHUTDOWN_PULSE_MS);
        fade_down();

        /* If the Pi went down without a touch (`sudo reboot`, or a software
         * `sudo poweroff`), it may be booting straight back up. Replay the boot
         * animation and wait for READY (no J2 — the Pi is coming up on its
         * own). If it returns within the grace window, hand the LED line back;
         * otherwise it was a real shutdown → stay off and drop into IDLE. */
        if (pi_self_booting) {
            fade_up(0);
            if (pulse_until_ready_or_timeout(REBOOT_GRACE_MS, 0)) {
                led_release_ownership();
                goto running_after_recovery;
            }
            fade_down();
        }
    }
}
