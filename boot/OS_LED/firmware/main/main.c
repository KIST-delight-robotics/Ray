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
 * Self-initiated Pi shutdown (no touch — GUI/terminal reboot or poweroff):
 *   The ATtiny can't tell reboot from poweroff by READY alone, so the Pi's
 *   system-shutdown hook (pi/os-led-poweroff-ack) pulses READY HIGH 300 ms
 *   at the very end of a real poweroff. Firmware breathes while classifying:
 *   short READY pulse → poweroff → fade out immediately; READY held HIGH
 *   600 ms → reboot finished → hand LED line back. Neither within 120 s →
 *   assume dead, fade out. Requires the Pi-side hook to be installed and
 *   the daemon's HANDOFF_WAIT_S to exceed the 600 ms hold.
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
#include <avr/eeprom.h>

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

#define PULSE_MIN         38    /* = RAY 대기 디밍 최저(0.15*255)와 동일. Pi 데몬
                                   BREATH_MIN(0.15)·RAY BreathingAnimation과 동기 유지 */
#define PULSE_MAX        255
#define PULSE_STEPS      128    /* one breath cycle */
#define PULSE_PEAK_IDX    64    /* parabola peaks at idx = N/2 */
#define PULSE_STEP_MS     32    /* 128 * 32 ≈ 4.1 s per breath cycle */
#define HANDOFF_RAMP_STEP_MS  8   /* 부팅 인수 램프 전용 4배속 (ramp_breath_to_peak_at 주석 참고) */
#define SHUTDOWN_PULSE_MS 2048  /* half cycle: 0 → peak; smooth from Pi's
                                   black/rainbow last frame to dim colour,
                                   then ramp up to PULSE_MAX for fade_down */

/* Breath colour at full level (RGB). All animation levels 0..255 scale
 * this colour linearly, so show(255) == full (233,233,50). */
#define BREATH_R         233
#define BREATH_G         233
#define BREATH_B          50

#define BOOT_TOUCH_HOLD_MS       500
#define SHUTDOWN_TOUCH_HOLD_MS  2000
#define EMERGENCY_HOLD_MS       5000

#define BOOT_TIMEOUT_MS     300000UL
#define SHUTDOWN_TIMEOUT_MS  30000UL
#define COLD_BOOT_TIMEOUT_MS  90000UL  /* 12V auto-power-on: wait this long for READY before falling back to IDLE */
#define SELF_DOWN_TIMEOUT_MS 120000UL  /* no-touch READY drop: max wait for reboot-return / poweroff ACK */
#define READY_BOOT_HOLD_MS      600    /* READY high this long = Pi genuinely up → release LED line */
#define SHUTDOWN_IGNORE_BACKSTOP_MS 25000UL /* touch poweroff: ACK를 못 봐도 이 시간이면 완전 정지로 간주 */
#define ACK_SETTLE_MS            2000  /* ACK 후 커널 완전 정지까지 여유 — 이 전에 J2를 쏘면 삼켜질 수 있음 */
#define ACK_PULSE_MIN_MS         60    /* READY pulse in [MIN, HOLD) = poweroff ACK from Pi shutdown hook */

#define TTP223_CAL_MS       1500

/* Auto-retry tuning */
#define BOOT_RETRY_GRACE_MS         45000UL  /* don't re-pulse J2 before this — Pi may be slow booting */
#define BOOT_RETRY_COOLDOWN_MS       3000UL  /* min between J2 re-pulses (touch- or auto-triggered) */
#define BOOT_AUTO_RETRY_AFTER_MS    90000UL  /* start ATtiny-driven auto re-pulses after this */
#define BOOT_AUTO_RETRY_PERIOD_MS   30000UL  /* auto re-pulse cadence once auto-retry is active */
#define SHUTDOWN_RETRY_INTERVAL_MS  10000UL  /* auto re-pulse PB4 every 10 s */
#define DRAIN_TIMEOUT_MS            10000UL  /* if touch stuck HIGH this long, force_reset */

static uint8_t led_buf[NUM_LEDS * 3];

/* ─── EEPROM 블랙박스 ──────────────────────────────────────────
 * 512 B EEPROM에 이벤트 링버퍼를 남긴다. 로봇 보드에는 플래시 배선이 없어
 * 실기기 디버깅 수단이 없으므로, 문제 재현 후 칩을 프로그래머 보드로 옮겨
 *   avrdude ... -U eeprom:r:log.bin:r
 * 로 덤프해 되짚는 블랙박스다. (EESAVE 퓨즈를 설정하면 재플래시에도 보존 — make fuses)
 * 레이아웃: [0]=magic 0xA5, [1]=다음 쓰기 슬롯, [2..]=슬롯당 2 B {event, arg}. */
#define BB_MAGIC       0xA5
#define BB_SLOTS       250
#define BB_BASE        2

#define EV_POWERON     0x01   /* arg = MCUSR 리셋 원인 (bit0 PORF, 1 EXTRF, 2 BORF, 3 WDRF) */
#define EV_IDLE        0x02
#define EV_BOOT_J2     0x03   /* J2 펄스 발사 */
#define EV_RUNNING     0x04   /* LED 라인 반납, Pi 동작 확인 */
#define EV_TOUCH_OFF   0x05   /* 터치 종료 시작 */
#define EV_SELF_DOWN   0x06   /* 무터치 READY 드랍 (리부트/소프트 종료) */
#define EV_ACK         0x07   /* poweroff ACK 감지, arg = 펄스폭/10 ms */
#define EV_ACK_MISSED  0x08   /* 백스톱 타임아웃 — ACK를 끝내 못 봄 */
#define EV_REBOOT_BACK 0x09   /* READY 유지 — 리부트 복귀로 판정 */
#define EV_FADE_OUT    0x0A   /* 최종 소등 */

static void bb_log(uint8_t event, uint8_t arg) {
    uint8_t slot = eeprom_read_byte((uint8_t*)1);
    if (eeprom_read_byte((uint8_t*)0) != BB_MAGIC || slot >= BB_SLOTS) {
        eeprom_update_byte((uint8_t*)0, BB_MAGIC);
        slot = 0;
    }
    uint16_t addr = BB_BASE + (uint16_t)slot * 2;
    eeprom_update_byte((uint8_t*)addr, event);
    eeprom_update_byte((uint8_t*)(addr + 1), arg);
    eeprom_update_byte((uint8_t*)1, (uint8_t)((slot + 1) % BB_SLOTS));
}

/* ─── 비차단 ACK 감시자 ────────────────────────────────────────
 * 종료 연출(호흡/페이드) 도중에도 READY를 놓치지 않기 위한 누적식 펄스 분류기.
 * 애니메이션 루프들이 스텝마다 ack_poll(dt)을 불러주면, 블로킹 측정 없이
 * 상승→하강 에지에서 펄스폭을 분류한다. ACK가 연출 중에 도착해도 잡힌다
 * (예전에는 연출 3.7 s가 사각지대라 ACK를 놓치면 복불복이 났다). */
static uint16_t ack_high_ms = 0;
static uint8_t  ack_seen = 0;

static void ack_watch_reset(void) {
    ack_high_ms = 0;
    ack_seen = 0;
}

static void ack_poll(uint8_t dt_ms) {
    if (pi_ready()) {
        if (ack_high_ms < 60000) ack_high_ms += dt_ms;
    } else {
        if (ack_high_ms >= ACK_PULSE_MIN_MS && ack_high_ms < READY_BOOT_HOLD_MS && !ack_seen) {
            ack_seen = 1;
            bb_log(EV_ACK, (uint8_t)(ack_high_ms / 10));
        }
        ack_high_ms = 0;
    }
}

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

/* Strict "Pi is genuinely up" check: READY must stay HIGH continuously for
 * READY_BOOT_HOLD_MS. Filters out the Pi's 300 ms poweroff-ACK pulse (see
 * os-led-poweroff-ack shutdown hook), which would otherwise be mistaken for
 * a boot and adopt a dead Pi. Used at IDLE adoption points; the boot-wait
 * paths keep the fast 80 ms debounce (no ACK pulse can occur mid-boot). */
static uint8_t pi_ready_adopt(void) {
    if (!pi_ready()) return 0;
    uint16_t held = 0;
    while (pi_ready()) {
        _delay_ms(TOUCH_POLL_MS);
        wdt_reset();
        held += TOUCH_POLL_MS;
        if (held >= READY_BOOT_HOLD_MS) return 1;
    }
    return 0;   /* went LOW before the hold — was a pulse/glitch, not a boot */
}

static void force_reset(void) {
    cli();
    wdt_enable(WDTO_15MS);
    for (;;) { }
}

static void fill_color(uint8_t level) {
    uint8_t g = (uint8_t)(((uint16_t)BREATH_G * level) / 255);
    uint8_t r = (uint8_t)(((uint16_t)BREATH_R * level) / 255);
    uint8_t b = (uint8_t)(((uint16_t)BREATH_B * level) / 255);
    for (uint16_t i = 0; i < NUM_LEDS * 3; i += 3) {
        led_buf[i]     = g;    /* WS2812 wire order: G, R, B */
        led_buf[i + 1] = r;
        led_buf[i + 2] = b;
    }
}

static void show(uint8_t level) {
    fill_color(level);
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
    if (pulse_j2) {
        bb_log(EV_BOOT_J2, 0);
        PORTB |= NPN_MASK;
    }

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
        ack_poll(ANTICIPATION_DIP_STEP_MS);
    }
    for (uint8_t i = ANTICIPATION_DEPTH; i > 0; i--) {
        show((uint8_t)(255 - i + 1));
        _delay_ms(ANTICIPATION_RIS_STEP_MS);
        wdt_reset();
        ack_poll(ANTICIPATION_RIS_STEP_MS);
    }
    show(255);

    for (uint16_t step = 0; step < FADE_STEPS; step++) {
        uint16_t sq = (uint16_t)step * (uint16_t)step;
        uint16_t shifted = sq >> 4;
        uint8_t darken = (shifted > 255) ? 255 : (uint8_t)shifted;
        show((uint8_t)(255 - darken));
        _delay_ms(FADE_STEP_MS);
        wdt_reset();
        ack_poll(FADE_STEP_MS);
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
    /* READY 확인 — 현재 위상에서 피크까지 빠르게 차올린 뒤 정지(인수 프레임).
     * 예전의 show(PULSE_MAX) 즉시 점프는 어두운 위상에서 걸리면 6%→100% 스냅이 보였다. */
    ramp_breath_to_peak_at(idx, HANDOFF_RAMP_STEP_MS);
    return 1;
}

/* _delay_ms는 컴파일타임 상수만 정확하므로, 가변 ms는 1 ms 단위로 돈다. */
static void delay_ms_var(uint8_t ms) {
    while (ms--) _delay_ms(1);
}

/* Walk the breath from wherever it is to the parabola peak (PULSE_MAX) so
 * the next frame (fade_down, or the Pi daemon's peak-start breathing) picks
 * up without a brightness jump. step_ms: 종료 경로는 호흡 속도(PULSE_STEP_MS),
 * 부팅 인수 경로는 HANDOFF_RAMP_STEP_MS(4배속) — Pi가 READY HIGH 후
 * HANDOFF_WAIT(1.2 s) 만에 SPI를 잡으므로 그 안에 램프+해제가 끝나야 한다
 * (최악 64스텝×8 ms=512 ms + 디바운스 80 ms < 1.2 s). */
static void ramp_breath_to_peak_at(uint8_t idx, uint8_t step_ms) {
    while (idx != PULSE_PEAK_IDX) {
        show(breath_level(idx));
        delay_ms_var(step_ms);
        wdt_reset();
        if (idx < PULSE_PEAK_IDX) idx++;
        else idx--;
    }
    show(PULSE_MAX);
}

static void ramp_breath_to_peak(uint8_t idx) {
    ramp_breath_to_peak_at(idx, PULSE_STEP_MS);
}

/* Shutdown breath: start at idx=0 (PULSE_MIN, dim colour) so the transition
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
        ack_poll(PULSE_STEP_MS);   /* 연출 중에도 ACK 감시 (사각지대 제거) */
        elapsed += PULSE_STEP_MS;
        idx = (uint8_t)((idx + 1) & (PULSE_STEPS - 1));
    }
    ramp_breath_to_peak(idx);
}

/* After a no-touch READY drop (GUI/terminal reboot OR poweroff — the ATtiny
 * can't tell which yet), breathe and watch READY to classify:
 *   - READY held HIGH >= READY_BOOT_HOLD_MS  → Pi rebooted and its daemon is
 *     back up. Return 1 (caller releases the LED line).
 *   - READY pulse in [ACK_PULSE_MIN_MS, HOLD) → the Pi's system-shutdown hook
 *     saying "this is a real poweroff". Return 0 (caller fades out NOW).
 *   - SELF_DOWN_TIMEOUT_MS with neither       → assume dead (hook missing or
 *     crash). Return 0.
 * On return 0 the breath is ramped to peak so fade_down starts from 255. */
static uint8_t self_down_wait_classify(void) {
    uint8_t idx = 0;    /* start dim — smooth from Pi's last (black) frame */
    uint32_t elapsed = 0;
    uint16_t emerg_held = 0;

    while (elapsed < SELF_DOWN_TIMEOUT_MS) {
        show(breath_level(idx));
        _delay_ms(PULSE_STEP_MS);
        wdt_reset();
        elapsed += PULSE_STEP_MS;
        idx = (uint8_t)((idx + 1) & (PULSE_STEPS - 1));

        if (touch_active()) {
            emerg_held += PULSE_STEP_MS;
            if (emerg_held >= EMERGENCY_HOLD_MS) force_reset();
        } else {
            emerg_held = 0;
        }

        if (pi_ready()) {
            /* READY rose: measure how long it stays HIGH, still breathing. */
            uint16_t high_ms = 0;
            while (pi_ready() && high_ms < READY_BOOT_HOLD_MS) {
                show(breath_level(idx));
                _delay_ms(PULSE_STEP_MS);
                wdt_reset();
                high_ms += PULSE_STEP_MS;
                elapsed += PULSE_STEP_MS;
                idx = (uint8_t)((idx + 1) & (PULSE_STEPS - 1));
            }
            if (high_ms >= READY_BOOT_HOLD_MS) {
                /* 재부팅 복귀 — 피크까지 빠른 램프 후 정지 (즉시 점프 방지) */
                ramp_breath_to_peak_at(idx, HANDOFF_RAMP_STEP_MS);
                return 1;          /* reboot complete — Pi is driving soon   */
            }
            if (high_ms >= ACK_PULSE_MIN_MS) {
                ramp_breath_to_peak(idx);
                return 0;          /* poweroff ACK — shutdown is final       */
            }
            /* shorter than ACK_PULSE_MIN_MS: noise — keep waiting */
        }
    }
    ramp_breath_to_peak(idx);
    return 0;
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
        if (pi_ready_adopt()) return 0;
        _delay_ms(TOUCH_POLL_MS);
        wdt_reset();
        drain_elapsed += TOUCH_POLL_MS;
        if (drain_elapsed >= DRAIN_TIMEOUT_MS) force_reset();
    }
    _delay_ms(TOUCH_RELEASE_MS);
    wdt_reset();

    for (;;) {
        uint16_t reblack_ms = 0;
        while (!touch_active()) {
            if (pi_ready_adopt()) return 0;
            _delay_ms(TOUCH_POLL_MS);
            wdt_reset();
            /* 소등 대기 중 1초마다 검정 재전송: 라인/전원 노이즈로 아무 픽셀이
             * 쓰레기 색을 래치해도 다음 재전송에서 지워진다 (유령 픽셀 방지). */
            reblack_ms += TOUCH_POLL_MS;
            if (reblack_ms >= 1000) {
                show(0);
                reblack_ms = 0;
            }
        }
        uint16_t held = 0;
        while (touch_active() && held < hold_ms) {
            if (pi_ready_adopt()) return 0;
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
    uint8_t reset_cause = MCUSR;   /* bit0 PORF, 1 EXTRF, 2 BORF, 3 WDRF */
    MCUSR = 0;
    wdt_enable(WDTO_8S);
    bb_log(EV_POWERON, reset_cause);
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
        bb_log(EV_IDLE, 0);
        uint8_t want_boot = wait_sustained_touch_or_pi_up(BOOT_TOUCH_HOLD_MS);
        if (!want_boot) {
            led_release_ownership();
            goto running_after_recovery;
        }

        /* Final pi_ready check before pulsing J2 — Pi could have come up
         * during the 0.5 s hold. Pulsing J2 against a running Pi would
         * trigger an unwanted shutdown. */
        if (pi_ready_adopt()) {
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
        bb_log(EV_RUNNING, 0);
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
        if (pi_self_booting) {
            /* Pi went down on its own (GUI/terminal reboot OR poweroff).
             * Breathe while the Pi tells us which it was:
             *  - poweroff → shutdown hook pulses READY 300 ms → fade out NOW
             *  - reboot   → READY comes back and stays HIGH → hand line back */
            bb_log(EV_SELF_DOWN, 0);
            if (self_down_wait_classify()) {
                led_release_ownership();
                goto running_after_recovery;
            }
            fade_down();
        } else {
            /* Touch-initiated shutdown. Pi가 종료를 마칠 때까지(ACK + 정착 2 s,
             * 또는 백스톱 25 s) 터치를 무시한다 — 이 창에서 J2를 쏘면 죽어가는
             * OS가 삼켜 헛발이 되기 때문. ACK 감시는 종료 연출 중에도 ack_poll로
             * 이어지므로 사각지대가 없고, 무시 구간이 끝나면 IDLE로 돌아가
             * 터치 0.5 s = 정상 부팅이 된다. (터치 큐잉 방식은 폐기 — 상태 표시
             * 없는 대기가 제품 UX로 부적절, 2026-09) */
            bb_log(EV_TOUCH_OFF, 0);
            ack_watch_reset();
            pulse_for_ms(SHUTDOWN_PULSE_MS);
            fade_down();
            bb_log(EV_FADE_OUT, 0);
            uint32_t ignore_elapsed = 0;
            uint16_t emerg_held = 0;
            while (!ack_seen && ignore_elapsed < SHUTDOWN_IGNORE_BACKSTOP_MS) {
                _delay_ms(TOUCH_POLL_MS);
                wdt_reset();
                ack_poll(TOUCH_POLL_MS);
                ignore_elapsed += TOUCH_POLL_MS;
                if (touch_active()) {          /* 무시 — 단 5 s 비상 리셋 탈출구는 유지 */
                    emerg_held += TOUCH_POLL_MS;
                    if (emerg_held >= EMERGENCY_HOLD_MS) force_reset();
                } else {
                    emerg_held = 0;
                }
            }
            if (ack_seen) {
                /* ACK는 커널 완전 정지 직전 신호 — J2가 확실히 듣도록 2 s 정착 */
                for (uint16_t st = 0; st < ACK_SETTLE_MS; st += TOUCH_POLL_MS) {
                    _delay_ms(TOUCH_POLL_MS);
                    wdt_reset();
                }
            } else {
                bb_log(EV_ACK_MISSED, 0);
            }
        }
    }
}
