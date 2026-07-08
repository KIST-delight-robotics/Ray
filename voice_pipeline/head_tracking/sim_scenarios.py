"""DOA_SCENARIOS.md 시나리오를 controller.py 로직으로 시뮬레이션(하드웨어 없음).

controller._run 의 결정 로직(연속추적 + deadzone + 거리비례 복귀)을 그대로 반영해,
가상 DOA 입력열에 대해 어떤 look_at(yaw,dur)이 나오는지 출력한다.
입력 각도는 '정면=0°' 규약 → XVF3800 DOA(정면=90°)로 변환해서 넣는다.

실행: uv run python voice_pipeline/head_tracking/sim_scenarios.py
"""

from __future__ import annotations

import random

from voice_pipeline.head_tracking.controller import HeadTrackingConfig, _wrap180, load_head_tracking_config

CFG = load_head_tracking_config()
DT = 1.0 / CFG.poll_hz
random.seed(42)

# 시나리오: (종류, 지속초, 파라미터). 각도는 정면=0° 규약.
SCENARIOS = {
    "conversation": [
        ("sil", 1.5, {}), ("talk", 2.6, {"a": -55}), ("sil", 0.6, {}), ("talk", 2.2, {"a": 40}),
        ("sil", 0.5, {}), ("talk", 1.6, {"a": -55}), ("sil", 0.4, {}),
        ("multi", 1.5, {"src": [(-55, 0.5), (40, 0.5)]}), ("talk", 2.8, {"a": 40}),
        ("sil", 1.2, {}), ("talk", 2.4, {"a": 15}), ("sil", 2.0, {}),
    ],
    "single": [("sil", 1.0, {}), ("talk", 4.0, {"a": 70}), ("sil", 2.5, {})],
    "left_quiet_right_loud": [("sil", 1.0, {}), ("multi", 5.0, {"src": [(-55, 0.28), (45, 0.60)]}), ("sil", 2.5, {})],
    "blip_only": [("sil", 1.0, {}), ("blip", 0.4, {}), ("sil", 1.0, {}), ("blip", 0.3, {}), ("sil", 1.5, {})],
    "moving": [("sil", 0.8, {}), ("sweep", 6.0, {"f": -60, "t": 60}), ("sil", 2.0, {})],
    "two_alternating": [
        ("sil", 1.0, {}), ("talk", 2.0, {"a": -50}), ("sil", 0.5, {}), ("talk", 2.0, {"a": 45}),
        ("sil", 0.5, {}), ("talk", 2.0, {"a": -50}), ("sil", 1.5, {}),
    ],
    "behind": [("sil", 1.0, {}), ("talk", 4.0, {"a": 150}), ("sil", 2.0, {})],
    "whisper": [("sil", 1.0, {}), ("talk", 5.0, {"a": -30}), ("sil", 2.0, {})],
}


def gen_frame(kind: str, p: dict, frac: float) -> tuple[float, int]:
    """(정면=0° 각도, speech) 한 프레임 생성."""
    if kind == "sil":
        return 0.0, 0
    if kind == "talk":
        return p["a"] + random.uniform(-9, 9), 1
    if kind == "blip":
        return random.uniform(-180, 180), 1
    if kind == "sweep":
        return p["f"] + (p["t"] - p["f"]) * frac + random.uniform(-6, 6), 1
    if kind == "multi":
        src = p["src"]
        total = sum(lv for _, lv in src)
        r = random.uniform(0, total)
        acc = 0.0
        for ang, lv in src:
            acc += lv
            if r <= acc:
                return ang + random.uniform(-9, 9), 1
        return src[-1][0], 1
    return 0.0, 0


def run(name: str, segs: list, c: HeadTrackingConfig) -> None:
    """controller._run의 '락' 로직을 그대로 미러링(검증용 출력)."""
    last_target = 0.0
    last_speech_t = -1e9
    onset_t: float | None = None
    locked = False
    t = 0.0
    sends: list[tuple[float, str, float, float]] = []
    for kind, dur, p in segs:
        n = int(dur / DT)
        for i in range(n):
            ang0, speech = gen_frame(kind, p, i / max(n - 1, 1))
            doa = (ang0 + c.doa_front_deg) % 360  # 정면0 → DOA(정면90)
            if speech:
                last_speech_t = t
                if not locked:
                    if onset_t is None:
                        onset_t = t
                    elif t - onset_t >= c.lock_confirm_s:
                        rel = _wrap180(doa - c.doa_front_deg)
                        yaw = max(-c.max_yaw_deg, min(c.max_yaw_deg, c.doa_sign * rel))
                        dd = max(c.t_min_s, min(c.t_max_s, abs(yaw - last_target) * c.t_per_deg))
                        sends.append((t, "LOCK", round(yaw, 1), round(dd, 2)))
                        last_target, locked = yaw, True
            else:
                onset_t = None
                if locked and (t - last_speech_t) > c.return_delay_s:
                    dd = max(c.t_min_s, min(c.return_duration_s, abs(last_target) * c.return_t_per_deg))
                    sends.append((t, "RETURN", 0.0, round(dd, 2)))
                    last_target, locked = 0.0, False
            t += DT
    n_lock = sum(1 for s in sends if s[1] == "LOCK")
    n_ret = sum(1 for s in sends if s[1] == "RETURN")
    print(f"\n=== {name} ===  (락 {n_lock}회, 복귀 {n_ret}회)")
    for ts, kind, yaw, dd in sends:
        print(f"  t={ts:4.1f}s  {kind:6}  yaw={yaw:+6.1f}°  dur={dd:.2f}s")


def main() -> None:
    print(f"params: front={CFG.doa_front_deg} sign={CFG.doa_sign} max=±{CFG.max_yaw_deg}° "
          f"deadzone={CFG.deadzone_deg}° return_delay={CFG.return_delay_s}s t/deg={CFG.return_t_per_deg}")
    for name, segs in SCENARIOS.items():
        run(name, segs, CFG)


if __name__ == "__main__":
    main()
