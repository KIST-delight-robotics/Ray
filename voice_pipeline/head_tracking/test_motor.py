"""yaw 값을 직접 입력해 모터 구동 (마이크/DOA 없이). ./build/Ray 먼저 실행 필요.

look_at(yaw_deg, duration)을 순서대로 보내 머리가 그 yaw로 움직이는지 확인한다.
방향(+왼/−오)·클램프(±35)·회전속도·복귀를 결정적으로 검증할 때 사용.

실행 (./build/Ray 띄워두고):
    uv run python voice_pipeline/head_tracking/test_motor.py            # 기본 데모 시퀀스
    uv run python voice_pipeline/head_tracking/test_motor.py 35 -35 0 15  # 지정 yaw들 순서대로
    uv run python voice_pipeline/head_tracking/test_motor.py --hold 3 20 -20
"""

from __future__ import annotations

import argparse
import time

from voice_pipeline.bridge.cpp_bridge import CppBridge
from voice_pipeline.head_tracking import load_head_tracking_config

# 기본 데모: 정면→좌max→정면→우max→정면→중간들→정면
DEMO = [35, 0, -35, 0, 15, -15, 25, -25, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("yaws", nargs="*", type=float, help="순서대로 보낼 yaw(°) 목록 (+왼 −오, 비우면 데모)")
    ap.add_argument("--hold", type=float, default=2.5, help="각 목표 유지 시간(초)")
    a = ap.parse_args()

    cfg = load_head_tracking_config()
    seq = a.yaws if a.yaws else DEMO
    lim = cfg.max_yaw_deg

    bridge = CppBridge()
    bridge.connect()
    print(f"[motor] 한계 ±{lim}° · 회전 {1/cfg.t_per_deg:.0f}°/s · 시퀀스={seq}")

    prev = 0.0
    try:
        for raw in seq:
            yaw = max(-lim, min(lim, raw))  # 안전 클램프(C++도 tick 클램프)
            delta = abs(yaw - prev)
            dur = max(cfg.t_min_s, min(cfg.t_max_s, delta * cfg.t_per_deg))
            bridge.send_look_at(yaw, dur)
            tick = round(-yaw * 4096 / 360 + 3600)  # 참고용 예상 tick
            side = "왼" if yaw > 0 else "오" if yaw < 0 else "정면"
            print(f"  → yaw={yaw:+6.1f}°  dur={dur:.2f}s  (예상 tick≈{tick}, {side})")
            time.sleep(a.hold)
            prev = yaw
        # 정면 복귀(느리게)
        dur = max(cfg.t_min_s, min(cfg.return_duration_s, abs(prev) * cfg.return_t_per_deg))
        bridge.send_look_at(0.0, dur)
        print(f"  → 정면 복귀 dur={dur:.2f}s")
        time.sleep(dur + 0.5)
    finally:
        bridge.disconnect()
        print("[motor] 종료")


if __name__ == "__main__":
    main()
