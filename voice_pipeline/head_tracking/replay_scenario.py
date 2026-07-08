"""DOA 시나리오를 실제 모터에 재생 (마이크 없이 가상 DOA 주입). ./build/Ray 먼저 실행.

DOA_SCENARIOS.md의 가상 DOA 입력열을 controller의 '실제' 로직(_run)에 주입해,
계산된 yaw가 look_at으로 C++에 가서 머리가 그 시나리오대로 움직이는지 본다.
(controller 코드를 그대로 사용 — 시뮬과 동일 결과가 실제 모터로 재생됨.)

실행 (./build/Ray 띄워두고):
    HEAD_TRACK_DEBUG=1 uv run python voice_pipeline/head_tracking/replay_scenario.py          # 8종 전부 순서대로
    HEAD_TRACK_DEBUG=1 uv run python voice_pipeline/head_tracking/replay_scenario.py single    # 하나만
    ... 시나리오: conversation single left_quiet_right_loud blip_only moving two_alternating behind whisper
"""

from __future__ import annotations

import sys
import threading
import time

from voice_pipeline.bridge.cpp_bridge import CppBridge
from voice_pipeline.head_tracking import HeadTrackingController, load_head_tracking_config
from voice_pipeline.head_tracking.sim_scenarios import SCENARIOS, gen_frame


class _FakeDOA:
    """controller가 기대하는 .read()/.close() 인터페이스로 시나리오 프레임을 실시간 제공."""

    def __init__(self, segs: list, front_deg: float) -> None:
        self._segs = segs
        self._front = front_deg
        self._t0 = time.monotonic()
        self.total = sum(d for _, d, _ in segs)

    def read(self) -> tuple[int, bool]:
        el = time.monotonic() - self._t0
        acc = 0.0
        for kind, dur, p in self._segs:
            if el < acc + dur:
                ang0, speech = gen_frame(kind, p, (el - acc) / dur)
                return int((ang0 + self._front) % 360), bool(speech)
            acc += dur
        return int(self._front), False  # 끝난 뒤엔 정면·무음

    def close(self) -> None:
        pass


def main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-26s %(levelname)-5s %(message)s")

    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg in ("all", ""):
        names = list(SCENARIOS)
    elif arg in SCENARIOS:
        names = [arg]
    else:
        print(f"알 수 없는 시나리오: {arg}\n가능: all, {', '.join(SCENARIOS)}")
        return

    bridge = CppBridge()
    bridge.connect()
    cfg = load_head_tracking_config()

    ctrl = HeadTrackingController(bridge, cfg)
    ctrl._doa = _FakeDOA(SCENARIOS[names[0]], cfg.doa_front_deg)  # 초기(thread 시작용, paused)
    ctrl._thread = threading.Thread(target=ctrl._run, name="head-tracking-sim", daemon=True)
    ctrl._thread.start()

    try:
        for i, name in enumerate(names, 1):
            fake = _FakeDOA(SCENARIOS[name], cfg.doa_front_deg)
            ctrl._doa = fake          # 시나리오 주입(클록 리셋)
            ctrl.resume()             # 웨이크워드 우회, 추적 시작
            print(f"\n>>> [{i}/{len(names)}] '{name}' 재생 ({fake.total:.1f}s) — 머리가 시뮬대로 움직임")
            time.sleep(fake.total + 1.5)
            ctrl.pause()              # 시나리오 사이 정면 복귀
            time.sleep(2.5)           # 복귀 + 간격
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.stop()
        bridge.disconnect()
        print("\n[replay] 종료")


if __name__ == "__main__":
    main()
