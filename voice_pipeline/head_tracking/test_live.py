"""head-tracking 직접 테스트 (웨이크워드 없이 바로 추적).

./build/Ray 를 먼저 실행해 두고 이 스크립트를 돌리면, DOA를 읽어 고개가 소리 쪽으로 움직인다.
(웨이크워드 경로를 우회 — 순수 head-tracking 동작 확인용. config [head_tracking].enabled=true 필요.)

실행:
    # 터미널1: ./build/Ray
    # 터미널2:
    HEAD_TRACK_DEBUG=1 uv run python voice_pipeline/head_tracking/test_live.py
    HEAD_TRACK_DEBUG=1 uv run python voice_pipeline/head_tracking/test_live.py 45   # 45초
"""

from __future__ import annotations

import logging
import sys
import time

from voice_pipeline.bridge.cpp_bridge import CppBridge
from voice_pipeline.head_tracking import HeadTrackingController, load_head_tracking_config


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-28s %(levelname)-5s %(message)s")
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0

    bridge = CppBridge()
    bridge.connect()
    cfg = load_head_tracking_config()
    print(f"[test] front={cfg.doa_front_deg} sign={cfg.doa_sign} max=±{cfg.max_yaw_deg}° "
          f"deadzone={cfg.deadzone_deg}° return: {1/cfg.return_t_per_deg:.0f}°/s")

    ctrl = HeadTrackingController(bridge, cfg)
    if not ctrl.start():
        print("[test] DOA 리더 시작 실패 (ReSpeaker 연결/권한 확인)")
        bridge.disconnect()
        return
    ctrl.resume()  # 웨이크워드 없이 바로 추적 시작

    print(f">>> {dur:.0f}초 동안 로봇의 좌/우/정면에서 말해보세요. 고개가 따라갑니다. (Ctrl+C 종료)")
    try:
        time.sleep(dur)
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.pause()      # 정면 복귀
        time.sleep(0.8)
        ctrl.stop()
        bridge.disconnect()
        print("[test] 종료")


if __name__ == "__main__":
    main()
