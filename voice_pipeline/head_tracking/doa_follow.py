"""실시간 DOA → yaw 연속 추종 + 라이브 값 출력 (캘리브레이션용).

./build/Ray 를 먼저 띄워두고 실행. 락 없이 매 프레임 DOA를 바로 yaw로 매핑해
연속 추종(look_at duration=0 = pursue)하면서, DOA·yaw를 실시간으로 한 줄씩 출력한다.
어느 방향에서 말하든 머리가 화자 쪽으로 오는지 + 좌/우 클램프 지점을 눈+값으로 동시 확인.

yaw = clamp(sign * (DOA - front), ±max)   # 안전범위 ±max 클램프 고정
DOA==0(방향추정 실패)·speech=0 프레임은 무시(직전 방향 유지).

실행 (./build/Ray 띄워두고, Ctrl+C로 종료):
    uv run python voice_pipeline/head_tracking/doa_follow.py                 # 기본 front=115 sign=-1 max=35
    uv run python voice_pipeline/head_tracking/doa_follow.py 115 -1 35       # front sign max 지정
    uv run python voice_pipeline/head_tracking/doa_follow.py 115 1 35        # sign 뒤집어 비교
"""

from __future__ import annotations

import subprocess
import sys
import time

from voice_pipeline.bridge.cpp_bridge import CppBridge
from voice_pipeline.head_tracking.xvf3800_doa import XVF3800DOA


def wrap180(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


def _start_capture() -> subprocess.Popen | None:
    """ReSpeaker 캡처 스트림을 열어둔다(DOA 갱신에 필요). 실패해도 추종은 진행."""
    try:
        return subprocess.Popen(
            ["arecord", "-D", "plughw:CARD=L16K6Ch,DEV=0", "-c", "6", "-r", "16000",
             "-f", "S16_LE", "/dev/null"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    front = float(sys.argv[1]) if len(sys.argv) > 1 else 115.0
    sign = float(sys.argv[2]) if len(sys.argv) > 2 else -1.0
    ymax = float(sys.argv[3]) if len(sys.argv) > 3 else 35.0

    capture = _start_capture()  # 마이크 캡처 활성(DOA 보호)
    time.sleep(1.0)
    bridge = CppBridge()
    bridge.connect()
    doa_reader = XVF3800DOA()
    print(f"[follow] front={front} sign={sign:+g} max=±{ymax}°  (Ctrl+C 종료)")
    print("  어느 방향에서든 말하면 머리가 그쪽으로 따라옵니다. 실시간 값:")

    last_yaw = 0.0
    yaw_min, yaw_max = 0.0, 0.0
    try:
        while True:
            d, sp = doa_reader.read()
            valid = sp and d != 0
            if valid:
                rel = wrap180(d - front)
                yaw = max(-ymax, min(ymax, sign * rel))
                bridge.send_look_at(yaw, 0.0)  # pursue(연속 추종)
                last_yaw = yaw
                yaw_min, yaw_max = min(yaw_min, yaw), max(yaw_max, yaw)
            side = "<<왼" if last_yaw > 1 else "오>>" if last_yaw < -1 else "정면"
            clamp = " [클램프]" if valid and abs(last_yaw) >= ymax - 0.01 else ""
            mark = "" if valid else "  (무효: 무시)"
            print(f"  DOA={d:3d} spch={int(sp)} → yaw={last_yaw:+6.1f}° {side}{clamp}{mark}")
            time.sleep(0.1)  # 10Hz
    except KeyboardInterrupt:
        pass
    finally:
        bridge.send_look_at(0.0, 0.6)  # 정면 복귀
        time.sleep(0.8)
        doa_reader.close()
        bridge.disconnect()
        if capture is not None:
            capture.terminate()
        print(f"\n[follow] yaw 도달: 최좌={yaw_max:+.1f}°  최우={yaw_min:+.1f}°  (±{ymax} 클램프)")
        print("[follow] 종료")


if __name__ == "__main__":
    main()
