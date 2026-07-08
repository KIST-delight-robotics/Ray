"""순수 raw DOA/VAD 실시간 모니터 (모터·브릿지·매핑 전부 없음).

XVF3800에서 (DOA, speech)만 읽어 실시간 출력. 칩의 DOA가 방향에 따라 실제로 변하는지,
아니면 특정 값(예: 90)에 고정되는지 확인용. Ray 안 띄워도 됨.

실행 (Ctrl+C 종료):
    uv run python voice_pipeline/head_tracking/doa_raw.py
"""

from __future__ import annotations

import time

from voice_pipeline.head_tracking.xvf3800_doa import XVF3800DOA


def main() -> None:
    doa = XVF3800DOA()
    print("[raw] DOA/VAD 실시간 (정면/왼쪽/오른쪽에서 번갈아 크게 말해보세요). Ctrl+C 종료")
    seen: dict[int, int] = {}
    try:
        while True:
            d, sp = doa.read()
            if sp:
                seen[d] = seen.get(d, 0) + 1
            bar = "#" * (d // 10)  # DOA 크기 막대(0~359)
            print(f"  DOA={d:3d}  VAD={int(sp)}  {bar}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        doa.close()
        if seen:
            top = sorted(seen.items(), key=lambda kv: -kv[1])[:8]
            print("\n[raw] speech=1일 때 자주 나온 DOA(횟수): " + ", ".join(f"{k}°×{v}" for k, v in top))


if __name__ == "__main__":
    main()
