"""C++ audio_sync 로그로 소리-모션 어긋남을 요약한다.

    uv run python scripts/hardware/audio_sync_report.py [var/log/motion/<dir>/audio_sync_RESPONSES.csv]

인자가 없으면 가장 최근 motion 로그 디렉터리의 audio_sync_*.csv 를 쓴다.
lag_ms = expected_ms − (playing_ms − silence_ms): 양수면 소리가 입보다 늦다.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        dirs = sorted(Path("var/log/motion").iterdir(), key=lambda p: p.name, reverse=True)
        files = [f for d in dirs for f in sorted(d.glob("audio_sync_*.csv"))]
        if not files:
            raise SystemExit("audio_sync_*.csv 가 없습니다 (C++ 가 MOTOR_ENABLED 로 빌드되어야 기록)")
        path = files[0]
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"{path}: 비어 있음")
    print(f"{path}: {len(rows)} 틱, {float(rows[-1]['elapsed_ms']) / 1000:.0f}초")

    def lag(r: dict[str, str]) -> float:
        return float(r["expected_ms"]) - (float(r["playing_ms"]) - float(r["silence_ms"]))

    header = ("구간(s)", "lag 중앙(ms)", "lag 최대(ms)", "무음 삽입 누적(ms)", "틱 지연 최대(ms)")
    print(f"{header[0]:>10} {header[1]:>12} {header[2]:>12} {header[3]:>18} {header[4]:>16}")
    win = 10_000
    b = 0.0
    while b < float(rows[-1]["elapsed_ms"]):
        seg = [r for r in rows if b <= float(r["elapsed_ms"]) < b + win]
        if seg:
            lags = sorted(lag(r) for r in seg)
            sched = [float(r["elapsed_ms"]) - float(r["expected_ms"]) for r in seg]  # 틱이 자기 예정 시각보다 늦은 양
            print(
                f"{b / 1000:4.0f}~{(b + win) / 1000:3.0f} {lags[len(lags) // 2]:12.0f} {lags[-1]:12.0f} "
                f"{float(seg[-1]['silence_ms']):18.0f} {max(sched):16.0f}"
            )
        b += win
    last = rows[-1]
    print(f"\n최종: 무음 삽입 {float(last['silence_ms']):.0f} ms, lag {lag(last):.0f} ms (양수 = 소리가 입보다 늦음)")


if __name__ == "__main__":
    main()
