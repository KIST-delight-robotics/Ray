"""네트워크 정지 구간 판별 — 세션 동안 Wi-Fi 게이트웨이와 외부 호스트에 200 ms 간격 ping 을 나란히 보내 RTT·손실을
시각과 함께 남기고, 파이프라인 로그의 "Audio chunk gap" 과 대조해 정지가 어느 구간이었는지 가른다.

    uv run python scripts/hardware/net_probe.py                       # 기록 시작 (Ctrl+C 로 종료, 요약 출력)
    uv run python scripts/hardware/net_probe.py --analyze var/log/net/<ts> [--pipeline var/log/pipeline/<ts>.log]

판별 논리 (조각 정지 시각 ±윈도 안의 ping 상태):
  게이트웨이 튐 (손실 또는 RTT ≥ 문턱)            → Wi-Fi 한 홉 문제
  게이트웨이 정상, 외부만 튐                       → 연구소 망 이상 상위 경로
  둘 다 정상                                       → 서버 쪽 전송 문제 (경로는 살아 있었음)

기록 형식 (var/log/net/<ts>/): <label>.csv = epoch,seq,rtt_ms(빈칸=손실). events.txt = 연속 이상 구간 요약.
200 ms 는 일반 권한에서 허용되는 최소 간격이고, 1 초 정지가 5 개 연속 이상으로 찍혀 단발 손실과 갈린다.
Wi-Fi 정지는 손실보다 "큐에 갇혔다가 늦게 도착" 이 많아 RTT 도 같이 본다.
"""

from __future__ import annotations

import argparse
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

LOG_ROOT = Path("var/log/net")
INTERVAL_SEC = 0.2
LATE_RTT_MS = 300.0  # 이 이상이면 "튐" 으로 본다 (정상 게이트웨이 RTT 2~10 ms, 외부 10~40 ms)
GAP_WINDOW_SEC = 0.5  # 조각 정지 구간 앞뒤로 함께 보는 폭
PIPELINE_GAP_RE = re.compile(r"^(\S+ \S+) .*Audio chunk gap (\d+)ms \((\w+)")
PING_REPLY_RE = re.compile(r"^\[(\d+\.\d+)\] .*icmp_seq=(\d+) .*time=([\d.]+) ms")
PING_LOST_RE = re.compile(r"^\[(\d+\.\d+)\] no answer yet for icmp_seq=(\d+)")


def default_gateway() -> str:
    out = subprocess.run(["ip", "-4", "route", "show", "default"], capture_output=True, text=True, check=True).stdout
    m = re.search(r"default via (\S+)", out)
    if not m:
        raise SystemExit("기본 게이트웨이를 찾지 못했다")
    return m.group(1)


class Pinger(threading.Thread):
    """ping -D -O -i 0.2 를 돌리며 응답·손실을 CSV 로 쓴다."""

    def __init__(self, label: str, host: str, out_dir: Path) -> None:
        super().__init__(name=f"ping-{label}", daemon=True)
        self.label, self.host = label, host
        self._path = out_dir / f"{label}.csv"
        self._proc: subprocess.Popen[str] | None = None
        self.sent = self.lost = self.late = 0

    def run(self) -> None:
        self._proc = subprocess.Popen(
            ["ping", "-D", "-O", "-n", "-i", str(INTERVAL_SEC), self.host],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert self._proc.stdout is not None
        with self._path.open("w") as f:
            f.write("epoch,seq,rtt_ms\n")
            for line in self._proc.stdout:
                if m := PING_REPLY_RE.match(line):
                    self.sent += 1
                    rtt = float(m.group(3))
                    if rtt >= LATE_RTT_MS:
                        self.late += 1
                    f.write(f"{m.group(1)},{m.group(2)},{rtt}\n")
                elif m := PING_LOST_RE.match(line):
                    self.sent += 1
                    self.lost += 1
                    f.write(f"{m.group(1)},{m.group(2)},\n")
                f.flush()

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.send_signal(signal.SIGINT)
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()


def record(args: argparse.Namespace) -> Path:
    out_dir = LOG_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True)
    gw = args.gateway or default_gateway()
    pingers = [Pinger("gateway", gw, out_dir), Pinger("external", args.external, out_dir)]
    for p in pingers:
        p.start()
    print(f"기록 중: gateway={gw}, external={args.external}, {INTERVAL_SEC * 1000:.0f} ms 간격 → {out_dir}")
    print("(Ctrl+C 로 종료)")
    try:
        while all(p.is_alive() for p in pingers):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in pingers:
            p.stop()
        for p in pingers:
            p.join(timeout=3)
    for p in pingers:
        print(f"  {p.label:8s} {p.host:15s} sent {p.sent}, lost {p.lost}, late(≥{LATE_RTT_MS:.0f}ms) {p.late}")
    return out_dir


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def load_csv(path: Path) -> list[tuple[float, float | None]]:
    rows: list[tuple[float, float | None]] = []
    for ln in path.read_text().splitlines()[1:]:
        t, _, rtt = ln.split(",")
        rows.append((float(t), float(rtt) if rtt else None))
    return rows


def bad_events(rows: list[tuple[float, float | None]]) -> list[tuple[float, float, int, int, float]]:
    """연속 이상(손실 또는 RTT ≥ 문턱) 구간 → (시작, 끝, 개수, 손실 수, 최대 RTT). 정상 ping 하나로 구간이 끊긴다."""
    events = []
    cur: list[tuple[float, float | None]] = []
    for t, rtt in rows:
        if rtt is None or rtt >= LATE_RTT_MS:
            cur.append((t, rtt))
        elif cur:
            events.append(_summ(cur))
            cur = []
    if cur:
        events.append(_summ(cur))
    return events


def _summ(cur: list[tuple[float, float | None]]) -> tuple[float, float, int, int, float]:
    rtts = [r for _, r in cur if r is not None]
    # 손실은 응답이 없어 도착 시각이 없다 — 응답이 늦게 온 것은 도착 시각에서 RTT 를 빼면 발신 시각
    start = min(t - (r or 0) / 1000 for t, r in cur)
    return start, cur[-1][0], len(cur), sum(1 for _, r in cur if r is None), max(rtts) if rtts else float("nan")


def status_in(rows: list[tuple[float, float | None]], t0: float, t1: float) -> str:
    sel = [(t, r) for t, r in rows if t0 <= t - (r or 0) / 1000 <= t1 or t0 <= t <= t1]
    if not sel:
        return "기록 없음"
    lost = sum(1 for _, r in sel if r is None)
    late = [r for _, r in sel if r is not None and r >= LATE_RTT_MS]
    if not lost and not late:
        return f"정상 ({len(sel)}개, 최대 {max(r for _, r in sel if r is not None):.0f} ms)"
    parts = []
    if lost:
        parts.append(f"손실 {lost}")
    if late:
        parts.append(f"지연 {len(late)}개 최대 {max(late):.0f} ms")
    return "튐: " + ", ".join(parts)


def fmt(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%H:%M:%S.%f")[:-3]


def analyze(run_dir: Path, pipeline: Path | None) -> str:
    gw = load_csv(run_dir / "gateway.csv")
    ext = load_csv(run_dir / "external.csv")
    lines = [f"# net probe — {run_dir}", ""]
    for label, rows in (("gateway", gw), ("external", ext)):
        lost = sum(1 for _, r in rows if r is None)
        rtts = sorted(r for _, r in rows if r is not None)
        if not rows:
            lines.append(f"{label}: 기록 없음")
            continue
        p50 = rtts[len(rtts) // 2] if rtts else float("nan")
        p99 = rtts[int(len(rtts) * 0.99)] if rtts else float("nan")
        lines.append(
            f"{label}: {len(rows)}개 ({(rows[-1][0] - rows[0][0]) / 60:.1f}분), 손실 {lost}, "
            f"RTT p50 {p50:.0f} / p99 {p99:.0f} ms, 이상 구간 {len(bad_events(rows))}개"
        )
        for s, e, n, nl, mx in bad_events(rows):
            if n >= 2:
                lines.append(f"    {fmt(s)} ~ {fmt(e)}  {n}개 (손실 {nl}, 최대 RTT {mx:.0f} ms)")
    if pipeline is None:
        (run_dir / "events.txt").write_text("\n".join(lines))
        return "\n".join(lines)

    lines += ["", f"## 파이프라인 조각 정지 대조 — {pipeline}", ""]
    lines.append(f"{'끝난 시각':>12}  {'정지':>7}  {'상태':>8}  {'게이트웨이':<34} {'외부':<34} 판정")
    n_gap = 0
    for ln in pipeline.read_text(errors="replace").splitlines():
        m = PIPELINE_GAP_RE.match(ln)
        if not m:
            continue
        t_end = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f").timestamp()
        gap = int(m.group(2)) / 1000
        t0, t1 = t_end - gap - GAP_WINDOW_SEC, t_end + GAP_WINDOW_SEC
        if not gw or not (gw[0][0] - 1 <= t_end <= gw[-1][0] + 1):
            continue
        n_gap += 1
        sg, se = status_in(gw, t0, t1), status_in(ext, t0, t1)
        if sg.startswith("튐"):
            verdict = "Wi-Fi 홉"
        elif se.startswith("튐"):
            verdict = "상위 경로"
        elif sg.startswith("기록") or se.startswith("기록"):
            verdict = "판정 불가"
        else:
            verdict = "경로 정상 → 서버 쪽"
        lines.append(f"{fmt(t_end):>12}  {gap * 1000:5.0f}ms  {m.group(3):>8}  {sg:<34} {se:<34} {verdict}")
    if n_gap == 0:
        lines.append("(ping 기록 시간대와 겹치는 조각 정지가 없다)")
    (run_dir / "events.txt").write_text("\n".join(lines))
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gateway", default=None, help="게이트웨이 주소 (기본: 기본 경로에서 자동)")
    p.add_argument("--external", default="1.1.1.1", help="외부 호스트")
    p.add_argument("--analyze", type=Path, default=None, help="기록 디렉터리만 분석")
    p.add_argument("--pipeline", type=Path, default=None, help="대조할 파이프라인 로그 (기본: var/log/pipeline 최신)")
    args = p.parse_args()

    run_dir = args.analyze or record(args)
    pipeline = args.pipeline
    if pipeline is None:
        logs = sorted(Path("var/log/pipeline").glob("*.log"))
        pipeline = logs[-1] if logs else None
    print()
    print(analyze(run_dir, pipeline))


if __name__ == "__main__":
    sys.exit(main())
