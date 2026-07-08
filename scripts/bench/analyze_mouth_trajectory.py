#!/usr/bin/env python3
"""Analyze captured online mouth-trajectory logs and emit an HTML report.

Reads pos4_audio CSVs captured by scripts/hardware/capture_mouth_logs.sh
(logs/mouth_inspection/<name>.csv) and evaluates, per utterance, how SMOOTH
and HUMAN-LIKE the mouth motion is — not just whether it tracks the audio.

Columns used (one row per 40 ms control tick):
    scheduled_ms, wall_elapsed_ms, target_pos4, actual_pos4, audio_raw_point_40ms

Focus metrics:
  - envelope↔target correlation + lag      (does the mouth follow speech)
  - target↔actual tracking error/lag/overshoot (PID/servo quality)
  - velocity / acceleration / JERK of target & actual (smoothness)
  - chatter: velocity sign-reversals per second (robotic jitter)
  - saturation: % ticks pinned closed / fully open
  - open-amount distribution (is motion monotone/binary = robotic)
  - real-time jitter: wall_elapsed_ms − scheduled_ms

Usage:
    uv run python scripts/bench/analyze_mouth_trajectory.py
    uv run python scripts/bench/analyze_mouth_trajectory.py --indir logs/mouth_inspection
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

DT_MS = 40.0  # control tick period
TICK_OPEN = 400.0  # min_mouth: full open travel (ticks)


def _f(x: str) -> float:
    try:
        return float(x)
    except (ValueError, TypeError):
        return math.nan


def read_csv(path: Path) -> dict[str, list[float]]:
    cols = {"scheduled_ms": [], "wall_ms": [], "target": [], "actual": [], "audio": []}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cols["scheduled_ms"].append(_f(row.get("scheduled_ms", "nan")))
            cols["wall_ms"].append(_f(row.get("wall_elapsed_ms", "nan")))
            cols["target"].append(_f(row.get("target_pos4", "nan")))
            cols["actual"].append(_f(row.get("actual_pos4", "nan")))
            cols["audio"].append(_f(row.get("audio_raw_point_40ms", "nan")))
    return cols


def diff(series: list[float]) -> list[float]:
    return [series[i + 1] - series[i] for i in range(len(series) - 1)]


def rms(xs: list[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return math.nan
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def corr(a: list[float], b: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    if len(pairs) < 3:
        return math.nan
    xs, ys = zip(*pairs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return math.nan
    return num / (dx * dy)


def best_lag_corr(a: list[float], b: list[float], max_lag: int = 8) -> tuple[float, int]:
    """Best correlation of b shifted relative to a, lag in ticks (b lags a if >0)."""
    best_r, best_l = -2.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            aa, bb = a[: len(a) - lag], b[lag:]
        else:
            aa, bb = a[-lag:], b[: len(b) + lag]
        r = corr(list(aa), list(bb))
        if not math.isnan(r) and r > best_r:
            best_r, best_l = r, lag
    return best_r, best_l


def sign_reversals(vel: list[float], eps: float = 2.0) -> int:
    """Count velocity sign reversals (ignoring near-zero moves) = chatter proxy."""
    prev = 0
    rev = 0
    for v in vel:
        s = 0 if abs(v) < eps else (1 if v > 0 else -1)
        if s != 0 and prev != 0 and s != prev:
            rev += 1
        if s != 0:
            prev = s
    return rev


def analyze(name: str, cols: dict[str, list[float]]) -> dict:
    target = cols["target"]
    actual = cols["actual"]
    audio = cols["audio"]
    n = len(target)
    dur = n * DT_MS / 1000.0

    # velocity (tick/tick = tick/40ms), accel, jerk
    vt, va = diff(target), diff(actual)
    at, aa = diff(vt), diff(va)
    jt, ja = diff(at), diff(aa)

    # envelope ↔ target
    env_r, env_lag = best_lag_corr(audio, target)
    # target ↔ actual tracking
    trk_r, trk_lag = best_lag_corr(target, actual)
    track_err = [t - a for t, a in zip(target, actual) if not (math.isnan(t) or math.isnan(a))]

    # overshoot: how often actual exceeds target's local range (past full open)
    valid_t = [t for t in target if not math.isnan(t)]
    valid_a = [a for a in actual if not math.isnan(a)]
    over = sum(1 for a in valid_a if a > max(valid_t) + 5) if valid_t else 0

    # saturation
    closed = sum(1 for t in valid_t if t <= 5)
    full = sum(1 for t in valid_t if t >= TICK_OPEN - 5)

    # real-time jitter
    jit = [w - s for w, s in zip(cols["wall_ms"], cols["scheduled_ms"]) if not (math.isnan(w) or math.isnan(s))]
    jit_sorted = sorted(jit)
    p95 = jit_sorted[int(0.95 * (len(jit_sorted) - 1))] if jit_sorted else math.nan

    dur_s = max(dur, 1e-9)
    return {
        "name": name,
        "n": n,
        "duration_s": round(dur, 1),
        "env_corr": round(env_r, 3),
        "env_lag_ms": int(env_lag * DT_MS),
        "track_corr": round(trk_r, 3),
        "track_lag_ms": int(trk_lag * DT_MS),
        "track_rms_tick": round(rms(track_err), 1),
        "track_max_tick": round(max((abs(x) for x in track_err), default=0.0), 1),
        "overshoot_ticks": over,
        "open_mean_pct": round(100 * statistics.fmean(valid_t) / TICK_OPEN, 1) if valid_t else 0,
        "open_max_pct": round(100 * max(valid_t, default=0) / TICK_OPEN, 1),
        "closed_pct": round(100 * closed / max(len(valid_t), 1), 1),
        "full_open_pct": round(100 * full / max(len(valid_t), 1), 1),
        "vel_max_target": round(max((abs(v) for v in vt), default=0.0), 1),
        "vel_max_actual": round(max((abs(v) for v in va), default=0.0), 1),
        "jerk_rms_target": round(rms(jt), 1),
        "jerk_rms_actual": round(rms(ja), 1),
        "chatter_target_hz": round(sign_reversals(vt) / dur_s, 2),
        "chatter_actual_hz": round(sign_reversals(va) / dur_s, 2),
        "jitter_p95_ms": round(p95, 1),
        "jitter_max_ms": round(max(jit, default=0.0), 1),
        # series for plotting (downsample audio normalized to 0..400)
        "_series": {
            "t": [round(i * DT_MS / 1000.0, 3) for i in range(n)],
            "audio": _norm(audio),
            "target": [None if math.isnan(x) else round(x, 1) for x in target],
            "actual": [None if math.isnan(x) else round(x, 1) for x in actual],
        },
    }


def _norm(audio: list[float]) -> list[float | None]:
    vals = [a for a in audio if not math.isnan(a)]
    hi = max(vals) if vals else 1.0
    hi = hi or 1.0
    return [None if math.isnan(a) else round(TICK_OPEN * a / hi, 1) for a in audio]


def verdict(m: dict) -> list[str]:
    """Heuristic flags focused on smoothness / human-likeness."""
    flags = []
    if not math.isnan(m["env_corr"]) and m["env_corr"] < 0.5:
        flags.append(f"⚠ 음성-입 상관 낮음 ({m['env_corr']})")
    if not math.isnan(m["track_corr"]) and m["track_corr"] < 0.9:
        flags.append(f"⚠ 서보 추종 약함 (r={m['track_corr']})")
    if m["track_lag_ms"] >= 80:
        flags.append(f"⚠ 추종 지연 {m['track_lag_ms']}ms")
    if m["track_rms_tick"] >= 40:
        flags.append(f"⚠ 추종오차 큼 (RMS {m['track_rms_tick']} tick)")
    if m["chatter_actual_hz"] >= 6:
        flags.append(f"⚠ 실제 떨림 {m['chatter_actual_hz']}Hz (로봇틱)")
    if m["full_open_pct"] >= 15:
        flags.append(f"⚠ 만개 포화 {m['full_open_pct']}%")
    if m["overshoot_ticks"] > 0:
        flags.append(f"⚠ 오버슈트 {m['overshoot_ticks']} ticks")
    if not flags:
        flags.append("✅ 부드럽고 양호")
    return flags


def build_html(rows: list[dict]) -> str:
    data = json.dumps([{**{k: v for k, v in r.items() if k != "_series"}, "series": r["_series"]} for r in rows])
    th = (
        "name|예문", "duration_s|길이(s)", "env_corr|음성↔입 r", "env_lag_ms|입지연(ms)",
        "track_corr|추종 r", "track_lag_ms|추종지연", "track_rms_tick|추종RMS",
        "open_mean_pct|평균개구%", "full_open_pct|만개%", "closed_pct|닫힘%",
        "vel_max_actual|최대속도", "jerk_rms_actual|저크RMS", "chatter_actual_hz|떨림Hz",
        "jitter_p95_ms|지터p95",
    )
    head = "".join(f"<th title='{k}'>{lbl}</th>" for k, lbl in (x.split("|") for x in th))
    body = ""
    for r in rows:
        cells = "".join(f"<td>{r[k.split('|')[0]]}</td>" for k in th)
        body += f"<tr data-name='{r['name']}'>{cells}</tr>"
        body += (
            f"<tr class='vrow'><td colspan='{len(th)}'>"
            + "  ".join(verdict(r))
            + "</td></tr>"
        )

    return f"""<!doctype html><html lang=ko><head><meta charset=utf-8>
<title>온라인 입 궤적 점검 — 부드러움/사람다움</title>
<style>
 body{{font-family:system-ui,'Apple SD Gothic Neo',sans-serif;margin:24px;background:#0f1117;color:#e6e6e6}}
 h1{{font-size:20px}} .sub{{color:#9aa;font-size:13px;margin-bottom:16px}}
 table{{border-collapse:collapse;font-size:12px;width:100%}}
 th,td{{border:1px solid #2a2f3a;padding:4px 7px;text-align:right}}
 th:first-child,td:first-child{{text-align:left}}
 th{{background:#1a1f2b;position:sticky;top:0;cursor:help}}
 tr[data-name]{{cursor:pointer}} tr[data-name]:hover{{background:#1c2330}}
 .vrow td{{text-align:left;color:#cdb;background:#141821;font-size:11px}}
 .charts{{margin-top:24px}} canvas{{background:#11151d;border:1px solid #2a2f3a;border-radius:6px;width:100%;height:200px}}
 . chartwrap{{margin-bottom:18px}} .clab{{font-size:13px;margin:14px 0 4px}}
 .legend{{font-size:11px;color:#9aa}} .legend b{{padding:0 6px}}
 .a{{color:#5ab0ff}} .t{{color:#ffd24a}} .ac{{color:#7CFC9A}}
</style></head><body>
<h1>🤖 온라인 입 모터 궤적 점검 — 부드러움 · 사람다움</h1>
<div class=sub>예문별 40ms 틱 로그(pos4_audio). 열 머리글에 마우스를 올리면 원본 지표명. 행 클릭 시 아래 그래프로 스크롤.
 <span class=legend><b class=a>■ 오디오 엔벨로프</b><b class=t>■ 목표(target)</b><b class=ac>■ 실제(actual)</b> (모두 0–400 tick 스케일)</span></div>
<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>
<div class=charts id=charts></div>
<script>
const DATA = {data};
const charts = document.getElementById('charts');
function draw(rec){{
  const w=charts.clientWidth, h=200, pad=24;
  const wrap=document.createElement('div'); wrap.className='chartwrap';
  const lab=document.createElement('div'); lab.className='clab';
  lab.textContent='▶ '+rec.name+'  ('+rec.duration_s+'s)'; wrap.appendChild(lab);
  const cv=document.createElement('canvas'); cv.id='cv_'+rec.name; wrap.appendChild(cv); charts.appendChild(wrap);
  const dpr=window.devicePixelRatio||1; cv.width=w*dpr; cv.height=h*dpr;
  const x=cv.getContext('2d'); x.scale(dpr,dpr);
  const s=rec.series, n=s.t.length, T=s.t[n-1]||1;
  const X=t=>pad+(w-2*pad)*t/T, Y=v=>h-pad-(h-2*pad)*(v/400);
  x.strokeStyle='#2a2f3a'; x.lineWidth=1;
  for(let g=0;g<=4;g++){{const yy=pad+(h-2*pad)*g/4; x.beginPath();x.moveTo(pad,yy);x.lineTo(w-pad,yy);x.stroke();}}
  function line(arr,col,wd){{x.strokeStyle=col;x.lineWidth=wd;x.beginPath();let st=false;
    for(let i=0;i<n;i++){{const v=arr[i]; if(v==null){{st=false;continue;}} const px=X(s.t[i]),py=Y(v);
      if(!st){{x.moveTo(px,py);st=true;}} else x.lineTo(px,py);}} x.stroke();}}
  line(s.audio,'#5ab0ff',1); line(s.target,'#ffd24a',1.6); line(s.actual,'#7CFC9A',1.4);
}}
DATA.forEach(draw);
document.querySelectorAll('tr[data-name]').forEach(tr=>tr.onclick=()=>{{
  const el=document.getElementById('cv_'+tr.dataset.name); if(el) el.scrollIntoView({{behavior:'smooth',block:'center'}});}});
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="logs/mouth_inspection")
    ap.add_argument("--out", default="logs/mouth_inspection/report.html")
    args = ap.parse_args()

    indir = Path(args.indir)
    csvs = sorted(p for p in indir.glob("*.csv") if not p.name.endswith("_play.log"))
    if not csvs:
        print(f"No CSVs in {indir}")
        return

    rows = []
    for p in csvs:
        cols = read_csv(p)
        if len(cols["target"]) < 5:
            print(f"skip (too few rows): {p.name}")
            continue
        rows.append(analyze(p.stem, cols))

    # console summary
    print(f"\n{'example':28} {'env_r':>6} {'trk_r':>6} {'trkLag':>7} {'trkRMS':>7} "
          f"{'chat_a':>7} {'jerk_a':>7} {'full%':>6} {'jitP95':>7}")
    for m in rows:
        print(f"{m['name']:28} {m['env_corr']:>6} {m['track_corr']:>6} {m['track_lag_ms']:>6}m "
              f"{m['track_rms_tick']:>7} {m['chatter_actual_hz']:>7} {m['jerk_rms_actual']:>7} "
              f"{m['full_open_pct']:>6} {m['jitter_p95_ms']:>7}")

    Path(args.out).write_text(build_html(rows), encoding="utf-8")
    print(f"\nHTML report -> {args.out}")


if __name__ == "__main__":
    main()
