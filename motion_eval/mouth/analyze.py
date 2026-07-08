#!/usr/bin/env python3
"""Objective evaluation of ONLINE mouth-trajectory logs (motion_eval/mouth/logs/*.csv).

Single-DOF jaw → judged against phonetic expectation, not raw envelope.
Metrics (all pure-stdlib):
  - syllable-event match: audio syllable nuclei ↔ mouth-open events (precision/recall/offset)
  - modulation spectrum: dominant rhythm of audio vs mouth (should sit in 3–8 Hz)
  - sync lag: audio→target, audio→actual (perceptual lip-sync)
  - tracking: target↔actual corr / lag / RMS / overshoot (servo)
  - smoothness: jerk RMS, chatter (velocity sign-reversals/s)
  - saturation: % pinned full-open / closed
  - silence closure: is the mouth closed where the audio is silent? (over-trigger check)

Usage:
    uv run python motion_eval/mouth/analyze.py
    uv run python motion_eval/mouth/analyze.py --indir motion_eval/mouth/logs --out motion_eval/mouth/reports/report.html
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

DT_MS = 40.0
FS = 1000.0 / DT_MS  # 25 Hz tick rate
TICK_OPEN = 400.0

# phonetic expectation per utterance (gen_wavs.py names)
EXPECT = {
    "bilabial_closure": "양순음 다수 → 입이 자주 닫혀야 (무음폐쇄율↑ 기대)",
    "vowel_alternation": "개↔폐모음 교대 → 규칙적 진동",
    "sustained_vowel": "지속모음 → 연 채 유지(떨림 낮아야)",
    "plosive_bursts": "파열음 → 또렷한 개폐, 빠른 추종",
    "silence_gaps": "단어간 묵음 → 묵음 닫힘(과검출 점검)",
    "soft_to_loud": "약→강 → 포화 없이 크기 반영",
    "fast_counting": "빠른 음절률 → 변조 고역, 병합 없이",
    "natural_sentence": "자연문 기준선",
}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def read_csv(path: Path):
    c = {"sched": [], "wall": [], "target": [], "actual": [], "audio": []}
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            c["sched"].append(_f(r.get("scheduled_ms")))
            c["wall"].append(_f(r.get("wall_elapsed_ms")))
            c["target"].append(_f(r.get("target_pos4")))
            c["actual"].append(_f(r.get("actual_pos4")))
            c["audio"].append(_f(r.get("audio_raw_point_40ms")))
    return c


def read_offline(path: Path):
    """offline_mouth_handoff CSV: time_s, mouth_tick (0-250), open_ratio (0-1)."""
    out = []
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            out.append(_f(r.get("open_ratio")))
    return out


def norm_max(s):
    v = nz(s)
    hi = max(v) if v else 0.0
    return [math.nan if math.isnan(x) else (x / hi if hi else 0.0) for x in s]


def nz(xs):
    return [x for x in xs if not math.isnan(x)]


def diff(s):
    return [s[i + 1] - s[i] for i in range(len(s) - 1)]


def rms(xs):
    xs = nz(xs)
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else math.nan


def smooth3(s):
    out = list(s)
    for i in range(1, len(s) - 1):
        out[i] = (s[i - 1] + s[i] + s[i + 1]) / 3.0
    return out


def corr(a, b):
    p = [(x, y) for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    if len(p) < 3:
        return math.nan
    xs, ys = zip(*p)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return math.nan
    return sum((x - mx) * (y - my) for x, y in p) / (dx * dy)


def best_lag(a, b, ml=8):
    br, bl = -2.0, 0
    for lag in range(-ml, ml + 1):
        aa, bb = (a[: len(a) - lag], b[lag:]) if lag >= 0 else (a[-lag:], b[: len(b) + lag])
        r = corr(list(aa), list(bb))
        if not math.isnan(r) and r > br:
            br, bl = r, lag
    return br, bl


def find_peaks(s, thr_frac=0.15, min_dist=3, min_prom=0.0):
    """Local maxima above thr_frac*max, separated by >= min_dist ticks."""
    v = nz(s)
    if not v:
        return []
    hi = max(v)
    thr = thr_frac * hi
    cand = []
    for i in range(1, len(s) - 1):
        if math.isnan(s[i]):
            continue
        if s[i] >= s[i - 1] and s[i] > s[i + 1] and s[i] >= thr:
            cand.append(i)
    # enforce min distance, keep higher peak
    peaks = []
    for i in cand:
        if peaks and i - peaks[-1] < min_dist:
            if s[i] > s[peaks[-1]]:
                peaks[-1] = i
        else:
            peaks.append(i)
    return peaks


def mod_spectrum_peak(s, fmin=1.5, fmax=9.0, step=0.1, detrend_win=25):
    """Dominant modulation freq (Hz) in the syllable band.

    Highpass via moving-average subtraction (window ~1s) first, so the slow
    overall envelope build/decay doesn't swamp the syllable-rate rhythm.
    """
    x = nz(s)
    n = len(x)
    if n < 16:
        return math.nan
    half = detrend_win // 2
    hp = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        hp.append(x[i] - sum(x[lo:hi]) / (hi - lo))
    best_f, best_p = math.nan, -1.0
    f = fmin
    while f <= fmax:
        w = 2 * math.pi * f / FS
        re = sum(hp[i] * math.cos(w * i) for i in range(n))
        im = sum(hp[i] * math.sin(w * i) for i in range(n))
        p = re * re + im * im
        if p > best_p:
            best_p, best_f = p, f
        f += step
    return round(best_f, 2)


def sign_reversals(vel, eps=2.0):
    prev, rev = 0, 0
    for v in vel:
        s = 0 if abs(v) < eps else (1 if v > 0 else -1)
        if s and prev and s != prev:
            rev += 1
        if s:
            prev = s
    return rev


def analyze(name, c, offline=None, words=None):
    target, actual, audio = c["target"], c["actual"], c["audio"]
    n = len(target)
    dur = max(n * DT_MS / 1000.0, 1e-9)

    # online↔offline comparison (handoff §4: normalize each to own max, then r/lag/RMSE)
    off_r = off_lag = off_rmse = None
    off_series = None
    if offline:
        m = min(n, len(offline))
        on_n = norm_max(target[:m])      # online target → 0..1 by own max
        of_n = norm_max(offline[:m])     # offline open_ratio (already ~0..1)
        off_r = corr(on_n, of_n)
        # cross-corr lag: positive = online lags offline (offline is zero-phase, leads)
        _, lag = best_lag(of_n, on_n, ml=10)
        off_lag = lag * DT_MS
        err = [a - b for a, b in zip(on_n, of_n) if not (math.isnan(a) or math.isnan(b))]
        off_rmse = rms(err)
        off_series = [None if math.isnan(x) else round(TICK_OPEN * x, 1) for x in of_n]

    aud_s = smooth3(audio)
    tgt_s = smooth3(target)
    nuclei = find_peaks(aud_s, thr_frac=0.18, min_dist=3)   # audio syllable nuclei
    events = find_peaks(tgt_s, thr_frac=0.15, min_dist=3)    # mouth-open events

    # greedy match within tolerance (±3 ticks = ±120 ms)
    tol = 3
    matched, offsets, used = 0, [], set()
    for ni in nuclei:
        best, bd = -1, tol + 1
        for j, ei in enumerate(events):
            if j in used:
                continue
            d = abs(ei - ni)
            if d < bd:
                bd, best = d, j
        if best >= 0:
            used.add(best)
            matched += 1
            offsets.append((events[best] - ni) * DT_MS)
    recall = matched / len(nuclei) if nuclei else math.nan
    precision = matched / len(events) if events else math.nan
    mean_off = statistics.fmean(offsets) if offsets else math.nan

    # sync lag
    _, env_lag = best_lag(audio, target)
    _, act_lag = best_lag(audio, actual)
    # tracking
    trk_r, trk_lag = best_lag(target, actual)
    terr = [t - a for t, a in zip(target, actual) if not (math.isnan(t) or math.isnan(a))]
    vt, va = diff(target), diff(actual)
    jt = diff(diff(vt))
    ja = diff(diff(va))

    # tendon drive: opening (string pulls) vs closing (passive return) asymmetry.
    # actual_pos4 = ID5 spool position, not true jaw aperture (see README caveat).
    def _oc(vel, eps=2.0):
        op = [v for v in vel if v > eps]
        cl = [-v for v in vel if v < -eps]
        vo = statistics.fmean(op) if op else 0.0
        vc = statistics.fmean(cl) if cl else 0.0
        return vo, vc
    vo_t, vc_t = _oc(vt)
    vo_a, vc_a = _oc(va)
    asym_a = (vc_a / vo_a) if vo_a > 1e-6 else math.nan   # <1 → 닫힘이 열림보다 느림
    asym_t = (vc_t / vo_t) if vo_t > 1e-6 else math.nan

    vt_t = nz(target)
    va_a = nz(actual)
    over = sum(1 for a in va_a if a > max(vt_t) + 5) if vt_t else 0

    # saturation + open stats
    closed = sum(1 for t in vt_t if t <= 5)
    full = sum(1 for t in vt_t if t >= TICK_OPEN - 5)

    # silence closure: where audio < 8% of max, is target near closed?
    av = nz(audio)
    sil_thr = 0.08 * max(av) if av else 0
    sil_idx = [i for i, a in enumerate(audio) if not math.isnan(a) and a < sil_thr]
    sil_closed = (
        statistics.fmean([100 * (1 - min(target[i], TICK_OPEN) / TICK_OPEN) for i in sil_idx if not math.isnan(target[i])])
        if sil_idx else math.nan
    )

    jit = [w - s for w, s in zip(c["wall"], c["sched"]) if not (math.isnan(w) or math.isnan(s))]

    # 자막: TTS 단어를 (단어 중앙시각 기준) 가장 가까운 음절핵에 스냅해 배치
    subs = None
    if words:
        nuc_t = [i * DT_MS / 1000.0 for i in nuclei]
        subs = []
        for wd in words:
            try:
                wt = (float(wd["start"]) + float(wd["end"])) / 2.0
            except (KeyError, TypeError, ValueError):
                continue
            t = wt
            if nuc_t:
                t = min(nuc_t, key=lambda x: abs(x - wt))
            subs.append({"t": round(t, 3), "w": str(wd.get("word", "")).strip()})

    syl_rate = round(len(nuclei) / dur, 2)
    return {
        "name": name,
        "n": n,
        "dur": round(dur, 1),
        "syl_count": len(nuclei),
        "syl_rate_hz": syl_rate,
        "evt_count": len(events),
        "recall": round(recall, 2) if not math.isnan(recall) else None,
        "precision": round(precision, 2) if not math.isnan(precision) else None,
        "evt_offset_ms": round(mean_off) if not math.isnan(mean_off) else None,
        "mod_audio_hz": mod_spectrum_peak(aud_s),
        "mod_mouth_hz": mod_spectrum_peak([a for a in actual]),
        "env_lag_ms": int(env_lag * DT_MS),
        "act_lag_ms": int(act_lag * DT_MS),
        "track_r": round(trk_r, 3),
        "track_lag_ms": int(trk_lag * DT_MS),
        "track_rms": round(rms(terr), 1),
        "overshoot": over,
        "open_mean_pct": round(100 * statistics.fmean(vt_t) / TICK_OPEN, 1) if vt_t else 0,
        "full_pct": round(100 * full / max(len(vt_t), 1), 1),
        "closed_pct": round(100 * closed / max(len(vt_t), 1), 1),
        "sil_closed_pct": round(sil_closed, 1) if not math.isnan(sil_closed) else None,
        "off_r": round(off_r, 3) if off_r is not None and not math.isnan(off_r) else None,
        "off_lag_ms": int(off_lag) if off_lag is not None else None,
        "off_rmse": round(off_rmse, 3) if off_rmse is not None and not math.isnan(off_rmse) else None,
        "jerk_a": round(rms(ja), 1),
        "vopen_a": round(vo_a, 1),
        "vclose_a": round(vc_a, 1),
        "asym_a": round(asym_a, 2) if not math.isnan(asym_a) else None,
        "asym_t": round(asym_t, 2) if not math.isnan(asym_t) else None,
        "chatter_a_hz": round(sign_reversals(va) / dur, 2),
        "jitter_p95_ms": round(sorted(jit)[int(0.95 * (len(jit) - 1))], 1) if jit else None,
        "_series": {
            "t": [round(i * DT_MS / 1000.0, 3) for i in range(n)],
            "audio": _norm(audio),
            "target": [None if math.isnan(x) else round(x, 1) for x in target],
            "actual": [None if math.isnan(x) else round(x, 1) for x in actual],
            "nuclei": [round(i * DT_MS / 1000.0, 3) for i in nuclei],
            "events": [round(i * DT_MS / 1000.0, 3) for i in events],
            "offline": off_series,
            "subs": subs,
        },
    }


def _norm(audio):
    v = nz(audio)
    hi = max(v) if v else 1.0
    hi = hi or 1.0
    return [None if math.isnan(a) else round(TICK_OPEN * a / hi, 1) for a in audio]


def verdict(m):
    f = []
    base = m["name"].replace("ref_tts_", "").replace("ref_", "")
    # 립싱크: 실제(ID5 모터) 기준. 영상이 소리보다 ~125ms 뒤까지는 지각상 자연스러움.
    al = m["act_lag_ms"]
    sign = "뒤" if al > 0 else ("앞" if al < 0 else "정렬")
    if abs(al) <= 125:
        f.append(f"✅싱크 {abs(al)}ms {sign}(지각 OK)")
    else:
        f.append(f"⚠싱크 {abs(al)}ms {sign}(지각 한계 초과)")
    if m["recall"] is not None and m["recall"] < 0.8:
        f.append(f"⚠ 음절 놓침 (recall {m['recall']})")
    if m["precision"] is not None and m["precision"] < 0.7:
        f.append(f"⚠ 가짜 개폐 (precision {m['precision']})")
    if not math.isnan(m["mod_mouth_hz"]) and not (2.5 <= m["mod_mouth_hz"] <= 8.5):
        f.append(f"⚠ 입 리듬 {m['mod_mouth_hz']}Hz (음절대역 밖)")
    if m["track_lag_ms"] >= 100:
        f.append(f"⚠ 추종지연 {m['track_lag_ms']}ms")
    if m["chatter_a_hz"] >= 6:
        f.append(f"⚠ 떨림 {m['chatter_a_hz']}Hz")
    if m["full_pct"] >= 15:
        f.append(f"⚠ 만개포화 {m['full_pct']}%")
    if m["asym_a"] is not None and m["asym_a"] < 0.7:
        f.append(f"ℹ 열림>닫힘 (신호 {m['asym_a']}) — 실제 턱은 텐던 슬랙으로 더 느릴 수 있음(영상검증)")
    if "sustained" in m["name"] and m["chatter_a_hz"] >= 4:
        f.append(f"⚠ 지속모음 떨림 {m['chatter_a_hz']}Hz")
    if ("silence" in m["name"] or "bilabial" in m["name"]) and m["sil_closed_pct"] is not None and m["sil_closed_pct"] < 70:
        f.append(f"⚠ 묵음 폐쇄 약함 ({m['sil_closed_pct']}%)")
    if not f:
        f.append("✅ 기대대로 양호")
    return f


def build_html(rows):
    cols = [
        ("name", "예문"), ("dur", "s"), ("syl_rate_hz", "음절률Hz"),
        ("recall", "recall"), ("precision", "precis"),
        ("mod_audio_hz", "오디오Hz"), ("mod_mouth_hz", "입Hz"),
        ("env_lag_ms", "명령싱크"), ("act_lag_ms", "실제싱크"), ("evt_offset_ms", "이벤트정렬"),
        ("track_r", "추종r"), ("track_lag_ms", "추종ms"),
        ("track_rms", "추종RMS"), ("open_mean_pct", "개구%"), ("full_pct", "만개%"),
        ("closed_pct", "닫힘%"), ("sil_closed_pct", "묵음폐쇄%"),
        ("off_r", "오프r"), ("off_lag_ms", "오프지연"), ("off_rmse", "오프RMSE"),
        ("chatter_a_hz", "떨림Hz"), ("jitter_p95_ms", "지터"),
    ]
    head = "".join(f"<th>{lbl}</th>" for _, lbl in cols)
    body = ""
    for r in rows:
        tds = "".join(f"<td>{'' if r.get(k) is None else r.get(k)}</td>" for k, _ in cols)
        exp = EXPECT.get(r["name"], "발화체 기준")
        body += f"<tr data-name='{r['name']}' title='{exp}'>{tds}</tr>"
        body += f"<tr class=vrow><td colspan={len(cols)}>📌 {exp} &nbsp;|&nbsp; " + "  ".join(verdict(r)) + "</td></tr>"
    data = json.dumps([{**{k: v for k, v in r.items() if k != '_series'}, "series": r["_series"]} for r in rows])
    return f"""<!doctype html><html lang=ko><head><meta charset=utf-8>
<title>온라인 입 궤적 객관 평가</title><style>
 body{{font-family:system-ui,'Apple SD Gothic Neo',sans-serif;margin:22px;background:#0f1117;color:#e7e7e7}}
 h1{{font-size:19px}} .sub{{color:#9aa;font-size:12px;margin-bottom:14px;line-height:1.5}}
 table{{border-collapse:collapse;font-size:11.5px;width:100%}}
 th,td{{border:1px solid #2a2f3a;padding:3px 6px;text-align:right}}
 th:first-child,td:first-child{{text-align:left}} th{{background:#1a1f2b;position:sticky;top:0}}
 tr[data-name]{{cursor:pointer}} tr[data-name]:hover{{background:#1c2330}}
 .vrow td{{text-align:left;color:#cdb;background:#141821;font-size:11px}}
 canvas{{background:#11151d;border:1px solid #2a2f3a;border-radius:6px;width:100%;height:190px}}
 .clab{{font-size:13px;margin:16px 0 4px}} .legend b{{padding:0 6px}}
 .a{{color:#5ab0ff}}.t{{color:#ffd24a}}.ac{{color:#7CFC9A}}.nu{{color:#ff6b6b}}.ev{{color:#d08bff}}
</style></head><body>
<h1>🎯 온라인 입 모터 궤적 — 객관 평가 (음성학 기준)</h1>
<div class=sub>턱 1자유도. 음절 핵마다 한 번 열리는가(recall/precision), 입 리듬이 음절대역인가, 묵음에 닫히는가, <b>립싱크</b>가 맞는가를 본다. 행 클릭 → 그래프.<br>
<b>싱크 3종</b>: 명령싱크(오디오→목표), 실제싱크(오디오→ID5모터, +면 입이 소리보다 뒤), 이벤트정렬(음절핵↔입이벤트 평균 시간차). 지각 허용 ≈ 입이 소리보다 +125ms 뒤까지 자연스러움.<br>
<span class=legend><b class=a>■오디오</b><b class=t>■목표(ID5명령)</b><b class=ac>■실제(ID5스풀)</b><b style="color:#e8590c">■오프라인(zero-phase)</b><b class=nu>● 음절핵</b><b class=ev>● 입이벤트</b></span><br>
<span style="color:#9aa">자막: TTS 단어를 가장 가까운 음절핵에 스냅해 표시 (단어가 그 음절핵에서 발화됨).</span><br>
<span style="color:#778">※ 실제=ID5 모터(실 스풀) 위치이며 실제 턱 벌어짐(mm)이 아님 — 실 슬랙/늘어짐 때문에 모터 추종이 좋아도 물리 개구는 다를 수 있음(진짜 개구량은 영상 필요). 닫힘은 능동 푸시가 아니라 수동 복원 → 닫/열 비대칭 주목.</span></div>
<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>
<div id=charts></div>
<script>
const DATA={data};const charts=document.getElementById('charts');
function draw(rec){{const w=charts.clientWidth,h=230,pad=24,subH=34;
 const wrap=document.createElement('div');const lab=document.createElement('div');lab.className='clab';
 lab.textContent='▶ '+rec.name+' ('+rec.dur+'s)  recall='+rec.recall+' precision='+rec.precision+' 입리듬='+rec.mod_mouth_hz+'Hz';
 wrap.appendChild(lab);const cv=document.createElement('canvas');cv.id='cv_'+rec.name;wrap.appendChild(cv);charts.appendChild(wrap);
 const dpr=window.devicePixelRatio||1;cv.width=w*dpr;cv.height=h*dpr;cv.style.height=h+'px';const x=cv.getContext('2d');x.scale(dpr,dpr);
 const s=rec.series,n=s.t.length,T=s.t[n-1]||1;const plotB=h-pad-subH,plotH=plotB-pad;
 const X=t=>pad+(w-2*pad)*t/T,Y=v=>plotB-plotH*(v/400);
 x.strokeStyle='#2a2f3a';x.lineWidth=1;for(let g=0;g<=4;g++){{const yy=pad+plotH*g/4;x.beginPath();x.moveTo(pad,yy);x.lineTo(w-pad,yy);x.stroke();}}
 function ln(arr,c,wd){{if(!arr)return;x.strokeStyle=c;x.lineWidth=wd;x.beginPath();let st=false;for(let i=0;i<arr.length;i++){{const v=arr[i];if(v==null){{st=false;continue;}}const px=X(s.t[i]),py=Y(v);st?x.lineTo(px,py):x.moveTo(px,py);st=true;}}x.stroke();}}
 ln(s.offline,'#e8590c',1.4);ln(s.audio,'#5ab0ff',1);ln(s.target,'#ffd24a',1.6);ln(s.actual,'#7CFC9A',1.4);
 x.strokeStyle='#ff6b6b';x.lineWidth=1;s.nuclei.forEach(t=>{{x.beginPath();x.moveTo(X(t),plotB);x.lineTo(X(t),plotB+5);x.stroke();x.beginPath();x.arc(X(t),pad,2.5,0,7);x.fillStyle='#ff6b6b';x.fill();}});
 x.fillStyle='#d08bff';s.events.forEach(t=>{{x.beginPath();x.arc(X(t),pad+8,2.5,0,7);x.fill();}});
 if(s.subs){{x.font='10px system-ui';x.textAlign='center';let lastX=-99;s.subs.forEach((sb,i)=>{{const px=X(sb.t);const row=(px-lastX<22)?1:0;lastX=px;
   x.fillStyle='#9fb3c8';x.fillText(sb.w,px,plotB+16+row*12);}});}}
}}
DATA.forEach(draw);
document.querySelectorAll('tr[data-name]').forEach(tr=>tr.onclick=()=>{{const e=document.getElementById('cv_'+tr.dataset.name);if(e)e.scrollIntoView({{behavior:'smooth',block:'center'}});}});
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="motion_eval/mouth/logs")
    ap.add_argument("--offdir", default="motion_eval/mouth/offline")
    ap.add_argument("--txtdir", default="motion_eval/mouth/text")
    ap.add_argument("--out", default="motion_eval/mouth/reports/report.html")
    a = ap.parse_args()
    offdir = Path(a.offdir)
    txtdir = Path(a.txtdir)
    csvs = sorted(p for p in Path(a.indir).glob("*.csv"))
    rows = []
    for p in csvs:
        c = read_csv(p)
        if len(c["target"]) < 8:
            print(f"skip {p.name}")
            continue
        offp = offdir / p.name
        offline = read_offline(offp) if offp.exists() else None
        txtp = txtdir / f"{p.stem}.json"
        words = json.loads(txtp.read_text(encoding="utf-8")).get("words") if txtp.exists() else None
        rows.append(analyze(p.stem, c, offline, words))
    if not rows:
        print(f"no logs in {a.indir} — run capture.sh first")
        return
    print(f"\n{'example':24}{'syl/s':>6}{'recall':>7}{'prec':>6}{'actLag':>7}{'mthHz':>6}"
          f"{'trkR':>6}{'silCl%':>7}{'offR':>6}{'offLag':>7}{'offRMSE':>8}")
    for m in rows:
        print(f"{m['name']:24}{m['syl_rate_hz']:>6}{str(m['recall']):>7}{str(m['precision']):>6}"
              f"{str(m['act_lag_ms']):>6}m{m['mod_mouth_hz']:>6}{m['track_r']:>6}{str(m['sil_closed_pct']):>7}"
              f"{str(m['off_r']):>6}{str(m['off_lag_ms']):>6}m{str(m['off_rmse']):>8}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(build_html(rows), encoding="utf-8")
    print(f"\nreport -> {a.out}")


if __name__ == "__main__":
    main()
