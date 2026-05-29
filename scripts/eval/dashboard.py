"""Generate an HTML dashboard from scored eval results.

Usage:
    uv run python scripts/eval/dashboard.py data/eval/results/scored.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_CSS = """
* { box-sizing: border-box; }
body { font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; background: #f0f2f5; color: #1a1a2e; }
header { background: #1a1a2e; color: white; padding: 1.5rem 2rem; }
header h1 { margin: 0; font-size: 1.4rem; font-weight: 600; }
header .meta { color: #8b8fa3; font-size: 0.82rem; margin-top: 0.3rem; }
main { max-width: 1100px; margin: 0 auto; padding: 1.5rem; }
section { margin-bottom: 2rem; }
h2 { font-size: 1.1rem; font-weight: 600; margin: 0 0 1rem 0; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
.card { background: white; border-radius: 12px; padding: 1.2rem; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
.card .label { font-size: 0.78rem; color: #6b7280; letter-spacing: 0.02em; }
.card .value { font-size: 2rem; font-weight: 700; margin-top: 0.3rem; line-height: 1; }
.card .sub { font-size: 0.78rem; color: #9ca3af; margin-top: 0.4rem; }
.good { color: #059669; }
.warn { color: #d97706; }
.bad { color: #dc2626; }
.panel { background: white; border-radius: 12px; padding: 1.2rem; box-shadow: 0 1px 4px rgba(0,0,0,0.06); margin-bottom: 1rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.85rem; table-layout: fixed; }
.asr-table th:last-child, .asr-table td:last-child { width: 70px; white-space: nowrap; text-align: center; }
.asr-table th:first-child, .asr-table td:first-child { width: 45%; }
th { text-align: left; padding: 0.5rem 0.7rem; color: #6b7280; font-weight: 500; border-bottom: 2px solid #e5e7eb; font-size: 0.78rem; }
td { padding: 0.5rem 0.7rem; border-bottom: 1px solid #f3f4f6; }
tr:last-child td { border-bottom: none; }
.tag { display: inline-block; padding: 0.2rem 0.55rem; border-radius: 6px; font-size: 0.72rem; font-weight: 600; }
.tag-ok { background: #d1fae5; color: #065f46; }
.tag-fail { background: #fee2e2; color: #991b1b; }
.tag-warn { background: #fef3c7; color: #92400e; }
.tag-mute { background: #f3f4f6; color: #6b7280; }
.bar-bg { display: inline-block; width: 120px; height: 6px; background: #e5e7eb; border-radius: 3px; vertical-align: middle; }
.bar-fill { display: block; height: 100%; border-radius: 3px; }
.bar-good { background: #059669; }
.bar-warn { background: #d97706; }
.bar-bad { background: #dc2626; }
.mono { font-family: 'SF Mono', 'Consolas', monospace; font-size: 0.82rem; }
.text-mute { color: #9ca3af; }
.divider { border: none; border-top: 1px solid #e5e7eb; margin: 1.5rem 0; }
"""


def _color_class(value: float, good: float, bad: float) -> str:
    if value <= good:
        return "good"
    if value >= bad:
        return "bad"
    return "warn"


def _bar_class(value: float, good: float, bad: float) -> str:
    if value <= good:
        return "bar-good"
    if value >= bad:
        return "bar-bad"
    return "bar-warn"


def _latency_bar(ms: float, max_ms: float, good: float = 1500, bad: float = 3000) -> str:
    pct = min(ms / max_ms * 100, 100) if max_ms else 0
    cls = _bar_class(ms, good, bad)
    return f'<span class="bar-bg"><span class="bar-fill {cls}" style="width:{pct:.0f}%"></span></span>'


def _wer_tag(wer: float) -> str:
    if wer == 0:
        return '<span class="tag tag-ok">정확</span>'
    if wer <= 0.15:
        return f'<span class="tag tag-warn">{wer:.0%}</span>'
    return f'<span class="tag tag-fail">{wer:.0%}</span>'


def _outcome_tag(outcome: str | None) -> str:
    tags = {
        "truncated": ("tag-ok", "중단됨"),
        "completed": ("tag-fail", "미감지"),
        "cancelled": ("tag-warn", "취소됨"),
    }
    if outcome in tags:
        cls, label = tags[outcome]
        return f'<span class="tag {cls}">{label}</span>'
    return f'<span class="tag tag-mute">{outcome or "—"}</span>'


def _stat_line(stats: dict, label: str = "") -> str:
    if not stats:
        return ""
    return (
        f"{label + ' ' if label else ''}"
        f'<span class="mono">{stats["median_ms"]:.0f}ms</span> '
        f'<span class="text-mute">(P95 {stats["p95_ms"]:.0f}ms)</span>'
    )


def build_html(scored: dict) -> str:
    scores = scored.get("scores", {})
    turns = scored.get("turns", [])
    asr_data = scores.get("asr", {})
    latency = scores.get("latency", {})
    interruption = scores.get("interruption", {})

    parts = [
        f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><title>평가 대시보드</title>
<style>{_CSS}</style></head><body>
<header>
<h1>Ray 파이프라인 평가</h1>
<div class="meta">{scored.get("started_at", "")} — {scored.get("finished_at", "")} &middot; {scored.get("total", 0)}턴 실행</div>
</header>
<main>
"""
    ]

    # ================================================================
    # Overview cards
    # ================================================================
    parts.append('<section><div class="cards">')

    if asr_data:
        wer = asr_data.get("mean_wer", 0)
        parts.append(
            f'<div class="card"><div class="label">음성 인식 정확도</div>'
            f'<div class="value {_color_class(wer, 0.05, 0.2)}">{(1 - wer):.0%}</div>'
            f'<div class="sub">{asr_data.get("perfect_count", 0)}/{asr_data.get("total_scored", 0)}건 완벽 인식</div></div>'
        )

    td_stats = latency.get("turn_detection_delay_ms", {})
    if td_stats:
        med = td_stats.get("median_ms", 0)
        parts.append(
            f'<div class="card"><div class="label">턴 감지 속도</div>'
            f'<div class="value {_color_class(med, 800, 2000)}">{med:.0f}<span style="font-size:0.5em">ms</span></div>'
            f'<div class="sub">침묵 시작 → 턴 감지 (중위값)</div></div>'
        )

    ts_pb = latency.get("turn_shift_to_playback_ms", {})
    if ts_pb:
        med = ts_pb.get("median_ms", 0)
        parts.append(
            f'<div class="card"><div class="label">응답 속도</div>'
            f'<div class="value {_color_class(med, 1500, 3000)}">{med:.0f}<span style="font-size:0.5em">ms</span></div>'
            f'<div class="sub">턴 감지 → 응답 재생 시작 (중위값)</div></div>'
        )

    if interruption and interruption.get("testable"):
        dr = interruption.get("detection_rate", 0)
        parts.append(
            f'<div class="card"><div class="label">인터럽션 감지율</div>'
            f'<div class="value {_color_class(1 - dr, 0.1, 0.3)}">{dr:.0%}</div>'
            f'<div class="sub">{interruption.get("detected", 0)}/{interruption.get("testable", 0)}건 감지</div></div>'
        )

    parts.append("</div></section>")

    # ================================================================
    # ASR section — grouped by suite
    # ================================================================
    asr_turns = [t for t in turns if "asr_score" in t]
    if asr_turns:
        asr_by_suite = scores.get("asr", {}).get("by_suite", {})
        parts.append("<section><h2>음성 인식 (ASR)</h2>")

        if asr_by_suite:
            parts.append('<div class="panel"><table>')
            parts.append("<tr><th>Suite</th><th>평균 WER</th><th>완벽 인식</th></tr>")
            for suite, stats in asr_by_suite.items():
                wer = stats["mean_wer"]
                parts.append(
                    f"<tr><td>{suite}</td>"
                    f'<td class="{_color_class(wer, 0.05, 0.2)}">{wer:.1%}</td>'
                    f"<td>{stats['perfect_count']}/{stats['total_scored']}</td></tr>"
                )
            parts.append("</table></div>")

        from itertools import groupby

        sorted_turns = sorted(asr_turns, key=lambda t: t["suite_name"])
        for suite_name, group in groupby(sorted_turns, key=lambda t: t["suite_name"]):
            suite_turns = list(group)
            parts.append('<div class="panel"><table class="asr-table">')
            parts.append(f"<tr><th colspan='3'>{suite_name}</th></tr>")
            parts.append("<tr><th>원본 텍스트</th><th>인식 결과</th><th>정확도</th></tr>")
            for t in suite_turns:
                wer = t["asr_score"]["wer"]
                asr_text = t["asr_text"] or '<span class="text-mute">인식 실패</span>'
                sys_text = t.get("system_text", "")
                diff_mark = ""
                if sys_text and sys_text != t["asr_text"]:
                    diff_mark = f' <span class="tag tag-warn">시스템: {sys_text}</span>'
                parts.append(
                    f"<tr><td>{t['input_text']}</td><td>{asr_text}{diff_mark}</td><td>{_wer_tag(wer)}</td></tr>"
                )
            parts.append("</table></div>")
        parts.append("</section>")

    # ================================================================
    # Turn-taking section
    # ================================================================
    tt_turns = [t for t in turns if t.get("latency", {}).get("turn_shift_to_playback_ms")]
    if tt_turns:
        max_lat = max(t["latency"]["turn_shift_to_playback_ms"] for t in tt_turns)
        parts.append("<section><h2>턴테이킹</h2>")

        if latency:
            _LABELS = {
                "turn_detection_delay": "턴 감지 (침묵 → 감지)",
                "turn_shift_to_playback": "응답 생성 (감지 → 재생)",
                "llm_ttft": "LLM 첫 토큰",
                "tts_ttfc": "TTS 첫 청크",
                "bridge": "Bridge 전송",
            }
            parts.append('<div class="panel"><table>')
            parts.append("<tr><th>구간</th><th>중위값</th><th>P95</th><th>범위</th></tr>")
            for key, stats in latency.items():
                if not stats:
                    continue
                label = _LABELS.get(key.removesuffix("_ms"), key.removesuffix("_ms"))
                parts.append(
                    f"<tr><td>{label}</td>"
                    f'<td class="mono">{stats["median_ms"]:.0f}ms</td>'
                    f'<td class="mono">{stats["p95_ms"]:.0f}ms</td>'
                    f'<td class="mono text-mute">{stats["min_ms"]:.0f} – {stats["max_ms"]:.0f}ms</td></tr>'
                )
            parts.append("</table></div>")

        parts.append('<div class="panel"><table>')
        parts.append("<tr><th>질문</th><th>Suite</th><th>턴 감지</th><th>응답 지연</th><th></th><th>결과</th></tr>")
        for t in tt_turns:
            lat = t["latency"]["turn_shift_to_playback_ms"]
            td_delay = t.get("turn_detection_delay_ms")
            td_str = f'<span class="mono">{td_delay:.0f}ms</span>' if td_delay else "—"
            bar = _latency_bar(lat, max_lat)
            parts.append(
                f"<tr><td>{t['question_id']}</td><td>{t['suite_name']}</td>"
                f"<td>{td_str}</td>"
                f'<td class="mono">{lat:.0f}ms</td>'
                f"<td>{bar}</td>"
                f"<td>{_outcome_tag(t.get('outcome'))}</td></tr>"
            )
        parts.append("</table></div></section>")

    # ================================================================
    # Interruption section
    # ================================================================
    int_turns = [t for t in turns if "interrupt_delay_sec" in t]
    if int_turns and interruption:
        parts.append("<section><h2>인터럽션</h2>")

        if interruption.get("by_delay"):
            parts.append('<div class="panel"><table>')
            parts.append("<tr><th>Delay</th><th>감지율</th><th>결과 분포</th></tr>")
            for delay_str, b in interruption["by_delay"].items():
                testable = b["testable"]
                if testable == 0:
                    rate_str = '<span class="text-mute">N/A</span>'
                    dist = f'<span class="text-mute">응답 종료 {b["na"]}건</span>'
                else:
                    rate = b["detected"] / testable
                    rate_cls = _color_class(1 - rate, 0.1, 0.3)
                    rate_str = f'<span class="{rate_cls}">{rate:.0%}</span> ({b["detected"]}/{testable})'
                    dist_parts = []
                    if b["truncated"]:
                        dist_parts.append(f'<span class="tag tag-ok">중단 {b["truncated"]}</span>')
                    if b["completed"]:
                        dist_parts.append(f'<span class="tag tag-fail">미감지 {b["completed"]}</span>')
                    if b["na"]:
                        dist_parts.append(f'<span class="tag tag-mute">N/A {b["na"]}</span>')
                    dist = " ".join(dist_parts)
                parts.append(f"<tr><td>{delay_str}초</td><td>{rate_str}</td><td>{dist}</td></tr>")
            parts.append("</table>")

            int_lat = interruption.get("latency", {})
            if int_lat:
                parts.append(f'<p style="margin:0.8rem 0 0; font-size:0.85rem">감지 지연: {_stat_line(int_lat)}</p>')
            parts.append("</div>")

        parts.append('<div class="panel"><table>')
        parts.append("<tr><th>질문</th><th>끼어들기</th><th>Delay</th><th>결과</th></tr>")
        for t in int_turns:
            audio = t.get("interrupt_audio", "—")
            outcome = _outcome_tag(t.get("outcome"))
            not_played = "" if t.get("interrupt_played") else ' <span class="tag tag-mute">미재생</span>'
            parts.append(
                f"<tr><td>{t['question_id']}</td>"
                f"<td>{audio}</td>"
                f"<td>{t['interrupt_delay_sec']:.1f}초</td>"
                f"<td>{outcome}{not_played}</td></tr>"
            )
        parts.append("</table></div></section>")

    parts.append("</main></body></html>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate eval HTML dashboard")
    parser.add_argument("scored", help="Path to scored.json")
    parser.add_argument("--output", default=None, help="Output HTML path")
    args = parser.parse_args()

    scored_path = Path(args.scored)
    if not scored_path.exists():
        print(f"Error: {scored_path} not found", file=sys.stderr)
        sys.exit(1)

    scored = json.loads(scored_path.read_text())
    html = build_html(scored)

    output_path = Path(args.output) if args.output else scored_path.parent / "dashboard.html"
    output_path.write_text(html)
    print(f"Dashboard saved: {output_path}")


if __name__ == "__main__":
    main()
